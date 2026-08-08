"""Бот работает непрерывно и ведёт НЕСКОЛЬКО каналов сразу (см. channels: в
config.yaml). Раз в config.run.interval_hours конвейер проходит по всем
каналам по очереди: собирает статьи по своим источникам и ключевым словам,
готовит черновик поста и присылает его админу в личку на модерацию с пометкой,
к какому каналу он относится. Публикация происходит только после нажатия
"Опубликовать" — в тот канал, что указан в черновике.

Запуск:
    python main.py              обычный режим — работает и слушает, пока не остановишь (Ctrl+C)
    python main.py --dry-run    один проход конвейера по всем каналам, черновики
                                 печатаются в лог, в Telegram ничего не уходит
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import images
import publisher
from clusterer import cluster_articles, rank_clusters
from fetcher import fetch_for_channel
from storage import Storage
from synthesizer import Synthesizer

BASE_DIR = Path(__file__).resolve().parent

log = logging.getLogger("bot")


def setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-12s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic", "telegram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def get_channel(config: dict, channel_id: str) -> dict | None:
    for channel in config["channels"]:
        if channel["id"] == channel_id:
            return channel
    return None


def read_secrets(config: dict, dry_run: bool) -> dict | None:
    """Читает ключи из .env.

    Помимо общих ключей, для каждого канала из config.yaml требуется своя
    переменная окружения (channels[].channel_id_env) — так один бот и один
    .env обслуживают сразу все каналы. PEXELS_API_KEY необязателен.
    """
    load_dotenv(BASE_DIR / ".env")

    provider = config["synthesis"].get("provider", "anthropic").strip().lower()
    llm_key_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "ANTHROPIC_API_KEY"

    required = [llm_key_name]
    if not dry_run:
        required += ["TELEGRAM_BOT_TOKEN", "TELEGRAM_ADMIN_ID"]
        required += [ch["channel_id_env"] for ch in config["channels"]]

    values = {name: (os.getenv(name) or "").strip() for name in required}
    missing = [name for name, value in values.items() if not value]

    if missing:
        log.error(
            "Не заполнены переменные окружения: %s. "
            "Скопируй .env.example в .env и впиши значения.",
            ", ".join(missing),
        )
        return None

    secrets = {
        "LLM_API_KEY": values.pop(llm_key_name),
        "TELEGRAM_BOT_TOKEN": values.get("TELEGRAM_BOT_TOKEN", ""),
        "TELEGRAM_ADMIN_ID": values.get("TELEGRAM_ADMIN_ID", ""),
        "PEXELS_API_KEY": (os.getenv("PEXELS_API_KEY") or "").strip(),
        # channel config id -> реальный telegram chat id канала
        "channel_ids": {
            ch["id"]: values.get(ch["channel_id_env"], "") for ch in config["channels"]
        },
    }
    return secrets


# --- сборка черновиков постов (общая часть для --dry-run и обычного режима) ------


def prepare_drafts_for_channel(
    channel_cfg: dict,
    config: dict,
    storage: Storage,
    synthesizer: Synthesizer,
    secrets: dict,
    fetch_cache: dict,
    used_images: set[str],
) -> list[dict]:
    """Прогоняет fetch -> dedup -> cluster -> synthesize -> картинка для ОДНОГО канала.

    fetch_cache и used_images общие на весь прогон (все каналы) — чтобы не
    скачивать одну и ту же ленту дважды и не выдавать двум постам одну картинку.
    """
    max_posts = config["synthesis"]["max_posts_per_run"]
    keywords = channel_cfg.get("keywords") or []

    articles = fetch_for_channel(channel_cfg["sources"], keywords, config, fetch_cache)
    added = storage.save_new(articles)
    log.info(
        "[%s] Новых статей записано в базу: %d (дублей: %d)",
        channel_cfg["id"], added, len(articles) - added,
    )

    pending = storage.load_pending(config["fetch"]["max_age_hours"], keywords=keywords)
    log.info("[%s] В очереди на обработку: %d статей", channel_cfg["id"], len(pending))
    if not pending:
        return []

    clusters = rank_clusters(cluster_articles(pending, config))
    selected = clusters[:max_posts]
    log.info("[%s] Берём в работу %d групп из %d", channel_cfg["id"], len(selected), len(clusters))

    drafts: list[dict] = []
    pexels_key = secrets.get("PEXELS_API_KEY", "")
    fallback_queries = channel_cfg.get("fallback_image_queries") or config.get("images", {}).get(
        "fallback_queries"
    )

    for index, cluster in enumerate(selected, start=1):
        result = synthesizer.synthesize(cluster, channel_cfg["topic"])
        if result is None:
            log.warning("[%s] Группа %d: пост не создан, пропускаем", channel_cfg["id"], index)
            continue
        text, articles_block = result

        image_url = None
        image_query = None
        if pexels_key:
            image_query = images.pick_query(cluster[0].title, fallback_queries)
            image_url = images.search_image(image_query, pexels_key, exclude_urls=used_images)
            if image_url:
                used_images.add(image_url)

        drafts.append(
            {
                "channel_id": channel_cfg["id"],
                "channel_name": channel_cfg["name"],
                "cluster": cluster,
                "text": text,
                "articles_block": articles_block,
                "image_url": image_url,
                "image_query": image_query,
            }
        )

    return drafts


def prepare_all_drafts(config: dict, storage: Storage, synthesizer: Synthesizer, secrets: dict) -> list[dict]:
    """Прогоняет все каналы по очереди, возвращает общий список черновиков."""
    fetch_cache: dict = {}
    used_images: set[str] = storage.get_used_image_urls() if secrets.get("PEXELS_API_KEY") else set()

    all_drafts: list[dict] = []
    for channel_cfg in config["channels"]:
        drafts = prepare_drafts_for_channel(
            channel_cfg, config, storage, synthesizer, secrets, fetch_cache, used_images
        )
        all_drafts.extend(drafts)
    return all_drafts


# --- обычный (непрерывный) режим --------------------------------------------------


async def pipeline_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Периодическая задача JobQueue: готовит черновики по всем каналам и шлёт админу."""
    config = context.bot_data["config"]
    secrets = context.bot_data["secrets"]
    synthesizer: Synthesizer = context.bot_data["synthesizer"]
    db_path = context.bot_data["db_path"]

    log.info("=== Плановый прогон конвейера (все каналы) ===")
    with Storage(db_path) as storage:
        try:
            drafts = prepare_all_drafts(config, storage, synthesizer, secrets)
        except Exception:  # noqa: BLE001 — сбой одного прогона не должен убивать бота
            log.exception("Ошибка при подготовке черновиков")
            return

        if not drafts:
            log.info("Новых черновиков нет ни по одному каналу")
            return

        for draft in drafts:
            post_id = storage.create_post(
                channel_id=draft["channel_id"],
                article_urls=[a.url for a in draft["cluster"]],
                articles_block=draft["articles_block"],
                text=draft["text"],
                image_url=draft["image_url"],
                image_query=draft["image_query"],
            )
            # Помечаем статьи как обработанные сразу, пока черновик ждёт
            # модерации — иначе они снова попадут в выборку в следующий
            # плановый прогон и породят дублирующий черновик той же темой.
            storage.mark_processed([a.url for a in draft["cluster"]])

            ok = await publisher.send_draft(
                context.bot,
                secrets["TELEGRAM_ADMIN_ID"],
                post_id,
                draft["channel_name"],
                draft["text"],
                draft["image_url"],
            )
            log.info(
                "[%s] Черновик поста %d отправлен админу на модерацию: %s",
                draft["channel_id"], post_id, ok,
            )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Нажатие кнопки '✅ Опубликовать' или '✏️ Доработать' под черновиком."""
    query = update.callback_query
    admin_id = context.bot_data["secrets"]["TELEGRAM_ADMIN_ID"]

    if str(update.effective_chat.id) != str(admin_id):
        await query.answer("Модерация недоступна", show_alert=False)
        return

    await query.answer()
    action, _, raw_post_id = (query.data or "").partition(":")
    if not raw_post_id.isdigit():
        return
    post_id = int(raw_post_id)

    config = context.bot_data["config"]
    db_path = context.bot_data["db_path"]

    with Storage(db_path) as storage:
        post = storage.get_post(post_id)
        if post is None:
            await query.edit_message_reply_markup(reply_markup=None)
            return

        channel_cfg = get_channel(config, post["channel_id"])
        if channel_cfg is None:
            log.error("Пост %d ссылается на неизвестный канал %s", post_id, post["channel_id"])
            await publisher.notify(context.bot, admin_id, f"Канал поста {post_id} не найден в config.yaml")
            return

        if action == "approve":
            channel_chat_id = context.bot_data["secrets"]["channel_ids"].get(channel_cfg["id"], "")
            ok = await publisher.publish_post(
                context.bot, channel_chat_id, post["text"], post["image_url"], config
            )
            if ok:
                storage.mark_post_published(post_id)
                log.info("Пост %d опубликован в канал %s", post_id, channel_cfg["id"])
            else:
                await publisher.notify(
                    context.bot, admin_id, f"Не удалось опубликовать пост {post_id}, смотри лог."
                )
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:  # noqa: BLE001 — не критично, если разметку не убрать
                pass

        elif action == "edit":
            context.bot_data["awaiting_edit_post_id"] = post_id
            await publisher.ask_for_edit_note(context.bot, admin_id)
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:  # noqa: BLE001
                pass

        elif action == "cancel":
            storage.mark_post_rejected(post_id)
            log.info("Пост %d отменён админом", post_id)
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:  # noqa: BLE001
                pass


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Текстовое сообщение от админа — либо замечание к посту, либо игнор."""
    admin_id = context.bot_data["secrets"]["TELEGRAM_ADMIN_ID"]
    if str(update.effective_chat.id) != str(admin_id):
        return

    post_id = context.bot_data.get("awaiting_edit_post_id")
    if post_id is None:
        return  # не ждём замечаний прямо сейчас — молча игнорируем сообщение

    context.bot_data["awaiting_edit_post_id"] = None
    edit_note = (update.message.text or "").strip()
    if not edit_note:
        return

    config = context.bot_data["config"]
    synthesizer: Synthesizer = context.bot_data["synthesizer"]
    db_path = context.bot_data["db_path"]

    with Storage(db_path) as storage:
        post = storage.get_post(post_id)
        if post is None:
            return

        channel_cfg = get_channel(config, post["channel_id"])
        if channel_cfg is None:
            return

        new_text = synthesizer.revise(post["articles_block"], post["text"], edit_note, channel_cfg["topic"])
        if new_text is None:
            await publisher.notify(
                context.bot, admin_id, f"Не получилось переписать пост {post_id}, попробуй ещё раз."
            )
            return

        storage.update_post_text(post_id, new_text)
        await publisher.send_draft(
            context.bot, admin_id, post_id, channel_cfg["name"], new_text, post["image_url"]
        )


async def on_startup(application: Application) -> None:
    log.info("Бот запущен, жду планового прогона и модерации")


def run_forever(config: dict, secrets: dict, db_path: str) -> None:
    interval_hours = config.get("run", {}).get("interval_hours", 4)

    application = Application.builder().token(secrets["TELEGRAM_BOT_TOKEN"]).post_init(on_startup).build()

    application.bot_data["config"] = config
    application.bot_data["secrets"] = secrets
    application.bot_data["db_path"] = db_path
    application.bot_data["synthesizer"] = Synthesizer(config, secrets["LLM_API_KEY"])
    application.bot_data["awaiting_edit_post_id"] = None

    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.job_queue.run_repeating(
        pipeline_job, interval=interval_hours * 3600, first=15
    )

    channel_names = ", ".join(ch["name"] for ch in config["channels"])
    log.info(
        "Запуск в непрерывном режиме: %d канал(ов) — %s. Прогон каждые %d ч. Останов — Ctrl+C.",
        len(config["channels"]), channel_names, interval_hours,
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


# --- dry-run: один проход без Telegram -------------------------------------------


def run_dry_run(config: dict, secrets: dict, db_path: str) -> int:
    synthesizer = Synthesizer(config, secrets["LLM_API_KEY"])
    with Storage(db_path) as storage:
        drafts = prepare_all_drafts(config, storage, synthesizer, secrets)

    if not drafts:
        log.info("Черновиков нет (нечего было обрабатывать)")
        return 0

    for index, draft in enumerate(drafts, start=1):
        log.info(
            "--- Черновик %d/%d [канал: %s] (картинка: %s) ---\n%s",
            index,
            len(drafts),
            draft["channel_name"],
            draft["image_url"] or "нет",
            draft["text"],
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RSS -> LLM -> модерация -> Telegram (несколько каналов)")
    parser.add_argument("--config", default="config.yaml", help="путь к конфигу")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="один проход конвейера по всем каналам, вывод в лог, без Telegram",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(resolve(args.config))
    secrets = read_secrets(config, args.dry_run)
    if secrets is None:
        return 1

    db_path = str(resolve(config["storage"]["db_path"]))

    if args.dry_run:
        log.info("=== Запуск (dry-run) ===")
        try:
            return run_dry_run(config, secrets, db_path)
        except Exception:  # noqa: BLE001
            log.exception("Фатальная ошибка")
            return 1

    log.info("=== Запуск (непрерывный режим) ===")
    try:
        run_forever(config, secrets, db_path)
    except Exception:  # noqa: BLE001
        log.exception("Фатальная ошибка, бот остановлен")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
