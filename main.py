"""Бот работает непрерывно (не по cron): раз в config.run.interval_hours собирает
статьи, готовит черновики постов и присылает их админу в личку на модерацию.
Публикация в канал происходит только после нажатия "Опубликовать".

Запуск:
    python main.py              обычный режим — работает и слушает, пока не остановишь (Ctrl+C)
    python main.py --dry-run    один проход конвейера, черновики печатаются в лог,
                                 в Telegram ничего не уходит (для проверки настроек)
"""

import argparse
import json
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
from fetcher import Article, fetch_all
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


def read_secrets(config: dict, dry_run: bool) -> dict[str, str] | None:
    """Читает ключи из .env. PEXELS_API_KEY необязателен (картинок просто не будет)."""
    load_dotenv(BASE_DIR / ".env")

    provider = config["synthesis"].get("provider", "anthropic").strip().lower()
    llm_key_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "ANTHROPIC_API_KEY"

    required = [llm_key_name]
    if not dry_run:
        required += ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL_ID", "TELEGRAM_ADMIN_ID"]

    secrets = {name: (os.getenv(name) or "").strip() for name in required}
    missing = [name for name, value in secrets.items() if not value]

    if missing:
        log.error(
            "Не заполнены переменные окружения: %s. "
            "Скопируй .env.example в .env и впиши значения.",
            ", ".join(missing),
        )
        return None

    secrets["LLM_API_KEY"] = secrets.pop(llm_key_name)
    secrets["PEXELS_API_KEY"] = (os.getenv("PEXELS_API_KEY") or "").strip()
    return secrets


# --- сборка черновиков постов (общая часть для --dry-run и обычного режима) ------


def prepare_drafts(config: dict, storage: Storage, synthesizer: Synthesizer, secrets: dict) -> list[dict]:
    """Прогоняет fetch -> dedup -> cluster -> synthesize -> подбор картинки.

    Возвращает список готовых к отправке на модерацию черновиков (ещё не в базе).
    Каждый элемент: {cluster, text, articles_block, image_url, image_query}
    """
    max_posts = config["synthesis"]["max_posts_per_run"]

    articles = fetch_all(config)
    added = storage.save_new(articles)
    log.info("Новых статей записано в базу: %d (дублей: %d)", added, len(articles) - added)

    pending = storage.load_pending(config["fetch"]["max_age_hours"])
    log.info("В очереди на обработку: %d статей", len(pending))
    if not pending:
        return []

    clusters = rank_clusters(cluster_articles(pending, config))
    selected = clusters[:max_posts]
    log.info("Берём в работу %d групп из %d", len(selected), len(clusters))

    drafts: list[dict] = []
    pexels_key = secrets.get("PEXELS_API_KEY", "")

    for index, cluster in enumerate(selected, start=1):
        result = synthesizer.synthesize(cluster)
        if result is None:
            log.warning("Группа %d: пост не создан, пропускаем", index)
            continue
        text, articles_block = result

        image_url = None
        image_query = None
        if pexels_key:
            image_query = images.pick_query(cluster[0].title, config)
            image_url = images.search_image(image_query, pexels_key)
        else:
            log.info("PEXELS_API_KEY не задан — пост будет без картинки")

        drafts.append(
            {
                "cluster": cluster,
                "text": text,
                "articles_block": articles_block,
                "image_url": image_url,
                "image_query": image_query,
            }
        )

    return drafts


# --- обычный (непрерывный) режим --------------------------------------------------


async def pipeline_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Периодическая задача JobQueue: готовит черновики и шлёт их админу на модерацию."""
    config = context.bot_data["config"]
    secrets = context.bot_data["secrets"]
    synthesizer: Synthesizer = context.bot_data["synthesizer"]
    db_path = context.bot_data["db_path"]

    log.info("=== Плановый прогон конвейера ===")
    with Storage(db_path) as storage:
        try:
            drafts = prepare_drafts(config, storage, synthesizer, secrets)
        except Exception:  # noqa: BLE001 — сбой одного прогона не должен убивать бота
            log.exception("Ошибка при подготовке черновиков")
            return

        if not drafts:
            log.info("Новых черновиков нет")
            return

        for draft in drafts:
            post_id = storage.create_post(
                article_urls=[a.url for a in draft["cluster"]],
                articles_block=draft["articles_block"],
                text=draft["text"],
                image_url=draft["image_url"],
                image_query=draft["image_query"],
            )
            ok = await publisher.send_draft(
                context.bot, secrets["TELEGRAM_ADMIN_ID"], post_id, draft["text"], draft["image_url"]
            )
            log.info("Черновик поста %d отправлен админу на модерацию: %s", post_id, ok)


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

        if action == "approve":
            ok = await publisher.publish_post(
                context.bot,
                context.bot_data["secrets"]["TELEGRAM_CHANNEL_ID"],
                post["text"],
                post["image_url"],
                config,
            )
            if ok:
                storage.mark_post_published(post_id)
                log.info("Пост %d опубликован в канал", post_id)
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

        new_text = synthesizer.revise(post["articles_block"], post["text"], edit_note)
        if new_text is None:
            await publisher.notify(
                context.bot, admin_id, f"Не получилось переписать пост {post_id}, попробуй ещё раз."
            )
            return

        storage.update_post_text(post_id, new_text)
        await publisher.send_draft(context.bot, admin_id, post_id, new_text, post["image_url"])


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

    log.info(
        "Запуск в непрерывном режиме: прогон конвейера каждые %d ч. Останов — Ctrl+C.",
        interval_hours,
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


# --- dry-run: один проход без Telegram -------------------------------------------


def run_dry_run(config: dict, secrets: dict, db_path: str) -> int:
    synthesizer = Synthesizer(config, secrets["LLM_API_KEY"])
    with Storage(db_path) as storage:
        drafts = prepare_drafts(config, storage, synthesizer, secrets)

    if not drafts:
        log.info("Черновиков нет (нечего было обрабатывать)")
        return 0

    for index, draft in enumerate(drafts, start=1):
        log.info(
            "--- Черновик %d/%d (картинка: %s) ---\n%s",
            index,
            len(drafts),
            draft["image_url"] or "нет",
            draft["text"],
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RSS -> LLM -> модерация -> Telegram")
    parser.add_argument("--config", default="config.yaml", help="путь к конфигу")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="один проход конвейера, вывод в лог, без обращений к Telegram",
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
