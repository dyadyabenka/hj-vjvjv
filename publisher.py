"""Публикация в канал и отправка черновиков на модерацию через python-telegram-bot.

Раньше здесь были прямые вызовы Bot API через requests. Теперь используется
telegram.Bot из библиотеки python-telegram-bot — она нужна в проекте всё
равно (для кнопок и обработки нажатий в moderation.py), так что логичнее
отправлять сообщения через один и тот же клиент.
"""

import html
import logging

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

log = logging.getLogger(__name__)

# Жёсткий лимит Telegram на длину текстового сообщения и на подпись к фото
MAX_MESSAGE_LENGTH = 4096
MAX_CAPTION_LENGTH = 1024


def to_html(text: str) -> str:
    """Готовит текст модели к отправке с parse_mode=HTML.

    Экранируем всё (в тексте могут быть <, > и &, которые Telegram примет
    за разметку), затем добавляем жирный заголовок первой строкой и жирный
    подзаголовок "Источники:". Голые ссылки Telegram делает кликабельными сам.
    """
    lines = text.strip().split("\n")
    formatted: list[str] = []

    for index, line in enumerate(lines):
        safe = html.escape(line)
        stripped = line.strip().lower()

        if index == 0 and stripped:
            safe = f"<b>{safe}</b>"
        elif stripped.startswith("источник"):
            safe = f"<b>{safe}</b>"

        formatted.append(safe)

    return "\n".join(formatted)


def split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Режет длинный пост на части по границам строк (разметка построчная,
    поэтому резать по строкам безопасно — теги не разорвутся).
    """
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""

    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]

        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            parts.append(current)
            current = line
        else:
            current = candidate

    if current:
        parts.append(current)

    return parts


def _moderation_keyboard(post_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Опубликовать", callback_data=f"approve:{post_id}"),
                InlineKeyboardButton("✏️ Доработать", callback_data=f"edit:{post_id}"),
            ]
        ]
    )


async def send_draft(bot: Bot, admin_chat_id: str, post_id: int, text: str, image_url: str | None) -> bool:
    """Отправляет черновик поста админу в личку с кнопками approve/edit.

    Возвращает True, если сообщение (хотя бы часть) успешно доставлено.
    """
    html_text = to_html(text)
    keyboard = _moderation_keyboard(post_id)

    try:
        if image_url and len(html_text) <= MAX_CAPTION_LENGTH:
            await bot.send_photo(
                chat_id=admin_chat_id,
                photo=image_url,
                caption=html_text,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
            return True

        if image_url:
            # Текст не помещается в подпись — фото отдельно, текст отдельно.
            # Пытаемся отправить фото, но если не получится (битый URL и т.п.),
            # это не должно мешать отправить сам текст на модерацию.
            try:
                await bot.send_photo(chat_id=admin_chat_id, photo=image_url)
            except TelegramError as exc:
                log.warning("Пост %d: не удалось отправить фото админу: %s", post_id, exc)

        parts = split_message(html_text)
        for index, part in enumerate(parts):
            is_last = index == len(parts) - 1
            await bot.send_message(
                chat_id=admin_chat_id,
                text=part,
                parse_mode="HTML",
                reply_markup=keyboard if is_last else None,
                disable_web_page_preview=True,
            )
        return True

    except TelegramError as exc:
        log.error("Пост %d: не удалось отправить черновик админу: %s", post_id, exc)
        return False


async def ask_for_edit_note(bot: Bot, admin_chat_id: str) -> None:
    """Просит админа прислать текстом, что поправить в посте."""
    try:
        await bot.send_message(
            chat_id=admin_chat_id,
            text="Напиши, что поправить в посте (короче/подробнее/другой акцент и т.д.) — "
            "следующим сообщением.",
        )
    except TelegramError as exc:
        log.warning("Не удалось отправить запрос замечаний админу: %s", exc)


async def notify(bot: Bot, admin_chat_id: str, text: str) -> None:
    """Служебное уведомление админу (ошибка, статус и т.п.), без разметки и кнопок."""
    try:
        await bot.send_message(chat_id=admin_chat_id, text=text)
    except TelegramError as exc:
        log.warning("Не удалось отправить уведомление админу: %s", exc)


async def publish_post(
    bot: Bot, channel_id: str, text: str, image_url: str | None, config: dict
) -> bool:
    """Публикует готовый (одобренный) пост в канал. True — если ушло успешно."""
    cfg = config["publishing"]
    disable_preview = cfg["disable_web_page_preview"]
    html_text = to_html(text)

    try:
        if image_url and len(html_text) <= MAX_CAPTION_LENGTH:
            await bot.send_photo(
                chat_id=channel_id, photo=image_url, caption=html_text, parse_mode="HTML"
            )
            return True

        if image_url:
            try:
                await bot.send_photo(chat_id=channel_id, photo=image_url)
            except TelegramError as exc:
                log.warning("Не удалось отправить фото в канал, публикуем без него: %s", exc)

        for part in split_message(html_text):
            await bot.send_message(
                chat_id=channel_id,
                text=part,
                parse_mode="HTML",
                disable_web_page_preview=disable_preview,
            )
        return True

    except TelegramError as exc:
        log.error("Публикация в канал не удалась: %s", exc)
        return False
