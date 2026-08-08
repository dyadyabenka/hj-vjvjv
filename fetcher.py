"""Сбор статей из RSS-лент.

Публичный вход — fetch_for_channel(sources, keywords, config, cache).
Возвращает список Article для одного канала, уже отфильтрованный по
возрасту и ключевым словам. cache общий на весь прогон (все каналы) —
чтобы одна и та же лента не скачивалась повторно для каждого канала.
"""

import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import feedparser
import requests

log = logging.getLogger(__name__)

# Некоторые сайты отдают 403 на запрос без User-Agent
USER_AGENT = "Mozilla/5.0 (compatible; RSSNewsBot/1.0)"

# Простейшая чистка HTML из описания: <p>, <a>, <img> и прочее нам не нужно
_TAG_RE = re.compile(r"<[^>]+>")
_SPACES_RE = re.compile(r"\s+")


@dataclass
class Article:
    """Одна статья из RSS-ленты."""

    title: str
    url: str
    summary: str
    source: str
    published_at: datetime  # всегда в UTC

    def as_prompt_block(self, index: int) -> str:
        """Оформление статьи для промпта синтеза."""
        return (
            f"[{index}] {self.title}\n"
            f"Источник: {self.source}\n"
            f"Ссылка: {self.url}\n"
            f"Текст: {self.summary or '(описание отсутствует)'}"
        )


def strip_html(raw: str) -> str:
    """Убирает HTML-теги и лишние пробелы из текста описания."""
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    return _SPACES_RE.sub(" ", text).strip()


def _parse_date(entry) -> datetime | None:
    """Достаёт дату публикации. feedparser уже приводит её к UTC."""
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return None


def _download(url: str, timeout: int) -> bytes | None:
    """Скачивает ленту. При любой сетевой ошибке возвращает None."""
    try:
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        log.warning("Не удалось скачать ленту %s: %s", url, exc)
        return None


def fetch_source(name: str, url: str, cfg: dict, cutoff: datetime) -> list[Article]:
    """Забирает статьи из одной ленты."""
    raw = _download(url, cfg["timeout_seconds"])
    if raw is None:
        return []

    feed = feedparser.parse(raw)
    if feed.bozo and not feed.entries:
        log.warning("Лента %s не распарсилась: %s", name, feed.bozo_exception)
        return []

    articles: list[Article] = []
    skipped_old = 0

    for entry in feed.entries[: cfg["max_articles_per_source"]]:
        link = (entry.get("link") or "").strip()
        title = strip_html(entry.get("title") or "")
        if not link or not title:
            continue

        published_at = _parse_date(entry)
        if published_at is None:
            # Даты нет — считаем статью свежей, иначе потеряем её навсегда
            published_at = datetime.now(timezone.utc)
        elif published_at < cutoff:
            skipped_old += 1
            continue

        summary = strip_html(
            entry.get("summary") or entry.get("description") or ""
        )[: cfg["description_max_chars"]]

        articles.append(
            Article(
                title=title,
                url=link,
                summary=summary,
                source=name,
                published_at=published_at,
            )
        )

    log.info(
        "Источник %-35s: взято %d, отброшено по возрасту %d",
        name,
        len(articles),
        skipped_old,
    )
    return articles


def _matches_keywords(article: Article, keywords: list[str]) -> bool:
    """True, если заголовок или описание содержат хотя бы одно ключевое слово."""
    if not keywords:
        return True
    haystack = f"{article.title} {article.summary}".lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def fetch_for_channel(
    sources: list[dict], keywords: list[str], config: dict, cache: dict[str, list[Article]]
) -> list[Article]:
    """Собирает и фильтрует статьи для одного канала.

    cache — словарь url -> список статей, общий на весь прогон (все каналы).
    Если несколько каналов используют одну и ту же ленту, она реально
    скачивается только один раз, а не по разу на каждый канал.
    """
    cfg = config["fetch"]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg["max_age_hours"])

    all_articles: list[Article] = []
    for source in sources:
        url = source["url"]
        if url not in cache:
            try:
                cache[url] = fetch_source(source["name"], url, cfg, cutoff)
            except Exception:  # noqa: BLE001 — одна битая лента не должна ронять запуск
                log.exception("Ошибка при обработке источника %s", source.get("name"))
                cache[url] = []
        all_articles.extend(cache[url])

    if keywords:
        filtered = [a for a in all_articles if _matches_keywords(a, keywords)]
        log.info(
            "После фильтра по ключевым словам: %d из %d", len(filtered), len(all_articles)
        )
        return filtered

    return all_articles
