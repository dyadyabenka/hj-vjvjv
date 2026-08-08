"""Учёт статей в SQLite.

Задача модуля — помнить, что мы уже видели и что уже опубликовали,
чтобы один и тот же материал не ушёл в канал дважды.

Жизненный цикл статьи:
    new       — только что найдена, ждёт обработки
    processed — обработана, но в пост не попала (устарела или не попала в топ)
    published — попала в опубликованный пост
"""

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone

from fetcher import Article

log = logging.getLogger(__name__)

STATUS_NEW = "new"
STATUS_PROCESSED = "processed"
STATUS_PUBLISHED = "published"

POST_PENDING = "pending_review"
POST_PUBLISHED = "published"
POST_REJECTED = "rejected"

# Формат дат в базе: фиксированная длина, чтобы строки корректно сравнивались
_DB_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    url          TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    title_hash   TEXT NOT NULL,
    summary      TEXT,
    source       TEXT,
    published_at TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    status       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_title_hash ON articles (title_hash);
CREATE INDEX IF NOT EXISTS idx_articles_status ON articles (status);

CREATE TABLE IF NOT EXISTS posts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id    TEXT NOT NULL,  -- id канала из config.yaml (channels[].id)
    article_urls  TEXT NOT NULL,   -- JSON-список ссылок статей-источников
    articles_block TEXT NOT NULL,  -- текст статей, как он ушёл в промпт (нужен для доработки)
    text          TEXT NOT NULL,
    image_url     TEXT,
    image_query   TEXT,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_status ON posts (status);
CREATE INDEX IF NOT EXISTS idx_posts_channel ON posts (channel_id);
"""

_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")


def title_hash(title: str) -> str:
    """Хэш нормализованного заголовка.

    Нормализация нужна, чтобы «Компания X купила Y» и «Компания X купила Y!»
    считались одной и той же новостью.
    """
    normalized = _NON_WORD_RE.sub(" ", title.lower())
    normalized = _SPACES_RE.sub(" ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _to_db(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime(_DB_TIME_FORMAT)


def _from_db(value: str) -> datetime:
    return datetime.strptime(value, _DB_TIME_FORMAT).replace(tzinfo=timezone.utc)


class Storage:
    """Тонкая обёртка над SQLite. Используется как контекстный менеджер."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # --- контекстный менеджер -------------------------------------------------

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # --- запись ---------------------------------------------------------------

    def save_new(self, articles: list[Article]) -> int:
        """Сохраняет статьи, которых ещё нет в базе.

        Дубликаты отсекаются двумя способами: по url (PRIMARY KEY)
        и по хэшу заголовка (одна новость на разных сайтах).
        Возвращает количество реально добавленных записей.
        """
        now = _to_db(datetime.now(timezone.utc))
        added = 0

        for article in articles:
            digest = title_hash(article.title)
            if self._exists(article.url, digest):
                continue

            self.conn.execute(
                """
                INSERT OR IGNORE INTO articles
                    (url, title, title_hash, summary, source,
                     published_at, fetched_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.url,
                    article.title,
                    digest,
                    article.summary,
                    article.source,
                    _to_db(article.published_at),
                    now,
                    STATUS_NEW,
                ),
            )
            added += 1

        self.conn.commit()
        return added

    def _exists(self, url: str, digest: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM articles WHERE url = ? OR title_hash = ? LIMIT 1",
            (url, digest),
        ).fetchone()
        return row is not None

    def _set_status(self, urls: list[str], status: str) -> None:
        if not urls:
            return
        self.conn.executemany(
            "UPDATE articles SET status = ? WHERE url = ?",
            [(status, url) for url in urls],
        )
        self.conn.commit()

    def mark_published(self, urls: list[str]) -> None:
        self._set_status(urls, STATUS_PUBLISHED)

    def mark_processed(self, urls: list[str]) -> None:
        self._set_status(urls, STATUS_PROCESSED)

    def expire_old(self, max_age_hours: int) -> int:
        """Помечает залежавшиеся «новые» статьи как обработанные.

        Иначе статья, не попавшая в топ за несколько запусков, висела бы
        в очереди вечно.
        """
        cutoff = _to_db(datetime.now(timezone.utc) - timedelta(hours=max_age_hours))
        cursor = self.conn.execute(
            "UPDATE articles SET status = ? WHERE status = ? AND published_at < ?",
            (STATUS_PROCESSED, STATUS_NEW, cutoff),
        )
        self.conn.commit()
        return cursor.rowcount

    # --- чтение ---------------------------------------------------------------

    def load_pending(self, max_age_hours: int, keywords: list[str] | None = None) -> list[Article]:
        """Статьи со статусом new и свежее указанного возраста.

        keywords — если задан, дополнительно фильтрует по вхождению хотя бы
        одного слова в заголовок/описание (регистр не важен). Нужно, потому
        что таблица articles общая на все каналы: статья, ещё не разобранная
        в пост в свой прошлый прогон, иначе могла бы "утечь" не в свой канал.
        """
        cutoff = _to_db(datetime.now(timezone.utc) - timedelta(hours=max_age_hours))
        rows = self.conn.execute(
            """
            SELECT url, title, summary, source, published_at
            FROM articles
            WHERE status = ? AND published_at >= ?
            ORDER BY published_at DESC
            """,
            (STATUS_NEW, cutoff),
        ).fetchall()

        articles = [
            Article(
                title=row["title"],
                url=row["url"],
                summary=row["summary"] or "",
                source=row["source"] or "",
                published_at=_from_db(row["published_at"]),
            )
            for row in rows
        ]

        if not keywords:
            return articles

        keywords_lower = [k.lower() for k in keywords]
        return [
            a for a in articles
            if any(k in f"{a.title} {a.summary}".lower() for k in keywords_lower)
        ]

    def stats(self) -> dict[str, int]:
        """Сводка по статусам — для лога в конце запуска."""
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS count FROM articles GROUP BY status"
        ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    # --- посты на модерации ----------------------------------------------------

    def create_post(
        self,
        channel_id: str,
        article_urls: list[str],
        articles_block: str,
        text: str,
        image_url: str | None,
        image_query: str | None,
    ) -> int:
        """Сохраняет черновик поста со статусом pending_review. Возвращает id поста."""
        cursor = self.conn.execute(
            """
            INSERT INTO posts
                (channel_id, article_urls, articles_block, text, image_url, image_query,
                 status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel_id,
                json.dumps(article_urls, ensure_ascii=False),
                articles_block,
                text,
                image_url,
                image_query,
                POST_PENDING,
                _to_db(datetime.now(timezone.utc)),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_post(self, post_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM posts WHERE id = ?", (post_id,)
        ).fetchone()
        if row is None:
            return None
        post = dict(row)
        post["article_urls"] = json.loads(post["article_urls"])
        return post

    def update_post_text(self, post_id: int, text: str) -> None:
        self.conn.execute("UPDATE posts SET text = ? WHERE id = ?", (text, post_id))
        self.conn.commit()

    def update_post_image(self, post_id: int, image_url: str | None, image_query: str | None) -> None:
        self.conn.execute(
            "UPDATE posts SET image_url = ?, image_query = ? WHERE id = ?",
            (image_url, image_query, post_id),
        )
        self.conn.commit()

    def mark_post_published(self, post_id: int) -> None:
        post = self.get_post(post_id)
        self.conn.execute(
            "UPDATE posts SET status = ? WHERE id = ?", (POST_PUBLISHED, post_id)
        )
        self.conn.commit()
        if post:
            self.mark_published(post["article_urls"])

    def mark_post_rejected(self, post_id: int) -> None:
        """Отмена черновика кнопкой '❌ Отменить'.

        Статьи-источники остаются в статусе processed (mark_processed
        проставляется уже при создании черновика — см. main.py), так что
        отменённая тема сама по себе не всплывёт снова в следующем прогоне.
        """
        self.conn.execute(
            "UPDATE posts SET status = ? WHERE id = ?", (POST_REJECTED, post_id)
        )
        self.conn.commit()

    def get_used_image_urls(self) -> set[str]:
        """Все image_url, что уже стоят у каких-либо постов (любого статуса).

        Используется, чтобы не подбирать повторно уже показанную картинку —
        учитываются и опубликованные посты, и черновики на модерации
        (мало ли их несколько штук с одинаковым запросом одновременно).
        """
        rows = self.conn.execute(
            "SELECT DISTINCT image_url FROM posts WHERE image_url IS NOT NULL"
        ).fetchall()
        return {row["image_url"] for row in rows}
