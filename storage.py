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
    check_note    TEXT,  -- замечание от synthesizer.verify(), если проверка что-то нашла
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_status ON posts (status);
CREATE INDEX IF NOT EXISTS idx_posts_channel ON posts (channel_id);

CREATE TABLE IF NOT EXISTS style_examples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id  TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_style_examples_channel ON style_examples (channel_id);
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
        self._migrate()

    def _migrate(self) -> None:
        """Добавляет колонки, которых не было в более старой версии базы.

        CREATE TABLE IF NOT EXISTS не трогает уже существующую таблицу, так
        что новые поля в posts (например, check_note) нужно добавлять руками
        для баз, созданных до этого изменения. Ошибка "duplicate column"
        означает, что колонка уже есть — это нормальный случай, не проблема.
        """
        try:
            self.conn.execute("ALTER TABLE posts ADD COLUMN check_note TEXT")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # колонка уже существует

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
        check_note: str | None = None,
    ) -> int:
        """Сохраняет черновик поста со статусом pending_review. Возвращает id поста."""
        cursor = self.conn.execute(
            """
            INSERT INTO posts
                (channel_id, article_urls, articles_block, text, image_url, image_query,
                 check_note, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel_id,
                json.dumps(article_urls, ensure_ascii=False),
                articles_block,
                text,
                image_url,
                image_query,
                check_note,
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

    def get_pending_posts(self) -> list[dict]:
        """Все черновики, ещё ждущие решения (не опубликованы и не отменены).

        Нужно для команды /pending — переслать их заново с кнопками, если
        кнопки под исходным сообщением в Telegram почему-то пропали
        (например, из-за прежнего бага при неудачной публикации).
        """
        rows = self.conn.execute(
            "SELECT * FROM posts WHERE status = ? ORDER BY id", (POST_PENDING,)
        ).fetchall()
        posts = [dict(row) for row in rows]
        for post in posts:
            post["article_urls"] = json.loads(post["article_urls"])
        return posts

    def update_post_text(self, post_id: int, text: str) -> None:
        """Обновляет текст поста после доработки. check_note сбрасывается —
        замечание относилось к прежнему тексту, оставлять его при новом
        варианте было бы вводящей в заблуждение "залежавшейся" пометкой.
        """
        self.conn.execute(
            "UPDATE posts SET text = ?, check_note = NULL WHERE id = ?", (text, post_id)
        )
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

    def get_recent_published_texts(self, channel_id: str, days: int) -> list[str]:
        """Тексты постов этого канала, опубликованных за последние days дней.

        Нужно для проверки на повтор темы (clusterer.is_duplicate_of_recent) —
        сравниваем новый черновик с недавней историей публикаций, а не со
        всей историей канала, иначе со временем темы неизбежно повторятся
        (например, раз в год снова пишут про то же исследование).
        """
        cutoff = _to_db(datetime.now(timezone.utc) - timedelta(days=days))
        rows = self.conn.execute(
            "SELECT text FROM posts WHERE channel_id = ? AND status = ? AND created_at >= ?",
            (channel_id, POST_PUBLISHED, cutoff),
        ).fetchall()
        return [row["text"] for row in rows]

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

    # --- примеры постов для стиля (channels[].style_examples из бота) --------

    def add_style_example(self, channel_id: str, text: str) -> int:
        """Сохраняет присланный админом пост-образец для канала. Возвращает id."""
        cursor = self.conn.execute(
            "INSERT INTO style_examples (channel_id, text, created_at) VALUES (?, ?, ?)",
            (channel_id, text, _to_db(datetime.now(timezone.utc))),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_style_examples(self, channel_id: str) -> list[str]:
        """Все сохранённые примеры для канала, от старых к новым."""
        rows = self.conn.execute(
            "SELECT text FROM style_examples WHERE channel_id = ? ORDER BY id",
            (channel_id,),
        ).fetchall()
        return [row["text"] for row in rows]

    def count_style_examples(self) -> dict[str, int]:
        """Сколько примеров сохранено на каждый канал — для команды /examples."""
        rows = self.conn.execute(
            "SELECT channel_id, COUNT(*) AS count FROM style_examples GROUP BY channel_id"
        ).fetchall()
        return {row["channel_id"]: row["count"] for row in rows}

    def clear_style_examples(self, channel_id: str) -> int:
        """Удаляет все примеры канала. Возвращает, сколько удалено."""
        cursor = self.conn.execute(
            "DELETE FROM style_examples WHERE channel_id = ?", (channel_id,)
        )
        self.conn.commit()
        return cursor.rowcount
