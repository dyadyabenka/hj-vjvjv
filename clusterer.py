"""Группировка статей об одном и том же событии.

Подход намеренно простой: TF-IDF по заголовкам + косинусное сходство.
Никаких embedding-моделей — не хочется тянуть torch ради склейки заголовков.

Важная деталь: анализатор символьный (char_wb, n-граммы 3-5), а не словарный.
Для русского это работает заметно лучше — «Яндекс купил» и «Яндекса покупка»
имеют общие символьные n-граммы, а как отдельные слова они не совпадут вовсе.
"""

import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fetcher import Article

log = logging.getLogger(__name__)


def _text_for_comparison(article: Article) -> str:
    """Что сравниваем: заголовок + начало описания.

    Заголовок берём дважды, чтобы он весил больше описания.
    """
    return f"{article.title} {article.title} {article.summary[:200]}"


def cluster_articles(articles: list[Article], config: dict) -> list[list[Article]]:
    """Разбивает статьи на группы. Группа из одной статьи — тоже валидная группа.

    Алгоритм — жадная кластеризация по принципу «одного связного соседа»:
    статья попадает в первую группу, где нашёлся достаточно похожий сосед.
    """
    cfg = config["clustering"]
    threshold = cfg["similarity_threshold"]
    max_size = cfg["max_articles_per_cluster"]

    if not articles:
        return []
    if len(articles) == 1:
        return [[articles[0]]]

    texts = [_text_for_comparison(article) for article in articles]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Бывает на совсем пустом словаре — тогда каждая статья сама по себе
        log.warning("TF-IDF не построился, каждая статья идёт отдельным постом")
        return [[article] for article in articles]

    similarity = cosine_similarity(matrix)

    # cluster_of[i] — индекс группы, куда попала статья i
    cluster_of: list[int] = [-1] * len(articles)
    clusters: list[list[int]] = []

    for i in range(len(articles)):
        if cluster_of[i] != -1:
            continue

        # Новая группа начинается с текущей статьи
        current = [i]
        cluster_of[i] = len(clusters)

        for j in range(i + 1, len(articles)):
            if cluster_of[j] != -1 or len(current) >= max_size:
                continue
            # Достаточно похожести хотя бы на одного члена группы
            if any(similarity[j][member] >= threshold for member in current):
                current.append(j)
                cluster_of[j] = len(clusters)

        clusters.append(current)

    result = [[articles[i] for i in group] for group in clusters]

    multi = sum(1 for group in result if len(group) > 1)
    log.info(
        "Кластеризация: %d статей -> %d групп (из них склеенных: %d)",
        len(articles),
        len(result),
        multi,
    )
    return result


def rank_clusters(clusters: list[list[Article]]) -> list[list[Article]]:
    """Сортировка групп по приоритету: сначала крупные, потом свежие.

    Событие, о котором написали три источника, интереснее одиночной заметки.
    """
    return sorted(
        clusters,
        key=lambda group: (
            len(group),
            max(article.published_at for article in group),
        ),
        reverse=True,
    )


def is_duplicate_of_recent(text: str, recent_texts: list[str], threshold: float) -> bool:
    """True, если text слишком похож на один из недавно ОПУБЛИКОВАННЫХ постов.

    Тот же принцип, что и в cluster_articles (символьные TF-IDF n-граммы),
    но здесь сравнивается не заголовок статьи, а сам готовый текст поста —
    защита от того, что похожая история всплывёт заново через недели и
    сгенерирует по сути повторный пост в канале.
    """
    if not recent_texts:
        return False

    texts = [text, *recent_texts]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return False

    similarity = cosine_similarity(matrix[0:1], matrix[1:])[0]
    best = float(similarity.max()) if len(similarity) else 0.0

    if best >= threshold:
        log.info("Пост похож на уже опубликованный (сходство %.2f >= %.2f) — пропускаем", best, threshold)
        return True
    return False
