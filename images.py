"""Поиск реального фото по теме поста через Pexels API.

Pexels выбран из-за простой авторизации (один заголовок, без OAuth) и
щедрого бесплатного лимита. Поиск на русском Pexels понимает похуже, чем
на английском, поэтому запрос строится на английском по ключевым словам
темы канала — см. pick_query().
"""

import logging
import random

import requests

log = logging.getLogger(__name__)

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

# Соответствие русских ключевых слов темы (из config.yaml -> fetch.keywords)
# английским поисковым запросам для Pexels. Ключ ищется по вхождению в
# заголовок статьи; если ничего не подошло — берётся случайный запрос
# из fallback_queries в config.yaml.
_KEYWORD_TO_QUERY = {
    "паразит": "parasite science microscope",
    "гельминт": "parasite science microscope",
    "глист": "parasite science microscope",
    "инвази": "parasite science microscope",
    "микробиом": "gut microbiome bacteria",
    "микрофлор": "gut microbiome bacteria",
    "кишечн": "digestive health science",
    "пробиотик": "probiotics yogurt bacteria",
    "лактобактер": "probiotics lab bacteria",
    "бифидобактер": "probiotics lab bacteria",
    "дисбактериоз": "digestive health science",
}


def pick_query(article_title: str, config: dict) -> str:
    """Подбирает англоязычный поисковый запрос по заголовку статьи."""
    title_lower = article_title.lower()
    for keyword, query in _KEYWORD_TO_QUERY.items():
        if keyword in title_lower:
            return query

    fallback = config.get("images", {}).get("fallback_queries") or ["science laboratory"]
    return random.choice(fallback)


def search_image(query: str, api_key: str) -> str | None:
    """Возвращает URL картинки (large) по запросу или None, если не нашлось/ошибка."""
    try:
        response = requests.get(
            PEXELS_SEARCH_URL,
            headers={"Authorization": api_key},
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Ошибка запроса к Pexels ('%s'): %s", query, exc)
        return None

    try:
        photos = response.json().get("photos") or []
    except ValueError:
        log.warning("Pexels вернул не-JSON ответ")
        return None

    if not photos:
        log.info("Pexels не нашёл фото по запросу '%s'", query)
        return None

    return photos[0]["src"]["large"]
