"""Поиск реального фото по теме поста через Pexels API.

Pexels выбран из-за простой авторизации (один заголовок, без OAuth) и
щедрого бесплатного лимита. Поиск на русском Pexels понимает похуже, чем
на английском, поэтому запрос строится на английском по ключевым словам
темы канала — см. pick_query().

search_image() умеет пропускать уже использованные фото (exclude_urls),
чтобы одна и та же картинка не повторялась в разных постах канала.
"""

import logging
import random

import requests

log = logging.getLogger(__name__)

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

# Сколько результатов запрашивать у Pexels за раз, чтобы было из чего
# выбрать не использованную ранее картинку (сам по себе запрос бесплатный —
# ограничение только по количеству запросов в час, не по объёму ответа).
RESULTS_PER_QUERY = 20

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
    "бактери": "bacteria microscope science",
    "иммунитет": "immune system science",
    "иммунн": "immune system science",
    "вирус": "virus microscope science",
    "инфекц": "infection medical science",
    "микроб": "microbe microscope science",
    "антибиотик": "antibiotics pills lab",
    "пищеварен": "digestive health science",
    "жкт": "digestive health science",
    "простейш": "microorganism microscope science",
    # Сон
    "сон": "sleep science bedroom night",
    "бессонниц": "insomnia sleepless night",
    "циркадн": "circadian rhythm clock night",
    "мелатонин": "melatonin sleep supplement",
    "засыпан": "person sleeping peaceful",
    "просыпа": "waking up morning bedroom",
    "недосып": "sleep deprivation tired",
    "сновиден": "dream sleep science",
    # Питание
    "питани": "healthy food nutrition science",
    "диет": "diet food plate healthy",
    "витамин": "vitamins supplements health",
    "нутриент": "nutrition food science",
    "калори": "calories food measurement",
    "белк": "protein food nutrition",
    "углевод": "carbohydrates food nutrition",
    "рацион": "healthy meal plate food",
    "сахар": "sugar food health",
    "клетчатк": "fiber vegetables food",
    # Крипта
    "биткоин": "bitcoin cryptocurrency coin",
    "bitcoin": "bitcoin cryptocurrency coin",
    "крипто": "cryptocurrency blockchain digital",
    "блокчейн": "blockchain technology network",
    "эфириум": "ethereum cryptocurrency coin",
    "ethereum": "ethereum cryptocurrency coin",
    "альткоин": "cryptocurrency coins altcoin",
    "токен": "crypto token digital asset",
    "майнинг": "crypto mining hardware",
    "биржа": "crypto trading exchange chart",
    "defi": "defi decentralized finance",
    "nft": "nft digital art blockchain",
    "стейблкоин": "stablecoin digital currency",
    "web3": "web3 blockchain technology",
}


def pick_query(article_title: str, fallback_queries: list[str]) -> str:
    """Подбирает англоязычный поисковый запрос по заголовку статьи.

    fallback_queries — список запросов-заглушек конкретного канала
    (channels[].fallback_image_queries в config.yaml), берётся, если по
    заголовку не нашлось совпадения по ключевым словам.
    """
    title_lower = article_title.lower()
    for keyword, query in _KEYWORD_TO_QUERY.items():
        if keyword in title_lower:
            return query

    queries = fallback_queries or ["science laboratory research"]
    return random.choice(queries)


def search_image(query: str, api_key: str, exclude_urls: set[str] | None = None) -> str | None:
    """Возвращает URL картинки (large) по запросу, пропуская уже использованные.

    exclude_urls — множество image_url, которые уже стоят у других постов
    (обычно вся история из storage.get_used_image_urls() плюс картинки,
    выбранные ранее в этом же прогоне). Если все найденные фото уже
    использованы — возвращается None (пост в этом случае уйдёт без фото,
    это не считается ошибкой).
    """
    exclude_urls = exclude_urls or set()

    try:
        response = requests.get(
            PEXELS_SEARCH_URL,
            headers={"Authorization": api_key},
            params={"query": query, "per_page": RESULTS_PER_QUERY, "orientation": "landscape"},
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

    for photo in photos:
        url = photo["src"]["large"]
        if url not in exclude_urls:
            return url

    log.info(
        "Все %d фото по запросу '%s' уже использовались ранее — пост будет без картинки",
        len(photos),
        query,
    )
    return None
