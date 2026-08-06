"""Синтез и доработка поста через Anthropic-совместимый API (Claude или DeepSeek).

synthesize()  — из группы статей делает первый черновик поста.
revise()      — переписывает уже существующий черновик с учётом замечаний
                админа (кнопка "доработать" в модерации).

Любая ошибка API возвращает None: вызывающий код просто пропускает эту группу
(для synthesize) или сообщает об ошибке админу (для revise).
"""

import logging

import anthropic

from fetcher import Article

log = logging.getLogger(__name__)

# DeepSeek отдаёт Anthropic-совместимый эндпоинт — значит, тот же SDK
# и тот же формат messages.create() работают без изменений,
# нужно только подменить base_url и ключ.
DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"

SUPPORTED_PROVIDERS = ("anthropic", "deepseek")

SYNTHESIS_PROMPT_TEMPLATE = """Ты — редактор научно-популярного Telegram-канала на тему: {topic}.

Вот несколько статей по одной теме из разных источников:

{articles}

Задача:
1. Напиши связный пост (150-300 слов) своими словами, объединяющий факты из всех источников.
2. НЕ копируй формулировки дословно — только пересказ своими словами.
3. Если источники расходятся в фактах или оценках — отметь это явно ("по разным данным...").
4. Тема медицинская — будь особенно аккуратен: не давай прямых рекомендаций
   "принимайте X" или "лечите Y с помощью Z", только пересказывай, что показало
   исследование. Если статья описывает предварительное или единичное
   исследование (а не устоявшийся научный консенсус) — прямо укажи это
   словами вроде "по предварительным данным" или "авторы одного исследования".
5. Тон: нейтральный, информативный, без желтизны и кликбейта, без "чудо-открытий".
6. Не добавляй вступление вроде "Вот пост" — сразу текст поста.
7. Первая строка — короткий заголовок (до 80 символов), с одним уместным эмодзи в начале.
8. Затем пустая строка и текст поста. Внутри текста можно использовать 2-4 emoji
   к месту (не в каждом предложении), чтобы пост легче читался в Telegram.
9. В конце — блок "Источники:" со списком ссылок.
10. Пиши по-русски, даже если исходные статьи на английском.

Формат ответа: только готовый текст поста, без пояснений от себя."""

REVISE_PROMPT_TEMPLATE = """Ты — редактор научно-популярного Telegram-канала на тему: {topic}.

Вот исходные статьи, на основе которых написан пост:

{articles}

Вот текущий черновик поста:

---
{previous_post}
---

Редактор канала (админ) попросил внести правку: "{edit_note}"

Перепиши пост с учётом этой правки, но не выходи за рамки фактов из исходных
статей выше — если просьба требует информации, которой нет в статьях, сделай
лучшее, что можно в рамках имеющихся фактов, и не выдумывай данные.

Требования к формату те же, что и всегда:
- своими словами, без дословного копирования источников
- первая строка — короткий заголовок с одним emoji, затем пустая строка
- 2-4 emoji по тексту к месту, без перебора
- в конце блок "Источники:" со списком ссылок
- по-русски

Формат ответа: только готовый текст поста, без пояснений от себя."""


def build_articles_block(cluster: list[Article]) -> str:
    """Форматирует статьи для промпта. Этот же текст сохраняется в базе,
    чтобы revise() мог позже переписать пост, не имея на руках объектов Article.
    """
    return "\n\n".join(
        article.as_prompt_block(i) for i, article in enumerate(cluster, start=1)
    )


class Synthesizer:
    """Обёртка над Anthropic-совместимым API (Anthropic или DeepSeek)."""

    def __init__(self, config: dict, api_key: str):
        cfg = config["synthesis"]
        self.provider = cfg.get("provider", "anthropic").strip().lower()

        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Неизвестный provider '{self.provider}' в config.yaml. "
                f"Допустимые значения: {', '.join(SUPPORTED_PROVIDERS)}"
            )

        if self.provider == "deepseek":
            # Тот же клиент anthropic.Anthropic, просто с другим base_url —
            # DeepSeek специально сделал эндпоинт совместимым, чтобы
            # не нужно было переписывать код под их SDK.
            self.client = anthropic.Anthropic(
                api_key=api_key, base_url=DEEPSEEK_BASE_URL
            )
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

        self.model = cfg["model"]
        self.max_tokens = cfg["max_tokens"]
        self.topic = config["channel_topic"]

        log.info("Синтезатор: провайдер=%s, модель=%s", self.provider, self.model)

    def synthesize(self, cluster: list[Article]) -> tuple[str, str] | None:
        """Первый черновик поста из группы статей.

        Возвращает (текст_поста, articles_block) или None при ошибке.
        articles_block нужно сохранить в storage — он понадобится revise().
        """
        articles_block = build_articles_block(cluster)
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(topic=self.topic, articles=articles_block)

        text = self._complete(prompt)
        if text is None:
            return None
        return text, articles_block

    def revise(self, articles_block: str, previous_post: str, edit_note: str) -> str | None:
        """Переписывает пост с учётом замечаний админа. Возвращает новый текст или None."""
        prompt = REVISE_PROMPT_TEMPLATE.format(
            topic=self.topic,
            articles=articles_block,
            previous_post=previous_post,
            edit_note=edit_note,
        )
        return self._complete(prompt)

    def _complete(self, prompt: str) -> str | None:
        """Общая логика вызова модели и разбора ответа — используется и synthesize, и revise."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AuthenticationError:
            key_name = "DEEPSEEK_API_KEY" if self.provider == "deepseek" else "ANTHROPIC_API_KEY"
            log.error("Неверный %s — проверь .env", key_name)
            return None
        except anthropic.RateLimitError:
            # SDK сам делает несколько повторов; сюда попадаем, когда они кончились
            log.error("Лимит запросов к API (%s) исчерпан", self.provider)
            return None
        except anthropic.APIConnectionError as exc:
            log.error("Нет связи с API (%s): %s", self.provider, exc)
            return None
        except anthropic.APIStatusError as exc:
            log.error(
                "API (%s) вернул ошибку %s: %s", self.provider, exc.status_code, exc.message
            )
            return None

        if response.stop_reason == "refusal":
            log.warning("Модель отказалась писать/переписывать пост")
            return None
        if response.stop_reason == "max_tokens":
            log.warning(
                "Ответ обрезан по max_tokens (%d) — увеличь лимит в config.yaml",
                self.max_tokens,
            )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        if not text:
            log.warning("Модель вернула пустой ответ")
            return None

        log.info(
            "Готово: %d символов (токенов: %d вход / %d выход)",
            len(text),
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return text
