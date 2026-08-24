"""Единый формат ошибок для LLM.

Инструменты MCP никогда не пробрасывают голые исключения наружу
(архитектурный инвариант №4, CLAUDE.md): вместо трейсбека модель
получает структурированный ответ, по которому может исправиться на
следующем шаге без участия пользователя.

Формат ошибки специально совпадает по форме с обычным успешным
результатом инструмента — плоский словарь с ``ok: bool`` — а не с
протокольной ошибкой MCP. Если ошибка уйдёт как сбой протокола,
клиент покажет красную плашку, а модель не увидит ни текст, ни
suggestion. Поэтому слой MCP-инструментов (этап 5) обязан перехватывать
``SkillError`` и возвращать ``payload.to_dict()`` как обычный результат
вызова.
"""

from dataclasses import dataclass, field
from difflib import get_close_matches


@dataclass
class ErrorPayload:
    """Структурированная ошибка, которую видит LLM вместо трейсбека.

    Attributes:
        error_code: Машиночитаемый код ошибки, например
            ``"unknown_dataset_id"``.
        message: Описание проблемы, понятное модели.
        allowed: Список допустимых значений, если применимо.
        suggestion: Ближайшее допустимое значение, если удалось
            подобрать по расстоянию редактирования.
        ok: Всегда ``False`` — присутствует, чтобы форма ошибки
            совпадала с формой успешного результата инструмента.
    """

    error_code: str
    message: str
    allowed: list[str] | None = None
    suggestion: str | None = None
    ok: bool = field(default=False, init=False)

    def to_dict(self) -> dict[str, object]:
        """Сериализует ошибку в словарь — обычный результат инструмента.

        Returns:
            Словарь с ключами ``ok``, ``error_code``, ``message`` и,
            если заданы, ``allowed`` и ``suggestion``.
        """
        payload: dict[str, object] = {
            "ok": self.ok,
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.allowed is not None:
            payload["allowed"] = self.allowed
        if self.suggestion is not None:
            payload["suggestion"] = self.suggestion
        return payload


class SkillError(Exception):
    """Исключение-носитель структурированной ошибки для LLM.

    Скиллы поднимают это исключение вместо ручной сборки словаря
    ошибки на месте, сохраняя обычный control flow (``raise``). Слой
    MCP-инструментов обязан перехватывать ``SkillError`` и возвращать
    ``payload.to_dict()`` как результат вызова, а не как протокольный
    сбой.

    Attributes:
        payload: Структурированное содержимое ошибки.
    """

    def __init__(self, payload: ErrorPayload) -> None:
        """Создаёт исключение с готовым содержимым ошибки.

        Args:
            payload: Структурированное описание того, что пошло не так.
        """
        super().__init__(payload.message)
        self.payload = payload


def suggest_closest(value: str, allowed: list[str]) -> str | None:
    """Подбирает ближайшее допустимое значение к введённому.

    Args:
        value: Значение, которое не нашлось среди допустимых.
        allowed: Список допустимых значений.

    Returns:
        Ближайшее совпадение или ``None``, если ничего похожего нет.
    """
    matches = get_close_matches(value, allowed, n=1, cutoff=0.6)
    return matches[0] if matches else None


def unknown_value_error(error_code: str, label: str, value: str, allowed: list[str]) -> SkillError:
    """Строит SkillError для случая «значение не входит в допустимые».

    Используется везде, где модель могла передать не тот идентификатор
    или имя колонки — например, ``dataset_id`` или колонку ``'Sale'``
    вместо ``'Sales'``.

    Args:
        error_code: Машиночитаемый код ошибки, например
            ``"unknown_column"``.
        label: Человекочитаемое обозначение того, что не найдено,
            например ``"Колонка"`` или ``"Идентификатор датасета"``.
        value: Введённое (некорректное) значение.
        allowed: Список допустимых значений.

    Returns:
        SkillError с заданным кодом, списком допустимых значений и,
        если удалось подобрать, полем suggestion.
    """
    suggestion = suggest_closest(value, allowed)
    message = f"'{value}' — недопустимое значение параметра «{label}»."
    if suggestion:
        message += f" Возможно, имелось в виду '{suggestion}'."
    return SkillError(
        ErrorPayload(
            error_code=error_code,
            message=message,
            allowed=allowed,
            suggestion=suggestion,
        )
    )
