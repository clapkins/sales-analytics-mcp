"""Проверки колонок, общие для скиллов.

Любой скилл, принимающий имя колонки от модели, обязан проверить его
одинаково: несуществующая колонка должна давать структурированную
ошибку со списком допустимых значений и подсказкой (инвариант №4,
CLAUDE.md). Правила живут здесь в одном экземпляре, а не копируются в
каждый скилл, которому понадобилась валидация.
"""

import pandas as pd

from core.errors import ErrorPayload, SkillError, unknown_value_error


def validate_column(df: pd.DataFrame, column: str) -> None:
    """Проверяет, что колонка существует в датасете.

    Args:
        df: Датафрейм.
        column: Имя колонки, переданное вызывающим.

    Raises:
        SkillError: С кодом ``"unknown_column"``, списком реальных
            колонок и, если удалось подобрать, ближайшим совпадением.
    """
    if column not in df.columns:
        raise unknown_value_error("unknown_column", "Колонка", column, list(df.columns))


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Возвращает имена настоящих числовых колонок.

    Булевы колонки исключаются: ``is_outlier`` после очистки для pandas
    числовая, но предлагать её как величину для графика или агрегации
    бессмысленно.

    Args:
        df: Датафрейм.

    Returns:
        Список имён числовых колонок.
    """
    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column])
    ]


def require_numeric_column(df: pd.DataFrame, column: str) -> None:
    """Проверяет, что колонка существует, числовая и не булева.

    Args:
        df: Датафрейм.
        column: Имя колонки, ожидаемой как числовая величина.

    Raises:
        SkillError: Если колонки нет или её тип не подходит; в списке
            допустимых значений — только настоящие числовые колонки.
    """
    validate_column(df, column)
    series = df[column]
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        raise SkillError(
            ErrorPayload(
                error_code="column_not_numeric",
                message=f"Колонка '{column}' не числовая.",
                allowed=numeric_columns(df),
            )
        )


def require_datetime_column(df: pd.DataFrame, column: str) -> None:
    """Проверяет, что колонка существует и имеет тип datetime64.

    Args:
        df: Датафрейм.
        column: Имя колонки, ожидаемой как ось времени.

    Raises:
        SkillError: Если колонки нет или её тип — не дата; в списке
            допустимых значений — только колонки с датами.
    """
    validate_column(df, column)
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        datetime_columns = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        raise SkillError(
            ErrorPayload(
                error_code="column_not_datetime",
                message=(
                    f"Колонка '{column}' не является датой, а для этой операции "
                    "нужна колонка с датами."
                ),
                allowed=datetime_columns,
            )
        )
