"""Скилл базовой описательной статистики — владелец инструмента describe_data.

Классифицирует колонки по типу (числовые, категориальные, датовые) и
считает статистику, подходящую каждому типу. Ничего не интерпретирует
и не формулирует выводов — это задача ``InsightGenerationSkill`` и
самой LLM (инвариант №6, CLAUDE.md).
"""

from typing import Any

import pandas as pd

from core.registry import BaseSkill, register_skill
from core.serialization import to_json_safe
from core.session import SessionStore

_TOP_CATEGORIES = 5


def _numeric_stats(series: pd.Series) -> dict[str, float]:
    """Считает среднее, медиану, стандартное отклонение, квартили, min/max.

    Args:
        series: Числовая колонка.

    Returns:
        Словарь с базовой статистикой.
    """
    described = series.describe()
    return {
        "mean": described["mean"],
        "median": series.median(),
        "std": described["std"],
        "min": described["min"],
        "q25": described["25%"],
        "q50": described["50%"],
        "q75": described["75%"],
        "max": described["max"],
    }


def _categorical_stats(series: pd.Series) -> dict[str, Any]:
    """Считает число уникальных значений и топ-5 по частоте.

    Args:
        series: Категориальная (текстовая) колонка.

    Returns:
        Словарь с ``unique_count`` и списком ``top_values``.
    """
    top_counts = series.value_counts().head(_TOP_CATEGORIES)
    return {
        "unique_count": series.nunique(dropna=True),
        "top_values": [{"value": value, "count": count} for value, count in top_counts.items()],
    }


def _datetime_stats(series: pd.Series) -> dict[str, Any]:
    """Определяет диапазон периода для колонки с датами.

    Args:
        series: Колонка типа datetime64.

    Returns:
        Словарь с ``min_date`` и ``max_date``.
    """
    return {"min_date": series.min(), "max_date": series.max()}


@register_skill
class DescriptiveStatsSkill(BaseSkill):
    """Считает базовую статистику по датасету — по типу каждой колонки."""

    name = "describe_data"
    description = (
        "Возвращает базовую статистику по датасету: для числовых "
        "колонок — среднее, медиану, стандартное отклонение, квартили, "
        "min/max; для категориальных — число уникальных значений и "
        "топ-5 по частоте; для колонок с датами — диапазон периода. "
        "Принимает dataset_id, а не сами данные."
    )

    def __init__(self, session: SessionStore) -> None:
        """Инициализирует скилл с хранилищем датасетов.

        Args:
            session: Хранилище, откуда берётся датасет по его id.
        """
        self.session = session

    def run(self, dataset_id: str) -> dict[str, Any]:
        """Считает статистику по датасету, сгруппированную по типу колонки.

        Args:
            dataset_id: Идентификатор датасета в session store.

        Returns:
            Словарь с ``numeric``, ``categorical`` и ``datetime``
            разделами статистики по соответствующим колонкам.

        Raises:
            SkillError: Если dataset_id неизвестен (поднимается
                session store).
        """
        df = self.session.get(dataset_id)

        numeric: dict[str, Any] = {}
        categorical: dict[str, Any] = {}
        datetime_stats: dict[str, Any] = {}

        for column in df.columns:
            series = df[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_stats[column] = _datetime_stats(series)
            elif pd.api.types.is_bool_dtype(series):
                # Булева колонка (например, is_outlier после очистки) —
                # is_numeric_dtype(bool) тоже True, но describe() для
                # bool возвращает count/unique/top/freq, а не mean/std,
                # поэтому её нужно отличать от чисел раньше, чем от них.
                categorical[column] = _categorical_stats(series)
            elif pd.api.types.is_numeric_dtype(series):
                numeric[column] = _numeric_stats(series)
            else:
                categorical[column] = _categorical_stats(series)

        result = {
            "ok": True,
            "dataset_id": dataset_id,
            "rows": df.shape[0],
            "numeric": numeric,
            "categorical": categorical,
            "datetime": datetime_stats,
        }
        return to_json_safe(result)
