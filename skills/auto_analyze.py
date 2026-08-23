"""Скилл автоматического выбора типа графика по типу колонки.

Не рисует ничего сам — решает, какой из существующих чарт-скиллов
вызвать (``TrendChartSkill``, ``DistributionChartSkill``,
``TopNChartSkill``), и подбирает вторую колонку, если она нужна для
графика, но не указана явно. Так модель может попросить «покажи
что-нибудь про Region» без явного выбора типа графика и второй
колонки.
"""

from typing import Any

import pandas as pd

from core.errors import ErrorPayload, SkillError, unknown_value_error
from core.registry import BaseSkill, register_skill
from skills.visualization import DistributionChartSkill, TopNChartSkill, TrendChartSkill

_DEFAULT_TOP_N = 10


def _first_numeric_column(df: pd.DataFrame, exclude: str) -> str | None:
    """Находит первую числовую (не булеву) колонку, кроме исключённой.

    Args:
        df: Датафрейм.
        exclude: Колонка, которую не следует предлагать самой себе в
            пару (например, если сама она уже числовая).

    Returns:
        Имя колонки или ``None``, если подходящей нет.
    """
    for column in df.columns:
        if column == exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
            return column
    return None


@register_skill
class AutoAnalyzeSkill(BaseSkill):
    """Сама выбирает тип графика по типу указанной колонки и строит его."""

    name = "auto_analyze"
    description = (
        "Автоматически выбирает подходящий тип графика по типу колонки "
        "и строит его: колонка с датами → тренд по числовой колонке, "
        "числовая колонка → гистограмма распределения, категориальная "
        "колонка → топ-N по числовой колонке. Вторая колонка (для "
        "тренда и топ-N) подбирается автоматически. Используй, когда "
        "не важен конкретный тип графика — важно быстро увидеть что-то "
        "содержательное про колонку."
    )

    def run(self, dataset_id: str, column: str) -> dict[str, Any]:
        """Строит график, тип которого выбран по типу колонки.

        Args:
            dataset_id: Идентификатор датасета в session store.
            column: Колонка, для которой нужно подобрать график.

        Returns:
            Результат соответствующего чарт-скилла (``chart_type``,
            ``path``, ``description``) с добавленным полем
            ``auto_selected``, поясняющим сделанный выбор.

        Raises:
            SkillError: Если dataset_id или колонка неизвестны, либо
                для тренда/топ-N не нашлось ни одной числовой колонки
                в пару.
        """
        df = self.session.get(dataset_id)
        if column not in df.columns:
            raise unknown_value_error("unknown_column", "Колонка", column, list(df.columns))

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            y_col = _first_numeric_column(df, exclude=column)
            if y_col is None:
                raise SkillError(
                    ErrorPayload(
                        error_code="no_numeric_column_for_trend",
                        message="Для графика тренда по датам нужна хотя бы одна числовая колонка.",
                    )
                )
            result = TrendChartSkill(self.session).run(dataset_id, x_col=column, y_col=y_col)
            result["auto_selected"] = {"chart_type": "trend", "paired_column": y_col}
            return result

        if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
            result = DistributionChartSkill(self.session).run(dataset_id, column=column)
            result["auto_selected"] = {"chart_type": "distribution"}
            return result

        value_col = _first_numeric_column(df, exclude=column)
        if value_col is None:
            raise SkillError(
                ErrorPayload(
                    error_code="no_numeric_column_for_top_n",
                    message="Для топ-N по категории нужна хотя бы одна числовая колонка.",
                )
            )
        result = TopNChartSkill(self.session).run(
            dataset_id, category_col=column, value_col=value_col, n=_DEFAULT_TOP_N, agg="sum"
        )
        result["auto_selected"] = {"chart_type": "top_n", "paired_column": value_col}
        return result
