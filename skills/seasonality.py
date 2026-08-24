"""Скилл анализа сезонности по месяцам и кварталам.

Добавлен последним, отдельно от остальных, как проверка заявления о
расширяемости: новый инструмент — это один файл в ``skills/``, без
единой правки в ``core/`` и в точках входа сервера. Дифф коммита,
которым добавлен этот файл, — само доказательство.

Отвечает на вопрос «когда продаём», который в демонстрационном прогоне
модель вынуждена была собирать по крупицам из описания графика тренда:
там сезонность упоминается одной фразой про сильнейший и слабейший
календарный месяц, без чисел по остальным.
"""

from typing import Any

import pandas as pd

from core.registry import BaseSkill, register_skill
from core.serialization import to_json_safe
from core.validation import require_datetime_column, require_numeric_column

_MONTH_NAMES = (
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
)


def _period_breakdown(totals_by_period: pd.Series, labels: dict[int, str]) -> list[dict[str, Any]]:
    """Превращает суммы по периодам в список записей с долей от целого.

    Args:
        totals_by_period: Суммы, индексированные номером периода.
        labels: Человекочитаемые названия периодов по их номеру.

    Returns:
        Список записей ``{"period", "total", "share_pct"}``,
        отсортированный по возрастанию номера периода.
    """
    grand_total = totals_by_period.sum()
    return [
        {
            "period": labels[int(number)],
            "total": round(float(total), 2),
            "share_pct": round(float(total) / grand_total * 100, 1) if grand_total else 0.0,
        }
        for number, total in totals_by_period.sort_index().items()
    ]


@register_skill
class SeasonalitySkill(BaseSkill):
    """Считает сезонность: распределение показателя по месяцам и кварталам."""

    name = "analyze_seasonality"
    description = (
        "Анализирует сезонность: суммирует числовую колонку по "
        "календарным месяцам и кварталам за весь период, показывает "
        "долю каждого месяца и квартала, называет сильнейший и "
        "слабейший период и отношение между ними. Отвечает на вопрос "
        "«когда продаём», в отличие от plot_trend, который показывает "
        "динамику по календарной оси. Графика не строит — возвращает "
        "числа."
    )

    def run(self, dataset_id: str, date_col: str, value_col: str) -> dict[str, Any]:
        """Считает сезонность показателя по месяцам и кварталам.

        Args:
            dataset_id: Идентификатор датасета в session store.
            date_col: Колонка с датами (например, Date).
            value_col: Числовая колонка, сезонность которой считается
                (например, Sales).

        Returns:
            Словарь с разбивкой ``by_month`` и ``by_quarter``, именами
            сильнейшего и слабейшего месяца, отношением пика к провалу и
            числом охваченных лет.

        Raises:
            SkillError: Если dataset_id неизвестен, ``date_col`` не
                является датой или ``value_col`` не числовая.
        """
        df = self.session.get(dataset_id)
        require_datetime_column(df, date_col)
        require_numeric_column(df, value_col)

        dates = df[date_col]
        values = df[value_col]

        by_month_totals = values.groupby(dates.dt.month).sum()
        by_quarter_totals = values.groupby(dates.dt.quarter).sum()

        month_labels = {number: _MONTH_NAMES[number - 1] for number in range(1, 13)}
        quarter_labels = {number: f"Q{number}" for number in range(1, 5)}

        peak_month_number = int(by_month_totals.idxmax())
        trough_month_number = int(by_month_totals.idxmin())
        peak_value = float(by_month_totals.max())
        trough_value = float(by_month_totals.min())

        return to_json_safe(
            {
                "ok": True,
                "dataset_id": dataset_id,
                "value_col": value_col,
                "years_covered": int(dates.dt.year.nunique()),
                "by_month": _period_breakdown(by_month_totals, month_labels),
                "by_quarter": _period_breakdown(by_quarter_totals, quarter_labels),
                "peak_month": month_labels[peak_month_number],
                "trough_month": month_labels[trough_month_number],
                "peak_to_trough_ratio": (
                    round(peak_value / trough_value, 2) if trough_value else None
                ),
            }
        )
