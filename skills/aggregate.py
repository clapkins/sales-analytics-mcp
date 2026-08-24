"""Скилл сводных таблиц без построения графика.

Появился по итогам демонстрационного прогона: чтобы посчитать маржу по
товарам, модели пришлось дважды вызвать ``plot_top_n`` (по выручке и по
прибыли) и делить числа вручную — то есть сгенерировать два PNG ради
арифметики. Инструмент закрывает самый частый аналитический вопрос
(«а сколько X в разрезе Y») напрямую и заметно дешевле по токенам, чем
картинка с описанием.

Принимает сразу несколько числовых колонок: маржа по товарам считается
одним вызовом с ``value_cols=["Sales", "Profit"]``, а не двумя.
"""

from typing import Any

from core.errors import unknown_value_error
from core.registry import BaseSkill, register_skill
from core.serialization import to_json_safe
from core.validation import require_numeric_column, validate_column

_AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count")
_DEFAULT_ROW_LIMIT = 20


@register_skill
class AggregateSkill(BaseSkill):
    """Считает сводную таблицу по категориям без построения графика."""

    name = "aggregate"
    description = (
        "Считает агрегаты числовых колонок в разрезе категориальной "
        "колонки и возвращает таблицу числами, без построения графика. "
        "Используй, когда нужен сам расчёт, а не картинка — например, "
        "чтобы сравнить выручку и прибыль по товарам и посчитать маржу: "
        "один вызов с value_cols=['Sales', 'Profit'] вместо двух "
        "графиков. agg: sum, mean, median, min, max или count."
    )

    def run(
        self,
        dataset_id: str,
        group_by: str,
        value_cols: list[str],
        agg: str = "sum",
        limit: int = _DEFAULT_ROW_LIMIT,
    ) -> dict[str, Any]:
        """Группирует датасет и считает агрегаты по нескольким колонкам.

        Args:
            dataset_id: Идентификатор датасета в session store.
            group_by: Колонка, по которой группируются строки
                (например, Product или Region).
            value_cols: Числовые колонки, по которым считается агрегат.
            agg: Способ агрегации — ``"sum"`` (по умолчанию), ``"mean"``,
                ``"median"``, ``"min"``, ``"max"`` или ``"count"``.
            limit: Сколько групп вернуть, отсортировав по убыванию
                первой из ``value_cols``.

        Returns:
            Словарь с ``rows`` — списком записей вида
            ``{group_by: значение, колонка: агрегат, ...}`` — и полями
            ``groups_total`` (сколько групп всего) и ``truncated``.

        Raises:
            SkillError: Если dataset_id, колонки или способ агрегации
                неизвестны.
        """
        if agg not in _AGGREGATIONS:
            raise unknown_value_error(
                "invalid_aggregation", "Способ агрегации", agg, list(_AGGREGATIONS)
            )

        df = self.session.get(dataset_id)
        validate_column(df, group_by)
        if not value_cols:
            raise unknown_value_error(
                "empty_value_cols",
                "Список колонок для агрегации",
                "[]",
                [column for column in df.columns if column != group_by],
            )
        for column in value_cols:
            require_numeric_column(df, column)

        grouped = df.groupby(group_by)[value_cols].agg(agg)
        grouped = grouped.sort_values(value_cols[0], ascending=False)

        limited = grouped.head(limit)
        rows = [
            {group_by: index, **{column: row[column] for column in value_cols}}
            for index, row in limited.iterrows()
        ]

        return to_json_safe(
            {
                "ok": True,
                "dataset_id": dataset_id,
                "group_by": group_by,
                "agg": agg,
                "rows": rows,
                "groups_total": len(grouped),
                "truncated": len(grouped) > len(limited),
            }
        )
