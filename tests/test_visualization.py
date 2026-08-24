"""Тесты скилла визуализации.

Графики строятся на реальном очищенном датасете, а не на трёх точках:
только так видно, не наезжают ли подписи месяцев друг на друга, не
превращается ли топ-N в частокол и т.п. — визуальная часть чек-листа
CLAUDE.md проверяется отдельно, глазами, после этого прогона.
"""

from pathlib import Path

import pandas as pd
import pytest

from core.errors import SkillError
from core.session import SessionStore
from skills.cleaning import DataCleaningSkill
from skills.loading import DataLoadingSkill
from skills.visualization import (
    CorrelationChartSkill,
    DistributionChartSkill,
    TopNChartSkill,
    TrendChartSkill,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


@pytest.fixture
def cleaned_dataset_id() -> tuple[SessionStore, str]:
    """Реальный датасет: загружен и очищен, как в типичном сценарии диалога."""
    session = SessionStore()
    raw_id = DataLoadingSkill(session).run(str(DATA_PATH))["dataset_id"]
    clean_id = DataCleaningSkill(session).run(raw_id)["dataset_id"]
    return session, clean_id


def test_trend_chart_created_and_described(cleaned_dataset_id: tuple[SessionStore, str]) -> None:
    """plot_trend сохраняет PNG и описывает пик/минимум/изменение числами."""
    session, dataset_id = cleaned_dataset_id
    result = TrendChartSkill(session).run(dataset_id, x_col="Date", y_col="Sales")

    assert result["ok"] is True
    assert result["chart_type"] == "trend"
    assert Path(result["path"]).exists()
    assert "Sales" in result["description"]
    assert "%" in result["description"]


def test_trend_chart_rejects_non_datetime_x_col(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Нечисловая/недатовая x_col — структурированная ошибка, а не трейсбек."""
    session, dataset_id = cleaned_dataset_id
    with pytest.raises(SkillError) as exc_info:
        TrendChartSkill(session).run(dataset_id, x_col="Product", y_col="Sales")

    assert exc_info.value.payload.error_code == "column_not_datetime"


def test_distribution_chart_created_and_described(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """plot_distribution сохраняет PNG и упоминает диапазон значений."""
    session, dataset_id = cleaned_dataset_id
    result = DistributionChartSkill(session).run(dataset_id, column="Sales")

    assert result["chart_type"] == "distribution"
    assert Path(result["path"]).exists()
    assert "диапазоне" in result["description"]


def test_distribution_chart_rejects_categorical_column(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Категориальная колонка для гистограммы — структурированная ошибка."""
    session, dataset_id = cleaned_dataset_id
    with pytest.raises(SkillError) as exc_info:
        DistributionChartSkill(session).run(dataset_id, column="Region")

    assert exc_info.value.payload.error_code == "column_not_numeric"


def test_correlation_chart_created_and_describes_strongest_pair(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """correlation_analysis находит сильнее всего связанную пару колонок."""
    session, dataset_id = cleaned_dataset_id
    result = CorrelationChartSkill(session).run(dataset_id)

    assert result["chart_type"] == "correlation"
    assert Path(result["path"]).exists()
    # Sales и Profit связаны по построению данных (Profit = Sales * маржа) —
    # ожидаемо самая сильная пара в корреляционной матрице.
    assert "Sales" in result["description"]
    assert "Profit" in result["description"]


def test_correlation_description_lists_every_pair_without_nan(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Регрессия: описание перечисляет все пары и не содержит nan.

    Раньше называлась только сильнейшая пара, и отрицательная связь
    Quantity↔Profit была для модели невидима — она не видит саму карту.
    """
    session, dataset_id = cleaned_dataset_id
    description = CorrelationChartSkill(session).run(dataset_id)["description"]

    assert "nan" not in description.lower()
    # Три числовые колонки дают ровно три пары — все должны быть названы.
    assert description.count(":") >= 3
    assert "Quantity" in description


def test_correlation_chart_requires_two_numeric_columns() -> None:
    """Меньше двух числовых колонок — явная ошибка, а не пустой график."""
    session = SessionStore()
    dataset_id = session.put(pd.DataFrame({"value": [1, 2, 3], "label": ["a", "b", "c"]}), meta={})

    with pytest.raises(SkillError) as exc_info:
        CorrelationChartSkill(session).run(dataset_id)

    assert exc_info.value.payload.error_code == "not_enough_numeric_columns"


def test_top_n_chart_created_and_identifies_leader(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """plot_top_n находит лидера по сумме продаж среди продуктов."""
    session, dataset_id = cleaned_dataset_id
    result = TopNChartSkill(session).run(
        dataset_id, category_col="Product", value_col="Sales", n=5, agg="sum"
    )

    assert result["chart_type"] == "top_n"
    assert Path(result["path"]).exists()
    assert "%" in result["description"]


def test_top_n_description_lists_full_ranking_not_just_leader(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Регрессия: описание перечисляет весь рейтинг, а не только лидера.

    Пока называли одного лидера, описания топ-N по Sales и по Profit
    были почти одинаковы ('лидер Laptop Pro'), и главная находка
    датасета — товар с высоким оборотом и низкой маржой — была для
    модели принципиально ненаблюдаема: картинку она не видит.
    """
    session, dataset_id = cleaned_dataset_id
    by_sales = TopNChartSkill(session).run(
        dataset_id, category_col="Product", value_col="Sales", n=10, agg="sum"
    )["description"]
    by_profit = TopNChartSkill(session).run(
        dataset_id, category_col="Product", value_col="Profit", n=10, agg="sum"
    )["description"]

    # В рейтинге должны быть названы все шесть товаров, а не только лидер.
    for product in ("Laptop Pro", "Wireless Mouse", "Desk Lamp"):
        assert product in by_sales
        assert product in by_profit

    # Заложенная в данные находка: Wireless Mouse даёт заметно большую
    # долю оборота, чем прибыли, — по описаниям это должно быть видно.
    sales_rank = by_sales.index("Wireless Mouse")
    profit_rank = by_profit.index("Wireless Mouse")
    assert sales_rank < profit_rank


def test_trend_description_uses_yearly_totals_not_noisy_endpoints(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Регрессия: динамика считается по годам, а не по двум крайним месяцам.

    Сравнение первого месяца с последним давало «+250%» на данных, где
    год к году рост около нуля, — модель добросовестно перенесла бы эту
    цифру в отчёт как факт.
    """
    session, dataset_id = cleaned_dataset_id
    description = TrendChartSkill(session).run(dataset_id, x_col="Date", y_col="Sales")[
        "description"
    ]

    assert "По годам" in description
    assert "2023" in description
    assert "2024" in description


def test_top_n_chart_rejects_invalid_aggregation(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Недопустимый agg возвращает структурированную ошибку с allowed."""
    session, dataset_id = cleaned_dataset_id
    with pytest.raises(SkillError) as exc_info:
        TopNChartSkill(session).run(
            dataset_id, category_col="Product", value_col="Sales", agg="median"
        )

    payload = exc_info.value.payload.to_dict()
    assert payload["error_code"] == "invalid_aggregation"
    assert set(payload["allowed"]) == {"sum", "mean", "count"}
