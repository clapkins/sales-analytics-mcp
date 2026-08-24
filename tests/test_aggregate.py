"""Тесты скилла сводных таблиц."""

from pathlib import Path

import pandas as pd
import pytest

from core.errors import SkillError
from core.session import SessionStore
from skills.aggregate import AggregateSkill
from skills.cleaning import DataCleaningSkill
from skills.loading import DataLoadingSkill

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


@pytest.fixture
def cleaned_dataset_id() -> tuple[SessionStore, str]:
    """Реальный очищенный датасет."""
    session = SessionStore()
    raw_id = DataLoadingSkill(session).run(str(DATA_PATH))["dataset_id"]
    clean_id = DataCleaningSkill(session).run(raw_id)["dataset_id"]
    return session, clean_id


def test_margin_case_is_answered_by_a_single_call(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Кейс, ради которого скилл появился: маржа по товарам одним вызовом.

    В демо-прогоне модели пришлось звать plot_top_n дважды и делить
    числа вручную, породив два PNG ради арифметики.
    """
    session, dataset_id = cleaned_dataset_id
    result = AggregateSkill(session).run(
        dataset_id, group_by="Product", value_cols=["Sales", "Profit"], agg="sum"
    )

    by_product = {row["Product"]: row for row in result["rows"]}
    assert set(by_product) == {
        "Laptop Pro",
        "Monitor 27",
        "Wireless Mouse",
        "Office Chair",
        "Webcam HD",
        "Desk Lamp",
    }

    # Заложенная в данные находка должна считаться прямо из ответа.
    mouse = by_product["Wireless Mouse"]
    lamp = by_product["Desk Lamp"]
    assert mouse["Profit"] / mouse["Sales"] < 0.10
    assert lamp["Profit"] / lamp["Sales"] > 0.20


def test_rows_sorted_by_first_value_column(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Строки отсортированы по убыванию первой из value_cols."""
    session, dataset_id = cleaned_dataset_id
    result = AggregateSkill(session).run(
        dataset_id, group_by="Product", value_cols=["Sales"], agg="sum"
    )

    sales = [row["Sales"] for row in result["rows"]]
    assert sales == sorted(sales, reverse=True)


def test_limit_truncates_and_reports_it() -> None:
    """При обрезке списка это явно помечается флагом truncated."""
    session = SessionStore()
    df = pd.DataFrame({"cat": [f"c{i}" for i in range(10)], "value": range(10)})
    dataset_id = session.put(df, meta={})

    result = AggregateSkill(session).run(dataset_id, group_by="cat", value_cols=["value"], limit=3)

    assert len(result["rows"]) == 3
    assert result["groups_total"] == 10
    assert result["truncated"] is True


def test_unknown_column_suggests_correct_name(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Опечатка в имени колонки даёт подсказку, а не трейсбек."""
    session, dataset_id = cleaned_dataset_id
    with pytest.raises(SkillError) as exc_info:
        AggregateSkill(session).run(dataset_id, group_by="Product", value_cols=["Sale"])

    payload = exc_info.value.payload.to_dict()
    assert payload["error_code"] == "unknown_column"
    assert payload["suggestion"] == "Sales"


def test_invalid_aggregation_lists_allowed(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Недопустимый agg возвращает список допустимых значений."""
    session, dataset_id = cleaned_dataset_id
    with pytest.raises(SkillError) as exc_info:
        AggregateSkill(session).run(
            dataset_id, group_by="Product", value_cols=["Sales"], agg="stddev"
        )

    payload = exc_info.value.payload.to_dict()
    assert payload["error_code"] == "invalid_aggregation"
    assert "sum" in payload["allowed"]


def test_empty_value_cols_is_rejected_with_hint(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Пустой список колонок — понятная ошибка, а не пустая таблица."""
    session, dataset_id = cleaned_dataset_id
    with pytest.raises(SkillError) as exc_info:
        AggregateSkill(session).run(dataset_id, group_by="Product", value_cols=[])

    assert exc_info.value.payload.error_code == "empty_value_cols"
