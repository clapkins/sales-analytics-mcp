"""Тесты скилла анализа сезонности."""

from pathlib import Path

import pandas as pd
import pytest

from core.errors import SkillError
from core.session import SessionStore
from skills.cleaning import DataCleaningSkill
from skills.loading import DataLoadingSkill
from skills.seasonality import SeasonalitySkill

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


@pytest.fixture
def cleaned_dataset_id() -> tuple[SessionStore, str]:
    """Реальный очищенный датасет."""
    session = SessionStore()
    raw_id = DataLoadingSkill(session).run(str(DATA_PATH))["dataset_id"]
    clean_id = DataCleaningSkill(session).run(raw_id)["dataset_id"]
    return session, clean_id


def test_finds_seasonality_laid_into_the_dataset(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Генератор данных закладывал пик в конце года и провал летом."""
    session, dataset_id = cleaned_dataset_id
    result = SeasonalitySkill(session).run(dataset_id, date_col="Date", value_col="Sales")

    assert result["ok"] is True
    assert result["years_covered"] == 2
    assert result["peak_month"] in ("ноябрь", "декабрь", "февраль", "март")
    assert result["trough_month"] in ("июнь", "июль", "январь")
    assert result["peak_to_trough_ratio"] > 1


def test_month_and_quarter_shares_sum_to_hundred(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Доли периодов образуют полное распределение, а не произвольные числа."""
    session, dataset_id = cleaned_dataset_id
    result = SeasonalitySkill(session).run(dataset_id, date_col="Date", value_col="Sales")

    assert len(result["by_month"]) == 12
    assert len(result["by_quarter"]) == 4
    assert sum(item["share_pct"] for item in result["by_month"]) == pytest.approx(100, abs=0.5)
    assert sum(item["share_pct"] for item in result["by_quarter"]) == pytest.approx(100, abs=0.5)


def test_months_are_ordered_calendar_not_by_size(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Месяцы идут в календарном порядке — так видно форму сезонной кривой."""
    session, dataset_id = cleaned_dataset_id
    result = SeasonalitySkill(session).run(dataset_id, date_col="Date", value_col="Sales")

    periods = [item["period"] for item in result["by_month"]]
    assert periods[0] == "январь"
    assert periods[-1] == "декабрь"


def test_non_datetime_column_returns_structured_error(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Недатовая колонка — структурированная ошибка со списком датовых."""
    session, dataset_id = cleaned_dataset_id
    with pytest.raises(SkillError) as exc_info:
        SeasonalitySkill(session).run(dataset_id, date_col="Product", value_col="Sales")

    payload = exc_info.value.payload.to_dict()
    assert payload["error_code"] == "column_not_datetime"
    assert payload["allowed"] == ["Date"]


def test_unknown_value_column_suggests_correct_name(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Опечатка в имени колонки значения даёт подсказку."""
    session, dataset_id = cleaned_dataset_id
    with pytest.raises(SkillError) as exc_info:
        SeasonalitySkill(session).run(dataset_id, date_col="Date", value_col="Sale")

    assert exc_info.value.payload.to_dict()["suggestion"] == "Sales"


def test_single_month_dataset_does_not_break() -> None:
    """Вырожденный случай: все продажи в одном месяце — не деление на ноль."""
    session = SessionStore()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-03-01", "2024-03-15", "2024-03-20"]),
            "Sales": [100.0, 200.0, 300.0],
        }
    )
    dataset_id = session.put(df, meta={})

    result = SeasonalitySkill(session).run(dataset_id, date_col="Date", value_col="Sales")

    assert result["peak_month"] == "март"
    assert result["trough_month"] == "март"
    assert len(result["by_month"]) == 1
