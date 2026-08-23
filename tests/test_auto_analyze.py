"""Тесты скилла автоматического выбора типа графика."""

from pathlib import Path

import pytest

from core.errors import SkillError
from core.session import SessionStore
from skills.auto_analyze import AutoAnalyzeSkill
from skills.cleaning import DataCleaningSkill
from skills.loading import DataLoadingSkill

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


@pytest.fixture
def cleaned_dataset_id() -> tuple[SessionStore, str]:
    """Реальный очищенный датасет для проверки автовыбора графика."""
    session = SessionStore()
    raw_id = DataLoadingSkill(session).run(str(DATA_PATH))["dataset_id"]
    clean_id = DataCleaningSkill(session).run(raw_id)["dataset_id"]
    return session, clean_id


def test_datetime_column_triggers_trend_chart(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Колонка с датами → auto_analyze строит тренд, подобрав числовую колонку."""
    session, dataset_id = cleaned_dataset_id
    result = AutoAnalyzeSkill(session).run(dataset_id, column="Date")

    assert result["chart_type"] == "trend"
    assert result["auto_selected"]["chart_type"] == "trend"
    assert Path(result["path"]).exists()


def test_numeric_column_triggers_distribution_chart(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Числовая колонка → auto_analyze строит гистограмму распределения."""
    session, dataset_id = cleaned_dataset_id
    result = AutoAnalyzeSkill(session).run(dataset_id, column="Sales")

    assert result["chart_type"] == "distribution"
    assert result["auto_selected"] == {"chart_type": "distribution"}


def test_categorical_column_triggers_top_n_chart(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Категориальная колонка → auto_analyze строит топ-N, подобрав числовую колонку."""
    session, dataset_id = cleaned_dataset_id
    result = AutoAnalyzeSkill(session).run(dataset_id, column="Region")

    assert result["chart_type"] == "top_n"
    assert result["auto_selected"]["chart_type"] == "top_n"
    assert result["auto_selected"]["paired_column"] in ("Sales", "Quantity", "Profit")


def test_unknown_column_returns_structured_error(
    cleaned_dataset_id: tuple[SessionStore, str],
) -> None:
    """Несуществующая колонка — структурированная ошибка, а не трейсбек."""
    session, dataset_id = cleaned_dataset_id
    with pytest.raises(SkillError) as exc_info:
        AutoAnalyzeSkill(session).run(dataset_id, column="NoSuchColumn")

    assert exc_info.value.payload.error_code == "unknown_column"
