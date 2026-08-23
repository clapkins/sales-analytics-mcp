"""Тесты скилла подготовки контекста для выводов."""

from pathlib import Path

import pytest

from core.session import SessionStore
from skills.cleaning import DataCleaningSkill
from skills.insights import InsightGenerationSkill
from skills.loading import DataLoadingSkill

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


@pytest.fixture
def loaded_and_cleaned() -> tuple[SessionStore, str, str]:
    """Session store с исходным и очищенным датасетом."""
    session = SessionStore()
    raw_id = DataLoadingSkill(session).run(str(DATA_PATH))["dataset_id"]
    clean_id = DataCleaningSkill(session).run(raw_id)["dataset_id"]
    return session, raw_id, clean_id


def test_context_includes_stats_and_cleaning_log_for_cleaned_dataset(
    loaded_and_cleaned: tuple[SessionStore, str, str],
) -> None:
    """Для очищенного датасета в контексте есть и статистика, и лог очистки."""
    session, _raw_id, clean_id = loaded_and_cleaned

    result = InsightGenerationSkill(session).run(clean_id)

    assert result["ok"] is True
    assert "Sales" in result["stats"]["numeric"]
    assert result["cleaning_log"]["duplicates_removed"] == 5
    assert result["chart_descriptions"] == []


def test_context_has_no_cleaning_log_for_raw_dataset(
    loaded_and_cleaned: tuple[SessionStore, str, str],
) -> None:
    """Для исходного (неочищенного) датасета лога очистки нет — это не ошибка."""
    session, raw_id, _clean_id = loaded_and_cleaned

    result = InsightGenerationSkill(session).run(raw_id)

    assert result["cleaning_log"] is None


def test_chart_descriptions_are_passed_through(
    loaded_and_cleaned: tuple[SessionStore, str, str],
) -> None:
    """Описания графиков, переданные вызывающим, попадают в контекст как есть."""
    session, _raw_id, clean_id = loaded_and_cleaned
    charts = [
        {"chart_type": "trend", "path": "charts/trend_Sales_by_Date.png", "description": "..."},
        {
            "chart_type": "correlation",
            "path": "charts/correlation_matrix.png",
            "description": "...",
        },
    ]

    result = InsightGenerationSkill(session).run(clean_id, chart_descriptions=charts)

    assert result["chart_descriptions"] == charts
