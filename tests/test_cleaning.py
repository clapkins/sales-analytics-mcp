"""Тесты скилла очистки данных.

Ключевые тесты прогоняются на реальном data/sales_data.csv, а не на
трёх синтетических строках: только так можно проверить, что очистка
находит ровно те дефекты, что заложены в data/README.md.
"""

from pathlib import Path

import pandas as pd
import pytest

from core.errors import SkillError
from core.session import SessionStore
from skills.cleaning import DataCleaningSkill
from skills.describe import DescriptiveStatsSkill
from skills.loading import DataLoadingSkill

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


@pytest.fixture
def loaded_dataset_id() -> tuple[SessionStore, str]:
    """Реальный датасет, загруженный в общий session store."""
    session = SessionStore()
    dataset_id = DataLoadingSkill(session).run(str(DATA_PATH))["dataset_id"]
    return session, dataset_id


def test_cleaning_finds_all_injected_duplicates(
    loaded_dataset_id: tuple[SessionStore, str],
) -> None:
    """Из data/README.md: заложено ровно 5 полных дублей строк."""
    session, dataset_id = loaded_dataset_id
    result = DataCleaningSkill(session).run(dataset_id)

    assert result["operations_log"]["duplicates_removed"] == 5


def test_cleaning_finds_all_injected_missing_values(
    loaded_dataset_id: tuple[SessionStore, str],
) -> None:
    """Из data/README.md: по 12 пропусков в Sales и Profit (~7% от 175 строк)."""
    session, dataset_id = loaded_dataset_id
    result = DataCleaningSkill(session).run(dataset_id)

    assert result["operations_log"]["missing_filled"]["Sales"] == 12
    assert result["operations_log"]["missing_filled"]["Profit"] == 12


def test_cleaning_finds_injected_sales_outliers(
    loaded_dataset_id: tuple[SessionStore, str],
) -> None:
    """Из data/README.md: минимум 4 выброса по Sales — детектор не должен их упустить."""
    session, dataset_id = loaded_dataset_id
    result = DataCleaningSkill(session).run(dataset_id)

    assert result["operations_log"]["outliers"]["by_column"]["Sales"] >= 4


def test_cleaning_normalizes_region_to_four_canonical_values(
    loaded_dataset_id: tuple[SessionStore, str],
) -> None:
    """После очистки Region содержит ровно 4 канонических значения, а не разнобой регистра."""
    session, dataset_id = loaded_dataset_id
    clean_result = DataCleaningSkill(session).run(dataset_id)

    described = DescriptiveStatsSkill(session).run(clean_result["dataset_id"])
    assert described["categorical"]["Region"]["unique_count"] == 4


def test_cleaning_keeps_original_dataset_accessible(
    loaded_dataset_id: tuple[SessionStore, str],
) -> None:
    """Очистка не должна лишать модель доступа к исходному датасету."""
    session, dataset_id = loaded_dataset_id
    original_rows = session.get(dataset_id).shape[0]

    DataCleaningSkill(session).run(dataset_id)

    assert session.get(dataset_id).shape[0] == original_rows


def test_cleaning_creates_derived_dataset_id(loaded_dataset_id: tuple[SessionStore, str]) -> None:
    """Новый dataset_id строится из id родителя, а не берётся с потолка."""
    session, dataset_id = loaded_dataset_id
    result = DataCleaningSkill(session).run(dataset_id)

    assert result["dataset_id"] == f"{dataset_id}_clean"
    assert result["parent_dataset_id"] == dataset_id


def test_mark_action_keeps_row_count_remove_action_drops_outliers() -> None:
    """outlier_action='mark' сохраняет строки, 'remove' убирает строки-выбросы."""
    session = SessionStore()
    # Значения намеренно все разные — иначе одинаковые строки задел бы
    # шаг удаления дублей раньше, чем дело дойдёт до детекции выбросов.
    df = pd.DataFrame({"value": [9, 10, 11, 12, 13, 1000]})
    dataset_id = session.put(df, meta={})

    marked = DataCleaningSkill(session).run(dataset_id, outlier_action="mark")
    assert marked["rows_after"] == 6
    assert session.get(marked["dataset_id"])["is_outlier"].sum() == 1

    dataset_id_2 = session.put(df, meta={})
    removed = DataCleaningSkill(session).run(dataset_id_2, outlier_action="remove")
    assert removed["rows_after"] == 5


def test_invalid_outlier_action_returns_structured_error() -> None:
    """Недопустимое значение outlier_action — структурированная ошибка с allowed."""
    session = SessionStore()
    dataset_id = session.put(pd.DataFrame({"value": [1, 2, 3]}), meta={})

    with pytest.raises(SkillError) as exc_info:
        DataCleaningSkill(session).run(dataset_id, outlier_action="delete_please")

    payload = exc_info.value.payload.to_dict()
    assert payload["error_code"] == "invalid_outlier_action"
    assert set(payload["allowed"]) == {"mark", "remove"}


def test_text_normalization_collapses_case_and_whitespace_variants() -> None:
    """Юнит-проверка нормализации текста на маленьком примере, без файла."""
    session = SessionStore()
    df = pd.DataFrame(
        {
            "Region": ["North", "  north ", "NORTH", "South"],
            "Sales": [100, 200, 150, 300],
        }
    )
    dataset_id = session.put(df, meta={})

    result = DataCleaningSkill(session).run(dataset_id)

    cleaned = session.get(result["dataset_id"])
    assert set(cleaned["Region"]) == {"North", "South"}
    assert result["operations_log"]["text_normalized"]["Region"] == 2
