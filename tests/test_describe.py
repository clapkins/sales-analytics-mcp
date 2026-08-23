"""Тесты скилла описательной статистики."""

from pathlib import Path

import pytest

from core.session import SessionStore
from skills.describe import DescriptiveStatsSkill
from skills.loading import DataLoadingSkill

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


@pytest.fixture
def loaded_dataset_id() -> tuple[SessionStore, str]:
    """Реальный датасет, загруженный в общий session store."""
    session = SessionStore()
    dataset_id = DataLoadingSkill(session).run(str(DATA_PATH))["dataset_id"]
    return session, dataset_id


def test_numeric_columns_have_full_stat_set(loaded_dataset_id: tuple[SessionStore, str]) -> None:
    """Числовые колонки содержат весь требуемый набор статистик."""
    session, dataset_id = loaded_dataset_id
    result = DescriptiveStatsSkill(session).run(dataset_id)

    sales_stats = result["numeric"]["Sales"]
    for key in ("mean", "median", "std", "min", "q25", "q50", "q75", "max"):
        assert key in sales_stats
    assert sales_stats["min"] <= sales_stats["q25"] <= sales_stats["q50"] <= sales_stats["q75"]


def test_categorical_top_values_are_limited_to_five(
    loaded_dataset_id: tuple[SessionStore, str],
) -> None:
    """Категориальная статистика отдаёт не больше топ-5 значений."""
    session, dataset_id = loaded_dataset_id
    result = DescriptiveStatsSkill(session).run(dataset_id)

    product_stats = result["categorical"]["Product"]
    assert product_stats["unique_count"] == 6
    assert len(product_stats["top_values"]) <= 5


def test_dirty_region_has_more_raw_variants_than_canonical_regions(
    loaded_dataset_id: tuple[SessionStore, str],
) -> None:
    """До очистки Region показывает больше 4 значений — разнобой регистра/пробелов."""
    session, dataset_id = loaded_dataset_id
    result = DescriptiveStatsSkill(session).run(dataset_id)

    assert result["categorical"]["Region"]["unique_count"] > 4


def test_datetime_column_has_period_range(loaded_dataset_id: tuple[SessionStore, str]) -> None:
    """Колонка Date даёт диапазон периода в формате ISO-даты."""
    session, dataset_id = loaded_dataset_id
    result = DescriptiveStatsSkill(session).run(dataset_id)

    date_stats = result["datetime"]["Date"]
    assert date_stats["min_date"] < date_stats["max_date"]
