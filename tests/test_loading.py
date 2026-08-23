"""Тесты скилла загрузки данных."""

from pathlib import Path

import pandas as pd
import pytest

from core.errors import SkillError
from core.session import SessionStore
from skills.loading import DataLoadingSkill

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


@pytest.fixture
def skill() -> DataLoadingSkill:
    """Скилл загрузки с чистым session store."""
    return DataLoadingSkill(SessionStore())


def test_loads_real_dataset_without_manual_hints(skill: DataLoadingSkill) -> None:
    """data/sales_data.csv загружается без подсказок по кодировке и формату дат."""
    result = skill.run(str(DATA_PATH))

    assert result["ok"] is True
    assert result["rows"] == 180
    assert result["columns"] == 6
    assert "datetime64" in result["column_types"]["Date"]
    assert len(result["sample_rows"]) == 3


def test_missing_counts_match_intentionally_injected_defects(skill: DataLoadingSkill) -> None:
    """Пропуски, заложенные в data/README.md (по 12 в Sales и Profit), видны при загрузке."""
    result = skill.run(str(DATA_PATH))

    assert result["missing_by_column"]["Sales"] == 12
    assert result["missing_by_column"]["Profit"] == 12


def test_unknown_file_raises_structured_error(skill: DataLoadingSkill) -> None:
    """Несуществующий файл — структурированная ошибка, а не голое исключение."""
    with pytest.raises(SkillError) as exc_info:
        skill.run("no_such_file.csv")

    payload = exc_info.value.payload.to_dict()
    assert payload["ok"] is False
    assert payload["error_code"] == "file_not_found"


def test_unsupported_extension_lists_allowed_formats(
    skill: DataLoadingSkill, tmp_path: Path
) -> None:
    """Неподдерживаемое расширение возвращает список допустимых форматов."""
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(SkillError) as exc_info:
        skill.run(str(bad_file))

    payload = exc_info.value.payload.to_dict()
    assert payload["error_code"] == "unsupported_file_format"
    assert ".csv" in payload["allowed"]


def test_falls_back_to_cp1251_when_utf8_fails(skill: DataLoadingSkill, tmp_path: Path) -> None:
    """Файл в cp1251 (нет валидного utf-8) должен читаться через перебор кодировок."""
    cp1251_file = tmp_path / "cyrillic.csv"
    content = "Регион,Продажи\nВосток,100\nЗапад,200\n"
    cp1251_file.write_bytes(content.encode("cp1251"))

    result = skill.run(str(cp1251_file))

    assert result["ok"] is True
    assert result["rows"] == 2
    assert result["sample_rows"][0]["Регион"] == "Восток"


def test_load_puts_dataframe_in_session_store(skill: DataLoadingSkill) -> None:
    """dataset_id из результата действительно указывает на датафрейм в store."""
    result = skill.run(str(DATA_PATH))

    stored = skill.session.get(result["dataset_id"])
    assert isinstance(stored, pd.DataFrame)
    assert stored.shape[0] == 180
