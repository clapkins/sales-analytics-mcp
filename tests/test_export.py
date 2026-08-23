"""Тесты скилла сохранения отчёта."""

import json

import pandas as pd
import pytest

from core.errors import SkillError
from core.session import SessionStore
from skills.export import ExportReportSkill


@pytest.fixture
def dataset_id() -> tuple[SessionStore, str]:
    """Простой датасет для проверки экспорта."""
    session = SessionStore()
    ds_id = session.put(pd.DataFrame({"a": [1, 2, 3]}), meta={})
    return session, ds_id


def test_markdown_export_writes_text_as_is(dataset_id: tuple[SessionStore, str]) -> None:
    """format='markdown' сохраняет переданный текст без изменений."""
    session, ds_id = dataset_id
    report_text = "# Отчёт\n\nПродажи растут."

    result = ExportReportSkill(session).run(ds_id, report_text=report_text, format="markdown")

    saved_path = result["path"]
    assert saved_path.endswith(".md")
    with open(saved_path, encoding="utf-8") as f:
        assert f.read() == report_text


def test_json_export_wraps_text_with_metadata(dataset_id: tuple[SessionStore, str]) -> None:
    """format='json' сохраняет текст вместе с базовыми метаданными датасета."""
    session, ds_id = dataset_id
    report_text = "Продажи растут."

    result = ExportReportSkill(session).run(ds_id, report_text=report_text, format="json")

    with open(result["path"], encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["report_text"] == report_text
    assert payload["dataset_id"] == ds_id
    assert payload["rows"] == 3


def test_invalid_format_returns_structured_error(dataset_id: tuple[SessionStore, str]) -> None:
    """Недопустимый format — структурированная ошибка с allowed."""
    session, ds_id = dataset_id
    with pytest.raises(SkillError) as exc_info:
        ExportReportSkill(session).run(ds_id, report_text="текст", format="pdf")

    payload = exc_info.value.payload.to_dict()
    assert payload["error_code"] == "invalid_export_format"
    assert set(payload["allowed"]) == {"markdown", "json"}


def test_does_not_write_own_conclusions(dataset_id: tuple[SessionStore, str]) -> None:
    """Скилл не добавляет ничего к тексту модели — сохраняет ровно то, что получил."""
    session, ds_id = dataset_id
    report_text = "ровно этот текст и ничего больше"

    result = ExportReportSkill(session).run(ds_id, report_text=report_text, format="markdown")

    with open(result["path"], encoding="utf-8") as f:
        content = f.read()
    assert content == report_text
