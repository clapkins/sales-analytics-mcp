"""Тесты session store."""

import pandas as pd
import pytest

from core.errors import SkillError
from core.session import SessionStore


def test_put_assigns_sequential_short_ids() -> None:
    """Датасеты без родителя получают короткие последовательные id."""
    store = SessionStore()
    first_id = store.put(pd.DataFrame({"a": [1]}), meta={})
    second_id = store.put(pd.DataFrame({"a": [2]}), meta={})

    assert first_id == "ds_1"
    assert second_id == "ds_2"


def test_put_with_parent_builds_derived_id() -> None:
    """Датасет-потомок получает id вида {parent_id}_{operation}."""
    store = SessionStore()
    parent_id = store.put(pd.DataFrame({"a": [1]}), meta={})

    child_id = store.put(
        pd.DataFrame({"a": [1]}),
        meta={},
        parent_id=parent_id,
        operation="clean",
    )

    assert child_id == "ds_1_clean"


def test_put_with_parent_requires_operation() -> None:
    """parent_id без operation — ошибка вызывающего кода, а не тихая порча id."""
    store = SessionStore()
    parent_id = store.put(pd.DataFrame({"a": [1]}), meta={})

    with pytest.raises(ValueError):
        store.put(pd.DataFrame({"a": [1]}), meta={}, parent_id=parent_id)


def test_get_unknown_dataset_id_lists_existing_ones() -> None:
    """Запрос неизвестного id возвращает подсказку с реально доступными id."""
    store = SessionStore()
    known_id = store.put(pd.DataFrame({"a": [1]}), meta={})

    with pytest.raises(SkillError) as exc_info:
        store.get("ds_999")

    payload = exc_info.value.payload.to_dict()
    assert payload["ok"] is False
    assert payload["error_code"] == "unknown_dataset_id"
    assert payload["allowed"] == [known_id]


def test_list_returns_summary_with_parent_link() -> None:
    """list() отражает связь родитель-потомок для отчёта об очистке."""
    store = SessionStore()
    parent_id = store.put(pd.DataFrame({"a": [1, 2]}), meta={"source": "load"})
    child_id = store.put(
        pd.DataFrame({"a": [1]}),
        meta={"source": "clean"},
        parent_id=parent_id,
        operation="clean",
    )

    summaries = {item["dataset_id"]: item for item in store.list()}

    assert summaries[parent_id]["parent_id"] is None
    assert summaries[parent_id]["rows"] == 2
    assert summaries[child_id]["parent_id"] == parent_id
    assert summaries[child_id]["operation"] == "clean"
    assert summaries[child_id]["rows"] == 1
