"""Тесты скилла списка датасетов."""

import pandas as pd

from core.session import SessionStore
from skills.dataset_registry import ListDatasetsSkill


def test_lists_all_datasets_with_lineage() -> None:
    """list_datasets отражает и исходный, и производный датасет."""
    session = SessionStore()
    parent_id = session.put(pd.DataFrame({"a": [1, 2]}), meta={})
    child_id = session.put(
        pd.DataFrame({"a": [1]}), meta={}, parent_id=parent_id, operation="clean"
    )

    result = ListDatasetsSkill(session).run()

    ids = {item["dataset_id"] for item in result["datasets"]}
    assert ids == {parent_id, child_id}


def test_empty_session_returns_empty_list() -> None:
    """На пустой сессии — пустой список, а не ошибка."""
    session = SessionStore()

    result = ListDatasetsSkill(session).run()

    assert result == {"ok": True, "datasets": []}
