"""Тесты формата ошибок для LLM."""

from core.errors import unknown_value_error

ALLOWED_COLUMNS = ["Date", "Product", "Region", "Sales", "Quantity", "Profit"]


def test_unknown_column_suggests_closest_match() -> None:
    """Опечатка 'Sale' должна предложить ближайшее допустимое имя 'Sales'."""
    error = unknown_value_error("unknown_column", "Колонка", "Sale", ALLOWED_COLUMNS)
    payload = error.payload.to_dict()

    assert payload["ok"] is False
    assert payload["error_code"] == "unknown_column"
    assert payload["suggestion"] == "Sales"
    assert payload["allowed"] == ALLOWED_COLUMNS


def test_unknown_value_without_close_match_has_no_suggestion() -> None:
    """Если ничего похожего нет, suggestion отсутствует, а не пустая строка."""
    error = unknown_value_error("unknown_column", "Колонка", "zzz", ALLOWED_COLUMNS)
    payload = error.payload.to_dict()

    assert "suggestion" not in payload
