"""Тесты приведения значений pandas/numpy к JSON-совместимым типам."""

import numpy as np
import pandas as pd

from core.serialization import to_json_safe


def test_numpy_scalars_become_native_python_types() -> None:
    """numpy.int64/float64 превращаются в обычные int/float."""
    assert to_json_safe(np.int64(5)) == 5
    assert isinstance(to_json_safe(np.int64(5)), int)
    assert to_json_safe(np.float64(1.5)) == 1.5
    assert isinstance(to_json_safe(np.float64(1.5)), float)


def test_timestamp_becomes_iso_date_string() -> None:
    """pd.Timestamp сериализуется в 'YYYY-MM-DD'."""
    assert to_json_safe(pd.Timestamp("2023-05-14")) == "2023-05-14"


def test_missing_values_become_none() -> None:
    """NaN, NaT и None приводятся к None."""
    assert to_json_safe(float("nan")) is None
    assert to_json_safe(pd.NaT) is None
    assert to_json_safe(None) is None


def test_nested_structures_are_converted_recursively() -> None:
    """Словари и списки обходятся рекурсивно."""
    value = {"a": [np.int64(1), pd.Timestamp("2024-01-01")], "b": np.float64(2.0)}
    assert to_json_safe(value) == {"a": [1, "2024-01-01"], "b": 2.0}
