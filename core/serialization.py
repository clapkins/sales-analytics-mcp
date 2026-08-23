"""Приведение значений pandas/numpy к JSON-совместимым типам.

Скиллы возвращают результаты в виде словарей, которые в итоге уходят
клиенту через MCP как JSON. numpy-скаляры (``numpy.int64``,
``numpy.float64``), объекты pandas (``Timestamp``, ``NaT``, ``NaN``) и
булевы ``numpy.bool_`` не сериализуются стандартным json-модулем
напрямую — эта функция приводит их к обычным Python-типам рекурсивно,
один раз для всех скиллов, а не в каждом по отдельности.
"""

from typing import Any

import numpy as np
import pandas as pd


def to_json_safe(value: Any) -> Any:
    """Рекурсивно приводит значение к JSON-совместимому виду.

    Args:
        value: Произвольное значение — скаляр, словарь, список,
            numpy-скаляр или объект pandas.

    Returns:
        Эквивалент на встроенных типах Python (``int``, ``float``,
        ``str``, ``bool``, ``None``, ``dict``, ``list``). Пропуски
        (``NaN``, ``NaT``) приводятся к ``None``.
    """
    if isinstance(value, dict):
        return {key: to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if pd.isna(value):
        return None
    return value
