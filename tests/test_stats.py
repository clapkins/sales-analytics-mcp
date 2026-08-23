"""Тесты общей IQR-утилиты, используемой очисткой и визуализацией."""

import pandas as pd

from core.stats import iqr_outlier_mask


def test_iqr_outlier_mask_flags_only_extreme_value() -> None:
    """Одно резко выделяющееся значение помечается, остальные — нет."""
    series = pd.Series([9, 10, 11, 12, 13, 1000])

    mask = iqr_outlier_mask(series)

    assert mask.tolist() == [False, False, False, False, False, True]


def test_iqr_outlier_mask_empty_for_uniform_data() -> None:
    """На однородных данных выбросов не находится."""
    series = pd.Series([10, 11, 9, 10, 11, 10, 9])

    mask = iqr_outlier_mask(series)

    assert not mask.any()
