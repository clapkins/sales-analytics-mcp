"""Тесты статистических утилит, общих для очистки и визуализации."""

import pandas as pd

from core.stats import describe_correlation_strength, iqr_outlier_mask


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


def test_weak_correlation_on_real_sample_is_marked_indistinguishable() -> None:
    """Реальный случай из демо-прогона: -0.13 на 175 строках — это шум.

    Именно такой коэффициент попал в отчёт как аргумент («объём не
    конвертируется в деньги»), хотя на этой выборке он неотличим от нуля.
    """
    assert "неотличима от нуля" in describe_correlation_strength(-0.13, 175)


def test_same_coefficient_becomes_meaningful_on_large_sample() -> None:
    """Тот же -0.13 на большой выборке уже перестаёт быть шумом."""
    assert describe_correlation_strength(-0.13, 5000) == "слабая"


def test_strength_labels_match_magnitude() -> None:
    """Сильная и умеренная связь называются своими именами."""
    assert describe_correlation_strength(0.71, 175) == "сильная"
    assert describe_correlation_strength(0.45, 175) == "умеренная"
