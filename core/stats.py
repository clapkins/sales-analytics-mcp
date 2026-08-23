"""Статистические утилиты, переиспользуемые несколькими скиллами.

IQR-детекция выбросов нужна и ``DataCleaningSkill`` (пометить/удалить
строку-выброс), и ``VisualizationSkill`` (упомянуть число выбросов в
описании гистограммы для модели) — правило определено здесь один раз,
а не продублировано в обоих скиллах.
"""

import pandas as pd

_IQR_MULTIPLIER = 1.5


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    """Строит булеву маску выбросов по правилу 1.5×IQR.

    Args:
        series: Числовая колонка.

    Returns:
        Булева маска той же длины: ``True`` — значение вне диапазона
        ``[Q1 - 1.5×IQR, Q3 + 1.5×IQR]``.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - _IQR_MULTIPLIER * iqr
    upper_bound = q3 + _IQR_MULTIPLIER * iqr
    return (series < lower_bound) | (series > upper_bound)
