"""Статистические утилиты, переиспользуемые несколькими скиллами.

IQR-детекция выбросов нужна и ``DataCleaningSkill`` (пометить/удалить
строку-выброс), и ``VisualizationSkill`` (упомянуть число выбросов в
описании гистограммы для модели) — правило определено здесь один раз,
а не продублировано в обоих скиллах.
"""

import math

import pandas as pd

_IQR_MULTIPLIER = 1.5
_WEAK_CORRELATION = 0.3
_STRONG_CORRELATION = 0.7


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


def describe_correlation_strength(value: float, sample_size: int) -> str:
    """Характеризует силу связи словами, с поправкой на объём выборки.

    Нужна, потому что голый коэффициент модель склонна трактовать как
    доказательство: на выборке в пару сотен строк ``-0.13`` — это шум,
    но в тексте отчёта он легко превращается в утверждение о связи.

    Порог — эвристика ``2/sqrt(n)``, а не проверка статистической
    гипотезы: полноценный тест значимости потребовал бы scipy, которого
    нет в зависимостях. Поэтому и формулировки осторожные («слабая
    связь на выборке такого размера»), без утверждений вроде
    «статистически незначима», которых этот расчёт не обосновывает.

    Args:
        value: Коэффициент корреляции Пирсона.
        sample_size: Число строк, по которым он посчитан.

    Returns:
        Короткая характеристика силы связи: слишком слабая для выводов
        на такой выборке, либо слабая, умеренная или сильная.
    """
    if sample_size > 1:
        heuristic_threshold = 2 / math.sqrt(sample_size)
        if abs(value) < heuristic_threshold:
            return "слабая связь на выборке такого размера, для выводов не годится"

    if abs(value) < _WEAK_CORRELATION:
        return "слабая"
    if abs(value) < _STRONG_CORRELATION:
        return "умеренная"
    return "сильная"
