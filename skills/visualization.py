"""Скилл визуализации: тренд, распределение, корреляция, топ-N.

Требования к графикам — прямой критерий оценки (ТЗ, CLAUDE.md): единая
цветовая палитра на весь проект, dpi=150, подписи осей с единицами
измерения, сетка, никаких дефолтных серых matplotlib-графиков. Модель
картинку не видит (архитектурный инвариант №5), поэтому каждый график
возвращается с текстовым описанием, посчитанным из реальных данных —
диапазон, тренд, выбросы, а не шаблонной фразой.

По плану инструменты — четыре отдельных тонких скилла (по одному на
тип графика, имена совпадают с ТЗ: ``plot_trend``, ``plot_distribution``,
``correlation_analysis``, плюс дополнительный ``plot_top_n``), но вся
отрисовка вынесена в общие приватные функции этого модуля, чтобы не
дублировать логику между ними.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.config import CHARTS_DIR
from core.errors import ErrorPayload, SkillError, unknown_value_error
from core.registry import BaseSkill, register_skill
from core.serialization import to_json_safe
from core.session import SessionStore
from core.stats import iqr_outlier_mask

sns.set_theme(style="whitegrid")

ACCENT_COLOR = "#3B6FA0"
HEATMAP_CMAP = "coolwarm"
CHART_DPI = 150
_UNIT_HINTS = {"Sales": "$", "Profit": "$", "Quantity": "шт."}
_TOP_N_AGGREGATIONS = ("sum", "mean", "count")


def _axis_label(column: str) -> str:
    """Добавляет единицу измерения к имени колонки, если она известна.

    Args:
        column: Имя колонки.

    Returns:
        ``"Sales ($)"`` для известных колонок, иначе просто ``column``.
    """
    unit = _UNIT_HINTS.get(column)
    return f"{column} ({unit})" if unit else column


def _validate_column(df: pd.DataFrame, column: str) -> None:
    """Проверяет, что колонка существует в датасете.

    Args:
        df: Датафрейм.
        column: Имя колонки, переданное вызывающим.

    Raises:
        SkillError: С кодом ``"unknown_column"``, списком реальных
            колонок и, если удалось подобрать, ближайшим совпадением.
    """
    if column not in df.columns:
        raise unknown_value_error("unknown_column", "Колонка", column, list(df.columns))


def _require_datetime_column(df: pd.DataFrame, column: str) -> None:
    """Проверяет, что колонка существует и имеет тип datetime64.

    Args:
        df: Датафрейм.
        column: Имя колонки, ожидаемой как ось времени.

    Raises:
        SkillError: Если колонки нет или её тип — не дата; в списке
            допустимых значений — только колонки с датами.
    """
    _validate_column(df, column)
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        datetime_columns = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        raise SkillError(
            ErrorPayload(
                error_code="column_not_datetime",
                message=(
                    f"Колонка '{column}' не является датой, а plot_trend строит "
                    "динамику по времени."
                ),
                allowed=datetime_columns,
            )
        )


def _require_numeric_column(df: pd.DataFrame, column: str) -> None:
    """Проверяет, что колонка существует, числовая и не булева.

    Булевы колонки (например, ``is_outlier`` после очистки) технически
    числовые для pandas, но не подходят для графика распределения или
    тренда — их пришлось бы исключить отдельно на графике.

    Args:
        df: Датафрейм.
        column: Имя колонки, ожидаемой как числовая величина.

    Raises:
        SkillError: Если колонки нет или её тип не подходит; в списке
            допустимых значений — только настоящие числовые колонки.
    """
    _validate_column(df, column)
    series = df[column]
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        numeric_columns = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        ]
        raise SkillError(
            ErrorPayload(
                error_code="column_not_numeric",
                message=f"Колонка '{column}' не числовая.",
                allowed=numeric_columns,
            )
        )


def _save_chart(fig: plt.Figure, filename: str) -> Path:
    """Сохраняет фигуру в charts/ с dpi=150 и без обрезки подписей.

    Args:
        fig: Готовая фигура matplotlib.
        filename: Имя файла без расширения.

    Returns:
        Путь к сохранённому PNG.
    """
    path = CHARTS_DIR / f"{filename}.png"
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_trend_chart(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[plt.Figure, str]:
    """Строит линейный график помесячной суммы y_col по датам x_col.

    Args:
        df: Датафрейм с проверенными колонками (дата + число).
        x_col: Колонка с датами.
        y_col: Числовая колонка.

    Returns:
        Кортеж: готовая фигура и текстовое описание для модели.
    """
    monthly = df.set_index(x_col)[y_col].resample("ME").sum().dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.to_numpy(), marker="o", color=ACCENT_COLOR, linewidth=2)
    ax.set_title(f"Динамика «{y_col}» по месяцам")
    ax.set_xlabel("Месяц")
    ax.set_ylabel(_axis_label(y_col))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(monthly.index.min(), monthly.index.max())
    fig.autofmt_xdate(rotation=45)

    peak_month = monthly.idxmax()
    trough_month = monthly.idxmin()
    first_value, last_value = monthly.iloc[0], monthly.iloc[-1]
    change_pct = ((last_value - first_value) / first_value * 100) if first_value else 0.0

    description = (
        f"Линейный график помесячной суммы '{y_col}' с {monthly.index.min():%Y-%m} "
        f"по {monthly.index.max():%Y-%m}. Пик — {peak_month:%Y-%m} "
        f"({monthly.loc[peak_month]:.0f}), минимум — {trough_month:%Y-%m} "
        f"({monthly.loc[trough_month]:.0f}). Изменение от первого к последнему "
        f"месяцу: {change_pct:+.1f}%."
    )
    return fig, description


def _build_distribution_chart(df: pd.DataFrame, column: str) -> tuple[plt.Figure, str]:
    """Строит гистограмму распределения числовой колонки.

    Args:
        df: Датафрейм с проверенной числовой колонкой.
        column: Числовая колонка.

    Returns:
        Кортеж: готовая фигура и текстовое описание для модели.
    """
    series = df[column].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(series, bins=20, kde=True, color=ACCENT_COLOR, ax=ax)
    ax.set_title(f"Распределение «{column}»")
    ax.set_xlabel(_axis_label(column))
    ax.set_ylabel("Количество записей")

    outlier_count = int(iqr_outlier_mask(series).sum())
    outlier_phrase = (
        f"{outlier_count} значений выходят за пределы 1.5×IQR от основной массы."
        if outlier_count
        else "Явных выбросов за пределами 1.5×IQR не обнаружено."
    )

    description = (
        f"Гистограмма распределения '{column}': {len(series)} значений в "
        f"диапазоне от {series.min():.0f} до {series.max():.0f}, среднее "
        f"{series.mean():.0f}, медиана {series.median():.0f}. {outlier_phrase}"
    )
    return fig, description


def _build_correlation_chart(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Строит тепловую карту корреляций числовых колонок.

    Args:
        df: Датафрейм.

    Returns:
        Кортеж: готовая фигура и текстовое описание для модели.

    Raises:
        SkillError: Если числовых колонок меньше двух.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise SkillError(
            ErrorPayload(
                error_code="not_enough_numeric_columns",
                message="Для корреляционной матрицы нужно минимум две числовые колонки.",
                allowed=list(numeric_df.columns),
            )
        )

    corr = numeric_df.corr()
    size = max(4.0, 1.1 * len(corr.columns))

    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=HEATMAP_CMAP,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Коэффициент корреляции"},
        ax=ax,
    )
    ax.set_title("Корреляционная матрица числовых показателей")
    fig.autofmt_xdate(rotation=45)

    upper_triangle_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(upper_triangle_mask).stack()
    strongest_pair = pairs.abs().idxmax()
    strongest_value = pairs.loc[strongest_pair]

    description = (
        f"Тепловая карта корреляций между колонками: {', '.join(corr.columns)}. "
        f"Сильнее всего связаны '{strongest_pair[0]}' и '{strongest_pair[1]}' "
        f"(коэффициент {strongest_value:+.2f})."
    )
    return fig, description


def _build_top_n_chart(
    df: pd.DataFrame, category_col: str, value_col: str, n: int, agg: str
) -> tuple[plt.Figure, str]:
    """Строит горизонтальную столбчатую диаграмму топ-N категорий.

    Горизонтальная ориентация выбрана намеренно: при длинных названиях
    категорий и/или их большом числе вертикальные подписи оси X
    превращаются в нечитаемый частокол (чек-лист CLAUDE.md).

    Args:
        df: Датафрейм с проверенными колонками.
        category_col: Категориальная колонка для группировки.
        value_col: Числовая колонка для агрегации.
        n: Сколько топ-категорий показать.
        agg: Способ агрегации — ``"sum"``, ``"mean"`` или ``"count"``.

    Returns:
        Кортеж: готовая фигура и текстовое описание для модели.
    """
    grouped = df.groupby(category_col)[value_col].agg(agg).sort_values(ascending=False)
    top = grouped.head(n)

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.5 * len(top) + 1)))
    ax.barh(top.index[::-1].astype(str), top.to_numpy()[::-1], color=ACCENT_COLOR)
    ax.set_title(f"Топ-{len(top)} по «{category_col}» ({agg} от «{value_col}»)")
    ax.set_xlabel(_axis_label(value_col))
    ax.set_ylabel(category_col)

    leader, leader_value = top.index[0], top.iloc[0]
    total = grouped.sum()
    share_pct = (leader_value / total * 100) if total else 0.0

    description = (
        f"Горизонтальная столбчатая диаграмма: топ-{len(top)} значений "
        f"'{category_col}' по {agg} от '{value_col}'. Лидер — '{leader}' "
        f"({leader_value:.0f}, {share_pct:.0f}% от суммы по всем категориям)."
    )
    return fig, description


@register_skill
class TrendChartSkill(BaseSkill):
    """Строит линейный график помесячной динамики числовой колонки."""

    name = "plot_trend"
    description = (
        "Строит линейный график динамики числовой колонки по месяцам. "
        "x_col должен быть колонкой с датами (например, Date), y_col — "
        "числовой колонкой (например, Sales). Возвращает путь к PNG и "
        "текстовое описание графика — модель картинку не видит."
    )

    def __init__(self, session: SessionStore) -> None:
        """Инициализирует скилл с хранилищем датасетов.

        Args:
            session: Хранилище, откуда берётся датасет.
        """
        self.session = session

    def run(self, dataset_id: str, x_col: str, y_col: str) -> dict[str, Any]:
        """Строит и сохраняет график тренда.

        Args:
            dataset_id: Идентификатор датасета в session store.
            x_col: Колонка с датами.
            y_col: Числовая колонка.

        Returns:
            Словарь с ``chart_type``, ``path`` и ``description``.

        Raises:
            SkillError: Если dataset_id неизвестен или колонки не
                подходят по типу.
        """
        df = self.session.get(dataset_id)
        _require_datetime_column(df, x_col)
        _require_numeric_column(df, y_col)

        fig, description = _build_trend_chart(df, x_col, y_col)
        path = _save_chart(fig, f"trend_{y_col}_by_{x_col}")

        return to_json_safe(
            {
                "ok": True,
                "chart_type": "trend",
                "path": str(path),
                "description": description,
            }
        )


@register_skill
class DistributionChartSkill(BaseSkill):
    """Строит гистограмму распределения числовой колонки."""

    name = "plot_distribution"
    description = (
        "Строит гистограмму распределения числовой колонки с кривой "
        "плотности. Подходит для ответа на вопросы вида «как "
        "распределены продажи». Возвращает путь к PNG и текстовое "
        "описание — диапазон, среднее, медиану, наличие выбросов."
    )

    def __init__(self, session: SessionStore) -> None:
        """Инициализирует скилл с хранилищем датасетов.

        Args:
            session: Хранилище, откуда берётся датасет.
        """
        self.session = session

    def run(self, dataset_id: str, column: str) -> dict[str, Any]:
        """Строит и сохраняет гистограмму распределения.

        Args:
            dataset_id: Идентификатор датасета в session store.
            column: Числовая колонка.

        Returns:
            Словарь с ``chart_type``, ``path`` и ``description``.

        Raises:
            SkillError: Если dataset_id неизвестен или колонка не
                числовая.
        """
        df = self.session.get(dataset_id)
        _require_numeric_column(df, column)

        fig, description = _build_distribution_chart(df, column)
        path = _save_chart(fig, f"distribution_{column}")

        return to_json_safe(
            {
                "ok": True,
                "chart_type": "distribution",
                "path": str(path),
                "description": description,
            }
        )


@register_skill
class CorrelationChartSkill(BaseSkill):
    """Строит тепловую карту корреляций числовых колонок датасета."""

    name = "correlation_analysis"
    description = (
        "Строит тепловую карту корреляций между всеми числовыми "
        "колонками датасета. Не требует указания колонок — использует "
        "их все. Возвращает путь к PNG и текстовое описание — самую "
        "сильную по модулю пару связанных показателей."
    )

    def __init__(self, session: SessionStore) -> None:
        """Инициализирует скилл с хранилищем датасетов.

        Args:
            session: Хранилище, откуда берётся датасет.
        """
        self.session = session

    def run(self, dataset_id: str) -> dict[str, Any]:
        """Строит и сохраняет корреляционную тепловую карту.

        Args:
            dataset_id: Идентификатор датасета в session store.

        Returns:
            Словарь с ``chart_type``, ``path`` и ``description``.

        Raises:
            SkillError: Если dataset_id неизвестен или числовых
                колонок меньше двух.
        """
        df = self.session.get(dataset_id)
        fig, description = _build_correlation_chart(df)
        path = _save_chart(fig, "correlation_matrix")

        return to_json_safe(
            {
                "ok": True,
                "chart_type": "correlation",
                "path": str(path),
                "description": description,
            }
        )


@register_skill
class TopNChartSkill(BaseSkill):
    """Строит горизонтальную столбчатую диаграмму топ-N категорий."""

    name = "plot_top_n"
    description = (
        "Строит горизонтальную столбчатую диаграмму топ-N категорий "
        "(например, топ продуктов по сумме продаж или топ регионов по "
        "средней прибыли). category_col — категориальная колонка, "
        "value_col — числовая, agg — 'sum', 'mean' или 'count'."
    )

    def __init__(self, session: SessionStore) -> None:
        """Инициализирует скилл с хранилищем датасетов.

        Args:
            session: Хранилище, откуда берётся датасет.
        """
        self.session = session

    def run(
        self,
        dataset_id: str,
        category_col: str,
        value_col: str,
        n: int = 10,
        agg: str = "sum",
    ) -> dict[str, Any]:
        """Строит и сохраняет диаграмму топ-N категорий.

        Args:
            dataset_id: Идентификатор датасета в session store.
            category_col: Категориальная колонка для группировки.
            value_col: Числовая колонка для агрегации.
            n: Сколько топ-категорий показать (по умолчанию 10).
            agg: Способ агрегации — ``"sum"``, ``"mean"`` или ``"count"``.

        Returns:
            Словарь с ``chart_type``, ``path`` и ``description``.

        Raises:
            SkillError: Если dataset_id неизвестен, колонки не
                подходят по типу или ``agg`` не входит в допустимые
                значения.
        """
        df = self.session.get(dataset_id)
        _validate_column(df, category_col)
        _require_numeric_column(df, value_col)

        if agg not in _TOP_N_AGGREGATIONS:
            raise unknown_value_error(
                "invalid_aggregation",
                "Способ агрегации",
                agg,
                list(_TOP_N_AGGREGATIONS),
            )

        fig, description = _build_top_n_chart(df, category_col, value_col, n, agg)
        path = _save_chart(fig, f"top_n_{value_col}_by_{category_col}")

        return to_json_safe(
            {
                "ok": True,
                "chart_type": "top_n",
                "path": str(path),
                "description": description,
            }
        )
