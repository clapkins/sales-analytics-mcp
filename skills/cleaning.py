"""Скилл очистки данных.

Выполняет весь набор операций очистки в одном проходе — нормализацию
текстовых колонок, удаление точных дублей, заполнение пропусков и
детекцию выбросов по IQR — и логирует каждую операцию отдельно, не
сворачивая их в одно число «почищено N строк»: лог операций попадёт в
отчёт, и там важно видеть, сколько именно дублей удалено, а сколько
пропусков заполнено.

Результат — новый dataset_id, исходный датасет в session store не
трогается: без этого модель потеряла бы возможность вернуться к сырым
данным после очистки.
"""

from typing import Any

import pandas as pd

from core.errors import unknown_value_error
from core.registry import BaseSkill, register_skill
from core.serialization import to_json_safe
from core.stats import iqr_outlier_mask

_OUTLIER_ACTIONS = ("mark", "remove")


def _normalize_text_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Убирает пробелы по краям и сводит регистр к наиболее частому варианту.

    Значения текстовой колонки группируются по ключу (strip + lower
    регистр), и все варианты одной группы заменяются на самый частый
    исходный вариант. Это устраняет разнобой вроде ``'NORTH'`` /
    ``'north'`` / ``'  North  '``, не трогая колонки без такого
    разнобоя (например, ``Product``, где регистр уже единообразен).

    Args:
        df: Исходный датафрейм.

    Returns:
        Кортеж: датафрейм с нормализованным текстом и лог — сколько
        значений изменено по каждой затронутой колонке.
    """
    df = df.copy()
    changes: dict[str, int] = {}

    for column in df.select_dtypes(include=["object", "str"]).columns:
        original = df[column]
        mask = original.notna()
        if not mask.any():
            continue

        stripped = original[mask].astype(str).str.strip()
        keys = stripped.str.lower()
        canonical_by_key = stripped.groupby(keys).agg(lambda values: values.value_counts().idxmax())
        normalized = keys.map(canonical_by_key)

        changed = int((normalized.to_numpy() != original[mask].to_numpy()).sum())
        if changed:
            changes[column] = changed
            df.loc[mask, column] = normalized.to_numpy()

    return df, changes


def _fill_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Заполняет пропуски: медианой для чисел, модой для категорий.

    Колонки с датами не трогаются — заполнение пропущенной даты
    медианой или модой не имеет содержательного смысла и не
    предусмотрено планом очистки.

    Args:
        df: Датафрейм после нормализации текста и удаления дублей.

    Returns:
        Кортеж: датафрейм без пропусков в обработанных колонках и лог
        — сколько значений заполнено по каждой колонке.
    """
    df = df.copy()
    filled: dict[str, int] = {}

    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        if missing == 0:
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            fill_value = series.median()
        else:
            mode = series.mode(dropna=True)
            if mode.empty:
                continue
            fill_value = mode.iloc[0]

        df[column] = series.fillna(fill_value)
        filled[column] = missing

    return df, filled


def _handle_outliers(df: pd.DataFrame, action: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Находит выбросы по IQR во всех числовых колонках и помечает/удаляет их.

    IQR считается глобально по всей колонке, без разбивки по товару
    или региону — простое и прозрачное правило, соответствующее
    объёму задания. Из-за этого наряду с намеренно внесёнными
    выбросами возможны находки на границе распределения — это
    ожидаемое поведение простого IQR-правила, а не ошибка.

    Args:
        df: Датафрейм после заполнения пропусков.
        action: ``"mark"`` — добавить колонку ``is_outlier``;
            ``"remove"`` — удалить строки-выбросы из результата.

    Returns:
        Кортеж: итоговый датафрейм и лог — сколько выбросов найдено
        по каждой числовой колонке.
    """
    df = df.copy()
    combined_mask = pd.Series(False, index=df.index)
    by_column: dict[str, int] = {}

    for column in df.select_dtypes(include="number").columns:
        mask = iqr_outlier_mask(df[column])
        count = int(mask.sum())
        if count:
            by_column[column] = count
            combined_mask |= mask

    if action == "mark":
        df["is_outlier"] = combined_mask
    else:
        df = df[~combined_mask].reset_index(drop=True)

    return df, by_column


@register_skill
class DataCleaningSkill(BaseSkill):
    """Очищает датасет: дубли, пропуски, разнобой в тексте, выбросы."""

    name = "clean_data"
    description = (
        "Очищает загруженный датасет: удаляет точные дубли строк, "
        "заполняет пропуски (медианой для чисел, модой для категорий), "
        "нормализует текстовые колонки (пробелы, регистр) и находит "
        "выбросы по правилу IQR — помечает их колонкой is_outlier или "
        "удаляет, в зависимости от outlier_action. Создаёт новый "
        "dataset_id, исходный датасет остаётся доступен для сравнения."
    )

    def run(self, dataset_id: str, outlier_action: str = "mark") -> dict[str, Any]:
        """Выполняет полный цикл очистки датасета.

        Args:
            dataset_id: Id исходного датасета в session store.
            outlier_action: ``"mark"`` (по умолчанию) — пометить
                выбросы колонкой ``is_outlier``; ``"remove"`` — удалить
                строки-выбросы из результата.

        Returns:
            Словарь с новым ``dataset_id``, ссылкой на исходный,
            размерностью до/после очистки и логом операций.

        Raises:
            SkillError: Если ``dataset_id`` неизвестен (поднимается
                session store) или ``outlier_action`` не входит в
                допустимые значения.
        """
        if outlier_action not in _OUTLIER_ACTIONS:
            raise unknown_value_error(
                "invalid_outlier_action",
                "Режим обработки выбросов",
                outlier_action,
                list(_OUTLIER_ACTIONS),
            )

        df = self.session.get(dataset_id)
        rows_before = df.shape[0]

        df, text_normalized = _normalize_text_columns(df)

        duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)

        df, missing_filled = _fill_missing(df)
        df, outliers_by_column = _handle_outliers(df, outlier_action)

        operations_log = {
            "text_normalized": text_normalized,
            "duplicates_removed": duplicates_removed,
            "missing_filled": missing_filled,
            "outliers": {
                "action": outlier_action,
                "by_column": outliers_by_column,
            },
        }

        new_dataset_id = self.session.put(
            df,
            meta={
                "parent_dataset_id": dataset_id,
                "operation": "clean",
                "operations_log": operations_log,
            },
            parent_id=dataset_id,
            operation="clean",
        )

        result = {
            "ok": True,
            "dataset_id": new_dataset_id,
            "parent_dataset_id": dataset_id,
            "rows_before": rows_before,
            "rows_after": df.shape[0],
            "operations_log": operations_log,
        }
        return to_json_safe(result)
