"""Скилл загрузки данных из CSV/Excel/JSON.

Владеет всей логикой разбора файла и приведения типов: подбор
кодировки и разделителя для CSV, парсинг колонок с датами из разных
текстовых форматов. Сам датафрейм наружу не возвращается (инвариант
№1, CLAUDE.md) — кладётся в session store, наружу уходит dataset_id и
компактная сводка по структуре данных.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from core.errors import ErrorPayload, SkillError, unknown_value_error
from core.registry import BaseSkill, register_skill
from core.serialization import to_json_safe
from core.session import SessionStore

_ENCODINGS = ("utf-8", "utf-8-sig", "cp1251")
_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y")
_SUPPORTED_SUFFIXES = (".csv", ".xlsx", ".xls", ".json")
_DATE_PARSE_SUCCESS_THRESHOLD = 0.8


def _read_csv_with_encoding(path: Path) -> pd.DataFrame:
    """Читает CSV, перебирая кодировки и автоопределяя разделитель.

    Args:
        path: Путь к CSV-файлу.

    Returns:
        Загруженный датафрейм.

    Raises:
        SkillError: Если файл не удалось прочитать ни в одной из
            поддерживаемых кодировок.
    """
    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue

    raise SkillError(
        ErrorPayload(
            error_code="unsupported_encoding",
            message=(
                f"Не удалось прочитать файл '{path.name}' ни в одной из "
                f"поддерживаемых кодировок: {', '.join(_ENCODINGS)}."
            ),
        )
    ) from last_error


def _load_raw(path: Path) -> pd.DataFrame:
    """Загружает файл в датафрейм в зависимости от расширения.

    Args:
        path: Путь к файлу.

    Returns:
        Загруженный датафрейм без приведения типов.

    Raises:
        SkillError: Если расширение файла не поддерживается.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_with_encoding(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)

    raise unknown_value_error(
        "unsupported_file_format",
        "Формат файла",
        suffix,
        list(_SUPPORTED_SUFFIXES),
    )


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Парсит колонку, где даты вперемешку в нескольких форматах.

    Пробует каждый из известных форматов по очереди, а для строк, не
    подошедших ни под один, — общий разбор через ``pd.to_datetime``.

    Args:
        series: Колонка с датами-строками.

    Returns:
        Колонка типа datetime64; значения, которые не удалось
        распознать, становятся ``NaT``.
    """

    def parse_one(value: object) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT
        text = str(value).strip()
        for fmt in _DATE_FORMATS:
            try:
                return pd.to_datetime(text, format=fmt)
            except ValueError:
                continue
        return pd.to_datetime(text, errors="coerce")

    return series.map(parse_one)


def _convert_date_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Определяет и конвертирует колонки, похожие на даты.

    Колонка считается «похожей на дату», если после разбора долей
    успешно распознанных значений среди непустых — не меньше
    ``_DATE_PARSE_SUCCESS_THRESHOLD``. Так формат определяется по
    содержимому, а не по имени колонки.

    Args:
        df: Датафрейм с сырыми (ещё не типизированными) колонками.

    Returns:
        Датафрейм, где распознанные колонки дат имеют тип datetime64.
    """
    df = df.copy()
    for column in df.select_dtypes(include=["object", "str"]).columns:
        original = df[column]
        non_null = original.notna().sum()
        if non_null == 0:
            continue

        parsed = _parse_mixed_dates(original)
        success_ratio = parsed.notna().sum() / non_null
        if success_ratio >= _DATE_PARSE_SUCCESS_THRESHOLD:
            df[column] = parsed

    return df


@register_skill
class DataLoadingSkill(BaseSkill):
    """Загружает CSV/Excel/JSON и кладёт результат в session store."""

    name = "load_data"
    description = (
        "Загружает табличные данные из файла (CSV, Excel или JSON), "
        "автоматически определяя кодировку, разделитель и формат дат "
        "в колонках. Возвращает dataset_id для дальнейших вызовов "
        "и сводку по структуре данных: размерность, типы колонок, "
        "количество пропусков и несколько строк примера."
    )

    def __init__(self, session: SessionStore) -> None:
        """Инициализирует скилл с хранилищем датасетов.

        Args:
            session: Хранилище, куда будет положен загруженный датафрейм.
        """
        self.session = session

    def run(self, file_path: str) -> dict[str, Any]:
        """Загружает файл и кладёт результат в session store.

        Args:
            file_path: Путь к файлу CSV, Excel (.xlsx/.xls) или JSON.

        Returns:
            Словарь с ``dataset_id``, размерностью, типами колонок,
            количеством пропусков по колонкам и тремя строками примера.

        Raises:
            SkillError: Если файл не найден, формат не поддерживается
                или не удалось подобрать кодировку.
        """
        path = Path(file_path)
        if not path.exists():
            raise SkillError(
                ErrorPayload(
                    error_code="file_not_found",
                    message=f"Файл '{file_path}' не найден.",
                )
            )

        df = _load_raw(path)
        df = _convert_date_like_columns(df)

        dataset_id = self.session.put(df, meta={"source_file": str(path)})

        result = {
            "ok": True,
            "dataset_id": dataset_id,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_by_column": df.isna().sum().to_dict(),
            "sample_rows": df.head(3).to_dict(orient="records"),
        }
        return to_json_safe(result)
