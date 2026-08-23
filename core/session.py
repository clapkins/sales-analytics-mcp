"""Хранилище датасетов в памяти (session store).

Ключевой архитектурный инвариант проекта (CLAUDE.md, п.1): DataFrame
никогда не пересекает границу LLM. Модель работает только с коротким
``dataset_id``, а сам датафрейм живёт здесь, в памяти процесса сервера.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.errors import unknown_value_error


@dataclass
class DatasetEntry:
    """Одна запись хранилища: датафрейм и его происхождение.

    Attributes:
        dataset_id: Идентификатор, под которым датасет виден модели.
        df: Сам датафрейм.
        meta: Метаданные скилла, положившего датасет (форма, источник,
            лог операций и т.п.) — произвольная структура, формируется
            и интерпретируется вызывающим скиллом.
        parent_id: Идентификатор датасета-родителя, если этот датасет
            получен из другого преобразованием (например, очисткой).
            ``None`` для датасетов, загруженных напрямую.
        operation: Имя операции, породившей этот датасет из родителя
            (например, ``"clean"``). ``None`` без родителя.
    """

    dataset_id: str
    df: pd.DataFrame
    meta: dict[str, Any]
    parent_id: str | None = None
    operation: str | None = None


class SessionStore:
    """Хранилище датафреймов в памяти процесса сервера.

    Не является постоянным хранилищем: данные живут, пока жив процесс
    MCP-сервера (персистентность — сознательно не реализуемая часть,
    см. ARCHITECTURE.md, «что доделал бы дальше»).
    """

    def __init__(self) -> None:
        self._entries: dict[str, DatasetEntry] = {}
        self._counter = 0

    def put(
        self,
        df: pd.DataFrame,
        meta: dict[str, Any],
        *,
        parent_id: str | None = None,
        operation: str | None = None,
    ) -> str:
        """Кладёт датафрейм в хранилище и возвращает его dataset_id.

        Идентификатор короткий и читаемый, а не UUID: модель копирует
        его между вызовами инструментов текстом, и длинный случайный
        id — источник опечаток. Датасет без родителя получает id вида
        ``ds_1``, ``ds_2`` по порядку добавления. Датасет, порождённый
        из другого (например, очисткой), получает id вида
        ``{parent_id}_{operation}`` — так ``ds_1`` после очистки
        становится ``ds_1_clean``.

        Args:
            df: Датафрейм для сохранения.
            meta: Метаданные скилла, положившего датасет.
            parent_id: Id датасета-родителя, если этот датасет получен
                из другого преобразованием.
            operation: Имя операции, породившей датасет из родителя.
                Обязателен, если указан ``parent_id``.

        Returns:
            Присвоенный dataset_id.

        Raises:
            ValueError: Если передан ``parent_id`` без ``operation``.
        """
        if parent_id is not None and not operation:
            raise ValueError("operation обязателен, если указан parent_id")

        if parent_id is None:
            self._counter += 1
            dataset_id = f"ds_{self._counter}"
        else:
            base = f"{parent_id}_{operation}"
            dataset_id = base
            suffix = 2
            while dataset_id in self._entries:
                dataset_id = f"{base}_{suffix}"
                suffix += 1

        self._entries[dataset_id] = DatasetEntry(
            dataset_id=dataset_id,
            df=df,
            meta=meta,
            parent_id=parent_id,
            operation=operation,
        )
        return dataset_id

    def get(self, dataset_id: str) -> pd.DataFrame:
        """Возвращает датафрейм по его dataset_id.

        Args:
            dataset_id: Идентификатор датасета.

        Returns:
            Сохранённый датафрейм.

        Raises:
            SkillError: Если dataset_id не найден — с кодом
                ``"unknown_dataset_id"`` и списком реально
                существующих идентификаторов, чтобы модель могла
                исправиться на следующем шаге.
        """
        entry = self._entries.get(dataset_id)
        if entry is None:
            raise unknown_value_error(
                "unknown_dataset_id",
                "Идентификатор датасета",
                dataset_id,
                list(self._entries),
            )
        return entry.df

    def list(self) -> list[dict[str, Any]]:
        """Возвращает сводку по всем датасетам в хранилище.

        Returns:
            Список словарей с полями ``dataset_id``, ``parent_id``,
            ``operation``, размерностью (``rows``, ``columns``) и
            ``meta`` — по одному на каждый сохранённый датасет.
        """
        return [
            {
                "dataset_id": entry.dataset_id,
                "parent_id": entry.parent_id,
                "operation": entry.operation,
                "rows": entry.df.shape[0],
                "columns": entry.df.shape[1],
                "meta": entry.meta,
            }
            for entry in self._entries.values()
        ]
