"""Скилл списка датасетов, доступных в текущей сессии.

Тонкая обёртка над ``SessionStore.list()`` — вынесена в отдельный
скилл, а не оставлена «внутренним» методом хранилища, потому что
модели нужен способ узнать, какие ``dataset_id`` вообще существуют
(например, после серии load_data/clean_data в диалоге), не запоминая
их все из истории переписки.
"""

from typing import Any

from core.registry import BaseSkill, register_skill
from core.serialization import to_json_safe


@register_skill
class ListDatasetsSkill(BaseSkill):
    """Возвращает список всех датасетов текущей сессии."""

    name = "list_datasets"
    description = (
        "Возвращает список всех датасетов, доступных в этой сессии: "
        "dataset_id, размерность, родительский датасет (если получен "
        "через clean_data) и метаданные. Полезно, если нужно "
        "вспомнить, какие dataset_id уже загружены или очищены."
    )

    def run(self) -> dict[str, Any]:
        """Собирает сводку по всем датасетам в session store.

        Returns:
            Словарь с ``ok`` и списком ``datasets`` — по одному на
            каждый датасет, сохранённый в этой сессии.
        """
        return to_json_safe({"ok": True, "datasets": self.session.list()})
