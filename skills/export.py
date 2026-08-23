"""Скилл сохранения готового текстового отчёта.

Не пишет ни слова сам — принимает уже готовый текст (модель формирует
его на основе ``prepare_insights_context``) и сохраняет в ``reports/``.
Отход от буквальной сигнатуры ``export_report(dataset_id, format)`` из
PLAN.md: добавлен обязательный ``report_text`` — без него инструменту
было бы физически нечего сохранять, кроме как сочинить текст самому,
а это прямо запрещено архитектурным инвариантом №6 (CLAUDE.md).
"""

import json
from typing import Any

from core.config import REPORTS_DIR
from core.errors import unknown_value_error
from core.registry import BaseSkill, register_skill
from core.serialization import to_json_safe

_EXPORT_FORMATS = ("markdown", "json")


@register_skill
class ExportReportSkill(BaseSkill):
    """Сохраняет текстовый отчёт, написанный моделью, в файл."""

    name = "export_report"
    description = (
        "Сохраняет готовый текстовый отчёт (написанный тобой на основе "
        "результата prepare_insights_context) в файл в папке reports/. "
        "Сам текст выводов не формирует — только сохраняет переданный. "
        "format: 'markdown' — текст как есть с расширением .md; "
        "'json' — текст вместе с базовыми метаданными датасета."
    )

    def run(self, dataset_id: str, report_text: str, format: str = "markdown") -> dict[str, Any]:
        """Сохраняет отчёт в reports/.

        Args:
            dataset_id: Идентификатор датасета, к которому относится
                отчёт — используется в имени файла и (для json) в
                метаданных.
            report_text: Готовый текст отчёта, написанный моделью.
            format: ``"markdown"`` (по умолчанию) или ``"json"``.

        Returns:
            Словарь с путём к сохранённому файлу и форматом.

        Raises:
            SkillError: Если dataset_id неизвестен или format не
                входит в допустимые значения.
        """
        if format not in _EXPORT_FORMATS:
            raise unknown_value_error(
                "invalid_export_format", "Формат экспорта", format, list(_EXPORT_FORMATS)
            )

        df = self.session.get(dataset_id)

        if format == "markdown":
            path = REPORTS_DIR / f"report_{dataset_id}.md"
            path.write_text(report_text, encoding="utf-8")
        else:
            path = REPORTS_DIR / f"report_{dataset_id}.json"
            payload = {
                "dataset_id": dataset_id,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "report_text": report_text,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return to_json_safe({"ok": True, "path": str(path), "format": format})
