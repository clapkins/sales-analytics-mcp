"""Скилл подготовки контекста для генерации выводов.

Не пишет текст отчёта сам — собирает и структурирует то, что уже
известно из предыдущих вызовов (статистика, лог очистки, описания
графиков) в один пакет фактов. Формулировки выводов и рекомендаций —
задача самой LLM (архитектурный инвариант №6, CLAUDE.md): здесь только
факты, ни одной готовой фразы вида «продажи выросли на X%».

Статистика и лог очистки восстанавливаются по dataset_id напрямую из
session store (переиспользуя ``DescriptiveStatsSkill`` и метаданные,
которые ``DataCleaningSkill`` кладёт при создании датасета) — модели
не нужно копировать их обратно текстом. Описания графиков — исключение:
это выбор модели, какие графики построить в этом диалоге, и
восстановить его по одному dataset_id невозможно, поэтому они
передаются явным аргументом.
"""

from typing import Any

from core.registry import BaseSkill, register_skill
from core.serialization import to_json_safe
from skills.describe import DescriptiveStatsSkill


@register_skill
class InsightGenerationSkill(BaseSkill):
    """Собирает статистику, лог очистки и описания графиков для отчёта."""

    name = "prepare_insights_context"
    description = (
        "Собирает статистику по датасету, лог операций очистки (если "
        "датасет получен через clean_data) и переданные описания "
        "графиков в единый структурированный пакет фактов. Не пишет "
        "текст выводов сам — используй результат этого вызова как "
        "материал для собственного текстового отчёта с выводами и "
        "рекомендациями."
    )

    def run(
        self, dataset_id: str, chart_descriptions: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Собирает факты по датасету в единый контекст для отчёта.

        Args:
            dataset_id: Датасет, для которого готовится отчёт — обычно
                результат ``clean_data``, чтобы в контекст попал лог
                очистки.
            chart_descriptions: Описания графиков, построенных ранее в
                этом же диалоге (результаты ``plot_trend``,
                ``plot_distribution``, ``correlation_analysis``,
                ``plot_top_n``) — эти данные существуют только в
                истории диалога, скилл не может восстановить их
                самостоятельно по одному dataset_id.

        Returns:
            Словарь с ``stats`` (результат ``describe_data``),
            ``cleaning_log`` (``None``, если датасет не был очищен) и
            переданными ``chart_descriptions``.

        Raises:
            SkillError: Если dataset_id неизвестен (поднимается
                session store).
        """
        stats = DescriptiveStatsSkill(self.session).run(dataset_id)
        meta = self.session.get_meta(dataset_id)

        result = {
            "ok": True,
            "dataset_id": dataset_id,
            "stats": stats,
            "cleaning_log": meta.get("operations_log"),
            "chart_descriptions": chart_descriptions or [],
        }
        return to_json_safe(result)
