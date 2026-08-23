"""Общие фикстуры pytest.

``anyio_backend`` нужен тестам с ``@pytest.mark.anyio`` (проверка
MCP-сервера через настоящего ``fastmcp.Client``, который работает
асинхронно) — плагин ``anyio`` уже приходит транзитивной зависимостью
fastmcp, отдельный ``pytest-asyncio`` не нужен.
"""

import pytest

import skills.export
import skills.visualization


@pytest.fixture
def anyio_backend() -> str:
    """Ограничивает анти-бэкенды тестов только asyncio (без trio)."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _isolate_output_dirs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Перенаправляет графики и отчёты тестов во временную папку.

    Тесты намеренно работают на реальных данных и реальном коде
    (CLAUDE.md прямо требует не тестировать на трёх синтетических
    строках), поэтому они по-настоящему сохраняют PNG и файлы отчётов.
    Без этой фикстуры каждый прогон pytest дописывал бы файлы в
    отслеживаемые charts/ и reports/ рядом с осознанно отобранными для
    репозитория демо-графиками — и требовал бы ручной чистки перед
    каждым коммитом.
    """
    monkeypatch.setattr(skills.visualization, "CHARTS_DIR", tmp_path)
    monkeypatch.setattr(skills.export, "REPORTS_DIR", tmp_path)
