"""Тесты реестра скиллов."""

from typing import Any

import pytest

from core import registry
from core.registry import BaseSkill, discover, register_skill


@pytest.fixture(autouse=True)
def _isolated_registry() -> Any:
    """Изолирует каждый тест от глобального состояния реестра.

    Без этого регистрация фиктивного скилла в одном тесте была бы
    видна остальным тестам модуля, включая тест на пустой реестр.
    """
    original = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original)


def test_discover_returns_empty_list_without_skills() -> None:
    """Пакет skills/ пуст на этом этапе — discover() не падает и возвращает []."""
    assert discover() == []


def test_register_skill_is_found_by_discover() -> None:
    """Скилл, зарегистрированный декоратором, попадает в результат discover()."""

    @register_skill
    class FakeSkill(BaseSkill):
        name = "fake_skill"
        description = "Тестовый скилл для проверки реестра."

        def run(self) -> str:
            return "ok"

    discovered_names = [skill.name for skill in discover()]

    assert "fake_skill" in discovered_names


def test_register_skill_rejects_duplicate_name() -> None:
    """Повторная регистрация того же имени — ошибка, а не молчаливая перезапись."""

    @register_skill
    class FirstSkill(BaseSkill):
        name = "duplicate_name"
        description = "Первый скилл."

        def run(self) -> str:
            return "first"

    with pytest.raises(ValueError):

        @register_skill
        class SecondSkill(BaseSkill):
            name = "duplicate_name"
            description = "Второй скилл с тем же именем."

            def run(self) -> str:
                return "second"
