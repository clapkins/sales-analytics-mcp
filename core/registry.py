"""Реестр скиллов: автодискавери и единая точка регистрации.

Список инструментов MCP генерируется из этого реестра (архитектурный
инвариант №3, CLAUDE.md): добавление скилла — это новый файл в
``skills/`` с декоратором ``@register_skill``, без единой правки в
``core/`` или в слое MCP-инструментов.
"""

import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    """Базовый класс скилла.

    Attributes:
        name: Уникальное машиночитаемое имя скилла — используется как
            имя инструмента MCP.
        description: Описание для LLM: что делает скилл и когда его
            стоит вызывать.
    """

    name: str
    description: str

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Выполняет логику скилла.

        Сигнатура намеренно свободная — каждый скилл сам определяет
        свои параметры. Валидация аргументов и сериализация результата
        — обязанность слоя MCP-инструментов (этап 5), не самого скилла.
        """


_REGISTRY: dict[str, type[BaseSkill]] = {}


def register_skill(skill_cls: type[BaseSkill]) -> type[BaseSkill]:
    """Регистрирует класс скилла по декоратору.

    Args:
        skill_cls: Класс-наследник ``BaseSkill``.

    Returns:
        Тот же класс без изменений — декоратор используется ради
        побочного эффекта регистрации.

    Raises:
        ValueError: Если скилл с таким ``name`` уже зарегистрирован.
    """
    name = skill_cls.name
    if name in _REGISTRY:
        raise ValueError(f"Скилл с именем '{name}' уже зарегистрирован")
    _REGISTRY[name] = skill_cls
    return skill_cls


def discover() -> list[type[BaseSkill]]:
    """Импортирует все модули пакета ``skills`` и возвращает реестр.

    Импорт модуля запускает декораторы ``@register_skill`` внутри
    него, поэтому вызывать эту функцию нужно один раз при старте
    сервера, до генерации списка MCP-инструментов.

    Returns:
        Список зарегистрированных классов скиллов.
    """
    import skills

    for module_info in pkgutil.iter_modules(skills.__path__, prefix="skills."):
        importlib.import_module(module_info.name)

    return list(_REGISTRY.values())
