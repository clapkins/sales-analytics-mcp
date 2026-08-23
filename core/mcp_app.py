"""Сборка MCP-сервера из реестра скиллов.

Единая точка, которую используют оба транспорта (``server_stdio.py`` и
``server_http.py``) и генератор ``scripts/generate_openapi.py``:
создаёт session store, находит все скиллы через ``core.registry.discover()``
и регистрирует каждый как MCP-инструмент. Имя, описание и схема
параметров инструмента берутся из самого скилла — из ``name``,
``description`` и сигнатуры ``run()`` (FastMCP разбирает Google-style
докстринг и достаёт из него даже описания отдельных параметров).
Добавление скилла — это новый файл в ``skills/``, без единой правки
здесь (проверяется этапом 8).
"""

import functools
from pathlib import Path
from typing import Any, Callable

from fastmcp import FastMCP

from core.errors import SkillError
from core.registry import BaseSkill, discover
from core.session import SessionStore

SERVER_NAME = "sales-analytics"
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.md"
SYSTEM_PROMPT_NAME = "sales_analysis_workflow"
SYSTEM_PROMPT_DESCRIPTION = (
    "Последовательность шагов для анализа данных о продажах: "
    "загрузка → структура → статистика → очистка → графики → выводы."
)


def _wrap_skill_errors(skill: BaseSkill) -> Callable[..., Any]:
    """Оборачивает ``run()`` скилла так, чтобы SkillError не был протокольным сбоем.

    Архитектурный инвариант №4 (CLAUDE.md): ошибка должна прийти модели
    как обычный результат вызова (``{"ok": false, ...}``), а не как
    сбой уровня MCP — иначе клиент покажет плашку ошибки, и модель не
    увидит ни текст, ни suggestion, ни allowed.

    ``functools.wraps`` копирует ``__annotations__`` и выставляет
    ``__wrapped__``, поэтому ``inspect.signature``/``get_type_hints``
    внутри FastMCP видят настоящую сигнатуру ``run()`` (без ``self``),
    а не ``(*args, **kwargs)`` — иначе схема инструмента получилась бы
    пустой.

    Args:
        skill: Экземпляр скилла с уже привязанным session store.

    Returns:
        Функция с той же сигнатурой, что и ``skill.run``, готовая к
        регистрации через ``mcp.tool()``.
    """

    @functools.wraps(skill.run)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return skill.run(*args, **kwargs)
        except SkillError as error:
            return error.payload.to_dict()

    return wrapper


def _register_system_prompt(mcp: FastMCP) -> None:
    """Регистрирует содержимое prompts/system_prompt.md как MCP-примитив.

    Текст читается из файла при каждом вызове, а не дублируется здесь
    строкой — единственный источник истины для инструкции остаётся
    один файл, который заодно вставляется в README.

    Args:
        mcp: Сервер, на котором регистрируется примитив ``prompt``.
    """

    @mcp.prompt(name=SYSTEM_PROMPT_NAME, description=SYSTEM_PROMPT_DESCRIPTION)
    def sales_analysis_workflow() -> str:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def build_mcp_server() -> FastMCP:
    """Создаёт MCP-сервер с инструментами, сгенерированными из реестра скиллов.

    Returns:
        Настроенный экземпляр FastMCP, общий для обоих транспортов —
        с одним session store на все инструменты этого процесса.
    """
    mcp = FastMCP(SERVER_NAME)
    session = SessionStore()

    for skill_cls in discover():
        skill = skill_cls(session)
        mcp.tool(_wrap_skill_errors(skill), name=skill.name, description=skill.description)

    _register_system_prompt(mcp)

    return mcp
