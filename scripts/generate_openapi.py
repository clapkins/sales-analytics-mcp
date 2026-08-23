"""Генерирует openapi.json — спецификацию инструментов в формате OpenAPI.

MCP — это не REST и не описывается нативно через OpenAPI: у него свой
протокол поверх JSON-RPC. Этот файл — не спецификация HTTP-маршрутов
``server_http.py`` (тот транспорт говорит на MCP, а не на обычном
REST), а представление каждого MCP-инструмента в виде одного
POST-эндпоинта с той же JSON-схемой параметров, что видит модель.
Такой формат нужен ТЗ для сценария интеграции через Custom GPT Action
или другой клиент, которому нужен именно OpenAPI, а не MCP-клиент.

Запуск:
    python scripts/generate_openapi.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Скрипт запускается как `python scripts/generate_openapi.py`, поэтому
# без этой строки sys.path[0] — это scripts/, а не корень репозитория,
# и `import core.mcp_app` не находится.
sys.path.insert(0, str(PROJECT_ROOT))

from core.mcp_app import build_mcp_server  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "openapi.json"


async def _build_paths() -> dict[str, Any]:
    """Строит секцию ``paths`` OpenAPI-документа из зарегистрированных инструментов.

    Returns:
        Словарь путей вида ``/tools/{имя_инструмента}`` с методом POST.
    """
    mcp = build_mcp_server()
    tools = await mcp.list_tools()

    paths: dict[str, Any] = {}
    for tool in sorted(tools, key=lambda t: t.name):
        paths[f"/tools/{tool.name}"] = {
            "post": {
                "summary": tool.description or tool.name,
                "operationId": tool.name,
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": tool.parameters}},
                },
                "responses": {
                    "200": {
                        "description": "Результат вызова инструмента.",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        }
    return paths


def _build_spec(paths: dict[str, Any]) -> dict[str, Any]:
    """Собирает полный OpenAPI-документ вокруг готовой секции paths.

    Args:
        paths: Секция ``paths``, построенная из инструментов сервера.

    Returns:
        Словарь — полный OpenAPI 3.1 документ.
    """
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "sales-analytics-mcp",
            "version": "0.1.0",
            "description": (
                "Инструменты MCP-сервера аналитической системы, "
                "представленные в формате OpenAPI для интеграций, "
                "которым нужен REST, а не протокол MCP (например, "
                "Custom GPT Action)."
            ),
        },
        "paths": paths,
    }


def main() -> None:
    """Генерирует openapi.json в корне репозитория."""
    paths = asyncio.run(_build_paths())
    spec = _build_spec(paths)
    OUTPUT_PATH.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
