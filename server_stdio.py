"""Точка входа MCP-сервера через транспорт stdio.

Используется для интеграции с локальными MCP-клиентами (Claude
Desktop, Cherry Studio) — клиент сам запускает этот файл как дочерний
процесс и общается с ним через стандартные потоки ввода-вывода.
Инструменты не перечисляются здесь вручную — весь список собирается
в ``core.mcp_app.build_mcp_server()`` из реестра скиллов.
"""

from core.mcp_app import build_mcp_server

mcp = build_mcp_server()

if __name__ == "__main__":
    mcp.run()
