"""Точка входа MCP-сервера через транспорт streamable-http.

Второй транспорт для того же набора инструментов — для клиентов,
которые говорят с MCP-сервером по HTTP, а не запускают его как
локальный дочерний процесс. Набор инструментов идентичен
``server_stdio.py``: оба используют одну и ту же сборку
``core.mcp_app.build_mcp_server()``, различается только транспорт.
"""

from core.mcp_app import build_mcp_server

mcp = build_mcp_server()

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
