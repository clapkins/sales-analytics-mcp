"""Тесты сборки MCP-сервера из реестра скиллов.

Ключевой тест здесь — сквозной, через настоящий fastmcp.Client, а не
вызов skill.run() напрямую: он проверяет ровно то, что видит реальный
MCP-клиент (список инструментов, схемы, и то, что ошибка приходит как
обычный результат вызова, а не протокольный сбой) — главный критерий
приёмки этапа 5.
"""

from pathlib import Path

import pytest
from fastmcp.client import Client

from core.mcp_app import build_mcp_server

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"

REQUIRED_TOOL_NAMES = {
    "load_data",
    "describe_data",
    "plot_trend",
    "plot_distribution",
    "correlation_analysis",
}


@pytest.mark.anyio
async def test_all_required_and_extra_tools_are_registered() -> None:
    """Клиент видит и обязательные по ТЗ имена, и дополнительные инструменты."""
    mcp = build_mcp_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()

    names = {tool.name for tool in tools}
    assert REQUIRED_TOOL_NAMES <= names
    assert {"clean_data", "list_datasets", "auto_analyze", "export_report"} <= names


@pytest.mark.anyio
async def test_tool_schemas_are_not_generic_object() -> None:
    """Схема параметров реально построена из сигнатуры run(), а не пустой object."""
    mcp = build_mcp_server()
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    load_data_schema = tools["load_data"].inputSchema
    assert set(load_data_schema["properties"]) == {"file_path"}
    assert load_data_schema["required"] == ["file_path"]


@pytest.mark.anyio
async def test_system_prompt_is_registered_and_matches_file() -> None:
    """MCP-примитив prompt отдаёт тот же текст, что лежит в prompts/system_prompt.md."""
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.md"
    expected_text = prompt_path.read_text(encoding="utf-8")

    mcp = build_mcp_server()
    async with Client(mcp) as client:
        prompts = await client.list_prompts()
        assert any(p.name == "sales_analysis_workflow" for p in prompts)

        result = await client.get_prompt("sales_analysis_workflow")

    assert result.messages[0].content.text == expected_text


@pytest.mark.anyio
async def test_unknown_column_returns_hint_not_protocol_error() -> None:
    """Главный критерий приёмки этапа 5: неверная колонка → allowed/suggestion, не трейсбек."""
    mcp = build_mcp_server()
    async with Client(mcp) as client:
        load_result = await client.call_tool("load_data", {"file_path": str(DATA_PATH)})
        dataset_id = load_result.data["dataset_id"]

        bad_result = await client.call_tool(
            "plot_distribution", {"dataset_id": dataset_id, "column": "Sale"}
        )

    assert bad_result.is_error is False
    assert bad_result.data["ok"] is False
    assert bad_result.data["error_code"] == "unknown_column"
    assert "Sales" in bad_result.data["allowed"]
    assert bad_result.data["suggestion"] == "Sales"


@pytest.mark.anyio
async def test_full_pipeline_through_client_load_clean_plot_insights() -> None:
    """Полный сценарий диалога через клиента: загрузка → очистка → график → инсайты."""
    mcp = build_mcp_server()
    async with Client(mcp) as client:
        load_result = await client.call_tool("load_data", {"file_path": str(DATA_PATH)})
        raw_id = load_result.data["dataset_id"]

        clean_result = await client.call_tool("clean_data", {"dataset_id": raw_id})
        clean_id = clean_result.data["dataset_id"]
        assert clean_result.data["operations_log"]["duplicates_removed"] == 5

        trend_result = await client.call_tool(
            "plot_trend", {"dataset_id": clean_id, "x_col": "Date", "y_col": "Sales"}
        )
        assert trend_result.data["ok"] is True

        insights_result = await client.call_tool(
            "prepare_insights_context",
            {
                "dataset_id": clean_id,
                "chart_descriptions": [
                    {
                        "chart_type": "trend",
                        "path": trend_result.data["path"],
                        "description": trend_result.data["description"],
                    }
                ],
            },
        )

    assert insights_result.data["cleaning_log"]["duplicates_removed"] == 5
    assert len(insights_result.data["chart_descriptions"]) == 1
