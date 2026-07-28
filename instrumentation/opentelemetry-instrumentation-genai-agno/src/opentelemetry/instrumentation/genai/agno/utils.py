# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Agno instrumentation."""

from __future__ import annotations

from typing import Any, Iterable

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    ToolDefinition,
)


def _extract_desc(tool: Any) -> str | None:
    desc = getattr(tool, "description", None)
    if not desc and getattr(tool, "entrypoint", None):
        desc = getattr(tool.entrypoint, "__doc__", None)
    return str(desc).strip() if desc else None


def prepare_tool_definitions(
    tools: Iterable[Any] | None,
) -> list[ToolDefinition] | None:
    """Extract tool definitions from Agno Agent tools."""
    if not tools:
        return None

    seen_names: set[str] = set()
    definitions: list[ToolDefinition] = []

    def _add_def(name: str, desc: str | None, params: Any) -> None:
        if not name or name in seen_names:
            return
        seen_names.add(name)
        definitions.append(
            FunctionToolDefinition(
                name=name,
                description=desc,
                parameters=params,
            )
        )

    for tool in tools:
        if isinstance(tool, dict):
            if (
                "type" in tool
                and tool.get("type") == "function"
                and isinstance(tool.get("function"), dict)
            ):
                func_dict = tool["function"]
                _add_def(
                    str(func_dict.get("name") or ""),
                    str(func_dict["description"])
                    if func_dict.get("description") is not None
                    else None,
                    func_dict.get("parameters"),
                )
            elif "name" in tool:
                _add_def(
                    str(tool.get("name") or ""),
                    str(tool["description"])
                    if tool.get("description") is not None
                    else None,
                    tool.get("parameters"),
                )
        elif hasattr(tool, "get_functions") and callable(
            getattr(tool, "get_functions", None)
        ):
            try:
                funcs = tool.get_functions()
                if isinstance(funcs, dict):
                    sub_defs = prepare_tool_definitions(list(funcs.values()))
                    if sub_defs:
                        for defn in sub_defs:
                            _add_def(
                                getattr(defn, "name", "") or "",
                                getattr(defn, "description", None),
                                getattr(defn, "parameters", None),
                            )
            except Exception:
                pass
        elif hasattr(tool, "name") and hasattr(tool, "parameters"):
            name = getattr(tool, "name", "") or ""
            desc = _extract_desc(tool)
            params = getattr(tool, "parameters", None)
            _add_def(
                str(name),
                desc,
                params,
            )
        elif callable(tool):
            try:
                from agno.tools.function import Function

                func = Function.from_callable(tool)
                name = getattr(func, "name", "") or ""
                desc = _extract_desc(func)
                params = getattr(func, "parameters", None)
                _add_def(
                    str(name),
                    desc,
                    params,
                )
            except Exception:
                name = getattr(tool, "__name__", str(tool))
                desc = getattr(tool, "__doc__", None)
                _add_def(
                    str(name),
                    str(desc).strip() if desc is not None else None,
                    None,
                )

    return definitions or None
