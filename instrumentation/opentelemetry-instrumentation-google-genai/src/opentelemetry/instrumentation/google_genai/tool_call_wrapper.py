# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from typing import Any, Callable, Optional, Union

from google.genai.types import (
    ToolListUnion,
    ToolListUnionDict,
    ToolOrDict,
)

from opentelemetry.util.genai.handler import TelemetryHandler

ToolFunction = Callable[..., Any]


def _get_function_args(wrapped_function, function_args, function_kwargs):
    """Records the details about a function invocation as span attributes."""
    function_arg_attr = {}
    signature = inspect.signature(wrapped_function)
    params = list(signature.parameters.values())
    for index, entry in enumerate(function_args):
        param_name = f"args[{index}]"
        if index < len(params):
            param_name = params[index].name
        function_arg_attr[f"code.function.parameters.{param_name}.type"] = (
            type(entry).__name__
        )
        function_arg_attr[f"code.function.parameters.{param_name}.value"] = (
            entry
        )
    for key, value in function_kwargs.items():
        function_arg_attr[f"code.function.parameters.{key}.type"] = type(
            value
        ).__name__
        function_arg_attr[f"code.function.parameters.{key}.value"] = value
    return function_arg_attr


def _wrap_tool_function(
    tool_function: ToolFunction,
    telemetry_handler: TelemetryHandler,
    capture_content_on_span: bool,
):
    if inspect.iscoroutinefunction(tool_function):

        @functools.wraps(tool_function)
        async def wrapped_function(*args, **kwargs):
            with telemetry_handler.tool(
                tool_function.__name__, tool_description=tool_function.__doc__
            ) as tool_invocation:
                result = await tool_function(*args, **kwargs)
                if capture_content_on_span:
                    tool_invocation.arguments = _get_function_args(
                        tool_function, args, kwargs
                    )
                    tool_invocation.tool_result = result
            return result
    else:

        @functools.wraps(tool_function)
        def wrapped_function(*args, **kwargs):
            with telemetry_handler.tool(
                tool_function.__name__, tool_description=tool_function.__doc__
            ) as tool_invocation:
                result = tool_function(*args, **kwargs)
                if capture_content_on_span:
                    tool_invocation.arguments = _get_function_args(
                        tool_function, args, kwargs
                    )
                    tool_invocation.tool_result = result
            return result

    return wrapped_function


def wrapped_tool(
    tool_or_tools: Optional[
        Union[ToolFunction, ToolOrDict, ToolListUnion, ToolListUnionDict]
    ],
    telemetry_handler: TelemetryHandler,
    capture_content_on_span: bool,
):
    if tool_or_tools is None:
        return None
    if isinstance(tool_or_tools, list):
        return [
            wrapped_tool(tool, telemetry_handler, capture_content_on_span)
            for tool in tool_or_tools
        ]
    if isinstance(tool_or_tools, dict):
        return {
            key: wrapped_tool(tool, telemetry_handler, capture_content_on_span)
            for (key, tool) in tool_or_tools.items()
        }
    if callable(tool_or_tools):
        return _wrap_tool_function(
            tool_or_tools, telemetry_handler, capture_content_on_span
        )
    return tool_or_tools
