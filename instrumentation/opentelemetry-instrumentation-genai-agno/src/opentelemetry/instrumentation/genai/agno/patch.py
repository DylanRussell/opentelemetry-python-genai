# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Agno instrumentation."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, cast

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    Text,
)

logger = logging.getLogger(__name__)

_AGNO_MODULE = "agno.agent"
_AGENT_CLASS = "Agent"


def patch_agent(handler: TelemetryHandler) -> None:
    """Apply patches to agno.agent.Agent class methods."""
    wrap_function_wrapper(
        _AGNO_MODULE,
        f"{_AGENT_CLASS}.run",
        _agent_run(handler),
    )
    wrap_function_wrapper(
        _AGNO_MODULE,
        f"{_AGENT_CLASS}.arun",
        _agent_arun(handler),
    )


def unpatch_agent() -> None:
    """Remove patches from agno.agent.Agent class methods."""
    try:
        import agno.agent

        unwrap(agno.agent.Agent, "run")
        unwrap(agno.agent.Agent, "arun")
    except (ImportError, AttributeError):
        pass


def _extract_input_content(input_val: Any) -> str:
    if isinstance(input_val, str):
        return input_val
    if hasattr(input_val, "content"):
        return str(getattr(input_val, "content"))
    return str(input_val)


def _extract_output_content(result: Any) -> str:
    if result is None:
        return ""
    if hasattr(result, "content") and getattr(result, "content") is not None:
        content = getattr(result, "content")
        if isinstance(content, str):
            return content
        return str(content)
    return str(result)


def _agent_run(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        agent_name = getattr(instance, "name", None) or "Agent"
        invocation = handler.invoke_local_agent(agent_name=agent_name)

        agent_id = getattr(instance, "agent_id", None)
        if agent_id:
            invocation.agent_id = str(agent_id)

        if capture_content and (args or "input" in kwargs):
            input_val = args[0] if args else kwargs.get("input")
            if input_val is not None:
                content_str = _extract_input_content(input_val)
                invocation.input_messages = [
                    InputMessage(role="user", parts=[Text(content=content_str)])
                ]

        try:
            result = wrapped(*args, **kwargs)
            if capture_content and result is not None:
                output_str = _extract_output_content(result)
                invocation.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=[Text(content=output_str)],
                        finish_reason="stop",
                    )
                ]
            if hasattr(result, "session_id") and getattr(result, "session_id"):
                invocation.conversation_id = str(getattr(result, "session_id"))
            if hasattr(result, "metrics") and getattr(result, "metrics"):
                metrics = getattr(result, "metrics")
                if (
                    hasattr(metrics, "input_tokens")
                    and getattr(metrics, "input_tokens") is not None
                ):
                    invocation.input_tokens = int(
                        getattr(metrics, "input_tokens")
                    )
                if (
                    hasattr(metrics, "output_tokens")
                    and getattr(metrics, "output_tokens") is not None
                ):
                    invocation.output_tokens = int(
                        getattr(metrics, "output_tokens")
                    )

            invocation.stop()
            return result
        except Exception as exc:
            invocation.fail(exc)
            raise

    return traced_method


def _agent_arun(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        agent_name = getattr(instance, "name", None) or "Agent"
        invocation = handler.invoke_local_agent(agent_name=agent_name)

        agent_id = getattr(instance, "agent_id", None)
        if agent_id:
            invocation.agent_id = str(agent_id)

        if capture_content and (args or "input" in kwargs):
            input_val = args[0] if args else kwargs.get("input")
            if input_val is not None:
                content_str = _extract_input_content(input_val)
                invocation.input_messages = [
                    InputMessage(role="user", parts=[Text(content=content_str)])
                ]

        try:
            result = await wrapped(*args, **kwargs)
            if capture_content and result is not None:
                output_str = _extract_output_content(result)
                invocation.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=[Text(content=output_str)],
                        finish_reason="stop",
                    )
                ]
            if hasattr(result, "session_id") and getattr(result, "session_id"):
                invocation.conversation_id = str(getattr(result, "session_id"))
            if hasattr(result, "metrics") and getattr(result, "metrics"):
                metrics = getattr(result, "metrics")
                if (
                    hasattr(metrics, "input_tokens")
                    and getattr(metrics, "input_tokens") is not None
                ):
                    invocation.input_tokens = int(
                        getattr(metrics, "input_tokens")
                    )
                if (
                    hasattr(metrics, "output_tokens")
                    and getattr(metrics, "output_tokens") is not None
                ):
                    invocation.output_tokens = int(
                        getattr(metrics, "output_tokens")
                    )

            invocation.stop()
            return result
        except Exception as exc:
            invocation.fail(exc)
            raise

    return cast(Callable[..., Any], traced_method)
