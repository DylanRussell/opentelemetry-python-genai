# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from collections.abc import AsyncIterable, Callable, Iterable
from typing import Any, cast

from google.genai._interactions._streaming import Stream
from google.genai._interactions.resources.interactions import (
    AsyncInteractionsResource,
    InteractionsResource,
)
from google.genai._interactions.types.interaction import Interaction, Usage
from google.genai._interactions.types.interaction_create_params import Input
from google.genai._interactions.types.interaction_sse_event import (
    InteractionSSEEvent,
)
from google.genai.types import Content
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
)
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import (
    Blob,
    GenericPart,
    InputMessage,
    MessagePart,
    OutputMessage,
    Reasoning,
    ServerToolCall,
    ServerToolCallResponse,
    Text,
    ToolCallRequest,
    ToolCallResponse,
    Uri,
)

from .generate_content import GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY


class _InteractionsMethodsSnapshot:
    def __init__(self) -> None:
        self._original_create = InteractionsResource.create
        self._original_async_create = AsyncInteractionsResource.create

    def restore(self) -> None:
        InteractionsResource.create = self._original_create
        AsyncInteractionsResource.create = self._original_async_create


def _apply_interaction_response_attributes(
    response: Interaction,
    invocation: InferenceInvocation,
    telemetry_handler: TelemetryHandler,
) -> None:
    invocation.response_model_name = response.model

    usage = response.usage or Usage()

    invocation.input_tokens = usage.total_input_tokens
    invocation.output_tokens = usage.total_output_tokens
    invocation.thinking_tokens = usage.total_thought_tokens
    invocation.cache_read_input_tokens = usage.total_cached_tokens

    if telemetry_handler.should_capture_content():
        invocation.output_messages = _interactions_response_to_messages(response)


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)

# Logic for parsing Input is tricky:
# https://github.com/open-telemetry/donation-openinference/blob/6cdd644d79fccf50aedcb614187f924ddfcafb7b/python/instrumentation/openinference-instrumentation-google-genai/src/openinference/instrumentation/google_genai/interactions_attributes.py#L103
def _content_param_to_part(part: Any) -> MessagePart:
    part_type = _get_field(part, "type")

    if part_type == "text":
        return Text(content=_get_field(part, "text") or "")

    part_text = _get_field(part, "text")
    if part_text is not None:
        return Text(content=part_text)

    inline_data = _get_field(part, "inline_data")
    if inline_data:
        return Blob(
            mime_type=_get_field(inline_data, "mime_type"),
            modality="image",
            content=_get_field(inline_data, "data") or b"",
        )

    file_data = _get_field(part, "file_data")
    if file_data:
        return Uri(
            mime_type=_get_field(file_data, "mime_type"),
            modality="image",
            uri=_get_field(file_data, "file_uri") or "",
        )

    fn_call = _get_field(part, "function_call")
    if fn_call:
        return ToolCallRequest(
            id=_get_field(fn_call, "id"),
            name=_get_field(fn_call, "name") or "",
            arguments=_get_field(fn_call, "args") or {},
        )

    fn_resp = _get_field(part, "function_response")
    if fn_resp:
        return ToolCallResponse(
            id=_get_field(fn_resp, "id"),
            response=_get_field(fn_resp, "response") or {},
        )

    mime_type = _get_field(part, "mime_type")
    uri = _get_field(part, "uri")
    data = _get_field(part, "data")

    if uri:
        return Uri(
            mime_type=mime_type,
            modality=part_type or "image",
            uri=uri,
        )
    elif data:
        content_bytes = data
        if isinstance(data, str):
            try:
                content_bytes = base64.b64decode(data)
            except Exception:
                content_bytes = data.encode("utf-8")
        return Blob(
            mime_type=mime_type,
            modality=part_type or "image",
            content=content_bytes,
        )

    return GenericPart(value=type(part).__name__)


def _get_thought_text(summary_list: Iterable[Any]) -> str:
    texts = []
    for s in summary_list:
        text = _get_field(s, "text")
        if text:
            texts.append(text)
    return "\n".join(texts)


def _interactions_input_to_messages(input_data: Input | None) -> list[InputMessage]:
    # None will end up raising an exception by the SDK
    if input_data is None:
        return []
    if isinstance(input_data, str):
        return [InputMessage(role="user", parts=[Text(content=input_data)])]

    # Content is iterable over key/value pairs, but is not a list..
    if not isinstance(input_data, Iterable) or isinstance(input_data, Content):
        input_data = [input_data]

    messages = []
    for item in input_data:
        if isinstance(item, Content):
            item = {"type": "user_input", "content": item.parts}

        item_type = _get_field(item, "type")
        if item_type == "user_input":
            parts = []
            content = _get_field(item, "content")
            for part in content:
                parts.append(_content_param_to_part(part))
            messages.append(InputMessage(role="user", parts=parts))
        elif item_type == "model_output":
            parts = []
            content = _get_field(item, "content")
            for part in content:
                parts.append(_content_param_to_part(part))
            messages.append(InputMessage(role="assistant", parts=parts))
        elif item_type == "thought":
            summary = _get_field(item, "summary")
            text = _get_thought_text(summary)
            messages.append(InputMessage(role="assistant", parts=[Reasoning(content=text)]))
        elif item_type == "function_call":
            call_id = _get_field(item, "id")
            name = _get_field(item, "name")
            arguments = _get_field(item, "arguments")
            part = ToolCallRequest(id=call_id, name=name or "", arguments=arguments)
            messages.append(InputMessage(role="assistant", parts=[part]))
        elif item_type == "function_result":
            call_id = _get_field(item, "call_id")
            result = _get_field(item, "result")
            part = ToolCallResponse(id=call_id, response=result)
            messages.append(InputMessage(role="tool", parts=[part]))
        elif item_type in (
            "code_execution_call",
            "google_search_call",
            "google_maps_call",
            "file_search_call",
            "mcp_server_tool_call",
            "url_context_call",
        ):
            call_id = _get_field(item, "id")
            arguments = _get_field(item, "arguments")
            part = ServerToolCall(name=str(item_type), server_tool_call=arguments, id=call_id)
            messages.append(InputMessage(role="assistant", parts=[part]))
        elif item_type in (
            "code_execution_result",
            "google_search_result",
            "google_maps_result",
            "file_search_result",
            "mcp_server_tool_result",
            "url_context_result",
        ):
            call_id = _get_field(item, "call_id")
            result = _get_field(item, "result")
            part = ServerToolCallResponse(server_tool_call_response=result, id=call_id)
            messages.append(InputMessage(role="tool", parts=[part]))
        elif isinstance(item, str):
            messages.append(
                InputMessage(role="user", parts=[Text(content=item)])
            )
        elif item_type is not None:
            part = GenericPart(value=type(item).__name__)
            messages.append(InputMessage(role="user", parts=[part]))

    return messages


def _interactions_response_to_messages(interaction: Interaction) -> list[OutputMessage]:
    messages = []
    for step in _get_field(interaction, "steps") or []:
        if _get_field(step, "type") == "model_output":
            parts = []
            for part in _get_field(step, "content") or []:
                part_text = _get_field(part, "text")
                if part_text is not None:
                    parts.append(Text(content=part_text))
            messages.append(
                OutputMessage(
                    role="assistant",
                    parts=parts,
                    finish_reason="stop",
                )
            )
            break
    return messages


class InteractionsStreamWrapper(SyncStreamWrapper[InteractionSSEEvent]):
    def __init__(
        self,
        stream: Iterable[InteractionSSEEvent],
        invocation: InferenceInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream)
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_last_interaction: Interaction | None = None

    def _process_chunk(self, chunk: InteractionSSEEvent) -> None:
        event_type = _get_field(chunk, "event_type")
        if event_type == "interaction_completed":
            interaction = _get_field(chunk, "interaction")
            if interaction:
                self._self_last_interaction = interaction

    def _on_stream_end(self) -> None:
        if self._self_last_interaction:
            _apply_interaction_response_attributes(
                self._self_last_interaction,
                self._self_invocation,
                self._self_telemetry_handler,
            )
        self._self_invocation.stop()

    def _on_stream_error(self, error: Exception) -> None:
        self._self_invocation.fail(error)


class AsyncInteractionsStreamWrapper(AsyncStreamWrapper[InteractionSSEEvent]):
    def __init__(
        self,
        stream: AsyncIterable[InteractionSSEEvent],
        invocation: InferenceInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream)
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_last_interaction: Interaction | None = None

    def _process_chunk(self, chunk: InteractionSSEEvent) -> None:
        event_type = _get_field(chunk, "event_type")
        if event_type == "interaction_completed":
            interaction = _get_field(chunk, "interaction")
            if interaction:
                self._self_last_interaction = interaction

    def _on_stream_end(self) -> None:
        if self._self_last_interaction:
            _apply_interaction_response_attributes(
                self._self_last_interaction,
                self._self_invocation,
                self._self_telemetry_handler,
            )
        self._self_invocation.stop()

    def _on_stream_error(self, error: Exception) -> None:
        self._self_invocation.fail(error)


def _create_instrumented_interactions_create(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Interaction | Stream[InteractionSSEEvent]],
        InteractionsResource,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Interaction | InteractionsStreamWrapper,
]:
    def instrumented_interactions_create(
        wrapped: Callable[..., Interaction | Stream[InteractionSSEEvent]],
        instance: InteractionsResource,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Interaction | InteractionsStreamWrapper:
        # Verrex ai does not support the interactions API yet, but eventually will.
        # SDK will raise an exception if model or agent is not passed or if input data is not passed.
        invocation = telemetry_handler.inference(
            provider=(
                GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
                if getattr(instance._client, "_is_vertex", False)
                else GenAIAttributes.GenAiSystemValues.GEMINI.value
            ),
            request_model=kwargs.get("model") or kwargs.get("agent") or "unknown",
            operation_name="interactions.create",
            server_address=getattr(instance._client, "server", None),
        )

        attrs = context_api.get_value(
            GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
        )
        if attrs:
            invocation.attributes.update(dict(attrs))

        if telemetry_handler.should_capture_content():
            invocation.input_messages = _interactions_input_to_messages(kwargs.get("input"))
            if system_instruction := kwargs.get("system_instruction"):
                invocation.system_instruction = [Text(content=system_instruction)]

        if kwargs.get("stream", False):
            return InteractionsStreamWrapper(
                wrapped(*args, **kwargs), invocation, telemetry_handler
        )
        try:
            response = wrapped(*args, **kwargs)
            _apply_interaction_response_attributes(
                response, invocation, telemetry_handler
            )
            invocation.stop()
            return response
        except Exception as exc:
            invocation.fail(exc)
            raise

    return instrumented_interactions_create


def _create_instrumented_async_interactions_create(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Any],
        AsyncInteractionsResource,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Any,
]:
    async def instrumented_interactions_create(
        wrapped: Callable[..., Any],
        instance: AsyncInteractionsResource,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Interaction | AsyncInteractionsStreamWrapper:
        invocation = telemetry_handler.inference(
            provider=(
                GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
                if getattr(instance._client, "_is_vertex", False)
                else GenAIAttributes.GenAiSystemValues.GEMINI.value
            ),
            request_model=kwargs.get("model") or kwargs.get("agent") or "unknown",
            operation_name="interactions.create",
            server_address=getattr(instance._client, "server", None),
        )

        attrs = context_api.get_value(
            GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
        )
        if attrs:
            invocation.attributes.update(dict(attrs))

        if telemetry_handler.should_capture_content():
            invocation.input_messages = _interactions_input_to_messages(kwargs.get("input"))
            if system_instruction := kwargs.get("system_instruction"):
                invocation.system_instruction = [Text(content=system_instruction)]

        if kwargs.get("stream", False):
            return AsyncInteractionsStreamWrapper(
                await wrapped(*args, **kwargs),
                invocation,
                telemetry_handler,
            )
        try:
            response = cast(Interaction, await wrapped(*args, **kwargs))
            _apply_interaction_response_attributes(
                response, invocation, telemetry_handler
            )
            invocation.stop()
            return response
        except Exception as exc:
            invocation.fail(exc)
            raise

    return instrumented_interactions_create


def uninstrument_interactions(snapshot: object) -> None:
    assert isinstance(snapshot, _InteractionsMethodsSnapshot)
    snapshot.restore()


def instrument_interactions(
    telemetry_handler: TelemetryHandler,
) -> object:
    snapshot = _InteractionsMethodsSnapshot()
    wrap_function_wrapper(
        "google.genai._interactions.resources.interactions",
        "InteractionsResource.create",
        _create_instrumented_interactions_create(telemetry_handler),
    )
    wrap_function_wrapper(
        "google.genai._interactions.resources.interactions",
        "AsyncInteractionsResource.create",
        _create_instrumented_async_interactions_create(telemetry_handler),
    )
    return snapshot
