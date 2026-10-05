# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Context helpers for GenAI inference attributes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Final

from opentelemetry.context import Context, get_value, set_value
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    SystemInstructionPart,
    ToolDefinition,
)
from opentelemetry.util.types import AttributeValue

INFERENCE_CONTEXT_KEY: Final[str] = "opentelemetry.genai.inference_context"
_INFERENCE_CONTEXT_KEY = INFERENCE_CONTEXT_KEY


@dataclass
class InferenceData:
    """Typed data passed from inner inference invocations to the outer invocation."""

    provider: str | None = None
    request_model: str | None = None
    response_model: str | None = None
    server_address: str | None = None
    server_port: int | None = None
    response_id: str | None = None
    finish_reasons: list[str] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    seed: int | None = None
    request_choice_count: int | None = None
    output_type: str | None = None
    request_stream: bool | None = None
    ttfc_seconds: float | None = None
    cache_write_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    text_input_tokens: int | None = None
    image_input_tokens: int | None = None
    audio_input_tokens: int | None = None
    text_output_tokens: int | None = None
    image_output_tokens: int | None = None
    audio_output_tokens: int | None = None
    text_cache_read_input_tokens: int | None = None
    image_cache_read_input_tokens: int | None = None
    audio_cache_read_input_tokens: int | None = None
    reasoning_level: str | None = None
    previous_response_id: str | None = None
    conversation_compacted: bool | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    input_messages: list[InputMessage] = field(
        default_factory=list[InputMessage]
    )
    output_messages: list[OutputMessage] = field(
        default_factory=list[OutputMessage]
    )
    system_instruction: list[SystemInstructionPart] | list[MessagePart] = (
        field(default_factory=list[SystemInstructionPart])
    )
    prompt_variables: Mapping[str, object] | None = None
    tool_definitions: list[ToolDefinition] | None = None
    attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )
    metric_attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )

    def merge(self, other: InferenceData, *, overwrite: bool = True) -> None:
        """Merge another context data instance into this one.

        Args:
            other: The context data to merge from.
            overwrite: If True, values from ``other`` overwrite existing values
                (used when inner invocations publish to context). If False,
                existing non-None values in ``self`` are preserved (used when
                the root invocation enriches from context).
        """
        for f in fields(self):
            if f.name in ("attributes", "metric_attributes"):
                continue
            val = getattr(other, f.name)
            if val is not None and (
                overwrite or getattr(self, f.name) is None
            ):
                setattr(self, f.name, val)

        if overwrite:
            self.attributes.update(other.attributes)
            self.metric_attributes.update(other.metric_attributes)
        else:
            for k, v in other.attributes.items():
                self.attributes.setdefault(k, v)
            for k, v in other.metric_attributes.items():
                self.metric_attributes.setdefault(k, v)


InferenceContextData = InferenceData
InferenceNonContentCaptureData = InferenceData


def set_inference_context_data(
    data: InferenceData,
    context: Context | None = None,
) -> Context:
    """Return a Context with the given inference context data attached.

    Args:
        data: The mutable inference context data object.
        context: The context to attach to. Defaults to the current context.

    Returns:
        A new Context containing the inference context data.
    """
    return set_value(_INFERENCE_CONTEXT_KEY, data, context=context)


def get_inference_context_data(
    context: Context | None = None,
) -> InferenceData | None:
    """Return the active inference context data from context, if any.

    Args:
        context: The context to inspect. Defaults to the current context.

    Returns:
        The active inference context data, or None if not set.
    """
    data = get_value(_INFERENCE_CONTEXT_KEY, context=context)
    if isinstance(data, InferenceData):
        return data
    return None


__all__ = [
    "INFERENCE_CONTEXT_KEY",
    "InferenceContextData",
    "InferenceData",
    "InferenceNonContentCaptureData",
    "get_inference_context_data",
    "set_inference_context_data",
]
