# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Context helpers for GenAI inference spans."""

from __future__ import annotations

from opentelemetry.context import (
    Context,
    get_value,
    set_value,
)
from opentelemetry.trace import Span

_INFERENCE_SPAN_KEY = "opentelemetry.genai.inference_span"


def set_inference_span_in_context(
    span: Span,
    context: Context | None = None,
) -> Context:
    """Return a Context with the given GenAI inference span attached.

    Args:
        span: The GenAI inference span to attach.
        context: The context to attach to. Defaults to the current context.

    Returns:
        A new Context containing the inference span.
    """
    return set_value(_INFERENCE_SPAN_KEY, span, context=context)


def get_current_inference_span(
    context: Context | None = None,
) -> Span | None:
    """Return the active GenAI inference span from context, if any.

    Args:
        context: The context to inspect. Defaults to the current context.

    Returns:
        The active inference Span, or None if no inference span is in context.
    """
    span = get_value(_INFERENCE_SPAN_KEY, context=context)
    if isinstance(span, Span):
        return span
    return None
