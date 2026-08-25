# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from opentelemetry.util.genai.context import (
    get_current_inference_span,
    set_inference_span_in_context,
)

__all__ = [
    "get_current_inference_span",
    "set_inference_span_in_context",
]
