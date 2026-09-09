# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.util.genai.context import (
    INFERENCE_ATTRIBUTES_KEY,
    get_inference_attributes,
    set_inference_attributes,
)

__all__ = [
    "INFERENCE_ATTRIBUTES_KEY",
    "get_inference_attributes",
    "set_inference_attributes",
]
