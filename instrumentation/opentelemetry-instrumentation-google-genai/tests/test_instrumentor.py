# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GoogleGenAiSdkInstrumentor."""

from google.genai.models import AsyncModels, Models

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)


def _get_interaction_classes():
    try:
        from google.genai._interactions.resources.interactions import (
            AsyncInteractionsResource,
            InteractionsResource,
        )

        return InteractionsResource, AsyncInteractionsResource
    except ImportError:
        from google.genai._gaos.interactions import (
            AsyncInteractions,
            Interactions,
        )

        return Interactions, AsyncInteractions


def test_co_filename_on_wrapped_functions():
    # ADK is relying on the __code__ attribute to suppress their instrumentation:
    # https://github.com/google/adk-python/blob/0d4d3783f7825a620c95a7b9dca919db790b879f/src/google/adk/telemetry/tracing.py#L650
    instrumentor = GoogleGenAiSdkInstrumentor()
    instrumentor.instrument()

    try:
        sync_interactions, async_interactions = _get_interaction_classes()
        wrapped_functions = [
            Models.generate_content,
            Models.generate_content_stream,
            AsyncModels.generate_content,
            AsyncModels.generate_content_stream,
            Models.embed_content,
            AsyncModels.embed_content,
            sync_interactions.create,
            async_interactions.create,
        ]

        for func in wrapped_functions:
            assert (
                "opentelemetry/instrumentation/google_genai"
                in func.__code__.co_filename
            ), f"Expected opentelemetry/instrumentation/google_genai in {func}.__code__.co_filename, got {func.__code__.co_filename}"
    finally:
        instrumentor.uninstrument()
