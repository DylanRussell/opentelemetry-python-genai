# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GoogleGenAiSdkInstrumentor."""

from google.genai.models import AsyncModels, Models

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)


from opentelemetry.test_util_genai.instrumentor import instrument


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


def test_co_filename_on_wrapped_functions(
    tracer_provider, logger_provider, meter_provider
):
    # ADK is relying on the __code__ attribute to suppress their instrumentation:
    # https://github.com/google/adk-python/blob/0d4d3783f7825a620c95a7b9dca919db790b879f/src/google/adk/telemetry/tracing.py#L650
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

    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        for func in wrapped_functions:
            co_filename = func.__code__.co_filename.replace("\\", "/")
            assert (
                "opentelemetry/instrumentation/google_genai" in co_filename
            ), f"Expected opentelemetry/instrumentation/google_genai in {co_filename}"

    for func in wrapped_functions:
        co_filename = func.__code__.co_filename.replace("\\", "/")
        assert (
            "opentelemetry/instrumentation/google_genai" not in co_filename
        ), f"Expected opentelemetry/instrumentation/google_genai removed from {co_filename} upon uninstrument"
