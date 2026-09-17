# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for standalone Agno Embedder instrumentation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from agno.knowledge.embedder.base import Embedder

from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
)
from opentelemetry.trace.status import StatusCode


@dataclass
class SimpleEmbedder(Embedder):
    id: str = "text-embedding-3-small"
    provider: str = "openai"

    def get_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5]


@dataclass
class UsageEmbedder(Embedder):
    model: str = "text-embedding-usage"
    provider: str = "custom_provider"

    def get_embedding_and_usage(
        self, text: str
    ) -> tuple[list[float], dict[str, int]]:
        return [0.1, 0.2, 0.3], {"input_tokens": 12, "total_tokens": 12}


@dataclass
class AsyncEmbedder(Embedder):
    id: str = "async-embed-model"
    provider: str = "cohere"

    async def async_get_embedding(self, text: str) -> list[float]:
        return [0.7, 0.8]

    async def async_get_embedding_and_usage(
        self, text: str
    ) -> tuple[list[float], dict[str, int]]:
        return [0.7, 0.8], {"prompt_tokens": 5}


@dataclass
class NestedCallingEmbedder(Embedder):
    id: str = "nested-embedder"

    def get_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2]

    def get_embedding_and_usage(
        self, text: str
    ) -> tuple[list[float], dict[str, int]]:
        # Internally calls get_embedding
        vec = self.get_embedding(text)
        return vec, {"input_tokens": 4}

    async def async_get_embedding(self, text: str) -> list[float]:
        return [0.3, 0.4]

    async def async_get_embedding_and_usage(
        self, text: str
    ) -> tuple[list[float], dict[str, int]]:
        # Internally calls async_get_embedding
        vec = await self.async_get_embedding(text)
        return vec, {"input_tokens": 6}


@dataclass
class FailingEmbedder(Embedder):
    id: str = "fail-embedder"

    def get_embedding(self, text: str) -> list[float]:
        raise ValueError("Invalid embedding request")


def test_embedder_get_embedding_sync(
    instrument_agno,
    span_exporter,
) -> None:
    """Test sync get_embedding on an Embedder subclass."""
    embedder = SimpleEmbedder()
    res = embedder.get_embedding("test string for embedding")
    assert res == [0.1, 0.2, 0.3, 0.4, 0.5]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings text-embedding-3-small"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_PROVIDER_NAME) == "openai"
    assert (
        span.attributes.get(GEN_AI_REQUEST_MODEL) == "text-embedding-3-small"
    )
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 5
    assert span.status.status_code != StatusCode.ERROR


def test_embedder_get_embedding_and_usage_sync(
    instrument_agno,
    span_exporter,
) -> None:
    """Test sync get_embedding_and_usage records input_tokens attribute."""
    embedder = UsageEmbedder()
    vec, usage = embedder.get_embedding_and_usage("sample text")
    assert vec == [0.1, 0.2, 0.3]
    assert usage.get("input_tokens") == 12

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings text-embedding-usage"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_PROVIDER_NAME) == "custom_provider"
    assert span.attributes.get(GEN_AI_REQUEST_MODEL) == "text-embedding-usage"
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 3
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 12


def test_embedder_async_get_embedding(
    instrument_agno,
    span_exporter,
) -> None:
    """Test async async_get_embedding on an Embedder subclass."""
    embedder = AsyncEmbedder()

    async def _test() -> None:
        res = await embedder.async_get_embedding("async text")
        assert res == [0.7, 0.8]

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings async-embed-model"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_PROVIDER_NAME) == "cohere"
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2


def test_embedder_async_get_embedding_and_usage(
    instrument_agno,
    span_exporter,
) -> None:
    """Test async async_get_embedding_and_usage records usage."""
    embedder = AsyncEmbedder()

    async def _test() -> None:
        vec, usage = await embedder.async_get_embedding_and_usage("async text")
        assert vec == [0.7, 0.8]
        assert usage.get("prompt_tokens") == 5

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 5
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2


def test_embedder_nested_call_suppression_sync(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that when get_embedding_and_usage calls get_embedding, only 1 span is emitted."""
    embedder = NestedCallingEmbedder()
    vec, usage = embedder.get_embedding_and_usage("nested test")
    assert vec == [0.1, 0.2]
    assert usage.get("input_tokens") == 4

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 4


def test_embedder_nested_call_suppression_async(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that when async_get_embedding_and_usage calls async_get_embedding, only 1 span is emitted."""
    embedder = NestedCallingEmbedder()

    async def _test() -> None:
        vec, usage = await embedder.async_get_embedding_and_usage(
            "nested test"
        )
        assert vec == [0.3, 0.4]
        assert usage.get("input_tokens") == 6

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 6


def test_embedder_error_handling(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error raised by get_embedding marks the span as failed."""
    embedder = FailingEmbedder()

    with pytest.raises(ValueError, match="Invalid embedding request"):
        embedder.get_embedding("failing input")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "ValueError"


def test_embedder_uninstrument_restores_untraced_behavior(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that uninstrumenting restores clean behavior and stops emitting spans."""
    embedder = SimpleEmbedder()
    embedder.get_embedding("traced")
    assert len(span_exporter.get_finished_spans()) == 1

    instrument_agno.uninstrument()
    span_exporter.clear()

    embedder.get_embedding("untraced")
    assert len(span_exporter.get_finished_spans()) == 0
