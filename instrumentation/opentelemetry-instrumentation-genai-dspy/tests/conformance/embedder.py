# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Embedder execution for DSPy."""

from __future__ import annotations

from typing import Any

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class _FallbackArray:
    def __init__(self, data: list[Any]) -> None:
        self._data = data
        if data and isinstance(data[0], list):
            self.shape: tuple[int, ...] = (len(data), len(data[0]))
            self.size = len(data) * len(data[0])
        else:
            self.shape = (len(data),)
            self.size = len(data)

    def __getitem__(self, idx: int) -> Any:
        item = self._data[idx]
        return _FallbackArray(item) if isinstance(item, list) else item


class _FallbackNP:
    float32 = float

    @staticmethod
    def array(data: Any, dtype: Any = None) -> _FallbackArray:
        if isinstance(data, _FallbackArray):
            return data
        return _FallbackArray(list(data))


def _dummy_embedder(texts: list[str], **kwargs: Any) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class EmbedderScenario(Scenario):
    expected_spans = {"embeddings": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="gen_ai.response.model",
        ),
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="gen_ai.usage.input_tokens",
        ),
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        import dspy.clients.embedding

        try:
            _ = dspy.clients.embedding.np.array
        except ImportError:
            dspy.clients.embedding.np = _FallbackNP()

        with instrument(
            DSPyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            embedder = dspy.Embedder(
                _dummy_embedder,
                caching=False,
                encoding_format="float",
                api_base="https://api.openai.com:443/v1",
            )
            embedder(["hello", "world"])
