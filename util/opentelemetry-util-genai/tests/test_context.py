# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai import (
    get_current_inference_span,
    set_inference_span_in_context,
)
from opentelemetry.util.genai.handler import TelemetryHandler


class TestInferenceSpanContext(unittest.TestCase):
    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        self.tracer = self.tracer_provider.get_tracer(__name__)

    def test_get_current_inference_span_none_by_default(self):
        assert get_current_inference_span() is None

    def test_set_and_get_inference_span_in_context(self):
        span = self.tracer.start_span("test_span")
        ctx = set_inference_span_in_context(span)
        assert get_current_inference_span(ctx) is span
        assert get_current_inference_span() is None

        token = attach(ctx)
        try:
            assert get_current_inference_span() is span
        finally:
            detach(token)
        assert get_current_inference_span() is None
        span.end()

    def test_plain_string_key_interoperability(self):
        from opentelemetry.context import get_value, set_value

        span = self.tracer.start_span("external_native_span")
        # An external native library sets the well-known string key
        ctx = set_value("opentelemetry.genai.inference_span", span)
        # Our helper can read it
        assert get_current_inference_span(ctx) is span

        # Our helper sets it, and an external library reading the string key gets it
        ctx2 = set_inference_span_in_context(span)
        assert get_value("opentelemetry.genai.inference_span", ctx2) is span
        span.end()

    def test_inference_invocation_attaches_and_cleans_up_context(self):
        assert get_current_inference_span() is None

        invocation = self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        )
        assert get_current_inference_span() is invocation.span

        invocation.stop()
        assert get_current_inference_span() is None

    def test_inference_invocation_cleans_up_on_fail(self):
        assert get_current_inference_span() is None

        invocation = self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        )
        assert get_current_inference_span() is invocation.span

        invocation.fail(ValueError("test error"))
        assert get_current_inference_span() is None

    def test_inference_invocation_context_manager(self):
        assert get_current_inference_span() is None

        with self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        ) as invocation:
            assert get_current_inference_span() is invocation.span

            # Downstream instrumentation can fetch and modify the span
            downstream_span = get_current_inference_span()
            assert downstream_span is not None
            downstream_span.set_attribute("custom.attribute", "enriched")

        assert get_current_inference_span() is None

        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("custom.attribute") == "enriched"

    def test_non_inference_invocations_do_not_set_inference_span(self):
        assert get_current_inference_span() is None

        with self.handler.invoke_local_agent(agent_name="MathTutor"):
            assert get_current_inference_span() is None

            # Nested inference invocation properly sets the inference span
            with self.handler.inference(
                "openai", request_model="gpt-4o-mini"
            ) as inf_inv:
                assert get_current_inference_span() is inf_inv.span

            assert get_current_inference_span() is None

        assert get_current_inference_span() is None
