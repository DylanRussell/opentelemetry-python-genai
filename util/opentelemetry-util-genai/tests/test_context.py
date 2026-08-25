# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from opentelemetry.context import attach, detach, get_value, set_value
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai import (
    INFERENCE_SPAN_KEY,
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

    def test_key_constant_value(self):
        self.assertEqual(
            INFERENCE_SPAN_KEY, "opentelemetry.genai.inference_span"
        )

    def test_get_current_inference_span_none_by_default(self):
        self.assertIsNone(get_current_inference_span())

    def test_set_and_get_inference_span_in_context(self):
        span = self.tracer.start_span("test_span")
        ctx = set_inference_span_in_context(span)
        self.assertIs(get_current_inference_span(ctx), span)
        self.assertIsNone(get_current_inference_span())

        token = attach(ctx)
        try:
            self.assertIs(get_current_inference_span(), span)
        finally:
            detach(token)
        self.assertIsNone(get_current_inference_span())
        span.end()

    def test_plain_string_key_interoperability(self):
        span = self.tracer.start_span("external_native_span")
        # An external native library sets the well-known string key
        ctx = set_value("opentelemetry.genai.inference_span", span)
        # Our helper can read it
        self.assertIs(get_current_inference_span(ctx), span)

        # Our helper sets it, and an external library reading the string key gets it
        ctx2 = set_inference_span_in_context(span)
        self.assertIs(
            get_value("opentelemetry.genai.inference_span", ctx2), span
        )
        self.assertIs(get_value(INFERENCE_SPAN_KEY, ctx2), span)
        span.end()

    def test_inference_invocation_attaches_and_cleans_up_context(self):
        self.assertIsNone(get_current_inference_span())

        invocation = self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        )
        self.assertIs(get_current_inference_span(), invocation.span)

        invocation.stop()
        self.assertIsNone(get_current_inference_span())

    def test_inference_invocation_cleans_up_on_fail(self):
        self.assertIsNone(get_current_inference_span())

        invocation = self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        )
        self.assertIs(get_current_inference_span(), invocation.span)

        invocation.fail(ValueError("test error"))
        self.assertIsNone(get_current_inference_span())

    def test_inference_invocation_context_manager(self):
        self.assertIsNone(get_current_inference_span())

        with self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        ) as invocation:
            self.assertIs(get_current_inference_span(), invocation.span)

            # Downstream instrumentation can fetch and modify the span
            downstream_span = get_current_inference_span()
            self.assertIsNotNone(downstream_span)
            if downstream_span is not None:
                downstream_span.set_attribute("custom.attribute", "enriched")

        self.assertIsNone(get_current_inference_span())

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes.get("custom.attribute"), "enriched"
        )

    def test_non_inference_invocations_do_not_set_inference_span(self):
        self.assertIsNone(get_current_inference_span())

        with self.handler.invoke_local_agent(agent_name="MathTutor"):
            self.assertIsNone(get_current_inference_span())

            # Nested inference invocation properly sets the inference span
            with self.handler.inference(
                "openai", request_model="gpt-4o-mini"
            ) as inf_inv:
                self.assertIs(get_current_inference_span(), inf_inv.span)

            self.assertIsNone(get_current_inference_span())

        self.assertIsNone(get_current_inference_span())
