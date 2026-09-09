# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from unittest.mock import patch

from opentelemetry.context import attach, detach
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai import (
    INFERENCE_ATTRIBUTES_KEY,
    get_inference_attributes,
    set_inference_attributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.types import AttributeValue


class TestInferenceContext(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def _harvest_metrics(self) -> dict[str, list[object]]:
        metrics = self.get_sorted_metrics()
        metrics_by_name: dict[str, list[object]] = {}
        for metric in metrics or []:
            points = getattr(metric.data, "data_points", None) or []
            metrics_by_name.setdefault(metric.name, []).extend(points)
        return metrics_by_name

    def test_context_key_constant_value(self) -> None:
        self.assertEqual(
            INFERENCE_ATTRIBUTES_KEY,
            "opentelemetry.genai.inference_attributes",
        )

    def test_get_inference_attributes_none_by_default(self) -> None:
        self.assertIsNone(get_inference_attributes())

    def test_set_and_get_inference_attributes(self) -> None:
        attrs = {"test.attr": "value"}
        ctx = set_inference_attributes(attrs)
        self.assertIs(get_inference_attributes(ctx), attrs)
        self.assertIsNone(get_inference_attributes())

        token = attach(ctx)
        try:
            self.assertIs(get_inference_attributes(), attrs)
        finally:
            detach(token)
        self.assertIsNone(get_inference_attributes())

    def test_in_place_mutation_of_inference_attributes(self) -> None:
        attrs: dict[str, AttributeValue] = {"initial": 1}
        ctx = set_inference_attributes(attrs)
        token = attach(ctx)
        try:
            current = get_inference_attributes()
            self.assertIsNotNone(current)
            assert current is not None
            current.update({"updated": 2, "initial": 10})

            after = get_inference_attributes()
            assert after is not None
            self.assertEqual(after["initial"], 10)
            self.assertEqual(after["updated"], 2)
            self.assertIs(after, attrs)
        finally:
            detach(token)

    def test_inference_invocation_sets_attributes_on_context(self) -> None:
        self.assertIsNone(get_inference_attributes())

        with self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        ) as invocation:
            self.assertFalse(invocation.already_started)
            attrs = get_inference_attributes()
            self.assertIsNotNone(attrs)
            assert attrs is not None
            # Attributes are not published to context until publish_to_context() is called
            self.assertEqual(attrs, {})

            invocation.publish_to_context()
            self.assertEqual(attrs.get(GenAI.GEN_AI_PROVIDER_NAME), "openai")
            self.assertEqual(
                attrs.get(GenAI.GEN_AI_REQUEST_MODEL), "gpt-4o-mini"
            )

        self.assertIsNone(get_inference_attributes())

    def test_inference_invocation_automatic_publish_on_finish(self) -> None:
        with self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        ) as invocation:
            self.assertFalse(invocation.already_started)
            invocation.input_tokens = 10
            # did not call publish_to_context()

        # Span attributes were populated automatically upon finish
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes.get(GenAI.GEN_AI_PROVIDER_NAME), "openai"
        )
        self.assertEqual(
            spans[0].attributes.get(GenAI.GEN_AI_USAGE_INPUT_TOKENS), 10
        )

    def test_non_inference_invocations_do_not_set_inference_attributes(
        self,
    ) -> None:
        self.assertIsNone(get_inference_attributes())

        with self.handler.invoke_local_agent(agent_name="MathTutor"):
            self.assertIsNone(get_inference_attributes())

            # Nested inference invocation properly sets the inference attributes
            with self.handler.inference(
                "openai", request_model="gpt-4o-mini"
            ) as inf_inv:
                self.assertFalse(inf_inv.already_started)
                self.assertIsNotNone(get_inference_attributes())

            self.assertIsNone(get_inference_attributes())

    def test_nested_inference_deduplication_and_enrichment(self) -> None:
        log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(log_exporter)
        )
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
            logger_provider=logger_provider,
        )

        with patch.dict(
            os.environ, {"OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true"}
        ):
            with handler.inference(
                "openai", request_model="gpt-4o"
            ) as root_inv:
                self.assertFalse(root_inv.already_started)
                root_inv.publish_to_context()

                with handler.inference(
                    "openai",
                    request_model="gpt-4o",
                    server_address="api.openai.com",
                    server_port=443,
                ) as nested_inv:
                    self.assertTrue(nested_inv.already_started)
                    nested_inv.publish_to_context()
                    nested_inv.input_tokens = 15
                    nested_inv.output_tokens = 25
                    nested_inv.response_model_name = "gpt-4o-2024-08-06"
                    nested_inv.attributes["custom.downstream"] = "enriched"

                # While still in root context, context attributes contain downstream enrichment
                attrs = get_inference_attributes()
                self.assertIsNotNone(attrs)
                assert attrs is not None
                self.assertEqual(
                    attrs.get(server_attributes.SERVER_ADDRESS),
                    "api.openai.com",
                )
                self.assertEqual(attrs.get(server_attributes.SERVER_PORT), 443)
                self.assertEqual(
                    attrs.get("gen_ai.response.model"),
                    "gpt-4o-2024-08-06",
                )
                self.assertEqual(attrs.get("gen_ai.usage.input_tokens"), 15)
                self.assertEqual(attrs.get("gen_ai.usage.output_tokens"), 25)
                self.assertEqual(attrs.get("custom.downstream"), "enriched")

                # After nested exit, span is NOT ended yet (still recording)
                self.assertTrue(root_inv.span.is_recording())
                spans = self.span_exporter.get_finished_spans()
                self.assertEqual(len(spans), 0)

            # After root exit, root span is ended and event is emitted
            spans = self.span_exporter.get_finished_spans()
            self.assertEqual(len(spans), 1)
            span = spans[0]
            self.assertEqual(
                span.attributes.get("gen_ai.provider.name"), "openai"
            )
            self.assertEqual(
                span.attributes.get(server_attributes.SERVER_ADDRESS),
                "api.openai.com",
            )
            self.assertEqual(
                span.attributes.get(server_attributes.SERVER_PORT), 443
            )
            self.assertEqual(
                span.attributes.get("gen_ai.response.model"),
                "gpt-4o-2024-08-06",
            )
            self.assertEqual(
                span.attributes.get("gen_ai.usage.input_tokens"), 15
            )
            self.assertEqual(
                span.attributes.get("gen_ai.usage.output_tokens"), 25
            )
            self.assertEqual(
                span.attributes.get("custom.downstream"), "enriched"
            )

            logs = log_exporter.get_finished_logs()
            self.assertEqual(len(logs), 1)
            event = logs[0].log_record
            self.assertEqual(
                event.event_name, "gen_ai.client.inference.operation.details"
            )
            self.assertIsNotNone(event.attributes)
            assert event.attributes is not None
            self.assertEqual(
                event.attributes.get(server_attributes.SERVER_ADDRESS),
                "api.openai.com",
            )
            self.assertEqual(
                event.attributes.get(server_attributes.SERVER_PORT), 443
            )
            self.assertEqual(
                event.attributes.get("gen_ai.response.model"),
                "gpt-4o-2024-08-06",
            )
            self.assertEqual(
                event.attributes.get("gen_ai.usage.input_tokens"), 15
            )
            self.assertEqual(
                event.attributes.get("gen_ai.usage.output_tokens"), 25
            )
            self.assertEqual(
                event.attributes.get("custom.downstream"), "enriched"
            )

            # Check metrics - exactly 1 operation duration point for the deduplicated invocation
            metrics = self._harvest_metrics()
            self.assertIn("gen_ai.client.operation.duration", metrics)
            duration_points = metrics["gen_ai.client.operation.duration"]
            self.assertEqual(len(duration_points), 1)

    def test_nested_inference_invocation_does_not_end_span_on_fail(
        self,
    ) -> None:
        with self.handler.inference(
            "upstream", request_model="gpt-4o"
        ) as root_inv:
            with self.assertRaises(ValueError) as handler_error:
                with self.handler.inference(
                    "downstream", request_model="gpt-4o"
                ) as nested_inv:
                    self.assertTrue(nested_inv.already_started)
                    raise ValueError("downstream network failure")

            self.assertEqual(
                str(handler_error.exception), "downstream network failure"
            )
            # Root span must NOT be ended yet
            self.assertTrue(root_inv.span.is_recording())
            self.assertEqual(len(self.span_exporter.get_finished_spans()), 0)

            # Downstream error is captured in context attributes
            attrs = get_inference_attributes()
            self.assertIsNotNone(attrs)
            assert attrs is not None
            self.assertEqual(
                attrs.get(error_attributes.ERROR_TYPE), "ValueError"
            )

        # After root finishes normally, 1 span is ended
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

    def test_downstream_streaming_record_stream_chunk(self) -> None:
        with self.handler.inference(
            "upstream", request_model="gpt-4o"
        ) as root_inv:
            self.assertFalse(root_inv.already_started)
            with self.handler.inference("downstream") as nested_inv:
                self.assertTrue(nested_inv.already_started)
                nested_inv.record_stream_chunk()
                nested_inv.record_stream_chunk()
                self.assertIsNotNone(nested_inv._ttfc_seconds)

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

    def test_llm_invocation_already_started(self) -> None:
        from opentelemetry.util.genai._inference_invocation import (
            LLMInvocation,
        )

        inv = LLMInvocation(request_model="test")
        self.assertFalse(inv.already_started)

        with self.handler.inference("upstream"):
            nested_inv = LLMInvocation(request_model="nested")
            self.handler.start_llm(nested_inv)
            self.assertTrue(nested_inv.already_started)
            nested_inv.attributes["custom.llm"] = "val"
            nested_inv.publish_to_context()
            attrs = get_inference_attributes()
            self.assertIsNotNone(attrs)
            assert attrs is not None
            self.assertEqual(attrs.get("custom.llm"), "val")
            self.assertEqual(attrs.get(GenAI.GEN_AI_REQUEST_MODEL), "nested")
            self.handler.stop_llm(nested_inv)
