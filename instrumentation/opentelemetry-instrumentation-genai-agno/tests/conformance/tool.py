# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: function tool execution for Agno."""

from __future__ import annotations

from typing import Any

from agno.tools.function import Function, FunctionCall

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class ToolScenario(Scenario):
    expected_spans = {"execute_tool": 1}

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):

            def sample_tool(x: int) -> int:
                """Double a number."""
                return x * 2

            func = Function.from_callable(sample_tool)
            func_call = FunctionCall(
                function=func,
                arguments={"x": 5},
                call_id="call-conformance",
            )
            try:
                func_call.execute()
            except Exception:
                pass
