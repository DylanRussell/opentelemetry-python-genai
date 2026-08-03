# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: basic workflow run for Agno."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("agno.workflow.workflow")

from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.workflow.workflow import Workflow
from tests.mock_model import MockModel

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class WorkflowScenario(Scenario):
    expected_spans = {"invoke_workflow": 1, "invoke_agent": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

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
            agent = Agent(
                name="workflow-agent",
                model=MockModel(id="mock-model"),
                session_id="session-workflow",
            )
            workflow = Workflow(
                name="test-conformance-workflow",
                agent=agent,
                session_id="session-workflow",
            )
            mock_output = ModelResponse(content="Workflow Conformance Hello!")
            with (
                patch.object(Workflow, "run", wraps=workflow.run),
                patch.object(Agent, "run", wraps=agent.run),
                patch(
                    "agno.models.base.Model.response", return_value=mock_output
                ),
            ):
                workflow.run("hello workflow conformance")
