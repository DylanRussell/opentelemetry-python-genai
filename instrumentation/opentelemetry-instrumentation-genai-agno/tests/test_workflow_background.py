# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Workflow background execution instrumentation."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.run.workflow import RunStatus
from agno.workflow.workflow import Workflow
from tests.mock_model import MockModel

from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_CONVERSATION_ID,
    GEN_AI_OPERATION_NAME,
)
from opentelemetry.semconv._incubating.attributes.user_attributes import (
    USER_ID,
)
from opentelemetry.trace.status import StatusCode


async def _wait_for_background_tasks(timeout: float = 5.0) -> None:
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current]
    if pending:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=timeout,
        )


def test_workflow_background_execution_non_streaming(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Workflow.arun(background=True) traces the background task until completion."""

    async def async_step(step_input: str) -> str:
        await asyncio.sleep(0.05)
        return f"processed: {step_input}"

    workflow = Workflow(
        name="test-bg-workflow",
        steps=[async_step],
    )

    async def _test() -> None:
        placeholder = await workflow.arun("hello background", background=True)
        # Verify placeholder returned immediately is pending
        assert placeholder.status == RunStatus.pending

        # At this point, the span should NOT be finished yet
        assert len(span_exporter.get_finished_spans()) == 0

        # Wait for background task execution to complete
        await _wait_for_background_tasks()

        assert placeholder.status == RunStatus.completed

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-bg-workflow"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_workflow"
    assert span.status.status_code != StatusCode.ERROR


def test_workflow_background_execution_identity(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that background workflow captures user_id and session_id attributes."""

    async def quick_step(step_input: str) -> str:
        return "quick done"

    workflow = Workflow(
        name="test-bg-id-workflow",
        steps=[quick_step],
    )

    async def _test() -> None:
        placeholder = await workflow.arun(
            "bg input",
            background=True,
            user_id="bg-user-999",
            session_id="bg-sess-888",
        )
        assert placeholder.status == RunStatus.pending
        await _wait_for_background_tasks()
        assert placeholder.status == RunStatus.completed

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get(USER_ID) == "bg-user-999"
    assert span.attributes.get(GEN_AI_CONVERSATION_ID) == "bg-sess-888"


def test_workflow_background_execution_child_parentage(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that steps/agents executed inside background workflow are parented under the workflow span."""
    agent = Agent(name="inner-bg-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Agent output inside background step")

    async def agent_step(step_input: str) -> str:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            res = await agent.arun("step agent input")
            return str(getattr(res, "content", "ok"))

    workflow = Workflow(
        name="test-bg-parent-workflow",
        steps=[agent_step],
    )

    async def _test() -> None:
        placeholder = await workflow.arun("start parent test", background=True)
        assert placeholder.status == RunStatus.pending
        await _wait_for_background_tasks()
        assert placeholder.status == RunStatus.completed

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    # Find workflow span and agent span
    wf_spans = [
        s
        for s in spans
        if s.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_workflow"
    ]
    agent_spans = [
        s
        for s in spans
        if s.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_agent"
    ]
    assert len(wf_spans) == 1
    assert len(agent_spans) == 1

    wf_span = wf_spans[0]
    agent_span = agent_spans[0]

    assert agent_span.parent is not None
    assert agent_span.parent.span_id == wf_span.context.span_id


def test_workflow_background_execution_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error inside background workflow execution fails the span."""

    async def failing_custom_executor(execution_input: object) -> str:
        raise RuntimeError("boom in background")

    workflow = Workflow(
        name="test-bg-fail-workflow",
        steps=failing_custom_executor,
    )

    async def _test() -> None:
        placeholder = await workflow.arun("fail test", background=True)
        assert placeholder.status == RunStatus.pending
        await _wait_for_background_tasks()
        assert placeholder.status == RunStatus.error

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "RuntimeError"
