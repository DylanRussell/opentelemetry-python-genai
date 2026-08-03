# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Agent instrumentation."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from agno.agent import Agent
from agno.models.message import Message
from agno.models.response import ModelResponse
from agno.team import Team
from agno.tools.function import Function, FunctionCall
from tests.mock_model import MockModel

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def test_agent_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.run emits an invoke_agent span."""
    agent = Agent(name="test-sync-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Hello back!")

    with (
        patch.object(Agent, "run", wraps=agent.run),
        patch("agno.models.base.Model.response", return_value=mock_output),
    ):
        res = agent.run("hello world")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-sync-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-sync-agent"
    )


def test_agent_arun_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.arun emits an invoke_agent span."""
    agent = Agent(name="test-async-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Async hello back!")

    async def _run_async() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            res = await agent.arun("hello async world")
            assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-async-agent"
    )


def test_tool_call_execute_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that FunctionCall.execute emits an execute_tool span."""

    def sample_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    func = Function.from_callable(sample_tool)
    func_call = FunctionCall(
        function=func,
        arguments={"x": 5},
        call_id="call-123",
    )
    res = func_call.execute()
    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool sample_tool"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "execute_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_NAME) == "sample_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID) == "call-123"
    )


def test_tool_call_aexecute_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that FunctionCall.aexecute emits an execute_tool span."""

    def sample_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    func = Function.from_callable(sample_tool)
    func_call = FunctionCall(
        function=func,
        arguments={"x": 5},
        call_id="call-456",
    )

    async def _run_async() -> None:
        res = await func_call.aexecute()
        assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool sample_tool"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "execute_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_NAME) == "sample_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID) == "call-456"
    )


def test_team_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Team.run emits an invoke_agent span."""
    member = Agent(name="member-agent", model=MockModel(id="mock-model"))
    team = Team(
        name="test-sync-team",
        members=[member],
        model=MockModel(id="mock-model"),
    )
    mock_output = ModelResponse(content="Hello back from team!")

    with patch("agno.models.base.Model.response", return_value=mock_output):
        res = team.run("hello team world")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-sync-team"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-sync-team"
    )


def test_team_arun_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Team.arun emits an invoke_agent span."""
    member = Agent(name="member-agent", model=MockModel(id="mock-model"))
    team = Team(
        name="test-async-team",
        members=[member],
        model=MockModel(id="mock-model"),
    )
    mock_output = ModelResponse(content="Async hello back from team!")

    async def _run_async() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            res = await team.arun("hello async team world")
            assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-team"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-async-team"
    )


def test_model_response_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Model.response emits a chat span."""
    model = MockModel(id="test-mock-model")
    mock_output = ModelResponse(content="Hello from model!", role="assistant")

    with patch.object(MockModel, "invoke", return_value=mock_output):
        res = model.response(
            messages=[Message(role="user", content="hello model")]
        )
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat test-mock-model"
    assert span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "test-mock-model"
    )


def test_model_aresponse_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Model.aresponse emits a chat span."""
    model = MockModel(id="test-mock-model-async")
    mock_output = ModelResponse(
        content="Async hello from model!", role="assistant"
    )

    async def _run_async() -> None:
        with patch.object(MockModel, "ainvoke", return_value=mock_output):
            res = await model.aresponse(
                messages=[Message(role="user", content="hello async model")]
            )
            assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat test-mock-model-async"
    assert span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "test-mock-model-async"
    )


def test_workflow_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.run emits an invoke_workflow span."""
    pytest.importorskip("fastapi")
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow  # noqa: PLC0415

    workflow = Workflow(name="test-workflow")
    with patch.object(Workflow, "run", wraps=workflow.run):
        try:
            workflow.run("test input")
        except Exception:
            pass

    spans = span_exporter.get_finished_spans()
    assert any(
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
        for span in spans
    )


@pytest.mark.asyncio
async def test_workflow_arun_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.arun emits an invoke_workflow span."""
    pytest.importorskip("fastapi")
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow  # noqa: PLC0415

    workflow = Workflow(name="test-workflow-async")
    with patch.object(Workflow, "arun", wraps=workflow.arun):
        try:
            await workflow.arun("test input")
        except Exception:
            pass

    spans = span_exporter.get_finished_spans()
    assert any(
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
        for span in spans
    )


def test_step_execute_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Step.execute emits an invoke_workflow span."""
    pytest.importorskip("fastapi")
    pytest.importorskip("agno.workflow.step")
    from agno.workflow.step import Step  # noqa: PLC0415

    step = Step(name="test-step", executor=lambda step_input: "test output")
    with patch.object(Step, "execute", wraps=step.execute):
        try:
            step.execute("test input")
        except Exception:
            pass

    spans = span_exporter.get_finished_spans()
    assert any(
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
        for span in spans
    )


@pytest.mark.asyncio
async def test_step_aexecute_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Step.aexecute emits an invoke_workflow span."""
    pytest.importorskip("fastapi")
    pytest.importorskip("agno.workflow.step")
    from agno.workflow.step import Step  # noqa: PLC0415

    step = Step(
        name="test-step-async", executor=lambda step_input: "test output"
    )
    with patch.object(Step, "aexecute", wraps=step.aexecute):
        try:
            await step.aexecute("test input")
        except Exception:
            pass

    spans = span_exporter.get_finished_spans()
    assert any(
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
        for span in spans
    )
