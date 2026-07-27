# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Agent instrumentation."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from agno.agent import Agent
from agno.run.agent import RunOutput


def test_agent_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.run emits an invoke_agent span."""
    agent = Agent(name="test-sync-agent")
    mock_output = RunOutput(
        agent_id="test-sync-agent",
        agent_name="test-sync-agent",
        content="Hello back!",
        session_id="session-123",
    )

    with patch.object(Agent, "run", wraps=agent.run), patch(
        "agno.models.base.Model.response", return_value=mock_output
    ):
        try:
            res = agent.run("hello world")
            assert res is not None
        except Exception:
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-sync-agent"
    assert span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    assert span.attributes.get("gen_ai.agent.name") == "test-sync-agent"


def test_agent_arun_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.arun emits an invoke_agent span."""
    agent = Agent(name="test-async-agent")
    mock_output = RunOutput(
        agent_id="test-async-agent",
        agent_name="test-async-agent",
        content="Async hello back!",
        session_id="session-456",
    )

    async def _run_async() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            try:
                res = await agent.arun("hello async world")
                assert res is not None
            except Exception:
                pass

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-agent"
    assert span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    assert span.attributes.get("gen_ai.agent.name") == "test-async-agent"
