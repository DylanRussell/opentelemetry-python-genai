# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Agno instrumentation tests."""
# pylint: disable=redefined-outer-name

from __future__ import annotations

import pytest

pytest_plugins = ["opentelemetry.test_util_genai.fixtures"]


@pytest.fixture
def instrument_agno(
    tracer_provider, logger_provider, meter_provider
):
    """Fixture to instrument Agno with test providers."""
    from opentelemetry.instrumentation.genai.agno import (  # noqa: PLC0415
        AgnoInstrumentor,
    )

    instrumentor = AgnoInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture
def uninstrument_agno():
    """Fixture to ensure Agno is uninstrumented after test."""
    yield
    from opentelemetry.instrumentation.genai.agno import (  # noqa: PLC0415
        AgnoInstrumentor,
    )

    AgnoInstrumentor().uninstrument()
