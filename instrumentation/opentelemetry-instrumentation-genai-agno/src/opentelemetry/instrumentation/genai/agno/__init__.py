# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Agno Instrumentation
==================================

Instrumentation for `Agno <https://github.com/agno-agi/agno>`_.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor

    # Enable instrumentation
    AgnoInstrumentor().instrument()

Configuration
-------------

Message content capture can be enabled by setting the environment variable:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``

API
---
"""

from __future__ import annotations

from typing import Any, Collection

from opentelemetry.instrumentation.genai.agno.package import (
    _instruments,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

__all__ = ["AgnoInstrumentor"]


class AgnoInstrumentor(BaseInstrumentor):
    """An instrumentor for Agno."""

    def __init__(self) -> None:
        super().__init__()
        self._tracer = None
        self._logger = None
        self._meter = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Agno instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
        """
        # Patching will be added in a follow-up PR

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Agno instrumentation.

        This removes all patches applied during instrumentation.
        """
        # Unpatching will be added in a follow-up PR
