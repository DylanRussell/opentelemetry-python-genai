Suppressing Inference Events Programmatically
=============================================

This example demonstrates how to suppress ``gen_ai.client.inference.operation.details``
log events programmatically in Python by wrapping an OpenTelemetry
``LogRecordProcessor``.

Overview
--------

When capturing content with ``EVENT_ONLY`` or ``SPAN_AND_EVENT``,
the OpenTelemetry GenAI instrumentation emits a
``gen_ai.client.inference.operation.details`` log event containing input/output
messages and metadata.

If you want to collect spans and traces while selectively dropping the inference
event without disabling logging globally, you can wrap your log record processor
with a custom processor:

.. code-block:: python

    from typing import Any

    from opentelemetry.sdk._logs import LogRecordProcessor


    class DropGenAiEventsProcessor(LogRecordProcessor):
        def __init__(self, processor: LogRecordProcessor) -> None:
            self._processor = processor

        def enabled(
            self, *, event_name: str | None = None, **kwargs: Any
        ) -> bool:
            if event_name == "gen_ai.client.inference.operation.details":
                return False
            if hasattr(self._processor, "enabled"):
                return self._processor.enabled(event_name=event_name, **kwargs)
            return True

        def on_emit(self, log_data: Any) -> None:
            record = getattr(log_data, "log_record", log_data)
            if (
                getattr(record, "event_name", None)
                == "gen_ai.client.inference.operation.details"
            ):
                return
            self._processor.on_emit(log_data)

        def shutdown(self) -> None:
            self._processor.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return self._processor.force_flush(timeout_millis)

How It Works
------------

- **Early filtering with** ``enabled()``: When supported by the SDK, the logger
  checks ``enabled(event_name=...)`` before constructing and emitting the event.
  Returning ``False`` prevents unnecessary payload building.
- **Filtering on** ``on_emit()``: If an event record is emitted, ``on_emit()``
  inspects the wrapped record's ``event_name`` attribute and drops the record
  before forwarding to the underlying batch processor or exporter.

Setup
-----

Update `.env <.env>`_ with your ``OPENAI_API_KEY`` (and ensure an OTLP-compatible
endpoint is listening on http://localhost:4317).

Run
---

Run the example using ``uv``:

.. code-block:: sh

    uv run --env-file .env python main.py

The OpenAI chat completion runs, span data is exported, and the inference event
is suppressed by ``DropGenAiEventsProcessor``.
