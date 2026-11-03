#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Demo script that exercises the PR #394 Gemini Live API OpenTelemetry instrumentation.

Run with uv from this directory:
    uv run --env-file=main.env run_gemini_live_telemetry.py

Connects to the live Gemini API endpoint (generativelanguage.googleapis.com)
using GEMINI_API_KEY (or GOOGLE_API_KEY) and sends a real 16kHz PCM audio
prompt and tool calls.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import struct
import uuid
from typing import Any

from google import genai
from google.genai import types as genai_types

from opentelemetry._logs import set_logger_provider
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, set_tracer_provider
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)


def _generate_pcm_sine_wave(
    freq_hz: float = 440.0,
    duration_s: float = 0.15,
    sample_rate: int = 16000,
) -> bytes:
    """Generate 16-bit little-endian mono PCM audio bytes."""
    num_samples = int(sample_rate * duration_s)
    samples = [
        int(8000 * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
        for i in range(num_samples)
    ]
    return struct.pack(f"<{num_samples}h", *samples)


def _setup_telemetry() -> tuple[
    InMemorySpanExporter,
    InMemoryLogRecordExporter,
    InMemoryMetricReader,
    TracerProvider,
    LoggerProvider,
    MeterProvider,
]:
    os.environ.setdefault(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    project_id = os.environ.get("PROJECT_ID")
    if project_id == "PROJECT_ID":
        project_id = None
    location = os.environ.get("LOCATION", "us-central1")

    resource_attrs: dict[str, str] = {
        "service.name": "gemini-live-semconv-pr394-prototype",
        "service.instance.id": str(uuid.uuid4()),
    }
    if project_id:
        resource_attrs["gcp.project_id"] = project_id
        resource_attrs["location"] = location

    resource = Resource.create(resource_attrs)

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    metric_reader = InMemoryMetricReader()
    metric_readers: list[Any] = [metric_reader]

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if project_id or otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        endpoint = otlp_endpoint or "telemetry.googleapis.com"
        exporter_kwargs: dict[str, Any] = {"endpoint": endpoint}

        if "telemetry.googleapis.com" in endpoint:
            import google.auth
            import google.auth.transport.grpc
            import google.auth.transport.requests
            import grpc
            from google.auth.transport.grpc import AuthMetadataPlugin

            credentials, _ = google.auth.default(quota_project_id=project_id)
            request = google.auth.transport.requests.Request()
            auth_metadata_plugin = AuthMetadataPlugin(
                credentials=credentials, request=request
            )
            channel_creds = grpc.composite_channel_credentials(
                grpc.ssl_channel_credentials(),
                grpc.metadata_call_credentials(auth_metadata_plugin),
            )
            exporter_kwargs["credentials"] = channel_creds

        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(**exporter_kwargs),
                max_export_batch_size=1,
            )
        )
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(**exporter_kwargs),
                max_export_batch_size=1,
            )
        )
        metric_readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(**exporter_kwargs),
                export_interval_millis=60000,
            )
        )

    meter_provider = MeterProvider(
        resource=resource, metric_readers=metric_readers
    )

    set_tracer_provider(tracer_provider)
    set_logger_provider(logger_provider)
    set_meter_provider(meter_provider)

    GoogleGenAiSdkInstrumentor().instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    return (
        span_exporter,
        log_exporter,
        metric_reader,
        tracer_provider,
        logger_provider,
        meter_provider,
    )


async def _run_live_session(
    client: genai.Client,
    model: str,
) -> None:
    config = genai_types.LiveConnectConfig(
        response_modalities=[genai_types.Modality.AUDIO],
        system_instruction=(
            "You are a helpful, ultra-concise voice assistant. "
            "CRITICAL RULE: Every spoken response MUST be a single word "
            "(for example: 'Hi.', 'Sixteen.', 'Warm.', 'Tokyo.', 'Hundred.'). "
            "Never speak more than 1 word in any turn."
        ),
        temperature=0.6,
        realtime_input_config=genai_types.RealtimeInputConfig(
            automatic_activity_detection=genai_types.AutomaticActivityDetection(
                disabled=True
            )
        ),
        input_audio_transcription=genai_types.AudioTranscriptionConfig(),
        output_audio_transcription=genai_types.AudioTranscriptionConfig(),
        tools=[
            genai_types.Tool(
                function_declarations=[
                    genai_types.FunctionDeclaration(
                        name="get_weather",
                        description="Get current weather for a city.",
                        parameters=genai_types.Schema(
                            type=genai_types.Type.OBJECT,
                            properties={
                                "city": genai_types.Schema(
                                    type=genai_types.Type.STRING,
                                    description="City name",
                                )
                            },
                            required=["city"],
                        ),
                    ),
                    genai_types.FunctionDeclaration(
                        name="get_stock_price",
                        description="Get current stock price for a ticker symbol.",
                        parameters=genai_types.Schema(
                            type=genai_types.Type.OBJECT,
                            properties={
                                "ticker": genai_types.Schema(
                                    type=genai_types.Type.STRING,
                                    description="Stock ticker symbol",
                                )
                            },
                            required=["ticker"],
                        ),
                    ),
                ]
            )
        ],
    )

    pcm_chunk = _generate_pcm_sine_wave(
        freq_hz=440.0, duration_s=0.1, sample_rate=16000
    )

    turns = [
        "Hi! Say hello in one word.",
        "What is the weather in Seattle? Use get_weather.",
        "And what is the weather in Tokyo? Use get_weather.",
        "Based on those two temperatures, which city is warmer?",
        "Is the stock price of GOOG above 100? Use get_stock_price.",
    ]

    async with client.aio.live.connect(model=model, config=config) as session:
        for idx, prompt in enumerate(turns, 1):
            await session.send_realtime_input(
                activity_start=genai_types.ActivityStart()
            )
            await session.send_realtime_input(
                audio=genai_types.Blob(
                    data=pcm_chunk, mime_type="audio/pcm;rate=16000"
                )
            )
            await session.send_realtime_input(text=prompt)
            await session.send_realtime_input(
                activity_end=genai_types.ActivityEnd()
            )

            transcript_parts: list[str] = []
            while True:
                msg = await session._receive()
                if msg.tool_call and msg.tool_call.function_calls:
                    for fc in msg.tool_call.function_calls:
                        args_dict = dict(fc.args or {})
                        print(f"[Turn {idx} Tool Call] {fc.name}({args_dict})")
                        if fc.name == "get_weather":
                            city = str(args_dict.get("city", "Seattle"))
                            is_tokyo = "tokyo" in city.lower()
                            tool_resp: dict[str, Any] = {
                                "city": city,
                                "temp_c": 24 if is_tokyo else 16,
                                "condition": "sunny"
                                if is_tokyo
                                else "partly cloudy",
                            }
                        else:
                            tool_resp = {
                                "ticker": str(
                                    args_dict.get("ticker", "GOOG")
                                ),
                                "price_usd": 192.45,
                            }
                        await session.send_tool_response(
                            function_responses=[
                                genai_types.FunctionResponse(
                                    id=fc.id,
                                    name=fc.name,
                                    response=tool_resp,
                                )
                            ]
                        )
                if (
                    msg.server_content
                    and msg.server_content.output_transcription
                    and msg.server_content.output_transcription.text
                ):
                    transcript_parts.append(
                        msg.server_content.output_transcription.text
                    )
                if (
                    msg.server_content
                    and msg.server_content.turn_complete is True
                ):
                    print(
                        f"[Turn {idx}] User: {prompt!r} -> "
                        f"Model: {''.join(transcript_parts).strip()!r}"
                    )
                    break


def _format_telemetry_report(
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    metric_reader: InMemoryMetricReader,
) -> dict[str, Any]:
    spans_out = []
    for s in span_exporter.get_finished_spans():
        attrs = dict(s.attributes or {})
        for json_key in (
            "gen_ai.input.messages",
            "gen_ai.output.messages",
            "gen_ai.system_instructions",
            "gen_ai.tool.definitions",
        ):
            if json_key in attrs and isinstance(attrs[json_key], str):
                try:
                    attrs[json_key] = json.loads(attrs[json_key])
                except ValueError:
                    pass
        spans_out.append(
            {
                "name": s.name,
                "kind": s.kind.name
                if isinstance(s.kind, SpanKind)
                else str(s.kind),
                "trace_id": f"{s.context.trace_id:032x}",
                "span_id": f"{s.context.span_id:016x}",
                "parent_span_id": f"{s.parent.span_id:016x}"
                if s.parent
                else None,
                "status": s.status.status_code.name,
                "attributes": attrs,
            }
        )

    events_out = []
    for l in log_exporter.get_finished_logs():
        rec = l.log_record
        events_out.append(
            {
                "event_name": rec.event_name,
                "trace_id": f"{rec.trace_id:032x}" if rec.trace_id else None,
                "span_id": f"{rec.span_id:016x}" if rec.span_id else None,
                "severity": rec.severity_number.name
                if rec.severity_number
                else None,
                "attributes": dict(rec.attributes or {}),
            }
        )

    metrics_out = []
    metrics_data = metric_reader.get_metrics_data()
    if metrics_data:
        for rm in metrics_data.resource_metrics:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    points = []
                    for dp in m.data.data_points:
                        points.append(
                            {
                                "attributes": dict(dp.attributes or {}),
                                "count": getattr(dp, "count", None),
                                "sum": getattr(dp, "sum", None),
                            }
                        )
                    metrics_out.append(
                        {
                            "name": m.name,
                            "unit": m.unit,
                            "data_points": points,
                        }
                    )

    return {
        "spans": spans_out,
        "events": events_out,
        "metrics": metrics_out,
    }


async def async_main(args: argparse.Namespace) -> None:
    (
        span_exporter,
        log_exporter,
        metric_reader,
        tracer_provider,
        logger_provider,
        meter_provider,
    ) = _setup_telemetry()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY must be set in the environment."
        )
    tracer = tracer_provider.get_tracer(__name__)

    try:
        client = genai.Client(api_key=api_key)
        with tracer.start_as_current_span("gemini_live_session") as root_span:
            print(
                f"[Active Context] trace_id={root_span.get_span_context().trace_id:032x} "
                f"root_span_id={root_span.get_span_context().span_id:016x}"
            )
            await _run_live_session(client, model=args.model)

        report = _format_telemetry_report(
            span_exporter, log_exporter, metric_reader
        )
    finally:
        tracer_provider.shutdown()
        logger_provider.shutdown()
        meter_provider.shutdown()

    formatted = json.dumps(
        report,
        indent=2,
        default=lambda o: base64.b64encode(o).decode("ascii")
        if isinstance(o, (bytes, bytearray))
        else str(o),
    )
    print(formatted)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(formatted + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini Live API with PR #394 OpenTelemetry GenAI instrumentation."
    )
    parser.add_argument(
        "--model",
        default="models/gemini-2.5-flash-native-audio-preview-12-2025",
        help="Model name to use when connecting to the live Gemini API.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional file path to write the captured spans, events, and metrics JSON.",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
