# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

from agno.agent import Agent
from agno.models.google import Gemini
import google.auth
from google.auth.transport.grpc import AuthMetadataPlugin
import google.auth.transport.requests
from google.protobuf import text_format
import grpc

from opentelemetry import _logs as logs
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.common._internal import (
    _log_encoder,
    metrics_encoder,
    trace_encoder,
)
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
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
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
)

PROJECT_ID = os.environ.get("PROJECT_ID", "ai-observability-demo-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL = os.environ.get("MODEL", "gemini-2.5-flash")
PROMPT = os.environ.get(
    "PROMPT", "What is 123 + 456? Use the add tool to compute."
)
GCP_OTLP_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT", "https://telemetry.googleapis.com:443"
)

credentials, _ = google.auth.default()
credentials = credentials.with_quota_project(PROJECT_ID)
request = google.auth.transport.requests.Request()
auth_metadata_plugin = AuthMetadataPlugin(
    credentials=credentials, request=request
)
channel_creds = grpc.composite_channel_credentials(
    grpc.ssl_channel_credentials(),
    grpc.metadata_call_credentials(auth_metadata_plugin),
)

resource = Resource.create(
    attributes={
        SERVICE_NAME: "repro-issue-682-duplicate-inference",
        "gcp.project_id": PROJECT_ID,
        "cloud.region": LOCATION,
    }
)

in_memory_log_exporter = InMemoryLogRecordExporter()
in_memory_span_exporter = InMemorySpanExporter()
in_memory_metric_reader = InMemoryMetricReader()

trace_provider = TracerProvider(resource=resource)
trace_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
trace_provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            credentials=channel_creds,
            endpoint=GCP_OTLP_ENDPOINT,
        )
    )
)
trace.set_tracer_provider(trace_provider)

logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(
    SimpleLogRecordProcessor(in_memory_log_exporter)
)
logger_provider.add_log_record_processor(
    BatchLogRecordProcessor(
        OTLPLogExporter(
            credentials=channel_creds,
            endpoint=GCP_OTLP_ENDPOINT,
        )
    )
)
logs.set_logger_provider(logger_provider)

otlp_metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(
        credentials=channel_creds,
        endpoint=GCP_OTLP_ENDPOINT,
    ),
    export_interval_millis=60000,
)
meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[in_memory_metric_reader, otlp_metric_reader],
)
metrics.set_meter_provider(meter_provider)


def write_logs_to_file(file_name: str) -> None:
    log_records = in_memory_log_exporter.get_finished_logs()
    log_proto = _log_encoder.encode_logs(log_records)
    with open(
        os.path.join(os.getcwd(), file_name + ".textproto"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text_format.MessageToString(log_proto))


def write_spans_to_file(file_name: str) -> None:
    spans = in_memory_span_exporter.get_finished_spans()
    span_proto = trace_encoder.encode_spans(spans)
    with open(
        os.path.join(os.getcwd(), file_name + ".textproto"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text_format.MessageToString(span_proto))


def write_metrics_to_file(file_name: str) -> None:
    metrics_data = in_memory_metric_reader.get_metrics_data()
    if metrics_data is None:
        return
    metric_proto = metrics_encoder.encode_metrics(metrics_data)
    with open(
        os.path.join(os.getcwd(), file_name + ".textproto"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text_format.MessageToString(metric_proto))


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def print_repro_summary() -> None:
    spans = in_memory_span_exporter.get_finished_spans()
    log_records = in_memory_log_exporter.get_finished_logs()
    print("\n=== REPRO SUMMARY (Issue #682: Duplicate Inference Telemetry) ===")
    if spans:
        trace_id_hex = format(spans[0].context.trace_id, "032x")
        print(f"GCP Project: {PROJECT_ID}")
        print(f"Trace ID:    {trace_id_hex}")
        print(
            f"Cloud Trace: https://console.cloud.google.com/traces/explorer;traceId={trace_id_hex}?project={PROJECT_ID}"
        )
    print(f"\nSpans emitted ({len(spans)} total):")
    for s in spans:
        span_id = format(s.context.span_id, "016x")
        parent_id = format(s.parent.span_id, "016x") if s.parent else "root"
        scope = s.instrumentation_scope.name if s.instrumentation_scope else ""
        attrs = dict(s.attributes or {})
        print(
            f"  - [{span_id} <- {parent_id}] {s.name!r} (scope={scope}, "
            f"op={attrs.get('gen_ai.operation.name')}, "
            f"provider={attrs.get('gen_ai.provider.name')}, "
            f"server.address={attrs.get('server.address')}, "
            f"conversation.id={attrs.get('gen_ai.conversation.id')}, "
            f"in_tokens={attrs.get('gen_ai.usage.input_tokens')}, "
            f"out_tokens={attrs.get('gen_ai.usage.output_tokens')})"
        )
    print(f"\nLog/Event records emitted ({len(log_records)} total):")
    for lr in log_records:
        scope = (
            lr.instrumentation_scope.name if lr.instrumentation_scope else ""
        )
        span_id = format(lr.log_record.span_id, "016x")
        attrs = dict(lr.log_record.attributes or {})
        print(
            f"  - [span={span_id}] event={lr.log_record.event_name!r} "
            f"(scope={scope}, op={attrs.get('gen_ai.operation.name')}, "
            f"provider={attrs.get('gen_ai.provider.name')}, "
            f"server.address={attrs.get('server.address')})"
        )


def main() -> None:
    GoogleGenAiSdkInstrumentor().instrument()
    AgnoInstrumentor().instrument()

    agent = Agent(
        model=Gemini(
            id=MODEL,
            vertexai=True,
            project_id=PROJECT_ID,
            location=LOCATION,
            temperature=0.2,
        ),
        tools=[add],
        name="google-genai-agent",
        session_id="repro-session-682",
    )

    response = agent.run(PROMPT)

    write_spans_to_file("span_agno_google_genai")
    write_logs_to_file("log_agno_google_genai")
    write_metrics_to_file("metric_agno_google_genai")
    print_repro_summary()

    trace_provider.shutdown()
    logger_provider.shutdown()
    meter_provider.shutdown()
    print("\nExported spans, logs, and metrics to GCP Telemetry API!")
    print("Agent response:", response.content)


if __name__ == "__main__":
    main()
