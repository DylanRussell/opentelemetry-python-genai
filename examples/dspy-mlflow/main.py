# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file

import os

from dotenv import load_dotenv
import dspy
import google.auth
from google.auth.transport.requests import AuthorizedSession
import mlflow
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv()


def setup_gcp_otel_tracing():
    """Configures OpenTelemetry to send traces to the GCP OTLP HTTP endpoint."""
    credentials, _ = google.auth.default()
    trace_provider = TracerProvider(
        resource=Resource.create(
            attributes={
                SERVICE_NAME: os.getenv(
                    "OTEL_SERVICE_NAME", "dspy-gemini-mlflow"
                )
            }
        )
    )
    processor = BatchSpanProcessor(
        OTLPSpanExporter(
            session=AuthorizedSession(credentials),
            endpoint=os.getenv(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                "https://telemetry.googleapis.com:443/v1/traces",
            ),
        )
    )
    trace_provider.add_span_processor(processor)
    trace.set_tracer_provider(trace_provider)
    return trace_provider


def setup_mlflow_tracing():
    """Configures MLflow to use OTel GenAI Semantic Conventions and trace DSPy."""
    # Enable OpenTelemetry GenAI Semantic Conventions in MLflow
    os.environ["MLFLOW_ENABLE_OTEL_GENAI_SEMCONV"] = "true"
    # Enable dual export to send traces to both MLflow tracking and the OTel exporter
    os.environ["MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT"] = "true"

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(
        os.getenv("MLFLOW_EXPERIMENT_NAME", "dspy-gemini-observability")
    )

    # Enable automatic tracing for DSPy
    mlflow.dspy.autolog()


class QuestionAnswer(dspy.Signature):
    """Answer questions with brief, accurate explanations."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="A detailed answer to the question.")


def main():
    # 1. Setup OpenTelemetry tracing to export to the GCP OTLP HTTP endpoint
    trace_provider = setup_gcp_otel_tracing()

    # 2. Setup MLflow tracing with OpenTelemetry GenAI Semantic Conventions
    setup_mlflow_tracing()

    # 3. Configure DSPy with a Google Gemini model
    model_name = os.getenv("MODEL", "gemini/gemini-2.0-flash")
    lm = dspy.LM(model_name)
    dspy.configure(lm=lm)

    # 4. Run DSPy ChainOfThought module
    qa_module = dspy.ChainOfThought(QuestionAnswer)
    prompt = os.getenv("PROMPT", "Why is the sky blue?")
    print(f"Running DSPy question answering with prompt: {prompt}")

    result = qa_module(question=prompt)
    print(f"\nAnswer:\n{result.answer}")

    # 5. Flush and shutdown the trace provider to ensure all spans are exported
    trace_provider.force_flush()
    trace_provider.shutdown()


if __name__ == "__main__":
    main()
