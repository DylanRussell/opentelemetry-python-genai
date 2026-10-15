"""Basic ADK agent with OpenTelemetry tracing enabled for Google Cloud Trace.

Reference: https://adk.dev/observability/traces/
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.adk.telemetry.google_cloud import (
    get_gcp_exporters,
    get_gcp_resource,
)
from google.adk.telemetry.setup import maybe_set_otel_providers
from google.genai import types

from opentelemetry import trace
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)


def setup_telemetry() -> None:
    """Configures OpenTelemetry tracing to export spans to Google Cloud Trace."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    # Obtain GCP exporters as documented in https://adk.dev/observability/traces/
    gcp_exporters = get_gcp_exporters(
        enable_cloud_tracing=True,
    )

    # telemetry.googleapis.com requires the Resource to have 'gcp.project_id'.
    # get_gcp_resource sets gcp.project_id, cloud.account.id, and detects platform metadata.
    otel_resource = get_gcp_resource(project_id)

    # Register OTel providers with ADK
    maybe_set_otel_providers([gcp_exporters], otel_resource=otel_resource)

    # Instrument the Google GenAI SDK
    GoogleGenAiSdkInstrumentor().instrument()


async def run_agent(prompt: str) -> None:
    """Initializes and runs a basic ADK agent with the given user prompt."""
    model_name = os.getenv("ADK_MODEL", "gemini-2.5-flash")

    # Define a basic ADK agent
    agent = Agent(
        name="basic_adk_agent",
        model=model_name,
        instruction="You are a helpful assistant. Provide clear and concise answers.",
    )

    runner = InMemoryRunner(agent=agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="user-1",
    )

    user_message = types.Content(
        parts=[types.Part(text=prompt)],
        role="user",
    )

    print(f"Agent '{agent.name}' running with model '{model_name}'...")
    print(f"Prompt: {prompt}")
    print("Response:\n" + "-" * 40)

    # Run the agent asynchronously and stream response events
    async for event in runner.run_async(
        user_id="user-1",
        session_id=session.id,
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text, end="", flush=True)

    print("\n" + "-" * 40)
    print("Agent run complete.")


async def _async_main() -> None:
    # Load environment variables from .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    # Initialize GCP Cloud Trace telemetry
    setup_telemetry()

    prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.getenv(
            "ADK_PROMPT", "Explain quantum computing in two sentences."
        )
    )

    try:
        await run_agent(prompt)
    finally:
        # Flush pending trace spans before process exits
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            print("Flushing telemetry spans to Google Cloud Trace...")
            tracer_provider.force_flush(timeout_millis=10000)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
