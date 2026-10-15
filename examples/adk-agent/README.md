# ADK Agent with Google Cloud Tracing

This example sets up a basic Google Agent Development Kit ([ADK](https://adk.dev/)) agent with OpenTelemetry tracing enabled and exported to **Google Cloud Trace** per the [ADK Traces documentation](https://adk.dev/observability/traces/).

## Prerequisites

1. Python `>=3.11`
2. [uv](https://docs.astral.sh/uv/) installed
3. A Google Cloud project with the **Cloud Trace API** enabled
4. Authentication credentials (either GCP Application Default Credentials or a Gemini API key)

## Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and configure the following variables:
   - `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID (where traces will be exported).
   - Authentication (choose one):
     - **Gemini Developer API**: Set `GEMINI_API_KEY`.
     - **Vertex AI**: Uncomment `GOOGLE_GENAI_USE_VERTEXAI=true` and `GOOGLE_CLOUD_LOCATION=us-central1`.
   - **GCP Trace Authentication**: Ensure you have authenticated to GCP:
     ```bash
     gcloud auth application-default login
     ```
     Or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`.

3. Install dependencies:
   ```bash
   uv sync
   ```

## Running the Agent

Run the agent script using `uv`:

```bash
uv run main.py
```

Or pass a custom prompt directly on the command line:

```bash
uv run main.py "What are three interesting facts about black holes?"
```

## Viewing Traces in Google Cloud Console

Once the script completes, visit Google Cloud Console:
1. Navigate to **Trace** > **Trace Explorer** in your GCP project.
2. Filter by service name `basic-adk-agent` (or whatever `OTEL_SERVICE_NAME` you set in `.env`).
3. You will see spans representing the agent's execution, including `invoke_agent`, `generate_content`, and LLM calls.
