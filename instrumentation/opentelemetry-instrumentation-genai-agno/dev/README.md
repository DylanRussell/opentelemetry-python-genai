# Reproducing Duplicate Nested GenAI Inference Telemetry ([semantic-conventions-genai#682](https://github.com/open-telemetry/semantic-conventions-genai/issues/682))

This folder contains a runnable reproduction of [open-telemetry/semantic-conventions-genai#682](https://github.com/open-telemetry/semantic-conventions-genai/issues/682) using real dependencies (`agno` + `google-genai`) when both the outer agent/framework instrumentation (`opentelemetry-instrumentation-genai-agno`) and the inner provider SDK instrumentation (`opentelemetry-instrumentation-google-genai`) are enabled in the same process **without** nested operation suppression.

---

## What Changed on This Branch (`repro-issue-682-duplicate-inference-spans`)

To demonstrate the problem that nested client invocation suppression solves:
1. **`util/opentelemetry-util-genai/src/opentelemetry/util/genai/handler.py`**: Disabled `Suppressed*Invocation` selection in `TelemetryHandler.inference()`, `embedding()`, `retrieval()`, and `fetch_response()` so every instrumentation layer always creates a full invocation.
2. **`util/opentelemetry-util-genai/src/opentelemetry/util/genai/_invocation.py`**: Disabled attaching `_ContextData` (`CLIENT_*_CONTEXT_KEY`) to the OpenTelemetry `Context`, preventing inner instrumentations from backfilling attributes onto the outer invocation.

---

## Prerequisites

1. **Python & `uv`**: Python `>=3.12` and [`uv`](https://docs.astral.sh/uv/) installed.
2. **Google Cloud Authentication (ADC)**:
   Authenticate with Application Default Credentials for a GCP project that has the **Vertex AI API** and **Cloud Telemetry API** (`telemetry.googleapis.com`) enabled:
   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project ai-observability-demo-project
   ```

---

## How to Run

1. Navigate to this directory:
   ```bash
   cd instrumentation/opentelemetry-instrumentation-genai-agno/dev
   ```

2. Review or update `main.env` with your target GCP project, region, model, and prompt:
   ```dotenv
   OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="SPAN_AND_EVENT"
   PROJECT_ID="ai-observability-demo-project"
   LOCATION="us-central1"
   MODEL="gemini-2.5-flash"
   PROMPT="What is 123 + 456? Use the add tool to compute."
   ```

3. Install dependencies from the branch into an isolated `.venv`:
   ```bash
   uv sync
   ```

4. Run the reproduction script:
   ```bash
   uv run --env-file=main.env call_instrumentation_lib_and_save_telemetry.py
   ```

---

## What the Script Does

1. **Configures OpenTelemetry SDK Exporters**:
   - **GCP Telemetry API (`https://telemetry.googleapis.com:443`)**: Exports spans (`OTLPSpanExporter`), logs/events (`OTLPLogExporter`), and metrics (`OTLPMetricExporter`) over gRPC using Google Cloud Application Default Credentials.
   - **Local `.textproto` Snapshots**: Simultaneously records all emitted telemetry in memory and writes:
     - `span_agno_google_genai.textproto`
     - `log_agno_google_genai.textproto`
     - `metric_agno_google_genai.textproto`
2. **Enables Layered Instrumentations**:
   - `GoogleGenAiSdkInstrumentor().instrument()` (inner Google GenAI SDK layer)
   - `AgnoInstrumentor().instrument()` (outer Agno framework layer)
3. **Executes an Agent Workflow**:
   - Runs an Agno `Agent` configured with `Gemini(id="gemini-2.5-flash", vertexai=True)` and an `add(a, b)` tool.
   - The agent makes **2 underlying LLM calls** (first to decide to call `add(123, 456)`, second to synthesize the final answer `579`).

---

## Observed Issues Without Suppression & Context Propagation

### 1. Duplicate Nested `CLIENT` Inference Spans (6 spans emitted instead of 4)
Each logical model call produces two nested `CLIENT` inference spans:
```text
invoke_agent google-genai-agent (INTERNAL, scope=opentelemetry.instrumentation.genai.agno)
├── chat gemini-2.5-flash (CLIENT, scope=opentelemetry.instrumentation.genai.agno)
│   └── generate_content gemini-2.5-flash (CLIENT, scope=opentelemetry.instrumentation.google_genai)  <-- DUPLICATE
├── execute_tool add (INTERNAL, scope=opentelemetry.instrumentation.genai.agno)
└── chat gemini-2.5-flash (CLIENT, scope=opentelemetry.instrumentation.genai.agno)
    └── generate_content gemini-2.5-flash (CLIENT, scope=opentelemetry.instrumentation.google_genai)  <-- DUPLICATE
```

### 2. Split & Inconsistent Attributes Across Layers
Neither span has the complete, accurate picture of the operation:
- **Outer Agno span (`chat gemini-2.5-flash`)**:
  - Missing `server.address` (`None` instead of `us-central1-aiplatform.googleapis.com`), `gen_ai.response.id`, `gen_ai.response.model`, and modality token breakdowns (`gen_ai.usage.text.input_tokens`, `gen_ai.usage.text.output_tokens`).
- **Inner Google GenAI SDK span (`generate_content gemini-2.5-flash`)**:
  - Captures `server.address = "us-central1-aiplatform.googleapis.com"`, `gen_ai.response.id`, `gen_ai.response.model`, and modality token breakdowns.

### 3. Duplicate `gen_ai.client.inference.operation.details` Events in Cloud Logging
4 event log records are emitted instead of 2 (one from `opentelemetry.instrumentation.genai.agno` and one from `opentelemetry.instrumentation.google_genai` for each model call). You can inspect them in Cloud Logging using the Trace ID printed by the script:
```bash
gcloud logging read 'trace="projects/ai-observability-demo-project/traces/<TRACE_ID>"' \
  --project=ai-observability-demo-project
```

### 4. Double-Counted Metrics in Cloud Monitoring
Both instrumentation scopes record `gen_ai.client.operation.duration` (`count: 2` each = 4 operations recorded for 2 calls) and `gen_ai.client.token.usage` (`sum: 126` input tokens recorded twice = 252 input tokens).
