DSPy with Gemini, MLflow Tracing, and GCP OTLP Export
=====================================================

This example demonstrates how to:

1. Configure **DSPy** to use a Google Gemini model.
2. Enable **MLflow Tracing** for DSPy using ``mlflow.dspy.autolog()`` (see `DSPy Observability Tutorial <https://dspy.ai/tutorials/observability/>`_).
3. Configure MLflow to emit traces following the **OpenTelemetry GenAI Semantic Conventions** via ``MLFLOW_ENABLE_OTEL_GENAI_SEMCONV="true"`` (see `MLflow OpenTelemetry GenAI Semantic Conventions <https://mlflow.org/docs/latest/genai/tracing/opentelemetry/genai-semconv/>`_).
4. Send the OpenTelemetry traces directly to the **Google Cloud (GCP) Cloud Trace OTLP HTTP endpoint** using ``AuthorizedSession`` credentials (see `GCP OTLP HTTP Example <https://github.com/GoogleCloudPlatform/opentelemetry-samples/blob/main/python/otlptrace/example_http.py>`_).

Prerequisites
-------------

1. Set your Gemini API key in your environment (or in a ``.env`` file):

   .. code-block:: bash

      export GEMINI_API_KEY="your-api-key"

2. Authenticate with Google Cloud so Application Default Credentials (ADC) are available for exporting traces to Cloud Trace:

   .. code-block:: bash

      gcloud auth application-default login

Installation
------------

Install the required dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Running the Example
-------------------

Execute the sample script:

.. code-block:: bash

   python main.py
