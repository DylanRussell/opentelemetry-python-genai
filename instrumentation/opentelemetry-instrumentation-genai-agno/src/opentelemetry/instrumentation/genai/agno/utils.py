# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Agno instrumentation."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from agno.knowledge.document.base import Document

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiProviderNameValues,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    RetrievalDocument,
    ToolDefinition,
)


def format_retrieval_document(doc: Document) -> RetrievalDocument:
    """Format an Agno Document into a RetrievalDocument model."""
    score: float | None = None
    if doc.reranking_score is not None:
        try:
            score = float(doc.reranking_score)
        except (ValueError, TypeError):
            pass

    metadata: dict[str, Any] | None = None
    if doc.meta_data:
        metadata = dict(doc.meta_data)

    return RetrievalDocument(
        content=doc.content,
        id=str(doc.id) if doc.id is not None else None,
        score=score,
        metadata=metadata,
    )


@runtime_checkable
class _ModelDumpJson(Protocol):
    def model_dump_json(self) -> str: ...


@runtime_checkable
class _JsonDump(Protocol):
    def json(self) -> str: ...


@runtime_checkable
class _ModelDump(Protocol):
    def model_dump(self) -> Any: ...


@runtime_checkable
class _DictDump(Protocol):
    def dict(self) -> Any: ...


def _json_default(obj: object) -> object:
    if isinstance(obj, type):
        return str(obj)
    if isinstance(obj, _ModelDump):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, _DictDump):
        try:
            return obj.dict()
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.asdict(obj)
        except Exception:
            pass
    return str(obj)


def format_content(val: object) -> str:
    """Format content into a string, converting structured objects to JSON."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, type):
        return str(val)
    if isinstance(val, _ModelDumpJson):
        try:
            return str(val.model_dump_json())
        except Exception:
            pass
    if isinstance(val, _JsonDump):
        try:
            return str(val.json())
        except Exception:
            pass
    if isinstance(val, _ModelDump):
        try:
            return json.dumps(
                val.model_dump(), ensure_ascii=False, default=_json_default
            )
        except Exception:
            pass
    if isinstance(val, _DictDump):
        try:
            return json.dumps(
                val.dict(), ensure_ascii=False, default=_json_default
            )
        except Exception:
            pass
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        try:
            return json.dumps(
                dataclasses.asdict(val),
                ensure_ascii=False,
                default=_json_default,
            )
        except Exception:
            pass
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False, default=_json_default)
        except Exception:
            pass
    return str(cast(object, val))


def _get_property_value(obj: Any, property_name: str) -> Any:
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj).get(property_name)

    return getattr(obj, property_name, None)


def _extract_desc(tool: Any) -> str | None:
    desc = _get_property_value(tool, "description")
    if not desc:
        entrypoint = _get_property_value(tool, "entrypoint")
        if entrypoint:
            desc = _get_property_value(entrypoint, "__doc__")
    return str(desc).strip() if desc else None


def prepare_tool_definitions(
    tools: Iterable[Any] | str | None,
) -> list[ToolDefinition] | None:
    """Extract tool definitions from Agno Agent tools."""
    if not tools:
        return None

    raw_tools: list[Any]
    if isinstance(tools, str):
        try:
            parsed_str_val: Any = json.loads(tools)
            if isinstance(parsed_str_val, list):
                raw_tools = cast(list[Any], parsed_str_val)
            elif isinstance(parsed_str_val, dict):
                raw_tools = [cast(dict[str, Any], parsed_str_val)]
            else:
                return None
        except Exception:
            return None
    elif (
        isinstance(tools, (list, tuple))
        and tools
        and all(isinstance(c, str) and len(c) == 1 for c in tools)
    ):
        try:
            parsed_coerced: Any = json.loads("".join(tools))
            if isinstance(parsed_coerced, list):
                raw_tools = cast(list[Any], parsed_coerced)
            elif isinstance(parsed_coerced, dict):
                raw_tools = [cast(dict[str, Any], parsed_coerced)]
            else:
                return None
        except Exception:
            return None
    elif isinstance(tools, dict):
        raw_tools = [cast(dict[str, Any], tools)]
    else:
        try:
            raw_tools = list(tools)
        except TypeError:
            return None

    seen_names: set[str] = set()
    definitions: list[ToolDefinition] = []

    def _add_def(name: str, desc: str | None, params: Any) -> None:
        if not name or name in seen_names:
            return
        seen_names.add(name)
        definitions.append(
            FunctionToolDefinition(
                name=name,
                description=desc,
                parameters=params,
            )
        )

    for tool_item in raw_tools:
        tool: Any = tool_item
        if isinstance(tool, str):
            try:
                parsed_tool: Any = json.loads(tool)
                if isinstance(parsed_tool, dict):
                    tool = cast(dict[str, Any], parsed_tool)
                elif isinstance(parsed_tool, list):
                    sub_defs = prepare_tool_definitions(
                        cast(list[Any], parsed_tool)
                    )
                    if sub_defs:
                        for defn in sub_defs:
                            _add_def(
                                str(_get_property_value(defn, "name") or ""),
                                _get_property_value(defn, "description"),
                                _get_property_value(defn, "parameters"),
                            )
                    continue
                else:
                    continue
            except Exception:
                continue

        # Skip tool execution records (which have tool_call_id)
        if isinstance(tool, dict) and "tool_call_id" in cast(
            dict[str, Any], tool
        ):
            continue
        if (
            not isinstance(tool, dict)
            and getattr(cast(object, tool), "tool_call_id", None) is not None
        ):
            continue

        if isinstance(tool, dict):
            if (
                "type" in tool
                and _get_property_value(tool, "type") == "function"
                and isinstance(_get_property_value(tool, "function"), dict)
            ):
                func_dict = _get_property_value(tool, "function")
                _add_def(
                    str(_get_property_value(func_dict, "name") or ""),
                    str(_get_property_value(func_dict, "description"))
                    if _get_property_value(func_dict, "description")
                    is not None
                    else None,
                    _get_property_value(func_dict, "parameters"),
                )
            elif "name" in tool:
                _add_def(
                    str(_get_property_value(tool, "name") or ""),
                    str(_get_property_value(tool, "description"))
                    if _get_property_value(tool, "description") is not None
                    else None,
                    _get_property_value(tool, "parameters"),
                )
        elif hasattr(tool, "functions") or hasattr(tool, "get_functions"):
            try:
                funcs = None
                if hasattr(tool, "get_functions") and callable(
                    _get_property_value(tool, "get_functions")
                ):
                    funcs_fn = _get_property_value(tool, "get_functions")
                    if callable(funcs_fn):
                        funcs = funcs_fn()
                else:
                    funcs = _get_property_value(tool, "functions")
                if isinstance(funcs, dict):
                    sub_defs = prepare_tool_definitions(
                        list(cast(dict[str, Any], funcs).values())
                    )
                    if sub_defs:
                        for defn in sub_defs:
                            _add_def(
                                _get_property_value(defn, "name") or "",
                                _get_property_value(defn, "description"),
                                _get_property_value(defn, "parameters"),
                            )
            except Exception:
                pass
        elif hasattr(tool, "name") and hasattr(tool, "parameters"):
            name = _get_property_value(tool, "name") or ""
            desc = _extract_desc(tool)
            params = _get_property_value(tool, "parameters")
            _add_def(
                str(name),
                desc,
                params,
            )
        elif callable(tool):
            try:
                import agno.tools.function  # pylint: disable=import-outside-toplevel

                fn_cls = _get_property_value(agno.tools.function, "Function")
                func = fn_cls.from_callable(tool)
                name = _get_property_value(func, "name") or ""
                desc = _extract_desc(func)
                params = _get_property_value(func, "parameters")
                _add_def(
                    str(name),
                    desc,
                    params,
                )
            except Exception:
                name = _get_property_value(tool, "__name__") or str(tool)
                desc = _get_property_value(tool, "__doc__")
                _add_def(
                    str(name),
                    str(desc).strip() if desc is not None else None,
                    None,
                )

    return definitions or None


def extract_user_id(
    instance: Any = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    run_response: Any = None,
) -> str | None:
    """Extract user_id from call arguments, instance, or response."""
    if kwargs and (user_id := kwargs.get("user_id")) is not None:
        return str(user_id)
    if args and len(args) > 2 and args[2] is not None:
        return str(args[2])
    if instance:
        if (user_id := getattr(instance, "user_id", None)) is not None:
            return str(user_id)
        if (user := getattr(instance, "user", None)) is not None:
            return str(user)
    if run_response and (
        (user_id := getattr(run_response, "user_id", None)) is not None
    ):
        return str(user_id)
    return None


def extract_session_id(
    instance: Any = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    run_response: Any = None,
) -> str | None:
    """Extract session_id from call arguments, instance, or response."""
    if kwargs and (session_id := kwargs.get("session_id")) is not None:
        return str(session_id)
    if args and len(args) > 4 and args[4] is not None:
        return str(args[4])
    if instance and (
        (session_id := getattr(instance, "session_id", None)) is not None
    ):
        return str(session_id)
    if run_response and (
        (session_id := getattr(run_response, "session_id", None)) is not None
    ):
        return str(session_id)
    return None


def set_invocation_user_id(
    invocation: Any,
    instance: Any = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    run_response: Any = None,
) -> None:
    """Extract and set user.id on the invocation attributes if present."""
    from opentelemetry.semconv._incubating.attributes.user_attributes import (  # pylint: disable=import-outside-toplevel
        USER_ID,
    )

    user_id = extract_user_id(instance, args, kwargs, run_response)
    if user_id is not None:
        invocation.attributes[USER_ID] = user_id


_UNKNOWN_PROVIDER = "unknown"

# Mapping of raw provider identifiers to GenAI semantic conventions standard values.
_KNOWN_PROVIDERS: dict[str, str] = {
    "openai": GenAiProviderNameValues.OPENAI.value,
    "azure": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure_openai": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure-openai": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure_ai": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "azure_ai_inference": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "azure-ai-inference": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "aws_bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "aws-bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "amazon_bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "anthropic": GenAiProviderNameValues.ANTHROPIC.value,
    "cohere": GenAiProviderNameValues.COHERE.value,
    "google": GenAiProviderNameValues.GCP_GEMINI.value,
    "gemini": GenAiProviderNameValues.GCP_GEMINI.value,
    "google_generativeai": GenAiProviderNameValues.GCP_GEMINI.value,
    "vertex_ai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "vertexai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "google_vertexai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "gcp_vertex_ai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "mistral": GenAiProviderNameValues.MISTRAL_AI.value,
    "mistralai": GenAiProviderNameValues.MISTRAL_AI.value,
    "mistral_ai": GenAiProviderNameValues.MISTRAL_AI.value,
    "groq": GenAiProviderNameValues.GROQ.value,
    "deepseek": GenAiProviderNameValues.DEEPSEEK.value,
    "watsonx": GenAiProviderNameValues.IBM_WATSONX_AI.value,
    "ibm_watsonx_ai": GenAiProviderNameValues.IBM_WATSONX_AI.value,
    "perplexity": GenAiProviderNameValues.PERPLEXITY.value,
    "xai": GenAiProviderNameValues.X_AI.value,
    "x_ai": GenAiProviderNameValues.X_AI.value,
    "ollama": "ollama",
    "fireworks": "fireworks",
    "together": "together",
    "voyage": "voyageai",
    "voyageai": "voyageai",
    "voyage_ai": "voyageai",
    "fastembed": "fastembed",
    "sentence_transformer": "sentence_transformer",
    "sentence_transformers": "sentence_transformer",
    "sentence-transformers": "sentence_transformer",
    "huggingface": "huggingface",
    "langdb": "langdb",
    "nebius": "nebius",
    "vllm": "vllm",
    "jina": "jina",
}

# Mapping of known embedder class names to provider values.
_CLASS_NAME_TO_PROVIDER: dict[str, str] = {
    "OpenAIEmbedder": GenAiProviderNameValues.OPENAI.value,
    "AzureOpenAIEmbedder": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "AwsBedrockEmbedder": GenAiProviderNameValues.AWS_BEDROCK.value,
    "CohereEmbedder": GenAiProviderNameValues.COHERE.value,
    "MistralEmbedder": GenAiProviderNameValues.MISTRAL_AI.value,
    "OllamaEmbedder": "ollama",
    "FireworksEmbedder": "fireworks",
    "TogetherEmbedder": "together",
    "VoyageAIEmbedder": "voyageai",
    "FastEmbedEmbedder": "fastembed",
    "SentenceTransformerEmbedder": "sentence_transformer",
    "HuggingfaceCustomEmbedder": "huggingface",
    "LangDBEmbedder": "langdb",
    "NebiusEmbedder": "nebius",
    "VLLMEmbedder": "vllm",
    "JinaEmbedder": "jina",
}


def resolve_embedder_provider(embedder: Any) -> str:
    """Resolve the ``gen_ai.provider.name`` value for an Agno embedder instance."""
    # 1. Explicit provider attribute on the embedder
    provider_attr = getattr(embedder, "provider", None)
    if provider_attr is not None:
        if isinstance(provider_attr, str):
            p_name = provider_attr.strip().lower()
            if p_name in _KNOWN_PROVIDERS:
                return _KNOWN_PROVIDERS[p_name]
            if p_name and p_name != "none":
                return p_name
        else:
            cls_name = provider_attr.__class__.__name__.lower()
            if "provider" in cls_name and cls_name != "provider":
                p_name = cls_name.removesuffix("provider")
                if p_name in _KNOWN_PROVIDERS:
                    return _KNOWN_PROVIDERS[p_name]
                if p_name:
                    return p_name

    # 2. Check the embedder class hierarchy (most derived first)
    for cls in type(embedder).__mro__:
        cls_name = cls.__name__
        if cls_name in ("GeminiEmbedder", "GoogleEmbedder"):
            if getattr(embedder, "vertexai", False) or (
                os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower()
                == "true"
            ):
                return GenAiProviderNameValues.GCP_VERTEX_AI.value
            return GenAiProviderNameValues.GCP_GEMINI.value
        if cls_name == "OpenAILikeEmbedder":
            # OpenAILikeEmbedder is an adapter for arbitrary OpenAI-compatible endpoints.
            # It inherits from OpenAIEmbedder, so stop MRO traversal to avoid attributing it to OpenAI.
            break
        if cls_name in _CLASS_NAME_TO_PROVIDER:
            return _CLASS_NAME_TO_PROVIDER[cls_name]
        if cls_name == "Embedder":
            # Base class reached without matching a known embedder
            break

    # 3. Check module name if in agno.knowledge.embedder.<submodule>
    module = getattr(embedder, "__module__", "")
    if "agno.knowledge.embedder." in module:
        sub = module.split("agno.knowledge.embedder.")[-1].split(".")[0]
        if sub not in ("base", "openai_like"):
            if sub == "google":
                if getattr(embedder, "vertexai", False) or (
                    os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower()
                    == "true"
                ):
                    return GenAiProviderNameValues.GCP_VERTEX_AI.value
                return GenAiProviderNameValues.GCP_GEMINI.value
            if sub in _KNOWN_PROVIDERS:
                return _KNOWN_PROVIDERS[sub]

    # 4. Check model/id prefix if it has provider/model format
    model = (
        getattr(embedder, "id", None)
        or getattr(embedder, "model", None)
        or getattr(embedder, "name", None)
    )
    if model is not None and isinstance(model, str):
        model_str = model.strip()
        if "/" in model_str:
            prefix = model_str.split("/")[0].strip().lower()
            if prefix in _KNOWN_PROVIDERS:
                return _KNOWN_PROVIDERS[prefix]

    # 5. Unresolved - fallback to unknown
    return _UNKNOWN_PROVIDER
