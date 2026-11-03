# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import contextlib
import functools
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from opentelemetry.context import Context, get_current
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
    UserSpeechInvocation,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    MessagePart,
    Modality,
    OutputMessage,
    ReasoningPart,
    SystemInstructionPart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
    TranscriptionPart,
)
from opentelemetry.util.genai.utils import (
    decode_base64,
    get_argument,
    object_to_any_value,
)

from ._error_type import resolve_error_type
from .message import _to_part, to_input_messages, to_system_instructions

try:
    from google.genai import live as genai_live
    from google.genai.live import AsyncLive, AsyncSession

    _HAS_LIVE = True
except ImportError:  # pragma: no cover
    genai_live = None  # type: ignore[assignment]
    AsyncLive = None  # type: ignore[assignment,misc]
    AsyncSession = None  # type: ignore[assignment,misc]
    _HAS_LIVE = False

_ORIGINAL_ASYNC_LIVE_CONNECT = AsyncLive.connect if _HAS_LIVE else None


def _strip_model_prefix(model: Any) -> str | None:
    if not model:
        return None
    model_str = str(model)
    if model_str.startswith("models/"):
        return model_str[len("models/") :]
    return model_str


def _get_config_field(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _extract_live_endpoint_info(
    live_instance: Any,
) -> tuple[str, str | None, int | None]:
    """Return ``(provider, server_address, server_port)`` for an ``AsyncLive`` instance."""
    api_client = getattr(live_instance, "_api_client", None)
    is_vertex = bool(getattr(api_client, "vertexai", False))
    provider = (
        GenAI.GenAiProviderNameValues.GCP_VERTEX_AI.value
        if is_vertex
        else GenAI.GenAiProviderNameValues.GCP_GEMINI.value
    )

    raw_url: Any = None
    if api_client is not None:
        http_options = getattr(api_client, "_http_options", None)
        if http_options is not None:
            raw_url = getattr(http_options, "base_url", None)
        if not raw_url:
            raw_url = getattr(api_client, "custom_base_url", None)
        if not raw_url and hasattr(api_client, "_websocket_base_url"):
            try:
                ws_url = api_client._websocket_base_url()
                if isinstance(ws_url, bytes):
                    ws_url = ws_url.decode("utf-8")
                raw_url = ws_url
            except Exception:  # pylint: disable=broad-exception-caught
                raw_url = None

    server_address: str | None = None
    server_port: int | None = None
    if raw_url:
        url_str = str(raw_url)
        if "://" not in url_str:
            url_str = f"wss://{url_str}"
        parsed = urlsplit(url_str)
        server_address = parsed.hostname
        server_port = parsed.port
        if server_address and server_port is None:
            server_port = 80 if parsed.scheme in ("ws", "http") else 443

    if not server_address:
        server_address = (
            "aiplatform.googleapis.com"
            if is_vertex
            else "generativelanguage.googleapis.com"
        )
        server_port = 443

    return provider, server_address, server_port


def _resolve_output_type(config: Any, is_vertex: bool) -> str | None:
    generation_config = _get_config_field(config, "generation_config")
    modalities = _get_config_field(
        config, "response_modalities"
    ) or _get_config_field(generation_config, "response_modalities")
    if modalities:
        upper_modalities = [
            str(getattr(m, "value", m)).upper() for m in modalities
        ]
        if any(m.endswith("AUDIO") for m in upper_modalities):
            return "speech"
        if any(m.endswith("TEXT") for m in upper_modalities):
            return "text"
        if any(m.endswith("IMAGE") for m in upper_modalities):
            return "image"
    if is_vertex:
        # Vertex AI Live defaults responseModalities to ["AUDIO"] when omitted.
        return "speech"
    return None


def _extract_system_instruction(
    sys_inst: Any,
) -> list[SystemInstructionPart]:
    if sys_inst is None:
        return []
    if isinstance(sys_inst, str):
        return [TextPart(content=sys_inst)]
    if hasattr(sys_inst, "parts"):
        try:
            return to_system_instructions(content=sys_inst)
        except Exception:  # pylint: disable=broad-exception-caught
            return []
    if isinstance(sys_inst, Mapping):
        if "parts" in sys_inst:
            parts_out: list[SystemInstructionPart] = []
            for p in sys_inst.get("parts") or []:
                if isinstance(p, str):
                    parts_out.append(TextPart(content=p))
                elif isinstance(p, Mapping) and "text" in p:
                    parts_out.append(TextPart(content=str(p["text"])))
            return parts_out
        if "text" in sys_inst:
            return [TextPart(content=str(sys_inst["text"]))]
    if isinstance(sys_inst, Sequence):
        seq_out: list[SystemInstructionPart] = []
        for item in sys_inst:
            seq_out.extend(_extract_system_instruction(item))
        return seq_out
    if hasattr(sys_inst, "text") and getattr(sys_inst, "text") is not None:
        return [TextPart(content=str(sys_inst.text))]
    return []


def _extract_tool_definitions(
    tools: Any, capture_content: bool
) -> list[ToolDefinition]:
    if not tools or not isinstance(tools, Sequence):
        return []
    definitions: list[ToolDefinition] = []
    for item in tools:
        if callable(item):
            name = getattr(item, "__name__", "unknown")
            doc = getattr(item, "__doc__", None)
            description = doc.strip() if doc and capture_content else None
            definitions.append(
                FunctionToolDefinition(
                    name=name,
                    description=description,
                    parameters=None,
                )
            )
            continue
        item_dict = (
            item
            if isinstance(item, Mapping)
            else item.model_dump(exclude_none=True)
            if hasattr(item, "model_dump")
            else {}
        )
        if not isinstance(item_dict, Mapping):
            continue
        for key, value in item_dict.items():
            if key == "function_declarations" and isinstance(value, Sequence):
                for func in value:
                    func_dict = (
                        func
                        if isinstance(func, Mapping)
                        else func.model_dump(exclude_none=True)
                        if hasattr(func, "model_dump")
                        else {}
                    )
                    if not isinstance(func_dict, Mapping):
                        continue
                    fname = func_dict.get("name", "unknown")
                    fdesc = (
                        func_dict.get("description")
                        if capture_content
                        else None
                    )
                    fparams = (
                        (
                            func_dict.get("parameters")
                            or func_dict.get("parameters_json_schema")
                        )
                        if capture_content
                        else None
                    )
                    definitions.append(
                        FunctionToolDefinition(
                            name=fname,
                            description=fdesc,
                            parameters=fparams,
                        )
                    )
            elif value is not None:
                definitions.append(GenericToolDefinition(name=key, type=key))
    return definitions


def _extract_blob_bytes_and_mime(
    blob_or_dict: Any, default_mime: str = "audio/pcm"
) -> tuple[bytes, str] | None:
    if blob_or_dict is None:
        return None
    if isinstance(blob_or_dict, Mapping):
        raw_data = blob_or_dict.get("data")
        mime_type = (
            blob_or_dict.get("mime_type")
            or blob_or_dict.get("mimeType")
            or default_mime
        )
    else:
        raw_data = getattr(blob_or_dict, "data", None)
        mime_type = getattr(blob_or_dict, "mime_type", None) or default_mime

    if raw_data is None:
        return None
    if isinstance(raw_data, bytes):
        return raw_data, str(mime_type)
    if isinstance(raw_data, bytearray):
        return bytes(raw_data), str(mime_type)
    if isinstance(raw_data, str):
        decoded = decode_base64(raw_data)
        if decoded is not None:
            return decoded, str(mime_type)
        return raw_data.encode("utf-8"), str(mime_type)
    return None


def _build_tool_response_input_message(
    function_responses: Any,
) -> InputMessage | None:
    if function_responses is None:
        return None
    if isinstance(function_responses, Mapping) or not isinstance(
        function_responses, Sequence
    ):
        items: Sequence[Any] = [function_responses]
    else:
        items = function_responses

    parts: list[MessagePart] = []
    for item in items:
        if isinstance(item, Mapping):
            call_id = item.get("id")
            response_val = item.get("response")
        else:
            call_id = getattr(item, "id", None)
            response_val = getattr(item, "response", None)
        parts.append(
            ToolCallResponsePart(
                id=str(call_id) if call_id is not None else None,
                response=object_to_any_value(response_val),
            )
        )
    if not parts:
        return None
    return InputMessage(role="tool", parts=parts)


def _is_interaction_complete(server_content: Any) -> bool:
    if server_content is None:
        return False
    if genai_live is not None and hasattr(
        genai_live, "_is_interaction_complete"
    ):
        try:
            return bool(genai_live._is_interaction_complete(server_content))
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return bool(getattr(server_content, "turn_complete", False))


@dataclass
class _ActiveGeneration:
    invocation: InferenceInvocation
    input_messages: list[InputMessage] = field(default_factory=list)
    completed_output_messages: list[OutputMessage] = field(
        default_factory=list
    )
    input_transcript_deltas: list[str] = field(default_factory=list)
    output_transcript_deltas: list[str] = field(default_factory=list)
    output_audio_chunks: list[bytes] = field(default_factory=list)
    output_audio_mime_type: str | None = None
    output_text_deltas: list[str] = field(default_factory=list)
    output_thought_deltas: list[str] = field(default_factory=list)
    output_other_parts: list[MessagePart] = field(default_factory=list)
    usage_metadata: Any = None
    interrupted: bool = False
    generation_complete: bool = False
    turn_complete: bool = False
    awaiting_tool_response: bool = False

    def flush_current_turn_output_message(
        self, finish_reason: str | None
    ) -> OutputMessage | None:
        parts: list[MessagePart] = []
        if self.output_thought_deltas:
            parts.append(
                ReasoningPart(content="".join(self.output_thought_deltas))
            )
            self.output_thought_deltas.clear()
        if self.output_audio_chunks:
            parts.append(
                BlobPart(
                    mime_type=self.output_audio_mime_type or "audio/pcm",
                    modality=Modality.AUDIO,
                    content=b"".join(self.output_audio_chunks),
                )
            )
            self.output_audio_chunks.clear()
            self.output_audio_mime_type = None
        if self.output_text_deltas:
            parts.append(TextPart(content="".join(self.output_text_deltas)))
            self.output_text_deltas.clear()
        if self.output_transcript_deltas:
            parts.append(
                TranscriptionPart(
                    content="".join(self.output_transcript_deltas)
                )
            )
            self.output_transcript_deltas.clear()
        if self.output_other_parts:
            parts.extend(self.output_other_parts)
            self.output_other_parts.clear()

        if not parts:
            return None
        msg = OutputMessage(
            role="assistant",
            parts=parts,
            finish_reason=finish_reason,
        )
        self.completed_output_messages.append(msg)
        return msg


class _LiveSessionTracker:
    """Tracks state for a single ``AsyncSession`` established by ``AsyncLive.connect``."""

    def __init__(
        self,
        *,
        session: Any,
        telemetry_handler: TelemetryHandler,
        provider: str,
        is_vertex: bool,
        request_model: str | None,
        realtime_session_id: str | None,
        server_address: str | None,
        server_port: int | None,
        config: Any,
        caller_context: Context,
    ) -> None:
        self.session = session
        self._telemetry_handler = telemetry_handler
        self.provider = provider
        self.is_vertex = is_vertex
        self.request_model = request_model
        self.realtime_session_id = realtime_session_id
        self.server_address = server_address
        self.server_port = server_port
        self.config = config
        self.caller_context = caller_context

        self.output_type = _resolve_output_type(config, is_vertex)
        generation_config = _get_config_field(config, "generation_config")
        self.max_tokens = _get_config_field(
            config, "max_output_tokens"
        ) or _get_config_field(generation_config, "max_output_tokens")
        self.temperature = _get_config_field(
            config, "temperature"
        ) or _get_config_field(generation_config, "temperature")
        self.top_p = _get_config_field(config, "top_p") or _get_config_field(
            generation_config, "top_p"
        )
        self.top_k = _get_config_field(config, "top_k") or _get_config_field(
            generation_config, "top_k"
        )
        self.presence_penalty = _get_config_field(
            config, "presence_penalty"
        ) or _get_config_field(generation_config, "presence_penalty")
        self.frequency_penalty = _get_config_field(
            config, "frequency_penalty"
        ) or _get_config_field(generation_config, "frequency_penalty")
        self.seed = _get_config_field(config, "seed") or _get_config_field(
            generation_config, "seed"
        )
        thinking_config = _get_config_field(
            config, "thinking_config"
        ) or _get_config_field(generation_config, "thinking_config")
        thinking_level = _get_config_field(thinking_config, "thinking_level")
        self.reasoning_level = (
            str(getattr(thinking_level, "value", thinking_level)).lower()
            if thinking_level is not None
            else None
        )

        # Gemini Live 3.x models report the tool request and spoken answer
        # within a single generation; 2.x models report them as two sibling
        # generations.
        self.uses_single_generation_tool_exchange: bool = "gemini-3" in (
            (request_model or "").lower()
        )

        self.system_instruction: list[SystemInstructionPart] = []
        self.tool_definitions: list[ToolDefinition] = []
        if self._telemetry_handler.should_capture_content():
            sys_inst = _get_config_field(config, "system_instruction")
            if sys_inst is not None:
                self.system_instruction = _extract_system_instruction(sys_inst)
            tools = _get_config_field(config, "tools")
            if tools is not None:
                self.tool_definitions = _extract_tool_definitions(
                    tools, capture_content=True
                )

        self._pending_audio_chunks: list[bytes] = []
        self._pending_audio_mime_type: str | None = None
        self._pending_input_messages: list[InputMessage] = []
        self._active_gen: _ActiveGeneration | None = None
        self._speech_invocation: UserSpeechInvocation | None = None

    def _flush_pending_audio(self) -> None:
        if not self._pending_audio_chunks:
            return
        if self._telemetry_handler.should_capture_content():
            combined = b"".join(self._pending_audio_chunks)
            mime_type = self._pending_audio_mime_type or "audio/pcm"
            self._pending_input_messages.append(
                InputMessage(
                    role="user",
                    parts=[
                        BlobPart(
                            mime_type=mime_type,
                            modality=Modality.AUDIO,
                            content=combined,
                        )
                    ],
                )
            )
        self._pending_audio_chunks.clear()
        self._pending_audio_mime_type = None

    def _start_user_speech(self) -> None:
        if self._speech_invocation is not None:
            return
        self._speech_invocation = self._telemetry_handler.user_speech(
            provider=self.provider,
            realtime_session_id=self.realtime_session_id,
            error_type_resolver=resolve_error_type,
            context=get_current(),
            _attach_to_context=False,
        )

    def _stop_user_speech(
        self, error: BaseException | None = None
    ) -> None:
        if self._speech_invocation is None:
            return
        speech = self._speech_invocation
        self._speech_invocation = None
        speech.realtime_session_id = self.realtime_session_id
        if error is not None:
            speech.fail(error)
        else:
            speech.stop()

    def on_send_realtime_input(self, kwargs: Mapping[str, Any]) -> None:
        if kwargs.get("activity_start") is not None:
            self._start_user_speech()

        audio = kwargs.get("audio")
        media = kwargs.get("media")
        video = kwargs.get("video")
        text = kwargs.get("text")
        audio_stream_end = kwargs.get("audio_stream_end")
        activity_end = kwargs.get("activity_end")

        if audio is not None:
            extracted = _extract_blob_bytes_and_mime(
                audio, default_mime="audio/pcm"
            )
            if extracted is not None:
                data_bytes, mime_type = extracted
                self._pending_audio_chunks.append(data_bytes)
                if not self._pending_audio_mime_type:
                    self._pending_audio_mime_type = mime_type
        elif media is not None:
            extracted = _extract_blob_bytes_and_mime(
                media, default_mime="audio/pcm"
            )
            if extracted is not None:
                data_bytes, mime_type = extracted
                if "audio" in mime_type.lower():
                    self._pending_audio_chunks.append(data_bytes)
                    if not self._pending_audio_mime_type:
                        self._pending_audio_mime_type = mime_type
                else:
                    self._flush_pending_audio()
                    if self._telemetry_handler.should_capture_content():
                        modality = (
                            Modality.IMAGE
                            if "image" in mime_type.lower()
                            else Modality.VIDEO
                            if "video" in mime_type.lower()
                            else Modality.DOCUMENT
                        )
                        self._pending_input_messages.append(
                            InputMessage(
                                role="user",
                                parts=[
                                    BlobPart(
                                        mime_type=mime_type,
                                        modality=modality,
                                        content=data_bytes,
                                    )
                                ],
                            )
                        )
        elif video is not None:
            self._flush_pending_audio()
            extracted = _extract_blob_bytes_and_mime(
                video, default_mime="video/mp4"
            )
            if (
                extracted is not None
                and self._telemetry_handler.should_capture_content()
            ):
                data_bytes, mime_type = extracted
                self._pending_input_messages.append(
                    InputMessage(
                        role="user",
                        parts=[
                            BlobPart(
                                mime_type=mime_type,
                                modality=Modality.VIDEO,
                                content=data_bytes,
                            )
                        ],
                    )
                )
        elif text is not None:
            self._flush_pending_audio()
            if self._telemetry_handler.should_capture_content():
                self._pending_input_messages.append(
                    InputMessage(
                        role="user",
                        parts=[TextPart(content=str(text))],
                    )
                )

        if audio_stream_end:
            self._flush_pending_audio()
        if activity_end is not None:
            self._flush_pending_audio()
            self._stop_user_speech()

    def on_send_client_content(
        self, turns: Any, turn_complete: bool = True  # pylint: disable=unused-argument
    ) -> None:
        self._flush_pending_audio()
        if (
            turns is not None
            and self._telemetry_handler.should_capture_content()
        ):
            try:
                messages = to_input_messages(contents=turns)
                self._pending_input_messages.extend(messages)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def on_send_tool_response(self, function_responses: Any) -> None:
        self._flush_pending_audio()
        if (
            self._active_gen is not None
            and self._active_gen.awaiting_tool_response
        ):
            # Per PR #394 (model/gen-ai/spans.yaml), gen_ai.input.messages
            # represents the input that initiated the generation and SHOULD NOT
            # include intermediate events that occur while the generation is
            # active, such as a tool result sent in response to a tool call.
            self._active_gen.awaiting_tool_response = False
            return
        if not self._telemetry_handler.should_capture_content():
            return
        tool_msg = _build_tool_response_input_message(function_responses)
        if tool_msg is None:
            return
        self._pending_input_messages.append(tool_msg)

    def on_send_deprecated(
        self, input_val: Any, end_of_turn: bool | None = False  # pylint: disable=unused-argument
    ) -> None:
        if input_val is None:
            self._flush_pending_audio()
            return
        if isinstance(input_val, str):
            self.on_send_realtime_input({"text": input_val})
            return
        if isinstance(input_val, Mapping):
            if "data" in input_val:
                self.on_send_realtime_input({"media": input_val})
                return
            if "name" in input_val and "response" in input_val:
                self.on_send_tool_response(input_val)
                return
            if "function_responses" in input_val:
                self.on_send_tool_response(input_val["function_responses"])
                return
            if "turns" in input_val or "content" in input_val:
                self.on_send_client_content(
                    input_val.get("turns") or input_val.get("content")
                )
                return
            if "media_chunks" in input_val:
                for chunk in input_val.get("media_chunks") or []:
                    self.on_send_realtime_input({"media": chunk})
                return
        # Handle typed SDK objects (Blob, FunctionResponse, LiveClientContent, etc.)
        if hasattr(input_val, "media_chunks") and input_val.media_chunks:
            for chunk in input_val.media_chunks:
                self.on_send_realtime_input({"media": chunk})
            return
        if (
            hasattr(input_val, "function_responses")
            and input_val.function_responses
        ):
            self.on_send_tool_response(input_val.function_responses)
            return
        if hasattr(input_val, "turns") and input_val.turns is not None:
            self.on_send_client_content(input_val.turns)
            return
        if hasattr(input_val, "data") and hasattr(input_val, "mime_type"):
            self.on_send_realtime_input({"media": input_val})
            return
        if hasattr(input_val, "name") and hasattr(input_val, "response"):
            self.on_send_tool_response(input_val)
            return

    def on_send_context_update(
        self, system_instruction: Any = None, tools: Any = None
    ) -> None:
        if not self._telemetry_handler.should_capture_content():
            return
        if system_instruction is not None:
            self.system_instruction = _extract_system_instruction(
                system_instruction
            )
        if tools is not None:
            self.tool_definitions = _extract_tool_definitions(
                tools, capture_content=True
            )

    def ensure_active_generation(self) -> _ActiveGeneration:
        self._flush_pending_audio()
        if self._active_gen is None:
            invocation = self._telemetry_handler.realtime_inference(
                self.provider,
                request_model=self.request_model,
                server_address=self.server_address,
                server_port=self.server_port,
                realtime_session_id=self.realtime_session_id,
                error_type_resolver=resolve_error_type,
                context=get_current(),
                _attach_to_context=False,
            )
            if self.max_tokens is not None:
                invocation.max_tokens = self.max_tokens
            if self.temperature is not None:
                invocation.temperature = self.temperature
            if self.top_p is not None:
                invocation.top_p = self.top_p
            if self.top_k is not None:
                invocation.top_k = self.top_k
            if self.presence_penalty is not None:
                invocation.presence_penalty = self.presence_penalty
            if self.frequency_penalty is not None:
                invocation.frequency_penalty = self.frequency_penalty
            if self.seed is not None:
                invocation.seed = self.seed
            if self.reasoning_level is not None:
                invocation.reasoning_level = self.reasoning_level

            input_msgs = list(self._pending_input_messages)
            self._pending_input_messages.clear()
            self._active_gen = _ActiveGeneration(
                invocation=invocation,
                input_messages=input_msgs,
            )
        elif self._pending_input_messages:
            self._active_gen.input_messages.extend(
                self._pending_input_messages
            )
            self._pending_input_messages.clear()
        return self._active_gen

    def _populate_and_finish_generation(
        self,
        active_gen: _ActiveGeneration,
        finish_reason: str,
        error: BaseException | None = None,
    ) -> None:
        invocation = active_gen.invocation
        invocation.realtime_session_id = self.realtime_session_id

        # Attach any server-reported input audio transcription to the user message.
        if (
            active_gen.input_transcript_deltas
            and self._telemetry_handler.should_capture_content()
        ):
            input_transcript = "".join(active_gen.input_transcript_deltas)
            active_gen.input_transcript_deltas.clear()
            user_msg: InputMessage | None = None
            for msg in reversed(active_gen.input_messages):
                if msg.role == "user":
                    user_msg = msg
                    break
            if user_msg is not None:
                user_msg.parts.append(
                    TranscriptionPart(content=input_transcript)
                )
            else:
                active_gen.input_messages.append(
                    InputMessage(
                        role="user",
                        parts=[TranscriptionPart(content=input_transcript)],
                    )
                )

        has_non_tool_output = bool(
            active_gen.output_audio_chunks
            or active_gen.output_text_deltas
            or active_gen.output_transcript_deltas
            or active_gen.output_thought_deltas
            or active_gen.output_other_parts
        )
        active_gen.flush_current_turn_output_message(finish_reason)

        invocation.finish_reasons = [finish_reason]
        if finish_reason == "tool_call" and not has_non_tool_output:
            invocation.output_type = None
        else:
            invocation.output_type = self.output_type

        if self._telemetry_handler.should_capture_content():
            invocation.input_messages = active_gen.input_messages
            invocation.output_messages = active_gen.completed_output_messages
            if self.system_instruction:
                invocation.system_instruction = self.system_instruction
            if self.tool_definitions:
                invocation.tool_definitions = self.tool_definitions

        usage = active_gen.usage_metadata
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            if prompt_tokens is not None:
                invocation.input_tokens = prompt_tokens

            response_tokens = getattr(usage, "response_token_count", None)
            if response_tokens is not None:
                invocation.output_tokens = response_tokens
            else:
                candidates_tokens = getattr(
                    usage, "candidates_token_count", None
                )
                thoughts_tokens = getattr(usage, "thoughts_token_count", None)
                if candidates_tokens is not None:
                    invocation.output_tokens = candidates_tokens + (
                        thoughts_tokens or 0
                    )

            cached_tokens = getattr(usage, "cached_content_token_count", None)
            if cached_tokens is not None:
                invocation.cache_read_input_tokens = cached_tokens

            thoughts_tokens = getattr(usage, "thoughts_token_count", None)
            if thoughts_tokens is not None:
                invocation.thinking_tokens = thoughts_tokens

            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None:
                invocation.set_input_tokens(
                    (detail.modality, detail.token_count)
                    for detail in prompt_details
                )

            response_details = getattr(
                usage, "response_tokens_details", None
            ) or getattr(usage, "candidates_tokens_details", None)
            if response_details is not None:
                invocation.set_output_tokens(
                    (detail.modality, detail.token_count)
                    for detail in response_details
                )

            cache_details = getattr(usage, "cache_tokens_details", None)
            if cache_details is not None:
                invocation.set_cache_read_input_tokens(
                    (detail.modality, detail.token_count)
                    for detail in cache_details
                )

        if error is not None:
            invocation.fail(error)
        else:
            invocation.stop()

    def on_server_message(self, message: Any) -> None:
        if message is None:
            return

        # Update session_id if exposed in setup_complete.
        setup_complete = getattr(message, "setup_complete", None)
        if setup_complete is not None:
            sid = getattr(setup_complete, "session_id", None)
            if sid:
                self.realtime_session_id = str(sid)

        # Handle server-side VAD events if exposed by the provider.
        voice_activity = getattr(message, "voice_activity", None)
        if voice_activity is not None:
            va_type = str(
                getattr(
                    getattr(voice_activity, "voice_activity_type", None),
                    "value",
                    getattr(voice_activity, "voice_activity_type", ""),
                )
            ).upper()
            if "START" in va_type:
                self._start_user_speech()
            elif "END" in va_type:
                self._flush_pending_audio()
                self._stop_user_speech()

        vad_signal = getattr(
            message, "voice_activity_detection_signal", None
        )
        if vad_signal is not None:
            sig_type = str(
                getattr(
                    getattr(vad_signal, "vad_signal_type", None),
                    "value",
                    getattr(vad_signal, "vad_signal_type", ""),
                )
            ).upper()
            if "START" in sig_type or "BOS" in sig_type:
                self._start_user_speech()
            elif "END" in sig_type or "EOS" in sig_type:
                self._flush_pending_audio()
                self._stop_user_speech()

        server_content = getattr(message, "server_content", None)
        tool_call = getattr(message, "tool_call", None)
        usage_metadata = getattr(message, "usage_metadata", None)

        if (
            self._active_gen is None
            and server_content is None
            and tool_call is None
            and usage_metadata is None
        ):
            return

        active_gen = self.ensure_active_generation()

        if usage_metadata is not None:
            active_gen.usage_metadata = usage_metadata

        if server_content is not None:
            input_tx = getattr(server_content, "input_transcription", None)
            if input_tx is not None and getattr(input_tx, "text", None):
                active_gen.input_transcript_deltas.append(str(input_tx.text))

            output_tx = getattr(server_content, "output_transcription", None)
            if output_tx is not None and getattr(output_tx, "text", None):
                active_gen.output_transcript_deltas.append(str(output_tx.text))

            model_turn = getattr(server_content, "model_turn", None)
            if model_turn is not None and getattr(model_turn, "parts", None):
                for part in model_turn.parts:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data is not None and getattr(
                        inline_data, "data", None
                    ):
                        raw_data = inline_data.data
                        if isinstance(raw_data, str):
                            raw_data = (
                                decode_base64(raw_data)
                                or raw_data.encode("utf-8")
                            )
                        mime = (
                            getattr(inline_data, "mime_type", None)
                            or "audio/pcm"
                        )
                        if "audio" in mime.lower():
                            active_gen.output_audio_chunks.append(raw_data)
                            if not active_gen.output_audio_mime_type:
                                active_gen.output_audio_mime_type = mime
                        elif self._telemetry_handler.should_capture_content():
                            converted = _to_part(
                                part, len(active_gen.output_other_parts)
                            )
                            if converted is not None:
                                active_gen.output_other_parts.append(converted)
                    elif getattr(part, "text", None) is not None:
                        if getattr(part, "thought", False):
                            active_gen.output_thought_deltas.append(
                                str(part.text)
                            )
                        else:
                            active_gen.output_text_deltas.append(
                                str(part.text)
                            )
                    elif self._telemetry_handler.should_capture_content():
                        converted = _to_part(
                            part, len(active_gen.output_other_parts)
                        )
                        if converted is not None:
                            active_gen.output_other_parts.append(converted)

            if getattr(server_content, "interrupted", False):
                active_gen.interrupted = True
            if getattr(server_content, "generation_complete", False):
                active_gen.generation_complete = True
            if getattr(server_content, "turn_complete", False):
                active_gen.turn_complete = True

        if tool_call is not None and getattr(
            tool_call, "function_calls", None
        ):
            if self.uses_single_generation_tool_exchange:
                # Per PR #394, gen_ai.output.messages represents the output
                # produced by the completed generation and SHOULD NOT include
                # intermediate events that occur while the generation is active,
                # such as a tool call handled before the generation completes.
                active_gen.awaiting_tool_response = True
            else:
                tool_parts: list[MessagePart] = [
                    ToolCallRequestPart(
                        id=getattr(fc, "id", None),
                        name=getattr(fc, "name", None) or "",
                        arguments=dict(fc.args)
                        if getattr(fc, "args", None) is not None
                        else {},
                    )
                    for fc in tool_call.function_calls
                ]
                active_gen.flush_current_turn_output_message(
                    finish_reason=None
                )
                active_gen.completed_output_messages.append(
                    OutputMessage(
                        role="assistant",
                        parts=tool_parts,
                        finish_reason="tool_call",
                    )
                )
                self._active_gen = None
                self._populate_and_finish_generation(
                    active_gen, finish_reason="tool_call"
                )
                return

        if _is_interaction_complete(server_content):
            finish_reason = (
                "interrupted" if active_gen.interrupted else "stop"
            )
            self._active_gen = None
            self._populate_and_finish_generation(
                active_gen, finish_reason=finish_reason
            )

    def on_receive_error(self, exc: BaseException) -> None:
        if self._active_gen is None and (
            self._pending_input_messages or self._pending_audio_chunks
        ):
            self.ensure_active_generation()
        if self._active_gen is not None:
            active_gen = self._active_gen
            self._active_gen = None
            self._populate_and_finish_generation(
                active_gen, finish_reason="error", error=exc
            )

    def on_session_error(self, exc: BaseException) -> None:
        self._stop_user_speech(error=exc)
        self.on_receive_error(exc)

    def on_session_close(self) -> None:
        self._stop_user_speech()
        if self._active_gen is not None:
            active_gen = self._active_gen
            self._active_gen = None
            finish_reason = (
                "interrupted"
                if active_gen.interrupted
                else "stop"
                if active_gen.generation_complete or active_gen.turn_complete
                else "error"
            )
            self._populate_and_finish_generation(
                active_gen, finish_reason=finish_reason
            )


def _wrap_session(session: Any, tracker: _LiveSessionTracker) -> None:
    """Wrap the methods of a live ``AsyncSession`` instance to record telemetry."""
    orig_send_realtime_input = session.send_realtime_input
    orig_send_client_content = session.send_client_content
    orig_send_tool_response = session.send_tool_response
    orig_send = session.send
    orig_send_context_update = getattr(session, "send_context_update", None)
    orig_receive = session._receive

    @functools.wraps(orig_send_realtime_input)
    async def wrapped_send_realtime_input(
        *args: Any, **kwargs: Any
    ) -> Any:
        tracker.on_send_realtime_input(kwargs)
        return await orig_send_realtime_input(*args, **kwargs)

    @functools.wraps(orig_send_client_content)
    async def wrapped_send_client_content(
        *args: Any, **kwargs: Any
    ) -> Any:
        turns = get_argument("turns", orig_send_client_content, args, kwargs)
        turn_complete = get_argument(
            "turn_complete", orig_send_client_content, args, kwargs
        )
        tracker.on_send_client_content(
            turns,
            turn_complete=True if turn_complete is None else bool(turn_complete),
        )
        return await orig_send_client_content(*args, **kwargs)

    @functools.wraps(orig_send_tool_response)
    async def wrapped_send_tool_response(
        *args: Any, **kwargs: Any
    ) -> Any:
        function_responses = get_argument(
            "function_responses", orig_send_tool_response, args, kwargs
        )
        tracker.on_send_tool_response(function_responses)
        return await orig_send_tool_response(*args, **kwargs)

    @functools.wraps(orig_send)
    async def wrapped_send(*args: Any, **kwargs: Any) -> Any:
        input_val = get_argument("input", orig_send, args, kwargs)
        end_of_turn = get_argument("end_of_turn", orig_send, args, kwargs)
        tracker.on_send_deprecated(input_val, end_of_turn=end_of_turn)
        return await orig_send(*args, **kwargs)

    @functools.wraps(orig_receive)
    async def wrapped_receive(*args: Any, **kwargs: Any) -> Any:
        active_gen = tracker._active_gen
        try:
            if active_gen is not None:
                with active_gen.invocation.activate():
                    message = await orig_receive(*args, **kwargs)
            else:
                message = await orig_receive(*args, **kwargs)
        except BaseException as exc:
            tracker.on_receive_error(exc)
            raise
        tracker.on_server_message(message)
        return message

    session.send_realtime_input = wrapped_send_realtime_input
    session.send_client_content = wrapped_send_client_content
    session.send_tool_response = wrapped_send_tool_response
    session.send = wrapped_send
    session._receive = wrapped_receive

    if orig_send_context_update is not None:

        @functools.wraps(orig_send_context_update)
        async def wrapped_send_context_update(
            *args: Any, **kwargs: Any
        ) -> Any:
            sys_inst = get_argument(
                "system_instruction", orig_send_context_update, args, kwargs
            )
            tools = get_argument(
                "tools", orig_send_context_update, args, kwargs
            )
            tracker.on_send_context_update(
                system_instruction=sys_inst, tools=tools
            )
            return await orig_send_context_update(*args, **kwargs)

        session.send_context_update = wrapped_send_context_update


@contextlib.asynccontextmanager
async def _connect_context_manager(
    live_instance: Any,
    original_connect: Any,
    telemetry_handler: TelemetryHandler,
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    bound_args = (live_instance, *args)
    model = get_argument("model", original_connect, bound_args, kwargs)
    config = get_argument("config", original_connect, bound_args, kwargs)
    request_model_name = _strip_model_prefix(model)
    provider, server_address, server_port = _extract_live_endpoint_info(
        live_instance
    )
    api_client = getattr(live_instance, "_api_client", None)
    is_vertex = bool(getattr(api_client, "vertexai", False))
    caller_context = get_current()

    async with original_connect(live_instance, *args, **kwargs) as session:
        realtime_session_id = getattr(session, "session_id", None) or None
        telemetry_handler.emit_realtime_session_started(
            provider,
            request_model=request_model_name,
            realtime_session_id=realtime_session_id,
            server_address=server_address,
            server_port=server_port,
            context=caller_context,
        )
        tracker = _LiveSessionTracker(
            session=session,
            telemetry_handler=telemetry_handler,
            provider=provider,
            is_vertex=is_vertex,
            request_model=request_model_name,
            realtime_session_id=realtime_session_id,
            server_address=server_address,
            server_port=server_port,
            config=config,
            caller_context=caller_context,
        )
        _wrap_session(session, tracker)
        session_error: BaseException | None = None
        try:
            yield session
        except BaseException as exc:
            session_error = exc
            tracker.on_session_error(exc)
            raise
        finally:
            if session_error is None:
                tracker.on_session_close()
            telemetry_handler.emit_realtime_session_ended(
                provider,
                request_model=request_model_name,
                realtime_session_id=tracker.realtime_session_id,
                server_address=server_address,
                server_port=server_port,
                error=session_error,
                error_type_resolver=resolve_error_type,
                context=caller_context,
            )


class _LiveMethodsSnapshot:
    def __init__(self) -> None:
        self._original_connect = (
            AsyncLive.connect if _HAS_LIVE else None
        )
        self._original_connect_code = (
            AsyncLive.connect.__code__ if _HAS_LIVE else None
        )

    def restore(self) -> None:
        if _HAS_LIVE and self._original_connect is not None:
            if self._original_connect_code is not None:
                self._original_connect.__code__ = self._original_connect_code
            AsyncLive.connect = self._original_connect  # type: ignore[method-assign]


def instrument_live(
    telemetry_handler: TelemetryHandler,
) -> _LiveMethodsSnapshot:
    """Instrument ``AsyncLive.connect`` on ``google.genai.live``."""
    snapshot = _LiveMethodsSnapshot()
    if not _HAS_LIVE or snapshot._original_connect is None:
        return snapshot
    original_connect = snapshot._original_connect
    original_connect.__code__ = original_connect.__code__.replace(
        co_filename=__file__.replace("\\", "/")
    )

    @functools.wraps(original_connect)
    def instrumented_connect(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _connect_context_manager(
            self, original_connect, telemetry_handler, *args, **kwargs
        )

    AsyncLive.connect = instrumented_connect  # type: ignore[method-assign]
    return snapshot


def uninstrument_live(
    snapshot: _LiveMethodsSnapshot | None = None,
) -> None:
    """Restore ``AsyncLive.connect`` on ``google.genai.live``."""
    if snapshot is not None:
        snapshot.restore()
        return
    if not _HAS_LIVE or _ORIGINAL_ASYNC_LIVE_CONNECT is None:
        return
    AsyncLive.connect = _ORIGINAL_ASYNC_LIVE_CONNECT  # type: ignore[method-assign]
