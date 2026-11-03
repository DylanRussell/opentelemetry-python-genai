# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Google GenAI Live API (realtime) instrumentation (PR #394)."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from google.genai import types as genai_types
from google.genai.live import AsyncLive

from opentelemetry._logs import SeverityNumber
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)


class _FakeAsyncSession:
    """Fake google.genai.live.AsyncSession for deterministic unit tests."""

    def __init__(
        self,
        messages: list[genai_types.LiveServerMessage],
        session_id: str | None = "test-live-session-123",
    ) -> None:
        self._messages = list(messages)
        self.session_id = session_id
        self.sent_realtime_inputs: list[dict[str, Any]] = []
        self.sent_client_contents: list[dict[str, Any]] = []
        self.sent_tool_responses: list[dict[str, Any]] = []

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.sent_realtime_inputs.append(kwargs)

    async def send_client_content(self, **kwargs: Any) -> None:
        self.sent_client_contents.append(kwargs)

    async def send_tool_response(self, **kwargs: Any) -> None:
        self.sent_tool_responses.append(kwargs)

    async def send(self, **kwargs: Any) -> None:
        pass

    async def _receive(self) -> genai_types.LiveServerMessage:
        if not self._messages:
            return genai_types.LiveServerMessage()
        return self._messages.pop(0)

    async def receive(self) -> AsyncIterator[genai_types.LiveServerMessage]:
        while self._messages:
            msg = await self._receive()
            yield msg
            if (
                msg.server_content
                and msg.server_content.turn_complete is True
            ):
                break


def _make_fake_connect(fake_session: _FakeAsyncSession):
    @contextlib.asynccontextmanager
    async def _fake_connect(
        self: Any,
        *,
        model: str,
        config: genai_types.LiveConnectConfigOrDict | None = None,
    ) -> AsyncIterator[_FakeAsyncSession]:
        yield fake_session

    return _fake_connect


@pytest.fixture
def instrumented_live(
    tracer_provider, logger_provider, meter_provider, monkeypatch
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        yield


@pytest.mark.asyncio
async def test_live_voice_turn_spans_events_and_metrics(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    messages = [
        genai_types.LiveServerMessage(
            voice_activity=genai_types.VoiceActivity(
                voice_activity_type=genai_types.VoiceActivityType.ACTIVITY_START
            )
        ),
        genai_types.LiveServerMessage(
            voice_activity=genai_types.VoiceActivity(
                voice_activity_type=genai_types.VoiceActivityType.ACTIVITY_END
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="Say hello in three words."
                ),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            thought=True,
                            text="Thinking of three word greeting",
                        ),
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x00\x01" * 1200,
                                mime_type="audio/pcm;rate=24000",
                            )
                        ),
                    ],
                ),
                output_transcription=genai_types.Transcription(
                    text="Hello there friend."
                ),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                generation_complete=True,
                turn_complete=True,
            ),
            usage_metadata=genai_types.UsageMetadata(
                prompt_token_count=38,
                response_token_count=41,
                total_token_count=79,
                prompt_tokens_details=[
                    genai_types.ModalityTokenCount(
                        modality=genai_types.MediaModality.AUDIO,
                        token_count=32,
                    ),
                    genai_types.ModalityTokenCount(
                        modality=genai_types.MediaModality.TEXT,
                        token_count=6,
                    ),
                ],
                response_tokens_details=[
                    genai_types.ModalityTokenCount(
                        modality=genai_types.MediaModality.AUDIO,
                        token_count=30,
                    ),
                    genai_types.ModalityTokenCount(
                        modality=genai_types.MediaModality.TEXT,
                        token_count=11,
                    ),
                ],
            ),
        ),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-voice-1")

    with patch.object(
        AsyncLive, "connect", _make_fake_connect(fake_session)
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.AUDIO],
                system_instruction="Be concise.",
                temperature=0.7,
            )
            async with live.connect(
                model="models/gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x00" * 800,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                async for _ in session.receive():
                    pass

    # Verify spans: user_speech finishes before realtime_inference starts
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    speech_span, span = spans[0], spans[1]
    assert speech_span.name == "user_speech"
    assert speech_span.kind == SpanKind.INTERNAL
    assert speech_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "user_speech"
    assert speech_span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"
    assert speech_span.attributes["gen_ai.realtime_session.id"] == "sess-voice-1"
    assert speech_span.end_time <= span.start_time

    assert (
        span.name
        == "realtime_inference gemini-2.5-flash-native-audio-preview-12-2025"
    )
    assert span.kind == SpanKind.CLIENT
    attrs = dict(span.attributes)
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "realtime_inference"
    assert attrs[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"
    assert (
        attrs[GenAI.GEN_AI_REQUEST_MODEL]
        == "gemini-2.5-flash-native-audio-preview-12-2025"
    )
    assert attrs["gen_ai.realtime_session.id"] == "sess-voice-1"
    assert attrs[GenAI.GEN_AI_OUTPUT_TYPE] == "speech"
    assert attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 38
    assert attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 41
    assert attrs["gen_ai.usage.audio.input_tokens"] == 32
    assert attrs["gen_ai.usage.text.input_tokens"] == 6
    assert attrs["gen_ai.usage.audio.output_tokens"] == 30
    assert attrs["gen_ai.usage.text.output_tokens"] == 11
    assert attrs[ServerAttributes.SERVER_ADDRESS] == "generativelanguage.googleapis.com"
    assert attrs[ServerAttributes.SERVER_PORT] == 443

    input_msgs = json.loads(attrs[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "blob"
    assert input_msgs[0]["parts"][0]["modality"] == "audio"
    assert input_msgs[0]["parts"][1] == {
        "type": "transcription",
        "content": "Say hello in three words.",
    }

    output_msgs = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["finish_reason"] == "stop"
    part_types = [p["type"] for p in output_msgs[0]["parts"]]
    assert part_types == ["reasoning", "blob", "transcription"]
    assert (
        output_msgs[0]["parts"][2]["content"] == "Hello there friend."
    )

    sys_instr = json.loads(attrs[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS])
    assert sys_instr == [{"type": "text", "content": "Be concise."}]

    # Verify events: started, inference operation details, ended
    logs = log_exporter.get_finished_logs()
    event_names = [log.log_record.event_name for log in logs]
    assert "gen_ai.client.realtime_session.started" in event_names
    assert "gen_ai.client.realtime_session.ended" in event_names
    assert (
        "gen_ai.client.inference.operation.details" in event_names
    )

    started_log = next(
        l.log_record
        for l in logs
        if l.log_record.event_name == "gen_ai.client.realtime_session.started"
    )
    assert started_log.severity_number == SeverityNumber.INFO
    assert started_log.attributes["gen_ai.realtime_session.id"] == "sess-voice-1"
    assert started_log.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"

    ended_log = next(
        l.log_record
        for l in logs
        if l.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended_log.severity_number == SeverityNumber.INFO
    assert ended_log.attributes["gen_ai.realtime_session.id"] == "sess-voice-1"

    # Verify metrics (including per-modality token usage)
    metrics_data = metric_reader.get_metrics_data()
    metric_names = {}
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                metric_names[m.name] = m

    assert "gen_ai.client.operation.duration" in metric_names
    assert "gen_ai.client.token.usage" in metric_names
    token_dp = list(metric_names["gen_ai.client.token.usage"].data.data_points)
    assert len(token_dp) == 2
    token_totals = {
        dp.attributes.get(GenAI.GEN_AI_TOKEN_TYPE): dp.sum for dp in token_dp
    }
    assert token_totals == {"input": 38, "output": 41}


@pytest.mark.asyncio
async def test_live_gemini_2_5_two_generation_tool_call(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    gen1_messages = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="What is the weather in Paris?"
                )
            )
        ),
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[
                    genai_types.FunctionCall(
                        id="call-weather-1",
                        name="get_weather",
                        args={"city": "Paris"},
                    )
                ]
            )
        ),
    ]
    gen2_messages = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x01\x02" * 500,
                                mime_type="audio/pcm;rate=24000",
                            )
                        )
                    ],
                ),
                output_transcription=genai_types.Transcription(
                    text="It is 18 degrees and sunny in Paris."
                ),
                turn_complete=True,
            ),
            usage_metadata=genai_types.UsageMetadata(
                prompt_token_count=90,
                response_token_count=45,
            ),
        ),
    ]
    fake_session = _FakeAsyncSession(
        gen1_messages + gen2_messages, session_id="sess-tool-25"
    )

    with patch.object(
        AsyncLive, "connect", _make_fake_connect(fake_session)
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.AUDIO],
                tools=[
                    genai_types.Tool(
                        function_declarations=[
                            genai_types.FunctionDeclaration(
                                name="get_weather",
                                description="Get weather for a city",
                            )
                        ]
                    )
                ],
            )
            async with live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x00" * 400,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                # Receive gen 1 (input transcription + tool_call)
                msg1 = await session._receive()
                assert msg1.server_content is not None
                msg2 = await session._receive()
                assert msg2.tool_call is not None

                # Send tool response
                await session.send_tool_response(
                    function_responses=[
                        genai_types.FunctionResponse(
                            id="call-weather-1",
                            name="get_weather",
                            response={"temp_c": 18, "condition": "sunny"},
                        )
                    ]
                )

                # Receive gen 2 (spoken answer + turn_complete)
                msg3 = await session._receive()
                assert msg3.server_content.turn_complete is True

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    span1, span2 = spans[0], spans[1]
    assert span1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    assert GenAI.GEN_AI_OUTPUT_TYPE not in span1.attributes
    s1_out = json.loads(span1.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert s1_out[0]["finish_reason"] == "tool_call"
    assert s1_out[0]["parts"][0]["type"] == "tool_call"
    assert s1_out[0]["parts"][0]["name"] == "get_weather"

    assert span2.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert span2.attributes[GenAI.GEN_AI_OUTPUT_TYPE] == "speech"
    s2_in = json.loads(span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert s2_in[0]["role"] == "tool"
    assert s2_in[0]["parts"][0]["type"] == "tool_call_response"
    assert s2_in[0]["parts"][0]["id"] == "call-weather-1"


@pytest.mark.asyncio
async def test_live_gemini_3_single_generation_tool_call(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    messages = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="What is the weather in Tokyo?"
                )
            )
        ),
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[
                    genai_types.FunctionCall(
                        id="call-tokyo-1",
                        name="get_weather",
                        args={"city": "Tokyo"},
                    )
                ]
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x01\x02" * 500,
                                mime_type="audio/pcm;rate=24000",
                            )
                        )
                    ],
                ),
                output_transcription=genai_types.Transcription(
                    text="It is 22 degrees in Tokyo."
                ),
                turn_complete=True,
            ),
            usage_metadata=genai_types.UsageMetadata(
                prompt_token_count=110,
                response_token_count=55,
            ),
        ),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-tool-31")

    with patch.object(
        AsyncLive, "connect", _make_fake_connect(fake_session)
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            async with live.connect(
                model="gemini-3.1-flash-live-preview",
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x00" * 400,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                await session._receive()
                await session._receive()
                await session.send_tool_response(
                    function_responses=[
                        genai_types.FunctionResponse(
                            id="call-tokyo-1",
                            name="get_weather",
                            response={"temp_c": 22},
                        )
                    ]
                )
                await session._receive()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    # Per latest PR #394, gen_ai.input.messages and gen_ai.output.messages
    # on a single-generation tool exchange omit intermediate tool call and
    # tool result events.
    in_msgs = json.loads(span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(in_msgs) == 1
    assert in_msgs[0]["role"] == "user"

    out_msgs = json.loads(span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert len(out_msgs) == 1
    assert out_msgs[0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_live_user_speech_and_interrupted_and_session_error(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    messages = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x01\x02" * 200,
                                mime_type="audio/pcm;rate=24000",
                            )
                        )
                    ],
                ),
                interrupted=True,
                turn_complete=True,
            )
        )
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-err-1")

    with patch.object(
        AsyncLive, "connect", _make_fake_connect(fake_session)
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            with pytest.raises(RuntimeError, match="websocket dropped"):
                async with live.connect(
                    model="gemini-2.5-flash-native-audio-preview-12-2025"
                ) as session:
                    await session.send_realtime_input(
                        activity_start=genai_types.ActivityStart()
                    )
                    await session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=b"\x00\x00" * 100,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                    await session.send_realtime_input(
                        activity_end=genai_types.ActivityEnd()
                    )
                    await session._receive()
                    raise RuntimeError("websocket dropped")

    spans = span_exporter.get_finished_spans()
    span_by_op = {
        s.attributes[GenAI.GEN_AI_OPERATION_NAME]: s for s in spans
    }
    assert "user_speech" in span_by_op
    assert "realtime_inference" in span_by_op

    speech_span = span_by_op["user_speech"]
    assert speech_span.name == "user_speech"
    assert speech_span.kind == SpanKind.INTERNAL
    assert speech_span.attributes["gen_ai.realtime_session.id"] == "sess-err-1"

    inf_span = span_by_op["realtime_inference"]
    assert inf_span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "interrupted",
    )

    logs = log_exporter.get_finished_logs()
    ended_log = next(
        l.log_record
        for l in logs
        if l.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended_log.severity_number == SeverityNumber.WARN
    assert ended_log.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"
