# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import timeit
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Final

from opentelemetry._logs import Logger
from opentelemetry.context import Context, get_value
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._instruments import _Instruments
from opentelemetry.util.genai._invocation import (
    Error,
    GenAIInvocation,
    _ContextData,
)
from opentelemetry.util.genai.completion_hook import (
    CompletionHook,
    _NoOpCompletionHook,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
)
from opentelemetry.util.types import AttributeValue

WORKFLOW_CONTEXT_KEY: Final[str] = "opentelemetry.genai.workflow.context"


@dataclass
class WorkflowData(_ContextData):
    """Typed data passed from inner workflow invocations to the outer invocation."""

    workflow_name: str | None = None
    conversation_id: str | None = None
    input_messages: Sequence[InputMessage] | None = None
    output_messages: Sequence[OutputMessage] | None = None
    attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )
    metric_attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )


class WorkflowInvocation(GenAIInvocation):
    """
    Represents a predetermined sequence of operations (e.g. agent, LLM, tool,
    and retrieval invocations). A workflow groups multiple operations together,
    accepting input(s) and producing final output(s).

    Use handler.workflow(name) rather than constructing this directly.
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        name: str | None,
        *,
        content_capturing_mode: ContentCapturingMode | None = None,
        start_span: bool = True,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
    ) -> None:
        """Use handler.workflow(name) rather than calling this directly."""
        _operation_name = GenAI.GenAiOperationNameValues.INVOKE_WORKFLOW.value
        start_attributes: dict[str, AttributeValue] = (
            {GenAI.GEN_AI_WORKFLOW_NAME: name} if name is not None else {}
        )
        self.data: WorkflowData = WorkflowData(
            workflow_name=name,
            conversation_id=conversation_id,
            input_messages=[],
            output_messages=[],
        )
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=f"{_operation_name} {name}" if name else _operation_name,
            span_kind=SpanKind.INTERNAL,
            start_attributes=start_attributes,
            context=context,
            conversation_id=conversation_id,
            content_capturing_mode=content_capturing_mode,
            start_span=start_span,
            _attach_to_context=_attach_to_context,
            attributes=self.data.attributes,
            metric_attributes=self.data.metric_attributes,
            context_key=WORKFLOW_CONTEXT_KEY,
            dataclass_class_object=WorkflowData,
        )
        self._name: str | None = name
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes

    @property
    def workflow_name(self) -> str | None:
        return self.data.workflow_name

    @property
    def input_messages(self) -> list[InputMessage]:
        if isinstance(self.data.input_messages, list):
            return self.data.input_messages
        messages = (
            list(self.data.input_messages) if self.data.input_messages else []
        )
        self.data.input_messages = messages
        return messages

    @input_messages.setter
    def input_messages(self, value: Sequence[InputMessage]) -> None:
        self.data.input_messages = value

    @property
    def output_messages(self) -> list[OutputMessage]:
        if isinstance(self.data.output_messages, list):
            return self.data.output_messages
        messages = (
            list(self.data.output_messages)
            if self.data.output_messages
            else []
        )
        self.data.output_messages = messages
        return messages

    @output_messages.setter
    def output_messages(self, value: Sequence[OutputMessage]) -> None:
        self.data.output_messages = value

    def enrich_from_context(self, data: WorkflowData) -> None:
        """Enrich invocation attributes from context data published by inner invocations.

        Outer (root) attributes take precedence over inner values. Inner
        invocations never override content capture fields.
        """
        input_messages = self.data.input_messages
        output_messages = self.data.output_messages

        self.data.merge(data, overwrite=False)

        self.data.input_messages = input_messages
        self.data.output_messages = output_messages

    def _get_messages_for_span(self) -> dict[str, AttributeValue]:
        if not self._should_capture_content_on_span:
            return {}
        optional_attrs = (
            (
                GenAI.GEN_AI_INPUT_MESSAGES,
                gen_ai_json_dumps([asdict(m) for m in self.input_messages])
                if self.input_messages
                else None,
            ),
            (
                GenAI.GEN_AI_OUTPUT_MESSAGES,
                gen_ai_json_dumps([asdict(m) for m in self.output_messages])
                if self.output_messages
                else None,
            ),
        )
        return {
            key: value for key, value in optional_attrs if value is not None
        }

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        attrs: dict[str, AttributeValue] = {}
        if self.data.workflow_name is not None:
            attrs[GenAI.GEN_AI_WORKFLOW_NAME] = self.data.workflow_name
        attrs.update(self.metric_attributes)
        return attrs

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        ctx_data = get_value(WORKFLOW_CONTEXT_KEY, context=self._span_context)
        if isinstance(ctx_data, WorkflowData):
            self.enrich_from_context(ctx_data)
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        attributes: dict[str, AttributeValue] = self._get_messages_for_span()
        conv_id = self.conversation_id or self.data.conversation_id
        if conv_id is not None:
            attributes[GenAI.GEN_AI_CONVERSATION_ID] = conv_id
        if self.data.workflow_name is not None:
            attributes[GenAI.GEN_AI_WORKFLOW_NAME] = self.data.workflow_name
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._call_completion_hook(
            inputs=self.input_messages,
            outputs=self.output_messages,
        )
        self._record_metrics()

    def _record_metrics(self) -> None:
        duration_seconds = max(
            timeit.default_timer() - self._monotonic_start_s,
            0.0,
        )
        self._instruments.invoke_workflow_duration.record(
            duration_seconds,
            attributes=self._get_metric_attributes(),
            context=self._span_context,
        )


class SuppressedWorkflowInvocation(WorkflowInvocation):
    """Represents a workflow invocation running inside an active workflow context.

    Suppresses span creation and metrics. On stop or fail, publishes its
    attributes to the active workflow context.
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        name: str | None,
        *,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
    ) -> None:
        super().__init__(
            tracer,
            instruments,
            logger,
            _NoOpCompletionHook(),
            name,
            content_capturing_mode=ContentCapturingMode.NO_CONTENT,
            start_span=False,
            context=context,
            _attach_to_context=_attach_to_context,
            conversation_id=conversation_id,
        )

    def publish_to_context(self, data: WorkflowData) -> None:
        """Publish invocation attributes to the active workflow context."""
        self.data.conversation_id = self.conversation_id
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        data.merge(self.data, overwrite=True)

    def _finish(self, error: Error | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        ctx_data = get_value(WORKFLOW_CONTEXT_KEY, context=self._span_context)
        if isinstance(ctx_data, WorkflowData):
            self.publish_to_context(ctx_data)

    def _apply_finish(self, error: Error | None = None) -> None:
        pass
