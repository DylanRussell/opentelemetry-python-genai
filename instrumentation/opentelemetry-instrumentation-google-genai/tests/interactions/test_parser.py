# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
import unittest.mock

from google.genai.types import Content, Part

from opentelemetry.instrumentation.google_genai.interactions import (
    _interactions_input_to_messages,
    _interactions_response_to_messages,
)
from opentelemetry.util.genai.types import (
    Reasoning,
    ServerToolCall,
    ServerToolCallResponse,
    Text,
    ToolCallRequest,
    ToolCallResponse,
)


class TestInteractionsParser(unittest.TestCase):
    def test_input_to_messages_none(self) -> None:
        self.assertEqual(_interactions_input_to_messages(None), [])

    def test_input_to_messages_str(self) -> None:
        messages = _interactions_input_to_messages("Hello world")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Hello world")

    def test_input_to_messages_content_object(self) -> None:
        content = Content(parts=[Part(text="Hello content")])
        messages = _interactions_input_to_messages(content)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Hello content")

    def test_input_to_messages_steps_list_user_input(self) -> None:
        steps = [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "Hello step"}],
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Hello step")

    def test_input_to_messages_steps_list_model_output(self) -> None:
        steps = [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "Hello response"}],
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Hello response")

    def test_input_to_messages_steps_list_thought(self) -> None:
        steps = [
            {
                "type": "thought",
                "summary": [{"text": "First thought"}, {"text": "Second thought"}],
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Reasoning)
        self.assertEqual(messages[0].parts[0].content, "First thought\nSecond thought")

    def test_input_to_messages_steps_list_tool_call_request(self) -> None:
        steps = [
            {
                "type": "function_call",
                "id": "call-123",
                "name": "calc",
                "arguments": {"x": 5},
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], ToolCallRequest)
        self.assertEqual(messages[0].parts[0].id, "call-123")
        self.assertEqual(messages[0].parts[0].name, "calc")
        self.assertEqual(messages[0].parts[0].arguments, {"x": 5})

    def test_input_to_messages_steps_list_tool_call_response(self) -> None:
        steps = [
            {
                "type": "function_result",
                "call_id": "call-123",
                "result": {"val": 10},
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "tool")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], ToolCallResponse)
        self.assertEqual(messages[0].parts[0].id, "call-123")
        self.assertEqual(messages[0].parts[0].response, {"val": 10})

    def test_input_to_messages_steps_list_server_tool_call(self) -> None:
        steps = [
            {
                "type": "code_execution_call",
                "id": "code-123",
                "arguments": {"code": "print(1)"},
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], ServerToolCall)
        self.assertEqual(messages[0].parts[0].id, "code-123")
        self.assertEqual(messages[0].parts[0].name, "code_execution_call")
        self.assertEqual(messages[0].parts[0].server_tool_call, {"code": "print(1)"})

    def test_input_to_messages_steps_list_server_tool_call_response(self) -> None:
        steps = [
            {
                "type": "code_execution_result",
                "call_id": "code-123",
                "result": {"output": "1"},
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "tool")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], ServerToolCallResponse)
        self.assertEqual(messages[0].parts[0].id, "code-123")
        self.assertEqual(messages[0].parts[0].server_tool_call_response, {"output": "1"})

    def test_response_to_messages(self) -> None:
        mock_step_1 = unittest.mock.MagicMock()
        mock_step_1.type = "model_output"

        mock_part = unittest.mock.MagicMock()
        mock_part.text = "Model response text"
        mock_step_1.content = [mock_part]

        mock_step_2 = unittest.mock.MagicMock()
        mock_step_2.type = "model_output"
        mock_step_2.content = [unittest.mock.MagicMock(text="Second response")]

        mock_interaction = unittest.mock.MagicMock()
        mock_interaction.steps = [mock_step_1, mock_step_2]

        messages = _interactions_response_to_messages(mock_interaction)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(messages[0].finish_reason, "stop")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Model response text")
