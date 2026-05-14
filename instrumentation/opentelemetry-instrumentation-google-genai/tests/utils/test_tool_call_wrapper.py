# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import unittest

from google.genai import types as genai_types

from opentelemetry._logs import get_logger_provider
from opentelemetry.instrumentation.google_genai import tool_call_wrapper
from opentelemetry.metrics import get_meter_provider
from opentelemetry.trace import get_tracer_provider
from opentelemetry.util.genai.handler import TelemetryHandler

from ..common import otel_mocker


class TestCase(unittest.TestCase):
    def setUp(self):
        self._otel = otel_mocker.OTelMocker()
        self._otel.install()
        self._otel_wrapper = TelemetryHandler(
            tracer_provider=get_tracer_provider(),
            logger_provider=get_logger_provider(),
            meter_provider=get_meter_provider(),
        )

    @property
    def otel(self):
        return self._otel

    @property
    def otel_wrapper(self):
        return self._otel_wrapper

    def wrap(self, tool_or_tools, capture_content_on_span=True):
        return tool_call_wrapper.wrapped_tool(
            tool_or_tools,
            self.otel_wrapper,
            capture_content_on_span=capture_content_on_span,
        )

    def test_wraps_none(self):
        result = self.wrap(None)
        self.assertIsNone(result)

    def test_wraps_single_tool_function(self):
        def somefunction():
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        self.assertEqual(
            span.attributes["gen_ai.operation.name"], "execute_tool"
        )
        self.assertEqual(span.attributes["gen_ai.tool.name"], "somefunction")

    def test_wraps_multiple_tool_functions_as_list(self):
        def somefunction():
            pass

        def otherfunction():
            pass

        wrapped_functions = self.wrap([somefunction, otherfunction])
        wrapped_somefunction = wrapped_functions[0]
        wrapped_otherfunction = wrapped_functions[1]
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        somefunction()
        otherfunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_otherfunction()
        self.otel.assert_has_span_named("execute_tool otherfunction")

    def test_wraps_multiple_tool_functions_as_dict(self):
        def somefunction():
            pass

        def otherfunction():
            pass

        wrapped_functions = self.wrap(
            {"somefunction": somefunction, "otherfunction": otherfunction}
        )
        wrapped_somefunction = wrapped_functions["somefunction"]
        wrapped_otherfunction = wrapped_functions["otherfunction"]
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        somefunction()
        otherfunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_otherfunction()
        self.otel.assert_has_span_named("execute_tool otherfunction")

    def test_wraps_async_tool_function(self):
        async def somefunction():
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        asyncio.run(somefunction())
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        asyncio.run(wrapped_somefunction())
        self.otel.assert_has_span_named("execute_tool somefunction")

    def test_preserves_tool_dict(self):
        tool_dict = genai_types.ToolDict()
        wrapped_tool_dict = self.wrap(tool_dict)
        self.assertEqual(tool_dict, wrapped_tool_dict)

    def test_does_not_have_description_if_no_doc_string(self):
        def somefunction():
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        self.assertNotIn("gen_ai.tool.description", span.attributes)

    def test_has_description_if_doc_string_present(self):
        def somefunction():
            """An example tool call function."""

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        self.assertEqual(
            span.attributes["gen_ai.tool.description"],
            "An example tool call function.",
        )

    def test_handles_primitive_int_arg(self):
        def somefunction(arg=None):
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction(12345)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction(12345)
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(arguments["code.function.parameters.arg.type"], "int")
        self.assertEqual(
            arguments["code.function.parameters.arg.value"], 12345
        )

    def test_handles_primitive_string_arg(self):
        def somefunction(arg=None):
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction("a string value")
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction("a string value")
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(arguments["code.function.parameters.arg.type"], "str")
        self.assertEqual(
            arguments["code.function.parameters.arg.value"],
            "a string value",
        )

    def test_handles_dict_arg(self):
        def somefunction(arg=None):
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction({"key": "value"})
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction({"key": "value"})
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(
            arguments["code.function.parameters.arg.type"], "dict"
        )
        self.assertEqual(
            arguments["code.function.parameters.arg.value"],
            {'key': 'value'}
        )

    def test_handles_primitive_list_arg(self):
        def somefunction(arg=None):
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction([1, 2, 3])
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction([1, 2, 3])
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(
            arguments["code.function.parameters.arg.type"], "list"
        )
        self.assertEqual(
            arguments["code.function.parameters.arg.value"],
            [1, 2, 3],
        )

    def test_handles_heterogenous_list_arg(self):
        def somefunction(arg=None):
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction([123, "abc"])
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction([123, "abc"])
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(
            arguments["code.function.parameters.arg.type"], "list"
        )
        self.assertEqual(
            arguments["code.function.parameters.arg.value"],
            [123, "abc"],
        )

    def test_handle_with_different_capture_content_on_span_config(self):
        def somefunction(arg=None):
            return arg

        for capture_content_on_span in [False, True]:
            with self.subTest(
                f"capture_content_on_span: {capture_content_on_span}"
            ):
                self.setUp()
                wrapped_somefunction = self.wrap(
                    somefunction,
                    capture_content_on_span=capture_content_on_span,
                )
                wrapped_somefunction(12345)
                span = self.otel.get_span_named("execute_tool somefunction")

                if not capture_content_on_span:
                    self.assertNotIn(
                        "gen_ai.tool.call.arguments",
                        span.attributes,
                    )
                    self.assertNotIn(
                        "gen_ai.tool.call.result",
                        span.attributes,
                    )
                else:
                    self.assertIn(
                        "code.function.parameters.arg.value",
                        span.attributes["gen_ai.tool.call.arguments"],
                    )
                    self.assertEqual(
                        span.attributes["gen_ai.tool.call.result"],
                        "12345",
                    )
                self.tearDown()

    def test_function_that_throws_exception(self):
        def somefunction(arg=None):
            raise Exception("Something went wrong")

        wrapped_somefunction = self.wrap(
            somefunction,
            capture_content_on_span=True,
        )
        try:
            wrapped_somefunction(12345)
        except Exception:
            span = self.otel.get_span_named("execute_tool somefunction")
            self.assertEqual(span.attributes["error.type"], "Exception")
