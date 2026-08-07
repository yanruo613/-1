import json
import os
import unittest
from unittest.mock import Mock, patch

from llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    InternChatClient,
)


class InternChatClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(
            os.environ,
            {
                "INTERN_API_KEY": "test-token",
                "INTERN_MODEL": "test-model",
            },
            clear=True,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    @staticmethod
    def successful_response(message):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": message}]}
        return response

    @patch("llm_client.requests.post")
    def test_keeps_existing_defaults_and_text_response(self, post: Mock) -> None:
        post.return_value = self.successful_response(
            {"role": "assistant", "content": "hello"}
        )
        messages = [{"role": "user", "content": "hi"}]

        result = InternChatClient().chat(messages)

        self.assertEqual(result, "hello")
        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(
            payload,
            {
                "model": "test-model",
                "messages": messages,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
            },
        )

    @patch("llm_client.requests.post")
    def test_forwards_defaults_and_per_call_args(self, post: Mock) -> None:
        post.return_value = self.successful_response(
            {"role": "assistant", "content": "done"}
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        client = InternChatClient(
            default_args={
                "thinking_mode": True,
                "temperature": 0.6,
                "top_p": 0.9,
            },
            n=2,
        )

        client.chat(
            [{"role": "user", "content": "1 + 1"}],
            temperature=0.1,
            max_tokens=128,
            thinking_mode=False,
            tools=tools,
            tool_choice="auto",
            top_p=0.8,
        )

        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["max_tokens"], 128)
        self.assertIs(payload["thinking_mode"], False)
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["n"], 2)

    @patch("llm_client.requests.post")
    def test_returns_complete_message_for_tool_calls(self, post: Mock) -> None:
        message = {
            "role": "assistant",
            "content": "calculate",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression": "1 + 1"}',
                    },
                }
            ],
        }
        post.return_value = self.successful_response(message)

        result = InternChatClient().chat(
            [{"role": "user", "content": "1 + 1"}],
            tools=[],
        )

        self.assertEqual(result, message)

    @patch("llm_client.requests.post")
    def test_accepts_multimodal_and_tool_messages(self, post: Mock) -> None:
        post.return_value = self.successful_response(
            {"role": "assistant", "content": "done"}
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "calculate",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {
                "role": "tool",
                "content": '{"result": 2}',
                "tool_call_id": "call-1",
            },
        ]

        InternChatClient().chat(messages)

        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["messages"], messages)


if __name__ == "__main__":
    unittest.main()
