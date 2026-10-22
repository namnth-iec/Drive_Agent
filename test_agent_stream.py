"""Self-check for Agent.stream() with a fake LLM client (no network). Run: python test_agent_stream.py"""

from types import SimpleNamespace as NS
from unittest.mock import patch

import agent as agent_module


class FakeStream:
    def __init__(self, deltas, message):
        self.text_stream, self._message = deltas, message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


turns = [
    FakeStream(["Checking"], NS(stop_reason="tool_use", content=[
        NS(type="text", text="Checking"),
        NS(type="tool_use", name="no_such_tool", input={}, id="t1"),
    ])),
    FakeStream(["Hel", "lo"], NS(stop_reason="end_turn", content=[NS(type="text", text="Hello")])),
]
fake_client = NS(messages=NS(stream=lambda **kw: turns.pop(0)))

with patch.object(agent_module.anthropic, "Anthropic", return_value=fake_client):
    a = agent_module.Agent()
events = list(a.stream("hi"))

assert [e["type"] for e in events] == ["text", "tool", "text", "text", "done"], events
assert events[1] == {"type": "tool", "name": "no_such_tool", "status": "failed"}
assert events[-1]["text"] == "Hello"
# history: user, assistant(tool_use), user(tool_result), assistant(final)
assert [m["role"] for m in a.conversation_history] == ["user", "assistant", "user", "assistant"]
print("OK")
