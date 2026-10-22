"""
Agent Core - Connects Claude LLM with the Tool Registry.
Handles the full conversation loop with tool use.
"""

import json
import anthropic
from config import LLM_MODEL
from registry import ToolRegistry, ToolDefinition

from tools.google_drive import ALL_DRIVE_TOOLS
from tools.read_file import ALL_READ_FILE_TOOLS
from tools.memory import ALL_MEMORY_TOOLS

ALL_TOOLS: list[ToolDefinition] = ALL_DRIVE_TOOLS + ALL_READ_FILE_TOOLS + ALL_MEMORY_TOOLS

SYSTEM_PROMPT = """\
You are a powerful AI assistant with access to the following capabilities:

1. **Google Drive**: You can list all files and read their contents (supports many formats: PDF, DOCX, XLSX, PPTX, images, etc.).
2. **Long-term Memory**: You can save and search information across conversations using semantic search (RAG).

Guidelines:
- When the user asks to see files from Google Drive, use list_drive_files first, then read_drive_file for each file.
- Proactively save important information to memory (user preferences, key facts, task results).
- Before answering questions about past interactions, user preferences, or concepts from previously saved files, search_memory first.
- When the user asks to save/remember a Drive file's content, call save_drive_file_to_memory with its file ID. Never copy file content into save_memory yourself.
- read_drive_file returns at most 15000 chars; for questions about a long file, save it to memory and use search_memory.
- When showing a file's content, include the content itself in your reply, not just a summary.
- If a tool returns an error (e.g. missing permission), explain it to the user plainly.
- Always respond in the same language as the user's message.
- Be concise and helpful.
"""


class Agent:
    def __init__(self, service_api_key: str = "sk-admin-001"):
        self.client = anthropic.Anthropic()
        self.model = LLM_MODEL
        self.service_api_key = service_api_key
        self.conversation_history: list[dict] = []

        # Initialize Tool Registry with all tools
        self.registry = ToolRegistry()
        for tool in ALL_TOOLS:
            self.registry.register(tool)

    def get_tools_for_claude(self) -> list[dict]:
        return self.registry.list_tools()

    def run(self, user_message: str) -> str:
        """Blocking wrapper over stream(): returns only the final text."""
        final_response = ""
        for event in self.stream(user_message):
            if event["type"] == "done":
                final_response = event["text"]
        return final_response

    def stream(self, user_message: str):
        """
        Run the full agent loop, yielding events as they happen:
        1. Add user message to history
        2. Send to Claude with tools, yielding {"type": "text"} deltas
        3. If Claude uses tools → run through Registry pipeline, yielding {"type": "tool"}
        4. Loop until Claude gives final text response, then yield {"type": "done"}
        """
        print(f"\n{'#'*60}")
        print(f"  USER: {user_message}")
        print(f"{'#'*60}")

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        tools = self.get_tools_for_claude()

        while True:
            print(f"\n>>> Calling Claude LLM...")
            with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=self.conversation_history,
            ) as s:
                for delta in s.text_stream:
                    yield {"type": "text", "text": delta}
                response = s.get_final_message()

            print(f">>> Claude stop_reason: {response.stop_reason}")

            # Final response - anything but tool_use (end_turn, max_tokens, ...) ends the loop
            if response.stop_reason != "tool_use":
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                })

                text_parts = [
                    block.text for block in response.content
                    if hasattr(block, "text")
                ]
                final_response = "\n".join(text_parts)

                print(f"\n{'#'*60}")
                print(f"  ASSISTANT: {final_response[:500]}")
                if len(final_response) > 500:
                    print(f"  ... [truncated, total {len(final_response)} chars]")
                print(f"{'#'*60}\n")

                yield {"type": "done", "text": final_response}
                return

            # Process tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            tool_results = []
            tool_events = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    # Run through the 6-step Registry pipeline
                    result = self.registry.call(
                        tool_name=tool_name,
                        arguments=tool_input,
                        api_key=self.service_api_key,
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
                    tool_events.append({
                        "type": "tool",
                        "name": tool_name,
                        "status": "failed" if "error" in result else "ok",
                    })

            self.conversation_history.append({
                "role": "user",
                "content": tool_results,
            })
            # Yield only after history is consistent: a client disconnect here must not leave a tool_use without its tool_result
            yield from tool_events

    def clear_history(self):
        self.conversation_history = []
        print("[Agent] Conversation history cleared.")

    def get_audit_log(self) -> list[dict]:
        return self.registry.get_audit_log()
