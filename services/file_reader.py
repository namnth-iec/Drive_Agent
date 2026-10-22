"""
File Reader Service - Convert files to Markdown using MarkItDown.
"""

import os
from markitdown import MarkItDown

MAX_CHARS = 15000


def read_file(file_path: str, max_chars: int | None = MAX_CHARS) -> dict:
    """Read a local file and convert its content to Markdown using MarkItDown.
    max_chars=None returns the full content (for indexing, not for the LLM context)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")

    content = MarkItDown().convert(file_path).text_content or ""
    if max_chars is not None and len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[... truncated at {max_chars} of {len(content)} chars]"
    return {"file_name": os.path.basename(file_path), "content": content}
