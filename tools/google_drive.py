"""
Google Drive Tools - List files and read file content.
"""

from registry.models import ToolDefinition
from services import drive_service
from services.file_reader import read_file


# ============================================================
# Tool 1: LIST DRIVE FILES
# ============================================================

def list_drive_files(folder_id: str = "") -> dict:
    """List all files in Google Drive."""
    fid = folder_id if folder_id else None
    files = drive_service.list_files(folder_id=fid)
    return {
        "total_files": len(files),
        "files": files,
    }


list_files_tool = ToolDefinition(
    name="list_drive_files",
    description=(
        "List all files in Google Drive. "
        "Returns file names, IDs, types, sizes, and modification times. "
        "Optionally provide a folder_id to list files in a specific folder."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "folder_id": {
                "type": "string",
                "description": "Optional Google Drive folder ID to list files from. Leave empty for default folder.",
            },
        },
        "required": [],
    },
    required_scopes=["drive:read"],
    handler=list_drive_files,
)


# ============================================================
# Tool 2: READ DRIVE FILE
# ============================================================

def read_drive_file(file_id: str) -> dict:
    """Download a file from Google Drive and read its content."""
    import os
    download = drive_service.download_file(file_id=file_id)
    temp_path = download["temp_path"]
    try:
        result = read_file(temp_path)
    finally:
        os.unlink(temp_path)
    return {
        "file_id": download["file_id"],
        "file_name": download["file_name"],
        "mime_type": download["mime_type"],
        "content": result["content"],
    }


read_file_tool = ToolDefinition(
    name="read_drive_file",
    description=(
        "Read the content of a file from Google Drive by its file ID. "
        "Supports many formats: PDF, DOCX, XLSX, PPTX, images, Google Docs/Sheets/Slides, text files, and more. "
        "Content is converted to Markdown using MarkItDown. "
        "Use list_drive_files first to get file IDs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "The Google Drive file ID to read.",
            },
        },
        "required": ["file_id"],
    },
    required_scopes=["drive:read"],
    handler=read_drive_file,
)


# ============================================================
# Tool 3: SAVE DRIVE FILE TO MEMORY
# ============================================================

def save_drive_file_to_memory(file_id: str) -> dict:
    """Download a Drive file and save its FULL content to memory, server-side.
    The content never passes through the LLM, so it is neither truncated nor paraphrased."""
    import os
    from tools.memory import save_memory
    download = drive_service.download_file(file_id=file_id)
    temp_path = download["temp_path"]
    try:
        content = read_file(temp_path, max_chars=None)["content"]
    finally:
        os.unlink(temp_path)
    saved = save_memory(content, category=f"file:{download['file_name']}")
    return {
        "file_name": download["file_name"],
        "total_chars": len(content),
        "chunks": saved["chunks"],
        "category": saved["category"],
    }


save_file_to_memory_tool = ToolDefinition(
    name="save_drive_file_to_memory",
    description=(
        "Save the FULL content of a Google Drive file to long-term memory by its file ID. "
        "Use this (not save_memory) whenever the user asks to save/remember/index a file: "
        "the file is read and stored server-side with no length limit."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "The Google Drive file ID to save.",
            },
        },
        "required": ["file_id"],
    },
    required_scopes=["drive:read", "memory:write"],
    handler=save_drive_file_to_memory,
)


ALL_DRIVE_TOOLS = [list_files_tool, read_file_tool, save_file_to_memory_tool]
