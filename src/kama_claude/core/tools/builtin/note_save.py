from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from kama_claude.core.session.store import SessionStore
from kama_claude.core.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from kama_claude.core.context import ExecutionContext


class NoteSaveParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    content: str


class NoteSaveTool(BaseTool):
    params_model = NoteSaveParams
    name = "note_save"
    description = (
        "Save a concise fact, user constraint, decision, or task progress checkpoint "
        "to this session's notes. Save hard constraints when learned and update progress "
        "after milestones, including completed work and remaining tasks. "
        "Notes survive context compaction and are visible in this and future runs "
        "of the same session. For changing progress, explicitly supersede the older note."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The durable fact or decision to remember.",
            },
        },
        "required": ["content"],
    }

    # 绑定当前 session 与 run，使工具调用能写入对应 notes.md
    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        run_id: str,
        context: ExecutionContext | None = None,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._run_id = run_id
        self._context = context

    # 将非空 content 追加到 session notes.md
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        content = NoteSaveParams.model_validate(params).content.strip()
        if not content:
            return ToolResult(
                content="empty content",
                is_error=True,
                error_type="runtime_error",
            )
        self._store.append_note(self._session_id, content, self._run_id)
        if self._context is not None:
            self._context.session_notes = self._store.read_notes(self._session_id)
        return ToolResult(content="saved")
