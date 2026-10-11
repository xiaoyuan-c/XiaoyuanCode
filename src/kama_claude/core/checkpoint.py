from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kama_claude.core.context import ExecutionContext
    from kama_claude.core.llm.types import ToolCallBlock


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RunCheckpoint:
    run_id: str
    session_id: str
    goal: str
    phase: str
    status: str
    step: int
    max_steps: int
    messages: list[dict[str, Any]]
    reason: str | None = None
    result: str = ""
    system_prompt_override: str | None = None
    tool_whitelist: list[str] | None = None
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    last_stop_reason: str | None = None
    resume_count: int = 0
    thread_committed: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    version: int = 1

    # 将 checkpoint 转为可原子写入 JSON 的普通字典
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "goal": self.goal,
            "phase": self.phase,
            "status": self.status,
            "step": self.step,
            "max_steps": self.max_steps,
            "messages": self.messages,
            "reason": self.reason,
            "result": self.result,
            "system_prompt_override": self.system_prompt_override,
            "tool_whitelist": self.tool_whitelist,
            "pending_tool_calls": self.pending_tool_calls,
            "last_stop_reason": self.last_stop_reason,
            "resume_count": self.resume_count,
            "thread_committed": self.thread_committed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    # 从 checkpoint.json 的普通字典恢复运行快照
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunCheckpoint:
        messages = data.get("messages", [])
        pending = data.get("pending_tool_calls", [])
        whitelist = data.get("tool_whitelist")
        if not isinstance(messages, list) or not isinstance(pending, list):
            raise ValueError("invalid checkpoint message fields")
        if whitelist is not None and not isinstance(whitelist, list):
            raise ValueError("invalid checkpoint tool whitelist")
        return cls(
            version=int(data.get("version", 1)),
            run_id=str(data["run_id"]),
            session_id=str(data.get("session_id", "")),
            goal=str(data["goal"]),
            phase=str(data["phase"]),
            status=str(data["status"]),
            step=int(data.get("step", 0)),
            max_steps=int(data["max_steps"]),
            messages=[dict(message) for message in messages],
            reason=str(data["reason"]) if data.get("reason") is not None else None,
            result=str(data.get("result", "")),
            system_prompt_override=(
                str(data["system_prompt_override"])
                if data.get("system_prompt_override") is not None
                else None
            ),
            tool_whitelist=(
                [str(name) for name in whitelist] if whitelist is not None else None
            ),
            pending_tool_calls=[dict(call) for call in pending],
            last_stop_reason=(
                str(data["last_stop_reason"])
                if data.get("last_stop_reason") is not None
                else None
            ),
            resume_count=int(data.get("resume_count", 0)),
            thread_committed=bool(data.get("thread_committed", False)),
            created_at=str(data.get("created_at", _now())),
            updated_at=str(data.get("updated_at", _now())),
        )


class RunCheckpointStore:
    # 初始化单个 run 的 checkpoint 文件存储
    def __init__(self, path: Path) -> None:
        self._path = path

    # 返回 checkpoint 文件是否存在
    def exists(self) -> bool:
        return self._path.exists()

    # 从磁盘读取并校验 checkpoint
    def load(self) -> RunCheckpoint:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("checkpoint root must be an object")
        return RunCheckpoint.from_dict(data)

    # 使用同目录临时文件和 os.replace 原子覆盖 checkpoint
    def save(self, checkpoint: RunCheckpoint) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2) + "\n"
        with temp_path.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, self._path)


class RunCheckpointer:
    # 初始化运行期 checkpoint 写入器并保留创建时间和恢复次数
    def __init__(
        self,
        store: RunCheckpointStore,
        *,
        session_id: str,
        system_prompt_override: str | None,
        tool_whitelist: list[str] | None,
        previous: RunCheckpoint | None = None,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._system_prompt_override = system_prompt_override
        self._tool_whitelist = list(tool_whitelist) if tool_whitelist is not None else None
        self._created_at = previous.created_at if previous is not None else _now()
        self._resume_count = previous.resume_count if previous is not None else 0
        self._thread_committed = previous.thread_committed if previous is not None else False
        self._pending_tool_calls = (
            restore_pending_tool_calls(previous) if previous is not None else []
        )
        self._last_stop_reason = previous.last_stop_reason if previous is not None else None

    # 将当前上下文和待确认工具调用保存为最新 checkpoint
    def save(
        self,
        context: ExecutionContext,
        phase: str,
        pending_tool_calls: list[ToolCallBlock] | None = None,
        last_stop_reason: str | None = None,
        *,
        thread_committed: bool | None = None,
    ) -> None:
        if thread_committed is not None:
            self._thread_committed = thread_committed
        if pending_tool_calls is not None:
            self._pending_tool_calls = list(pending_tool_calls)
        if last_stop_reason is not None:
            self._last_stop_reason = last_stop_reason
        pending = [
            {"id": call.id, "name": call.name, "input": call.input}
            for call in self._pending_tool_calls
        ]
        checkpoint = RunCheckpoint(
            run_id=context.run_id,
            session_id=self._session_id,
            goal=context.goal,
            phase=phase,
            status=context.status,
            step=context.step,
            max_steps=context.max_steps,
            messages=context.messages,
            reason=context.reason,
            result=context.result,
            system_prompt_override=self._system_prompt_override,
            tool_whitelist=self._tool_whitelist,
            pending_tool_calls=pending,
            last_stop_reason=self._last_stop_reason,
            resume_count=self._resume_count,
            thread_committed=self._thread_committed,
            created_at=self._created_at,
            updated_at=_now(),
        )
        self._store.save(checkpoint)

    # 将恢复次数加一，供 run.resumed 事件和后续 checkpoint 使用
    def mark_resumed(self) -> int:
        self._resume_count += 1
        self._thread_committed = False
        return self._resume_count


# 将 checkpoint 中的工具调用字典恢复为 ToolCallBlock 列表
def restore_pending_tool_calls(checkpoint: RunCheckpoint) -> list[ToolCallBlock]:
    from kama_claude.core.llm.types import ToolCallBlock

    calls: list[ToolCallBlock] = []
    for item in checkpoint.pending_tool_calls:
        tool_input = item.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        calls.append(
            ToolCallBlock(
                id=str(item.get("id", "")),
                name=str(item.get("name", "")),
                input={str(key): value for key, value in tool_input.items()},
            )
        )
    return calls


# 为执行状态不确定且尚无结果的工具调用补齐错误结果，避免恢复时盲目重放副作用
def repair_interrupted_tool_calls(
    context: ExecutionContext, pending_tool_calls: list[ToolCallBlock]
) -> list[str]:
    completed_ids: set[str] = set()
    for message in context.messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                completed_ids.add(str(block.get("tool_use_id", "")))

    repaired: list[str] = []
    for call in pending_tool_calls:
        if not call.id or call.id in completed_ids:
            continue
        context.add_tool_result(
            call.id,
            "Error: the previous process stopped while this tool call was pending, so its "
            "final state is unknown. Inspect the current state before deciding whether to retry.",
            is_error=True,
        )
        repaired.append(call.id)
    return repaired
