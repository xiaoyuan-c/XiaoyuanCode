from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.tools.base import BaseTool, ToolResult


# Bash 仅以实际命令作为身份，忽略等待时长和执行时会被丢弃的额外参数
def _operation_key(call: ToolCallBlock) -> str:
    params = dict(call.input)
    if call.name == "bash":
        params = {"command": call.input.get("command")}
    return json.dumps([call.name, params], sort_keys=True, ensure_ascii=False)


@dataclass
class TimeoutRecovery:
    pending: ToolCallBlock | None = None
    cleanup_confirmed: bool | None = None
    evidence: dict[str, str] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    timeout_counts: dict[str, int] = field(default_factory=dict)
    retry_budget: dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""

    # 检查前置条件：未确认超时状态时仅允许只读检查与恢复确认工具
    def check(self, call: ToolCallBlock, tool: BaseTool) -> str | None:
        if self.stop_reason:
            return "Recovery is unresolved. End the task and report the recorded state."
        if _operation_key(call) in self.completed:
            return "This operation was confirmed completed. Do not execute it again."
        if self.pending and not tool.read_only and tool.name != ResolveToolTimeoutTool.name:
            return (
                f"Operation {self.pending.id} timed out; its effects are unconfirmed. "
                "Use read-only tools to inspect the actual state, then call "
                "resolve_tool_timeout before another state-changing operation."
            )
        if self.retry_budget.get(_operation_key(call)) == 0:
            return "The single confirmed retry for this operation has been used. Do not repeat it."
        return None

    # 在权限检查之后消费确认重试额度，工具内部重试不能增加实际执行次数
    def reserve_retry(self, call: ToolCallBlock) -> bool:
        key = _operation_key(call)
        if self.retry_budget.get(key) == 1:
            self.retry_budget[key] = 0
            return True
        return False

    # 保存超时与之后的只读检查结果，失败检查和超时之前的结果不能作为确认依据
    def observe(self, call: ToolCallBlock, result: ToolResult, tool: BaseTool | None) -> None:
        if tool is None or tool.name == ResolveToolTimeoutTool.name:
            return
        if result.is_error and result.error_type == "timeout" and not tool.read_only:
            self.pending = call
            self.cleanup_confirmed = result.execution_stopped
            if tool.name == "bash" and self.cleanup_confirmed is None:
                self.cleanup_confirmed = False
            self.evidence.clear()
            key = _operation_key(call)
            self.timeout_counts[key] = self.timeout_counts.get(key, 0) + 1
        elif self.pending and tool.read_only and not result.is_error:
            self.evidence[call.id] = json.dumps(
                {"tool": call.name, "params": call.input, "result": result.content},
                ensure_ascii=False,
            )

    # 根据带实际检查依据的结论恢复执行；不明、部分完成或反复超时直接停止
    def resolve(self, params: ResolveTimeoutParams) -> ToolResult:
        if self.pending is None or params.tool_use_id != self.pending.id:
            return ToolResult("No matching unresolved timeout.", True, "schema_error")
        if params.state in ("partial", "unknown"):
            self.stop_reason = (
                f"Operation {self.pending.id} ({self.pending.name}) has {params.state} "
                f"effects after timeout. {params.explanation}"
            )
            return ToolResult(self.stop_reason)
        if not params.evidence_ids or any(i not in self.evidence for i in params.evidence_ids):
            return ToolResult(
                "Use successful read-only checks performed after this timeout, and pass "
                "their tool_use IDs as evidence_ids. Earlier or failed calls do not count.",
                True, "schema_error",
            )
        key = _operation_key(self.pending)
        if self.cleanup_confirmed is False:
            self.stop_reason = "Timed-out process cleanup could not be confirmed; do not retry."
            return ToolResult(self.stop_reason)
        if params.state == "not_applied" and self.timeout_counts[key] > 1:
            self.stop_reason = "Operation timed out again after a confirmed retry; stop recovery."
            return ToolResult(self.stop_reason)
        if params.state == "completed":
            self.completed.add(key)
        else:
            self.retry_budget[key] = 1
        detail = json.dumps({
            "tool_use_id": self.pending.id, "state": params.state,
            "explanation": params.explanation,
            "evidence": [self.evidence[i] for i in params.evidence_ids],
        }, ensure_ascii=False)
        self.pending = None
        self.evidence.clear()
        return ToolResult(
            f"State confirmation recorded: {detail}. "
            + ("Do not repeat the completed operation." if params.state == "completed"
               else "One retry is permitted; normal tool permissions still apply.")
        )

    # 将待确认操作保留在系统提示中，避免上下文压缩或 Skill 覆盖丢失恢复要求
    def instruction(self) -> str:
        if self.pending is None:
            return ""
        operation = json.dumps({
            "tool_use_id": self.pending.id, "tool": self.pending.name,
            "params": self.pending.input, "execution_stopped": self.cleanup_confirmed,
            "successful_check_ids": list(self.evidence),
        }, ensure_ascii=False)
        return (
            "A state-changing operation timed out. Its effects are UNKNOWN, not rolled back. "
            f"Operation: {operation}. Only read-only checks and resolve_tool_timeout are "
            "allowed until its state is confirmed. Inspect the affected files or task state. "
            "Call resolve_tool_timeout with completed or not_applied only when supported "
            "by actual post-timeout check results. If checks are unavailable, inconclusive, "
            "or show partial effects, report unknown or partial and stop. Do not claim "
            "success while recovery is unresolved. This requirement survives compaction."
        )


class ResolveTimeoutParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tool_use_id: str
    state: Literal["completed", "not_applied", "partial", "unknown"]
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)


class ResolveToolTimeoutTool(BaseTool):
    name = "resolve_tool_timeout"
    params_model = ResolveTimeoutParams
    description = (
        "Confirm the state of a timed-out operation after read-only inspection. "
        "Supply the timed-out tool_use ID, actual successful post-timeout check IDs, "
        "and an explanation linking their results to the operation. completed prevents "
        "repetition; not_applied permits one retry; partial or unknown stops the task. "
        "Do not infer no effects from timeout alone. This tool records your assessment; "
        "it does not undo changes or independently verify the meaning of the evidence."
    )
    input_schema: dict[str, object] = ResolveTimeoutParams.model_json_schema()

    # 绑定本次执行的超时恢复状态，主 Agent 和子 Agent 各自维护
    def __init__(self, recovery: TimeoutRecovery) -> None:
        self._recovery = recovery

    # 校验检查依据并记录恢复结论，不自行重新执行超时操作
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return self._recovery.resolve(ResolveTimeoutParams.model_validate(params))
