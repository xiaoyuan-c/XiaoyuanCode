from __future__ import annotations

import hashlib
import json
from typing import Literal

from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.tools.base import ToolResult


class RepeatedFailureGuard:
    # 只按全失败的完整模型轮次计数，为相同失败保留一次有限的重新规划机会
    def __init__(self, threshold: int = 3, grace_steps: int = 2) -> None:
        if threshold < 2 or grace_steps < 1:
            raise ValueError("failure threshold must be >= 2 and grace_steps must be >= 1")
        self._threshold = threshold
        self._grace_steps = grace_steps
        self._signature: tuple[str, ...] = ()
        self.count = 0
        self.guidance = ""
        self.detail = ""

    # 成功或新信息重置计数；相同失败达到阈值时提示重规划，宽限期耗尽时停止
    def observe(
        self, outcomes: list[tuple[ToolCallBlock, ToolResult]],
    ) -> Literal["replan", "stop"] | None:
        if not outcomes or any(not result.is_error for _, result in outcomes):
            self._signature = ()
            self.count = 0
            self.guidance = ""
            self.detail = ""
            return None

        signature = tuple(sorted({_failure_fingerprint(call, result) for call, result in outcomes}))
        if signature != self._signature:
            self._signature = signature
            self.count = 0
            self.guidance = ""
        self.count += 1
        names = ", ".join(sorted({call.name for call, _ in outcomes}))
        self.detail = (
            f"Stopped: {names} failed with identical arguments and errors for "
            f"{self.count} consecutive rounds, including {self._grace_steps} rounds "
            "after a replanning request. Review the recorded tool errors before continuing."
        )
        if self.count == self._threshold:
            self.guidance = (
                f"Repeated tool failure detected: {names} failed with identical arguments "
                f"and errors for {self.count} consecutive rounds. Replan now. "
                "Do not repeat the unchanged failing calls. Inspect the cause, change "
                "the arguments or approach, or explain why the task is blocked and end "
                f"the turn. If the same failures persist for {self._grace_steps} more "
                "rounds, the runtime will stop this task."
            )
            return "replan"
        if self.count >= self._threshold + self._grace_steps:
            return "stop"
        return None


# 忽略调用 ID 和 JSON 对象键顺序，只比较工具、参数、错误类型与原始错误内容
def _failure_fingerprint(call: ToolCallBlock, result: ToolResult) -> str:
    payload = json.dumps(
        [call.name, call.input, result.error_type, result.content],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
