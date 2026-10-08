from __future__ import annotations

import pytest

from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.loop_guard import RepeatedFailureGuard
from kama_claude.core.tools.base import ToolResult


# 构造完整失败结果，不含工具内部重试，使测试聚焦跨模型轮次的行为
def _failure(
    params: dict[str, object] | None = None, message: str = "not found", kind: str = "runtime_error",
) -> tuple[ToolCallBlock, ToolResult]:
    return (
        ToolCallBlock(id="call", name="read_file", input=params or {"path": "missing.py"}),
        ToolResult(content=message, is_error=True, error_type=kind),
    )


# 功能：连续三轮相同失败先请求重新规划，再允许两轮尝试，最终提前停止
# 设计：按轮提交失败结果，检查准确的动作序列，避免在第一次失败时过早终止
def test_warn_then_stop_after_grace() -> None:
    guard = RepeatedFailureGuard()

    assert [guard.observe([_failure()]) for _ in range(5)] == [None, None, "replan", None, "stop"]
    assert "Replan now" in guard.guidance
    assert "5 consecutive rounds" in guard.detail


# 功能：同轮重复调用不增加轮次计数，调用 ID 与对象键顺序变化不绕过检测
# 设计：用重复结果和重排参数提交三轮，保持语义相同但序列化顺序不同
def test_round_count_ignores_duplicates_and_json_key_order() -> None:
    guard = RepeatedFailureGuard()
    first = _failure({"path": "x.py", "options": {"a": 1, "b": 2}})
    other = _failure({"options": {"b": 2, "a": 1}, "path": "x.py"})
    other[0].id = "different-call-id"

    assert guard.observe([first] * 4) is None
    assert guard.count == 1
    assert guard.observe([other]) is None
    assert guard.observe([first]) == "replan"


# 功能：参数、工具、错误内容或错误类型变化时重新计数并清除旧的恢复指令
# 设计：先达到警告阈值，再逐一变更真正用于决策的信息，验证新尝试不会沿用旧计数
@pytest.mark.parametrize("changed", ["params", "name", "message", "kind"])
def test_new_information_resets_failure_streak(changed: str) -> None:
    guard = RepeatedFailureGuard()
    for _ in range(3):
        guard.observe([_failure()])
    new = _failure()
    if changed == "params":
        new[0].input = {"path": "another.py"}
    elif changed == "name":
        new[0].name = "list_dir"
    elif changed == "message":
        new[1].content = "permission denied"
    else:
        new[1].error_type = "schema_error"

    assert guard.observe([new]) is None
    assert guard.count == 1
    assert guard.guidance == ""


# 功能：成功结果、正常状态轮询或无工具调用清除失败计数
# 设计：先制造重规划警告，再模拟成功结果或空轮次，检查旧告警不污染后续行动
@pytest.mark.parametrize("mode", ["success", "poll", "empty", "mixed"])
def test_progress_and_polling_do_not_trigger_guard(mode: str) -> None:
    guard = RepeatedFailureGuard()
    for _ in range(3):
        guard.observe([_failure()])
    content = "still running" if mode == "poll" else "new information"
    success = (ToolCallBlock(id="ok", name="agent_result", input={}), ToolResult(content=content))
    outcomes = [] if mode == "empty" else [success]
    if mode == "mixed":
        outcomes.append(_failure())

    for _ in range(10):
        assert guard.observe(outcomes) is None
    assert guard.count == 0
    assert guard.guidance == ""


# 功能：多工具失败集合顺序变化仍能识别相同轮次
# 设计：交替交换两个失败调用的顺序，确认批量调用不能仅靠重排逃过检测
def test_failed_batch_order_does_not_matter() -> None:
    guard = RepeatedFailureGuard(threshold=2)
    a, b = _failure(), _failure({"path": "other.py"})

    assert guard.observe([a, b]) is None
    assert guard.observe([b, a]) == "replan"


# 功能：无效阈值在初始化时被拒绝
# 设计：覆盖不能构成重复的阈值和没有重规划机会的宽限值，避免配置导致首次失败即终止
@pytest.mark.parametrize("threshold,grace", [(1, 2), (3, 0)])
def test_invalid_guard_thresholds_rejected(threshold: int, grace: int) -> None:
    with pytest.raises(ValueError):
        RepeatedFailureGuard(threshold, grace)
