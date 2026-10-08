from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kama_claude.core.config import KamaConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from kama_claude.core.loop import AgentLoop
from kama_claude.core.runner import AgentRunner
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.read_file import ReadFileTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.core.tools.timeout_recovery import (
    ResolveTimeoutParams,
    ResolveToolTimeoutTool,
    TimeoutRecovery,
)


# 构造工具请求，独立调用 ID 用于验证真实状态检查的引用关系
def _call(name: str, uid: str, **params: object) -> ToolCallBlock:
    return ToolCallBlock(id=uid, name=name, input=params)


# 构造包含执行清理状态的超时结果
def _timeout(stopped: bool | None = True) -> ToolResult:
    return ToolResult("timed out; effects unknown", True, "timeout", execution_stopped=stopped)


# 初始化待确认命令和超时后的成功读取，用于测试恢复状态转换
def _pending(stopped: bool | None = True) -> TimeoutRecovery:
    recovery = TimeoutRecovery()
    recovery.observe(_call("bash", "slow", command="write output"), _timeout(stopped), BashTool())
    recovery.observe(_call("read_file", "check", path="output"), ToolResult("actual state"), ReadFileTool())
    return recovery


# 功能：没有超时后的成功检查依据，不能确认完成或批准重试
# 设计：覆盖空依据、旧调用和失败读取，防止模型仅凭说明解除限制
@pytest.mark.parametrize("evidence_ids", [[], ["before"], ["failed"]])
def test_resolution_requires_post_timeout_successful_checks(evidence_ids: list[str]) -> None:
    recovery = TimeoutRecovery()
    read = ReadFileTool()
    recovery.observe(_call("read_file", "before", path="x"), ToolResult("old"), read)
    recovery.observe(_call("bash", "slow", command="write output"), _timeout(), BashTool())
    recovery.observe(_call("read_file", "failed", path="x"), ToolResult("missing", True), read)
    result = recovery.resolve(ResolveTimeoutParams(
        tool_use_id="slow", state="not_applied", evidence_ids=evidence_ids,
        explanation="No change found",
    ))
    assert result.is_error
    assert recovery.pending is not None
    assert recovery.check(_call("write_file", "retry", path="x"), WriteFileTool()) is not None


# 功能：确认完成后阻止相同操作重放，仅修改超时时长不能绕过检查
# 设计：确认依据指向成功读取，再用新调用 ID 和不同等待时间重复同一命令
def test_completed_operation_cannot_be_repeated() -> None:
    recovery = _pending()
    assert not recovery.resolve(ResolveTimeoutParams(
        tool_use_id="slow", state="completed", evidence_ids=["check"], explanation="Output is complete",
    )).is_error
    assert recovery.pending is None
    assert recovery.check(_call("bash", "again", command="write output", timeout=120, ignored="x"), BashTool())


# 功能：部分完成、状态不明或进程清理失败时停止恢复而不放行重试
# 设计：前三种不确定性分别触发停止分支，防止把已有副作用当作可安全重试
@pytest.mark.parametrize("state,stopped", [
    ("partial", True), ("unknown", True), ("not_applied", False), ("not_applied", None),
])
def test_uncertain_state_stops_recovery(state: str, stopped: bool | None) -> None:
    recovery = _pending(stopped)
    recovery.resolve(ResolveTimeoutParams.model_validate({
        "tool_use_id": "slow", "state": state,
        "evidence_ids": ["check"], "explanation": "Cannot safely continue",
    }))
    assert recovery.stop_reason
    assert recovery.pending is not None


# 功能：同一操作确认后仅允许一次重试，重复超时不能不断清除限制
# 设计：两次超时分别提供新检查结果，第二次确认没有生效仍必须停止
def test_confirmed_retry_is_bounded() -> None:
    recovery = _pending()
    params = ResolveTimeoutParams(
        tool_use_id="slow", state="not_applied", evidence_ids=["check"], explanation="No effect",
    )
    recovery.resolve(params)
    assert recovery.pending is None
    assert recovery.check(_call("bash", "slow2", command="write output"), BashTool()) is None
    recovery.observe(_call("bash", "slow2", command="write output"), _timeout(), BashTool())
    recovery.observe(_call("read_file", "check2", path="output"), ToolResult("no effect"), ReadFileTool())
    recovery.resolve(params.model_copy(update={"tool_use_id": "slow2", "evidence_ids": ["check2"]}))
    assert recovery.stop_reason


# 功能：只读工具超时不引入修改不确定性，也不会替换已待确认的修改操作
# 设计：在待确认状态下再制造读取超时，验证原操作和有效依据仍保留
def test_read_only_timeout_does_not_replace_pending_operation() -> None:
    recovery = _pending()
    recovery.observe(_call("read_file", "read-timeout", path="output"), _timeout(), ReadFileTool())
    assert recovery.pending is not None and recovery.pending.id == "slow"
    assert "check" in recovery.evidence


# 功能：同轮超时后的再次写入被执行端拦截，未确认状态不能作为成功任务结束
# 设计：真实 runner 配合模拟超时返回，检查工具调用次数、结果配对和持久化退出原因
async def test_timeout_blocks_same_batch_mutation_and_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invoke = AsyncMock(return_value=_timeout())
    monkeypatch.setattr(BashTool, "invoke", invoke)
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        LlmResponse(stop_reason="tool_use", tool_calls=[
            _call("bash", "slow", command="write output"),
            _call("bash", "again", command="changed command"),
        ]),
        LlmResponse(stop_reason="end_turn", text="Done"),
    ])
    result = await AgentRunner(KamaConfig(), provider=provider, runs_dir=tmp_path).run_and_capture("work")
    assert invoke.call_count == 1
    assert result.status == "failed" and result.reason == "timeout_state_unconfirmed"
    assert "Timeout State Verification" in provider.chat.call_args_list[1].kwargs["system"]
    results = provider.chat.call_args_list[1].kwargs["messages"][2]["content"]
    assert [block["tool_use_id"] for block in results] == ["slow", "again"]
    assert all(block["is_error"] for block in results)
    log_file = next(tmp_path.rglob("events.jsonl"))
    assert "recovery_required" in log_file.read_text(encoding="utf-8")
    assert "timeout_state_unconfirmed" in log_file.read_text(encoding="utf-8")


# 功能：检查实际文件并确认未生效后能重试，确认已完成时则跳过重复操作
# 设计：模拟命令超时、真实文件读取、确认和结束，覆盖恢复工具在 runner 中的注册与执行
@pytest.mark.parametrize("completed", [False, True])
async def test_file_inspection_resolves_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed: bool) -> None:
    target = tmp_path / "state.txt"
    target.write_text("done" if completed else "not applied", encoding="utf-8")
    invoke = AsyncMock(side_effect=[_timeout(), ToolResult("done")])
    monkeypatch.setattr(BashTool, "invoke", invoke)
    provider = MagicMock()
    responses = [
        LlmResponse(stop_reason="tool_use", tool_calls=[_call("bash", "slow", command="write output")]),
        LlmResponse(stop_reason="tool_use", tool_calls=[_call("read_file", "check", path=str(target))]),
        LlmResponse(stop_reason="tool_use", tool_calls=[_call(
            "resolve_tool_timeout", "confirm", tool_use_id="slow",
            state="completed" if completed else "not_applied", evidence_ids=["check"],
            explanation="The actual file confirms the state",
        )]),
    ]
    if not completed:
        responses.append(LlmResponse(stop_reason="tool_use", tool_calls=[
            _call("bash", "retry", command="write output"),
        ]))
    responses.append(LlmResponse(stop_reason="end_turn", text="Done"))
    provider.chat = AsyncMock(side_effect=responses)
    result = await AgentRunner(KamaConfig(), provider=provider, runs_dir=tmp_path / "runs").run_and_capture("work")
    assert result.status == "success"
    assert invoke.call_count == (1 if completed else 2)
    assert "Timeout State Verification" not in provider.chat.call_args_list[-1].kwargs["system"]


# 功能：确认后的重试只执行一次，运行时错误不会触发内部重试或再次请求重放
# 设计：超时后真实读取检查并确认，再模拟重试失败及重复请求，断言总计仅两次实际执行
async def test_confirmed_retry_has_one_execution_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "state"
    target.write_text("not applied", encoding="utf-8")
    invoke = AsyncMock(side_effect=[_timeout(), ToolResult("failed", True, "runtime_error")])
    monkeypatch.setattr(BashTool, "invoke", invoke)
    calls = [
        _call("bash", "slow", command="write output"),
        _call("read_file", "check", path=str(target)),
        _call("resolve_tool_timeout", "confirm", tool_use_id="slow", state="not_applied", evidence_ids=["check"], explanation="No effect"),
        _call("bash", "retry", command="write output"),
        _call("bash", "repeat", command="write output", timeout=120),
    ]
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        *[LlmResponse(stop_reason="tool_use", tool_calls=[call]) for call in calls],
        LlmResponse(stop_reason="end_turn", text="Cannot finish"),
    ])
    config = KamaConfig()
    config.agent.max_steps = 10
    await AgentRunner(config, provider=provider, runs_dir=tmp_path / "runs").run_and_capture("work")
    assert invoke.call_count == 2
    blocks = provider.chat.call_args_list[-1].kwargs["messages"][-2]["content"]
    assert blocks[0]["tool_use_id"] == "repeat"
    assert "single confirmed retry" in blocks[0]["content"]


# 功能：压缩消息和角色指令覆盖不会清除超时限制，状态不明时立即停止后续写入
# 设计：每轮压缩替换全部历史，再提交 unknown 确认，验证恢复控制独立于模型历史
async def test_unknown_resolution_stops_after_compaction() -> None:
    context = ExecutionContext("r", "work", 20, system_prompt_override="Role rules")
    registry = ToolRegistry()
    bash = BashTool()
    bash.invoke = AsyncMock(return_value=_timeout())
    registry.register(bash)
    registry.register(ResolveToolTimeoutTool(context.timeout_recovery))
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        LlmResponse(
            stop_reason="tool_use", tool_calls=[_call("bash", "slow", command="write output")],
            usage=UsageStats(input_tokens=10, output_tokens=10, context_pct=0.9),
        ),
        LlmResponse(stop_reason="tool_use", tool_calls=[
            _call("resolve_tool_timeout", "confirm", tool_use_id="slow", state="unknown", explanation="No checks available"),
            _call("bash", "retry", command="write output"),
        ]),
    ])
    compactor = MagicMock()

    # 模拟有损压缩，不保留原始工具调用消息
    async def compress(ctx: ExecutionContext, _: object) -> None:
        ctx.messages = [{"role": "user", "content": "summary"}]

    compactor.compact = AsyncMock(side_effect=compress)
    await AgentLoop(provider, registry, EventBus(), compactor=compactor).run(context)
    assert context.reason == "timeout_state_unconfirmed"
    assert context.step == 2 and bash.invoke.call_count == 1
    system = provider.chat.call_args_list[1].kwargs["system"]
    assert "Role rules" in system and "Timeout State Verification" in system
    assert "slow" in system


# 功能：白名单不含确认工具时保持限制，不通过恢复机制扩大可用工具范围
# 设计：只允许命令工具，超时后直接结束，确认恢复工具未注册且任务不能被报告为成功
async def test_whitelist_does_not_expand_for_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BashTool, "invoke", AsyncMock(return_value=_timeout()))
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        LlmResponse(stop_reason="tool_use", tool_calls=[_call("bash", "slow", command="work")]),
        LlmResponse(stop_reason="end_turn", text="Cannot inspect"),
    ])
    result = await AgentRunner(KamaConfig(), provider=provider, runs_dir=tmp_path).run_and_capture(
        "work", tool_whitelist=["bash"],
    )
    assert result.reason == "timeout_state_unconfirmed"
    assert [tool["name"] for tool in provider.chat.call_args_list[1].kwargs["tool_schemas"]] == ["bash"]
