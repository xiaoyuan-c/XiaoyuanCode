from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.registry import ToolRegistry

# --- stubs -------------------------------------------------------------------


class _MockProvider:
    """Returns canned responses in order; raises exc immediately if given."""

    def __init__(
        self,
        responses: list[LlmResponse],
        exc: BaseException | None = None,
    ) -> None:
        self._responses = iter(responses)
        self._exc = exc

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        if self._exc is not None:
            raise self._exc
        return next(self._responses)


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes msg"
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content=str(params["msg"]))


class _FailTool(BaseTool):
    name = "fail"
    description = "Always raises"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise RuntimeError("tool error")


# --- helpers -----------------------------------------------------------------


def _ctx(max_steps: int = 5) -> ExecutionContext:
    return ExecutionContext(run_id="r1", goal="test goal", max_steps=max_steps)


def _tc(name: str = "echo", inp: dict[str, object] | None = None, uid: str = "t1") -> ToolCallBlock:
    return ToolCallBlock(id=uid, name=name, input=inp or {"msg": "hi"})


def _make_loop(
    provider: _MockProvider,
    registry: ToolRegistry | None = None,
    bus: EventBus | None = None,
) -> tuple[AgentLoop, EventBus]:
    b = bus or EventBus()
    return AgentLoop(provider, registry or ToolRegistry(), b), b  # type: ignore[arg-type]


async def _events(bus: EventBus) -> list[BaseModel]:
    collected: list[BaseModel] = []

    async def _h(e: BaseModel) -> None:
        collected.append(e)

    bus.subscribe(_h)
    return collected


# --- tests -------------------------------------------------------------------


# 功能：验证 LLM 返回 end_turn 时 loop 将 context 标记为 success
# 设计：单步 provider 直接返回 end_turn，最简正常路径，确认 loop 的基本终止逻辑
async def test_end_turn_marks_success() -> None:
    provider = _MockProvider([LlmResponse(stop_reason="end_turn", text="done")])
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "success"
    assert ctx.step == 1


# 功能：验证达到 max_steps 时 loop 以 exceeded_max_steps 原因将 context 标记为 failed
# 设计：设置 max_steps=2 + 无限 tool_use provider，同时验证 step 数量和失败原因，确认计数器与终止逻辑联动正确
async def test_max_steps_marks_failed() -> None:
    tc = _tc("unknown", {})
    provider = _MockProvider([LlmResponse(stop_reason="tool_use", tool_calls=[tc])] * 10)
    loop, _ = _make_loop(provider)
    ctx = _ctx(max_steps=2)
    await loop.run(ctx)
    assert ctx.status == "failed"
    assert ctx.reason == "exceeded_max_steps"
    assert ctx.step == 2


# 功能：验证"调工具 → end_turn"的两步路径最终标记为 success
# 设计：provider 返回 [tool_use, end_turn] 序列，注册真实 EchoTool，覆盖最常见的正常工作路径
async def test_tool_use_then_end_turn_marks_success() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="end_turn", text="summary"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "success"
    assert ctx.step == 2


# 功能：验证工具结果按 Anthropic 格式（tool_result user 消息）追加到消息历史
# 设计：检查 messages[2]（tool_result 所在位置），断言 tool_use_id 和 content，确认 loop 正确调用了 context.add_tool_result
async def test_tool_result_appended_to_context() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc(inp={"msg": "hello"})]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    # messages: [goal, assistant(tool_use), user(tool_result), assistant(end_turn)]
    tool_result_msg = ctx.messages[2]
    assert tool_result_msg["role"] == "user"
    block = tool_result_msg["content"][0]  # type: ignore[index]
    assert block["tool_use_id"] == "t1"
    assert block["content"] == "hello"


# 功能：验证工具失败时 loop 不终止，而是将错误追加上下文让 LLM 重新决策
# 设计：工具始终 raise + provider 第二步返回 end_turn，确认 loop 最终到达 success；这是 agent 区别于普通脚本的核心特性
async def test_tool_failure_loop_continues_to_success() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc("fail", {})]),
        LlmResponse(stop_reason="end_turn", text="handled error"),
    ])
    registry = ToolRegistry()
    registry.register(_FailTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "success"
    assert ctx.step == 2


# 功能：验证工具失败的错误信息以 is_error=True 追加进上下文，让 LLM 能感知工具调用失败
# 设计：检查 tool_result block 中的 is_error 标记，与 test_tool_failure_loop_continues_to_success 互补
async def test_tool_failure_result_is_error_in_context() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc("fail", {})]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_FailTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    tool_result_msg = ctx.messages[2]
    block = tool_result_msg["content"][0]  # type: ignore[index]
    assert block.get("is_error") is True


# 功能：验证收到 CancelledError 时 loop 将 context 标记为 cancelled 后继续上抛 CancelledError
# 设计：用 pytest.raises 捕获 CancelledError，同时检查 context.status，确认优雅退出行为：先记录状态，再传播取消信号
async def test_cancelled_error_marks_failed_and_reraises() -> None:
    provider = _MockProvider([], exc=asyncio.CancelledError())
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    with pytest.raises(asyncio.CancelledError):
        await loop.run(ctx)
    assert ctx.status == "failed"
    assert ctx.reason == "cancelled"


# 功能：验证 LLM 调用异常被捕获并标记为 llm_error，不向上传播
# 设计：provider 抛 RuntimeError，确认 loop 不崩溃、context 状态为 failed/llm_error，异常被正确吸收
async def test_llm_api_error_marks_failed() -> None:
    provider = _MockProvider([], exc=RuntimeError("api error"))
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "failed"
    assert ctx.reason == "llm_error"


# 功能：验证每个步骤都发布 step.started 和 step.finished 事件
# 设计：注入 bus + 事件收集器，检查事件类型集合，确认步骤级事件的可观测性（S2 TUI 依赖这两个事件显示进度）
async def test_step_started_and_finished_events_published() -> None:
    bus = EventBus()
    events = await _events(bus)
    provider = _MockProvider([LlmResponse(stop_reason="end_turn")])
    loop, _ = _make_loop(provider, bus=bus)
    ctx = _ctx()
    await loop.run(ctx)
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert "step.started" in types
    assert "step.finished" in types


# 功能：验证多步执行后 step 计数器正确累积到步数总量
# 设计：三步序列 [tool_use, tool_use, end_turn]，确认 step==3，排除计数器初始化错误或某步未递增的情况
async def test_step_counter_increments_across_steps() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx(max_steps=10)
    await loop.run(ctx)
    assert ctx.step == 3
    assert ctx.status == "success"


# 功能：验证 LLM 文本响应以正确的 content block 格式追加到消息历史
# 设计：检查 messages[1] 的 role 和 content block 结构，确认 loop 构造的 assistant 消息符合 Anthropic 格式
async def test_assistant_message_blocks_added_to_context() -> None:
    provider = _MockProvider([LlmResponse(stop_reason="end_turn", text="answer")])
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    await loop.run(ctx)
    assistant_msg = ctx.messages[1]
    assert assistant_msg["role"] == "assistant"
    blocks = assistant_msg["content"]
    assert blocks[0]["type"] == "text"  # type: ignore[index]
    assert blocks[0]["text"] == "answer"  # type: ignore[index]


# 功能：相同无效调用跨轮重复时先把重规划指令交给模型，再提前终止并保留配对工具结果
# 设计：模拟八轮不存在工具的请求，检查第五轮停止、第四次调用能看到指令，以及最后一轮仍完成结果配对
async def test_repeated_failure_prompts_replan_then_stops() -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc("unknown", {}, uid=f"t{i}")])
        for i in range(8)
    ])
    ctx = _ctx(max_steps=20)
    bus = EventBus()
    events = await _events(bus)

    await AgentLoop(provider, ToolRegistry(), bus).run(ctx)

    assert ctx.status == "failed"
    assert ctx.reason == "repeated_tool_failure"
    assert ctx.step == provider.chat.call_count == 5
    assert "Replan now" not in provider.chat.call_args_list[2].kwargs["system"]
    assert "Replan now" in provider.chat.call_args_list[3].kwargs["system"]
    assert "unknown" in ctx.result
    assert ctx.messages[-1]["content"][0]["tool_use_id"] == "t4"
    assert getattr(events[-1], "type") == "step.finished"


# 功能：重规划后改用成功工具会清除警告，任务能继续正常完成
# 设计：三轮未知工具触发警告，第四轮执行真实 EchoTool，第五轮结束并检查系统提示中的恢复指令已移除
async def test_success_after_replan_clears_guidance() -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        *[LlmResponse(stop_reason="tool_use", tool_calls=[_tc("unknown")]) for _ in range(3)],
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="end_turn", text="Recovered"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    ctx = _ctx(max_steps=20)

    await AgentLoop(provider, registry, EventBus()).run(ctx)

    assert ctx.status == "success"
    assert "Replan now" in provider.chat.call_args_list[3].kwargs["system"]
    assert "Runtime Recovery Instructions" not in provider.chat.call_args_list[4].kwargs["system"]


# 功能：重复成功的状态查询不会被当作死循环
# 设计：六次返回同样的 still running 成功结果后结束，确认超过失败阈值也不触发警告
async def test_successful_polling_is_allowed() -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        *[LlmResponse(stop_reason="tool_use", tool_calls=[_tc(inp={"msg": "still running"})]) for _ in range(6)],
        LlmResponse(stop_reason="end_turn", text="Done"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    ctx = _ctx(max_steps=20)

    await AgentLoop(provider, registry, EventBus()).run(ctx)

    assert ctx.status == "success"
    assert ctx.step == 7
    assert all("Replan now" not in call.kwargs["system"] for call in provider.chat.call_args_list)


# 功能：连续同批的重复调用按轮计数而不是按工具调用次数计数
# 设计：每轮发三个相同失败请求，两轮后正常结束，验证未提前警告且所有工具结果均配对
async def test_many_failures_in_one_round_do_not_exhaust_guard() -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        *[LlmResponse(stop_reason="tool_use", tool_calls=[_tc("unknown", uid=f"t{i}") for i in range(3)]) for _ in range(2)],
        LlmResponse(stop_reason="end_turn", text="Blocked"),
    ])
    ctx = _ctx(max_steps=20)

    await AgentLoop(provider, ToolRegistry(), EventBus()).run(ctx)

    assert ctx.status == "success"
    assert all("Replan now" not in call.kwargs["system"] for call in provider.chat.call_args_list)
    assert len(ctx.messages[2]["content"]) == 3


# 功能：压缩消息和 Skill 系统提示覆盖都不能丢失重规划要求或失败计数
# 设计：模拟每轮压缩清空工具历史，并设置专用系统指令，验证仍在第五轮停止且模型看到了恢复指导
async def test_guard_survives_compaction_and_prompt_override() -> None:
    from kama_claude.core.llm.types import UsageStats

    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        LlmResponse(
            stop_reason="tool_use", tool_calls=[_tc("unknown")],
            usage=UsageStats(input_tokens=10, output_tokens=10, context_pct=0.9),
        ) for _ in range(8)
    ])
    compactor = MagicMock()

    # 模拟有损压缩，仅修改消息列表，不保留原错误或提示文本
    async def compress(context: ExecutionContext, provider: object) -> None:
        context.messages = [{"role": "user", "content": "summary"}]

    compactor.compact = AsyncMock(side_effect=compress)
    ctx = _ctx(max_steps=20)
    ctx.system_prompt_override = "Specialized skill rules"

    await AgentLoop(provider, ToolRegistry(), EventBus(), compactor=compactor).run(ctx)

    assert ctx.reason == "repeated_tool_failure"
    assert ctx.step == 5
    assert compactor.compact.call_count == 4
    assert "Specialized skill rules" in provider.chat.call_args_list[3].kwargs["system"]
    assert "Replan now" in provider.chat.call_args_list[3].kwargs["system"]


# 功能：工具内部重试不会消耗重复失败检测的模型轮次额度
# 设计：两轮工具失败各执行三次内部尝试，共六次失败仍可正常结束且不触发重规划
async def test_internal_retries_do_not_count_as_model_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kama_claude.core.tools.invocation._RETRY_BASE_S", 0)
    tool = _FailTool()
    tool.invoke = AsyncMock(side_effect=RuntimeError("tool error"))
    registry = ToolRegistry()
    registry.register(tool)
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        *[LlmResponse(stop_reason="tool_use", tool_calls=[_tc("fail", {})]) for _ in range(2)],
        LlmResponse(stop_reason="end_turn", text="Blocked"),
    ])
    ctx = _ctx(max_steps=20)

    await AgentLoop(provider, registry, EventBus()).run(ctx)

    assert tool.invoke.call_count == 6
    assert ctx.status == "success"
    assert all("Replan now" not in call.kwargs["system"] for call in provider.chat.call_args_list)
