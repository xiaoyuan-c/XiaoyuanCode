from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kama_claude.core.bus.events import StepFinishedEvent, StepStartedEvent
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.loop_guard import RepeatedFailureGuard
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from kama_claude.core.compact.compactor import Compactor
    from kama_claude.core.permissions.manager import PermissionManager


log = logging.getLogger(__name__)

type CheckpointCallback = Callable[
    [ExecutionContext, str, list[ToolCallBlock] | None, str | None], None
]


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class AgentLoop:
    # 初始化模型、工具、事件、权限、压缩及重复失败保护配置
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        bus: EventBus,
        *,
        permission_manager: PermissionManager | None = None,
        compactor: Compactor | None = None,
        compact_threshold: float = 0.80,
        session_id: str = "",
        repeat_failure_threshold: int = 3,
        replan_grace_steps: int = 2,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._bus = bus
        self._permission_manager = permission_manager
        self._compactor = compactor
        self._compact_threshold = compact_threshold
        self._session_id = session_id
        self._repeat_failure_threshold = repeat_failure_threshold
        self._replan_grace_steps = replan_grace_steps
        self._checkpoint_callback = checkpoint_callback

    # 将当前执行阶段交给 runner 的持久化回调，未配置时保持原有行为
    def _checkpoint(
        self,
        context: ExecutionContext,
        phase: str,
        pending_tool_calls: list[ToolCallBlock] | None = None,
        last_stop_reason: str | None = None,
    ) -> None:
        if self._checkpoint_callback is not None:
            self._checkpoint_callback(
                context, phase, pending_tool_calls, last_stop_reason
            )

    # 驱动 plan→act→observe 循环直到上下文终止；CancelledError 向上传播
    async def run(self, context: ExecutionContext) -> None:
        guard = RepeatedFailureGuard(self._repeat_failure_threshold, self._replan_grace_steps)
        while not context.is_done():
            context.step += 1
            self._checkpoint(context, "step_started", [])
            await self._bus.publish(
                StepStartedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )

            # [plan] call LLM — API errors terminate the run
            try:
                response = await self._provider.chat(
                    messages=context.messages,
                    tool_schemas=self._registry.tool_schemas(),
                    bus=self._bus,
                    run_id=context.run_id,
                    step=context.step,
                    system=context.system_prompt(
                        "You are a helpful AI assistant. "
                        "Use the available tools to complete the user's goal. "
                        "When the goal is fully achieved, respond with a final answer "
                        "and do not call any more tools."
                    ),
                )
            except asyncio.CancelledError:
                context.mark_failed("cancelled")
                self._checkpoint(context, "cancelled")
                raise
            except Exception:
                logging.getLogger(__name__).exception(
                    "LLM call failed run_id=%s step=%d", context.run_id, context.step
                )
                context.mark_failed("llm_error")
                self._checkpoint(context, "llm_failed")
                break

            # 恢复说明只用于中断后的第一次模型调用，避免后续每轮重复注入
            context.recovery_instruction = ""

            # [observe] append assistant content blocks to context
            # thinking blocks must come first and be preserved verbatim for extended thinking mode
            blocks: list[dict[str, object]] = list(response.thinking_blocks)
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                )
            context.add_assistant_message(blocks)
            self._checkpoint(
                context,
                "model_response",
                list(response.tool_calls),
                response.stop_reason,
            )

            # [act] execute each requested tool; errors become tool results so loop continues
            outcomes: list[tuple[ToolCallBlock, ToolResult]] = []
            if response.stop_reason == "tool_use":
                for index, tc in enumerate(response.tool_calls):
                    self._checkpoint(
                        context,
                        "tool_started",
                        list(response.tool_calls[index:]),
                        response.stop_reason,
                    )
                    result = await invoke_tool(
                        self._registry, tc, self._bus, context.run_id,
                        permission_manager=self._permission_manager,
                        session_id=self._session_id,
                        timeout_recovery=context.timeout_recovery,
                    )
                    context.timeout_recovery.observe(tc, result, self._registry.get(tc.name))
                    context.add_tool_result(tc.id, result.content, is_error=result.is_error)
                    outcomes.append((tc, result))
                    self._checkpoint(
                        context,
                        "tool_result",
                        list(response.tool_calls[index + 1:]),
                        response.stop_reason,
                    )
            elif response.stop_reason == "max_tokens" and response.tool_calls:
                # Output token limit hit mid-tool-call; input is incomplete.
                # Add synthetic error results so the conversation stays balanced.
                for tc in response.tool_calls:
                    context.add_tool_result(
                        tc.id,
                        "Error: output token limit reached before this tool call "
                        "could be completed. "
                        "Please break the task into smaller steps and try again.",
                        is_error=True,
                    )
                self._checkpoint(
                    context, "tool_result", [], response.stop_reason
                )

            # 整轮工具结果配对完成后再检查，内部重试和同轮重复调用不增加失败轮数
            recovery_action = guard.observe(outcomes)
            context.recovery_instruction = guard.guidance
            if recovery_action == "replan":
                log.warning(
                    "Repeated tool failure; requesting replan run_id=%s step=%d",
                    context.run_id, context.step,
                )

            # 正常结束和步数上限保持原有优先级，再检查重复失败提前终止
            if context.timeout_recovery.stop_reason or (
                response.stop_reason == "end_turn" and context.timeout_recovery.pending
            ):
                context.result = (
                    context.timeout_recovery.stop_reason
                    or "Timed-out operation state is unconfirmed. " + (response.text or "")
                )
                context.mark_failed("timeout_state_unconfirmed")
            elif response.stop_reason == "end_turn":
                context.result = response.text or ""
                context.mark_success()
            elif context.step >= context.max_steps:
                context.mark_failed("exceeded_max_steps")
            elif recovery_action == "stop":
                context.result = guard.detail
                context.mark_failed("repeated_tool_failure")
                log.warning("%s run_id=%s", guard.detail, context.run_id)

            # 工具结果追加完毕（messages 末尾为 user）后检查压缩，仅在 run 继续时触发
            # 此时压缩结果 [user_summary, assistant_ack] 对下一次 LLM 调用是合法输入
            if (
                not context.is_done()
                and response.stop_reason == "tool_use"
                and self._compactor is not None
                and self._compact_threshold > 0
                and response.usage is not None
                and response.usage.context_pct >= self._compact_threshold
            ):
                await self._compactor.compact(context, self._provider)
                self._checkpoint(
                    context, "context_compacted", [], response.stop_reason
                )

            await self._bus.publish(
                StepFinishedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )
            self._checkpoint(context, "step_finished", [], response.stop_reason)
