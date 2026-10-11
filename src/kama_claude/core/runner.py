from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kama_claude.core.bus.events import RunFinishedEvent, RunResumedEvent, RunStartedEvent
from kama_claude.core.checkpoint import (
    RunCheckpointer,
    RunCheckpointStore,
    repair_interrupted_tool_calls,
    restore_pending_tool_calls,
)
from kama_claude.core.compact.compactor import Compactor
from kama_claude.core.config import KamaConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus, EventHandler
from kama_claude.core.events.writer import EventWriter
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.provider import AnthropicProvider
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.mcp.server import McpServerManager
from kama_claude.core.memory.loader import load_context_file
from kama_claude.core.permissions.manager import PermissionManager
from kama_claude.core.rag.service import RagService
from kama_claude.core.rag.tools import KnowledgeReadTool, KnowledgeSearchTool
from kama_claude.core.runs import RUNS_DIR, new_run_id
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore
from kama_claude.core.skills.loader import SkillLoader
from kama_claude.core.skills.tool import ActivateSkillTool
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import AgentResultTool, SpawnAgentTool
from kama_claude.core.task.manager import TaskManager
from kama_claude.core.tools.builtin import (
    BashTool,
    ListDirTool,
    NoteSaveTool,
    ReadFileTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    WriteFileTool,
)
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.core.tools.timeout_recovery import ResolveToolTimeoutTool
from kama_claude.core.trace.provider import TracingProvider
from kama_claude.core.trace.writer import TraceWriter


def _now() -> str:
    return datetime.now(UTC).isoformat()


# 从消息历史中提取最近一条 assistant 文本，供终态响应恢复使用
def _last_assistant_text(messages: list[dict[str, object]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(texts)
    return ""


@dataclass
class RunOutcome:
    status: str
    result: str
    reason: str | None


class AgentRunner:
    # 组装所有运行时依赖，准备执行一次完整的 agent run
    def __init__(
        self,
        config: KamaConfig,
        *,
        bus: EventBus | None = None,
        provider: LLMProvider | None = None,
        extra_handlers: list[EventHandler] | None = None,
        runs_dir: Path | None = None,
        trace: TraceWriter | None = None,
        permission_manager: PermissionManager | None = None,
        mcp_manager: McpServerManager | None = None,
        rag_service: RagService | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._provider = provider
        self._extra_handlers: list[EventHandler] = extra_handlers or []
        self._runs_dir = runs_dir or RUNS_DIR
        self._trace = trace
        self._permission_manager = permission_manager
        self._mcp_manager = mcp_manager
        self._rag_service = rag_service
        # 跨 run 共享的后台 subagent 任务注册表
        self._task_registry = BackgroundTaskRegistry()

    # 构建工具注册表，注入 TaskManager（任务工具共享同一实例）；可选注入 SpawnAgentTool
    def _build_registry(
        self,
        task_manager: TaskManager,
        *,
        session: Session | None = None,
        store: SessionStore | None = None,
        run_id: str | None = None,
        provider: LLMProvider | None = None,
        bus: EventBus | None = None,
        child_runs_dir: Path | None = None,
        session_id: str = "",
        tool_whitelist: list[str] | None = None,
        context: ExecutionContext | None = None,
    ) -> ToolRegistry:
        allowed: set[str] | None = set(tool_whitelist) if tool_whitelist else None

        def _ok(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()
        if context is not None and _ok("resolve_tool_timeout"):
            registry.register(ResolveToolTimeoutTool(context.timeout_recovery))
        for t in [ReadFileTool(), BashTool(), WriteFileTool(), ListDirTool()]:
            if _ok(t.name):
                registry.register(t)
        for t in [
            TaskCreateTool(task_manager),
            TaskUpdateTool(task_manager),
            TaskListTool(task_manager),
            TaskGetTool(task_manager),
        ]:
            if _ok(t.name):
                registry.register(t)
        if session is not None and store is not None and run_id is not None:
            note_tool = NoteSaveTool(store, session.id, run_id, context=context)
            if _ok(note_tool.name):
                registry.register(note_tool)
        if provider is not None and bus is not None and run_id is not None:
            runs_dir = child_runs_dir or self._runs_dir
            if _ok("spawn_agent"):
                registry.register(
                    SpawnAgentTool(
                        provider=provider,
                        parent_bus=bus,
                        parent_run_id=run_id,
                        permission_manager=self._permission_manager,
                        max_steps=self._config.agent.max_steps,
                        task_registry=self._task_registry,
                        runs_dir=runs_dir,
                        session_id=session_id,
                        depth=0,
                    )
                )
            if _ok("agent_result"):
                registry.register(AgentResultTool(self._task_registry))
        if self._mcp_manager is not None:
            for mcp_tool in self._mcp_manager.get_tools():
                if _ok(mcp_tool.name):
                    registry.register(mcp_tool)
        if self._rag_service is not None:
            for rag_tool in (
                KnowledgeSearchTool(self._rag_service),
                KnowledgeReadTool(self._rag_service),
            ):
                if _ok(rag_tool.name):
                    registry.register(rag_tool)
        if (
            self._config.agent.auto_skills
            and context is not None
            and context.system_prompt_override is None
            and bus is not None
            and _ok("activate_skill")
        ):
            skills = [
                skill for skill in SkillLoader().list_all_skills()
                if skill.description.strip() and not skill.disable_model_invocation
            ]
            if skills:
                registry.register(ActivateSkillTool(skills, context, registry, bus))
        return registry

    # 执行一次完整的 agent run（委托给 run_and_capture，忽略返回值）
    async def run(self, goal: str, *, run_id: str | None = None) -> None:
        await self.run_and_capture(goal, run_id=run_id)

    # 执行 agent run 并返回 RunOutcome（含最终文字结果）
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        session: Session | None = None,
        store: SessionStore | None = None,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
        resume: bool = False,
    ) -> RunOutcome:
        if resume and run_id is None:
            raise ValueError("run_id is required when resuming")
        run_id = run_id or new_run_id()
        if session is not None and store is not None:
            run_path = store.runs_dir(session.id) / run_id
            history = store.read_messages(session.id)
            notes = store.read_notes(session.id)
        else:
            run_path = self._runs_dir / run_id
            history = [{"role": "user", "content": goal}]
            notes = ""
        run_path.mkdir(parents=True, exist_ok=True)

        checkpoint_store = RunCheckpointStore(run_path / "checkpoint.json")
        previous_checkpoint = checkpoint_store.load() if resume else None
        if resume and previous_checkpoint is None:
            raise ValueError(f"checkpoint not found for run {run_id}")
        if previous_checkpoint is not None:
            if previous_checkpoint.run_id != run_id:
                raise ValueError("checkpoint run_id does not match requested run")
            expected_session_id = session.id if session is not None else ""
            if previous_checkpoint.session_id != expected_session_id:
                raise ValueError("checkpoint session_id does not match requested session")
            goal = previous_checkpoint.goal
            system_prompt_override = previous_checkpoint.system_prompt_override
            tool_whitelist = previous_checkpoint.tool_whitelist

        global_ctx = load_context_file(Path("~/.kama/context.md").expanduser())
        project_ctx = load_context_file(Path(".kama/context.md"))

        task_manager = TaskManager(run_path / ".tasks")

        bus = self._bus if self._bus is not None else EventBus()
        for h in self._extra_handlers:
            bus.subscribe(h)

        if previous_checkpoint is None:
            context = ExecutionContext(
                run_id=run_id,
                goal=goal,
                max_steps=self._config.agent.max_steps,
                prefill_messages=history,
                session_notes=notes,
                global_context=global_ctx,
                project_context=project_ctx,
                system_prompt_override=system_prompt_override,
            )
        else:
            context = ExecutionContext(
                run_id=run_id,
                goal=goal,
                max_steps=max(
                    previous_checkpoint.max_steps,
                    previous_checkpoint.step + self._config.agent.max_steps,
                ),
                session_notes=notes,
                global_context=global_ctx,
                project_context=project_ctx,
                messages=previous_checkpoint.messages,
                step=previous_checkpoint.step,
                system_prompt_override=system_prompt_override,
            )
        prefill_len = len(history)

        session_id_str = session.id if session is not None else ""
        checkpointer = RunCheckpointer(
            checkpoint_store,
            session_id=session_id_str,
            system_prompt_override=system_prompt_override,
            tool_whitelist=tool_whitelist,
            previous=previous_checkpoint,
        )
        terminal_response_recovered = False
        resume_count = 0
        if previous_checkpoint is None:
            checkpointer.save(context, "created", [])
        else:
            previous_status = previous_checkpoint.status
            previous_reason = previous_checkpoint.reason
            pending_tool_calls = restore_pending_tool_calls(previous_checkpoint)
            repaired_ids = repair_interrupted_tool_calls(context, pending_tool_calls)
            context.status = "running"
            context.reason = None
            context.result = ""
            recovery_parts = [
                f"This run resumed from checkpoint phase '{previous_checkpoint.phase}'."
            ]
            if previous_status == "failed" and previous_reason:
                recovery_parts.append(f"The previous attempt stopped because: {previous_reason}.")
            if repaired_ids:
                recovery_parts.append(
                    "Some tool calls have an unknown final state. Inspect the current state "
                    "before retrying any side-effecting operation."
                )
            context.recovery_instruction = " ".join(recovery_parts)
            if (
                previous_checkpoint.phase == "model_response"
                and previous_checkpoint.last_stop_reason == "end_turn"
                and not pending_tool_calls
            ):
                context.result = _last_assistant_text(context.messages)
                context.mark_success()
                terminal_response_recovered = True
            resume_count = checkpointer.mark_resumed()
            checkpointer.save(
                context,
                "resumed",
                [],
                previous_checkpoint.last_stop_reason,
            )

        # 将 AgentLoop 的关键执行阶段同步写入当前 run 的 checkpoint.json
        def save_checkpoint(
            current: ExecutionContext,
            phase: str,
            pending: list[ToolCallBlock] | None,
            stop_reason: str | None,
        ) -> None:
            checkpointer.save(current, phase, pending, stop_reason)

        async with EventWriter(run_path / "events.jsonl") as writer:
            writer.subscribe(bus)
            if previous_checkpoint is None:
                await bus.publish(RunStartedEvent(run_id=run_id, goal=goal, ts=_now()))
            else:
                await bus.publish(
                    RunResumedEvent(
                        run_id=run_id,
                        from_phase=previous_checkpoint.phase,
                        from_step=previous_checkpoint.step,
                        resume_count=resume_count,
                        ts=_now(),
                    )
                )

            cancelled = False
            try:
                provider: LLMProvider = self._provider or AnthropicProvider(
                    self._config.llm.default_model
                )
                if self._trace is not None:
                    provider = TracingProvider(
                        provider,
                        self._trace,
                        include_payload=self._config.trace.include_llm_payload,
                    )
                child_runs_dir = (
                    store.runs_dir(session.id)
                    if session is not None and store is not None
                    else self._runs_dir
                )
                registry = self._build_registry(
                    task_manager,
                    session=session,
                    store=store,
                    run_id=run_id,
                    provider=provider,
                    bus=bus,
                    child_runs_dir=child_runs_dir,
                    session_id=session_id_str,
                    tool_whitelist=tool_whitelist,
                    context=context,
                )
                session_dir = (
                    store.session_dir(session.id)
                    if session is not None and store is not None
                    else run_path
                )
                compactor = Compactor(bus, session_dir, session_id_str)
                loop = AgentLoop(
                    provider, registry, bus,
                    permission_manager=self._permission_manager,
                    compactor=compactor,
                    compact_threshold=self._config.compaction.auto_threshold,
                    session_id=session_id_str,
                    checkpoint_callback=save_checkpoint,
                )
                if not terminal_response_recovered:
                    await loop.run(context)
            except asyncio.CancelledError:
                cancelled = True
                if not context.is_done():
                    context.mark_failed("cancelled")
            except Exception:
                logging.getLogger(__name__).exception(
                    "agent run failed run_id=%s step=%d", run_id, context.step
                )
                if not context.is_done():
                    context.mark_failed("llm_error")

            if session is not None and store is not None:
                if context.messages[:prefill_len] == history:
                    store.append_messages(
                        session.id, context.messages[prefill_len:], run_id=run_id
                    )
                else:
                    # 压缩或恢复会替换消息前缀，此时覆盖为完整合法上下文以避免重复或丢失
                    store.write_compacted(session.id, context.messages)

            final_phase = "cancelled" if cancelled else "finished"
            checkpointer.save(
                context,
                final_phase,
                thread_committed=session is not None and store is not None,
            )
            await bus.publish(
                RunFinishedEvent(
                    run_id=run_id,
                    status=context.status,
                    reason=context.reason,
                    steps=context.step,
                    ts=_now(),
                )
            )

        if cancelled:
            raise asyncio.CancelledError()

        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
        )
