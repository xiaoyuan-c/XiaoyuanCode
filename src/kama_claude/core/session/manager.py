from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kama_claude.core.bus.envelope import HandlerError
from kama_claude.core.bus.events import (
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionWaitingForInputEvent,
    SkillInvokedEvent,
)
from kama_claude.core.checkpoint import RunCheckpoint, RunCheckpointStore
from kama_claude.core.events.bus import EventBus
from kama_claude.core.runs import new_run_id
from kama_claude.core.session.model import Session, SessionMode
from kama_claude.core.session.store import SessionStore
from kama_claude.core.skills.loader import SkillLoader

if TYPE_CHECKING:
    from kama_claude.core.llm.base import LLMProvider
    from kama_claude.core.runner import AgentRunner, RunOutcome

SESSION_NOT_FOUND = -32010
SESSION_CLOSED = -32011
SESSION_BUSY = -32012
RUN_NOT_FOUND = -32013
RUN_NOT_RESUMABLE = -32014
NO_ACTIVE_RUN = -32015


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 检查外部传入的持久化对象 ID 不含路径分隔符或目录跳转
def _is_safe_id(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value


class SessionManager:
    # 初始化会话管理器，接入文件存储、runner 工厂、事件总线和可选的 LLM provider（用于手动压缩）
    def __init__(
        self,
        store: SessionStore,
        runner_factory: Callable[[], AgentRunner],
        bus: EventBus,
        provider: LLMProvider | None = None,
    ) -> None:
        self._store = store
        self._runner_factory = runner_factory
        self._bus = bus
        self._provider = provider
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._skill_loader = SkillLoader()
        self._active_runs: dict[str, asyncio.Task[RunOutcome]] = {}
        self._active_by_session: dict[str, str] = {}
        self._user_cancelled: set[str] = set()

    # 创建新 session 并写入 meta.json
    async def create(self, mode: SessionMode, title: str = "") -> Session:
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        ts = _now()
        session = Session(
            id=sid,
            mode=mode,
            status="active",
            title=title,
            created_at=ts,
            updated_at=ts,
            run_ids=[],
        )
        self._sessions[sid] = session
        self._locks[sid] = asyncio.Lock()
        self._store.write_meta(session)
        await self._bus.publish(SessionCreatedEvent(session_id=sid, mode=mode, ts=ts))
        return session

    # 处理用户消息，追加 thread 并启动一次 agent run
    async def send_message(self, sid: str, content: str, *, run_id: str | None = None) -> str:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")

        async with lock:
            if session.status == "closed":
                raise HandlerError(SESSION_CLOSED, "session already closed")

            if session.status == "waiting_for_input":
                await self._bus.publish(SessionResumedEvent(session_id=sid, ts=_now()))

            self._store.append_message(sid, "user", content)
            await self._bus.publish(
                SessionMessageReceivedEvent(session_id=sid, content=content, ts=_now())
            )

            if not session.title:
                session.title = content[:40]

            run_id = run_id or new_run_id()
            session.run_ids.append(run_id)
            session.updated_at = _now()
            self._store.write_meta(session)

            # Skill 解析：检测 "/" 前缀，展开为系统提示覆盖和工具白名单
            goal = content
            system_prompt_override: str | None = None
            tool_whitelist: list[str] | None = None
            if content.startswith("/"):
                parts = content[1:].split(None, 1)
                skill_name = parts[0]
                arguments = parts[1] if len(parts) > 1 else ""
                skill = self._skill_loader.resolve(skill_name)
                if skill is not None:
                    goal = self._skill_loader.render_prompt(skill, arguments)
                    system_prompt_override = goal
                    tool_whitelist = skill.allowed_tools or None
                    await self._bus.publish(
                        SkillInvokedEvent(
                            skill_name=skill_name,
                            arguments=arguments,
                            run_id=run_id,
                            ts=_now(),
                        )
                    )

            await self._execute_run(
                session,
                goal,
                run_id,
                system_prompt_override=system_prompt_override,
                tool_whitelist=tool_whitelist,
            )

            await self._finish_run(session, run_id)
            return run_id

    # 恢复指定 session 中已失败或中断的 run，并继续使用原 run_id
    async def resume_run(self, sid: str, run_id: str) -> RunOutcome:
        session = self._get_session(sid)
        if not _is_safe_id(run_id):
            raise HandlerError(RUN_NOT_FOUND, "run checkpoint not found")
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")

        checkpoint_store = RunCheckpointStore(
            self._store.runs_dir(sid) / run_id / "checkpoint.json"
        )
        if not checkpoint_store.exists():
            raise HandlerError(RUN_NOT_FOUND, "run checkpoint not found")
        try:
            checkpoint = checkpoint_store.load()
        except (OSError, ValueError, KeyError) as error:
            raise HandlerError(RUN_NOT_FOUND, "invalid run checkpoint") from error
        if checkpoint.session_id != sid or checkpoint.run_id != run_id:
            raise HandlerError(RUN_NOT_FOUND, "run checkpoint does not match session")
        if checkpoint.status == "success":
            raise HandlerError(RUN_NOT_RESUMABLE, "successful run cannot be resumed")

        async with lock:
            session.status = "active"
            session.updated_at = _now()
            if run_id not in session.run_ids:
                session.run_ids.append(run_id)
            self._store.write_meta(session)
            await self._bus.publish(SessionResumedEvent(session_id=sid, ts=session.updated_at))
            outcome = await self._execute_run(
                session,
                checkpoint.goal,
                run_id,
                resume=True,
            )
            await self._finish_run(session, run_id)
            return outcome

    # 取消指定 session 当前正在执行的 run，并等待 checkpoint 落盘
    async def cancel_current(self, sid: str) -> tuple[str, str]:
        self._get_session(sid)
        run_id = self._active_by_session.get(sid)
        if run_id is None:
            raise HandlerError(NO_ACTIVE_RUN, "session has no active run")
        task = self._active_runs.get(run_id)
        if task is None:
            raise HandlerError(NO_ACTIVE_RUN, "session has no active run")
        self._user_cancelled.add(run_id)
        task.cancel()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
        # 等待 send_message/resume_run 完成 session 状态收尾后再响应客户端
        async with self._locks[sid]:
            pass
        return run_id, "cancelled"

    # 读取指定 run 的最新 checkpoint 状态，供客户端展示和恢复前检查
    async def get_checkpoint(self, sid: str, run_id: str) -> RunCheckpoint:
        self._get_session(sid)
        if not _is_safe_id(run_id):
            raise HandlerError(RUN_NOT_FOUND, "run checkpoint not found")
        checkpoint_store = RunCheckpointStore(
            self._store.runs_dir(sid) / run_id / "checkpoint.json"
        )
        if not checkpoint_store.exists():
            raise HandlerError(RUN_NOT_FOUND, "run checkpoint not found")
        try:
            checkpoint = checkpoint_store.load()
        except (OSError, ValueError, KeyError) as error:
            raise HandlerError(RUN_NOT_FOUND, "invalid run checkpoint") from error
        if checkpoint.session_id != sid or checkpoint.run_id != run_id:
            raise HandlerError(RUN_NOT_FOUND, "run checkpoint does not match session")
        return checkpoint

    # 创建并跟踪 runner 子任务，使取消命令只中断目标 run 而不关闭客户端连接
    async def _execute_run(
        self,
        session: Session,
        goal: str,
        run_id: str,
        *,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
        resume: bool = False,
    ) -> RunOutcome:
        runner = self._runner_factory()
        if resume:
            coroutine = runner.run_and_capture(
                goal,
                run_id=run_id,
                session=session,
                store=self._store,
                resume=True,
            )
        else:
            coroutine = runner.run_and_capture(
                goal,
                run_id=run_id,
                session=session,
                store=self._store,
                system_prompt_override=system_prompt_override,
                tool_whitelist=tool_whitelist,
            )
        task = asyncio.create_task(coroutine)
        self._active_runs[run_id] = task
        self._active_by_session[session.id] = run_id
        try:
            return await task
        except asyncio.CancelledError:
            if run_id not in self._user_cancelled:
                raise
            from kama_claude.core.runner import RunOutcome
            return RunOutcome(status="failed", result="", reason="cancelled")
        finally:
            self._active_runs.pop(run_id, None)
            if self._active_by_session.get(session.id) == run_id:
                self._active_by_session.pop(session.id, None)
            self._user_cancelled.discard(run_id)

    # 根据 session 模式完成状态流转并通知客户端恢复输入或关闭会话
    async def _finish_run(self, session: Session, run_id: str) -> None:
        session.updated_at = _now()
        if session.mode == "one_shot":
            session.status = "closed"
            await self._bus.publish(
                SessionClosedEvent(session_id=session.id, ts=session.updated_at)
            )
        else:
            session.status = "waiting_for_input"
            await self._bus.publish(
                SessionWaitingForInputEvent(
                    session_id=session.id,
                    last_run_id=run_id,
                    ts=session.updated_at,
                )
            )
        self._store.write_meta(session)

    # 关闭指定 session 并更新 meta.json
    async def close(self, sid: str) -> None:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        async with lock:
            session.status = "closed"
            session.updated_at = _now()
            self._store.write_meta(session)
            await self._bus.publish(SessionClosedEvent(session_id=sid, ts=session.updated_at))

    # 手动压缩指定 session 的 thread，将摘要持久化写入 thread.jsonl
    async def compact(self, sid: str, focus: str = "") -> Any:
        self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        if self._provider is None:
            raise HandlerError(-32020, "provider not available for compaction")
        async with lock:
            from kama_claude.core.bus.commands import SessionCompactResult
            from kama_claude.core.compact.compactor import Compactor
            messages = self._store.read_messages(sid)
            session_dir = self._store.session_dir(sid)
            compactor = Compactor(self._bus, session_dir, sid)
            result = await compactor.compact_messages(messages, self._provider, focus=focus)
            if result is None:
                raise HandlerError(-32021, "compaction failed or not beneficial")
            self._store.write_compacted(sid, [
                {"role": "user", "content": result.summary_text},
                {"role": "assistant", "content": "Understood, I'll continue from this summary."},
            ])
            return SessionCompactResult(
                summary_tokens=result.summary_tokens,
                saved_tokens=max(0, result.original_token_estimate - result.summary_tokens),
            )

    # 读取指定 session 的完整 thread 历史
    async def get_history(self, sid: str) -> list[dict[str, Any]]:
        self._get_session(sid)
        return self._store.read_messages(sid)

    # 从内存索引取 session，不存在时抛 JSON-RPC 结构化错误
    def _get_session(self, sid: str) -> Session:
        session = self._sessions.get(sid)
        if session is None:
            if not _is_safe_id(sid) or not self._store.meta_exists(sid):
                raise HandlerError(SESSION_NOT_FOUND, "session not found")
            try:
                session = self._store.read_meta(sid)
            except (OSError, ValueError, KeyError) as error:
                raise HandlerError(SESSION_NOT_FOUND, "invalid session metadata") from error
            self._sessions[sid] = session
            self._locks[sid] = asyncio.Lock()
        return session
