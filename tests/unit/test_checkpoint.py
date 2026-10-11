from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kama_claude.core.checkpoint import RunCheckpoint, RunCheckpointStore
from kama_claude.core.config import KamaConfig
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.runner import AgentRunner
from kama_claude.core.session.manager import SessionManager
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore


class _ToolThenEndProvider:
    # 初始化两轮响应计数器
    def __init__(self) -> None:
        self.calls = 0

    # 第一轮请求读取目录，第二轮结束任务
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
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(id="tool-1", name="list_dir", input={"path": "."})
                ],
            )
        return LlmResponse(stop_reason="end_turn", text="done")


class _ResumeCapturingProvider:
    # 初始化恢复调用的消息和系统提示捕获字段
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.system = ""

    # 捕获恢复后的上下文并直接结束任务
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
        self.messages = [dict(message) for message in messages]
        self.system = system or ""
        return LlmResponse(stop_reason="end_turn", text="resumed")


class _BlockingProvider:
    # 初始化模型调用开始信号
    def __init__(self) -> None:
        self.started = asyncio.Event()

    # 模拟一个只能通过任务取消结束的长时间模型调用
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
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


# 创建测试用的最小 agent 配置
def _config() -> KamaConfig:
    config = KamaConfig()
    config.agent.max_steps = 5
    return config


# 创建并持久化一个 chat session
def _session(store: SessionStore, sid: str = "sess-test") -> Session:
    session = Session(
        id=sid,
        mode="chat",
        status="active",
        title="",
        created_at="2026-10-11T00:00:00+00:00",
        updated_at="2026-10-11T00:00:00+00:00",
    )
    store.write_meta(session)
    return session


# 功能：验证正常 run 的 checkpoint 最终包含成功状态、模型消息和工具结果
# 设计：真实执行一个无副作用的 list_dir 工具，覆盖模型返回、工具结果、会话提交和终态落盘
async def test_runner_persists_messages_and_tool_results_in_checkpoint(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = _session(store)
    store.append_message(session.id, "user", "list files")
    runner = AgentRunner(_config(), provider=_ToolThenEndProvider())  # type: ignore[arg-type]

    outcome = await runner.run_and_capture(
        "list files", run_id="run-checkpoint", session=session, store=store
    )

    checkpoint = RunCheckpointStore(
        store.runs_dir(session.id) / "run-checkpoint" / "checkpoint.json"
    ).load()
    assert outcome.status == "success"
    assert checkpoint.status == "success"
    assert checkpoint.phase == "finished"
    assert checkpoint.thread_committed is True
    assert any("tool_use" in str(message["content"]) for message in checkpoint.messages)
    assert any("tool_result" in str(message["content"]) for message in checkpoint.messages)


# 功能：验证恢复工具执行中断的 run 时不会直接重放状态未知的副作用工具
# 设计：手工构造 write_file 的 tool_started 快照，断言恢复后补错误结果并先交给模型重新判断
async def test_resume_repairs_pending_tool_without_replaying_it(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = _session(store)
    store.append_message(session.id, "user", "write output")
    run_id = "run-resume"
    target = tmp_path / "must-not-exist.txt"
    checkpoint_store = RunCheckpointStore(
        store.runs_dir(session.id) / run_id / "checkpoint.json"
    )
    checkpoint_store.save(
        RunCheckpoint(
            run_id=run_id,
            session_id=session.id,
            goal="write output",
            phase="tool_started",
            status="failed",
            step=1,
            max_steps=5,
            messages=[
                {"role": "user", "content": "write output"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "write-1",
                            "name": "write_file",
                            "input": {"path": str(target), "content": "unsafe replay"},
                        }
                    ],
                },
            ],
            reason="cancelled",
            pending_tool_calls=[
                {
                    "id": "write-1",
                    "name": "write_file",
                    "input": {"path": str(target), "content": "unsafe replay"},
                }
            ],
            last_stop_reason="tool_use",
        )
    )
    provider = _ResumeCapturingProvider()
    runner = AgentRunner(_config(), provider=provider)  # type: ignore[arg-type]

    outcome = await runner.run_and_capture(
        "ignored", run_id=run_id, session=session, store=store, resume=True
    )

    assert outcome.status == "success"
    assert not target.exists()
    assert "unknown" in str(provider.messages[-1]).lower()
    assert "Inspect the current state" in provider.system
    resumed = checkpoint_store.load()
    assert resumed.resume_count == 1
    assert resumed.status == "success"
    events = [
        json.loads(line)
        for line in (store.runs_dir(session.id) / run_id / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[0]["type"] == "run.resumed"


# 功能：验证用户取消长任务后 run 状态和消息会先写入 checkpoint，再恢复 session 输入状态
# 设计：用永不返回的 provider 模拟长调用，通过 SessionManager 取消入口覆盖任务传播、落盘和状态收尾
async def test_cancel_current_persists_resumable_checkpoint(tmp_path: Path) -> None:
    provider = _BlockingProvider()
    store = SessionStore(tmp_path / "sessions")
    manager = SessionManager(
        store,
        lambda: AgentRunner(_config(), provider=provider),  # type: ignore[arg-type]
        EventBus(),
    )
    session = await manager.create("chat")
    send_task = asyncio.create_task(manager.send_message(session.id, "long task"))
    await provider.started.wait()
    run_id = store.read_meta(session.id).run_ids[-1]

    cancelled_run_id, status = await manager.cancel_current(session.id)
    completed_run_id = await send_task

    assert cancelled_run_id == completed_run_id == run_id
    assert status == "cancelled"
    checkpoint = await manager.get_checkpoint(session.id, run_id)
    assert checkpoint.status == "failed"
    assert checkpoint.reason == "cancelled"
    assert checkpoint.phase == "cancelled"
    assert checkpoint.thread_committed is True
    assert store.read_meta(session.id).status == "waiting_for_input"


# 功能：验证 daemon 重启后 SessionManager 能从 meta.json 懒加载旧会话
# 设计：创建会话后丢弃原 manager，用新实例读取历史，覆盖断点续跑依赖的跨进程会话恢复路径
async def test_session_is_lazy_loaded_after_manager_restart(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = _session(store, "sess-persisted")
    store.append_message(session.id, "user", "persist me")
    manager = SessionManager(
        store,
        lambda: AgentRunner(_config(), provider=_ToolThenEndProvider()),  # type: ignore[arg-type]
        EventBus(),
    )

    history = await manager.get_history(session.id)

    assert history == [{"role": "user", "content": "persist me"}]
