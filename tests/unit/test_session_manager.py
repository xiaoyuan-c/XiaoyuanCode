from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.bus.envelope import HandlerError
from kama_claude.core.events.bus import EventBus
from kama_claude.core.runner import RunOutcome
from kama_claude.core.session.manager import SESSION_CLOSED, SESSION_NOT_FOUND, SessionManager
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore


class _Runner:
    # 模拟 AgentRunner，将 run 新消息写入 thread 后返回成功
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        session: Session | None = None,
        store: SessionStore | None = None,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome:
        assert run_id is not None
        assert session is not None
        assert store is not None
        store.append_messages(
            session.id,
            [{"role": "assistant", "content": [{"type": "text", "text": f"done {goal}"}]}],
            run_id,
        )
        return RunOutcome(status="success", result="done", reason=None)


# 功能：验证 create 会创建 active session、写入 meta 并发布 session.created 事件
# 设计：用真实 SessionStore + EventBus 收集事件，覆盖 manager 与 store/bus 的协作边界
async def test_create_session_writes_meta_and_event(tmp_path: Path) -> None:
    events: list[object] = []
    bus = EventBus()

    async def collect(event: object) -> None:
        events.append(event)

    bus.subscribe(collect)
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), bus)  # type: ignore[arg-type]

    session = await manager.create("chat", "title")

    assert session.status == "active"
    assert store.read_meta(session.id).title == "title"
    assert [e.type for e in events] == ["session.created"]  # type: ignore[attr-defined]


# 功能：验证 chat session 处理一条消息后进入 waiting_for_input，并保留 user/assistant thread
# 设计：mock runner 主动追加 assistant 消息，确认 send_message 负责 user 消息、状态流转和 run_id 记录
async def test_send_message_chat_enters_waiting_and_writes_thread(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("chat")

    run_id = await manager.send_message(session.id, "hello")

    loaded = store.read_meta(session.id)
    assert loaded.status == "waiting_for_input"
    assert loaded.run_ids == [run_id]
    messages = store.read_messages(session.id)
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1]["role"] == "assistant"


# 功能：验证 one_shot session 在单次消息完成后自动 closed
# 设计：复用 mock runner 的成功路径，聚焦 mode 对最终状态的影响，保证 kama run 的统一路径正确
async def test_one_shot_auto_closes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("one_shot")

    await manager.send_message(session.id, "hello")

    assert store.read_meta(session.id).status == "closed"


# 功能：验证不存在的 session_id 返回 session_not_found 错误码
# 设计：直接调用 get_history 的查找路径，断言 HandlerError code，覆盖 IPC handler 可结构化返回错误
async def test_missing_session_raises_handler_error(tmp_path: Path) -> None:
    manager = SessionManager(SessionStore(tmp_path), lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    with pytest.raises(HandlerError) as exc:
        await manager.get_history("missing")
    assert exc.value.code == SESSION_NOT_FOUND


# 功能：验证 closed session 不能继续 send_message
# 设计：先显式 close，再发送消息，断言 session_closed 错误码，覆盖状态机拒绝路径
async def test_closed_session_rejects_message(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("chat")
    await manager.close(session.id)

    with pytest.raises(HandlerError) as exc:
        await manager.send_message(session.id, "again")
    assert exc.value.code == SESSION_CLOSED


# 功能：验证手动压缩也补回遗漏笔记，并保留可回读的原始记录
# 设计：真实 manager/store 配合模拟摘要，覆盖手动入口不会绕过自动压缩的笔记保护
async def test_manual_compact_preserves_notes_and_history(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from kama_claude.core.llm.types import LlmResponse

    provider = MagicMock()
    provider.chat = AsyncMock(return_value=LlmResponse(stop_reason="end_turn", text="Summary"))
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus(), provider=provider)  # type: ignore[arg-type]
    session = await manager.create("chat")
    store.append_message(session.id, "user", "Do not modify config.py")
    store.append_note(session.id, "Do not modify config.py", "r")

    await manager.compact(session.id)

    history = await manager.get_history(session.id)
    assert "Do not modify config.py" in history[0]["content"]
    assert "Original history" in history[0]["content"]
    assert len(list(store.session_dir(session.id).glob("history_*.jsonl"))) == 1
    assert "Do not modify config.py" in store.read_notes(session.id)


# 功能：显式 Skill 命令展开系统模板参数并传递白名单，仍允许仅手动 Skill
# 设计：真实 SessionManager 调用捕获型 runner，验证手动入口优先于后端自动注册条件
async def test_explicit_skill_expands_system_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock

    directory = tmp_path / ".kama" / "skills"
    directory.mkdir(parents=True)
    (directory / "review.md").write_text(
        "---\nname: review\ndescription: Review code\ndisable-model-invocation: true\n"
        "allowed_tools:\n  - read_file\n---\nReview exactly $ARGUMENTS",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = MagicMock()
    runner.run_and_capture = AsyncMock(return_value=RunOutcome("success", "done", None))
    manager = SessionManager(SessionStore(tmp_path / "sessions"), lambda: runner, EventBus())
    session = await manager.create("chat")

    await manager.send_message(session.id, "/review src/main.py")

    kwargs = runner.run_and_capture.call_args.kwargs
    assert kwargs["system_prompt_override"] == "Review exactly src/main.py"
    assert kwargs["tool_whitelist"] == ["read_file"]
