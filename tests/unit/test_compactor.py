from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kama_claude.core.compact.compactor import Compactor
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, UsageStats


def _stub_provider(summary: str = "## 1. Original Goal\nTest\n## 2. Completed Steps\n- done") -> Any:
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=LlmResponse(
        stop_reason="end_turn",
        text=summary,
        usage=UsageStats(input_tokens=100, output_tokens=30),
    ))
    return provider


def _make_messages(n: int = 5) -> list[dict[str, Any]]:
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": "user message " + "x" * 200})
        msgs.append({"role": "assistant", "content": "assistant reply " + "y" * 200})
    return msgs


# 功能：验证 compact_messages 成功时 provider.chat 被调用一次且不传工具 schema
# 设计：stub provider 返回非空摘要，断言 chat 调用一次，tool_schemas=[]
async def test_compact_messages_calls_provider(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    messages = _make_messages()

    result = await compactor.compact_messages(messages, provider)

    assert result is not None
    provider.chat.assert_called_once()
    call_kwargs = provider.chat.call_args
    assert call_kwargs.kwargs.get("tool_schemas") == [] or call_kwargs.args[1] == []


# 功能：验证摘要保留模型输出，并附上可回读的原始历史归档路径
# 设计：固定模型输出，读取归档 JSONL 对比原始消息，验证摘要来源和恢复数据均可靠
async def test_compact_messages_returns_summary(tmp_path: Path) -> None:
    expected = "## 1. Original Goal\nDo X\n## 2. Completed\n- step one"
    provider = _stub_provider(summary=expected)
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")

    result = await compactor.compact_messages(_make_messages(), provider)

    assert result is not None
    assert result.summary_text.startswith(expected)
    assert result.history_path is not None
    assert str(result.history_path) in result.summary_text
    archived = [json.loads(line) for line in result.history_path.read_text(encoding="utf-8").splitlines()]
    assert archived == _make_messages()


# 功能：验证 compact() 将 context.messages 替换为两条摘要消息对
# 设计：调用 compact() 后断言 messages 长度为 2，role 分别为 user/assistant
async def test_compact_replaces_context_messages(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    ctx.messages = _make_messages()

    await compactor.compact(ctx, provider)

    assert len(ctx.messages) == 2
    assert ctx.messages[0]["role"] == "user"
    assert ctx.messages[1]["role"] == "assistant"


# 功能：验证 compact() 在 session 目录写入 summary_*.md 文件
# 设计：使用 tmp_path，调用 compact() 后检查目录内是否存在 summary_ 开头的文件
async def test_compact_writes_summary_file(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    ctx.messages = _make_messages()

    await compactor.compact(ctx, provider)

    summary_files = list(tmp_path.glob("summary_*.md"))
    assert len(summary_files) == 1


# 功能：验证 compact() 成功后发布 ContextCompactedEvent 事件
# 设计：订阅 EventBus，收集事件，断言收到类型为 context.compacted 的事件
async def test_compact_publishes_event(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    received: list[Any] = []

    async def handler(event: Any) -> None:
        received.append(event)

    bus.subscribe(handler)
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    ctx.messages = _make_messages()

    await compactor.compact(ctx, provider)

    types = [getattr(e, "type", None) for e in received]
    assert "context.compacted" in types


# 功能：验证 provider 抛异常时 context.messages 保持不变
# 设计：stub provider.chat 抛 RuntimeError，断言 compact() 返回 None 且 messages 未被修改
async def test_compact_failure_preserves_context(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=RuntimeError("LLM error"))
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    original_messages = _make_messages()
    ctx.messages = list(original_messages)

    result = await compactor.compact(ctx, provider)

    assert result is None
    assert ctx.messages == original_messages


# 功能：验证摘要遗漏笔记时补回正文，且最新磁盘笔记注入当前执行的系统提示
# 设计：上下文只含旧笔记、磁盘含新约束和进度，模型返回遗漏全部内容的摘要，覆盖同一 Run 中新笔记的保护
async def test_compact_restores_latest_notes(tmp_path: Path) -> None:
    notes = (
        "## Note (t1, r1)\n不能修改 config.py\n\n"
        "## Note (t2, r1)\n已完成解析模块，下一步运行测试\n\n"
    )
    (tmp_path / "notes.md").write_text(notes, encoding="utf-8")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5, session_notes="old note")
    provider = _stub_provider("Short summary")

    result = await Compactor(EventBus(), tmp_path, "sess-1").compact(ctx, provider)

    assert result is not None
    assert "不能修改 config.py" in result.summary_text
    assert "已完成解析模块，下一步运行测试" in result.summary_text
    assert ctx.session_notes == notes
    assert notes.strip() in ctx.system_prompt("BASE")
    request = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert notes in request
    assert "old note" not in result.summary_text


# 功能：验证已原样包含的笔记不会再次补入，重复存储的同一笔记只恢复一次
# 设计：模型保留一条笔记并遗漏另一条，磁盘包含重复正文，检查最终出现次数避免无谓增长
async def test_compact_only_restores_missing_notes(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(
        "## Note (t1, r1)\nkeep config.py\n\n"
        "## Note (t2, r1)\ntests pending\n\n"
        "## Note (t3, r1)\ntests pending\n\n",
        encoding="utf-8",
    )
    result = await Compactor(EventBus(), tmp_path, "s").compact_messages(
        _make_messages(), _stub_provider("Summary: keep config.py"),
    )

    assert result is not None
    assert result.summary_text.count("keep config.py") == 1
    assert result.summary_text.count("tests pending") == 1
    assert result.summary_tokens > 30


# 功能：验证归档无法写入时拒绝压缩并保留原始消息
# 设计：模拟存储层 PermissionError，排除摘要成功后不可恢复地丢弃历史的路径
async def test_archive_failure_preserves_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compactor = Compactor(EventBus(), tmp_path, "s")
    monkeypatch.setattr(compactor, "_archive_history", MagicMock(side_effect=PermissionError("read only")))
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    original = list(ctx.messages)

    assert await compactor.compact(ctx, _stub_provider()) is None
    assert ctx.messages == original


# 功能：验证连续压缩会保留不同的原始历史归档
# 设计：在同一秒内压缩两段不同消息，检查文件名和内容独立，避免恢复资料被覆盖
async def test_repeated_compactions_keep_separate_archives(tmp_path: Path) -> None:
    compactor = Compactor(EventBus(), tmp_path, "s")
    first = await compactor.compact_messages(_make_messages(1), _stub_provider())
    second = await compactor.compact_messages(_make_messages(2), _stub_provider())

    assert first is not None and second is not None
    assert first.history_path is not None and second.history_path is not None
    assert first.history_path != second.history_path
    assert len(first.history_path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(second.history_path.read_text(encoding="utf-8").splitlines()) == 4
