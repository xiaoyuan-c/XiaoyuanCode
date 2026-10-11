from __future__ import annotations

from pathlib import Path

from kama_claude.core.rag.errors import RagIndexError
from kama_claude.core.rag.models import KnowledgeChunk, SearchHit
from kama_claude.core.rag.tools import KnowledgeReadTool, KnowledgeSearchTool


class _FakeService:
    # 初始化固定知识片段，并记录搜索参数供断言
    def __init__(self) -> None:
        self.chunk = KnowledgeChunk(
            id="chunk-1",
            source_path="学习/Agent.md",
            title="Agent",
            heading_path=("循环",),
            tags=("Agent",),
            content="AgentLoop 会重复推理、调用工具和观察结果。",
            source_hash="hash",
            ordinal=0,
        )
        self.search_args: tuple[object, ...] | None = None

    # 返回固定搜索结果并捕获查询扩展参数
    def search(self, query: str, **kwargs: object) -> list[SearchHit]:
        self.search_args = (query, kwargs)
        return [SearchHit(self.chunk, 0.1, lexical_rank=1, vector_rank=2)]

    # 按固定 ID 返回片段，未知 ID 模拟索引错误
    def read(self, chunk_id: str) -> KnowledgeChunk:
        if chunk_id != self.chunk.id:
            raise RagIndexError("未找到知识片段")
        return self.chunk


# 功能：knowledge_search 将查询改写参数传给服务，并在结果中保留片段 ID 与来源
# 设计：使用轻量 fake 避免下载向量模型，聚焦 Agent 工具和检索服务之间的契约
async def test_knowledge_search_tool_formats_sources() -> None:
    service = _FakeService()
    tool = KnowledgeSearchTool(service)  # type: ignore[arg-type]

    result = await tool.invoke(
        {"query": "什么是 AgentLoop", "alternate_queries": ["ReAct 循环"], "top_k": 3}
    )

    assert not result.is_error
    assert "chunk_id=chunk-1" in result.content
    assert "source=学习/Agent.md" in result.content
    assert service.search_args is not None
    assert service.search_args[0] == "什么是 AgentLoop"


# 功能：knowledge_read 对不存在的片段返回不可重试的 knowledge_error
# 设计：让 fake 抛 RagIndexError，确认错误被包装而不是穿透并触发运行时自动重试
async def test_knowledge_read_tool_wraps_index_error() -> None:
    tool = KnowledgeReadTool(_FakeService())  # type: ignore[arg-type]

    result = await tool.invoke({"chunk_id": "missing"})

    assert result.is_error
    assert result.error_type == "knowledge_error"


# 功能：注入 RAG 服务后，Runner 会把两个知识工具注册进统一工具体系
# 设计：直接检查 ToolRegistry，验证 RAG 与内置工具共享注册表且无需加载真实模型
def test_runner_registers_rag_tools(tmp_path: Path) -> None:
    from kama_claude.core.config import KamaConfig
    from kama_claude.core.rag.service import RagService
    from kama_claude.core.runner import AgentRunner
    from kama_claude.core.task.manager import TaskManager

    config = KamaConfig()
    service = RagService(config.rag)
    runner = AgentRunner(config, rag_service=service)

    registry = runner._build_registry(TaskManager(tmp_path / "tasks"))

    assert registry.get("knowledge_search") is not None
    assert registry.get("knowledge_read") is not None

