from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.rag.errors import RagError
from kama_claude.core.rag.models import KnowledgeChunk, SearchHit
from kama_claude.core.rag.service import RagService
from kama_claude.core.tools.base import BaseTool, ToolResult


class KnowledgeSearchParams(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    query: str = Field(min_length=1, description="要在个人知识库中检索的问题或关键词")
    alternate_queries: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="可选的查询改写或同义表达，用于提高召回率",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="最多返回的知识片段数；不填时使用系统配置",
    )
    path_prefix: str | None = Field(
        default=None,
        description="可选的知识库相对目录前缀，用于缩小检索范围",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="可选标签，片段命中任意一个标签即可",
    )


class KnowledgeReadParams(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    chunk_id: str = Field(min_length=1, description="knowledge_search 返回的片段 ID")


# 将单条搜索结果整理成便于模型引用的文本
def _format_hit(index: int, hit: SearchHit) -> str:
    heading = " > ".join(hit.chunk.heading_path) or "（无标题层级）"
    score = hit.rerank_score if hit.rerank_score is not None else hit.score
    return (
        f"[{index}] chunk_id={hit.chunk.id}\n"
        f"source={hit.chunk.source_path}\n"
        f"heading={heading}\n"
        f"score={score:.6f}\n"
        f"content:\n{hit.chunk.content}"
    )


# 将完整片段整理成包含来源信息的文本
def _format_chunk(chunk: KnowledgeChunk) -> str:
    heading = " > ".join(chunk.heading_path) or "（无标题层级）"
    tags = ", ".join(chunk.tags) or "（无）"
    return (
        f"chunk_id={chunk.id}\n"
        f"source={chunk.source_path}\n"
        f"title={chunk.title}\n"
        f"heading={heading}\n"
        f"tags={tags}\n"
        f"content:\n{chunk.content}"
    )


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = (
        "在用户的只读 Obsidian 知识库索引中进行 BM25 与向量混合检索；"
        "适合查找个人笔记、学习资料和已有结论。"
    )
    params_model = KnowledgeSearchParams
    input_schema = KnowledgeSearchParams.model_json_schema()
    read_only = True

    # 绑定共享的 RAG 检索服务
    def __init__(self, service: RagService) -> None:
        self._service = service

    # 在线程池执行模型检索，避免阻塞 AgentLoop 的事件循环
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = KnowledgeSearchParams.model_validate(params)
        try:
            hits = await asyncio.to_thread(
                self._service.search,
                parsed.query,
                alternate_queries=tuple(parsed.alternate_queries),
                top_k=parsed.top_k,
                path_prefix=parsed.path_prefix,
                tags=tuple(parsed.tags),
            )
        except RagError as error:
            return ToolResult(str(error), is_error=True, error_type="knowledge_error")
        if not hits:
            return ToolResult("个人知识库中没有找到匹配内容。")
        return ToolResult("\n\n---\n\n".join(
            _format_hit(index, hit) for index, hit in enumerate(hits, start=1)
        ))


class KnowledgeReadTool(BaseTool):
    name = "knowledge_read"
    description = "根据 knowledge_search 返回的片段 ID，读取该索引片段的完整正文与来源。"
    params_model = KnowledgeReadParams
    input_schema = KnowledgeReadParams.model_json_schema()
    read_only = True

    # 绑定共享的 RAG 检索服务
    def __init__(self, service: RagService) -> None:
        self._service = service

    # 在线程池中按片段 ID 读取索引内容
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = KnowledgeReadParams.model_validate(params)
        try:
            chunk = await asyncio.to_thread(self._service.read, parsed.chunk_id)
        except RagError as error:
            return ToolResult(str(error), is_error=True, error_type="knowledge_error")
        return ToolResult(_format_chunk(chunk))

