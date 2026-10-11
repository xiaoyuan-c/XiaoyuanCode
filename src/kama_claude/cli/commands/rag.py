from __future__ import annotations

import json

from kama_claude.core.config import KamaConfig
from kama_claude.core.rag.errors import RagError
from kama_claude.core.rag.indexer import RagIndexer
from kama_claude.core.rag.service import RagService


# 校验 RAG 已启用且配置了只读知识库路径
def _validate_config(config: KamaConfig) -> None:
    if not config.rag.enabled:
        raise SystemExit("RAG 尚未启用，请在 .kama/config.toml 的 [rag] 中设置 enabled = true")
    if not config.rag.vault_path:
        raise SystemExit("未配置 rag.vault_path")


# 构建或增量更新个人知识库索引
def cmd_rag_index(config: KamaConfig, *, force: bool = False) -> None:
    _validate_config(config)
    try:
        result = RagIndexer(config.rag).build(force=force)
    except RagError as error:
        raise SystemExit(f"RAG 建索引失败：{error}") from error
    if result.skipped:
        print(f"索引无变化：{result.files} 个文件，{result.chunks} 个片段")
        return
    print(
        f"索引完成：{result.files} 个文件，{result.chunks} 个片段；"
        f"新生成 {result.embedded_chunks} 个向量，复用 {result.reused_chunks} 个向量"
    )


# 输出无需加载模型的 RAG 索引状态
def cmd_rag_status(config: KamaConfig) -> None:
    _validate_config(config)
    try:
        status = RagService(config.rag).status()
    except RagError as error:
        raise SystemExit(f"读取 RAG 状态失败：{error}") from error
    print(json.dumps(status, ensure_ascii=False, indent=2))


# 从命令行执行一次混合检索，便于独立验证召回效果
def cmd_rag_search(config: KamaConfig, query: str, *, top_k: int) -> None:
    _validate_config(config)
    try:
        hits = RagService(config.rag).search(query, top_k=top_k)
    except RagError as error:
        raise SystemExit(f"RAG 检索失败：{error}") from error
    if not hits:
        print("没有找到匹配内容")
        return
    for index, hit in enumerate(hits, start=1):
        heading = " > ".join(hit.chunk.heading_path) or "（无标题层级）"
        score = hit.rerank_score if hit.rerank_score is not None else hit.score
        print(
            f"[{index}] {hit.chunk.source_path} | {heading} | score={score:.6f}\n"
            f"chunk_id={hit.chunk.id}\n{hit.chunk.content}\n"
        )
