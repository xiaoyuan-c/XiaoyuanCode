from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from kama_claude.core.config import RagConfig
from kama_claude.core.rag.embeddings import (
    EmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
)
from kama_claude.core.rag.errors import RagDependencyError, RagIndexError
from kama_claude.core.rag.models import KnowledgeChunk, SearchHit
from kama_claude.core.rag.reranker import CrossEncoderReranker, RerankerBackend
from kama_claude.core.rag.store import RagIndexStore, StoredIndex


class RagService:
    # 创建延迟加载索引与模型的知识检索服务
    def __init__(
        self,
        config: RagConfig,
        *,
        embedder: EmbeddingBackend | None = None,
        reranker: RerankerBackend | None = None,
    ) -> None:
        self._config = config
        self._store = RagIndexStore(Path(config.index_dir))
        self._stored: StoredIndex | None = None
        self._embedder = embedder
        self._reranker = reranker

    # 返回本地索引状态而不加载任何模型
    def status(self) -> dict[str, Any]:
        return self._store.status()

    # 丢弃内存缓存，使下一次检索读取刚更新的索引
    def reload(self) -> None:
        self._stored = None

    # 确保索引已经加载，并检查其与当前配置使用相同的向量模型
    def _get_stored(self) -> StoredIndex:
        if self._stored is None:
            self._stored = self._store.load()
        model = self._stored.manifest.get("embedding_model")
        if model != self._config.embedding_model:
            raise RagIndexError(
                "索引所用 Embedding 模型与当前配置不一致，请重新执行：kama rag index --force"
            )
        return self._stored

    # 延迟创建查询向量模型
    def _get_embedder(self) -> EmbeddingBackend:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbeddingBackend(
                self._config.embedding_model
            )
        return self._embedder

    # 按配置决定是否启用交叉编码器重排
    def _get_reranker(self) -> RerankerBackend | None:
        if not self._config.reranker_model:
            return None
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self._config.reranker_model)
        return self._reranker

    # 判断片段是否满足可选路径与标签过滤条件
    @staticmethod
    def _matches_filters(
        chunk: KnowledgeChunk,
        path_prefix: str | None,
        tags: tuple[str, ...],
    ) -> bool:
        if path_prefix and not chunk.source_path.startswith(path_prefix.strip("/")):
            return False
        return not tags or bool(set(tags).intersection(chunk.tags))

    # 使用 BM25、Faiss、RRF、笔记层级筛选和可选 Reranker 完成混合检索
    def search(
        self,
        query: str,
        *,
        alternate_queries: tuple[str, ...] = (),
        top_k: int | None = None,
        path_prefix: str | None = None,
        tags: tuple[str, ...] = (),
    ) -> list[SearchHit]:
        stored = self._get_stored()
        allowed = {
            index
            for index, chunk in enumerate(stored.chunks)
            if self._matches_filters(chunk, path_prefix, tags)
        }
        if not allowed:
            return []
        queries = tuple(
            dict.fromkeys(value.strip() for value in (query, *alternate_queries) if value.strip())
        )
        scores: dict[int, float] = {}
        lexical_ranks: dict[int, int] = {}
        vector_ranks: dict[int, int] = {}
        candidate_k = min(self._config.candidate_k, len(allowed))

        for current_query in queries:
            lexical_scores = stored.bm25.score(current_query)
            lexical_order = sorted(
                allowed, key=lambda index: lexical_scores[index], reverse=True
            )[:candidate_k]
            for rank, index in enumerate(lexical_order, start=1):
                if lexical_scores[index] <= 0:
                    continue
                scores[index] = scores.get(index, 0.0) + 1.0 / (
                    self._config.rrf_k + rank
                )
                lexical_ranks[index] = min(lexical_ranks.get(index, rank), rank)

            query_vector = self._get_embedder().encode([current_query])[0]
            try:
                numpy = import_module("numpy")
            except ImportError as error:
                raise RagDependencyError("缺少 numpy，请执行：uv sync --extra rag") from error
            search_count = (
                len(stored.chunks)
                if path_prefix or tags
                else min(len(stored.chunks), max(candidate_k * 5, candidate_k))
            )
            _, raw_indexes = stored.vector_index.search(
                numpy.asarray([query_vector], dtype="float32"), search_count
            )
            vector_order = [int(index) for index in raw_indexes[0] if int(index) in allowed]
            for rank, index in enumerate(vector_order[:candidate_k], start=1):
                scores[index] = scores.get(index, 0.0) + 1.0 / (
                    self._config.rrf_k + rank
                )
                vector_ranks[index] = min(vector_ranks.get(index, rank), rank)

        ordered = sorted(scores, key=scores.__getitem__, reverse=True)
        if self._config.hierarchical and ordered:
            source_scores: dict[str, float] = {}
            for index in ordered:
                source = stored.chunks[index].source_path
                source_scores[source] = max(source_scores.get(source, 0.0), scores[index])
            selected_sources = {
                source
                for source, _ in sorted(
                    source_scores.items(), key=lambda item: item[1], reverse=True
                )[:self._config.note_candidate_k]
            }
            ordered = [
                index for index in ordered
                if stored.chunks[index].source_path in selected_sources
            ]
        ordered = ordered[:self._config.candidate_k]

        reranker = self._get_reranker()
        rerank_scores: dict[int, float] = {}
        if reranker is not None and ordered:
            values = reranker.score(
                query, [stored.chunks[index].retrieval_text() for index in ordered]
            )
            rerank_scores = dict(zip(ordered, values, strict=True))
            ordered.sort(key=rerank_scores.__getitem__, reverse=True)

        limit = top_k or self._config.top_k
        return [
            SearchHit(
                chunk=stored.chunks[index],
                score=scores[index],
                lexical_rank=lexical_ranks.get(index),
                vector_rank=vector_ranks.get(index),
                rerank_score=rerank_scores.get(index),
            )
            for index in ordered[:limit]
        ]

    # 按稳定片段 ID 返回索引中的完整内容
    def read(self, chunk_id: str) -> KnowledgeChunk:
        for chunk in self._get_stored().chunks:
            if chunk.id == chunk_id:
                return chunk
        raise RagIndexError(f"未找到知识片段：{chunk_id}")
