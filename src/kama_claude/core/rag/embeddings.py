from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

from kama_claude.core.rag.errors import RagDependencyError


class EmbeddingBackend(Protocol):
    # 批量生成已归一化的文本向量
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddingBackend:
    # 延迟加载向量模型，避免未启用 RAG 时引入重型依赖和启动开销
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    # 在第一次实际检索或建索引时加载 sentence-transformers 模型
    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as error:
            raise RagDependencyError(
                "RAG 向量模型依赖未安装，请执行：uv sync --extra rag"
            ) from error
        self._model = sentence_transformers.SentenceTransformer(self._model_name)
        return self._model

    # 批量生成适用于余弦相似度的归一化向量
    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 16,
        )
        return [[float(value) for value in row] for row in vectors]

