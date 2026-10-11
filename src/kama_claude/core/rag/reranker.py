from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

from kama_claude.core.rag.errors import RagDependencyError


class RerankerBackend(Protocol):
    # 对候选文档按查询相关性打分
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    # 延迟加载交叉编码器，只有实际重排候选时才占用模型资源
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    # 在第一次重排时加载 CrossEncoder 模型
    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as error:
            raise RagDependencyError(
                "RAG 重排模型依赖未安装，请执行：uv sync --extra rag"
            ) from error
        self._model = sentence_transformers.CrossEncoder(self._model_name)
        return self._model

    # 计算查询与每个候选文档的相关性分数
    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        raw_scores = self._get_model().predict(pairs, show_progress_bar=False)
        return [float(value) for value in raw_scores]
