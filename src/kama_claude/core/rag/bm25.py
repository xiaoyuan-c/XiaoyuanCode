from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from kama_claude.core.rag.tokenizer import tokenize


@dataclass
class BM25Index:
    tokenized_documents: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    # 从原始文档构建 BM25 索引
    @classmethod
    def build(cls, documents: list[str]) -> BM25Index:
        return cls([tokenize(document) for document in documents])

    # 计算查询相对于全部文档的 BM25 分数
    def score(self, query: str) -> list[float]:
        document_count = len(self.tokenized_documents)
        if document_count == 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return [0.0] * document_count

        lengths = [len(document) for document in self.tokenized_documents]
        average_length = sum(lengths) / document_count if document_count else 0.0
        frequencies = [Counter(document) for document in self.tokenized_documents]
        document_frequencies = Counter(
            token
            for token in set(query_tokens)
            for document in self.tokenized_documents
            if token in document
        )
        scores = [0.0] * document_count
        for token in set(query_tokens):
            frequency = document_frequencies[token]
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for index, counts in enumerate(frequencies):
                term_frequency = counts[token]
                if term_frequency == 0:
                    continue
                normalizer = term_frequency + self.k1 * (
                    1.0 - self.b
                    + self.b * lengths[index] / max(average_length, 1.0)
                )
                scores[index] += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0) / normalizer
                )
        return scores

    # 将 BM25 索引转换为 JSON 可序列化字典
    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenized_documents": self.tokenized_documents,
            "k1": self.k1,
            "b": self.b,
        }

    # 从持久化字典恢复 BM25 索引
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        raw_documents = data.get("tokenized_documents", [])
        documents = [
            [str(token) for token in document]
            for document in raw_documents
            if isinstance(document, list)
        ]
        return cls(
            tokenized_documents=documents,
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
        )
