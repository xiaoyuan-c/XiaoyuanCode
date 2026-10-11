from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    source_path: str
    title: str
    heading_path: tuple[str, ...]
    tags: tuple[str, ...]
    content: str
    source_hash: str
    ordinal: int

    # 拼接检索字段，使标题、层级标题和正文都能参与召回
    def retrieval_text(self) -> str:
        heading = " > ".join(self.heading_path)
        return "\n".join(part for part in (self.title, heading, self.content) if part)

    # 将知识片段转换为可持久化字典
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "title": self.title,
            "heading_path": list(self.heading_path),
            "tags": list(self.tags),
            "content": self.content,
            "source_hash": self.source_hash,
            "ordinal": self.ordinal,
        }

    # 从持久化字典恢复知识片段
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeChunk:
        return cls(
            id=str(data["id"]),
            source_path=str(data["source_path"]),
            title=str(data["title"]),
            heading_path=tuple(str(item) for item in data.get("heading_path", [])),
            tags=tuple(str(item) for item in data.get("tags", [])),
            content=str(data["content"]),
            source_hash=str(data["source_hash"]),
            ordinal=int(data["ordinal"]),
        )


@dataclass(frozen=True)
class LoadedNote:
    source_path: str
    source_hash: str
    chunks: tuple[KnowledgeChunk, ...]


@dataclass(frozen=True)
class SearchHit:
    chunk: KnowledgeChunk
    score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class IndexBuildResult:
    files: int
    chunks: int
    embedded_chunks: int
    reused_chunks: int
    skipped: bool = False

