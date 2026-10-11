from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from kama_claude.core.rag.bm25 import BM25Index
from kama_claude.core.rag.errors import RagDependencyError, RagIndexError
from kama_claude.core.rag.models import KnowledgeChunk

_INDEX_VERSION = 1


@dataclass
class StoredIndex:
    chunks: list[KnowledgeChunk]
    vectors: Any
    vector_index: Any
    bm25: BM25Index
    manifest: dict[str, Any]


class RagIndexStore:
    # 管理位于 XiaoCode 工作区内的 RAG 索引文件
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir.expanduser()

    # 判断本地索引的必要文件是否齐全
    def exists(self) -> bool:
        return all(
            (self.index_dir / name).is_file()
            for name in ("manifest.json", "chunks.jsonl", "vectors.npy", "faiss.index", "bm25.json")
        )

    # 返回无需加载模型即可读取的索引状态
    def status(self) -> dict[str, Any]:
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.is_file():
            return {"exists": False, "index_dir": str(self.index_dir)}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise RagIndexError(f"索引清单损坏：{manifest_path}") from error
        return {"exists": self.exists(), "index_dir": str(self.index_dir), **manifest}

    # 加载 numpy 与 faiss；仅在真正使用向量索引时要求安装可选依赖
    @staticmethod
    def _load_vector_dependencies() -> tuple[Any, Any]:
        try:
            numpy = import_module("numpy")
            faiss = import_module("faiss")
        except ImportError as error:
            raise RagDependencyError(
                "RAG 向量索引依赖未安装，请执行：uv sync --extra rag"
            ) from error
        return numpy, faiss

    # 将片段、向量、BM25 和清单保存到项目内的索引目录
    def save(
        self,
        chunks: list[KnowledgeChunk],
        vectors: list[list[float]],
        *,
        vault_path: Path,
        embedding_model: str,
        file_hashes: dict[str, str],
        build_options: dict[str, Any],
    ) -> None:
        if len(chunks) != len(vectors):
            raise RagIndexError("片段数量与向量数量不一致")
        if not chunks:
            raise RagIndexError("知识库没有可索引的 Markdown 正文")
        numpy, faiss = self._load_vector_dependencies()
        matrix = numpy.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise RagIndexError("Embedding 模型返回了无效向量")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        vector_index = faiss.IndexFlatIP(int(matrix.shape[1]))
        vector_index.add(matrix)
        faiss.write_index(vector_index, str(self.index_dir / "faiss.index"))
        numpy.save(self.index_dir / "vectors.npy", matrix)

        chunks_text = "\n".join(
            json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks
        ) + "\n"
        (self.index_dir / "chunks.jsonl").write_text(chunks_text, encoding="utf-8")
        bm25 = BM25Index.build([chunk.retrieval_text() for chunk in chunks])
        (self.index_dir / "bm25.json").write_text(
            json.dumps(bm25.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        manifest = {
            "version": _INDEX_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "vault_path": str(vault_path.expanduser().resolve()),
            "embedding_model": embedding_model,
            "files": len(file_hashes),
            "chunks": len(chunks),
            "dimensions": int(matrix.shape[1]),
            "file_hashes": file_hashes,
            "build_options": build_options,
        }
        (self.index_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 从项目索引目录完整加载检索所需数据
    def load(self) -> StoredIndex:
        if not self.exists():
            raise RagIndexError("RAG 索引不存在，请先执行：kama rag index")
        numpy, faiss = self._load_vector_dependencies()
        try:
            chunks = [
                KnowledgeChunk.from_dict(json.loads(line))
                for line in (self.index_dir / "chunks.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ]
            vectors = numpy.load(self.index_dir / "vectors.npy")
            vector_index = faiss.read_index(str(self.index_dir / "faiss.index"))
            bm25 = BM25Index.from_dict(
                json.loads((self.index_dir / "bm25.json").read_text(encoding="utf-8"))
            )
            manifest = json.loads(
                (self.index_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
            raise RagIndexError(f"RAG 索引损坏：{self.index_dir}") from error
        if not (len(chunks) == len(vectors) == len(bm25.tokenized_documents)):
            raise RagIndexError("RAG 索引中的片段、向量与 BM25 数据数量不一致")
        return StoredIndex(chunks, vectors, vector_index, bm25, manifest)

