from __future__ import annotations

from pathlib import Path

from kama_claude.core.config import RagConfig
from kama_claude.core.rag.embeddings import (
    EmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
)
from kama_claude.core.rag.loader import load_vault_notes
from kama_claude.core.rag.models import IndexBuildResult
from kama_claude.core.rag.store import RagIndexStore


class RagIndexer:
    # 创建支持增量复用向量的 Obsidian 索引器
    def __init__(
        self,
        config: RagConfig,
        embedder: EmbeddingBackend | None = None,
    ) -> None:
        self._config = config
        self._embedder = embedder or SentenceTransformerEmbeddingBackend(
            config.embedding_model
        )
        self._store = RagIndexStore(Path(config.index_dir))

    # 扫描只读知识库并构建索引，未变化的文件直接复用旧向量
    def build(self, *, force: bool = False) -> IndexBuildResult:
        vault_path = Path(self._config.vault_path).expanduser()
        notes = load_vault_notes(
            vault_path,
            excluded_dirs=tuple(self._config.excluded_dirs),
            chunk_size=self._config.chunk_size,
            overlap=self._config.chunk_overlap,
        )
        file_hashes = {note.source_path: note.source_hash for note in notes}
        build_options = {
            "chunk_size": self._config.chunk_size,
            "chunk_overlap": self._config.chunk_overlap,
            "excluded_dirs": sorted(self._config.excluded_dirs),
        }

        old_vectors: dict[str, list[float]] = {}
        old_hashes: dict[str, str] = {}
        if self._store.exists() and not force:
            stored = self._store.load()
            if stored.manifest.get("embedding_model") == self._config.embedding_model:
                old_hashes = {
                    str(path): str(digest)
                    for path, digest in stored.manifest.get("file_hashes", {}).items()
                }
                if (
                    old_hashes == file_hashes
                    and stored.manifest.get("build_options") == build_options
                ):
                    return IndexBuildResult(
                        files=len(notes),
                        chunks=len(stored.chunks),
                        embedded_chunks=0,
                        reused_chunks=len(stored.chunks),
                        skipped=True,
                    )
                old_vectors = {
                    chunk.id: [float(value) for value in stored.vectors[index].tolist()]
                    for index, chunk in enumerate(stored.chunks)
                    if old_hashes.get(chunk.source_path) == chunk.source_hash
                }

        chunks = [chunk for note in notes for chunk in note.chunks]
        vectors: list[list[float] | None] = [old_vectors.get(chunk.id) for chunk in chunks]
        missing_indexes = [index for index, vector in enumerate(vectors) if vector is None]
        if missing_indexes:
            new_vectors = self._embedder.encode(
                [chunks[index].retrieval_text() for index in missing_indexes]
            )
            for index, vector in zip(missing_indexes, new_vectors, strict=True):
                vectors[index] = vector

        complete_vectors = [vector for vector in vectors if vector is not None]
        self._store.save(
            chunks,
            complete_vectors,
            vault_path=vault_path,
            embedding_model=self._config.embedding_model,
            file_hashes=file_hashes,
            build_options=build_options,
        )
        return IndexBuildResult(
            files=len(notes),
            chunks=len(chunks),
            embedded_chunks=len(missing_indexes),
            reused_chunks=len(chunks) - len(missing_indexes),
        )

