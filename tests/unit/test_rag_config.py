from __future__ import annotations

import pytest

from kama_claude.core.config import KamaConfig, _apply_toml


# 功能：RAG TOML 配置能覆盖默认模型、路径和检索参数
# 设计：直接调用配置合并函数排除开发机本地配置干扰，覆盖新增配置契约
def test_rag_toml_config() -> None:
    config = KamaConfig()

    _apply_toml(
        config,
        {
            "rag": {
                "enabled": True,
                "vault_path": "/vault",
                "index_dir": "workspace/custom",
                "embedding_model": "embedding",
                "reranker_model": "",
                "chunk_size": 800,
                "chunk_overlap": 80,
                "candidate_k": 20,
                "top_k": 4,
                "rrf_k": 50,
                "hierarchical": False,
                "note_candidate_k": 6,
                "excluded_dirs": [".obsidian"],
            }
        },
    )

    assert config.rag.enabled
    assert config.rag.vault_path == "/vault"
    assert config.rag.chunk_size == 800
    assert config.rag.reranker_model == ""
    assert not config.rag.hierarchical


# 功能：重叠长度不允许达到或超过片段长度
# 设计：构造会导致分块无法前进的边界配置，确认启动前即可明确失败
def test_rag_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(SystemExit, match="chunk_overlap"):
        _apply_toml(KamaConfig(), {"rag": {"chunk_size": 100, "chunk_overlap": 100}})
