from __future__ import annotations

from kama_claude.core.rag.bm25 import BM25Index
from kama_claude.core.rag.tokenizer import tokenize


# 功能：中英文混合分词同时产生英文词元、中文单字和中文二元词元
# 设计：使用典型技术短语验证无需外部分词库也能覆盖精确关键词与中文局部匹配
def test_mixed_tokenizer() -> None:
    tokens = tokenize("Agent 使用向量检索和 BM25")

    assert "agent" in tokens
    assert "向量" in tokens
    assert "检索" in tokens
    assert "bm25" in tokens


# 功能：BM25 应把含有查询关键词的文档排在无关文档之前
# 设计：用中文专有词构造明显相关和无关文本，直接比较分数避免耦合排序实现
def test_bm25_ranks_keyword_match_first() -> None:
    index = BM25Index.build(["AgentLoop 使用工具调用", "红黑树保持平衡"])

    scores = index.score("AgentLoop 工具")

    assert scores[0] > scores[1]


# 功能：BM25 索引序列化后仍保持相同评分行为
# 设计：通过 to_dict/from_dict 往返恢复索引，验证落盘数据没有遗漏统计所需字段
def test_bm25_serialization_roundtrip() -> None:
    original = BM25Index.build(["BM25 关键词检索", "Embedding 向量检索"])
    restored = BM25Index.from_dict(original.to_dict())

    assert restored.score("关键词") == original.score("关键词")

