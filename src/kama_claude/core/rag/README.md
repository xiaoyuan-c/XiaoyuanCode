# XiaoCode RAG 开发说明

这一模块把指定的 Obsidian Markdown 仓库作为**只读知识源**，让 XiaoCode 在现有 AgentLoop 中调用 `knowledge_search` 和 `knowledge_read`，用于个人编码与学习问答。

## 安全边界

- 源知识库：`rag.vault_path`，只执行目录扫描和文件读取。
- 索引目录：`rag.index_dir`，默认是 XiaoCode 下的 `workspace/rag/obsidian/`。
- 代码不会在 Obsidian 仓库中创建、修改或删除任何文件。
- `.obsidian`、`.trash`、`.learning-agent` 默认不参与索引。
- 项目的个人路径配置 `.kama/config.toml` 已加入 `.gitignore`，不会提交到 Git。

## 当前检索链路

```text
Markdown 笔记
  → 按标题层级切分，并保留文件路径、标题路径、标签
  → BGE Embedding + Faiss 向量索引
  → BM25 中英文关键词索引
  → 两路候选通过 RRF 融合
  → 按笔记聚合后筛选候选笔记（层级化检索）
  → BGE Reranker 重排
  → Top-K 片段返回 AgentLoop
```

`knowledge_search` 允许模型同时提交原查询和最多四个查询改写，用于 Query Rewrite / Query Expansion。复杂问题则由现有 ReAct AgentLoop 多次调用检索工具，属于轻量的 Agentic RAG，而不是额外维护第二套 Agent 框架。

## 安装与配置

RAG 使用可选依赖，不影响 XiaoCode 的基础安装：

```bash
uv sync --extra rag
```

项目本地 `.kama/config.toml` 示例：

```toml
[rag]
enabled = true
vault_path = "/absolute/path/to/your/obsidian-vault"
index_dir = "workspace/rag/obsidian"
embedding_model = "BAAI/bge-small-zh-v1.5"
reranker_model = "BAAI/bge-reranker-base"
chunk_size = 1200
chunk_overlap = 150
candidate_k = 30
top_k = 5
rrf_k = 60
hierarchical = true
note_candidate_k = 8
excluded_dirs = [".obsidian", ".trash", ".learning-agent"]
```

如果机器资源有限，可以设置 `reranker_model = ""` 暂时关闭重排。也可以通过 `KAMA_RAG_ENABLED`、`KAMA_RAG_VAULT_PATH`、`KAMA_RAG_INDEX_DIR` 覆盖前三项。

## 使用命令

首次建索引；第一次运行会下载配置的模型：

```bash
uv run kama rag index
```

笔记未变化时会跳过；仅部分笔记变化时会复用其他文件已有的向量。强制重建：

```bash
uv run kama rag index --force
```

查看索引状态和独立测试检索：

```bash
uv run kama rag status
uv run kama rag search "我以前如何理解 AgentLoop？" --top-k 5
```

随后正常启动 `kama-core`。启用 RAG 后，运行时会把两个只读工具注册到统一工具注册表：

- `knowledge_search`：混合召回、查询扩展、层级筛选与重排。
- `knowledge_read`：根据搜索返回的稳定 `chunk_id` 获取完整片段。

内建 `study` skill 会指导模型先检索、再回答并标注来源。Skill 负责工作方法，真正的数据访问仍通过统一工具体系完成。

## 索引文件

所有产物都在 `rag.index_dir`：

- `manifest.json`：模型、文件哈希和索引统计。
- `chunks.jsonl`：切分后的文本及其来源元数据。
- `vectors.npy`：片段向量，用于增量复用。
- `faiss.index`：Faiss 内积索引；向量已归一化，等价于余弦相似度排序。
- `bm25.json`：关键词检索索引。

`workspace/` 已被 Git 忽略。

## 模块职责

- `loader.py`：只读扫描、frontmatter 提取、标题感知分块。
- `bm25.py` / `tokenizer.py`：不依赖外部搜索服务的关键词召回。
- `embeddings.py` / `reranker.py`：延迟加载 BGE 模型。
- `store.py`：仅在项目索引目录持久化数据。
- `indexer.py`：按文件哈希增量更新并复用未变化向量。
- `service.py`：混合检索、RRF、笔记级筛选、重排和 Top-K。
- `tools.py`：把检索能力接入 XiaoCode 的 `ToolRegistry`。

## 后续评估重点

不要只凭“看起来回答不错”判断效果。建议从真实学习问题中整理一小组评测集，记录期望命中的笔记或标题，比较关键词检索、向量检索、混合检索、查询改写和重排各自对 Recall@K、MRR 与响应耗时的影响，再决定是否微调 Embedding / Reranker。
