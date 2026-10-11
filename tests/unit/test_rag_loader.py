from __future__ import annotations

from pathlib import Path

from kama_claude.core.rag.loader import load_vault_notes, scan_markdown_files


# 功能：只扫描未排除目录中的 Markdown，忽略 Obsidian 元数据和非 Markdown 文件
# 设计：在临时知识库同时放置正常笔记、隐藏目录笔记和文本文件，验证扫描边界而不接触真实仓库
def test_scan_markdown_files_respects_exclusions(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "agent.md").write_text("# Agent", encoding="utf-8")
    (tmp_path / "notes" / "ignore.txt").write_text("text", encoding="utf-8")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "plugin.md").write_text("metadata", encoding="utf-8")

    files = scan_markdown_files(tmp_path, (".obsidian",))

    assert [file.relative_to(tmp_path).as_posix() for file in files] == ["notes/agent.md"]


# 功能：Markdown 加载器保留 frontmatter 标签与标题层级，并把长章节拆成多个片段
# 设计：构造含多级标题和超长正文的笔记，断言检索元数据完整且分块长度受控
def test_load_vault_notes_preserves_metadata_and_chunks(tmp_path: Path) -> None:
    note = tmp_path / "RAG.md"
    note.write_text(
        "---\ntitle: 个人检索\ntags: [RAG, 学习]\n---\n"
        "# 基础\nBM25 负责关键词召回。\n"
        "## 向量检索\n" + "向量语义检索。" * 20,
        encoding="utf-8",
    )

    notes = load_vault_notes(
        tmp_path,
        excluded_dirs=(),
        chunk_size=60,
        overlap=10,
    )

    assert len(notes) == 1
    assert len(notes[0].chunks) >= 3
    assert notes[0].chunks[0].title == "个人检索"
    assert notes[0].chunks[0].tags == ("RAG", "学习")
    assert notes[0].chunks[0].heading_path == ("基础",)
    assert any(chunk.heading_path == ("基础", "向量检索") for chunk in notes[0].chunks)
    assert all(chunk.source_path == "RAG.md" for chunk in notes[0].chunks)


# 功能：加载和分块不会改写知识库中的原始 Markdown
# 设计：读取前后比较源文件字节与修改时间，覆盖最重要的只读不变式
def test_load_vault_notes_does_not_modify_source(tmp_path: Path) -> None:
    note = tmp_path / "只读.md"
    original = "# 只读\n这是知识库正文。\n".encode()
    note.write_bytes(original)
    modified_before = note.stat().st_mtime_ns

    load_vault_notes(tmp_path, excluded_dirs=(), chunk_size=100, overlap=10)

    assert note.read_bytes() == original
    assert note.stat().st_mtime_ns == modified_before


# 功能：代码块中的井号注释不能被误判成 Markdown 标题
# 设计：在围栏代码块中放置 Python 注释，断言注释留在正文且标题路径仍是外层章节
def test_heading_parser_ignores_code_fence_comments(tmp_path: Path) -> None:
    (tmp_path / "代码.md").write_text(
        "# 二分查找\n```python\n# 这是代码注释\nvalue = 1\n```\n",
        encoding="utf-8",
    )

    notes = load_vault_notes(tmp_path, excluded_dirs=(), chunk_size=200, overlap=10)

    assert notes[0].chunks[0].heading_path == ("二分查找",)
    assert "# 这是代码注释" in notes[0].chunks[0].content

