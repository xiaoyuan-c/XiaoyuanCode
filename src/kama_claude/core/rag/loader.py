from __future__ import annotations

import hashlib
import re
from pathlib import Path

from kama_claude.core.rag.errors import RagError
from kama_claude.core.rag.models import KnowledgeChunk, LoadedNote

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


# 返回知识库内未被排除的 Markdown 文件，整个过程只读取源目录
def scan_markdown_files(vault_path: Path, excluded_dirs: tuple[str, ...]) -> list[Path]:
    root = vault_path.expanduser().resolve()
    if not root.is_dir():
        raise RagError(f"Obsidian 知识库目录不存在或不是目录：{root}")
    excluded = set(excluded_dirs)
    return sorted(
        path
        for path in root.rglob("*.md")
        if path.is_file() and not excluded.intersection(path.relative_to(root).parts[:-1])
    )


# 从简单 YAML frontmatter 中提取标题与标签，不引入额外 YAML 依赖
def _parse_frontmatter(text: str) -> tuple[str | None, tuple[str, ...], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, (), text
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration:
        return None, (), text

    title: str | None = None
    tags: list[str] = []
    reading_tags = False
    for line in lines[1:end]:
        stripped = line.strip()
        if stripped.startswith("title:"):
            title = stripped.removeprefix("title:").strip().strip("\"'") or None
            reading_tags = False
        elif stripped.startswith("tags:"):
            value = stripped.removeprefix("tags:").strip()
            reading_tags = True
            if value.startswith("[") and value.endswith("]"):
                tags.extend(
                    item.strip().strip("\"'")
                    for item in value[1:-1].split(",")
                    if item.strip()
                )
                reading_tags = False
            elif value:
                tags.extend(part for part in re.split(r"[,\s]+", value) if part)
                reading_tags = False
        elif reading_tags and stripped.startswith("-"):
            value = stripped.removeprefix("-").strip().strip("\"'")
            if value:
                tags.append(value)
        elif stripped and not line.startswith((" ", "\t")):
            reading_tags = False
    return title, tuple(dict.fromkeys(tags)), "\n".join(lines[end + 1:])


# 按 Markdown 标题层级切分正文，同时保留每段所在的标题路径
def _split_sections(body: str) -> list[tuple[tuple[str, ...], str]]:
    sections: list[tuple[tuple[str, ...], str]] = []
    heading_stack: list[str] = []
    content_lines: list[str] = []
    code_fence: str | None = None

    # 将当前标题路径下已经收集的正文写入章节列表
    def flush() -> None:
        content = "\n".join(content_lines).strip()
        if content:
            sections.append((tuple(heading_stack), content))

    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            code_fence = None if code_fence == marker else marker
            content_lines.append(line)
            continue
        if code_fence is not None:
            content_lines.append(line)
            continue
        match = _HEADING_RE.match(line)
        if match is None:
            content_lines.append(line)
            continue
        flush()
        content_lines = []
        level = len(match.group(1))
        heading_stack = heading_stack[:level - 1]
        heading_stack.append(match.group(2).strip())
    flush()
    return sections


# 将过长章节切成带少量重叠的字符窗口，避免把整篇长笔记一次送入模型
def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        tentative_end = min(start + chunk_size, len(text))
        end = tentative_end
        if tentative_end < len(text):
            boundary = text.rfind("\n", start + chunk_size // 2, tentative_end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# 读取单篇 Markdown 并生成带来源元数据的知识片段
def load_markdown_note(
    path: Path,
    vault_path: Path,
    *,
    chunk_size: int,
    overlap: int,
) -> LoadedNote:
    root = vault_path.expanduser().resolve()
    resolved = path.resolve()
    try:
        relative_path = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise RagError(f"拒绝读取知识库目录之外的文件：{resolved}") from error
    raw_bytes = resolved.read_bytes()
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8", errors="replace")
    frontmatter_title, tags, body = _parse_frontmatter(text)
    title = frontmatter_title or resolved.stem

    chunks: list[KnowledgeChunk] = []
    ordinal = 0
    for heading_path, section in _split_sections(body):
        for content in _split_long_text(section, chunk_size, overlap):
            chunk_key = f"{relative_path}\0{'/'.join(heading_path)}\0{ordinal}\0{content}"
            chunk_id = hashlib.sha256(chunk_key.encode()).hexdigest()[:24]
            chunks.append(
                KnowledgeChunk(
                    id=chunk_id,
                    source_path=relative_path,
                    title=title,
                    heading_path=heading_path,
                    tags=tags,
                    content=content,
                    source_hash=source_hash,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return LoadedNote(relative_path, source_hash, tuple(chunks))


# 扫描并加载整个知识库的 Markdown 笔记
def load_vault_notes(
    vault_path: Path,
    *,
    excluded_dirs: tuple[str, ...],
    chunk_size: int,
    overlap: int,
) -> list[LoadedNote]:
    return [
        load_markdown_note(
            path,
            vault_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for path in scan_markdown_files(vault_path, excluded_dirs)
    ]

