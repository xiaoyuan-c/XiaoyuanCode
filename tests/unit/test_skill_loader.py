from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.skills.loader import Skill, SkillLoader


# 功能：内建 review skill 应能被 SkillLoader 查找到
# 设计：直接调用 resolve("review")，不依赖文件系统之外的任何状态
def test_builtin_skill_found() -> None:
    loader = SkillLoader()
    skill = loader.resolve("review")
    assert skill is not None
    assert skill.name == "review"
    assert "审查" in skill.description or "review" in skill.description.lower()
    assert skill.system_prompt_template != ""


# 功能：内建 init / summarize / orchestrate skill 均可找到
# 设计：列举所有内建 skill 名，断言均能解析
@pytest.mark.parametrize("name", ["init", "review", "summarize", "orchestrate"])
def test_all_builtin_skills_found(name: str) -> None:
    loader = SkillLoader()
    skill = loader.resolve(name)
    assert skill is not None, f"builtin skill '{name}' not found"


# 功能：不存在的 skill 名应返回 None
# 设计：查找一个不存在的名称，断言 resolve 返回 None 而非抛异常
def test_unknown_skill_returns_none() -> None:
    loader = SkillLoader()
    result = loader.resolve("nonexistent_skill_xyz")
    assert result is None


# 功能：render_prompt 应将 $ARGUMENTS 替换为传入的参数字符串
# 设计：构造含 $ARGUMENTS 的 skill，验证 render_prompt 结果不含 "$ARGUMENTS" 且含参数值
def test_arguments_substituted() -> None:
    loader = SkillLoader()
    skill = Skill(
        name="test",
        description="test skill",
        system_prompt_template="Review this: $ARGUMENTS\nPlease be thorough.",
        allowed_tools=[],
    )
    rendered = loader.render_prompt(skill, "src/foo.py")
    assert "$ARGUMENTS" not in rendered
    assert "src/foo.py" in rendered


# 功能：frontmatter 中的 allowed_tools 列表应被正确解析
# 设计：构造含 allowed_tools 的 Markdown 文件，通过 _parse_skill_file 解析并验证结果
def test_frontmatter_parsed(tmp_path: Path) -> None:
    from kama_claude.core.skills.loader import _parse_skill_file

    content = """\
---
name: custom
description: 自定义 skill 测试
allowed_tools:
  - read_file
  - bash
---
你是一个测试助手，目标：$ARGUMENTS
"""
    p = tmp_path / "custom.md"
    p.write_text(content, encoding="utf-8")
    skill = _parse_skill_file(p)
    assert skill.name == "custom"
    assert skill.description == "自定义 skill 测试"
    assert "read_file" in skill.allowed_tools
    assert "bash" in skill.allowed_tools
    assert "$ARGUMENTS" in skill.system_prompt_template


# 功能：无 frontmatter 的 Markdown 文件仍可加载，allowed_tools 为空列表
# 设计：写入纯正文 Markdown，断言解析成功且 allowed_tools=[]
def test_no_frontmatter(tmp_path: Path) -> None:
    from kama_claude.core.skills.loader import _parse_skill_file

    content = "你是助手，请帮助用户完成任务：$ARGUMENTS\n"
    p = tmp_path / "plain.md"
    p.write_text(content, encoding="utf-8")
    skill = _parse_skill_file(p)
    assert skill.name == "plain"
    assert skill.allowed_tools == []
    assert "你是助手" in skill.system_prompt_template


# 功能：项目本地 skill 应覆盖内建同名 skill
# 设计：在 .kama/skills/ 中写入同名文件，用 monkeypatch 修改 cwd，断言加载到的是本地版本
def test_project_overrides_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_skills = tmp_path / ".kama" / "skills"
    local_skills.mkdir(parents=True)
    (local_skills / "review.md").write_text(
        "---\nname: review\ndescription: local override\n---\nlocal system prompt $ARGUMENTS\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loader = SkillLoader()
    skill = loader.resolve("review")
    assert skill is not None
    assert skill.description == "local override"
    assert "local system prompt" in skill.system_prompt_template


# 功能：仅手动标志仍允许显式加载，且支持现有和标准风格的字段名
# 设计：参数化两种字段拼写，加载真实文件而非构造 Skill 对象，覆盖解析与自动入口所需元数据
@pytest.mark.parametrize("key", ["disable-model-invocation", "disable_model_invocation"])
def test_manual_only_flag_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    directory = tmp_path / ".kama" / "skills"
    directory.mkdir(parents=True)
    (directory / "deploy.md").write_text(
        f"---\nname: deploy\ndescription: Deploy app\n{key}: true\n---\nDeploy $ARGUMENTS",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    skill = SkillLoader().resolve("deploy")

    assert skill is not None
    assert skill.disable_model_invocation
    assert skill.system_prompt_template == "Deploy $ARGUMENTS"


# 功能：标准风格的内联白名单也能被解析，其他字段的列表不能混入工具白名单
# 设计：覆盖空格串与内联数组，并添加无关列表，防止格式差异导致自动激活后限制失效
@pytest.mark.parametrize("value", ["read_file list_dir", "[read_file, list_dir]"])
def test_inline_tools_parsed(tmp_path: Path, value: str) -> None:
    from kama_claude.core.skills.loader import _parse_skill_file

    file = tmp_path / "review.md"
    file.write_text(
        f"---\nname: review\ndescription: Review code\nallowed-tools: {value}\n"
        "tags:\n  - ignored\n---\nReview instructions",
        encoding="utf-8",
    )

    assert _parse_skill_file(file).allowed_tools == ["read_file", "list_dir"]


# 功能：同目录同时存在两种格式时，自动目录与显式命令加载相同模板
# 设计：构造冲突的扁平文件和目录式 Skill，验证自动选择不会悄悄使用另一套指令和白名单
def test_catalog_matches_explicit_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / ".kama" / "skills"
    (directory / "review").mkdir(parents=True)
    (directory / "review.md").write_text(
        "---\nname: review\ndescription: Flat review\n---\nFlat instructions",
        encoding="utf-8",
    )
    (directory / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Directory review\n---\nDirectory instructions",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loader = SkillLoader()
    catalog_skill = next(skill for skill in loader.list_all_skills() if skill.name == "review")

    assert catalog_skill == loader.resolve("review")
    assert catalog_skill.system_prompt_template == "Flat instructions"
