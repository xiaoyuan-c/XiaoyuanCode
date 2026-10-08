from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kama_claude.core.compact.compactor import Compactor
from kama_claude.core.config import KamaConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.permissions.policy import PermissionDecision, evaluate
from kama_claude.core.runner import AgentRunner
from kama_claude.core.skills.loader import Skill, SkillLoader
from kama_claude.core.skills.tool import ActivateSkillTool
from kama_claude.core.task.manager import TaskManager
from kama_claude.core.tools.builtin import ReadFileTool, WriteFileTool
from kama_claude.core.tools.registry import ToolRegistry


# 构建含已注册工具的真实激活入口，使测试能验证工具分发和权限边界
def _activation(
    skills: list[Skill],
) -> tuple[ActivateSkillTool, ExecutionContext, ToolRegistry, EventBus]:
    context = ExecutionContext(run_id="r", goal="review src", max_steps=5)
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    bus = EventBus()
    tool = ActivateSkillTool(skills, context, registry, bus)
    registry.register(tool)
    return tool, context, registry, bus


# 功能：模型初始只看到目录，激活后参数展开、系统指令和白名单一起生效
# 设计：激活前检查 schema 不含正文，激活后检查原有写工具被移除而未注册工具不会被添加
async def test_activation_loads_instructions_and_restricts_tools() -> None:
    skill = Skill("review", "Review source code", "PRIVATE BODY: $ARGUMENTS", ["read_file", "bash"])
    tool, ctx, registry, bus = _activation([skill])
    received: list[object] = []

    # 收集运行时事件，验证沿用 TUI 订阅的 skill.invoked 展示路径
    async def collect(event: object) -> None:
        received.append(event)

    bus.subscribe(collect)
    assert "Review source code" in tool.description
    assert "PRIVATE BODY" not in str(registry.tool_schemas())

    result = await tool.invoke({"name": "review", "arguments": "src/main.py"})

    assert not result.is_error
    assert ctx.system_prompt_override == "PRIVATE BODY: src/main.py"
    assert "$ARGUMENTS" not in ctx.system_prompt("BASE")
    assert registry.get("read_file") is not None
    assert registry.get("write_file") is None
    assert registry.get("bash") is None
    assert registry.get("activate_skill") is None
    assert any(getattr(event, "type", "") == "skill.invoked" for event in received)


# 功能：仅手动 Skill 和不存在的名称都不能通过自动入口加载
# 设计：传入禁用自动触发的 Skill 及目录遍历名称，验证失败不修改系统指令或工具集合
@pytest.mark.parametrize("name", ["deploy", "../../review", "missing"])
async def test_unavailable_skills_do_not_change_context(name: str) -> None:
    skills = [
        Skill("review", "Review code", "Review instructions"),
        Skill("deploy", "Deploy code", "Deploy instructions", disable_model_invocation=True),
    ]
    tool, ctx, registry, _ = _activation(skills)
    original = registry.tool_schemas()
    assert "deploy" not in tool.description

    result = await tool.invoke({"name": name})

    assert result.is_error
    assert result.error_type == "schema_error"
    assert ctx.system_prompt_override is None
    assert registry.tool_schemas() == original


# 功能：空参数使用原任务，且一个 Run 不能再次激活或切换 Skill
# 设计：对同一个工具实例调用两次，验证幂等边界避免绕过收紧的白名单
async def test_activation_defaults_to_goal_and_rejects_second_activation() -> None:
    tool, ctx, registry, _ = _activation([Skill("review", "Review code", "$ARGUMENTS")])

    assert not (await tool.invoke({"name": "review"})).is_error
    assert ctx.system_prompt_override == ctx.goal
    assert registry.get("write_file") is not None
    assert (await tool.invoke({"name": "review"})).is_error


# 功能：同一模型响应中的后续调用也必须遵守刚激活 Skill 的白名单
# 设计：模拟 activate_skill 与 write_file 同批返回，验证执行循环逐次分发时拒绝写入
async def test_same_turn_forbidden_tool_is_not_executed(tmp_path: Path) -> None:
    tool, ctx, registry, bus = _activation([
        Skill("review", "Review code", "Review $ARGUMENTS", ["read_file"]),
    ])
    target = tmp_path / "must-not-exist.txt"
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        LlmResponse(stop_reason="tool_use", tool_calls=[
            ToolCallBlock(id="s", name=tool.name, input={"name": "review"}),
            ToolCallBlock(id="w", name="write_file", input={"path": str(target), "content": "bad"}),
        ]),
        LlmResponse(stop_reason="end_turn", text="Reviewed"),
    ])

    await AgentLoop(provider, registry, bus).run(ctx)

    assert ctx.status == "success"
    assert not target.exists()
    assert "unknown tool: write_file" in str(ctx.messages)
    next_call = provider.chat.call_args_list[1].kwargs
    assert "Review review src" in next_call["system"]
    assert [schema["name"] for schema in next_call["tool_schemas"]] == ["read_file"]


# 功能：已加载 Skill 的系统指令不会随消息摘要压缩丢失
# 设计：先激活再模拟有损摘要，检查正文保留在消息列表外的系统提示中
async def test_active_skill_survives_compaction(tmp_path: Path) -> None:
    tool, ctx, _, bus = _activation([Skill("review", "Review code", "Critical review rules")])
    await tool.invoke({"name": "review"})
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=LlmResponse(stop_reason="end_turn", text="Short summary"))

    await Compactor(bus, tmp_path, "s").compact(ctx, provider)

    assert "Critical review rules" in ctx.system_prompt("BASE")


# 功能：显式模板、工具白名单或关闭配置都阻止自动入口注册
# 设计：调用真实 runner 注册逻辑，验证自动选择不能覆盖显式选择或越过调用方的工具边界
@pytest.mark.parametrize("mode", ["explicit", "whitelist", "disabled"])
def test_explicit_selection_and_config_take_precedence(tmp_path: Path, mode: str) -> None:
    config = KamaConfig()
    config.agent.auto_skills = mode != "disabled"
    ctx = ExecutionContext(run_id="r", goal="test", max_steps=5)
    if mode == "explicit":
        ctx.system_prompt_override = "Explicit skill instructions"
    registry = AgentRunner(config)._build_registry(
        TaskManager(tmp_path / "tasks"), context=ctx, bus=EventBus(),
        tool_whitelist=["read_file"] if mode == "whitelist" else None,
    )

    assert registry.get("activate_skill") is None


# 功能：激活本身直接允许，但后续实际文件写入仍需权限确认
# 设计：对照已有权限策略的决策，验证自动加载指令不会给写入工具自动授权
def test_activation_does_not_grant_write_permissions() -> None:
    assert evaluate("activate_skill", {"name": "review"}) == PermissionDecision.ALLOW
    assert evaluate("write_file", {"path": "test.py"}) == PermissionDecision.ASK


# 功能：自然语言请求下 runner 提供自动入口，激活后的下轮调用使用模板与受限工具集合
# 设计：模拟模型主动调用 activate_skill，走真实 runner 的注册、执行、事件和下一次模型调用链路
async def test_runner_activates_skill_without_slash_command(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=[
        LlmResponse(stop_reason="tool_use", tool_calls=[
            ToolCallBlock(id="s", name="activate_skill", input={"name": "review", "arguments": "src"}),
        ]),
        LlmResponse(stop_reason="end_turn", text="Review complete"),
    ])
    runner = AgentRunner(KamaConfig(), provider=provider, runs_dir=tmp_path)

    result = await runner.run_and_capture("帮我审查 src 中的代码", run_id="r")

    assert result.status == "success"
    first = provider.chat.call_args_list[0].kwargs
    assert any(schema["name"] == "activate_skill" for schema in first["tool_schemas"])
    second = provider.chat.call_args_list[1].kwargs
    assert "代码审查员" in second["system"]
    assert "$ARGUMENTS" not in second["system"]
    assert {schema["name"] for schema in second["tool_schemas"]} == {
        "read_file", "list_dir", "bash", "resolve_tool_timeout",
    }
    assert "skill.invoked" in (tmp_path / "r" / "events.jsonl").read_text(encoding="utf-8")


# 功能：没有可自动调用的 Skill 时不注册空激活工具
# 设计：分别模拟空目录和全部仅手动的目录，验证模型不会看到没有合法选项的工具
@pytest.mark.parametrize("manual_only", [False, True])
def test_empty_auto_catalog_omits_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manual_only: bool) -> None:
    skills = [Skill("deploy", "Deploy app", "Deploy", disable_model_invocation=True)] if manual_only else []
    monkeypatch.setattr(SkillLoader, "list_all_skills", lambda self: skills)
    registry = AgentRunner(KamaConfig())._build_registry(
        TaskManager(tmp_path / "tasks"),
        context=ExecutionContext(run_id="r", goal="test", max_steps=5),
        bus=EventBus(),
    )

    assert registry.get("activate_skill") is None


# 功能：模型认为无需 Skill 时可以直接完成普通任务，无额外分类或强制激活
# 设计：单次 end_turn 响应完成任务，检查调用次数和事件记录，覆盖自动入口的可选路径
async def test_unrelated_task_can_finish_without_activation(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=LlmResponse(stop_reason="end_turn", text="Hello"))

    outcome = await AgentRunner(KamaConfig(), provider=provider, runs_dir=tmp_path).run_and_capture(
        "Say hello", run_id="r",
    )

    assert outcome.status == "success"
    provider.chat.assert_called_once()
    assert "skill.invoked" not in (tmp_path / "r" / "events.jsonl").read_text(encoding="utf-8")
