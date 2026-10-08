from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from kama_claude.core.bus.events import SkillInvokedEvent
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.skills.loader import Skill, SkillLoader
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.registry import ToolRegistry


class ActivateSkillParams(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    name: str
    arguments: str = ""


class ActivateSkillTool(BaseTool):
    name = "activate_skill"
    params_model = ActivateSkillParams

    # 把可自动触发的 Skill 名称与描述提供给模型，正文只在激活后进入模型上下文
    def __init__(
        self,
        skills: list[Skill],
        context: ExecutionContext,
        registry: ToolRegistry,
        bus: EventBus,
    ) -> None:
        self._skills = {
            skill.name: skill for skill in skills
            if skill.description.strip() and not skill.disable_model_invocation
        }
        self._context = context
        self._registry = registry
        self._bus = bus
        self._active = False
        catalog = [
            {"name": skill.name, "description": skill.description}
            for skill in self._skills.values()
        ]
        self.description = (
            "Load specialized instructions for the current task. If a task clearly "
            "matches a skill description below, activate that skill before doing the "
            "work. Do not activate a skill for unrelated tasks or a mere mention of "
            "its name. If none matches, proceed normally without this tool. "
            "Select at most one primary skill per run. Pass the task-specific target "
            "or request as arguments. Activation may narrow the available tools; "
            "request activation alone, then follow the loaded instructions. "
            "Available skills (name and description only):\n"
            + json.dumps(catalog, ensure_ascii=False)
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": list(self._skills)},
                "arguments": {
                    "type": "string",
                    "description": "Target path or task details for the skill template.",
                },
            },
            "required": ["name"],
        }

    # 激活一个 Skill、展开参数并收紧工具集合，沿用已有事件展示与工具权限检查
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = ActivateSkillParams.model_validate(params)
        skill = self._skills.get(parsed.name)
        if skill is None or self._active:
            return ToolResult(
                content="Skill is unavailable or a skill is already active in this run.",
                is_error=True,
                error_type="schema_error",
            )
        arguments = parsed.arguments.strip() or self._context.goal
        instructions = SkillLoader().render_prompt(skill, arguments)
        self._context.system_prompt_override = instructions
        allowed = (
            set(skill.allowed_tools) if skill.allowed_tools
            else {str(schema["name"]) for schema in self._registry.tool_schemas()}
        )
        allowed.discard(self.name)
        self._registry.restrict(allowed)
        self._active = True
        await self._bus.publish(SkillInvokedEvent(
            skill_name=skill.name,
            arguments=arguments,
            run_id=self._context.run_id,
            ts=datetime.now(UTC).isoformat(),
        ))
        return ToolResult(content=f"Activated skill: {skill.name}\n\n{instructions}")
