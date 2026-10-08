from __future__ import annotations

import asyncio
import os
import signal
import sys

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.tools.base import BaseTool, ToolResult

_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB
_DEFAULT_TIMEOUT = 60
_CLEANUP_TIMEOUT = 5


# 清理本次启动的进程树并等待输出管道结束；清理失败时保留未确认状态
async def _stop_process_tree(proc: asyncio.subprocess.Process) -> bool:
    try:
        if sys.platform == "win32":
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(proc.pid), "/T", "/F",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                code = await asyncio.wait_for(killer.wait(), timeout=_CLEANUP_TIMEOUT)
            except TimeoutError:
                killer.kill()
                await killer.wait()
                code = -1
            stopped = code == 0
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stopped = True
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await asyncio.wait_for(proc.wait(), timeout=_CLEANUP_TIMEOUT)
        return stopped
    except (OSError, TimeoutError):
        return False


class BashParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    command: str
    timeout: int = Field(default=_DEFAULT_TIMEOUT, ge=1, le=120)


class BashTool(BaseTool):
    params_model = BashParams
    name = "bash"
    description = (
        "Execute a shell command and return its output (stdout + stderr combined). "
        "Non-interactive only — commands requiring user input will hang and time out. "
        "Prefer short, focused commands. Output is truncated at 64 KB."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Maximum seconds to wait (default {_DEFAULT_TIMEOUT}, max 120).",
            },
        },
        "required": ["command"],
    }

    # 在子进程中执行 shell 命令，合并 stdout/stderr，超时或非零退出码时返回错误
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = BashParams.model_validate(params)
        command = p.command
        timeout = p.timeout

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=sys.platform != "win32",
            )
            communication = asyncio.create_task(proc.communicate())
            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    asyncio.shield(communication), timeout=timeout
                )
            except TimeoutError:
                stopped, output = await _cleanup_command(proc, communication)
                return ToolResult(
                    content=(
                        f"[timeout after {timeout}s; process cleanup "
                        f"{'confirmed' if stopped else 'unconfirmed'}]\n{output}\n"
                        "Changes may already have occurred. Inspect the actual state "
                        "before retrying; timeout does not roll back files."
                    ),
                    is_error=True,
                    error_type="timeout",
                    execution_stopped=stopped,
                )
            except asyncio.CancelledError:
                await _cleanup_command(proc, communication)
                raise
        except Exception as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="runtime_error")

        output = stdout_bytes.decode("utf-8", errors="replace")
        truncated = len(stdout_bytes) > _MAX_OUTPUT_BYTES
        if truncated:
            output = output[:_MAX_OUTPUT_BYTES] + "\n[truncated]"

        returncode = proc.returncode or 0
        if returncode != 0:
            return ToolResult(
                content=f"[exit {returncode}]\n{output}",
                is_error=True,
                error_type="runtime_error",
            )
        return ToolResult(content=output or "[no output]")


# 超时或取消时清理进程并限时回收输出，防止继承管道的子进程让清理无限等待
async def _cleanup_command(
    proc: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes, bytes | None]],
) -> tuple[bool, str]:
    stopped = await _stop_process_tree(proc)
    try:
        stdout, _ = await asyncio.wait_for(communication, timeout=_CLEANUP_TIMEOUT)
        output = stdout[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        return stopped, output
    except (TimeoutError, OSError):
        return False, "[output unavailable during cleanup]"
