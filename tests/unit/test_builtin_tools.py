from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.list_dir import ListDirTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool

# ── bash ──────────────────────────────────────────────────────────────────────


# 按当前系统的 shell 规则引用 Python 命令，避免依赖 sleep 等平台专属程序
def _python_command(*arguments: str) -> str:
    parts = [sys.executable, *arguments]
    return subprocess.list2cmdline(parts) if sys.platform == "win32" else shlex.join(parts)

# 功能：验证成功命令的 stdout 出现在 ToolResult.content 中，is_error 为 False
# 设计：用 echo 命令避免外部依赖，直接比较输出内容，无需 mock
@pytest.mark.asyncio
async def test_bash_success_stdout() -> None:
    result = await BashTool().invoke({"command": "echo hello"})
    assert not result.is_error
    assert "hello" in result.content


# 功能：验证非零退出码时 is_error=True 且 content 包含退出码标注
# 设计：`exit 2` 是最简单的非零退出；不依赖任何外部命令行为
@pytest.mark.asyncio
async def test_bash_nonzero_exit_is_error() -> None:
    result = await BashTool().invoke({"command": "exit 2"})
    assert result.is_error
    assert "[exit 2]" in result.content


# 功能：验证命令超时后 is_error=True，error_type 为 "timeout"
# 设计：使用当前解释器休眠避免 Windows 缺少 sleep；同时确认执行端已清理进程
@pytest.mark.asyncio
async def test_bash_timeout() -> None:
    result = await BashTool().invoke({
        "command": _python_command("-c", "import time; time.sleep(5)"), "timeout": 1,
    })
    assert result.is_error
    assert result.error_type == "timeout"
    assert result.execution_stopped is True


# 功能：Shell 超时或外层取消会停止子进程，保留已发生的修改并阻止其继续写入
# 设计：真实子进程先写开始标记、等待后再写结束标记，确认进程清理不是仅杀掉外层 shell
@pytest.mark.parametrize("cancel", [False, True])
async def test_bash_cleans_children_without_rolling_back(tmp_path: Path, cancel: bool) -> None:
    child = tmp_path / "child.py"
    started, finished = tmp_path / "started", tmp_path / "finished"
    child.write_text(
        "import sys, time\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text('partial')\n"
        "print('started', flush=True)\ntime.sleep(3)\n"
        "Path(sys.argv[2]).write_text('done')\n", encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, *sys.argv[1:]])\n", encoding="utf-8",
    )
    task = asyncio.create_task(BashTool().invoke({
        "command": _python_command(str(parent), str(child), str(started), str(finished)),
        "timeout": 10 if cancel else 1,
    }))
    if cancel:
        for _ in range(100):
            if started.exists():
                break
            await asyncio.sleep(0.02)
        assert started.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result.error_type == "timeout"
        assert result.execution_stopped is True
        assert "started" in result.content
    await asyncio.sleep(3.1)
    assert started.read_text() == "partial"
    assert not finished.exists()


# 功能：验证 stderr 被合并到 stdout 输出中
# 设计：只写 stderr 的命令（>&2 echo），输出应该出现在合并后的 content 里
@pytest.mark.asyncio
async def test_bash_stderr_merged() -> None:
    result = await BashTool().invoke({"command": "echo err >&2"})
    assert not result.is_error
    assert "err" in result.content


# ── write_file ────────────────────────────────────────────────────────────────

# 功能：验证 write_file 写入文件后内容可以被读取，返回字节数
# 设计：写入临时目录，断言文件存在且内容一致；用 tmp_path fixture 自动清理
@pytest.mark.asyncio
async def test_write_file_creates_and_returns_size(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    result = await WriteFileTool().invoke(
        {"path": str(target), "content": "hello world"}
    )
    assert not result.is_error
    assert "11" in result.content  # "hello world" = 11 bytes
    assert target.read_text() == "hello world"


# 功能：验证 write_file 自动创建不存在的父目录
# 设计：路径包含两层不存在的子目录，确认写入后目录结构被创建
@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.txt"
    result = await WriteFileTool().invoke({"path": str(target), "content": "x"})
    assert not result.is_error
    assert target.exists()


# 功能：验证 write_file 拒绝包含 .. 的路径并抛出 PermissionError
# 设计：.. 路径遍历与 read_file 遵循相同规则，用相同的断言模式保持一致性
@pytest.mark.asyncio
async def test_write_file_rejects_traversal() -> None:
    with pytest.raises(PermissionError):
        await WriteFileTool().invoke({"path": "../secret.txt", "content": "x"})


# ── list_dir ──────────────────────────────────────────────────────────────────

# 功能：验证 list_dir 输出包含目录中的文件名
# 设计：在 tmp_path 创建已知结构，断言文件名出现在 content 中；不约束格式细节
@pytest.mark.asyncio
async def test_list_dir_shows_files(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x")
    (tmp_path / "bar.md").write_text("y")
    result = await ListDirTool().invoke({"path": str(tmp_path)})
    assert not result.is_error
    assert "foo.py" in result.content
    assert "bar.md" in result.content


# 功能：验证 list_dir 按 max_depth 限制递归深度（depth=1 时不展示孙级目录内容）
# 设计：创建 parent/child/grandchild 三层，depth=1 时 grandchild 不应出现在输出中
@pytest.mark.asyncio
async def test_list_dir_respects_max_depth(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    grandchild = child / "grandchild"
    grandchild.mkdir()
    (grandchild / "deep.txt").write_text("x")

    result = await ListDirTool().invoke({"path": str(tmp_path), "max_depth": 1})
    assert not result.is_error
    assert "child" in result.content
    assert "deep.txt" not in result.content


# 功能：验证对不存在的路径 list_dir 抛出 FileNotFoundError
# 设计：直接传入不存在的路径字符串，预期抛出标准异常（invocation.py 捕获后返回 error ToolResult）
@pytest.mark.asyncio
async def test_list_dir_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        await ListDirTool().invoke({"path": "/this/does/not/exist"})


# 功能：验证 list_dir 拒绝包含 .. 的路径
# 设计：与 read_file 和 write_file 保持一致的安全规则
@pytest.mark.asyncio
async def test_list_dir_rejects_traversal() -> None:
    with pytest.raises(PermissionError):
        await ListDirTool().invoke({"path": "../"})
