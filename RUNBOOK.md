# 运维手册（RUNBOOK）

## 日常操作

### 启动守护进程

```bash
uv run kama-core
```

默认监听 `127.0.0.1:7437`，按 `Ctrl+C` 优雅退出。

### 验证连通

```bash
uv run kama ping
# → pong server=0.0.1 uptime=12ms latency=2ms
```

### 停止守护进程

```bash
kill $(pgrep -f kama-core)
```

---

## 配置

优先级（低 → 高）：**内建默认值 → `~/.kama/config.toml` → `.env` → 系统环境变量**。

### `~/.kama/config.toml`

```toml
[core]
host = "127.0.0.1"
port = 7437

[logging]
level  = "INFO"
file   = "~/.kama/logs/core.log"
format = "text"    # "text" | "json"
```

### `.env`

从 `.env.example` 复制后修改，存放本机配置与密钥（不提交 git）：

```bash
cp .env.example .env
```

### 系统环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KAMA_CONFIG` | `~/.kama/config.toml` | 覆盖配置文件路径 |
| `KAMA_HOST` | `127.0.0.1` | TCP 监听地址 |
| `KAMA_PORT` | `7437` | TCP 监听端口 |
| `KAMA_LOG_LEVEL` | `INFO` | 日志级别（DEBUG / INFO / WARNING / ERROR） |
| `KAMA_LOG_FILE` | `~/.kama/logs/core.log` | 日志文件路径（留空则仅输出 stderr） |
| `KAMA_LOG_FORMAT` | `text` | 日志格式（`text` 或 `json`） |

---

## Skill 自动触发

普通消息也支持加载 Skill。例如在 TUI 中输入“帮我审查 src 的代码”，模型会根据可用
Skill 的名称和描述判断是否加载 `review`，选中后才把完整模板加入上下文。
`/review src` 等显式命令仍然可用，并优先于自动选择。每个 Run 最多自动加载一个主 Skill，
加载时会显示已有的 Skill 事件，并按模板配置收紧工具白名单；实际命令和文件写入仍经过权限审批。

自动触发默认开启，可通过 `.env` 中的 `KAMA_AUTO_SKILLS=false` 关闭，或设置：

```toml
[agent]
auto_skills = false
```

某个 Skill 只允许手动调用时，在其 frontmatter 中添加 `disable-model-invocation: true`。
自动目录不包含这类 Skill，也不包含没有描述的 Skill。描述应写清楚适用任务和触发条件。

---

## 重复工具失败保护

主 Agent 和子 Agent 共用此保护：连续 3 个模型轮次的工具调用全部失败，且工具、参数及错误
完全相同时，在下一次模型调用的系统提示词中要求重新规划。接下来 2 轮仍是相同失败，就提前
停止任务，记录 `repeated_tool_failure`；TUI 会显示重复工具失败且无法恢复的原因。

按完整模型轮次计数，工具内部重试和同轮重复调用不会多计。任一工具成功，或失败调用集合变化，
都会清除计数和恢复指令。返回成功状态的后台任务轮询不会触发检测。检测状态独立于消息历史，
上下文压缩不会清除它，Skill 的系统提示词覆盖也不会丢失恢复指令。其他循环仍由最大步数兜底。

---

## 超时后确认状态再重试

在本次 Run 中，可能修改状态的工具（包括 Bash 和默认视为非只读的 MCP 工具）超时后，
执行端暂停后续有副作用的调用，包括同轮模型回复中的后续调用。此时只允许只读工具进行检查，
以及使用 `resolve_tool_timeout` 提交结论。当前只读工具包括 `read_file`、`list_dir`、
`task_get`、`task_list` 和 `agent_result`；任意 Bash 命令不会因模型声称只读而被放行。

模型需要检查实际状态，并引用超时之后成功执行的只读调用 ID。确认结论有四种：

- `completed`：已经完成，阻止同一操作重复执行；Bash 仅改变 `timeout` 也不能重复执行。
- `not_applied`：依据检查结果确认没有生效，只允许一次实际重试；该次执行不做内部自动重试，
  新调用 ID 或仅改变 Bash 的超时时长也不会增加额度。同一操作再次超时则停止恢复。
- `partial` 或 `unknown`：部分完成或状态不明，停止本次执行并记录 `timeout_state_unconfirmed`。

执行端检查依据是否来自实际的成功检查，模型负责判断结果是否足以支撑结论。目前没有通用的
状态验证器、事务回滚或跨 Run 的恢复账本；不能把任意读取结果当成操作没有生效的证明。
MCP 暂未声明可信的只读工具能力，无法用现有只读工具确认外部状态时应提交 `unknown`，停止恢复。
调用方、Skill 和子 Agent 的工具白名单仍然生效；没有检查或确认工具时，不会自动扩大白名单。

Shell 超时或取消时，在 Windows 上用 `taskkill /PID <本次进程> /T /F` 清理进程树，
在 POSIX 上清理本次创建的独立进程组，并限时等待清理完成。保留已收到的输出和清理状态；
清理不能确认时禁止恢复重试。已发生的文件修改和外部副作用不会被撤销，脱离进程树的后台进程
也不属于通用回滚保证。恢复状态保存在执行上下文中，Skill 覆盖和上下文压缩不会清除限制。
TUI 使用现有工具事件展示检查过程，并在无法恢复时显示停止原因。

---

## 开发命令

```bash
uv run ruff check src tests scripts   # lint
uv run mypy src                       # 类型检查
uv run pytest tests/ -v               # 全量测试
uv run pytest tests/unit/ -v         # 仅单元测试（无需启动 daemon）

make docs                             # 重新生成 WIRE_PROTOCOL.md
make verify-s0                        # 完整验证（lint + 类型 + 测试 + 协议同源检查）
```

---

## 日志

```bash
tail -f ~/.kama/logs/core.log
```

---

## 常见错误

| 报错 | 原因 | 处理 |
|------|------|------|
| `core already running at 127.0.0.1:7437` | 已有守护进程在运行 | `kill $(pgrep -f kama-core)` |
| `core not running` | 未启动守护进程 | `uv run kama-core` |
| `Address already in use` | 端口被其他进程占用 | `KAMA_PORT=8000 uv run kama-core` |
| `Config error: KAMA_PORT must be an integer` | `.env` 或环境变量中端口值非整数 | 检查 `KAMA_PORT` 的值 |
