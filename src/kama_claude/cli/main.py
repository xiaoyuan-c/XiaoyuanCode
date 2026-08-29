from __future__ import annotations

import argparse
import sys

from kama_claude.cli.commands.ping import cmd_ping
from kama_claude.cli.commands.version import cmd_version
from kama_claude.core.config import get_config
from kama_claude.core.logging_setup import setup_logging


# CLI 主入口：解析命令行参数并分发到对应子命令
def main() -> None:
    # 添加命令 kama
    parser = argparse.ArgumentParser(prog="kama", description="KamaClaude CLI")
    # 给命令 kama 添加参数 --version，动作是 store_true
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    # 创建子命令的容器
    subparsers = parser.add_subparsers(dest="command")
    # 给命令 kama 添加子命令 ping
    subparsers.add_parser("ping", help="Ping the core daemon")

    args = parser.parse_args() #  解析得到的命令行参数

    if args.version: # 如果命令 kama 有参数 --version
        cmd_version()
        return

    if args.command == "ping": # 如果命令 kama 的子命令是 ping，也就是 kama ping
        config = get_config()
        setup_logging(config)
        cmd_ping(config)
    else:
        parser.print_help() # 否则，打印帮助信息并退出
        sys.exit(1)
