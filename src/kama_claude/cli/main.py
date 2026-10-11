from __future__ import annotations

import argparse
import sys

from kama_claude.cli.commands.chat import cmd_chat
from kama_claude.cli.commands.core import cmd_core_start, cmd_core_status, cmd_core_stop
from kama_claude.cli.commands.ping import cmd_ping
from kama_claude.cli.commands.rag import cmd_rag_index, cmd_rag_search, cmd_rag_status
from kama_claude.cli.commands.run import cmd_run
from kama_claude.cli.commands.trace import cmd_trace
from kama_claude.cli.commands.version import cmd_version
from kama_claude.core.config import get_config
from kama_claude.core.logging_setup import setup_logging


# CLI 主入口：解析命令行参数并分发到对应子命令
def main() -> None:
    parser = argparse.ArgumentParser(prog="kama", description="KamaClaude CLI")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ping", help="Ping the core daemon")
    subparsers.add_parser("chat", help="Start a multi-turn chat session")

    run_parser = subparsers.add_parser("run", help="Run an agent task")
    run_parser.add_argument("--goal", required=True, help="Goal for the agent to accomplish")

    core_parser = subparsers.add_parser("core", help="Manage the core daemon")
    core_sub = core_parser.add_subparsers(dest="core_command")
    core_sub.add_parser("start", help="Start the daemon in the background")
    core_sub.add_parser("stop", help="Stop the running daemon")
    core_sub.add_parser("status", help="Show daemon status")

    trace_parser = subparsers.add_parser("trace", help="View system trace log")
    trace_parser.add_argument("run_id", nargs="?", default=None, help="Filter by run ID")
    trace_parser.add_argument("--layer", choices=["ipc", "event", "llm"], help="Filter by layer")
    trace_parser.add_argument("--direction", help="Filter by direction (e.g. CORE→LLM)")
    trace_parser.add_argument("--raw", action="store_true", help="Output raw NDJSON")
    trace_parser.add_argument("--follow", "-f", action="store_true", help="Follow new records")

    rag_parser = subparsers.add_parser("rag", help="Manage the local Obsidian RAG index")
    rag_sub = rag_parser.add_subparsers(dest="rag_command")
    rag_index_parser = rag_sub.add_parser("index", help="Build or update the RAG index")
    rag_index_parser.add_argument("--force", action="store_true", help="Rebuild every vector")
    rag_sub.add_parser("status", help="Show RAG index status")
    rag_search_parser = rag_sub.add_parser("search", help="Test knowledge retrieval")
    rag_search_parser.add_argument("query", help="Question or keywords to search")
    rag_search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if args.version:
        cmd_version()
        return

    config = get_config()
    setup_logging(config)

    if args.command == "ping":
        cmd_ping(config)
    elif args.command == "chat":
        cmd_chat(config)
    elif args.command == "run":
        cmd_run(args.goal, config)
    elif args.command == "core":
        if args.core_command == "start":
            cmd_core_start(config)
        elif args.core_command == "stop":
            cmd_core_stop(config)
        elif args.core_command == "status":
            cmd_core_status(config)
        else:
            core_parser.print_help()
            sys.exit(1)
    elif args.command == "trace":
        cmd_trace(
            args.run_id,
            config,
            layer=args.layer,
            direction=args.direction,
            raw=args.raw,
            follow=args.follow,
        )
    elif args.command == "rag":
        if args.rag_command == "index":
            cmd_rag_index(config, force=args.force)
        elif args.rag_command == "status":
            cmd_rag_status(config)
        elif args.rag_command == "search":
            cmd_rag_search(config, args.query, top_k=args.top_k)
        else:
            rag_parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
