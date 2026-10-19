"""`courtside proj ...` and `courtside dfs ...` command routing.

Each command module exposes `add_commands(sub) -> dict[str, handler]`, where `sub` is an
argparse subparsers object and `handler(args) -> int` (the same pattern as `mlops/cli.py`).
Modules that don't exist yet are skipped, so tasks can land their commands independently:
add the module name to `PROJ_MODULES` or `DFS_MODULES`.
"""

from __future__ import annotations

import argparse
import importlib

PROJ_MODULES: tuple[str, ...] = (
    "courtside.proj.ids",
    "courtside.analytics.stints_store",
)
DFS_MODULES: tuple[str, ...] = (
    "courtside.proj.dfs.rules",
    "courtside.proj.dfs.capture_cli",
    "courtside.proj.dfs.optimizer_cli",
    "courtside.proj.dfs.runner_cli",
)


def _run(prog: str, modules: tuple[str, ...], argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog=prog)
    sub = p.add_subparsers(dest="sub_cmd", required=True)
    handlers: dict = {}
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:
                raise
            continue
        if hasattr(mod, "add_commands"):
            handlers.update(mod.add_commands(sub))
    if not handlers:
        p.error("no commands available yet")
    args = p.parse_args(argv)
    return handlers[args.sub_cmd](args)


def proj_main(argv: list[str]) -> int:
    return _run("courtside proj", PROJ_MODULES, argv)


def dfs_main(argv: list[str]) -> int:
    return _run("courtside dfs", DFS_MODULES, argv)
