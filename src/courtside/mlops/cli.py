"""`courtside mlops ...` command routing.

Registry/retraining commands live in `cli_registry.py`; monitoring/checks/alerts commands in
`cli_monitor.py`. Each exposes `add_commands(sub) -> dict[str, handler]`, where `sub` is an
argparse subparsers object and `handler(args) -> int`.
"""

from __future__ import annotations

import argparse
import importlib


_MODULES = ("courtside.mlops.cli_registry", "courtside.mlops.cli_monitor")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="courtside mlops")
    sub = p.add_subparsers(dest="mlops_cmd", required=True)
    handlers: dict = {}
    for name in _MODULES:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:
                raise
            continue
        handlers.update(mod.add_commands(sub))
    args = p.parse_args(argv)
    return handlers[args.mlops_cmd](args)
