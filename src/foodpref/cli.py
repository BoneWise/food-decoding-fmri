"""Shared JSON configuration loading for command-line entry points.

JSON ``arguments`` keys use argparse destination names (underscores). Explicit
CLI arguments take precedence. Paths are relative to the working directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_with_config(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", help="JSON configuration file; CLI values override its arguments.")
    known, _ = probe.parse_known_args(argv)
    parser.add_argument("--config", help="JSON configuration file; CLI values override its arguments.")
    if known.config:
        payload = json.loads(Path(known.config).expanduser().read_text(encoding="utf-8"))
        values = payload.get("arguments", payload)
        if not isinstance(values, dict):
            parser.error("JSON configuration must contain an arguments object.")
        actions = {action.dest: action for action in parser._actions}
        unknown = set(values) - set(actions)
        if unknown or "config" in values:
            parser.error(f"Unknown or nested configuration arguments: {sorted(unknown | ({'config'} & set(values)))}")
        checked = {}
        for name, value in values.items():
            action = actions[name]
            if value is not None and action.type is not None:
                try:
                    value = action.type(value)
                except (TypeError, ValueError) as exc:
                    parser.error(f"Invalid configuration value for {name}: {exc}")
            if value is not None and action.choices is not None and value not in action.choices:
                parser.error(f"Invalid configuration value for {name}: {value!r}")
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)) and not isinstance(value, bool):
                parser.error(f"Configuration value for {name} must be true or false.")
            if action.required and value is None:
                parser.error(f"Required configuration value {name} cannot be null.")
            if name in values:
                action.required = False
            checked[name] = value
        parser.set_defaults(**checked)
    return parser.parse_args(argv)
