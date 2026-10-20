#!/usr/bin/env python3
"""Block agent tool calls that would expose gated match data to a cloud-hosted LLM.

Reads one tool call as JSON on stdin: {"tool_name": str, "tool_input": {...}} (Claude Code hook
shape; the pi extension sends the same shape). Rules live in config/data-guard.json.
Exit 0 = allow. Exit 2 = block, reason on stderr (Claude Code shows it to the model).
Stdlib only, so it runs under any python3 without the project env. Fails closed.
"""
import json
import re
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "data-guard.json"
PATH_KEYS = ("path", "file_path", "notebook_path")


def load_rules(config_path=CONFIG_PATH):
    config = json.loads(config_path.read_text())
    blocked = [(re.compile(p["regex"], re.IGNORECASE), p["why"]) for p in config["blockedPatterns"]]
    allowed = [re.compile(p["regex"]) for p in config["allowedCommands"]]
    return blocked, allowed, set(config["pathOnlyTools"])


def text_to_check(tool_name, tool_input, path_only_tools):
    """Write/edit tools are judged by their target path only: their content may mention data/ in prose."""
    if tool_name in path_only_tools:
        return " ".join(str(tool_input.get(k, "")) for k in PATH_KEYS)
    return json.dumps(tool_input, ensure_ascii=False)


def check(tool_name, tool_input, rules):
    """Return None when allowed, else the reason for blocking."""
    blocked, allowed, path_only_tools = rules
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if isinstance(command, str) and any(a.match(command.strip()) for a in allowed):
        return None
    text = text_to_check(tool_name, tool_input if isinstance(tool_input, dict) else {}, path_only_tools)
    for pattern, why in blocked:
        match = pattern.search(text)
        if match:
            return (
                f"DATA GUARD: blocked {tool_name} -- it references '{match.group(0)}' ({why}). "
                "Gated match data must never enter a cloud LLM context (docs/DATA_POLICY.md). "
                "Write a script that reads the data itself and prints only aggregate results, "
                "and run it without naming data paths (e.g. scripts/py <script>.py)."
            )
    return None


def main():
    try:
        call = json.load(sys.stdin)
        reason = check(call.get("tool_name", ""), call.get("tool_input") or {}, load_rules())
    except Exception as error:  # fail closed: a broken guard must not let data through
        reason = f"DATA GUARD: guard error, blocking as a precaution: {error!r}"
    if reason:
        print(reason, file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
