"""Rename a choice value inside Label Studio `reason_choice` annotations.

Example:
  python labelstudio/rename_reason_choice.py export.json old_name new_name

By default this writes a new file next to the input named
`<input>.renamed.json`. Use `--in-place` to overwrite the input file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rename a reason_choice value inside a Label Studio export JSON"
    )
    p.add_argument("input_json", type=Path, help="Label Studio export JSON file")
    p.add_argument("old_choice", type=str, help="Existing choice value to rename")
    p.add_argument("new_choice", type=str, help="Replacement choice value")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <input>.renamed.json)",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a new file",
    )
    return p.parse_args(argv)


def _rename_choice(node: Any, old_choice: str, new_choice: str) -> int:
    """Recursively rename matching reason_choice values.

    Only `choices` results whose `from_name` is `reason_choice` are updated.
    """
    changes = 0
    if isinstance(node, dict):
        if node.get("type") == "choices" and node.get("from_name") == "reason_choice":
            value = node.get("value")
            if isinstance(value, dict):
                choices = value.get("choices")
                if isinstance(choices, list):
                    updated_choices = []
                    for choice in choices:
                        choice_text = str(choice)
                        if choice_text == old_choice:
                            updated_choices.append(new_choice)
                            changes += 1
                        else:
                            updated_choices.append(choice_text)
                    value["choices"] = updated_choices
                elif isinstance(choices, str) and choices == old_choice:
                    value["choices"] = new_choice
                    changes += 1
        for child in node.values():
            changes += _rename_choice(child, old_choice, new_choice)
    elif isinstance(node, list):
        for item in node:
            changes += _rename_choice(item, old_choice, new_choice)
    return changes


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input_json.exists():
        raise FileNotFoundError(f"Input not found: {args.input_json}")
    if args.in_place:
        out_path = args.input_json
    else:
        out_path = args.out or args.input_json.with_name(
            f"{args.input_json.stem}{'.renamed' if not str(args.input_json).endswith('.renamed.json') else ''}{args.input_json.suffix}"
        )

    loaded = json.loads(args.input_json.read_text(encoding="utf-8"))
    changes = _rename_choice(loaded, args.old_choice, args.new_choice)
    out_path.write_text(json.dumps(loaded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({changes} choice value(s) renamed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
