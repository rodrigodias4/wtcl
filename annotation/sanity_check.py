"""Sanity-check Label Studio export JSON span boundaries.

This script reports spans that:
- end with trailing punctuation or whitespace,
- begin or end in the middle of a word.

Example:
  python labelstudio/sanity_check.py labelstudio/exports/project-2-at-2026-06-07-18-49-96127a47.json
"""

from __future__ import annotations

import argparse
import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Issue:
    task_id: Any
    annotation_id: Any
    result_id: Any
    start: int
    end: int
    issue: str
    span_text: str
    source_text: str
    row_id: Any


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check Label Studio export JSON span boundaries"
    )
    parser.add_argument("export_json", type=Path, help="Label Studio export JSON file")
    parser.add_argument(
        "--label-type",
        default="labels",
        choices=["labels"],
        help="Result type to inspect for spans (default: labels)",
    )
    return parser.parse_args(argv)


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON file, got: {path}")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        return [task for task in loaded if isinstance(task, dict)]
    if isinstance(loaded, dict):
        if isinstance(loaded.get("tasks"), list):
            return [task for task in loaded["tasks"] if isinstance(task, dict)]
        if isinstance(loaded.get("data"), list):
            return [task for task in loaded["data"] if isinstance(task, dict)]
    raise ValueError(
        "Expected a Label Studio export as a list of tasks or a wrapped dict"
    )


def _pick_annotation(task: dict[str, Any]) -> Optional[dict[str, Any]]:
    annotations = task.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        return None
    for annotation in reversed(annotations):
        if isinstance(annotation, dict):
            return annotation
    return None


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _tail_issue(text: str) -> Optional[str]:
    if not text:
        return None
    if text[-1].isspace():
        return "trailing whitespace"
    if text[-1] in string.punctuation and text[-1] not in ("%", "$", '"', "'"):
        return "trailing punctuation"
    return None


def _boundary_issues(
    source_text: str, start: int, end: int, span_text: str
) -> list[str]:
    issues: list[str] = []
    if start > 0 and start <= len(source_text):
        before = source_text[start - 1]
        at_start = source_text[start]
        if _is_word_char(before) and _is_word_char(at_start):
            issues.append("starts in the middle of a word")
    if 0 <= end < len(source_text):
        before_end = source_text[end - 1]
        at_end = source_text[end]
        if _is_word_char(before_end) and _is_word_char(at_end):
            issues.append("ends in the middle of a word")

    tail_issue = _tail_issue(span_text)
    if tail_issue is not None:
        issues.append(tail_issue)
    return issues


def _iter_spans(tasks: Iterable[dict[str, Any]]) -> Iterable[Issue]:
    for task in tasks:
        task_id = task.get("id")
        data = task.get("data") if isinstance(task.get("data"), dict) else {}
        source_text = str(data.get("text") or data.get("sentence") or "")

        annotation = _pick_annotation(task)
        if annotation is None:
            continue
        annotation_id = annotation.get("id")
        results = annotation.get("result")
        if not isinstance(results, list):
            continue

        for result in results:
            if not isinstance(result, dict):
                continue
            if result.get("type") != "labels":
                continue

            value = result.get("value")
            if not isinstance(value, dict):
                continue

            start = _as_int(value.get("start"))
            end = _as_int(value.get("end"))
            span_text = value.get("text")
            if start is None or end is None:
                continue
            if not isinstance(span_text, str):
                span_text = "" if span_text is None else str(span_text)

            issues = _boundary_issues(source_text, start, end, span_text)

            result_id = result.get("id")
            for issue in issues:
                yield Issue(
                    task_id=task_id,
                    annotation_id=annotation_id,
                    result_id=result_id,
                    start=start,
                    end=end,
                    issue=issue,
                    span_text=span_text,
                    source_text=(
                        source_text[start:end]
                        if 0 <= start <= end <= len(source_text)
                        else ""
                    ),
                    row_id=data.get("id"),
                )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.export_json.exists():
        raise FileNotFoundError(f"Export not found: {args.export_json}")

    tasks = _load_tasks(args.export_json)
    issues = list(_iter_spans(tasks))

    if not issues:
        print(f"No span issues found in {args.export_json}")
        return 0

    print(f"Found {len(issues)} span issue(s) in {args.export_json}")
    for issue in issues:
        print(
            f"id={issue.row_id} task={issue.task_id} annotation={issue.annotation_id} result={issue.result_id} "
            f"[{issue.start}, {issue.end}) {issue.issue}"
        )
        print(f"  span : {issue.span_text!r}")
        print(f"  slice: {issue.source_text!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
