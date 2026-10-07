"""Span-annotate sentences for check-worthiness using an LLM.

Reads a CSV containing a column of sentences, prompts an LLM to identify
check-worthy spans (verifiable factual claims), reconstructs offsets from the
returned span text, and writes a new CSV containing the original sentence plus
span annotations.

Default runtime targets Ollama's local HTTP API.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from rich.console import Console
import numpy as np

sys.path.append((Path(__file__).resolve().parent.parent / "training").as_posix())
from utils import create_progress_bar
from train import (
    compute_metrics_span_level,
    compute_metrics_token_level,
    encode,
    get_tokenizer,
)

console = Console()
progress = create_progress_bar(console)

import pandas as pd

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_NUM_EXAMPLES = 3


# Data structures for model responses and extracted spans
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Annotation:
    spans: List[Span]
    prompt_eval_duration: Optional[float] = None
    eval_duration: Optional[float] = None
    unmatched_spans_count: int = 0

    def to_public_dict(self) -> Dict[str, Any]:
        if isinstance(self.spans, list):
            spans = [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                }
                for s in self.spans
            ]
        else:
            spans = "PASS"

        return {
            "spans": spans,
            "prompt_eval_duration": self.prompt_eval_duration,
            "eval_duration": self.eval_duration,
            "unmatched_spans_count": self.unmatched_spans_count,
        }


# HTTP transport for the Ollama API


def _http_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read().decode("utf-8", errors="replace")

    return json.loads(data)


# Span parsing and normalization helpers
def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _extract_span_text(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item
    return None


def _span_pattern(span_text: str) -> str:
    pattern_parts = []
    for char in span_text:
        if char.isspace():
            if not pattern_parts or pattern_parts[-1] != r"\s+":
                pattern_parts.append(r"\s+")
        elif char in "'’":
            pattern_parts.append("['’]")
        elif char in '"“”':
            pattern_parts.append('["“”]')
        else:
            pattern_parts.append(re.escape(char))
    return "".join(pattern_parts)


def _validate_and_normalize(
    original_text: str,
    spans_raw: Any,
    prompt_eval_duration: Optional[float] = None,
    eval_duration: Optional[float] = None,
) -> Annotation:
    if spans_raw is None:
        spans_raw = []
    if not isinstance(spans_raw, list):
        raise ValueError(f"'spans' must be a list, spans_raw: {spans_raw}")

    spans: List[Span] = []
    candidate_spans: Dict[Tuple[int, int], Span] = {}
    unmatched_spans_count = 0

    for item in spans_raw:
        span_text = _extract_span_text(item)
        if not span_text:
            raise ValueError(
                f"Each span must be a string\nspans_raw: {spans_raw}\nitem: {item}"
            )

        matches = list(
            re.finditer(
                rf"(?<!\w){_span_pattern(span_text)}(?!\w)",
                original_text,
                flags=re.IGNORECASE,
            )
        )
        if not matches:
            console.print(
                f"[yellow]Warning:[/yellow] Span not found in original text: {span_text!r}"
            )
            unmatched_spans_count += 1

        for match in matches:
            start = match.start()
            matched_text = match.group(0)
            end = match.end()
            assert original_text[start:end] == matched_text, (
                f"Extracted span text does not match original text at offsets {start}-{end}: "
                f"expected {matched_text!r}, got {original_text[start:end]!r}"
            )
            candidate_spans.setdefault(
                (start, end), Span(start=start, end=end, text=matched_text)
            )

    # Resolve overlaps in text order so output ordering cannot affect matching
    for candidate in sorted(
        candidate_spans.values(),
        key=lambda span: (span.start, -(span.end - span.start)),
    ):
        interval = (candidate.start, candidate.end)
        if not any(_spans_overlap(interval, (span.start, span.end)) for span in spans):
            spans.append(candidate)

    spans.sort(key=lambda s: (s.start, s.end))
    # console.print(f"Extracted spans: {spans}")
    return Annotation(
        spans=spans,
        prompt_eval_duration=prompt_eval_duration,
        eval_duration=eval_duration,
        unmatched_spans_count=unmatched_spans_count,
    )


# CSV example formatting and prediction artifact helpers
def _parse_spans(value: Any) -> List[Tuple[int, int, str]]:
    """Parse the CSV span representation into character intervals."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        if not value.strip() or value == "PASS":
            return []
        value = ast.literal_eval(value)
    if not isinstance(value, list):
        raise ValueError(f"Expected a list of spans, got {value!r}")
    return [
        (int(span["start"]), int(span["end"]), str(span.get("text", "")))
        for span in value
        if isinstance(span, dict)
    ]


def _format_examples(examples: pd.DataFrame) -> str:
    formatted = ["Examples:"]
    for _, row in examples.iterrows():
        spans = [span[2] for span in _parse_spans(row["spans"])]
        formatted.append(f'Text: {json.dumps(str(row["text"]), ensure_ascii=False)}')
        formatted.append(f"Spans: {json.dumps(spans, ensure_ascii=False)}")
        formatted.append("")
    formatted.append("End of examples. Perform the task on the the upcoming text.")
    formatted.append("")
    return "\n".join(formatted)


def _span_dicts(value: Any) -> List[Dict[str, Any]]:
    """Return spans as JSON-compatible dictionaries for prediction artifacts."""
    return [
        {"start": start, "end": end, "text": text}
        for start, end, text in _parse_spans(value)
    ]


def _spans_to_bio(text: str, spans: Any, tokenizer: Any) -> List[int]:
    encoded = encode(text, _span_dicts(spans), tokenizer)
    return [
        label for label, mask in zip(encoded["labels"], encoded["crf_mask"]) if mask
    ]


def _compute_test_metrics(
    predictions: List[List[int]], labels: List[List[int]]
) -> Dict[str, Any]:
    metrics = compute_metrics_token_level(predictions, labels)
    metrics["span"] = compute_metrics_span_level(predictions, labels)
    return metrics


def _aggregate_test_metrics(
    fold_metrics: Dict[str, Dict[str, Any]], debates: Iterable[str]
) -> Dict[str, Any]:
    overall: Dict[str, Any] = {}
    for label in ["macro", "span", "B", "I", "O"]:
        overall[label] = {"mean": {}, "std": {}}
        for metric in ["f1", "precision", "recall"]:
            values = [fold_metrics[debate][label][metric] for debate in debates]
            overall[label]["mean"][metric] = float(np.mean(values))
            overall[label]["std"][metric] = float(np.std(values))

    values = [fold_metrics[debate]["macro"]["jaccard"] for debate in debates]
    overall["macro"]["mean"]["jaccard"] = float(np.mean(values))
    overall["macro"]["std"]["jaccard"] = float(np.std(values))
    return overall


# Ollama request and response handling
class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def annotate(
        self,
        prompt_template: str,
        text: str,
        guidelines_text: str,
        examples_text: str,
        temperature: float,
        seed: int,
    ) -> Annotation:
        prompt = prompt_template.format(
            text=text, guidelines=guidelines_text, examples=examples_text
        )
        # console.rule()
        # console.print(f"\n[blue]Text to annotate:[/blue]\n{text}\n")
        # console.print(f"\n[blue]Prompt sent to LLM:[/blue]\n{prompt}\n")
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "Return ONLY a single valid JSON array of strings.",
                },
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": float(temperature),
                "seed": seed,
                "top_k": 1,
                "top_p": 1,
            },
            "think": False,
        }

        resp = _http_json(
            f"{self.base_url}/api/chat", payload=payload, timeout_s=self.timeout_s
        )
        content = (resp.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise ValueError(
                f"Unexpected Ollama response format: missing message.content\nResponse: {resp}"
            )

        # console.print(f"\n[blue]Response:[/blue]\n{content}\n")
        array_start = content.find("[")
        array_end = content.rfind("]")
        if array_start == -1 or array_end == -1 or array_end < array_start:
            raise ValueError(
                f"Unexpected Ollama response format: missing array of spans\nContent: {content}"
            )
        content = content[array_start : array_end + 1]

        content = (
            content.replace("\“", "“")
            .replace("\”", "”")
            .replace("\‘", "‘")
            .replace("\’", "’")
        )

        try:
            spans_raw = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Unexpected Ollama response format (JSON decoding): {e}\nContent: {content}"
            )
        return _validate_and_normalize(
            text,
            spans_raw,
            prompt_eval_duration=resp.get("prompt_eval_duration"),
            eval_duration=resp.get("eval_duration"),
        )

    # Command-line configuration


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify check-worthy spans in a CSV via Ollama."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input CSV",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Name of the text column in the input CSV (default: text)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Path to prompt template text file"
    )
    parser.add_argument(
        "--guidelines",
        type=str,
        default=None,
        help="Path to guidelines text file (optional)",
    )
    parser.add_argument(
        "--examples-csv",
        type=Path,
        default=None,
        help="Annotated CSV used to sample in-context examples per fold (optional)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help=f"Number of examples sampled per fold (default: {DEFAULT_NUM_EXAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold example sampling (default: 42)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests (default: 0.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only annotate the first N rows in each fold (useful for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the LLM; output empty spans for all rows",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup requests to send to the LLM before timing (default: 0)",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the response from the LLM for debugging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    # Validate paths and establish output locations
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.examples_csv is not None and not args.examples_csv.exists():
        raise FileNotFoundError(f"Examples CSV not found: {args.examples_csv}")
    if args.num_examples < 0:
        raise ValueError("--num-examples must be non-negative")

    input_path = args.input.resolve()
    examples_path = args.examples_csv.resolve() if args.examples_csv else None
    if args.out is None:
        out_dir = (
            Path(__file__).parent
            / "output"
            / f"{args.model.replace('/', '-')}"
            / f"{input_path.stem.split('.')[0].split('_')[0]}_{datetime.now().strftime('%Y%m%d-%H%M%S')}{f' {args.comment}' if args.comment else ''}"
        ).resolve()
    else:
        out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "predictions.csv"
    folds_dir = out_dir / "folds"

    if args.guidelines:
        guidelines_path = Path(args.guidelines)
        if not guidelines_path.exists():
            raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")
        guidelines_text = (
            "Guidelines:\n" + guidelines_path.read_text(encoding="utf-8") + "\n\n"
        )
    else:
        guidelines_text = ""

    # Load the prompt and annotation data
    prompt_path = (
        Path(args.prompt) if args.prompt else Path(__file__).parent / "prompt.txt"
    )
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_path}")
    prompt_template = prompt_path.read_text(encoding="utf-8")

    df = pd.read_csv(input_path)
    examples_df = pd.read_csv(examples_path) if examples_path else None
    frames = [(df, "input")]
    if examples_df is not None:
        frames.append((examples_df, "examples"))
    for frame, name in frames:
        missing = {"debate_id", "text", "spans"} - set(frame.columns)
        if name == "input":
            missing.discard("text")
            if args.text_col not in frame.columns:
                missing.add(args.text_col)
        if missing:
            raise ValueError(
                f"{name} CSV is missing required columns: {sorted(missing)}"
            )

    df = df.sort_values(["debate_id", "id"] if "id" in df.columns else ["debate_id"])
    if examples_df is not None:
        examples_df = examples_df.drop_duplicates(subset=["debate_id", "text"])
    client = OllamaClient(base_url=args.ollama_url, model=args.model)
    tokenizer = get_tokenizer("FacebookAI/roberta-base")
    all_results: List[Dict[str, Any]] = []
    all_test_preds: Dict[str, List[List[int]]] = {}
    all_test_labels: Dict[str, List[List[int]]] = {}
    fold_metadata: List[Dict[str, Any]] = []
    fold_test_metrics: Dict[str, Dict[str, Any]] = {}
    times: Dict[str, Dict[str, List[Optional[float]]]] = {}
    failed = 0

    progress.start()
    task_warmup = progress.add_task("Warmup", total=args.warmup)
    for i in range(args.warmup):
        try:
            client.annotate(
                prompt_template=prompt_template,
                text="Warmup request",
                guidelines_text=guidelines_text,
                examples_text="Warmup examples",
                temperature=args.temperature,
                seed=args.seed,
            )
        except Exception as e:
            console.print(f"Error during warmup: {e}")
        finally:
            progress.advance(task_warmup)
    progress.remove_task(task_warmup)

    overall_start = time.perf_counter()

    # Run inference once for each held-out debate
    # folds_dir.mkdir(parents=True, exist_ok=True)

    debates = list(df["debate_id"].unique())
    task_folds = progress.add_task("Folds", total=len(debates))
    total_unmatched_spans_count = 0

    for fold_index, (test_debate, fold_df) in enumerate(
        df.groupby("debate_id", sort=True)
    ):
        failed_before_fold = failed
        fold_df = fold_df.head(args.limit) if args.limit is not None else fold_df
        if examples_df is None:
            sampled_examples = None
            examples_text = ""
        else:
            candidate_examples = examples_df[examples_df["debate_id"] != test_debate]
            sample_size = min(args.num_examples, len(candidate_examples))
            if args.num_examples > 0 and len(candidate_examples) > sample_size:
                sampled_examples = candidate_examples.sample(
                    n=sample_size,
                    random_state=args.seed + fold_index,
                )
            else:
                sampled_examples = candidate_examples
            examples_text = _format_examples(sampled_examples)
        all_test_preds[test_debate] = []
        all_test_labels[test_debate] = []
        times[test_debate] = {
            "prompt_eval_duration": [],
            "eval_duration": [],
        }
        console.rule(f"Fold {fold_index + 1}: held-out debate {test_debate}")
        console.print(
            f"Testing {len(fold_df)} rows with "
            f"{len(sampled_examples) if sampled_examples is not None else 0} examples"
        )
        if sampled_examples is not None:
            console.print(
                f"Examples sampled: {[f'{str(row['id'])}/{str(row['chunk_id'])}' for _, row in sampled_examples.iterrows()]}"
            )

        fold_results: List[Dict[str, Any]] = []
        task_rows = progress.add_task("Rows", total=len(fold_df))
        # Annotate the rows in the held-out fold
        for _, row in fold_df.iterrows():
            text = str(row[args.text_col])
            error = ""
            try:
                ann = (
                    Annotation(spans=[])
                    if args.dry_run
                    else client.annotate(
                        prompt_template=prompt_template,
                        text=text,
                        guidelines_text=guidelines_text,
                        examples_text=examples_text,
                        temperature=args.temperature,
                        seed=args.seed,
                    )
                )
            except (
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
            ) as exc:
                failed += 1
                error = str(exc)
                ann = Annotation(spans=[])
                console.print(f"[red]Error[/red] row {row.get('id', '')}: {exc}")

            ann_dict = ann.to_public_dict()
            predicted_spans = _span_dicts(ann_dict["spans"])
            gold_spans = _span_dicts(row["spans"])
            predicted_bio = _spans_to_bio(text, predicted_spans, tokenizer)
            gold_bio = _spans_to_bio(text, gold_spans, tokenizer)
            total_unmatched_spans_count += ann_dict.get("unmatched_spans_count", 0)
            times[test_debate]["prompt_eval_duration"].append(
                ann_dict["prompt_eval_duration"]
            )
            times[test_debate]["eval_duration"].append(ann_dict["eval_duration"])
            all_test_preds[test_debate].append(predicted_bio)
            all_test_labels[test_debate].append(gold_bio)
            result = {
                "debate_id": test_debate,
                "id": row.get("id", ""),
                "text": text,
                "gold_spans": row["spans"],
                "spans": json.dumps(predicted_spans, ensure_ascii=False),
                "error": error,
                "prompt_eval_duration": ann_dict["prompt_eval_duration"],
                "eval_duration": ann_dict["eval_duration"],
                "unmatched_spans_count": ann_dict["unmatched_spans_count"],
            }
            fold_results.append(result)
            all_results.append(result)
            progress.advance(task_rows)
            if args.sleep:
                time.sleep(args.sleep)

        progress.remove_task(task_rows)
        progress.advance(task_folds)

        fold_output = pd.DataFrame(fold_results)
        # fold_output.to_csv(folds_dir / f"{test_debate}.csv", index=False)
        fold_predictions = [
            _spans_to_bio(result["text"], result["spans"], tokenizer)
            for result in fold_results
        ]
        fold_labels = [
            _spans_to_bio(row[args.text_col], row["spans"], tokenizer)
            for _, row in fold_df.iterrows()
        ]
        fold_test_metrics[test_debate] = _compute_test_metrics(
            fold_predictions, fold_labels
        )
        fold_metadata.append(
            {
                "debate_id": test_debate,
                "test_metrics": fold_test_metrics[test_debate],
                "test": {
                    "rows": len(fold_output),
                    "output": f"folds/{test_debate}.csv",
                },
                "examples": (
                    len(sampled_examples) if sampled_examples is not None else 0
                ),
                "failed_rows": failed - failed_before_fold,
            }
        )
        console.print(
            f"Test: "
            f"M-F1={fold_test_metrics[test_debate]['macro']['f1']*100:.2f}% "
            f"M-P={fold_test_metrics[test_debate]['macro']['precision']*100:.1f}% "
            f"M-R={fold_test_metrics[test_debate]['macro']['recall']*100:.1f}% "
            f"B-F1={fold_test_metrics[test_debate]['B']['f1']*100:.2f}% "
            f"B-P={fold_test_metrics[test_debate]['B']['precision']*100:.1f}% "
            f"B-R={fold_test_metrics[test_debate]['B']['recall']*100:.1f}% "
            f"I-F1={fold_test_metrics[test_debate]['I']['f1']*100:.2f}% "
            f"I-P={fold_test_metrics[test_debate]['I']['precision']*100:.1f}% "
            f"I-R={fold_test_metrics[test_debate]['I']['recall']*100:.1f}% "
        )
    progress.remove_task(task_folds)
    progress.stop()

    # Save predictions, fold outputs, and run metadata
    out_df = pd.DataFrame(all_results)
    out_df.to_csv(output_path, index=False)
    test_preds_labels_path = out_dir / "test_preds_labels.json"
    test_preds_labels_path.write_text(
        json.dumps(
            {"preds": all_test_preds, "labels": all_test_labels},
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    times_path = out_dir / "times.json"
    times_path.write_text(
        json.dumps(times, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    debates = list(fold_test_metrics)
    overall_test_metrics = _aggregate_test_metrics(fold_test_metrics, debates)
    summary = {
        "input": str(input_path),
        "examples_csv": str(examples_path) if examples_path else None,
        "model": args.model,
        "seed": args.seed,
        "num_examples": args.num_examples,
        "folds": fold_metadata,
        "overall": {
            "rows": len(out_df),
            "failed_rows": failed,
            "unmatched_spans_count": total_unmatched_spans_count,
            "wall_time_seconds": time.perf_counter() - overall_start,
        },
    }
    results = {
        debate: {
            "test_metrics": fold_test_metrics[debate],
        }
        for debate, metadata in ((fold["debate_id"], fold) for fold in fold_metadata)
    }
    results["overall"] = {
        "test": overall_test_metrics,
        "training_time_seconds": time.perf_counter() - overall_start,
    }

    console.print(
        f"Overall test metrics: {json.dumps(overall_test_metrics, indent=2, ensure_ascii=False)}"
    )

    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    average_prompt_eval_duration = np.mean(
        [
            t
            for fold in times.values()
            for t in fold["prompt_eval_duration"]
            if t is not None
        ]
    )
    std_prompt_eval_duration = np.std(
        [
            t
            for fold in times.values()
            for t in fold["prompt_eval_duration"]
            if t is not None
        ]
    )
    average_eval_duration = np.mean(
        [t for fold in times.values() for t in fold["eval_duration"] if t is not None]
    )
    std_eval_duration = np.std(
        [t for fold in times.values() for t in fold["eval_duration"] if t is not None]
    )
    console.print(
        f"Average prompt evaluation duration: {average_prompt_eval_duration/(10**6):.4f} (±{std_prompt_eval_duration/(10**6):.4f}) ms"
    )
    console.print(
        f"Average evaluation duration: {average_eval_duration/(10**6):.4f} (±{std_eval_duration/(10**6):.4f}) ms"
    )

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as summary_file:
        summary_file.write(
            f"Rows: {len(out_df)}\n"
            f"Failed rows: {failed}\n"
            f"Unmatched spans count: {total_unmatched_spans_count}\n"
            f"Average prompt evaluation duration: {average_prompt_eval_duration/(10**6):.4f} (±{std_prompt_eval_duration/(10**6):.4f}) ms\n"
            f"Average evaluation duration: {average_eval_duration/(10**6):.4f} (±{std_eval_duration/(10**6):.4f}) ms\n"
        )
    console.print(f"Wrote: {output_path} ({len(out_df)} rows)")
    # console.print(f"Wrote fold outputs: {folds_dir}")
    console.print(f"Wrote predictions and labels: {test_preds_labels_path}")
    console.print(f"Wrote timings: {times_path}")
    console.print(f"Wrote results: {results_path}")
    console.print(f"Wrote run summary: {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
