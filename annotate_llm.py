"""Annotate sentence check-worthiness with a single claim span using an LLM.

Reads a CSV containing a column of sentences, prompts an LLM to return the
check-worthy span text itself (or `NULL` if none), and writes a new CSV
containing the original sentence plus normalized annotation columns.

Default runtime targets Ollama's local HTTP API with Llama 3.2 3B.

Example:
	python annotate_llm.py transcripts/20240910_trump_harris.csv --out annotated.csv

Prereqs (Ollama):
	1) Install Ollama: https://ollama.com
	2) Pull model: ollama pull llama3.2:3b
	3) Ensure Ollama is running (default: http://localhost:11434)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
	from tqdm import tqdm  # type: ignore
	HAS_TQDM = True
except Exception:  # pragma: no cover
	tqdm = None  # type: ignore
	HAS_TQDM = False


DEFAULT_MODEL = "mistral:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


# Keep the prompt in one place so you can tweak it later.
# This is the “example query” the user asked for.
EXAMPLE_QUERY_TEMPLATE = (
	"You are a meticulous fact-checking journalist. "
	"Task: Identify the single check-worthy span in the sentence below, if it exists. "
	"A check-worthy span is a verifiable factual claim (numbers, dates, events, "
	"attributions, policies, measurable comparisons) that would be worth investigating for truthfulness. "
	"Do NOT mark opinions, insults, vague rhetoric, or pure value judgments unless "
	"they contain a concrete factual claim.\n\n"
	"Return ONLY valid JSON with this exact schema:\n"
	"{{\n"
	"  \"check_worthy_span\": \"...\"\n"
	"}}\n\n"
	"Rules:\n"
	"- Return ONLY one string field named check_worthy_span.\n"
	"- If a check-worthy claim exists, the value must be an exact substring of the original sentence.\n"
	"- Prefer the smallest span that captures the strongest check-worthy claim.\n"
	"- Named entities, places and numbers are only check-worthy if they are part of a factual claim "
 	"(e.g. \"New York\" is not check-worthy by itself, but \"Crime in New York has increased by 10%\" contains a check-worthy span that includes the entity).\n"
	"- If no check-worthy claim exists, set check_worthy_span to the exact string \"NULL\".\n\n"
	"Examples:\n"
	"Sentence:\nThe U.S. economy grew by 2.5% in the last quarter.\n"
	"Annotation:\n{{\"check_worthy_span\": \"U.S. economy grew by 2.5%\"}}\n\n"
	"Sentence:\nI think the new policy is terrible and will hurt a lot of people.\n"
	"Annotation:\n{{\"check_worthy_span\": \"NULL\"}}\n\n"
	"Sentence:\n{sentence}"
)


@dataclass(frozen=True)
class Annotation:
	check_worthy: bool
	check_worthy_span: str

	def to_public_dict(self) -> Dict[str, Any]:
		return {
			"check_worthy": bool(self.check_worthy),
			"check_worthy_span": self.check_worthy_span,
		}


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


def _extract_json_object(text: str) -> str:
	"""Extract a JSON object from model text.

	We instruct the model to return ONLY JSON, but in practice it sometimes wraps
	it in code fences or preambles. This tries to recover the first {...} block.
	"""
	text = text.strip()
	if not text:
		raise ValueError("Empty model response")

	# Remove common code fence wrappers.
	if text.startswith("```"):
		lines = text.splitlines()
		# Drop the opening ``` or ```json
		lines = lines[1:]
		# Drop closing ``` if present
		if lines and lines[-1].strip().startswith("```"):
			lines = lines[:-1]
		text = "\n".join(lines).strip()

	# Find the first '{' and the last '}' and try that slice.
	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError(f"No JSON object found in response: {text[:200]!r}")
	return text[start : end + 1]


def _validate_and_normalize(sentence: str, obj: Dict[str, Any]) -> Annotation:
	raw_span = obj.get("check_worthy_span")
	if raw_span is None:
		raw_span = "NULL"
	if not isinstance(raw_span, str):
		raw_span = str(raw_span)

	span = raw_span.strip()
	if not span:
		span = "NULL"

	if span.upper() == "NULL":
		return Annotation(check_worthy=False, check_worthy_span="")

	# Normalize to an exact substring when the model drifts in punctuation/spacing.
	if span not in sentence:
		span_lower = span.lower()
		sentence_lower = sentence.lower()
		at = sentence_lower.find(span_lower)
		if at >= 0:
			span = sentence[at : at + len(span)]
		else:
			# If we cannot reliably map the text back to the sentence, treat as no span.
			span = "NULL"

	if span == "NULL":
		return Annotation(check_worthy=False, check_worthy_span="")

	return Annotation(check_worthy=True, check_worthy_span=span)


class OllamaClient:
	def __init__(self, base_url: str, model: str, timeout_s: int = 120):
		self.base_url = base_url.rstrip("/")
		self.model = model
		self.timeout_s = timeout_s

	def annotate(self, sentence: str, prompt_template: str, temperature: float) -> Annotation:
		prompt = prompt_template.format(sentence=sentence)
		payload = {
			"model": self.model,
			"stream": False,
			"messages": [
				{
					"role": "system",
					"content": "You output only JSON. No extra text.",
				},
				{"role": "user", "content": prompt},
			],
			"options": {
				"temperature": float(temperature),
			},
		}

		resp = _http_json(f"{self.base_url}/api/chat", payload=payload, timeout_s=self.timeout_s)
		content = (resp.get("message") or {}).get("content")
		if not isinstance(content, str):
			raise ValueError("Unexpected Ollama response format: missing message.content")

		json_text = _extract_json_object(content)
		obj = json.loads(json_text)
		if not isinstance(obj, dict):
			raise ValueError("Expected a JSON object")
		return _validate_and_normalize(sentence, obj)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Span-annotate a sentence CSV for check-worthiness via Llama 3.2 3B.")
	parser.add_argument(
		"input",
		type=Path,
		help="Path to input .csv containing a column of sentences",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Output .csv path (default: <input>_annotated.csv)",
	)
	parser.add_argument(
		"--sentence-col",
		type=str,
		default="sentence",
		help="Name of the sentence column in the input CSV (default: sentence)",
	)
	parser.add_argument(
		"--sample-frac",
		type=float,
		default=1.0,
		help="Fraction of candidate rows to annotate (random sample, default: 1.0)",
	)
	parser.add_argument(
		"--sample-seed",
		type=int,
		default=0,
		help="Random seed for sampling (default: 0)",
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help="If output CSV exists, load it and only annotate remaining rows",
	)
	parser.add_argument(
		"--redo-failed",
		action="store_true",
		help="With --resume, re-annotate rows that previously had a non-empty error",
	)
	parser.add_argument(
		"--model",
		type=str,
		default=DEFAULT_MODEL,
		help=f"Ollama model name (default: {DEFAULT_MODEL})",
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
		help="Only annotate the first N rows (useful for testing)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Do not call the LLM; output check_worthy_span='NULL' for all rows",
	)
	return parser.parse_args(argv)


def _iter_sentences(df: pd.DataFrame, sentence_col: str, limit: Optional[int]) -> Iterable[Tuple[int, str]]:
	if sentence_col not in df.columns:
		raise KeyError(f"Sentence column not found: {sentence_col!r}. Available: {list(df.columns)!r}")

	# If this is a previously-produced annotated CSV, prefer its stable row id column.
	row_series = df["row"] if "row" in df.columns else None

	count = 0
	for idx, value in df[sentence_col].items():
		if limit is not None and count >= limit:
			break

		row_i: Optional[int]
		if row_series is not None:
			row_val = row_series.loc[idx]
			if row_val is None:
				continue
			try:
				row_i = int(row_val)
			except Exception:
				continue
		else:
			row_i = int(idx)

		if value is None:
			continue
		sentence = str(value).strip()
		if not sentence:
			continue
		yield row_i, sentence
		count += 1


def _collect_sentences(df: pd.DataFrame, sentence_col: str, limit: Optional[int]) -> List[Tuple[int, str]]:
	return list(_iter_sentences(df, sentence_col=sentence_col, limit=limit))


def _load_existing_results(path: Path) -> Dict[int, Dict[str, Any]]:
	"""Load an existing output CSV (if it has a 'row' column) into a dict by row id."""
	df_prev = pd.read_csv(path)
	if "row" not in df_prev.columns:
		return {}

	results_by_row: Dict[int, Dict[str, Any]] = {}
	for rec in df_prev.to_dict(orient="records"):
		row_val = rec.get("row")
		if row_val is None or (isinstance(row_val, float) and math.isnan(row_val)):
			continue
		try:
			row_i = int(row_val)
		except Exception:
			continue
		results_by_row[row_i] = rec
	return results_by_row


def _done_rows_from_existing(df_prev: pd.DataFrame, redo_failed: bool) -> set[int]:
	"""Return set of row ids considered already annotated in an existing output file."""
	if "row" not in df_prev.columns:
		return set()

	# If the file is a partially-filled output (same schema), treat a row as done
	# only if it has non-null annotation fields.
	base_done = pd.Series(True, index=df_prev.index)
	if "check_worthy" in df_prev.columns and "check_worthy_span" in df_prev.columns:
		# Treat rows with check_worthy=False as complete even when the span is empty.
		cw = df_prev["check_worthy"]
		cw_str = cw.astype(str).str.strip().str.lower()
		cw_false = cw_str.isin({"false", "0", "0.0", "no", "n"})
		base_done = cw.notna() & (cw_false | df_prev["check_worthy_span"].notna())
	elif "check_worthy" in df_prev.columns and "spans" in df_prev.columns:
		# Backward compatibility with older output schema.
		base_done = df_prev["check_worthy"].notna() & df_prev["spans"].notna()

	if redo_failed and "error" in df_prev.columns:
		error_ok = df_prev["error"].isna() | df_prev["error"].astype(str).str.strip().eq("")
		done_mask = base_done & error_ok
	else:
		done_mask = base_done

	done_rows: set[int] = set()
	for v in df_prev.loc[done_mask, "row"].dropna().tolist():
		try:
			done_rows.add(int(v))
		except Exception:
			continue
	return done_rows


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	out_path = args.out
	if out_path is None:
		out_path = args.input.with_name(f"{args.input.stem}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

	df = pd.read_csv(args.input)

	if not (0.0 <= float(args.sample_frac) <= 1.0):
		raise ValueError("--sample-frac must be between 0.0 and 1.0")

	client = OllamaClient(base_url=args.ollama_url, model=args.model)

	results_by_row: Dict[int, Dict[str, Any]] = {}
	failed: int = 0

	# Optionally resume from an existing output CSV.
	done_rows: set[int] = set()
	if args.resume and out_path.exists():
		df_prev = pd.read_csv(out_path)
		results_by_row = _load_existing_results(out_path)
		done_rows = _done_rows_from_existing(df_prev, redo_failed=bool(args.redo_failed))

	items = _collect_sentences(df, args.sentence_col, args.limit)
	items = [(row_i, s) for (row_i, s) in items if row_i not in done_rows]

	if float(args.sample_frac) == 0.0:
		items = []
	elif float(args.sample_frac) < 1.0 and items:
		rng = random.Random(int(args.sample_seed))
		k = int(math.ceil(float(args.sample_frac) * len(items)))
		k = max(1, min(k, len(items)))
		items = rng.sample(items, k)
	iterable: Iterable[Tuple[int, str]]
	if tqdm is not None and sys.stderr.isatty():
		iterable = tqdm(items, total=len(items), unit="sent", desc="Annotating")
	else:
		iterable = items
		if (not HAS_TQDM) and sys.stderr.isatty():
			print("Tip: install tqdm for a progress bar: pip install tqdm", file=sys.stderr)

	for row_i, sentence in iterable:
		if args.dry_run:
			ann = Annotation(check_worthy=False, check_worthy_span="")
		else:
			try:
				ann = client.annotate(sentence, prompt_template=EXAMPLE_QUERY_TEMPLATE, temperature=args.temperature)
			except (ValueError, json.JSONDecodeError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
				failed += 1
				ann = Annotation(check_worthy=False, check_worthy_span="")
				results_by_row[int(row_i)] = {
					"row": int(row_i),
					"sentence": sentence,
					"check_worthy": ann.check_worthy,
					"check_worthy_span": ann.check_worthy_span,
					"error": str(e),
				}
				if args.sleep:
					time.sleep(args.sleep)
				continue

		results_by_row[int(row_i)] = {
			"row": int(row_i),
			"sentence": sentence,
			"check_worthy": ann.check_worthy,
			"check_worthy_span": ann.check_worthy_span,
			"error": "",
		}

		if args.sleep:
			time.sleep(args.sleep)

	rows_sorted = [results_by_row[k] for k in sorted(results_by_row.keys())]
	out_df = pd.DataFrame(rows_sorted)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(out_path, index=False)

	# Helpful terminal summary.
	print(f"Wrote: {out_path} ({len(out_df)} rows). Failed rows: {failed}.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
