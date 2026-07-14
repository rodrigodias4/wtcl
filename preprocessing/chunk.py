from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
import json
import argparse
from utils import console, progress

MODEL = "distilroberta-base"


def truncate(id: int, text: str, spans: list, tokenizer: AutoTokenizer, max_length):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Build token-level BIO labels
    labels = ["O"] * len(tokens)

    for span in spans:
        span_start = span["start"]
        span_end = span["end"]

        """ if span_end > max_length:
            raise ValueError(f"Span end {span_end} exceeds max_length {max_length} for row {id}. Consider using the chunking strategy instead.") """

        matching_tokens = []

        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start < span_end and tok_end > span_start:
                matching_tokens.append(i)

        if matching_tokens:
            labels[matching_tokens[0]] = "B"
            for idx in matching_tokens[1:]:
                labels[idx] = "I"

    return {"input_ids": input_ids[:max_length], "labels": labels[:max_length]}


def chunk_spans(
    id: int,
    text: str,
    spans: list,
    tokenizer: AutoTokenizer,
    stride: int,
    max_length: int,
):
    """
    Splits text into span-safe chunks while preserving raw text and remapped spans.
    """

    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    if len(input_ids) <= max_length:
        return [{"text": text, "spans": spans}]

    # -----------------------------
    # Build span token ranges
    # -----------------------------
    span_ranges = []

    for span in spans:
        token_range = []

        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start < span["end"] and tok_end > span["start"]:
                token_range.append(i)

        if token_range:
            span_ranges.append((token_range[0], token_range[-1] + 1))

    def build_sentence_boundaries(tokens):
        sentence_end_punct = {".", "!", "?"}
        return [i + 1 for i, tok in enumerate(tokens) if tok in sentence_end_punct]

    sentence_boundaries = build_sentence_boundaries(tokens)

    def snap_to_sentence(boundary, min_boundary=0):
        valid = [b for b in sentence_boundaries if b <= boundary]
        if not valid:
            return boundary
        return max(valid[-1], min_boundary)

    def make_span_safe(boundary, min_boundary=0):
        while boundary > min_boundary and any(
            s <= boundary < e for s, e in span_ranges
        ):
            boundary -= 1
        return boundary

    # -----------------------------
    # Chunking loop
    # -----------------------------
    chunks = []
    start = 0
    n = len(input_ids)

    while start < n:
        end = min(start + max_length, n)

        end = snap_to_sentence(end)
        end = make_span_safe(end)

        if end <= start:
            end = min(start + max_length, n)

        chunk_start_char = offsets[start][0]
        chunk_end_char = offsets[end - 1][1]

        chunk_text = text[chunk_start_char:chunk_end_char]

        chunk_spans = []
        for span in spans:
            if span["start"] < chunk_end_char and span["end"] > chunk_start_char:
                span_start = max(span["start"], chunk_start_char) - chunk_start_char
                span_end = min(span["end"], chunk_end_char) - chunk_start_char
                chunk_spans.append(
                    {
                        "start": span_start,
                        "end": span_end,
                        "text": chunk_text[span_start:span_end],
                    }
                )

        chunks.append(
            {
                "text": chunk_text,
                "spans": chunk_spans,
            }
        )

        next_start = snap_to_sentence(start + stride, min_boundary=start + stride // 2)
        next_start = make_span_safe(next_start, min_boundary=start + stride // 2)

        if next_start <= start:
            break

        start = next_start

    assert len(chunks) > 0, "No chunks created."

    console.print(
        f"Row {id}: Tokenized into {len(tokens)} tokens, "
        f"{len(spans)} spans, resulting in {len(chunks)} chunks."
    )

    if chunks[1]["spans"] != []:
        console.print(
            f"Example span in chunk 1: {chunks[1]['text'][chunks[1]['spans'][0]['start']:chunks[1]['spans'][0]['end']]}"
        )

    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert spans to BIO format")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=MODEL,
        help="Tokenizer model name (only for checking tokenization > max_length)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for chunking",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length for each chunk",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_file = f"{'.'.join(args.input_file.split('.')[:-1])}.chunked.csv"
    console.print(f"Processing file: {args.input_file}")
    console.print(f"Using model: {args.model_name}")
    console.print(f"Stride: {args.stride}, Max Length: {args.max_length}")
    console.print(f"Output will be saved to: {output_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    with open(args.input_file, "r") as f:
        data = pd.read_csv(f)

    all_rows = []
    overlapping_chunks = 0

    for _, row in progress.track(
        data.iterrows(), total=len(data), description="Converting spans to BIO"
    ):
        chunks = chunk_spans(
            row["id"],
            text=row["text"],
            spans=json.loads(row["spans"]),
            tokenizer=tokenizer,
            stride=args.stride,
            max_length=args.max_length,
        )
        if len(chunks) > 1:
            overlapping_chunks += len(chunks)
        new_rows = []

        for chunk_id, chunk in enumerate(chunks):
            new_rows.append(
                {
                    "debate_id": row["debate_id"],
                    "id": row["id"],
                    "chunk_id": chunk_id,
                    "speaker": row["speaker"],
                    "text": chunk["text"],
                    "spans": json.dumps(chunk["spans"], ensure_ascii=False),
                }
            )

        all_rows.extend(new_rows)

    console.print(f"Number of overlapping chunks: {overlapping_chunks}")
    console.print(f"Processed {len(data)} rows into {len(all_rows)} chunks.")

    pd.DataFrame(all_rows).sort_values(
        ["id", "chunk_id"] if "chunk_id" in all_rows[0] else ["id"]
    ).to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
