import argparse
import json
import random
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.text import Text
from transformers import AutoTokenizer

from train import B_ID, I_ID, MAX_LENGTH, encode, get_tokenizer

console = Console()


def spans_from_bio(labels, offsets):
    """
    Converts BIO labels aligned with first-subword tokens
    into character spans.
    """
    spans = []

    start_char = None
    end_char = None

    for label, (s, e) in zip(labels, offsets):

        if label == B_ID:

            if start_char is not None:
                spans.append((start_char, end_char))

            start_char = s
            end_char = e

        elif label == I_ID:

            if start_char is None:
                start_char = s

            end_char = e

        else:

            if start_char is not None:
                spans.append((start_char, end_char))
                start_char = None
                end_char = None

    if start_char is not None:
        spans.append((start_char, end_char))

    return spans


def style_per_character(length, pred_spans, gold_spans):

    styles = [None] * length

    pred_mask = [False] * length
    gold_mask = [False] * length

    for s, e in pred_spans:
        for i in range(s, e):
            if 0 <= i < length:
                pred_mask[i] = True

    for s, e in gold_spans:
        for i in range(s, e):
            if 0 <= i < length:
                gold_mask[i] = True

    for i in range(length):

        if pred_mask[i] and gold_mask[i]:
            styles[i] = "black on green"

        elif pred_mask[i]:
            styles[i] = "black on red"

        elif gold_mask[i]:
            styles[i] = "black on blue"

    return styles


def colored_text(text, pred_spans, gold_spans):

    styles = style_per_character(len(text), pred_spans, gold_spans)

    rich_text = Text()

    current_style = styles[0] if styles else None
    buffer = ""

    for ch, st in zip(text, styles):

        if st != current_style:
            rich_text.append(buffer, style=current_style)
            buffer = ch
            current_style = st
        else:
            buffer += ch

    if buffer:
        rich_text.append(buffer, style=current_style)

    return rich_text


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--tokenizer", required=True)

    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=20,
    )

    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    tokenizer = get_tokenizer(args.tokenizer)

    df = pd.read_csv(args.dataset)

    with open(args.predictions) as f:
        data = json.load(f)

    rows = []

    for debate in sorted(data["preds"].keys()):

        preds = data["preds"][debate]
        labels = data["labels"][debate]

        debate_rows = (
            df[df.debate_id == debate]
            .sort_values(["id", "chunk_id"])
            .reset_index(drop=True)
        )

        assert len(preds) == len(debate_rows)
        assert len(labels) == len(debate_rows)

        for (_, row), pred, gold in zip(
            debate_rows.iterrows(),
            preds,
            labels,
        ):

            rows.append(
                (
                    debate,
                    row,
                    pred,
                    gold,
                )
            )

    random.shuffle(rows)

    printed = 0

    console.print(
        "[green]Green[/green] = overlap   "
        "[red]Red[/red] = prediction only   "
        "[blue]Blue[/blue] = gold only"
    )

    for debate, row, pred, gold in rows:

        text = row.text

        enc = encode(text, [], tokenizer, max_length=MAX_LENGTH)

        filtered_offsets = [
            off for off, keep in zip(enc["offset_mapping"], enc["crf_mask"]) if keep
        ]

        if len(filtered_offsets) != len(pred):
            console.print(
                f"[yellow]Skipping because token count differs "
                f"({len(filtered_offsets)} vs {len(pred)})[/yellow]"
            )
            continue

        pred_spans = spans_from_bio(
            pred,
            filtered_offsets,
        )

        gold_spans = spans_from_bio(
            gold,
            filtered_offsets,
        )

        if pred_spans == gold_spans:
            continue

        console.rule(f"[bold]{debate}[/bold]   " f"id={row.id}   chunk={row.chunk_id}")

        console.print(
            colored_text(
                text,
                pred_spans,
                gold_spans,
            ),
            end="\n\n",
        )

        printed += 1

        if printed >= args.num_samples:
            break


if __name__ == "__main__":
    main()
