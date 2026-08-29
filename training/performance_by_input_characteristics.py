import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from matplotlib import cm, colors
import matplotlib.pyplot as plt
import colormaps as cmaps
import numpy as np
import pandas as pd
from spacy.lang.en import English
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))

from partial_span_analysis import bio_sequence_to_spans, compute_partial_span_metrics
from utils import console, label_list

DEFAULT_LENGTH_BIN_SIZE = 16
DEFAULT_SENTENCE_BIN_SIZE = 8


def bin_label(value: int, bin_size: int) -> str:
    lower = (value // bin_size) * bin_size
    return f"{lower}-{lower + bin_size - 1}"


def bin_start(value: int, bin_size: int, max_threshold: int | None = None) -> int:
    if max_threshold is not None and value >= max_threshold:
        return max_threshold
    return (value // bin_size) * bin_size


def collect_sequences(
    predictions: dict[str, list[Sequence]],
    labels: dict[str, list[Sequence]],
) -> list[tuple[Sequence, Sequence]]:
    pairs: list[tuple[Sequence, Sequence]] = []
    if set(predictions) != set(labels):
        raise ValueError("Predictions and labels must contain the same debate keys.")

    for debate in sorted(predictions):
        if len(predictions[debate]) != len(labels[debate]):
            raise ValueError(
                f"Debate '{debate}' has mismatched predictions and labels."
            )
        for prediction, label in zip(predictions[debate], labels[debate]):
            if len(prediction) != len(label):
                raise ValueError(
                    f"Debate '{debate}' has a prediction/label sequence length mismatch."
                )
            pairs.append((prediction, label))
    return pairs


def collect_dataset_sequences(
    predictions: dict[str, list[Sequence]],
    labels: dict[str, list[Sequence]],
    dataset_path: Path,
) -> list[tuple[str, Sequence, Sequence]]:
    dataset = pd.read_csv(dataset_path)
    required_columns = {"debate_id", "id", "chunk_id", "text"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    rows: list[tuple[str, Sequence, Sequence]] = []
    if set(predictions) != set(labels):
        raise ValueError("Predictions and labels must contain the same debate keys.")

    for debate in sorted(predictions):
        debate_rows = (
            dataset[dataset["debate_id"] == debate]
            .sort_values(["id", "chunk_id"])
            .reset_index(drop=True)
        )
        if len(predictions[debate]) != len(debate_rows) or len(labels[debate]) != len(
            debate_rows
        ):
            raise ValueError(
                f"Debate '{debate}' has mismatched dataset, predictions, and labels lengths."
            )

        for (_, row), prediction, label in zip(
            debate_rows.iterrows(), predictions[debate], labels[debate]
        ):
            if len(prediction) != len(label):
                raise ValueError(
                    f"Debate '{debate}' has a prediction/label sequence length mismatch."
                )
            rows.append((str(row["text"]), prediction, label))
    return rows


def split_into_sentences(
    text: str,
    prediction: Sequence,
    label: Sequence,
    word_offsets: list[tuple[int, int]],
) -> list[tuple[Sequence, Sequence]]:
    nlp = English()
    nlp.add_pipe("sentencizer")
    sentences = list(nlp(text).sents)
    if len(word_offsets) != len(label):
        raise ValueError(
            "The number of tokenizer word offsets does not match the word-level "
            f"labels ({len(word_offsets)} != {len(label)})."
        )

    sentence_sequences: list[tuple[Sequence, Sequence]] = []
    word_index = 0
    for sentence in sentences:
        sentence_start = word_index
        while (
            word_index < len(word_offsets)
            and word_offsets[word_index][0] < sentence.end_char
        ):
            word_index += 1
        if word_index > sentence_start:
            sentence_sequences.append(
                (
                    prediction[sentence_start:word_index],
                    label[sentence_start:word_index],
                )
            )

    if word_index != len(word_offsets):
        raise ValueError("Sentence splitting did not account for every dataset word.")
    return sentence_sequences


def get_word_offsets(
    text: str, labels: Sequence, tokenizer: AutoTokenizer
) -> list[tuple[int, int]]:
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    word_ids = encoding.word_ids()
    offsets = encoding["offset_mapping"]
    selected_offsets = []
    previous_word = None
    for word_id, offset in zip(word_ids, offsets):
        if word_id is None:
            continue
        if word_id != previous_word:
            selected_offsets.append(offset)
            previous_word = word_id
    if len(selected_offsets) != len(labels):
        raise ValueError(
            "The tokenizer produced a different number of word-level positions "
            f"than the saved labels ({len(selected_offsets)} != {len(labels)})."
        )
    return selected_offsets


def aggregate_by_sentence_length(
    dataset_rows: list[tuple[str, Sequence, Sequence]],
    sentence_bin_size: int,
    tokenizer: AutoTokenizer,
) -> dict[int, dict]:
    grouped: dict[int, tuple[list[Sequence], list[Sequence]]] = {}
    for text, prediction, label in dataset_rows:
        word_offsets = get_word_offsets(text, label, tokenizer)
        for sentence_prediction, sentence_label in split_into_sentences(
            text, prediction, label, word_offsets
        ):
            key = bin_start(len(sentence_label), sentence_bin_size, 64)
            pred_bucket, gold_bucket = grouped.setdefault(key, ([], []))
            pred_bucket.append(sentence_prediction)
            gold_bucket.append(sentence_label)
    return compute_bucket_metrics(grouped)


def aggregate_by_length(
    pairs: list[tuple[Sequence, Sequence]], length_bin_size: int
) -> dict[int, dict]:
    grouped: dict[int, tuple[list[Sequence], list[Sequence]]] = {}
    for prediction, label in pairs:
        key = bin_start(len(label), length_bin_size)
        pred_bucket, gold_bucket = grouped.setdefault(key, ([], []))
        pred_bucket.append(prediction)
        gold_bucket.append(label)
    return compute_bucket_metrics(grouped)


def aggregate_by_span_count(pairs: list[tuple[Sequence, Sequence]]) -> dict[int, dict]:
    grouped: dict[int, tuple[list[Sequence], list[Sequence]]] = {}
    for prediction, label in pairs:
        n_spans = len(bio_sequence_to_spans(label))
        key = n_spans
        pred_bucket, gold_bucket = grouped.setdefault(key, ([], []))
        pred_bucket.append(prediction)
        gold_bucket.append(label)
    return compute_bucket_metrics(grouped)


def compute_bucket_metrics(
    grouped: dict[int, tuple[list[Sequence], list[Sequence]]],
) -> dict[int, dict]:
    metrics = {}
    for key, (pred_bucket, gold_bucket) in grouped.items():
        flat_preds = [token for seq in pred_bucket for token in seq]
        flat_gold = [token for seq in gold_bucket for token in seq]
        macro_f1 = f1_score(
            flat_gold,
            flat_preds,
            labels=list(range(len(label_list))),
            average="macro",
            zero_division=0,
        )
        """ macro_f1 = compute_partial_span_metrics(
            gold_bucket, pred_bucket, thresholds=[0.5]
        )[0]["f1"] """
        metrics[key] = {"f1": 0.0 if not np.isfinite(macro_f1) else float(macro_f1)}
        metrics[key]["count"] = len(gold_bucket)
        metrics[key]["token_count"] = len(flat_gold)
    return metrics


def compute_global_macro_f1(pairs: list[tuple[Sequence, Sequence]]) -> float:
    flat_preds = [token for pred, _ in pairs for token in pred]
    flat_gold = [token for _, gold in pairs for token in gold]
    report = classification_report(
        flat_gold,
        flat_preds,
        labels=list(range(len(label_list))),
        output_dict=True,
        zero_division=0,
    )
    return float(report["macro avg"]["f1-score"])


def compute_metrics_by_characteristic(
    predictions: dict[str, list[Sequence]],
    labels: dict[str, list[Sequence]],
    length_bin_size: int,
) -> tuple[dict[str, dict], dict[str, dict]]:
    if length_bin_size <= 0:
        raise ValueError("Length bin size must be a positive integer.")
    pairs = collect_sequences(predictions, labels)
    return (
        aggregate_by_length(pairs, length_bin_size),
        aggregate_by_span_count(pairs),
    )


def plot_metrics_by_characteristic(
    metrics: dict[int, dict],
    output_path: Path,
    xlabel: str,
    histogram_mode: bool,
    figsize: tuple[int, int],
    max_threshold: int | None = None,
) -> None:
    console.print(metrics)
    groups = sorted(metrics)
    heights = [metrics[group]["f1"] for group in groups]
    counts = [metrics[group]["count"] for group in groups]

    cmap = cmaps.blue_8_5g2
    norm = colors.Normalize(
        vmin=min(counts) if counts else 0, vmax=max(counts) if counts else 1
    )
    bar_colors = [cmap(norm(c)) for c in counts]

    figure, axis = plt.subplots(figsize=figsize)
    if histogram_mode:
        bin_size = groups[1] - groups[0] if len(groups) > 1 else 1
        axis.bar(
            groups,
            heights,
            width=bin_size,
            align="edge",
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
            zorder=1,
        )
        xticks = [str(group) for group in groups]
        if max_threshold is not None:
            xticks = [
                f"{tick}+" if tick >= max_threshold else str(tick) for tick in groups
            ]
    else:
        axis.bar(
            groups,
            heights,
            width=0.8,
            align="center",
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
            zorder=1,
        )
        xticks = groups

    axis.set_xticks(groups, xticks)
    axis.set_xlabel(xlabel, fontsize=14)
    axis.set_ylabel("Macro F1", fontsize=14)
    axis.set_ylim(0, 1)
    axis.set_yticks(np.arange(0, 1.1, 0.1))
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    axis.margins(x=0.02)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = figure.colorbar(sm, ax=axis, pad=0.02)
    cbar.set_label("Support (Count)", fontsize=11)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot span-level performance by input length and gold span count."
    )
    parser.add_argument("preds_labels_file", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--length-bin-size", type=int, default=DEFAULT_LENGTH_BIN_SIZE)
    parser.add_argument(
        "--sentence-bin-size", type=int, default=DEFAULT_SENTENCE_BIN_SIZE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.preds_labels_file.open() as handle:
        payload = json.load(handle)

    predictions = payload.get("preds", payload.get("predictions"))
    labels = payload.get("labels")
    if predictions is None or labels is None:
        raise ValueError("The JSON file must contain 'preds' and 'labels'.")
    if args.sentence_bin_size <= 0:
        raise ValueError("Sentence bin size must be a positive integer.")

    length_metrics, span_metrics = compute_metrics_by_characteristic(
        predictions, labels, args.length_bin_size
    )
    pairs = collect_sequences(predictions, labels)
    global_macro_f1 = compute_global_macro_f1(pairs)
    dataset_rows = collect_dataset_sequences(predictions, labels, args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sentence_metrics = aggregate_by_sentence_length(
        dataset_rows, args.sentence_bin_size, tokenizer
    )
    output_dir = args.output_dir or args.preds_labels_file.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    length_plot_path = output_dir / "performance_by_sequence_length.png"
    span_plot_path = output_dir / "performance_by_span_count.png"
    sentence_plot_path = output_dir / "performance_by_sentence_length.png"
    plot_metrics_by_characteristic(
        length_metrics,
        length_plot_path,
        xlabel="Sequence length",
        histogram_mode=True,
        figsize=(14, 4),
    )
    plot_metrics_by_characteristic(
        span_metrics,
        span_plot_path,
        xlabel="Span count",
        histogram_mode=False,
        figsize=(7, 3),
    )
    plot_metrics_by_characteristic(
        sentence_metrics,
        sentence_plot_path,
        xlabel="Sentence length",
        histogram_mode=True,
        figsize=(7, 3),
        max_threshold=64,
    )

    console.print("Macro F1 by sequence length:")
    for length, values in sorted(length_metrics.items()):
        console.print(
            f"Length {bin_label(length, args.length_bin_size)}: n={values['count']} "
            f"F1={values['f1']:.1%}"
        )

    console.print("\nMacro F1 by span count:")
    for spans, values in sorted(span_metrics.items()):
        console.print(f"Spans {spans}: n={values['count']} " f"F1={values['f1']:.1%}")

    console.print("\nMacro F1 by sentence length:")
    for length, values in sorted(sentence_metrics.items()):
        console.print(
            f"Sentence length {bin_label(length, args.sentence_bin_size)}: "
            f"n={values['count']} F1={values['f1']:.1%}"
        )

    console.print(f"Saved plot to {length_plot_path}")
    console.print(f"Saved plot to {span_plot_path}")
    console.print(f"Saved plot to {sentence_plot_path}")


if __name__ == "__main__":
    main()
