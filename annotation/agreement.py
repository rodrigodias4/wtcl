from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path
import sys
from pandas import isnull, read_csv
from sklearn.metrics import classification_report, cohen_kappa_score
import krippendorff
from pygamma_agreement import Continuum
from pyannote.core import Segment
from rich.console import Console
from rich.progress import Progress

sys.path.append((Path(__file__).resolve().parent.parent / "training").as_posix())
from train import encode, get_tokenizer
from partial_span_analysis import compute_partial_span_metrics
from utils import label_list
from plot_cm import compute_metrics_span_level

console = Console()
progress = Progress(console=console, transient=True)


def add_label_runs(continuum, annotator, labels, offset):
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            continuum.add(
                annotator,
                Segment(offset + start, offset + end),
                str(labels[start]),
            )
            start = end


def parse_args():
    parser = ArgumentParser(description="Compute agreement metrics for annotations.")

    parser.add_argument("file_A", type=str, help="Path to the first annotation file.")

    parser.add_argument("file_B", type=str, help="Path to the second annotation file.")

    parser.add_argument(
        "--latex", action="store_true", help="Output results in LaTeX format."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load annotations from the files
    df_A = read_csv(args.file_A)
    df_B = read_csv(args.file_B)

    labels_A = []
    labels_B = []
    tokenizer = get_tokenizer("FacebookAI/roberta-base")

    for idx in range(len(df_A)):
        if idx > len(df_B) - 1:
            console.print(
                f"Halting at index {idx} as file_B has fewer entries than file_A."
            )
            break
        row_A = df_A.iloc[idx]
        row_B = df_B.iloc[idx]

        assert row_A["id"] == row_B["id"], f"Row IDs do not match at index {idx}."
        assert row_A["text"] == row_B["text"], f"Row texts do not match at index {idx}."

        if (
            row_A["spans"] is None
            or row_B["spans"] is None
            or isnull(row_A["spans"])
            or isnull(row_B["spans"])
            or row_A["spans"] == ""
            or row_B["spans"] == ""
            or row_A["spans"] == "PASS"
            or row_B["spans"] == "PASS"
        ):
            console.print(f"Skipping row {row_A['id']} due to missing spans.")
            continue

        if row_A["text"] != row_B["text"]:
            console.print(
                f"[yellow]Warning:[/yellow] Texts do not match at row {row_A['id']}."
            )

        spans_A = literal_eval(row_A["spans"])
        spans_B = literal_eval(row_B["spans"])

        if not isinstance(spans_A, list) or not isinstance(spans_B, list):
            console.print(f"Skipping row {idx} due to invalid span format.")
            continue

        # Encode the spans
        enc_A = encode(row_A["text"], spans_A, tokenizer)
        enc_B = encode(row_B["text"], spans_B, tokenizer)

        assert len(enc_A["labels"]) == len(
            enc_B["labels"]
        ), f"Encoded label lengths do not match at ID {row_A['id']}."

        labels_A.append(
            [label for label, mask in zip(enc_A["labels"], enc_A["crf_mask"]) if mask]
        )
        labels_B.append(
            [label for label, mask in zip(enc_B["labels"], enc_B["crf_mask"]) if mask]
        )
        assert len(labels_A[-1]) == len(
            labels_B[-1]
        ), f"Filtered label lengths do not match at ID {row_A['id']}."

    labels_A_flat = [label for sublist in labels_A for label in sublist]
    labels_B_flat = [label for sublist in labels_B for label in sublist]

    assert len(labels_A_flat) == len(
        labels_B_flat
    ), "Flattened label lists must be of the same length."

    progress.start()
    # Cohen's Kappa
    progress_temp = progress.add_task(
        description="Computing Cohen's Kappa...", total=None
    )
    kappa = cohen_kappa_score(labels_A_flat, labels_B_flat)
    progress.remove_task(progress_temp)

    # Krippendorff's Alpha
    progress_temp = progress.add_task(
        description="Computing Krippendorff's Alpha...", total=None
    )
    alpha = krippendorff.alpha(
        reliability_data=[labels_A_flat, labels_B_flat],
        level_of_measurement="nominal",
    )
    progress.remove_task(progress_temp)

    # Mathet's Gamma
    progress_temp = progress.add_task(
        description="Computing Mathet's Gamma...", total=None
    )
    gamma_continuum = Continuum()
    token_offset = 0
    for row_labels_A, row_labels_B in zip(labels_A, labels_B):
        add_label_runs(gamma_continuum, "A", row_labels_A, token_offset)
        add_label_runs(gamma_continuum, "B", row_labels_B, token_offset)
        token_offset += len(row_labels_A)
    gamma = gamma_continuum.compute_gamma(n_samples=30, fast=True).gamma
    progress.remove_task(progress_temp)

    progress_temp = progress.add_task(
        description="Computing token-level metrics...", total=None
    )
    token_level_metrics = classification_report(
        labels_A_flat,
        labels_B_flat,
        labels=list(range(len(label_list))),
        output_dict=True,
    )
    progress.remove_task(progress_temp)

    progress_temp = progress.add_task(
        description="Computing span-level metrics...", total=None
    )
    exact_span = compute_metrics_span_level(labels_A, labels_B)
    thresholds = [0.25, 0.5, 0.75]
    m = compute_partial_span_metrics(labels_A, labels_B, thresholds=thresholds)
    progress.remove_task(progress_temp)
    progress.stop()

    console.print(f"Mathet's Gamma: {gamma:.2%}")
    console.print(f"Cohen's Kappa: {kappa:.2%}")
    console.print(f"Krippendorff's Alpha: {alpha:.2%}")

    if args.latex:
        console.print("Token-level metrics (LaTeX format):")
        console.print(
            f"{token_level_metrics['macro avg']['f1-score'] * 100:.1f} & "
            f"{token_level_metrics['macro avg']['precision'] * 100:.1f} & "
            f"{token_level_metrics['macro avg']['recall'] * 100:.1f} & "
            f"{token_level_metrics['1']['f1-score'] * 100:.1f} & "
            f"{token_level_metrics['1']['precision'] * 100:.1f} & "
            f"{token_level_metrics['1']['recall'] * 100:.1f} & "
            f"{token_level_metrics['2']['f1-score'] * 100:.1f} & "
            f"{token_level_metrics['2']['precision'] * 100:.1f} & "
            f"{token_level_metrics['2']['recall'] * 100:.1f} & "
            f"{token_level_metrics['0']['f1-score'] * 100:.1f} & "
            f"{token_level_metrics['0']['precision'] * 100:.1f} & "
            f"{token_level_metrics['0']['recall'] * 100:.1f} \\\\"
        )

        console.print("Span-level metrics (LaTeX format):")
        console.print(
            f"{float(m[0]['f1']) * 100:.1f} & "
            f"{float(m[0]['precision']) * 100:.1f} & "
            f"{float(m[0]['recall']) * 100:.1f} & "
            f"{float(m[1]['f1']) * 100:.1f} & "
            f"{float(m[1]['precision']) * 100:.1f} & "
            f"{float(m[1]['recall']) * 100:.1f} & "
            f"{float(m[2]['f1']) * 100:.1f} & "
            f"{float(m[2]['precision']) * 100:.1f} & "
            f"{float(m[2]['recall']) * 100:.1f} & "
            f"{exact_span['f1'] * 100:.1f} & "
            f"{exact_span['precision'] * 100:.1f} & "
            f"{exact_span['recall'] * 100:.1f} \\\\"
        )
        return

    # Print token-level metrics
    # Macro metrics
    console.print(
        f"Macro metrics: "
        f"F1 = {token_level_metrics['macro avg']['f1-score']:.2%}, "
        f"Precision = {token_level_metrics['macro avg']['precision']:.2%}, "
        f"Recall = {token_level_metrics['macro avg']['recall']:.2%}"
    )
    # Print token-level metrics for each class
    for id in range(len(label_list)):
        console.print(
            f"Metrics for class {label_list[id]}: "
            f"F1 = {token_level_metrics[str(id)]['f1-score']:.2%}, "
            f"Precision = {token_level_metrics[str(id)]['precision']:.2%}, "
            f"Recall = {token_level_metrics[str(id)]['recall']:.2%}"
        )

    # Print exact span metrics
    console.print(
        f"Exact Span metrics: "
        f"F1 = {exact_span['f1']:.2%}, "
        f"Precision = {exact_span['precision']:.2%}, "
        f"Recall = {exact_span['recall']:.2%}"
    )

    # Print partial span metrics
    for i, t in enumerate(thresholds):
        console.print(
            f"IoU ≥ {t:.2f}: F1 = {float(m[i]['f1']):.2%} P={float(m[i]['precision']):.2%} R={(float(m[i]['recall'])):.2%}"
        )


if __name__ == "__main__":
    main()
