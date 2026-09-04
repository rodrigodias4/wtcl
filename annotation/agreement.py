from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path
import sys
from pandas import isnull, read_csv
from sklearn.metrics import classification_report, cohen_kappa_score
from rich.console import Console

sys.path.append((Path(__file__).resolve().parent.parent / "training").as_posix())
from train import encode, get_tokenizer
from partial_span_analysis import compute_partial_span_metrics
from utils import label_list
from plot_cm import compute_metrics_span_level

console = Console()


def parse_args():
    parser = ArgumentParser(description="Compute agreement metrics for annotations.")

    parser.add_argument("file_A", type=str, help="Path to the first annotation file.")

    parser.add_argument("file_B", type=str, help="Path to the second annotation file.")

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

    # Cohen's Kappa
    kappa = cohen_kappa_score(labels_A_flat, labels_B_flat)
    console.print(f"Cohen's Kappa: {kappa:.2%}")

    # Token-level metrics
    token_level_metrics = classification_report(
        labels_A_flat,
        labels_B_flat,
        labels=list(range(len(label_list))),
        output_dict=True,
    )
    console.print(
        f"Macro metrics: "
        f"F1 = {token_level_metrics['macro avg']['f1-score']:.2%}, "
        f"Precision = {token_level_metrics['macro avg']['precision']:.2%}, "
        f"Recall = {token_level_metrics['macro avg']['recall']:.2%}"
    )

    for id in range(len(label_list)):
        console.print(
            f"Metrics for class {label_list[id]}: "
            f"F1 = {token_level_metrics[str(id)]['f1-score']:.2%}, "
            f"Precision = {token_level_metrics[str(id)]['precision']:.2%}, "
            f"Recall = {token_level_metrics[str(id)]['recall']:.2%}"
        )

    # Exact Span F1
    exact_span = compute_metrics_span_level(labels_A, labels_B)
    console.print(
        f"Exact Span metrics: "
        f"F1 = {exact_span['f1']:.2%}, "
        f"Precision = {exact_span['precision']:.2%}, "
        f"Recall = {exact_span['recall']:.2%}"
    )

    # Partial Span F1 for different IoU thresholds
    thresholds = [0.25, 0.5, 0.75]
    m = compute_partial_span_metrics(labels_A, labels_B, thresholds=thresholds)
    for i, t in enumerate(thresholds):
        console.print(
            f"IoU ≥ {t:.2f}: F1 = {float(m[i]['f1']):.2%} P={float(m[i]['precision']):.2%} R={(float(m[i]['recall'])):.2%}"
        )


if __name__ == "__main__":
    main()
