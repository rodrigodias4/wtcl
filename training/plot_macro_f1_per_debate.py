#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = {
    "bert-base-cased": "BERT+CRF",
    "roberta-base": "RoBERTa+CRF",
    "distilbert-base-cased": "DistilBERT+CRF",
    "deberta-v3-base": "DeBERTa+CRF",
}


def extract_debate_macro_f1(results: dict) -> dict[str, float]:
    """Return a mapping of debate name -> test macro F1 from a results.json file."""
    scores = {}
    for key, value in results.items():
        if key == "overall":
            continue
        if not isinstance(value, dict):
            continue

        test_metrics = value.get("test_metrics") or value.get("test")
        if test_metrics is None:
            continue

        macro = test_metrics.get("macro")
        if macro is None:
            continue

        f1 = macro.get("f1")
        if f1 is not None:
            scores[key] = float(f1)

    return scores


def infer_model_label(path: str | Path) -> str:
    p = Path(path)

    if p.name == "results.json":
        if p.parent.parent.name and p.parent.parent.name not in {"training", "models"}:
            return MODEL_NAME.get(p.parent.parent.name, p.parent.parent.name)
        return p.parent.name

    return p.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot test macro-F1 by debate for multiple models. "
            "Provide one or more results.json files."
        )
    )
    parser.add_argument(
        "results_files",
        nargs="+",
        help="One or more results.json files to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Optional labels for each model, in the same order as the result files. "
            "If omitted, labels are inferred from the parent directory names."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save the output plot image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.labels is not None and len(args.labels) != len(args.results_files):
        raise ValueError(
            "The number of labels must match the number of results files. "
            f"Got {len(args.labels)} labels for {len(args.results_files)} files."
        )

    if args.output is None:
        args.output = Path(__file__).parent / "macro_f1_per_debate.png"

    model_scores = {}
    for path_str in args.results_files:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")

        with path.open("r", encoding="utf-8") as f:
            results = json.load(f)

        model_scores[path_str] = extract_debate_macro_f1(results)

    labels = args.labels or [infer_model_label(path) for path in args.results_files]

    debates = []
    for scores in model_scores.values():
        for debate in scores:
            if debate not in debates:
                debates.append(debate)

    if not debates:
        raise ValueError(
            "No debate-level macro-F1 entries were found in the provided files."
        )

    n_models = len(args.results_files)
    x = np.arange(len(debates))
    width = 0.8 / max(1, n_models)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(debates)), 4))

    for i, (path_str, label) in enumerate(zip(args.results_files, labels)):
        scores = model_scores[path_str]
        values = [scores.get(debate, np.nan) for debate in debates]
        offsets = x + (i - (n_models - 1) / 2) * width

        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=label,
            color=colors[i],
            edgecolor="black",
            linewidth=0.3,
            alpha=0.9,
            zorder=3,
        )

        """ for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                ) """

    ax.set_xticks(x)
    ax.set_xticklabels([debate[:4] for debate in debates], ha="center", rotation=0)
    ax.set_ylabel("Test Macro F1")
    ax.set_xlabel("Debate")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.xaxis.grid(False)
    ax.legend(title="Model", loc="upper left", frameon=True, bbox_to_anchor=(1, 1))

    fig.tight_layout()
    output_path = args.output if isinstance(args.output, Path) else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved grouped bar plot to {output_path}")


if __name__ == "__main__":
    main()
