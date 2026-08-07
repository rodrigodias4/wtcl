import argparse
import json
import os
from pathlib import Path
from rich.table import Table

from utils import console, progress

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
models_dir = script_dir / "models"
models = []
dataset_name = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank models based on validation and test performance."
    )
    parser.add_argument(
        "--transformer-name",
        help="Name of the transformer for which to rank models.",
        required=True,
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of the dataset for which to rank models.",
        default="wtcl-v2",
    )
    return parser.parse_args()


def collect_model_results(transformer_dir):
    # Iterate through the models directory and collect model results
    for model_name in os.listdir(transformer_dir):
        model_dir = transformer_dir / model_name
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith("."):
            continue

        if not (model_dir / "results.json").exists():
            console.print(f"Skipping {model_name} as results.json does not exist.")
            continue

        model_results = json.load(open(model_dir / "results.json", "r"))

        if "overall" not in model_results:
            console.print(
                f"Skipping {model_name} as 'overall' key is missing in results.json."
            )
            continue

        validation_results = model_results["overall"].get("validation", {})
        test_results = model_results["overall"].get("test", {})

        models.append(
            {
                "name": f"{model_name[5:10]} {model_name[20:]}",
                "validation": validation_results,
                "test": test_results,
            }
        )

    models_with_validation = [m for m in models if m["validation"]]
    models_ranked_by_validation = sorted(
        models_with_validation,
        key=lambda x: x["validation"]["macro"]["f1"],
        reverse=True,
    )
    models_ranked_by_test = sorted(
        models, key=lambda x: x["test"]["macro"]["f1"], reverse=True
    )

    return models_ranked_by_validation, models_ranked_by_test


def create_val_table():
    # Create a rich table to display the model rankings based on validation macro F1 scores
    table = Table(
        title=f"Model Validation Performance on {dataset_name} Dataset", expand=True
    )
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True, max_width=3)
    table.add_column("Model Name", style="magenta", max_width=32, no_wrap=True)
    table.add_column("F1", justify="right", style="green")
    table.add_column("P", justify="right", style="green")
    table.add_column("R", justify="right", style="green")
    table.add_column("B-F1", justify="right", style="green")
    table.add_column("B-P", justify="right", style="green")
    table.add_column("B-R", justify="right", style="green")
    table.add_column("I-F1", justify="right", style="green")
    table.add_column("I-P", justify="right", style="green")
    table.add_column("I-R", justify="right", style="green")
    table.add_column("S-F1", justify="right", style="green")
    table.add_column("Jacc", justify="right", style="green")

    return table


def create_test_table():
    # Print the table to the console

    # Create a rich table to display the model rankings based on test macro F1 scores
    table = Table(
        title=f"Model Test Performance on {dataset_name} Dataset", expand=True
    )
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True, max_width=3)
    table.add_column("Model Name", style="magenta")
    table.add_column("Test Macro F1", justify="right", style="green")
    return table


def fill_val_table(table, models_ranked_by_validation):
    # Add rows to the table based on the ranked models
    for rank, model in enumerate(models_ranked_by_validation, start=1):
        table.add_row(
            str(rank),
            model["name"],
            f"{(model['validation']['macro']['f1'] * 100):.1f}",
            f"{(model['validation']['macro']['precision'] * 100):.1f}",
            f"{(model['validation']['macro']['recall'] * 100):.1f}",
            f"{(model['validation']['B']['f1'] * 100):.2f}",
            f"{(model['validation']['B']['precision'] * 100):.1f}",
            f"{(model['validation']['B']['recall'] * 100):.1f}",
            f"{(model['validation']['I']['f1'] * 100):.2f}",
            f"{(model['validation']['I']['precision'] * 100):.1f}",
            f"{(model['validation']['I']['recall'] * 100):.1f}",
            (
                f"{(model['validation']['span']['f1'] * 100):.2f}"
                if "span" in model["validation"]
                else "N/A"
            ),
            (
                f"{(model['validation']['macro']['jaccard'] * 100):.2f}"
                if "jaccard" in model["validation"]["macro"]
                else "N/A"
            ),
        )


def fill_test_table(table, models_ranked_by_test):
    # Add rows to the table based on the ranked models
    for rank, model in enumerate(models_ranked_by_test, start=1):
        table.add_row(
            str(rank),
            model["name"],
            f"{(model['test']['macro']['f1'] * 100):.1f}",
        )


def main():
    global dataset_name
    args = parse_args()
    dataset_name = args.dataset_name
    transformer_dir = models_dir / dataset_name / args.transformer_name
    if not transformer_dir.is_dir():
        console.print(
            f"Transformer directory for {args.transformer_name} does not exist."
        )
        exit(1)

    models_ranked_by_validation, models_ranked_by_test = collect_model_results(
        transformer_dir
    )
    table_val = create_val_table()
    table_test = create_test_table()
    fill_val_table(table_val, models_ranked_by_validation)
    fill_test_table(table_test, models_ranked_by_test)
    console.print(table_val)
    console.print(table_test)


if __name__ == "__main__":
    main()
