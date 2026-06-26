import json
import os
from pathlib import Path
from rich.table import Table

from utils import console, progress

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
dataset_name = "wtcl-v1"
models_dir = script_dir / "models" / dataset_name
models = []

# Iterate through the models directory and collect model results
for transformer_name in os.listdir(models_dir):
    transformer_dir = models_dir / transformer_name
    if not transformer_dir.is_dir():
        continue
    for model_name in os.listdir(transformer_dir):
        model_dir = transformer_dir / model_name
        if not model_dir.is_dir():
            continue

        if not (model_dir / "results.json").exists():
            console.print(f"Skipping {model_name} as results.json does not exist.")
            continue

        model_results = json.load(open(model_dir / "results.json", "r"))
        validation_results = model_results["overall"]["validation"]
        test_results = model_results["overall"]["test"]

        models.append(
            {
                "name": model_name,
                "results": model_results,
                "validation": validation_results,
                "test": test_results,
            }
        )

        models_ranked_by_validation = sorted(
            models, key=lambda x: x["validation"]["macro"]["f1"], reverse=True
        )
        models_ranked_by_test = sorted(
            models, key=lambda x: x["test"]["macro"]["f1"], reverse=True
        )

# Create a rich table to display the model rankings based on validation macro F1 scores
table = Table(
    title=f"Model Validation Performance on {dataset_name} Dataset", expand=True
)
table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
table.add_column("Model Name", style="magenta")
table.add_column("M-F1", justify="right", style="green")
table.add_column("M-A", justify="right", style="green")
table.add_column("M-P", justify="right", style="green")
table.add_column("M-R", justify="right", style="green")
table.add_column("B-F1", justify="right", style="green")
table.add_column("I-F1", justify="right", style="green")
table.add_column("S-F1", justify="right", style="green")

# Add rows to the table based on the ranked models
for rank, model in enumerate(models_ranked_by_validation, start=1):
    table.add_row(
        str(rank),
        model["name"],
        f"{model['validation']['macro']['f1']:.4f}",
        f"{model['validation']['macro']['accuracy']:.4f}",
        f"{model['validation']['macro']['precision']:.4f}",
        f"{model['validation']['macro']['recall']:.4f}",
        f"{model['validation']['B']['f1']:.4f}",
        f"{model['validation']['I']['f1']:.4f}",
        (
            f"{model['validation']['span']['f1']:.4f}"
            if "span" in model["validation"]
            else "N/A"
        ),
    )

# Print the table to the console
console.print(table)


# Create a rich table to display the model rankings based on test macro F1 scores
table_test = Table(
    title=f"Model Test Performance on {dataset_name} Dataset", expand=True
)
table_test.add_column("Rank", justify="right", style="cyan", no_wrap=True)
table_test.add_column("Model Name", style="magenta")
table_test.add_column("Test Macro F1", justify="right", style="green")

# Add rows to the table based on the ranked models
for rank, model in enumerate(models_ranked_by_test, start=1):
    table_test.add_row(
        str(rank),
        model["name"],
        f"{model['test']['macro']['f1']:.4f}",
    )

# Print the test performance table to the console
console.print(table_test)
