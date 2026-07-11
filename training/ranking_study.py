from argparse import ArgumentParser
import json
from pathlib import Path

from ranking import create_val_table, create_test_table, fill_val_table, fill_test_table
from utils import console
from rich.table import Table


def parse_args():
    parser = ArgumentParser(
        description="Rank study models based on validation and test performance."
    )
    parser.add_argument(
        "study_path",
        help="Path to the study results JSON file.",
    )
    return parser.parse_args()


def collect_study_results(study_path):
    # Load the study results from the JSON file
    study_results = json.load(open(study_path, "r"))

    # Extract the model names and their corresponding results
    models = []
    for trial_number, trial_data in study_results["trials"].items():
        if trial_data.get("pruned", False):
            continue  # Skip pruned trials
        model_name = trial_number
        results = trial_data["results"]
        validation_results = results["overall"]["validation"]
        test_results = results["overall"]["test"]

        models.append(
            {
                "name": model_name,
                "results": results,
                "validation": validation_results,
                "test": test_results,
                "deciding_metric": validation_results["macro"]["f1"],
            }
        )

    return models


def main():
    args = parse_args()
    study_path = Path(args.study_path)
    if not study_path.is_file():
        raise ValueError("Study path is not a file.")

    models = collect_study_results(study_path)

    # Rank models by validation performance
    table_val = create_val_table()
    models_ranked_by_validation = sorted(
        models, key=lambda x: x["validation"]["macro"]["f1"], reverse=True
    )
    fill_val_table(table_val, models_ranked_by_validation)
    console.print(table_val)
    del models_ranked_by_validation, table_val

    # Display validation results per fold for each model
    val_per_fold = []
    table_val_per_fold = Table(title="Validation Results per Fold", expand=True)
    table_val_per_fold.add_column(
        "Model Name", style="magenta", max_width=32, no_wrap=True
    )
    table_val_per_fold.add_column("M-F1", justify="right", style="green")
    for fold in range(1, 8):
        table_val_per_fold.add_column(f"{fold}", justify="right", style="green")
    for model in models:
        folds = sorted(list(model["results"].keys()))
        folds.remove("overall")
        row = [
            model["name"],
            f"{model['deciding_metric']:.2%}",
        ]
        for i, fold in enumerate(folds):
            fold_results = model["results"][str(fold)]
            best_epoch = fold_results["best_epoch"] - 1
            row.append(
                f"{fold_results['validation_metrics'][best_epoch]['macro']['f1']:.2%}"
            )
        val_per_fold.append(row)

    val_per_fold = sorted(val_per_fold, key=lambda x: x[1], reverse=True)
    for row in val_per_fold:
        table_val_per_fold.add_row(*row)
    console.print(table_val_per_fold)
    del val_per_fold, table_val_per_fold

    # Rank models by test performance
    table_test = create_test_table()
    models_ranked_by_test = sorted(
        models, key=lambda x: x["test"]["macro"]["f1"], reverse=True
    )
    fill_test_table(table_test, models_ranked_by_test)
    console.print(table_test)
    del models_ranked_by_test, table_test


if __name__ == "__main__":
    main()
