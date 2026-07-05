from argparse import ArgumentParser
import json
from pathlib import Path

from ranking import create_val_table, create_test_table, fill_val_table, fill_test_table
from utils import console


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
        model_name = trial_number
        results = trial_data["results"]
        validation_results = results["overall"]["validation"]
        test_results = results["overall"]["test"]

        models.append(
            {
                "name": model_name,
                "validation": validation_results,
                "test": test_results,
            }
        )

    # Rank models by validation and test performance
    models_ranked_by_validation = sorted(
        models, key=lambda x: x["validation"]["macro"]["f1"], reverse=True
    )
    models_ranked_by_test = sorted(
        models, key=lambda x: x["test"]["macro"]["f1"], reverse=True
    )

    return models_ranked_by_validation, models_ranked_by_test


def main():
    args = parse_args()
    study_path = Path(args.study_path)
    if not study_path.is_file():
        raise ValueError("Study path is not a file.")

    table_val = create_val_table()
    table_test = create_test_table()

    models_ranked_by_validation, models_ranked_by_test = collect_study_results(
        study_path
    )

    fill_val_table(table_val, models_ranked_by_validation)
    fill_test_table(table_test, models_ranked_by_test)

    console.print(table_val)
    console.print(table_test)


if __name__ == "__main__":
    main()
