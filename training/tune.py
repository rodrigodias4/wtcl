import pandas as pd
import optuna

from training.train import build_model, compute_macro_f1_token_level, evaluate, train


# Hyperparameter sampling function
def sample_hparams(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "max_length": trial.suggest_categorical("max_length", [256, 512, 768]),
    }


def objective(trial, all_debates):
    hparams = sample_hparams(trial)

    outer_scores = []

    for test_debate in all_debates:
        print(f"Testing on debate: {test_debate.name} with hyperparameters: {hparams}")
        # -------------------------
        # OUTER SPLIT (LODO)
        # -------------------------
        train_val = [d for d in all_debates if d != test_debate]

        # -------------------------
        # INNER CV FOR TUNING SCORE
        # -------------------------
        inner_scores = []

        for val_debate in train_val:
            print(f"  Validating on debate: {val_debate.name}")

            train_debates = [d for d in train_val if d != val_debate]

            train_data = pd.merge(train_debates)
            val_data = val_debate

            model = build_model(hparams)

            train(model, train_data, hparams)
            preds = evaluate(model, val_data)

            score = compute_macro_f1_token_level(preds, val_data)
            inner_scores.append(score)

        mean_inner_score = sum(inner_scores) / len(inner_scores)
        outer_scores.append(mean_inner_score)
        trial.report(mean_inner_score, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return sum(outer_scores) / len(outer_scores)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Tune hyperparameters for the model.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    
    # ...
    
    study.optimize(lambda trial: objective(trial, debates), n_trials=30)
    best_params = study.best_trial.params


if __name__ == "__main__":
    main()
