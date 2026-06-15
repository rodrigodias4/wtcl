import pandas as pd

MODEL_DEFAULT = "distilroberta-base"

# Label mapping
label_list = ["O", "B", "I"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}


def get_validation_debate(df: pd.DataFrame, debates: list[str]) -> str:
    """
    Choosing validation debate that minimizes the distance to the mean
    ratio of positive tokens across all debates.
    This is a heuristic to select a validation debate that is representative of the overall dataset.
    """
    debate_ratios = {}
    for debate in debates:
        n_positive_tokens = 0
        n_tokens = 0
        debate_data = df[df["debate_id"] == debate]

        for index, row in debate_data.iterrows():
            labels = eval(row["labels"])
            n_positive_tokens += sum(label2id[label] > 0 for label in labels)
            n_tokens += len(labels)

        debate_ratios[debate] = n_positive_tokens / n_tokens if n_tokens > 0 else 0

    # Find the debate with the closest number of positive tokens to the mean
    mean_positive_tokens = sum(debate_ratios.values()) / len(debate_ratios)
    val_debate = min(
        debate_ratios, key=lambda x: abs(debate_ratios[x] - mean_positive_tokens)
    )

    return val_debate
