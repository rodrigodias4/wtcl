from json import load, dumps
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console

console = Console()

LLM_MODELS = (
    "qwen3.5:9b",
    "qwen3.5:9b+ICL",
    "mistral:7b",
    "mistral:7b+ICL",
    "llama3.1:8b",
    "llama3.1:8b+ICL",
)
BASE_DIR = Path(__file__).parent.parent
BASELINE_FILES = {
    "RoBERTa+CRF": BASE_DIR / "inference_times_2026-09-21_03-03-19.json",
    "RoBERTa+SC": BASE_DIR / "segment_inference_times_2026-09-21_03-09-35.json",
}


def _load_llm_samples(model: str):
    has_icl = model.endswith("+ICL")
    model_name = model.removesuffix("+ICL")
    timing_files = sorted(
        path
        for path in (Path(__file__).parent / "output" / model_name).glob("*/times.json")
        if ("icl" in path.parent.name.lower()) == has_icl
    )
    if not timing_files:
        return []

    with timing_files[-1].open() as timing_file:
        timings = load(timing_file)

    samples = []
    for debate_timings in timings.values():
        for component, key in [
            ("TTFT", "prompt_eval_duration"),
            ("Generation", "eval_duration"),
        ]:
            samples.extend(
                {
                    "model": model,
                    "component": component,
                    "time_ms": duration / 1e6,
                }
                for duration in debate_timings[key]
                if duration is not None
            )
    return samples


def _load_baseline_samples(model: str, path: Path):
    with path.open() as timing_file:
        timings = load(timing_file)
    return [
        {"model": model, "component": "Inference", "time_ms": duration * 1e3}
        for duration in timings
    ]


def _load_all_samples():
    samples = [sample for model in LLM_MODELS for sample in _load_llm_samples(model)]
    samples.extend(
        sample
        for model, path in BASELINE_FILES.items()
        for sample in _load_baseline_samples(model, path)
    )
    return samples


def _summarize(samples):
    data = pd.DataFrame(samples)
    summaries = {}
    for model in data["model"].unique():
        model_data = data[data["model"] == model]
        percentiles = model_data["time_ms"].quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95])
        summaries[model] = {
            "mean": model_data["time_ms"].mean(),
            "std": model_data["time_ms"].std(ddof=0),
            "median": model_data["time_ms"].median(),
            "percentiles": {
                "5": percentiles.loc[0.05],
                "25": percentiles.loc[0.25],
                "50": percentiles.loc[0.5],
                "75": percentiles.loc[0.75],
                "90": percentiles.loc[0.9],
                "95": percentiles.loc[0.95],
            },
        }
    return summaries


def plot_time_violins(output_path: Path, combine: bool = True):
    samples = _load_all_samples()
    if not samples:
        raise ValueError("No raw LLM timing samples were found")

    data = pd.DataFrame(samples)
    llm_data = data[data["component"] != "Inference"]
    baseline_data = data[data["component"] == "Inference"]
    if combine:
        model_order = [
            model
            for model in LLM_MODELS + tuple(BASELINE_FILES)
            if model in data["model"].unique()
        ]
        figure, axis = plt.subplots(figsize=(10, 5))
        sns.violinplot(
            data=llm_data,
            x="model",
            y="time_ms",
            hue="component",
            order=model_order,
            split=True,
            inner="quartile",
            cut=0,
            density_norm="width",
            common_norm=False,
            palette={"TTFT": "tab:pink", "Generation": "tab:purple"},
            linecolor="black",
            linewidth=0.7,
            gap=0.05,
            ax=axis,
        )
        sns.violinplot(
            data=baseline_data,
            x="model",
            y="time_ms",
            order=model_order,
            inner="quartile",
            cut=0,
            density_norm="width",
            color="tab:blue",
            linecolor="black",
            linewidth=0.7,
            gap=0.05,
            ax=axis,
        )
        axis.set_xlabel("Model")
        axis.set_ylabel("Running time (ms)")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.3, zorder=0)
        axis.set_axisbelow(True)
        axis.set_yscale("log")
        axis.legend(title="LLM time component (additive)", loc="upper right", ncols=2)
        figure.tight_layout()
        figure.savefig(output_path, dpi=500)
        plt.close(figure)
        return

    figure, (llm_axis, baseline_axis) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]}
    )
    sns.violinplot(
        data=llm_data,
        x="model",
        y="time_ms",
        hue="component",
        split=True,
        inner="quartile",
        cut=0,
        common_norm=False,
        palette={"TTFT": "tab:pink", "Generation": "tab:purple"},
        ax=llm_axis,
    )
    sns.violinplot(
        data=baseline_data,
        x="model",
        y="time_ms",
        inner="quartile",
        cut=0,
        color="tab:blue",
        ax=baseline_axis,
    )
    llm_axis.set_xlabel("LLM")
    llm_axis.set_ylabel("Running time (ms)")
    llm_axis.tick_params(axis="x", rotation=30)
    llm_axis.grid(axis="y", alpha=0.3, zorder=0)
    llm_axis.set_axisbelow(True)
    llm_axis.legend(
        title="LLM time component (additive)",
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
    )
    baseline_axis.set_xlabel("Model")
    baseline_axis.set_ylabel("")
    baseline_axis.tick_params(axis="x", rotation=30)
    baseline_axis.grid(axis="y", alpha=0.3, zorder=0)
    baseline_axis.set_axisbelow(True)
    llm_axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(output_path, dpi=500)
    plt.close(figure)


def plot_time_boxes(output_path: Path, combine: bool = True):
    samples = _load_all_samples()
    if not samples:
        raise ValueError("No raw timing samples were found")

    data = pd.DataFrame(samples)
    llm_data = data[data["component"] != "Inference"]
    baseline_data = data[data["component"] == "Inference"]
    palette = {"TTFT": "tab:pink", "Generation": "tab:purple"}
    if combine:
        model_order = [
            model
            for model in LLM_MODELS + tuple(BASELINE_FILES)
            if model in data["model"].unique()
        ]
        figure, axis = plt.subplots(figsize=(7, 5))
        sns.boxplot(
            data=llm_data,
            x="model",
            y="time_ms",
            hue="component",
            order=model_order,
            palette=palette,
            width=0.7,
            gap=0.1,
            fliersize=2,
            linewidth=0.8,
            linecolor="black",
            ax=axis,
        )
        sns.boxplot(
            data=baseline_data,
            x="model",
            y="time_ms",
            order=model_order,
            color="tab:blue",
            width=0.5,
            fliersize=2,
            linewidth=0.8,
            linecolor="black",
            ax=axis,
        )
        axis.set_xlabel("Model")
        axis.set_ylabel("Running time (ms)")
        axis.set_xticklabels(axis.get_xticklabels(), rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.3, zorder=0)
        axis.set_axisbelow(True)
        axis.set_yscale("log")
        leg = axis.legend(title="LLM time\ncomponent\n(additive)", loc="upper right")
        leg.get_title().set_multialignment("center")
        figure.tight_layout()
        figure.savefig(output_path, dpi=500)
        plt.close(figure)
        return

    figure, (llm_axis, baseline_axis) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]}
    )
    sns.boxplot(
        data=llm_data,
        x="model",
        y="time_ms",
        hue="component",
        palette=palette,
        width=0.7,
        fliersize=2,
        linewidth=0.8,
        ax=llm_axis,
    )
    sns.boxplot(
        data=baseline_data,
        x="model",
        y="time_ms",
        color="tab:blue",
        width=0.5,
        fliersize=2,
        linewidth=0.8,
        ax=baseline_axis,
    )
    llm_axis.set_xlabel("LLM")
    llm_axis.set_ylabel("Running time (ms)")
    llm_axis.tick_params(axis="x", rotation=30)
    llm_axis.grid(axis="y", alpha=0.3, zorder=0)
    llm_axis.set_axisbelow(True)
    llm_axis.legend(
        title="LLM time component (additive)",
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
    )
    baseline_axis.set_xlabel("Model")
    baseline_axis.set_ylabel("")
    baseline_axis.tick_params(axis="x", rotation=30)
    baseline_axis.grid(axis="y", alpha=0.3, zorder=0)
    baseline_axis.set_axisbelow(True)
    llm_axis.set_yscale("log")
    baseline_axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(output_path, dpi=500)
    plt.close(figure)


def plot_time_comparison(output_path: Path):
    summaries = _summarize(_load_all_samples())
    llm_models = [model for model in LLM_MODELS if model in summaries]
    baseline_models = list(BASELINE_FILES)
    labels = llm_models + baseline_models
    llm_ttft = [summaries[model]["mean"] for model in llm_models]
    llm_ttft_std = [summaries[model]["std"] for model in llm_models]
    llm_generate = [0.0] * len(llm_models)
    llm_generate_std = [0.0] * len(llm_models)
    for index, model in enumerate(llm_models):
        model_data = pd.DataFrame(_load_llm_samples(model))
        ttft = model_data.loc[model_data["component"] == "TTFT", "time_ms"]
        generate = model_data.loc[model_data["component"] == "Generation", "time_ms"]
        llm_ttft[index] = ttft.mean()
        llm_ttft_std[index] = ttft.std(ddof=0)
        llm_generate[index] = generate.mean()
        llm_generate_std[index] = generate.std(ddof=0)
    classifier_means = [summaries[model]["mean"] for model in baseline_models]
    classifier_stds = [summaries[model]["std"] for model in baseline_models]
    totals = [
        ttft + generate for ttft, generate in zip(llm_ttft, llm_generate)
    ] + classifier_means

    figure, axis = plt.subplots(figsize=(6, 4))
    positions = list(range(len(labels)))
    axis.bar(
        positions[: len(llm_models)],
        llm_ttft,
        label="TTFT",
        color="tab:pink",
        edgecolor="black",
        linewidth=0.4,
    )
    axis.bar(
        positions[: len(llm_models)],
        llm_generate,
        bottom=llm_ttft,
        label="Generation",
        color="tab:purple",
        edgecolor="black",
        linewidth=0.4,
    )
    axis.bar(
        positions[len(llm_models) :],
        classifier_means,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.4,
    )

    # axis.errorbar(
    #     [position - 0.2 for position in positions[: len(llm_models)]],
    #     llm_ttft,
    #     yerr=llm_ttft_std,
    #     fmt="none",
    #     ecolor="black",
    #     capsize=3,
    #     label="Std dev",
    #     zorder=3,
    # )
    # axis.errorbar(
    #     [position + 0.2 for position in positions[: len(llm_models)]],
    #     totals[: len(llm_models)],
    #     yerr=llm_generate_std,
    #     fmt="none",
    #     ecolor="black",
    #     capsize=3,
    #     zorder=3,
    # )
    # axis.errorbar(
    #     positions[len(llm_models) :],
    #     classifier_means,
    #     yerr=classifier_stds,
    #     fmt="none",
    #     ecolor="black",
    #     capsize=3,
    #     zorder=3,
    # )

    for position, total in zip(positions, totals):
        axis.text(position, total, f"{total:.1f}", ha="center", va="bottom", fontsize=9)

    axis.set_xlabel("Model")
    axis.set_ylabel("Mean Running time (ms)")
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    # upper_bounds = [
    #     total + max(ttft_std, generate_std)
    #     for total, ttft_std, generate_std in zip(
    #         totals[: len(llm_models)], llm_ttft_std, llm_generate_std
    #     )
    # ] + [mean + std for mean, std in zip(classifier_means, classifier_stds)]
    axis.set_yscale("log")
    axis.set_ylim(0, max(totals) * 1.5)
    axis.grid(axis="y", alpha=0.3, zorder=0)
    axis.set_axisbelow(True)
    leg = axis.legend(
        title="LLM time component",
        alignment="center",
        fontsize=8,
        title_fontsize=8,
    )
    leg.get_title().set_multialignment("center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    plot_time_comparison(Path(__file__).with_name("time_comparison.png"))
    plot_time_violins(Path(__file__).with_name("time_comparison_violin.png"))
    plot_time_boxes(Path(__file__).with_name("time_comparison_box.png"))

    console.print(f"Summary:\n{dumps(_summarize(_load_all_samples()), indent=2)}")
