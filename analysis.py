#!/usr/bin/env python

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

EXPERIMENT = "base"

MODELS = Path("./input") / "models.jsonl"

OUTPUT = Path("./output") / f"{EXPERIMENT}.jsonl"

ANALYSIS = Path("./analysis")


def plot_arrows(ax, data, quant):
    text_elements = []
    for i, ((size, short), g) in enumerate(data.groupby(["size", "short"])):
        y = g[quant].tolist()
        label = f"{short}-{size}B"

        color = "darkgreen"
        if y[1] < y[0]:
            color = "darkred"

        ax.plot(
            [i, i],
            y,
            color=color,
            linewidth=4,
        )

        text_elements.append(label)

    ax.set_xticks(range(len(text_elements)))
    ax.set_xticklabels(text_elements, rotation=60, ha="center")
    ax.set_xlim(-0.5, len(text_elements) - 0.5)

    ax.set_ylim(0, 1)
    ax.grid(True)


if __name__ == "__main__":
    with MODELS.open("r") as f:
        models = pd.DataFrame(json.loads(line) for line in f).set_index("name")

    with OUTPUT.open("r") as f:
        data = pd.DataFrame(json.loads(line) for line in f)

    grouped = data.groupby(["model", "suggest_empty"])

    metrics = []
    for (model_name, suggest_empty), group in grouped:
        total = len(group)
        evals = group["evaluation"].value_counts(normalize=True)

        answered = evals["CORRECT"] + evals["INCORRECT"]

        metrics.append(
            {
                "name": model_name,
                **models.loc[model_name, :].to_dict(),
                "suggest_empty": suggest_empty,
                **evals.to_dict(),
                "accuracy_given_answer": evals["CORRECT"] / answered if answered > 0 else None,
            }
        )

    metrics = pd.DataFrame(metrics).fillna(0.0).sort_values(by=["short", "size"])

    print(f"\n{metrics}\n")

    figsize = (1.8 * 2, 2 * 2)

    fig, axs = plt.subplots(2, 2, figsize=(figsize[0] * 2, figsize[1] * 2), squeeze=False)

    axs = axs.flatten()

    plot_arrows(axs[0], metrics, "CORRECT")
    axs[0].set_title("fraction of CORRECT answers")

    plot_arrows(axs[1], metrics, "INCORRECT")
    axs[1].set_title("fraction of INCORRECT answers")

    plot_arrows(axs[2], metrics, "DOUBT")
    axs[2].set_title("fraction of DOUBT answers")

    plot_arrows(axs[3], metrics, "accuracy_given_answer")
    axs[3].set_title("accuracy of answers = C/(C + INC)")

    fig.savefig(ANALYSIS / "analysis.png")

    pivoted = (
        data.pivot_table(
            index=["model", "question"],
            columns="suggest_empty",
            values="evaluation",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={False: "base_eval", True: "suggest_eval"})
    )

    transitions_df = pivoted.groupby(["base_eval", "suggest_eval"]).size().reset_index(name="count")

    print("\n=== Transition counts (base → suggest_empty) ===\n")
    print(transitions_df)

    crosstab = pd.crosstab(pivoted["base_eval"], pivoted["suggest_eval"])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(crosstab, annot=True, cmap="Blues", fmt="d")
    ax.set_title("Transition matrix")
    ax.set_ylabel("Base evaluation")
    ax.set_xlabel("\"Suggest empty\" evaluation")
    fig.savefig(ANALYSIS / "transition_matrix.png")
