#!/usr/bin/env python

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from IPython.display import display

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

    fig, axs = plt.subplots(2, 2, figsize=(1.8 * 4, 2 * 4), squeeze=False)
    axs = axs.flatten()

    plots = [
        ("CORRECT", "fraction of CORRECT answers"),
        ("INCORRECT", "fraction of INCORRECT answers"),
        ("DOUBT", "fraction of DOUBT answers"),
        ("accuracy_given_answer", "accuracy of answers = C/(C + INC)"),
    ]
    for i, (metric, title) in enumerate(plots):
        plot_arrows(axs[i], metrics, metric)
        axs[i].set_title(title)

    fig.savefig(ANALYSIS / "fractions.png")

    def calculate_ratios(group):
        true_row = group[group["suggest_empty"] == True].iloc[0]
        false_row = group[group["suggest_empty"] == False].iloc[0]

        ratios = {}
        for col in ["INCORRECT", "DOUBT", "CORRECT", "accuracy_given_answer"]:
            ratios[f"{col}_ratio"] = (
                (true_row[col] / false_row[col] - 1)*100 if false_row[col] != 0 else float("inf")
            )

        return pd.Series(ratios)

    ratios = metrics.groupby(["short", "size"]).apply(calculate_ratios, include_groups=False)
    print(ratios)

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

    fig, ax = plt.subplots(1, 1, figsize=(2.4 * 2, 1.4 * 2))
    sns.heatmap(crosstab, annot=True, cmap="Blues", fmt="d")
    ax.set_title("Transition matrix")
    ax.set_ylabel("Base evaluation")
    ax.set_xlabel('"Suggest empty" evaluation')
    fig.savefig(ANALYSIS / "transition_matrix.png")

    crosstab = pd.crosstab(pivoted["base_eval"], pivoted["suggest_eval"], normalize="index")

    fig, ax = plt.subplots(1, 1, figsize=(2.4 * 2, 1.4 * 2))
    sns.heatmap(crosstab, annot=True, cmap="Blues", fmt=".2f")
    ax.set_title("Transition matrix (row-wise normalized)")
    ax.set_ylabel("Base evaluation")
    ax.set_xlabel('"Suggest empty" evaluation')
    fig.savefig(ANALYSIS / "transition_matrix_normalized.png")

    question_transitions = (
        pivoted.groupby(["question", "base_eval", "suggest_eval"]).size().reset_index(name="count")
    )

    question_transitions = question_transitions[
        question_transitions["base_eval"] != question_transitions["suggest_eval"]
    ].sort_values(by="count", ascending=False)

    print("\n=== Transition counts per question ===\n")
    with pd.option_context("display.max_colwidth", None):
        display(question_transitions.head(20))

    transition_counts = question_transitions.groupby(["base_eval", "suggest_eval"]).value_counts(
        ["count"]
    )
    print("\n=== Distribution of transition count per question ===\n")
    print(transition_counts)
