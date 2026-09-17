"""
AI-Augmented Ticket Triage Agent
=================================
Phase 3 of the IT Service Ticket Classification initiative.

What this does
--------------
1. Loads the same 4,000-ticket sample used in Phase 1.
2. Splits it into a training pool (used only to build a handful of
   few-shot examples) and a held-out test set the agent never sees
   during "training".
3. Sends the held-out tickets to Claude (in small batches) and asks it
   to classify each one into one of the 8 known categories.
4. Scores the predictions against the real labels -- honestly, on data
   the model never saw -- and reports accuracy, a per-category
   breakdown, and a confusion matrix.
5. Flags any ticket the agent classified as "Storage" as a
   self-service candidate, because Phase 1 found Storage tickets are
   low-volume and the least complex category of all 8 -- the same
   finding that justified the Phase 2 BRD.

Requirements
------------
- An Anthropic API key, set as the environment variable ANTHROPIC_API_KEY
  (never hard-code a key in this file or commit one to git).
- pip install anthropic pandas scikit-learn matplotlib

Run
---
    python triage_agent.py
"""

import json
import os
import random
import sys
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    import anthropic
except ImportError:
    print("Missing dependency. Run: pip install anthropic")
    sys.exit(1)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_PATH = "data/it_tickets_sample_4000.csv"
MODEL = os.environ.get("TRIAGE_MODEL", "claude-haiku-4-5")
TEST_FRACTION = 0.06          # ~240 held-out tickets, kept small to control API cost/time
FEWSHOT_PER_CATEGORY = 2      # a couple of labeled examples per category, shown to the model
BATCH_SIZE = 15               # tickets classified per API call
RANDOM_SEED = 42
SELF_SERVICE_CATEGORY = "Storage"  # matches the Phase 1 finding / Phase 2 BRD scope

CATEGORIES = [
    "Hardware", "HR Support", "Access", "Miscellaneous",
    "Storage", "Purchase", "Internal Project", "Administrative rights",
]


def load_and_split():
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_FRACTION,
        stratify=df["Topic_group"],
        random_state=RANDOM_SEED,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_fewshot_examples(train_df):
    """Pick a couple of real, labeled tickets per category to show the
    model what each category looks like. These examples are held out
    of the test set already, since they come from train_df."""
    examples = []
    for cat in CATEGORIES:
        subset = train_df[train_df["Topic_group"] == cat]
        picks = subset.sample(n=min(FEWSHOT_PER_CATEGORY, len(subset)), random_state=RANDOM_SEED)
        for _, row in picks.iterrows():
            examples.append({"text": row["Document"][:300], "label": cat})
    return examples


def build_system_prompt(fewshot_examples):
    lines = [
        "You are an IT service-desk ticket classifier.",
        f"Classify each ticket into exactly one of these {len(CATEGORIES)} categories:",
        ", ".join(CATEGORIES) + ".",
        "",
        "Here are a few labeled examples:",
    ]
    for ex in fewshot_examples:
        lines.append(f'- "{ex["text"]}" -> {ex["label"]}')
    lines += [
        "",
        "You will be given a numbered batch of new tickets. Respond with ONLY a JSON array",
        'of objects like [{"id": 1, "category": "Hardware"}, ...], one entry per ticket,',
        "in the same order. Use exactly one of the category names listed above for every entry.",
        "No prose, no markdown fences, just the JSON array.",
    ]
    return "\n".join(lines)


def classify_batch(client, system_prompt, batch_df):
    numbered = "\n".join(
        f"{i+1}. {text[:500]}" for i, text in enumerate(batch_df["Document"].tolist())
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": numbered}],
    )
    raw = response.content[0].text.strip()
    # Be tolerant of accidental markdown fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print("Could not parse model response as JSON:\n", raw)
        parsed = []
    preds = ["Miscellaneous"] * len(batch_df)  # safe fallback if parsing partially fails
    for item in parsed:
        idx = item.get("id", 0) - 1
        cat = item.get("category", "")
        if 0 <= idx < len(preds) and cat in CATEGORIES:
            preds[idx] = cat
    return preds


def run_evaluation():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set. Export it before running this script.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    random.seed(RANDOM_SEED)

    train_df, test_df = load_and_split()
    fewshot_examples = build_fewshot_examples(train_df)
    system_prompt = build_system_prompt(fewshot_examples)

    print(f"Model: {MODEL}")
    print(f"Held-out test set: {len(test_df)} tickets (never used in the few-shot examples)")
    print(f"Few-shot examples shown to the model: {len(fewshot_examples)}\n")

    all_preds = []
    for start in range(0, len(test_df), BATCH_SIZE):
        batch = test_df.iloc[start:start + BATCH_SIZE]
        preds = classify_batch(client, system_prompt, batch)
        all_preds.extend(preds)
        print(f"Classified {min(start + BATCH_SIZE, len(test_df))}/{len(test_df)}")

    test_df = test_df.copy()
    test_df["Predicted"] = all_preds
    test_df["Self_Service_Candidate"] = test_df["Predicted"] == SELF_SERVICE_CATEGORY

    y_true = test_df["Topic_group"]
    y_pred = test_df["Predicted"]

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall accuracy on held-out tickets: {acc:.1%}")
    print("\nPer-category report:")
    print(classification_report(y_true, y_pred, labels=CATEGORIES, zero_division=0))

    os.makedirs("results", exist_ok=True)
    test_df.to_csv("results/predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    with open("results/metrics.json", "w") as f:
        json.dump({
            "model": MODEL,
            "test_size": len(test_df),
            "accuracy": acc,
            "categories": CATEGORIES,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    plot_confusion_matrix(cm, CATEGORIES)
    print("\nSaved: results/predictions.csv, results/metrics.json, results/confusion_matrix.png")


def plot_confusion_matrix(cm, categories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(categories)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title("Triage Agent — Confusion Matrix (held-out tickets)")

    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(len(categories)):
        for j in range(len(categories)):
            val = cm[i, j]
            color = "white" if val > max_val / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_evaluation()
