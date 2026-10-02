"""
Stage 4: evaluation.

Measures how well the app assigns severity and root cause, by comparing its
predictions against the ground-truth labels baked into the synthetic data.

Leakage handling: every incident we test also lives in the ChromaDB index,
so naive retrieval could return the incident itself as one of its own
"similar past incidents", flattering the score. We avoid that by retrieving
a few extra candidates and dropping any whose ID matches the incident under
test, so an item never informs its own assessment.

Run from the project root:  python -m rag.evaluate
"""

import json
import random

from rag.config import DATA_PATH, OPENAI_API_KEY, LLM_MODEL, TOP_K, check_key
from rag.labels import SEVERITY_LEVELS
from rag.retrieve import Retriever
from rag.generate import build_prompt, IncidentAssessment
from openai import OpenAI


SAMPLE_SIZE = 20
RANDOM_SEED = 42   # same sample every run, so results are reproducible


def retrieve_excluding_self(retriever, incident):
    """Top-k similar incidents, with the incident itself removed."""
    # Ask for one more than we need, so after dropping self we still have k.
    candidates = retriever.retrieve(incident["narrative"], k=TOP_K + 1)
    kept = [c for c in candidates if c["id"] != incident["id"]]
    return kept[:TOP_K]


def assess_one(client, retriever, incident):
    """Run the full pipeline on one incident and return its prediction."""
    retrieved = retrieve_excluding_self(retriever, incident)
    system, user = build_prompt(incident["narrative"], retrieved)
    completion = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=IncidentAssessment,
    )
    return completion.choices[0].message.parsed


def severity_distance(pred, true):
    """How many levels apart two severities are (0 = exact match)."""
    order = {level: i for i, level in enumerate(SEVERITY_LEVELS)}
    if pred not in order:          # model returned something off-scale
        return len(SEVERITY_LEVELS)
    return abs(order[pred] - order[true])


def severity_confusion_matrix(rows):
    """
    Build a true-vs-predicted count grid for severity.

    Returns a nested dict: matrix[true_label][predicted_label] = count.
    Any predicted value outside the known levels is bucketed as "other".
    """
    levels = SEVERITY_LEVELS + ["other"]
    matrix = {t: {p: 0 for p in levels} for t in SEVERITY_LEVELS}
    for r in rows:
        true = r["true_severity"]
        pred = r["pred_severity"] if r["pred_severity"] in SEVERITY_LEVELS else "other"
        matrix[true][pred] += 1
    return matrix


def print_confusion_matrix(matrix):
    """Print the severity confusion matrix as a readable grid."""
    cols = SEVERITY_LEVELS + ["other"]
    # Drop the "other" column if nothing landed there, to keep it tidy.
    if sum(matrix[t]["other"] for t in SEVERITY_LEVELS) == 0:
        cols = SEVERITY_LEVELS

    print("\nSeverity confusion matrix (rows = true, columns = predicted)")
    header = "true \\ pred".ljust(12) + "".join(c.rjust(8) for c in cols)
    print(header)
    for t in SEVERITY_LEVELS:
        line = t.ljust(12) + "".join(str(matrix[t][c]).rjust(8) for c in cols)
        print(line)


def main():
    check_key()

    with open(DATA_PATH) as f:
        all_incidents = json.load(f)

    # Pick a fixed random sample.
    random.seed(RANDOM_SEED)
    sample = random.sample(all_incidents, SAMPLE_SIZE)

    retriever = Retriever()
    client = OpenAI(api_key=OPENAI_API_KEY)

    severity_correct = 0
    severity_within_one = 0
    root_cause_correct = 0
    rows = []

    print(f"Evaluating on {SAMPLE_SIZE} incidents (leakage-excluded)...\n")

    for i, inc in enumerate(sample, 1):
        pred = assess_one(client, retriever, inc)

        sev_dist = severity_distance(pred.severity, inc["true_severity"])
        sev_ok = sev_dist == 0
        sev_within_one = sev_dist <= 1
        rc_ok = pred.root_cause == inc["true_root_cause"]

        severity_correct += sev_ok
        severity_within_one += sev_within_one
        root_cause_correct += rc_ok

        rows.append({
            "id": inc["id"],
            "type": inc["incident_type"],
            "true_severity": inc["true_severity"],
            "pred_severity": pred.severity,
            "severity_ok": sev_ok,
            "severity_distance": sev_dist,
            "true_root_cause": inc["true_root_cause"],
            "pred_root_cause": pred.root_cause,
            "root_cause_ok": rc_ok,
        })

        mark = "ok " if sev_ok else "X  "
        print(f"{i:2}. {inc['id']}  sev {inc['true_severity']:>6} -> "
              f"{pred.severity:>6} [{mark}]  "
              f"rc {inc['true_root_cause']} -> {pred.root_cause}")

    sev_acc = severity_correct / SAMPLE_SIZE
    sev_w1_acc = severity_within_one / SAMPLE_SIZE
    rc_acc = root_cause_correct / SAMPLE_SIZE

    print("\n" + "=" * 55)
    print("RESULTS")
    print(f"  Severity accuracy (exact):     {severity_correct}/{SAMPLE_SIZE} = {sev_acc:.0%}")
    print(f"  Severity accuracy (within 1):  {severity_within_one}/{SAMPLE_SIZE} = {sev_w1_acc:.0%}")
    print(f"  Root-cause accuracy:           {root_cause_correct}/{SAMPLE_SIZE} = {rc_acc:.0%}")
    print("=" * 55)

    # Severity confusion matrix: rows = true, columns = predicted.
    # Shows not just how often it was wrong, but which class it confused
    # for which, so systematic bias (e.g. hedging to "medium") is visible.
    matrix = severity_confusion_matrix(rows)
    print_confusion_matrix(matrix)

    # Save full results so you can inspect or chart them later.
    with open("eval_results.json", "w") as f:
        json.dump({
            "sample_size": SAMPLE_SIZE,
            "severity_accuracy_exact": sev_acc,
            "severity_accuracy_within_one": sev_w1_acc,
            "root_cause_accuracy": rc_acc,
            "severity_confusion_matrix": matrix,
            "rows": rows,
        }, f, indent=2)
    print("\nFull breakdown saved to eval_results.json")


if __name__ == "__main__":
    main()
