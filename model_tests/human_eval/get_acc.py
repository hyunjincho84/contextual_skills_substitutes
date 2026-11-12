#!/usr/bin/env python3
"""Print Top-k accuracy for the human-evaluation model predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path(
    "/home/jovyan/LEM_data3/hyunjincho/LEM/human_eval/"
    "skill_or_skill_shuffled.csv"
)

MODEL_COLUMNS = {
    "BERT": "bert_pred_top5",
    "GPT": "gpt_top5_pred",
    "W2V": "skill2vec_pred_top5",
    "CondProb": "condprob_pred_top5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Top-1/3/5 accuracy for each prediction model."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Prediction CSV (default: {DEFAULT_INPUT})",
    )
    return parser.parse_args()


def normalize_skill(value: str | None) -> str:
    return "" if value is None else value.strip().casefold()


def parse_predictions(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        normalized
        for item in value.split("|")
        if (normalized := normalize_skill(item))
    ][:5]


def evaluate(input_csv: Path) -> None:
    counts = {
        model: {"available": 0, "top1": 0, "top3": 0, "top5": 0}
        for model in MODEL_COLUMNS
    }
    total = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"mask_answer", *MODEL_COLUMNS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            total += 1
            answer = normalize_skill(row["mask_answer"])
            if not answer:
                raise ValueError(f"Empty mask_answer at CSV data row {total}")

            for model, column in MODEL_COLUMNS.items():
                predictions = parse_predictions(row[column])
                if predictions:
                    counts[model]["available"] += 1
                counts[model]["top1"] += int(answer in predictions[:1])
                counts[model]["top3"] += int(answer in predictions[:3])
                counts[model]["top5"] += int(answer in predictions[:5])

    if total == 0:
        raise ValueError(f"No data rows found: {input_csv}")

    print(f"Samples: {total:,}")
    for model, model_counts in counts.items():
        available = model_counts["available"]
        print(f"\n[{model}]")
        print(f"Coverage: {available:,}/{total:,} ({available / total:.4f})")
        print(
            f"Top-1 Acc: {model_counts['top1'] / total:.4f} "
            f"({model_counts['top1']:,}/{total:,})"
        )
        print(
            f"Top-3 Acc: {model_counts['top3'] / total:.4f} "
            f"({model_counts['top3']:,}/{total:,})"
        )
        print(
            f"Top-5 Acc: {model_counts['top5'] / total:.4f} "
            f"({model_counts['top5']:,}/{total:,})"
        )


def main() -> None:
    args = parse_args()
    evaluate(args.input_csv)


if __name__ == "__main__":
    main()
