import argparse
import json
import random
from pathlib import Path


def load_records(path):
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    indexed = {record["index"]: record for record in records}
    assert len(indexed) == len(records), f"重复 index: {path}"
    return indexed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()
    assert args.per_category > 0

    run_dir = Path(args.run)
    before = load_records(run_dir / "eval/step_0000/responses.jsonl")
    after = load_records(run_dir / "eval/step_0200/responses.jsonl")

    assert before.keys() == after.keys(), "前后评测题目不一致"

    groups = {
        "wrong_to_correct": [],
        "wrong_to_wrong": [],
        "correct_to_wrong": [],
        "correct_to_correct": [],
    }

    for index in sorted(before):
        old, new = before[index], after[index]
        assert old["question"] == new["question"]
        assert old["ground_truth"] == new["ground_truth"]

        old_correct = old["reward"] == 1.0
        new_correct = new["reward"] == 1.0

        if not old_correct and new_correct:
            category = "wrong_to_correct"
        elif not old_correct and not new_correct:
            category = "wrong_to_wrong"
        elif old_correct and not new_correct:
            category = "correct_to_wrong"
        else:
            category = "correct_to_correct"

        groups[category].append(index)

    rng = random.Random(0)
    comparisons = []

    for category, indices in groups.items():
        print(f"{category}: {len(indices)} / {len(before)}")

        selected = rng.sample(
            indices, min(args.per_category, len(indices))
        )
        for index in selected:
            comparisons.append({
                "category": category,
                "index": index,
                "question": before[index]["question"],
                "ground_truth": before[index]["ground_truth"],
                "before": before[index],
                "after": after[index],
            })

    # 不覆盖已有输出。
    output_path = run_dir / "response_comparisons.json"
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(
            {
                "selection_seed": 0,
                "category_counts": {
                    key: len(value) for key, value in groups.items()
                },
                "examples": comparisons,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()