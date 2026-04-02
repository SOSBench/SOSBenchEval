import argparse
import csv
import json
import os


def collect_summaries(run_root: str) -> list[dict]:
    rows = []
    for root, _, files in os.walk(run_root):
        if "summary.json" not in files:
            continue
        path = os.path.join(root, "summary.json")
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        row = {
            "run_dir": summary.get("run_dir", root),
            "dataset": summary.get("dataset"),
            "dataset_split": summary.get("dataset_split"),
            "generation_model": summary.get("generation_model"),
            "evaluation_model": summary.get("evaluation_model"),
            "overall_unsafe_rate": summary.get("overall_unsafe_rate"),
            "judged": summary.get("counts", {}).get("judged"),
            "blocked_by_provider": summary.get("counts", {}).get("blocked_by_provider"),
            "other_terminal_non_judged": summary.get("counts", {}).get("other_terminal_non_judged"),
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["dataset"] or "", row["generation_model"] or "", row["run_dir"] or ""))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SOSBench v2 summaries into a leaderboard view.")
    parser.add_argument("--run-root", default="./runs")
    parser.add_argument("--format", choices=["json", "csv", "table"], default="table")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = collect_summaries(args.run_root)
    if args.format == "json":
        payload = json.dumps(rows, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        else:
            print(payload)
        return

    if args.format == "csv":
        fieldnames = list(rows[0].keys()) if rows else [
            "run_dir",
            "dataset",
            "dataset_split",
            "generation_model",
            "evaluation_model",
            "overall_unsafe_rate",
            "judged",
            "blocked_by_provider",
            "other_terminal_non_judged",
        ]
        if args.output:
            handle = open(args.output, "w", encoding="utf-8", newline="")
            close_handle = True
        else:
            import sys

            handle = sys.stdout
            close_handle = False
        try:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        finally:
            if close_handle:
                handle.close()
        return

    if not rows:
        print("No summaries found.")
        return

    header = [
        "dataset",
        "generation_model",
        "evaluation_model",
        "overall_unsafe_rate",
        "judged",
        "blocked_by_provider",
        "run_dir",
    ]
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows)) for key in header}
    line = " | ".join(key.ljust(widths[key]) for key in header)
    print(line)
    print("-+-".join("-" * widths[key] for key in header))
    for row in rows:
        print(" | ".join(str(row.get(key, "")).ljust(widths[key]) for key in header))
