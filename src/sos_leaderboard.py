import argparse
import csv
import json
import os


def normalize_name(value: str) -> str:
    safe = []
    for char in value:
        safe.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(safe).strip("_") or "unknown"


def collect_summaries(run_root: str) -> list[dict]:
    rows = []
    all_domains: set[str] = set()
    for root, _, files in os.walk(run_root):
        if "summary.json" not in files:
            continue
        path = os.path.join(root, "summary.json")
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        per_domain_scores = (
            summary.get("per_subject_policy_violation_rate")
            or summary.get("per_subject_unsafe_rate")
            or {}
        )
        for domain in per_domain_scores:
            all_domains.add(domain)
        row = {
            "run_dir": summary.get("run_dir", root),
            "dataset": summary.get("dataset"),
            "dataset_split": summary.get("dataset_split"),
            "generation_model": summary.get("generation_model"),
            "evaluation_model": summary.get("evaluation_model"),
            "overall_policy_violation_rate": (
                summary.get("overall_policy_violation_rate")
                if "overall_policy_violation_rate" in summary
                else summary.get("overall_unsafe_rate")
            ),
            "judged": summary.get("counts", {}).get("judged"),
            "blocked_by_provider": summary.get("counts", {}).get("blocked_by_provider"),
            "other_terminal_non_judged": summary.get("counts", {}).get("other_terminal_non_judged"),
            "per_domain_policy_violation_rate": per_domain_scores,
        }
        rows.append(row)

    domain_columns = {
        domain: f"domain_{normalize_name(domain)}_policy_violation_rate"
        for domain in sorted(all_domains)
    }
    for row in rows:
        per_domain_scores = row.get("per_domain_policy_violation_rate", {}) or {}
        for domain, column in domain_columns.items():
            row[column] = per_domain_scores.get(domain)

    rows.sort(key=lambda row: (row["dataset"] or "", row["generation_model"] or "", row["run_dir"] or ""))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SOSBench v2 summaries into a leaderboard view.")
    parser.add_argument("--run-root", default="./runs")
    parser.add_argument("--format", choices=["json", "csv", "table"], default="csv")
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
            "overall_policy_violation_rate",
            "judged",
            "blocked_by_provider",
            "other_terminal_non_judged",
            "per_domain_policy_violation_rate",
        ]
        output_path = args.output or os.path.join(args.run_root, "leaderboard.csv")
        if args.output:
            handle = open(args.output, "w", encoding="utf-8", newline="")
            close_handle = True
        else:
            handle = open(output_path, "w", encoding="utf-8", newline="")
            close_handle = True
        try:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        finally:
            if close_handle:
                handle.close()
        if not args.output:
            print(output_path)
        return

    if not rows:
        print("No summaries found.")
        return

    header = [
        "dataset",
        "generation_model",
        "evaluation_model",
        "overall_policy_violation_rate",
        "per_domain_policy_violation_rate",
        "judged",
        "blocked_by_provider",
        "run_dir",
    ]

    def _format_cell(row: dict, key: str) -> str:
        value = row.get(key, "")
        if key == "per_domain_policy_violation_rate":
            value = json.dumps(value, sort_keys=True)
        return str(value)

    widths = {
        key: max([len(key)] + [len(_format_cell(row, key)) for row in rows])
        for key in header
    }
    line = " | ".join(key.ljust(widths[key]) for key in header)
    print(line)
    print("-+-".join("-" * widths[key] for key in header))
    for row in rows:
        print(" | ".join(_format_cell(row, key).ljust(widths[key]) for key in header))
