import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir",
    type=Path,
    default=Path("qwen_test_3pct"),
    help="Benchmark output directory containing join_stats.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output image path. Defaults to <results-dir>/join_quality_runtime.png.",
)
args = parser.parse_args()

root = args.results_dir / "join_stats"
output = args.output or args.results_dir / "join_quality_runtime.png"

rows = []

if not root.exists():
    raise SystemExit(f"No join_stats directory found at {root}")

for csv_path in root.rglob("*_metrics.csv"):
    df = pd.read_csv(csv_path)

    if df.empty or not {"approach_name", "f1_score"}.issubset(df.columns):
        continue

    for method, group in df.groupby("approach_name"):
        cache_json = csv_path.parent / "cache" / method / f"{method}.json"
        if not cache_json.exists():
            print(f"Warning: no runtime stats found for {method}: {cache_json}")
            continue

        with cache_json.open() as file:
            stats = json.load(file)

        runtimes = []
        for result in stats.values():
            component_times = result.get("cost", {}).get("component_times", {})
            runtime = component_times.get("end_to_end")
            if runtime is None:
                runtime = component_times.get("execution")
            if runtime is not None and runtime > 0:
                runtimes.append(runtime)

        if not runtimes:
            print(f"Warning: runtime stats are empty for {method}: {cache_json}")
            continue

        rows.append({
            "method": method,
            "runtime": sum(runtimes) / len(runtimes),
            "f1": group["f1_score"].mean(),
        })

if not rows:
    raise SystemExit(f"No usable metrics CSV files found below {root}")

points = pd.DataFrame(rows).groupby("method", as_index=False).mean(numeric_only=True)

fig, ax = plt.subplots(figsize=(8, 5))

for _, row in points.iterrows():
    ax.scatter(row["runtime"], row["f1"], s=70)
    ax.annotate(
        row["method"],
        (row["runtime"], row["f1"]),
        xytext=(5, 5),
        textcoords="offset points",
    )

ax.set_xscale("log")
ax.set_xlabel("Average execution runtime (seconds, log scale)")
ax.set_ylabel("Average F1 score")
ax.set_title("Join quality/runtime trade-off")
ax.grid(True, which="both", alpha=0.25)

plt.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=300)
print(f"Saved plot to {output}")
plt.show()