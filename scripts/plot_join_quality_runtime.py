import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir",
    type=Path,
    default=Path("qwen_test_1pct"),
    help="Benchmark output directory containing join_stats.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Defaults to <results-dir>/join_quality_runtime_panels.png.",
)
parser.add_argument(
    "--queries-per-panel",
    type=int,
    default=None,
    help="Optional number of queries per panel. If omitted, all queries use one plot.",
)
args = parser.parse_args()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
    }
)

root = args.results_dir / "join_stats"
output = args.output or args.results_dir / "join_quality_runtime_panels.png"

if not root.exists():
    raise SystemExit(f"No join_stats directory found at {root}")


def infer_join_flow(pipelines):
    for pipeline_text in pipelines:
        try:
            pipeline = json.loads(pipeline_text)
        except (json.JSONDecodeError, TypeError):
            continue

        for step in pipeline:
            operator = step.get("operator", "")
            if "ExtractAndMatchFilter" in operator:
                return "Extract + Match"
            if "ExtractAndQaFilter" in operator:
                return "Extract + QA"
            if "ExtractAndDecisionFilter" in operator:
                return "Extract + Decision"
            if "DecisionModelTextQABackend" in operator:
                return "Decision"
            if "TextQaFilter" in operator:
                return "Text QA"
            if "TraditionalJoin" in operator:
                return "Traditional Join"
    return "Unknown"


def get_runtime(stats, method):
    runtimes = []
    for result in stats.values():
        component_times = result.get("cost", {}).get("component_times", {})
        runtime = component_times.get("end_to_end")
        if runtime is None:
            runtime = component_times.get("execution")
        if runtime is not None and runtime > 0:
            runtimes.append(runtime)
    return sum(runtimes) / len(runtimes) if runtimes else None


rows = []

for csv_path in root.rglob("*_metrics.csv"):
    metrics = pd.read_csv(csv_path)
    required = {"approach_name", "query", "f1_score"}
    if metrics.empty or not required.issubset(metrics.columns):
        continue

    query_order = list(dict.fromkeys(metrics["query"]))
    query_numbers = {
        query: index + 1
        for index, query in enumerate(query_order)
    }

    for method, group in metrics.groupby("approach_name"):
        cache_json = csv_path.parent / "cache" / method / f"{method}.json"
        if not cache_json.exists():
            print(f"Warning: no runtime stats found for {method}: {cache_json}")
            continue

        with cache_json.open() as cache_file:
            stats = json.load(cache_file)

        runtime = get_runtime(stats, method)
        if runtime is None:
            print(f"Warning: runtime stats are empty for {method}: {cache_json}")
            continue

        for _, metric in group.iterrows():
            query = metric["query"]
            query_number = query_numbers.get(query)
            if query_number is None:
                continue
            if query_number in (9, 10):
                continue

            metric_runtime = metric.get("execution_cost_runtime")
            if pd.notna(metric_runtime) and float(metric_runtime) > 0:
                point_runtime = float(metric_runtime)
            else:
                point_runtime = runtime

            flow = "Unknown"
            for result in stats.values():
                result_flow = infer_join_flow(result.get("pipelines", []))
                if result_flow != "Unknown":
                    flow = result_flow
                    break

            rows.append({
                "panel": "All Queries",
                "query": query,
                "query_number": query_number,
                "method": method,
                "series": flow,
                "label": method,
                "runtime_sec": point_runtime,
                "f1_score": float(metric["f1_score"]),
            })

if not rows:
    raise SystemExit(f"No usable metrics CSV files found below {root}")

data = pd.DataFrame(rows)

METHOD_DISPLAY_NAMES = {
    "x-dec": "extract&decision",
}

# Keep each query point separate; each method becomes a line across its queries.
summary = data.copy()

panels = list(dict.fromkeys(summary["panel"]))
methods = list(dict.fromkeys(summary["method"]))
method_styles = {
    "Extract + Match": {"color": "#4C78A8", "marker": "o"},
    "Extract + QA": {"color": "#F2C14E", "marker": "s"},
    "Extract + Decision": {"color": "#A45BB7", "marker": "^"},
    "Decision": {"color": "#E67E22", "marker": "D"},
    "Text QA": {"color": "#A65628", "marker": "P"},
    "Traditional Join": {"color": "#59A14F", "marker": "X"},
    "Unknown": {"color": "#777777", "marker": "*"},
}

# Fall back to unique colors for method names such as vanillaQwen3B/x-dec/dec.
colors = list(plt.cm.tab10(range(max(len(methods), 1))))
method_colors = {
    method: method_styles.get(method, {"color": None})["color"]
    for method in methods
}
for index, method in enumerate(methods):
    if method_colors[method] is None:
        method_colors[method] = colors[index % len(colors)]

fig, axes = plt.subplots(
    1,
    1,
    figsize=(8.5, 5.8),
)
axes = [axes]

for ax, panel in zip(axes, panels):
    panel_data = summary[summary["panel"] == panel].copy()
    all_points = []

    for method in methods:
        method_data = panel_data[panel_data["method"] == method].sort_values(
            ["runtime_sec", "query_number"]
        )
        if method_data.empty:
            continue

        style = method_styles.get(method)
        marker = style["marker"] if style else "o"

        display_method = METHOD_DISPLAY_NAMES.get(method, method)
        ax.plot(
            method_data["runtime_sec"],
            method_data["f1_score"],
            color=method_colors[method],
            marker=marker,
            markersize=8,
            linewidth=2.0,
            alpha=0.95,
            label=display_method,
            zorder=3,
        )

        for _, point in method_data.iterrows():
            all_points.append(point)

    all_points.sort(key=lambda p: (p["runtime_sec"], p["f1_score"], p["method"]))
    offsets = [
        (8, 8, "left", "bottom"),
        (8, -10, "left", "top"),
        (-8, 8, "right", "bottom"),
        (-8, -10, "right", "top"),
    ]

    for idx, point in enumerate(all_points):
        dx, dy, ha, va = offsets[idx % len(offsets)]
        ax.annotate(
            f"Q{int(point['query_number'])}",
            (point["runtime_sec"], point["f1_score"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="medium",
            color="#222222",
            ha=ha,
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="#DDDDDD",
                lw=0.5,
                alpha=0.9,
            ),
            zorder=4,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_title(panel, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Avg Exec. Runtime (sec., log scale)", fontsize=10, labelpad=6)

    x_min, x_max = ax.get_xlim()
    if x_min > 0 and x_max > 0:
        ax.set_xlim(x_min * 0.65, x_max * 1.45)
    ax.set_ylim(-0.08, 1.08)

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5, zorder=1)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

axes[0].set_ylabel("Avg F1 Score", fontsize=10)

handles, labels = axes[-1].get_legend_handles_labels()
if handles:
    fig.legend(
        handles,
        labels,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=min(len(handles), 4),
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        handlelength=1.8,
        columnspacing=1.2,
    )

fig.suptitle(
    "Join quality / runtime trade-off",
    fontsize=13,
    fontweight="bold",
    y=0.97,
)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.15)

output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output}")