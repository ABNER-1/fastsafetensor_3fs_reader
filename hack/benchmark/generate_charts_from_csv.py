#!/usr/bin/env python3
"""Standalone chart generator from benchmark CSV files.

This script reads benchmark_summary.csv (and optionally benchmark_raw.csv)
and generates visualization charts without requiring the original benchmark
to be re-run.

Usage:
    # Basic usage - generate all charts
    python generate_charts_from_csv.py --csv-dir ./benchmark_results

    # Specify output directory
    python generate_charts_from_csv.py --csv-dir ./benchmark_results --output-dir ./charts

    # Generate only specific chart types
    python generate_charts_from_csv.py --csv-dir ./benchmark_results --charts heatmap,lineplot

    # Include data interpretation report
    python generate_charts_from_csv.py --csv-dir ./benchmark_results --generate-report

    # Generate PDF format instead of PNG
    python generate_charts_from_csv.py --csv-dir ./benchmark_results --format pdf
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CSV Data Loading
# ---------------------------------------------------------------------------

def load_summary_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load benchmark_summary.csv and return list of summary records."""
    records = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {
                "backend": row["backend"],
                "buffer_size_mb": float(row["buffer_size_mb"]),
                "chunk_size_mb": float(row["chunk_size_mb"]),
                "num_processes": int(row["num_processes"]),
                "throughput_gbps_median": float(row["throughput_gbps_median"]),
                "throughput_gbps_mean": float(row["throughput_gbps_mean"]),
                "throughput_gbps_min": float(row["throughput_gbps_min"]),
                "throughput_gbps_max": float(row["throughput_gbps_max"]),
                "throughput_gbps_std": float(row["throughput_gbps_std"]),
                "wall_time_median": float(row["wall_time_median"]),
                "wall_time_mean": float(row["wall_time_mean"]),
                "num_iterations": int(row["num_iterations"]),
            }
            records.append(record)
    return records


def load_raw_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load benchmark_raw.csv and return list of raw records."""
    records = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {
                "backend": row["backend"],
                "buffer_size": int(row["buffer_size"]),
                "chunk_size": int(row["chunk_size"]),
                "num_processes": int(row["num_processes"]),
                "rank": int(row["rank"]),
                "iteration": int(row["iteration"]),
                "num_files": int(row["num_files"]),
                "total_bytes_read": int(row["total_bytes_read"]),
                "read_time": float(row["read_time"]),
                "wall_time": float(row["wall_time"]),
                "throughput_mbps": float(row["throughput_mbps"]),
                "throughput_gbps": float(row["throughput_gbps"]),
                "round_wall_time": float(row["round_wall_time"]) if row["round_wall_time"] else None,
                "success": row["success"].lower() == "true",
                "error": row.get("error", ""),
            }
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Chart Generation
# ---------------------------------------------------------------------------

def generate_heatmaps(
    summaries: List[Dict[str, Any]],
    output_dir: str,
    fmt: str = "png",
) -> List[str]:
    """Generate heatmap charts showing buffer_size vs chunk_size for each backend and num_processes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"ERROR: Missing required dependency: {e}")
        print("Please install: pip install matplotlib numpy")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    generated: List[str] = []

    # Collect unique dimensions
    backends = sorted({s["backend"] for s in summaries})
    all_procs = sorted({s["num_processes"] for s in summaries})
    all_buf = sorted({s["buffer_size_mb"] for s in summaries})
    all_chunk = sorted({s["chunk_size_mb"] for s in summaries})

    # Build lookup: (backend, buf, chunk, procs) -> median throughput
    lookup: Dict[Tuple, float] = {}
    for s in summaries:
        key = (
            s["backend"],
            s["buffer_size_mb"],
            s["chunk_size_mb"],
            s["num_processes"],
        )
        lookup[key] = s["throughput_gbps_median"]

    for backend in backends:
        for nprocs in all_procs:
            # Build 2-D matrix: rows=buffer_size, cols=chunk_size
            matrix = np.full((len(all_buf), len(all_chunk)), np.nan)
            for i, buf in enumerate(all_buf):
                for j, chunk in enumerate(all_chunk):
                    val = lookup.get((backend, buf, chunk, nprocs))
                    if val is not None:
                        matrix[i, j] = val

            # Skip if all NaN
            if np.all(np.isnan(matrix)):
                continue

            fig, ax = plt.subplots(figsize=(10, 7))
            im = ax.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                cmap="YlOrRd",
            )
            ax.set_xticks(range(len(all_chunk)))
            ax.set_xticklabels([str(c) for c in all_chunk])
            ax.set_yticks(range(len(all_buf)))
            ax.set_yticklabels([str(b) for b in all_buf])
            ax.set_xlabel("chunk_size (MB)")
            ax.set_ylabel("buffer_size (MB)")
            ax.set_title(
                f"Throughput (GB/s) -- backend={backend}, procs={nprocs}"
            )
            fig.colorbar(im, ax=ax, label="GB/s")

            # Annotate cells with values
            for i in range(len(all_buf)):
                for j in range(len(all_chunk)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j, i, f"{val:.2f}",
                            ha="center", va="center",
                            color="black" if val < np.nanmax(matrix) * 0.7 else "white",
                            fontsize=8,
                        )

            fname = f"heatmap_{backend}_{nprocs}procs.{fmt}"
            path = os.path.join(output_dir, fname)
            fig.tight_layout()
            fig.savefig(path, dpi=150, format=fmt)
            plt.close(fig)
            generated.append(path)
            print(f"Chart saved: {path}")

    return generated


def generate_lineplot(
    summaries: List[Dict[str, Any]],
    output_dir: str,
    fmt: str = "png",
) -> Optional[str]:
    """Generate line plot showing throughput vs num_processes for each backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"ERROR: Missing required dependency: {e}")
        print("Please install: pip install matplotlib")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Collect unique dimensions
    backends = sorted({s["backend"] for s in summaries})
    all_procs = sorted({s["num_processes"] for s in summaries})

    # Build lookup
    lookup: Dict[Tuple, float] = {}
    for s in summaries:
        key = (
            s["backend"],
            s["buffer_size_mb"],
            s["chunk_size_mb"],
            s["num_processes"],
        )
        lookup[key] = s["throughput_gbps_median"]

    # Find best config per backend
    best_configs: Dict[str, Tuple[int, int]] = {}
    for backend in backends:
        backend_summaries = [s for s in summaries if s["backend"] == backend]
        if backend_summaries:
            best = max(
                backend_summaries, key=lambda s: s["throughput_gbps_median"]
            )
            best_configs[backend] = (
                best["buffer_size_mb"],
                best["chunk_size_mb"],
            )

    if not best_configs or len(all_procs) <= 1:
        print("Skipping line plot: insufficient data")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "p", "*"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#795548"]

    for idx, backend in enumerate(backends):
        if backend not in best_configs:
            continue
        buf_mb, chunk_mb = best_configs[backend]
        xs = []
        ys = []
        for nprocs in all_procs:
            val = lookup.get((backend, buf_mb, chunk_mb, nprocs))
            if val is not None:
                xs.append(nprocs)
                ys.append(val)
        if xs:
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]
            ax.plot(
                xs, ys,
                marker=marker,
                color=color,
                label=f"{backend} (buf={buf_mb}MB, chunk={chunk_mb}MB)",
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("num_processes")
    ax.set_ylabel("Aggregate Throughput (GB/s)")
    ax.set_title("Throughput vs Concurrency (best config per backend)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if all_procs:
        ax.set_xticks(all_procs)

    fname = f"lineplot_throughput_vs_procs.{fmt}"
    path = os.path.join(output_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150, format=fmt)
    plt.close(fig)
    print(f"Chart saved: {path}")
    return path


def generate_barplot(
    summaries: List[Dict[str, Any]],
    output_dir: str,
    fmt: str = "png",
) -> Optional[str]:
    """Generate bar chart comparing best config of each backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"ERROR: Missing required dependency: {e}")
        print("Please install: pip install matplotlib")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    backends = sorted({s["backend"] for s in summaries})

    if len(backends) <= 1:
        print("Skipping bar plot: need at least 2 backends for comparison")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))
    bar_labels = []
    bar_values = []
    bar_configs = []

    for backend in backends:
        backend_summaries = [
            s for s in summaries if s["backend"] == backend
        ]
        if backend_summaries:
            best = max(
                backend_summaries,
                key=lambda s: s["throughput_gbps_median"],
            )
            bar_labels.append(backend)
            bar_values.append(best["throughput_gbps_median"])
            bar_configs.append(
                f"buf={best['buffer_size_mb']}MB\n"
                f"chunk={best['chunk_size_mb']}MB\n"
                f"p={best['num_processes']}"
            )

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
    bars = ax.bar(
        bar_labels,
        bar_values,
        color=colors[: len(bar_labels)],
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar, val, cfg in zip(bars, bar_values, bar_configs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar_values) * 0.02,
            f"{val:.2f} GB/s\n{cfg}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title("Backend Comparison (best config each)")
    ax.set_ylim(0, max(bar_values) * 1.3 if bar_values else 1)

    fname = f"barplot_backend_comparison.{fmt}"
    path = os.path.join(output_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150, format=fmt)
    plt.close(fig)
    print(f"Chart saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Data Interpretation Report
# ---------------------------------------------------------------------------

def generate_interpretation_report(
    summaries: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """Generate a Markdown report interpreting the benchmark data."""
    os.makedirs(output_dir, exist_ok=True)

    # Group by backend
    by_backend: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        by_backend[s["backend"]].append(s)

    # Find best config per backend
    best_per_backend: Dict[str, Dict[str, Any]] = {}
    for backend, rows in by_backend.items():
        best = max(rows, key=lambda r: r["throughput_gbps_median"])
        best_per_backend[backend] = best

    # Find overall best
    overall_best = max(
        best_per_backend.items(),
        key=lambda kv: kv[1]["throughput_gbps_median"],
    )

    # Calculate speedups
    sorted_backends = sorted(
        best_per_backend.items(),
        key=lambda kv: kv[1]["throughput_gbps_median"],
    )
    baseline_tp = sorted_backends[0][1]["throughput_gbps_median"]

    # Analyze scalability
    scalability_analysis = {}
    for backend in by_backend.keys():
        backend_data = [s for s in summaries if s["backend"] == backend]
        procs_tp = {}
        for s in backend_data:
            key = (s["buffer_size_mb"], s["chunk_size_mb"])
            if key not in procs_tp:
                procs_tp[key] = {}
            procs_tp[key][s["num_processes"]] = s["throughput_gbps_median"]

        # Find best config's scalability
        best_key = (best_per_backend[backend]["buffer_size_mb"],
                    best_per_backend[backend]["chunk_size_mb"])
        if best_key in procs_tp:
            proc_data = procs_tp[best_key]
            sorted_procs = sorted(proc_data.items())
            if len(sorted_procs) >= 2:
                first_proc, first_tp = sorted_procs[0]
                last_proc, last_tp = sorted_procs[-1]
                ideal_scaling = last_proc / first_proc
                actual_scaling = last_tp / first_tp if first_tp > 0 else 0
                efficiency = (actual_scaling / ideal_scaling * 100) if ideal_scaling > 0 else 0
                scalability_analysis[backend] = {
                    "first_procs": first_proc,
                    "first_tp": first_tp,
                    "last_procs": last_proc,
                    "last_tp": last_tp,
                    "ideal_scaling": ideal_scaling,
                    "actual_scaling": actual_scaling,
                    "efficiency": efficiency,
                }

    # Generate report
    lines = [
        "# Benchmark Data Interpretation Report",
        "",
        f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Configurations Tested**: {len(summaries)}",
        "",
        "## Executive Summary",
        "",
        f"- **Best Overall Backend**: `{overall_best[0]}` with {overall_best[1]['throughput_gbps_median']:.3f} GB/s",
        f"  - Optimal config: buffer={overall_best[1]['buffer_size_mb']}MB, chunk={overall_best[1]['chunk_size_mb']}MB, procs={overall_best[1]['num_processes']}",
        "",
        "### Backend Performance Ranking",
        "",
        "| Rank | Backend | Best Throughput (GB/s) | Config | Speedup vs Baseline |",
        "|------|---------|------------------------|--------|---------------------|",
    ]

    for rank, (backend, best) in enumerate(reversed(sorted_backends), 1):
        config = f"buf={best['buffer_size_mb']}MB, chunk={best['chunk_size_mb']}MB, p={best['num_processes']}"
        speedup = best["throughput_gbps_median"] / baseline_tp if baseline_tp > 0 else 0
        lines.append(
            f"| {rank} | {backend} | {best['throughput_gbps_median']:.3f} | {config} | {speedup:.2f}x |"
        )

    lines.extend([
        "",
        "## Detailed Analysis by Backend",
        "",
    ])

    for backend in sorted(by_backend.keys()):
        rows = by_backend[backend]
        best = best_per_backend[backend]

        lines.extend([
            f"### {backend.upper()} Backend",
            "",
            f"**Best Configuration**:",
            f"- Buffer Size: {best['buffer_size_mb']} MB",
            f"- Chunk Size: {best['chunk_size_mb']} MB",
            f"- Num Processes: {best['num_processes']}",
            f"- Median Throughput: **{best['throughput_gbps_median']:.3f} GB/s**",
            f"- Std Dev: {best['throughput_gbps_std']:.3f} GB/s",
            f"- Min/Max: {best['throughput_gbps_min']:.3f} / {best['throughput_gbps_max']:.3f} GB/s",
            "",
        ])

        # Scalability analysis
        if backend in scalability_analysis:
            sa = scalability_analysis[backend]
            lines.extend([
                "**Scalability Analysis**:",
                f"- From {sa['first_procs']} to {sa['last_procs']} processes:",
                f"  - Throughput: {sa['first_tp']:.3f} -> {sa['last_tp']:.3f} GB/s",
                f"  - Actual scaling: {sa['actual_scaling']:.2f}x (ideal: {sa['ideal_scaling']:.2f}x)",
                f"  - Parallel efficiency: **{sa['efficiency']:.1f}%**",
                "",
            ])

            if sa["efficiency"] < 50:
                lines.append(
                    "⚠️ **Warning**: Low parallel efficiency detected. "
                    "Consider investigating potential bottlenecks (I/O contention, lock contention, etc.)."
                )
            elif sa["efficiency"] < 80:
                lines.append(
                    "ℹ️ **Note**: Moderate parallel efficiency. "
                    "Some overhead exists but scaling is reasonable."
                )
            else:
                lines.append(
                    "✅ **Good**: High parallel efficiency. "
                    "The backend scales well with increased concurrency."
                )
            lines.append("")

        # Parameter sensitivity
        lines.extend([
            "**Parameter Sensitivity**:",
            "",
        ])

        # Buffer size impact
        buf_sizes = sorted({r["buffer_size_mb"] for r in rows})
        if len(buf_sizes) > 1:
            buf_impact = []
            for buf in buf_sizes:
                buf_rows = [r for r in rows if r["buffer_size_mb"] == buf]
                avg_tp = sum(r["throughput_gbps_median"] for r in buf_rows) / len(buf_rows)
                buf_impact.append((buf, avg_tp))
            best_buf = max(buf_impact, key=lambda x: x[1])
            lines.append(f"- Buffer size impact: Best at {best_buf[0]}MB (avg {best_buf[1]:.3f} GB/s)")

        # Chunk size impact
        chunk_sizes = sorted({r["chunk_size_mb"] for r in rows})
        if len(chunk_sizes) > 1:
            chunk_impact = []
            for chunk in chunk_sizes:
                chunk_rows = [r for r in rows if r["chunk_size_mb"] == chunk]
                avg_tp = sum(r["throughput_gbps_median"] for r in chunk_rows) / len(chunk_rows)
                chunk_impact.append((chunk, avg_tp))
            best_chunk = max(chunk_impact, key=lambda x: x[1])
            lines.append(f"- Chunk size impact: Best at {best_chunk[0]}MB (avg {best_chunk[1]:.3f} GB/s)")

        lines.append("")

    lines.extend([
        "## Recommendations",
        "",
        "Based on the benchmark results:",
        "",
        f"1. **Use `{overall_best[0]}` backend** for best overall performance "
        f"({overall_best[1]['throughput_gbps_median']:.3f} GB/s).",
        "",
        f"2. **Optimal configuration**: buffer={overall_best[1]['buffer_size_mb']}MB, "
        f"chunk={overall_best[1]['chunk_size_mb']}MB, "
        f"num_processes={overall_best[1]['num_processes']}.",
        "",
    ])

    # Add scalability recommendation
    best_scalable = None
    best_efficiency = 0
    for backend, sa in scalability_analysis.items():
        if sa["efficiency"] > best_efficiency:
            best_efficiency = sa["efficiency"]
            best_scalable = backend

    if best_scalable and best_efficiency > 60:
        lines.append(
            f"3. **Best scalability**: `{best_scalable}` backend shows {best_efficiency:.1f}% "
            f"parallel efficiency, making it suitable for high-concurrency scenarios."
        )
    else:
        lines.append(
            "3. **Scalability concern**: All backends show limited scalability. "
            "Consider investigating I/O bottlenecks or reducing lock contention."
        )

    lines.extend([
        "",
        "## Raw Data Summary",
        "",
        f"- Total successful configurations: {len(summaries)}",
        f"- Backends tested: {', '.join(sorted(by_backend.keys()))}",
        f"- Process counts: {', '.join(map(str, sorted({s['num_processes'] for s in summaries})))}",
        f"- Buffer sizes: {', '.join(map(str, sorted({s['buffer_size_mb'] for s in summaries})))} MB",
        f"- Chunk sizes: {', '.join(map(str, sorted({s['chunk_size_mb'] for s in summaries})))} MB",
        "",
        "---",
        "*Generated by generate_charts_from_csv.py*",
    ])

    report_path = os.path.join(output_dir, "data_interpretation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Console Report (from benchmark_report.py)
# ---------------------------------------------------------------------------

def print_console_report(summaries: List[Dict[str, Any]]) -> None:
    """Print a formatted console report with per-backend tables."""
    if not summaries:
        print("No benchmark results to report.")
        return

    # Group by backend
    by_backend: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        by_backend[s["backend"]].append(s)

    # Find best config per backend
    best_per_backend: Dict[str, Dict[str, Any]] = {}
    for backend, rows in by_backend.items():
        best = max(rows, key=lambda r: r["throughput_gbps_median"])
        best_per_backend[backend] = best

    # Per-backend tables
    for backend in sorted(by_backend.keys()):
        rows = sorted(
            by_backend[backend],
            key=lambda r: (
                r["buffer_size_mb"],
                r["chunk_size_mb"],
                r["num_processes"],
            ),
        )
        best = best_per_backend[backend]

        print(f"\n{'=' * 90}")
        print(f"Backend: {backend}")
        print(f"{'=' * 90}")
        header = (
            f"{'buffer(MB)':<12} {'chunk(MB)':<12} {'procs':<8} "
            f"{'throughput(GB/s)':<20} {'wall_time(s)':<14} {'std':<10}"
        )
        print(header)
        print(f"{'-' * 90}")

        for r in rows:
            is_best = (
                r["buffer_size_mb"] == best["buffer_size_mb"]
                and r["chunk_size_mb"] == best["chunk_size_mb"]
                and r["num_processes"] == best["num_processes"]
            )
            marker = " ***" if is_best else ""
            print(
                f"{r['buffer_size_mb']:<12} {r['chunk_size_mb']:<12} "
                f"{r['num_processes']:<8} "
                f"{r['throughput_gbps_median']:<20.3f} "
                f"{r['wall_time_median']:<14.3f} "
                f"{r['throughput_gbps_std']:<10.3f}{marker}"
            )

        print(
            f"\n  Best: buffer={best['buffer_size_mb']}MB, "
            f"chunk={best['chunk_size_mb']}MB, "
            f"procs={best['num_processes']} -> "
            f"{best['throughput_gbps_median']:.3f} GB/s"
        )

    # Cross-backend comparison
    if len(best_per_backend) > 1:
        print(f"\n{'=' * 90}")
        print("Cross-Backend Comparison (best params per backend)")
        print(f"{'=' * 90}")
        header = (
            f"{'Backend':<10} {'Best Config':<30} "
            f"{'Throughput(GB/s)':<20} {'Speedup':<10}"
        )
        print(header)
        print(f"{'-' * 90}")

        sorted_backends = sorted(
            best_per_backend.items(),
            key=lambda kv: kv[1]["throughput_gbps_median"],
        )
        baseline_tp = sorted_backends[0][1]["throughput_gbps_median"]

        for backend, best in sorted_backends:
            config = (
                f"buf={best['buffer_size_mb']}MB "
                f"chunk={best['chunk_size_mb']}MB "
                f"p={best['num_processes']}"
            )
            speedup = (
                best["throughput_gbps_median"] / baseline_tp
                if baseline_tp > 0
                else 0.0
            )
            print(
                f"{backend:<10} {config:<30} "
                f"{best['throughput_gbps_median']:<20.3f} "
                f"{speedup:<10.2f}x"
            )

    print(f"\n{'=' * 90}\n")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate charts from benchmark CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all charts
  python generate_charts_from_csv.py --csv-dir ./benchmark_results

  # Specify output directory
  python generate_charts_from_csv.py --csv-dir ./benchmark_results --output-dir ./charts

  # Generate only heatmaps
  python generate_charts_from_csv.py --csv-dir ./benchmark_results --charts heatmap

  # Generate report with interpretation
  python generate_charts_from_csv.py --csv-dir ./benchmark_results --generate-report

  # Generate PDF format
  python generate_charts_from_csv.py --csv-dir ./benchmark_results --format pdf
        """,
    )

    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing benchmark_summary.csv (and optionally benchmark_raw.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="./charts_output",
        help="Output directory for generated charts (default: ./charts_output)",
    )
    parser.add_argument(
        "--charts",
        default="all",
        help="Comma-separated list of charts to generate: all,heatmap,lineplot,barplot (default: all)",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate data interpretation report (Markdown)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--console-report",
        action="store_true",
        default=True,
        help="Print console report (default: True)",
    )

    args = parser.parse_args()

    # Check dependencies early
    try:
        import matplotlib
        import numpy
    except ImportError as e:
        print(f"ERROR: Missing required dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install matplotlib numpy")
        print("\nOr if using uv:")
        print("  uv pip install matplotlib numpy")
        sys.exit(1)

    # Load CSV files
    summary_csv = os.path.join(args.csv_dir, "benchmark_summary.csv")
    raw_csv = os.path.join(args.csv_dir, "benchmark_raw.csv")

    if not os.path.exists(summary_csv):
        print(f"ERROR: Summary CSV not found: {summary_csv}")
        print("Please ensure benchmark_summary.csv exists in the specified directory.")
        sys.exit(1)

    print(f"Loading data from: {summary_csv}")
    summaries = load_summary_csv(summary_csv)

    # Optionally load raw data
    raw_data = None
    if os.path.exists(raw_csv):
        print(f"Loading raw data from: {raw_csv}")
        raw_data = load_raw_csv(raw_csv)

    print(f"Loaded {len(summaries)} summary records")

    # Print console report
    if args.console_report:
        print_console_report(summaries)

    # Parse chart types
    chart_types = [c.strip().lower() for c in args.charts.split(",")]
    generate_all = "all" in chart_types

    os.makedirs(args.output_dir, exist_ok=True)
    generated_files = []

    # Generate charts
    if generate_all or "heatmap" in chart_types:
        print("\nGenerating heatmaps...")
        files = generate_heatmaps(summaries, args.output_dir, args.format)
        generated_files.extend(files)

    if generate_all or "lineplot" in chart_types or "line" in chart_types:
        print("\nGenerating line plot...")
        path = generate_lineplot(summaries, args.output_dir, args.format)
        if path:
            generated_files.append(path)

    if generate_all or "barplot" in chart_types or "bar" in chart_types:
        print("\nGenerating bar chart...")
        path = generate_barplot(summaries, args.output_dir, args.format)
        if path:
            generated_files.append(path)

    # Generate report
    if args.generate_report:
        print("\nGenerating interpretation report...")
        report_path = generate_interpretation_report(summaries, args.output_dir)
        generated_files.append(report_path)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Generated {len(generated_files)} files in: {os.path.abspath(args.output_dir)}")
    print(f"{'=' * 60}")
    for f in generated_files:
        print(f"  - {os.path.basename(f)}")


if __name__ == "__main__":
    main()
