#!/usr/bin/env python3
"""Benchmark result aggregation, console reporting, CSV export, and chart
generation.

This module processes raw per-iteration result dicts produced by
``benchmark_worker.py`` and produces:

1. Console tables with best-parameter highlighting.
2. CSV files (raw + summary).
3. Matplotlib charts (heatmaps, line plots, bar charts).
"""

import csv
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. Data aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    raw_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate per-iteration results into per-configuration summaries.

    Groups by ``(backend, buffer_size, chunk_size, num_processes)`` and
    computes median / mean / min / max / std of throughput and wall time.

    Only successful results (``success == True``) are included.

    Returns:
        A list of summary dicts, one per unique parameter combination.
    """
    # Group results by configuration key
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in raw_results:
        if not r.get("success", False):
            continue
        key = (
            r["backend"],
            r["buffer_size"],
            r["chunk_size"],
            r["num_processes"],
        )
        groups[key].append(r)

    summaries: List[Dict[str, Any]] = []
    for (backend, buf, chunk, nprocs), records in sorted(groups.items()):
        # Aggregate across all ranks and iterations for this config.
        # Use the *aggregate* throughput per iteration: sum bytes across ranks
        # divided by the round wall time.
        #
        # First, group records by iteration to compute per-iteration aggregate.
        iter_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            iter_groups[rec["iteration"]].append(rec)

        iter_throughputs_gbps: List[float] = []
        iter_wall_times: List[float] = []
        for _iter_id, iter_recs in sorted(iter_groups.items()):
            total_bytes = sum(r["total_bytes_read"] for r in iter_recs)
            # Use round_wall_time if available (set by run_benchmark_round),
            # otherwise fall back to max wall_time across ranks.
            round_wall = iter_recs[0].get(
                "round_wall_time", max(r["wall_time"] for r in iter_recs)
            )
            if round_wall > 0:
                agg_gbps = (total_bytes / 1024 / 1024 / 1024) / round_wall
            else:
                agg_gbps = 0.0
            iter_throughputs_gbps.append(agg_gbps)
            iter_wall_times.append(round_wall)

        if not iter_throughputs_gbps:
            continue

        tp = iter_throughputs_gbps
        wt = iter_wall_times

        summaries.append(
            {
                "backend": backend,
                "buffer_size_mb": buf // (1024 * 1024),
                "chunk_size_mb": chunk // (1024 * 1024),
                "num_processes": nprocs,
                "throughput_gbps_median": statistics.median(tp),
                "throughput_gbps_mean": statistics.mean(tp),
                "throughput_gbps_min": min(tp),
                "throughput_gbps_max": max(tp),
                "throughput_gbps_std": statistics.stdev(tp) if len(tp) > 1 else 0.0,
                "wall_time_median": statistics.median(wt),
                "wall_time_mean": statistics.mean(wt),
                "num_iterations": len(tp),
            }
        )

    return summaries


# ---------------------------------------------------------------------------
# 2. Console report
# ---------------------------------------------------------------------------


def print_console_report(summaries: List[Dict[str, Any]]) -> None:
    """Print a formatted console report with per-backend tables and a
    cross-backend comparison."""

    if not summaries:
        print("No benchmark results to report.")
        return

    # Group by backend
    by_backend: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        by_backend[s["backend"]].append(s)

    # Find best config per backend (highest median throughput)
    best_per_backend: Dict[str, Dict[str, Any]] = {}
    for backend, rows in by_backend.items():
        best = max(rows, key=lambda r: r["throughput_gbps_median"])
        best_per_backend[backend] = best

    # ---- Per-backend tables ------------------------------------------------
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

    # ---- Cross-backend comparison ------------------------------------------
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

        # Use the slowest backend as the baseline
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
                best["throughput_gbps_median"] / baseline_tp if baseline_tp > 0 else 0.0
            )
            print(
                f"{backend:<10} {config:<30} "
                f"{best['throughput_gbps_median']:<20.3f} "
                f"{speedup:<10.2f}x"
            )

    print(f"\n{'=' * 90}\n")


# ---------------------------------------------------------------------------
# 3. CSV export
# ---------------------------------------------------------------------------


def export_csv(
    raw_results: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    output_dir: str,
) -> Tuple[str, str]:
    """Write raw and summary CSV files to *output_dir*.

    Returns:
        ``(raw_csv_path, summary_csv_path)``
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Raw CSV -----------------------------------------------------------
    raw_path = os.path.join(output_dir, "benchmark_raw.csv")
    raw_fields = [
        "backend",
        "buffer_size",
        "chunk_size",
        "num_processes",
        "rank",
        "iteration",
        "num_files",
        "total_bytes_read",
        "read_time",
        "wall_time",
        "throughput_mbps",
        "throughput_gbps",
        "round_wall_time",
        "success",
        "error",
    ]
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        for r in raw_results:
            writer.writerow(r)

    # ---- Summary CSV -------------------------------------------------------
    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    summary_fields = [
        "backend",
        "buffer_size_mb",
        "chunk_size_mb",
        "num_processes",
        "throughput_gbps_median",
        "throughput_gbps_mean",
        "throughput_gbps_min",
        "throughput_gbps_max",
        "throughput_gbps_std",
        "wall_time_median",
        "wall_time_mean",
        "num_iterations",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print(f"CSV exported: {raw_path}")
    print(f"CSV exported: {summary_path}")
    return raw_path, summary_path


# ---------------------------------------------------------------------------
# 4. Matplotlib chart generation
# ---------------------------------------------------------------------------


def generate_charts(
    summaries: List[Dict[str, Any]],
    output_dir: str,
) -> List[str]:
    """Generate benchmark visualisation charts and save them to *output_dir*.

    Charts produced:
    - Heatmap per (backend, num_processes): buffer_size vs chunk_size.
    - Line plot: throughput vs num_processes per backend (best buf/chunk).
    - Bar chart: cross-backend comparison at best config.

    Returns:
        List of generated file paths.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(
            "WARNING: matplotlib/numpy not available -- skipping chart " "generation."
        )
        print(f"\nMissing dependency: {e}")
        print("\nTo generate charts later, you can:")
        print("  1. Install dependencies: pip install matplotlib numpy")
        print("  2. Run the standalone chart generator:")
        print(f"     python hack/generate_charts_from_csv.py --csv-dir {output_dir}")
        print("\nThis will regenerate all charts from the existing CSV files.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    generated: List[str] = []

    # Collect unique dimensions
    backends = sorted({s["backend"] for s in summaries})
    all_procs = sorted({s["num_processes"] for s in summaries})
    all_buf = sorted({s["buffer_size_mb"] for s in summaries})
    all_chunk = sorted({s["chunk_size_mb"] for s in summaries})

    # Build a lookup: (backend, buf, chunk, procs) -> median throughput
    lookup: Dict[Tuple, float] = {}
    for s in summaries:
        key = (
            s["backend"],
            s["buffer_size_mb"],
            s["chunk_size_mb"],
            s["num_processes"],
        )
        lookup[key] = s["throughput_gbps_median"]

    # ---- Heatmaps ----------------------------------------------------------
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
            ax.set_title(f"Throughput (GB/s) -- backend={backend}, procs={nprocs}")
            fig.colorbar(im, ax=ax, label="GB/s")

            # Annotate cells with values
            for i in range(len(all_buf)):
                for j in range(len(all_chunk)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color="black" if val < np.nanmax(matrix) * 0.7 else "white",
                            fontsize=8,
                        )

            fname = f"heatmap_{backend}_{nprocs}procs.png"
            path = os.path.join(output_dir, fname)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            generated.append(path)
            print(f"Chart saved: {path}")

    # ---- Line plot: throughput vs num_processes ----------------------------
    # For each backend, use its best (buffer, chunk) config
    best_configs: Dict[str, Tuple[int, int]] = {}
    for backend in backends:
        backend_summaries = [s for s in summaries if s["backend"] == backend]
        if backend_summaries:
            best = max(backend_summaries, key=lambda s: s["throughput_gbps_median"])
            best_configs[backend] = (
                best["buffer_size_mb"],
                best["chunk_size_mb"],
            )

    if best_configs and len(all_procs) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        markers = ["o", "s", "^", "D", "v"]
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
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
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

        path = os.path.join(output_dir, "lineplot_throughput_vs_procs.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)
        print(f"Chart saved: {path}")

    # ---- Bar chart: cross-backend comparison at best config ----------------
    if len(backends) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        bar_labels = []
        bar_values = []
        bar_configs = []
        for backend in backends:
            backend_summaries = [s for s in summaries if s["backend"] == backend]
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

        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
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

        path = os.path.join(output_dir, "barplot_backend_comparison.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)
        print(f"Chart saved: {path}")

    return generated


# ---------------------------------------------------------------------------
# 5. headers_batch aggregation, reporting, and CSV export
# ---------------------------------------------------------------------------


def aggregate_headers_batch_results(
    raw_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate per-iteration headers_batch results into per-configuration summaries.

    Groups by ``(backend, batch_size, num_threads, num_processes)`` and
    computes median / mean / min / max / std of ``headers_per_second`` and
    ``wall_time``.

    Only successful results (``success == True``) are included.

    Returns:
        A list of summary dicts, one per unique parameter combination.
    """
    import statistics
    from collections import defaultdict

    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in raw_results:
        if not r.get("success", False):
            continue
        key = (
            r["backend"],
            r["batch_size"],
            r["num_threads"],
            r["num_processes"],
        )
        groups[key].append(r)

    summaries: List[Dict[str, Any]] = []
    for (backend, batch_size, num_threads, nprocs), records in sorted(groups.items()):
        # Group by iteration to compute per-iteration aggregate files/s
        iter_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            iter_groups[rec["iteration"]].append(rec)

        iter_fps: List[float] = []
        iter_wall_times: List[float] = []
        for _iter_id, iter_recs in sorted(iter_groups.items()):
            total_files = sum(r["total_files_read"] for r in iter_recs)
            round_wall = iter_recs[0].get(
                "round_wall_time", max(r["wall_time"] for r in iter_recs)
            )
            fps = total_files / round_wall if round_wall > 0 else 0.0
            iter_fps.append(fps)
            iter_wall_times.append(round_wall)

        if not iter_fps:
            continue

        summaries.append(
            {
                "backend": backend,
                "batch_size": batch_size,
                "num_threads": num_threads,
                "num_processes": nprocs,
                "headers_per_second_median": statistics.median(iter_fps),
                "headers_per_second_mean": statistics.mean(iter_fps),
                "headers_per_second_min": min(iter_fps),
                "headers_per_second_max": max(iter_fps),
                "headers_per_second_std": (
                    statistics.stdev(iter_fps) if len(iter_fps) > 1 else 0.0
                ),
                "wall_time_median": statistics.median(iter_wall_times),
                "wall_time_mean": statistics.mean(iter_wall_times),
                "num_iterations": len(iter_fps),
            }
        )

    return summaries


def print_headers_batch_report(summaries: List[Dict[str, Any]]) -> None:
    """Print a formatted console report for headers_batch benchmark results."""

    if not summaries:
        print("No headers_batch benchmark results to report.")
        return

    by_backend: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        by_backend[s["backend"]].append(s)

    best_per_backend: Dict[str, Dict[str, Any]] = {}
    for backend, rows in by_backend.items():
        best = max(rows, key=lambda r: r["headers_per_second_median"])
        best_per_backend[backend] = best

    for backend in sorted(by_backend.keys()):
        rows = sorted(
            by_backend[backend],
            key=lambda r: (r["batch_size"], r["num_threads"], r["num_processes"]),
        )
        best = best_per_backend[backend]

        print(f"\n{'=' * 90}")
        print(f"[headers_batch] Backend: {backend}")
        print(f"{'=' * 90}")
        header = (
            f"{'batch_size':<12} {'num_threads':<13} {'procs':<8} "
            f"{'files/s':<20} {'wall_time(s)':<14} {'std':<10}"
        )
        print(header)
        print(f"{'-' * 90}")

        for r in rows:
            is_best = (
                r["batch_size"] == best["batch_size"]
                and r["num_threads"] == best["num_threads"]
                and r["num_processes"] == best["num_processes"]
            )
            marker = " ***" if is_best else ""
            print(
                f"{r['batch_size']:<12} {r['num_threads']:<13} "
                f"{r['num_processes']:<8} "
                f"{r['headers_per_second_median']:<20.1f} "
                f"{r['wall_time_median']:<14.3f} "
                f"{r['headers_per_second_std']:<10.1f}{marker}"
            )

        print(
            f"\n  Best: batch_size={best['batch_size']}, "
            f"num_threads={best['num_threads']}, "
            f"procs={best['num_processes']} -> "
            f"{best['headers_per_second_median']:.1f} files/s"
        )

    # Cross-backend comparison
    if len(best_per_backend) > 1:
        print(f"\n{'=' * 90}")
        print("[headers_batch] Cross-Backend Comparison (best params per backend)")
        print(f"{'=' * 90}")
        header = (
            f"{'Backend':<10} {'Best Config':<35} " f"{'files/s':<20} {'Speedup':<10}"
        )
        print(header)
        print(f"{'-' * 90}")

        sorted_backends = sorted(
            best_per_backend.items(),
            key=lambda kv: kv[1]["headers_per_second_median"],
        )
        baseline_fps = sorted_backends[0][1]["headers_per_second_median"]

        for backend, best in sorted_backends:
            config = (
                f"batch={best['batch_size']} "
                f"threads={best['num_threads']} "
                f"p={best['num_processes']}"
            )
            speedup = (
                best["headers_per_second_median"] / baseline_fps
                if baseline_fps > 0
                else 0.0
            )
            print(
                f"{backend:<10} {config:<35} "
                f"{best['headers_per_second_median']:<20.1f} "
                f"{speedup:<10.2f}x"
            )

    print(f"\n{'=' * 90}\n")


def export_headers_batch_csv(
    raw_results: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    output_dir: str,
) -> Tuple[str, str]:
    """Write headers_batch raw and summary CSV files to *output_dir*.

    Returns:
        ``(raw_csv_path, summary_csv_path)``
    """
    import csv as _csv

    os.makedirs(output_dir, exist_ok=True)

    # ---- Raw CSV -----------------------------------------------------------
    raw_path = os.path.join(output_dir, "benchmark_headers_batch_raw.csv")
    raw_fields = [
        "op",
        "backend",
        "buffer_size",
        "batch_size",
        "num_threads",
        "num_processes",
        "rank",
        "iteration",
        "num_files",
        "total_files_read",
        "headers_per_second",
        "wall_time",
        "round_wall_time",
        "success",
        "error",
    ]
    with open(raw_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        for r in raw_results:
            writer.writerow(r)

    # ---- Summary CSV -------------------------------------------------------
    summary_path = os.path.join(output_dir, "benchmark_headers_batch_summary.csv")
    summary_fields = [
        "backend",
        "batch_size",
        "num_threads",
        "num_processes",
        "headers_per_second_median",
        "headers_per_second_mean",
        "headers_per_second_min",
        "headers_per_second_max",
        "headers_per_second_std",
        "wall_time_median",
        "wall_time_mean",
        "num_iterations",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print(f"CSV exported: {raw_path}")
    print(f"CSV exported: {summary_path}")
    return raw_path, summary_path
