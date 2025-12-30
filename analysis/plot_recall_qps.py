# 2025/12/30 Zhencan Peng: add a one-shot script to plot Recall vs QPS per range bucket.

"""
Plot Recall–QPS Curves (7 Range Buckets) from Benchmark Logs.

This script scans the repository `logs/` folder for log files produced by
`scripts/run_query_index.sh` (which runs `apps/query_index.cc`). It extracts
per-range (Recall, QPS) points and draws a single figure with 7 subplots:

- X-axis: Recall
- Y-axis: QPS
- One subplot per range bucket (1%, 2%, 4%, 8%, 16%, 32%, 64%)

Typical log line format (from `apps/query_index.cc`):

  [search_ef 64] [Range ratio 1.0000%] Recall=0.8123 Latency=0.4321 ms QPS=2313.1234 ...

Usage
-----
From the repo root:

  python3 analysis/plot_recall_qps.py

Plot a specific log file (recommended for a single run):

  python3 analysis/plot_recall_qps.py \
    --log_file logs/deep/search/deep_N1000000_k16_efc150_efm300_alpha1.0_20251228_230148.log \
    --out analysis/recall_qps.png

Or with a custom log folder and output path:

  python3 analysis/plot_recall_qps.py \
    --log_root /path/to/DynamicSegmentGraph/logs \
    --out /path/to/out.png

Complexity
----------
Let F be the number of log files and L be the total number of lines across
those files. Parsing is O(L) time and O(R) space for R extracted records.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

RECALL_X_MIN: float = 0.7
RECALL_X_MAX: float = 1.0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot Recall (x) vs QPS (y) for 7 range buckets from query_index logs."
    )
    p.add_argument(
        "--log_file",
        type=Path,
        action="append",
        default=None,
        help=(
            "Explicit log file(s) to plot. Can be passed multiple times. "
            "If set, file discovery via --log_root/--pattern/--include/--exclude is skipped."
        ),
    )
    p.add_argument(
        "--log_root",
        type=Path,
        default=None,
        help="Root directory containing logs/ (default: <repo_root>/logs).",
    )
    p.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern(s) under log_root (default: '**/*.log'). Can be passed multiple times.",
    )
    p.add_argument(
        "--include",
        type=str,
        default=None,
        help="Only include log files whose relative path matches this regex.",
    )
    p.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Exclude log files whose relative path matches this regex.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <repo_root>/analysis/recall_qps.png).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Recall vs QPS (7 ranges)",
        help="Figure title.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI for raster formats (default: 200).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively (in addition to saving).",
    )
    p.add_argument(
        "--legend",
        action="store_true",
        help="Show a global legend (auto-disabled if too many curves).",
    )
    p.add_argument(
        "--legend_max",
        type=int,
        default=12,
        help="Max curves to show in legend before auto-disabling (default: 12).",
    )
    return p


def _repo_root_from_this_file() -> Path:
    # analysis/plot_recall_qps.py -> repo root is parent of analysis/
    return Path(__file__).resolve().parents[1]


def _compile_optional_regex(expr: Optional[str]) -> Optional[re.Pattern[str]]:
    if expr is None:
        return None
    return re.compile(expr)


def _filter_paths(
    paths: Sequence[Path],
    base: Path,
    include: Optional[re.Pattern[str]],
    exclude: Optional[re.Pattern[str]],
) -> List[Path]:
    kept: List[Path] = []
    for p in paths:
        rel = str(p.relative_to(base))
        if include is not None and include.search(rel) is None:
            continue
        if exclude is not None and exclude.search(rel) is not None:
            continue
        kept.append(p)
    return kept


def main() -> int:
    args = _build_argparser().parse_args()

    repo_root = _repo_root_from_this_file()
    log_root: Path = args.log_root if args.log_root is not None else (repo_root / "logs")
    out_path: Path = args.out if args.out is not None else (repo_root / "analysis" / "recall_qps.png")
    patterns: List[str] = args.pattern if args.pattern is not None else ["**/*.log"]

    include_re = _compile_optional_regex(args.include)
    exclude_re = _compile_optional_regex(args.exclude)

    try:
        from analysis.log_record_extractor import RecordExtractor
    except ImportError:
        # Allow running as `python3 analysis/plot_recall_qps.py` from repo root
        # without installing the package.
        from log_record_extractor import RecordExtractor  # type: ignore

    extractor = RecordExtractor()
    if args.log_file is not None:
        log_files = [Path(p) for p in args.log_file]
        missing = [str(p) for p in log_files if not p.is_file()]
        if missing:
            print("[plot_recall_qps] These --log_file paths do not exist:")
            for p in missing:
                print(f"  - {p}")
            return 1
        # For labels, fall back to filenames when explicit paths are provided.
        label_base = None
    else:
        log_files = extractor.find_log_files(log_root, patterns=patterns)
        log_files = _filter_paths(log_files, log_root, include_re, exclude_re)
        label_base = log_root

    if not log_files:
        print(f"[plot_recall_qps] No log files found under: {log_root}")
        print(f"[plot_recall_qps] Patterns: {patterns}")
        return 1

    # Import matplotlib only when we know we have something to plot.
    try:
        warnings.filterwarnings(
            "ignore",
            message=r"Unable to import Axes3D\..*",
            category=UserWarning,
        )
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("[plot_recall_qps] matplotlib is required for plotting.")
        print(f"[plot_recall_qps] ImportError: {exc}")
        return 1

    range_ratios = extractor.range_ratios

    # Load all logs first.
    per_log_grouped: Dict[str, Dict[float, List[Tuple[float, float]]]] = {}
    for log_path in log_files:
        if label_base is not None:
            label = str(log_path.relative_to(label_base))
        else:
            label = log_path.name
        grouped = extractor.extract_file_grouped(log_path)
        points_by_ratio: Dict[float, List[Tuple[float, float]]] = {}
        for ratio in range_ratios:
            recs = grouped.get(ratio, [])
            pts = [(r.recall, r.qps) for r in recs]
            pts.sort(key=lambda x: x[0])  # sort by recall
            points_by_ratio[ratio] = pts
        per_log_grouped[label] = points_by_ratio

    # Create a 2x4 grid (7 used, 1 hidden).
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
    axes_flat = axes.flatten()

    # Pick a stable color per log file.
    cmap = plt.get_cmap("tab20")
    labels = sorted(per_log_grouped.keys())
    label_to_color = {lab: cmap(i % cmap.N) for i, lab in enumerate(labels)}

    label_to_handle = {}

    for idx, ratio in enumerate(range_ratios):
        ax = axes_flat[idx]
        pct = int(round(ratio * 100))
        ax.set_title(f"Range {pct}%")
        ax.set_xlabel("Recall")
        ax.set_ylabel("QPS")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        x_min = RECALL_X_MIN
        x_max = RECALL_X_MAX
        ax.set_xlim(RECALL_X_MIN, RECALL_X_MAX)

        y_max: Optional[float] = None
        for lab in labels:
            pts = per_log_grouped[lab][ratio]
            if not pts:
                continue
            # Filter to the visible recall window so scaling reflects what you see.
            pts = [(x, y) for x, y in pts if x_min <= x <= x_max]
            if not pts:
                continue
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            if ys:
                local_max = max(ys)
                y_max = local_max if y_max is None else max(y_max, local_max)
            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                color=label_to_color[lab],
                label=lab,
            )
            if lab not in label_to_handle:
                label_to_handle[lab] = line

        # Important: do not lock y-limits before plotting; otherwise QPS curves can
        # be clipped (Matplotlib would keep the default [0, 1] range).
        if y_max is not None:
            ax.set_ylim(0.0, y_max * 1.05)
        else:
            ax.set_ylim(0.0, 1.0)

    # Hide the unused 8th subplot.
    if len(axes_flat) > len(range_ratios):
        axes_flat[len(range_ratios)].axis("off")

    fig.suptitle(args.title)

    want_legend = bool(args.legend)
    if want_legend and len(label_to_handle) > args.legend_max:
        print(
            f"[plot_recall_qps] Too many curves ({len(label_to_handle)}) for legend; "
            f"auto-disabling legend (legend_max={args.legend_max})."
        )
        want_legend = False

    if want_legend and label_to_handle:
        fig.legend(
            list(label_to_handle.values()),
            list(label_to_handle.keys()),
            loc="lower center",
            ncol=min(4, max(1, len(label_to_handle))),
            fontsize="small",
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.18)
    else:
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot_recall_qps] Saved figure: {out_path}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


