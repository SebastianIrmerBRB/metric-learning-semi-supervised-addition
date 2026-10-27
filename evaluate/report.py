"""Build everything at once and put an HTML page in front of it.

This is the entry point. It collects, tabulates, plots, stitches the filmstrips,
and writes an ``index.html`` that shows them in the order a reader actually wants
them: what ran, what moved, then the pictures.

The page is plain local HTML with relative links -- open it in a browser, no
server, no build step. Everything it shows also exists as a CSV or a PNG beside
it, so nothing is trapped inside the report.

Examples::

    python evaluate/report.py --root logs/phase_1 --out evaluate_out
    python evaluate/report.py --root final/phase_1/seeded_runs_cars --out ~/thesis/tracking
    xdg-open evaluate_out/index.html
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import common, figures, filmstrip, tables
from evaluate.collect import collect

PAGE_STYLE = """
:root { color-scheme: light dark; --fg:#1f2328; --muted:#6b7280; --bg:#ffffff;
        --card:#f6f7f9; --line:#e5e7eb; --good:#2f7d4f; --bad:#c1442e; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e6e6; --muted:#9aa0a6; --bg:#14161a; --card:#1c1f24; --line:#2a2f36; }
}
* { box-sizing: border-box; }
body { margin:0; padding:0 16px 64px; background:var(--bg); color:var(--fg);
       font:14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width:1180px; margin:0 auto; }
h1 { font-size:24px; margin:28px 0 4px; }
h2 { font-size:18px; margin:36px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }
h3 { font-size:14px; margin:20px 0 8px; color:var(--muted); font-weight:600;
     text-transform:uppercase; letter-spacing:.04em; }
p.lede { color:var(--muted); margin:0 0 18px; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:8px 0 18px; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; white-space:nowrap; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.good { color:var(--good); font-weight:600; }
.bad { color:var(--bad); font-weight:600; }
.scroll { overflow-x:auto; }
figure { margin:0 0 22px; background:var(--card); border:1px solid var(--line);
         border-radius:10px; padding:10px; }
figure img { width:100%; height:auto; display:block; border-radius:6px; }
figure.strip { overflow-x:auto; }
figure.strip img { width:auto; max-width:none; height:300px; }
figcaption { color:var(--muted); font-size:12px; margin-top:8px; }
.grid { display:grid; gap:18px; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); }
code { background:var(--card); padding:1px 5px; border-radius:4px; font-size:12px; }
ul.files { columns:2; font-size:13px; }
@media (max-width:640px) { .grid { grid-template-columns:1fr; } ul.files { columns:1; } }
"""


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    if isinstance(value, float):
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4f}"
    return html.escape(str(value))


def _frame_to_html(frame: pd.DataFrame, numeric: set[str] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(
        f'<th class="num">{html.escape(str(c))}</th>' if c in numeric
        else f"<th>{html.escape(str(c))}</th>"
        for c in frame.columns
    )
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(
            f'<td class="num">{_cell(row[c])}</td>' if c in numeric else f"<td>{_cell(row[c])}</td>"
            for c in frame.columns
        )
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _summary_section(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "<p class='lede'>No metrics were tracked.</p>"
    blocks = []
    for group, metrics in common.METRIC_GROUPS.items():
        present = [m for m in metrics if m in set(summary["metric"])]
        if not present:
            continue
        rows = []
        for metric in present:
            for _, row in summary[summary["metric"] == metric].iterrows():
                delta = row["delta"]
                better = common.HIGHER_IS_BETTER.get(metric)
                css = ""
                if better is not None and not pd.isna(delta) and abs(delta) > 1e-12:
                    improved = delta > 0 if better else delta < 0
                    css = "good" if improved else "bad"
                sign = "+" if (not pd.isna(delta) and delta > 0) else ""
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(row['run']))}</td>"
                    f"<td>{html.escape(str(row['metric_label']))}</td>"
                    f'<td class="num">{int(row["folds"])}</td>'
                    f'<td class="num">{_cell(row["baseline"])}</td>'
                    f'<td class="num">{_cell(row["final"])}</td>'
                    f'<td class="num {css}">{sign}{_cell(delta)}</td>'
                    "</tr>"
                )
        blocks.append(
            f"<h3>{html.escape(group)}</h3>"
            '<div class="scroll"><table><thead><tr><th>Run</th><th>Metric</th>'
            '<th class="num">Folds</th><th class="num">Baseline (epoch -1)</th>'
            '<th class="num">Final</th><th class="num">Delta</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
    return "".join(blocks)


def _figure_block(path: Path, out: Path, caption: str, strip: bool = False) -> str:
    relative = path.relative_to(out).as_posix()
    css = "strip" if strip else ""
    return (
        f'<figure class="{css}"><img src="{html.escape(relative)}" alt="{html.escape(caption)}" loading="lazy">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def build_report(root: Path, out: Path, all_folds: bool = False) -> Path:
    collected = collect(root, out)
    table_paths = tables.write_tables(collected, out)
    figure_paths = figures.write_figures(collected, out)
    strip_paths = filmstrip.write_filmstrips(collected, out, all_folds=all_folds)

    runs = collected["runs"]
    display_columns = [
        c for c in ("run", "fold", "dataset", "method", "loss", "seed", "variant",
                    "labeled_per_class", "epochs", "track_interval", "run_dir")
        if c in runs.columns
    ]

    named = {path.stem: path for path in figure_paths}
    highlights = []
    for stem, caption in (
        ("delta_baseline", "How far each run moved each metric from its epoch -1 baseline."),
        ("trajectory", "Structure against score. Hollow marker is epoch -1; the arrow is the direction of training."),
    ):
        if stem in named:
            highlights.append(_figure_block(named[stem], out, caption))

    overviews = [
        _figure_block(path, out, f"{path.stem.replace('overview_', '').capitalize()} metrics across training.")
        for path in figure_paths if path.stem.startswith("overview_")
    ]
    curves = [
        _figure_block(
            path, out,
            common.METRIC_LABELS.get(path.stem.replace("curve_", ""), path.stem)
            + " -- line is the fold mean, band the fold min/max.",
        )
        for path in figure_paths if path.stem.startswith("curve_")
    ]
    strips = [
        _figure_block(path, out, f"{path.stem} -- epoch -1 at the left, trained model at the right.", strip=True)
        for path in strip_paths
    ]

    data_files = sorted(
        p for p in (out / "data").glob("*.csv")
    ) + sorted(p for p in (out / "tables").glob("*") if p.is_file())
    file_items = "".join(
        f'<li><a href="{html.escape(p.relative_to(out).as_posix())}">'
        f"{html.escape(p.relative_to(out).as_posix())}</a></li>"
        for p in data_files
    )

    fold_count = len(collected["records"])
    run_count = runs["run_id"].nunique() if not runs.empty else 0
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Epoch tracking report</title><style>{PAGE_STYLE}</style></head>
<body><div class="wrap">
<h1>Epoch tracking report</h1>
<p class="lede">{fold_count} tracked fold(s) across {run_count} run(s) from
<code>{html.escape(str(root))}</code>, generated {datetime.now():%Y-%m-%d %H:%M}.
Every number here was measured off the selection path: nothing on this page was
read back into training, early stopping, or Optuna.</p>

<h2>What ran</h2>
{_frame_to_html(runs[display_columns], numeric={"fold", "seed", "labeled_per_class", "epochs", "track_interval"})}

<h2>What moved</h2>
<p class="lede">Baseline is epoch -1, before any optimizer step. A green delta is
an improvement; metrics without a natural direction (a degree, a norm, a
component count) are left uncoloured.</p>
{_summary_section(collected["summary"])}

<h2>Highlights</h2>
<div class="grid">{"".join(highlights) or "<p class='lede'>Not enough data for the summary figures.</p>"}</div>

<h2>Metric families</h2>
<div class="grid">{"".join(overviews) or "<p class='lede'>No group overviews.</p>"}</div>

<h2>Individual curves</h2>
<div class="grid">{"".join(curves) or "<p class='lede'>No curves.</p>"}</div>

<h2>Filmstrips</h2>
<p class="lede">The per-epoch pictures in order, so a representation can be watched
unfolding rather than opened one file at a time. Scroll sideways.</p>
{"".join(strips) or "<p class='lede'>No per-epoch pictures found. Track <code>graph</code> or <code>embeddings</code> to get them.</p>"}

<h2>Files</h2>
<ul class="files">{file_items}</ul>
</div></body></html>
"""
    index_path = out / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="directory to search for tracked runs")
    parser.add_argument("--out", type=Path, default=Path("evaluate_out"), help="output directory")
    parser.add_argument("--all-folds", action="store_true", help="one filmstrip per fold, not per run")
    args = parser.parse_args()

    index_path = build_report(args.root, args.out, all_folds=args.all_folds)
    print(f"\nOpen this: {index_path}")


if __name__ == "__main__":
    main()
