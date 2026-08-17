#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend dashboard: the scalars from many captures, plotted against time.

The per-file QA plots answer "what did file 37 look like". They cannot answer
"is something drifting", which is the question that actually matters during a
run -- and answering it today means opening 80 PNGs side by side. This reads the
scalars.json that vmm_reduce wrote for every store in a run and draws each
quantity as a series over file index.

The quantities are the ones that have already caught real problems on this
setup: occupancy (dead or hot hybrids), live channel count (P2_MID's ~39 dead
channels), ADC percentiles (P2_IN's compressed spectrum -- the gain story),
efficiency and its coincidence width, and the corrupt-word fraction.

    python vmm_trend.py <run_or_subrun_dir> --out trend.png
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Categorical slots 1-3 of the reference palette, in fixed order. Station
# identity owns its colour: P2_IN is always blue, whatever else is on the plot.
# Never cycled, never reassigned when a station drops out of a run.
STATION_COLORS = {
    "P2_IN":  "#2a78d6",   # slot 1, blue
    "P2_MID": "#eb6834",   # slot 2, orange
    "P2_OUT": "#1baf7a",   # slot 3, aqua
}
STATION_ORDER = ["P2_IN", "P2_MID", "P2_OUT"]

INK = "#33322e"
INK_MUTED = "#6b6a65"
GRID = "#e6e5e1"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def collect(root) -> list:
    """Every scalars.json under root, sorted by capture sequence.

    Accepts a subrun dir, a run dir, or a hits_store dir -- whatever is handed
    to it, it finds the stores underneath.
    """
    hits = glob.glob(os.path.join(str(root), "**", "scalars.json"), recursive=True)
    out = []
    for path in hits:
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception:
            continue
        rec["_store"] = os.path.dirname(path)
        rec["_subrun"] = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(path))))
        out.append(rec)

    def _seq(rec):
        name = rec.get("capture", "")
        try:
            from common_functions import parse_pcapng_name
            parsed = parse_pcapng_name(name)
            if parsed:
                return (parsed[0], parsed[1])
        except Exception:
            pass
        return ("", rec.get("mtime", 0))

    return sorted(out, key=_seq)


def _series(records, getter):
    """(x, y) with missing points dropped, so a failed file leaves a gap."""
    xs, ys = [], []
    for i, r in enumerate(records):
        try:
            v = getter(r)
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            xs.append(i)
            ys.append(float(v))
    return xs, ys


def _station_ids():
    try:
        import vmm_reduce as vr
        return vr.station_vmms()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def _plot_stations(ax, records, getter_for, ylabel, label_ends=True):
    """One line per station, fixed colour, end-labelled."""
    any_data = False
    for st in STATION_ORDER:
        xs, ys = _series(records, getter_for(st))
        if not xs:
            continue
        any_data = True
        ax.plot(xs, ys, color=STATION_COLORS[st], linewidth=1.8,
                marker="o", markersize=3.5, markeredgewidth=0, label=st, zorder=3)
        if label_ends:
            # direct label: identity without hunting the legend (<=4 series)
            ax.annotate(st, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(5, 0), fontsize=7.5, color=INK_MUTED,
                        va="center", annotation_clip=False)
    _style(ax, ylabel)
    return any_data


def make_dashboard(records, out_png, title=""):
    """Render the trend dashboard. Returns the path, or None if there is nothing."""
    if not records:
        return None

    panels = []

    # 1. total hits per file -- the first thing to look at
    panels.append(("Hits per file", "total",
                   lambda st: (lambda r: r.get("n_hits"))))

    # 2-5. per-station quantities
    panels.append(("Efficiency", "station",
                   lambda st: (lambda r: (r.get("efficiency") or {})
                               .get(st, {}).get("efficiency"))))
    panels.append(("Coincidence sigma (ns)", "station",
                   lambda st: (lambda r: (r.get("efficiency") or {})
                               .get(st, {}).get("sigma_ns"))))
    panels.append(("ADC p50", "station",
                   lambda st: (lambda r: (r.get("per_station") or {})
                               .get(st, {}).get("adc_p50"))))

    n = len(panels) + 1
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.1 * n), sharex=True,
                             squeeze=False)
    axes = [a[0] for a in axes]
    fig.patch.set_facecolor("white")

    # total hits: a single series needs no legend, the title names it
    xs, ys = _series(records, lambda r: r.get("n_hits"))
    axes[0].plot(xs, ys, color=STATION_COLORS["P2_IN"], linewidth=1.8,
                 marker="o", markersize=3.5, markeredgewidth=0, zorder=3)
    _style(axes[0], "Hits per file")

    for ax, (ylabel, kind, gf) in zip(axes[1:], panels[1:]):
        _plot_stations(ax, records, gf, ylabel)

    # live channels: dead-channel watch, per station, summed over its VMMs
    ax = axes[-1]
    st_ids = _station_ids()
    if st_ids:
        for st in STATION_ORDER:
            ids = st_ids.get(st, set())
            if not ids:
                continue
            xs, ys = _series(records, lambda r, ids=ids: sum(
                int(v) for k, v in (r.get("live_channels_per_vmm") or {}).items()
                if int(k) in ids) or None)
            if xs:
                ax.plot(xs, ys, color=STATION_COLORS[st], linewidth=1.8,
                        marker="o", markersize=3.5, markeredgewidth=0,
                        label=st, zorder=3)
                ax.annotate(st, (xs[-1], ys[-1]), textcoords="offset points",
                            xytext=(5, 0), fontsize=7.5, color=INK_MUTED,
                            va="center", annotation_clip=False)
    _style(ax, "Live channels")

    axes[-1].set_xlabel("capture index (chronological)", fontsize=9, color=INK)
    from matplotlib.ticker import MaxNLocator
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # one legend for the whole figure: identity is never colour-alone
    handles = [plt.Line2D([], [], color=STATION_COLORS[s], linewidth=1.8,
                          marker="o", markersize=3.5, label=s)
               for s in STATION_ORDER]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.99, 0.995))

    subruns = sorted({r.get("_subrun", "") for r in records if r.get("_subrun")})
    sub = f"{len(records)} captures"
    if subruns:
        sub += f"  |  {', '.join(subruns[:4])}{' …' if len(subruns) > 4 else ''}"
    fig.suptitle(title or "VMM trend", fontsize=13, color=INK, x=0.012, ha="left",
                 y=0.998)
    fig.text(0.012, 0.972, sub, fontsize=8.5, color=INK_MUTED, ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_png, dpi=110, facecolor="white")
    plt.close(fig)
    return out_png


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="run dir, subrun dir, or hits_store dir")
    ap.add_argument("--out", default="trend.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    recs = collect(args.root)
    if not recs:
        print(f"no scalars.json found under {args.root}")
        return 1
    path = make_dashboard(recs, args.out,
                          args.title or f"VMM trend — {os.path.basename(str(args.root).rstrip('/'))}")
    print(f"{len(recs)} captures -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
