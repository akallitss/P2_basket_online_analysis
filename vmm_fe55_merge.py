#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_fe55_merge.py — stitch per-pad gain measurements from several Fe55 source
positions into one gain map of the whole detector.

The problem
-----------
A collimated Fe55 source lights ~100 of the detector's 896 instrumented pads.
A gain map of the whole area therefore means several runs at several source
positions, and those runs are NOT directly comparable: the source sits at a
different distance, the exposure differs, the mesh may have been re-ramped, the
gas may have drifted between the first position and the last. Simply pasting
the per-run photopeaks side by side would render every one of those differences
as a gain step at the seam between positions.

The method
----------
Each run r measures a subset of pads. Model the measurement as

    peak(r, p) = s_r * g_p

with s_r a per-run scale (everything about run r that is not the detector) and
g_p the pad's gain (what is wanted). Taking logs makes it linear,

    log peak(r, p) = log s_r + log g_p

which is a least-squares problem in (log s, log g), solved over every pad
measured by any run at once. Runs are tied together through the pads they SHARE,
so overlap between neighbouring source positions is not optional -- it is the
only thing that fixes their relative scale. The gauge is fixed by setting the
median log s_r to zero, so the merged map is in the units of a typical run.

Weights are sqrt(n_events): the fractional error on a photopeak estimated from
N events goes as 1/sqrt(N), so a pad in the core of the spot deserves more say
than one in the tail.

What comes out
--------------
  <out>_gain_map.png     merged relative gain over the whole fan
  <out>_coverage.png     how many runs measured each pad, and WHICH PADS ARE
                         STILL UNMEASURED -- i.e. where to put the source next
  <out>_scales.png       the fitted per-run scale factors; a run far from the
                         others means conditions changed, not the detector
  <out>_merged_pads.csv  one row per pad: merged gain, n_runs, spread
  <out>_summary.json

Usage
-----
  python3 vmm_fe55_merge.py pos1_pads.csv pos2_pads.csv ... [--out-dir DIR]
  python3 vmm_fe55_merge.py 'analysis/**/*_pads.csv' --name det1_gain

@author: ak271430 Alexandra Kallitsopoulou
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_det1_cabling as dc


def load_runs(paths, min_events=30):
    """Read the per-pad CSVs. Returns a long DataFrame of usable measurements."""
    frames = []
    for p in paths:
        t = pd.read_csv(p)
        need = {"channel_id", "peak_adc", "n_hits"}
        if not need <= set(t.columns):
            raise ValueError(f"{p}: missing columns {need - set(t.columns)}")
        t = t[t["peak_adc"].notna() & (t["peak_adc"] > 0)
              & (t["n_hits"] >= min_events)]
        if "hot" in t:
            t = t[~t["hot"].astype(bool)]
        if not len(t):
            print(f"  {os.path.basename(p)}: no usable pads, skipped")
            continue
        frames.append(pd.DataFrame({
            "run": os.path.basename(p).replace("_pads.csv", ""),
            "channel_id": t["channel_id"].to_numpy(),
            "peak_adc": t["peak_adc"].to_numpy(float),
            "n_hits": t["n_hits"].to_numpy(float),
        }))
        print(f"  {os.path.basename(p)}: {len(t)} pads")
    if not frames:
        raise SystemExit("no usable measurements in any input")
    return pd.concat(frames, ignore_index=True)


def solve_scales(obs):
    """Least-squares split of log peak into a per-run scale and a per-pad gain.

    Returns (scales, gains, info). `scales` is a Series indexed by run,
    `gains` a Series indexed by channel_id, both in linear units.
    """
    runs = sorted(obs["run"].unique())
    pads = sorted(obs["channel_id"].unique())
    ri = {r: i for i, r in enumerate(runs)}
    pi = {p: i for i, p in enumerate(pads)}
    n_r, n_p = len(runs), len(pads)

    y = np.log(obs["peak_adc"].to_numpy(float))
    w = np.sqrt(obs["n_hits"].to_numpy(float))

    # Design matrix: one column per run, one per pad. Dense is fine -- at most
    # a few dozen runs and 896 pads, so this is at worst a few thousand by ~950.
    A = np.zeros((len(obs) + 1, n_r + n_p))
    A[np.arange(len(obs)), [ri[r] for r in obs["run"]]] = w
    A[np.arange(len(obs)), n_r + np.array([pi[p] for p in obs["channel_id"]])] = w
    b = np.concatenate([y * w, [0.0]])
    # Gauge: log s is defined only up to a constant traded against log g, so
    # pin the mean run scale to 1. Without this the system is singular and the
    # split between "run was brighter" and "detector gain is higher" is
    # arbitrary.
    A[-1, :n_r] = 1.0

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    log_s, log_g = sol[:n_r], sol[n_r:]

    pred = log_s[[ri[r] for r in obs["run"]]] + \
        log_g[[pi[p] for p in obs["channel_id"]]]
    resid = y - pred
    info = {
        "n_runs": n_r, "n_pads": n_p, "n_obs": int(len(obs)),
        "residual_rms_log": float(np.sqrt(np.average(resid ** 2, weights=w))),
        "residual_rms_pct": float(100 * np.sqrt(np.average(resid ** 2,
                                                           weights=w))),
    }
    return (pd.Series(np.exp(log_s), index=runs),
            pd.Series(np.exp(log_g), index=pads), info, resid)


def overlap_matrix(obs):
    """Pads shared between each pair of runs — the connectivity that makes the
    stitch possible. A run sharing nothing with the rest cannot be placed on
    the common scale and must be reported, not silently floated."""
    runs = sorted(obs["run"].unique())
    sets = {r: set(obs.loc[obs["run"] == r, "channel_id"]) for r in runs}
    m = pd.DataFrame(0, index=runs, columns=runs, dtype=int)
    for a in runs:
        for b in runs:
            m.loc[a, b] = len(sets[a] & sets[b])
    return m


def disconnected_runs(ov):
    """Runs not connected to the largest overlap component."""
    runs = list(ov.index)
    if len(runs) < 2:
        return []
    seen, stack = {runs[0]}, [runs[0]]
    while stack:
        a = stack.pop()
        for b in runs:
            if b not in seen and ov.loc[a, b] > 0:
                seen.add(b)
                stack.append(b)
    return [r for r in runs if r not in seen]


def plot_merged(gains, full, out_png, title, n_runs):
    verts = dc.pad_polygons(full)
    g = full["channel_id"].map(gains).to_numpy(float)
    rel = g / np.nanmedian(g)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.add_collection(PolyCollection(verts, facecolors="0.93",
                                     edgecolors="0.82", linewidths=0.25))
    ok = np.isfinite(rel)
    if ok.any():
        pc = PolyCollection(verts[ok], array=rel[ok], cmap="coolwarm",
                            edgecolors="face", linewidths=0.2)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, label="relative gain (merged / median)")
    ax.set_xlabel("pad_cx [mm]")
    ax.set_ylabel("pad_cy [mm]")
    ax.set_aspect("equal")
    ax.autoscale_view()
    spread = np.nanstd(rel[ok]) if ok.sum() > 1 else np.nan
    ax.set_title(f"{title}\n{int(ok.sum())}/{len(full)} pads from {n_runs} "
                 f"source position(s), spread std = {spread:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(obs, full, out_png, title):
    """How many runs measured each pad, and what is still unmeasured."""
    n_runs = obs.groupby("channel_id")["run"].nunique()
    verts = dc.pad_polygons(full)
    cov = full["channel_id"].map(n_runs).fillna(0).to_numpy(float)
    inst = full["instrumented"].to_numpy(bool)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    ax = axes[0]
    ax.add_collection(PolyCollection(verts, facecolors="0.93",
                                     edgecolors="0.82", linewidths=0.25))
    m = cov > 0
    if m.any():
        pc = PolyCollection(verts[m], array=cov[m], cmap="viridis",
                            edgecolors="face", linewidths=0.2)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, label="source positions measuring this pad")
    ax.set_title("coverage — pads measured by >1 position are what\n"
                 "tie the positions onto a common gain scale", fontsize=9)

    ax = axes[1]
    todo = inst & (cov == 0)
    ax.add_collection(PolyCollection(verts[~inst], facecolors="0.93",
                                     edgecolors="0.82", linewidths=0.25))
    ax.add_collection(PolyCollection(verts[inst & ~todo], facecolors="#2f7d32",
                                     edgecolors="face", linewidths=0.2))
    if todo.any():
        ax.add_collection(PolyCollection(verts[todo], facecolors="#c62828",
                                         edgecolors="face", linewidths=0.2))
    ax.set_title(f"green = measured, red = still to do "
                 f"({int(todo.sum())}/{int(inst.sum())} pads), "
                 f"grey = not read out", fontsize=9)

    for ax in axes:
        ax.set_xlabel("pad_cx [mm]")
        ax.set_ylabel("pad_cy [mm]")
        ax.set_aspect("equal")
        ax.autoscale_view()
    fig.suptitle(title + " — coverage", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return int(todo.sum())


def plot_scales(scales, info, resid, out_png, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.bar(range(len(scales)), scales.to_numpy(), color="0.35")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.2)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(scales.index, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("fitted run scale $s_r$")
    ax.set_title("per-run scale — a run far from 1 means the CONDITIONS\n"
                 "changed (source distance, HV, gas), not the detector",
                 fontsize=9)
    ax = axes[1]
    ax.hist(100 * resid, bins=40, color="0.35")
    ax.set_xlabel("residual after the stitch [%]")
    ax.set_ylabel("pad-measurements")
    ax.set_title(f"stitch residual rms = {info['residual_rms_pct']:.1f} %\n"
                 "large tails mean the model peak = scale x gain is failing",
                 fontsize=9)
    fig.suptitle(title + " — stitching diagnostics", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stitch per-pad Fe55 gain measurements from several source "
                    "positions into one map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="*_pads.csv files from vmm_fe55_map.py, or a glob")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--name", default="det1_fe55_merged")
    ap.add_argument("--min-events", type=int, default=30,
                    help="ignore pad measurements with fewer hits")
    args = ap.parse_args(argv)

    paths = []
    for p in args.inputs:
        paths += sorted(glob.glob(p, recursive=True)) if any(c in p for c in "*?[") \
            else [p]
    paths = sorted(set(paths))
    if not paths:
        print("no input CSVs found", file=sys.stderr)
        return 2

    print(f"reading {len(paths)} per-pad table(s)")
    obs = load_runs(paths, min_events=args.min_events)

    ov = overlap_matrix(obs)
    orphans = disconnected_runs(ov)
    if orphans:
        print(f"\nWARNING: {len(orphans)} run(s) share no pad with the rest and "
              f"cannot be put on the common scale: {orphans}")
        print("         Their gains are still fitted, but their scale is "
              "unconstrained relative to the others —\n"
              "         overlap the source positions so neighbouring runs share "
              "a band of pads.")

    scales, gains, info, resid = solve_scales(obs)
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)
    full = dc.full_pad_map()
    title = f"det1 Fe55 merged gain map"

    plot_merged(gains, full, base + "_gain_map.png", title, len(scales))
    n_todo = plot_coverage(obs, full, base + "_coverage.png", title)
    plot_scales(scales, info, resid, base + "_scales.png", title)

    per_pad = obs.groupby("channel_id").agg(
        n_runs=("run", "nunique"), n_hits=("n_hits", "sum"))
    per_pad["gain_adc"] = gains
    per_pad["rel_gain"] = per_pad["gain_adc"] / per_pad["gain_adc"].median()
    per_pad = per_pad.join(
        full.set_index("channel_id")[["pad_cx", "pad_cy", "radius", "phi",
                                      "connector_number"]])
    csv = base + "_merged_pads.csv"
    per_pad.reset_index().to_csv(csv, index=False)

    summary = {
        "inputs": [os.path.basename(p) for p in paths],
        "runs": {r: float(s) for r, s in scales.items()},
        "fit": info,
        "pads_measured": int(len(per_pad)),
        "pads_instrumented": int(full["instrumented"].sum()),
        "pads_still_unmeasured": n_todo,
        "coverage_frac": float(len(per_pad) / max(int(full["instrumented"].sum()), 1)),
        "rel_gain_spread_std": float(per_pad["rel_gain"].std()),
        "runs_not_connected_by_overlap": orphans,
        "overlap_pads": ov.to_dict(),
    }
    with open(base + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nstitched {info['n_runs']} run(s), {info['n_obs']} pad-measurements"
          f" -> {info['n_pads']} pads")
    print(f"  residual rms {info['residual_rms_pct']:.1f} %")
    print(f"  run scales: "
          + ", ".join(f"{r}={s:.3f}" for r, s in scales.items()))
    print(f"  coverage {len(per_pad)}/{int(full['instrumented'].sum())} "
          f"instrumented pads; {n_todo} still unmeasured")
    print(f"  relative-gain spread std = {per_pad['rel_gain'].std():.3f}")
    print(f"wrote {base}_*.png, _merged_pads.csv, _summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
