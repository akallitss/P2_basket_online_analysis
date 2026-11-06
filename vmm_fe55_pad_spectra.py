#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_fe55_pad_spectra.py — is there really an Fe55 photopeak? Per-pad spectra,
starting from the pads that see the source most directly.

Why this exists
---------------
The detector-summed ADC spectrum of this detector is 50-80 % FWHM, which looks
nothing like a 5.9 keV line and led to the wrong conclusion that there was no
photopeak at all. That summed width is NOT the energy resolution: the pads
differ in gain by std/median ~ 0.33, so summing 896 pads whose peaks sit at
different ADC values smears the line by 2.355 x 0.33 = 78 % FWHM before the
intrinsic resolution contributes anything.

    measured summed FWHM 84 %  ~=  sqrt(31 %^2 + 78 %^2) = 84 %

The fix is to measure each pad's peak on its own and rescale before summing.
Done that way the line narrows from 63-80 % to 25-28 % FWHM, and in the
higher-gain run the Ar escape peak appears at exactly 0.50 x the main peak with
a clean valley between -- the textbook Fe55 signature in argon, and proof the
main peak is the 5.9 keV photopeak.

So: NEVER judge the photopeak from the summed spectrum on this detector.

Products
--------
  <base>_padspec_corrected.png   raw vs gain-corrected sum; the escape peak
  <base>_padspec_grid.png        individual spectra of the brightest pads
  <base>_padspec_checks.png      the null test (peak vs distance from source),
                                 per-pad resolution, and the gain distribution
  <base>_padspec.csv             per-pad peak, FWHM, resolution, distance
  <base>_padspec.json            summary

Usage
-----
  python3 vmm_fe55_pad_spectra.py <captures...> [--out-dir DIR] [--name NAME]
        [--n-pads 24] [--region-mm 120] [--fit-min 150] [--conditions "..."]

@author: ak271430 Alexandra Kallitsopoulou
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_det1_cabling as dc
import vmm_fe55_map as fm

# Reference ADC the gain-corrected spectrum is rescaled to. Arbitrary; 500 puts
# the 5.9 keV peak mid-range and the escape peak near 250, both comfortably
# inside the 10-bit axis at every gain setting seen so far.
REF_ADC = 500.0
CORR_BINS = 1400


def per_pad_peaks(adc, t, fit_min=150, drop_saturation=True):
    """Per-pad photopeak and FWHM, on a copy of the pad table.

    The ADC-1023 bin is zeroed first: it is saturation pile-up, not a spectral
    feature, and on the high-gain run it holds 14 % of all hits, which drags
    every median-based estimator upward and can become the argmax outright.
    """
    a = adc.astype(np.float64).copy()
    if drop_saturation:
        a[:, fm.ADC_MAX] = 0.0
    key = dc.pad_key(t["vmm"].to_numpy(), t["ch"].to_numpy())
    pk = np.full(len(t), np.nan)
    fw = np.full(len(t), np.nan)
    for i, k in enumerate(key):
        h = a[k].astype(np.int64)
        p = fm.windowed_median_peak(h, fit_min=fit_min)
        pk[i] = p
        fw[i] = fm.hist_fwhm(h, p)
    out = t.copy()
    out["pad_peak_adc"] = pk
    out["pad_fwhm_adc"] = fw
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pad_res_pct"] = 100.0 * fw / pk
    # NB: do not stash `key` in out.attrs -- pandas compares attrs elementwise
    # on concat/finalize and an ndarray there raises "truth value is ambiguous"
    # from deep inside unrelated operations. Callers recompute it; it is cheap.
    return out, a


def gain_corrected_sum(a, key, peaks, ref=REF_ADC, nbins=CORR_BINS):
    """Sum the per-pad spectra after rescaling each so its peak lands on `ref`.

    This is the whole point of the module: it removes the pad-to-pad gain
    spread, which otherwise dominates the summed width and hides the line.
    """
    out = np.zeros(nbins)
    src = np.arange(fm.ADC_BINS)
    used = 0
    for k, p in zip(key, peaks):
        if not np.isfinite(p) or p <= 0:
            continue
        h = a[k]
        nz = np.flatnonzero(h)
        if not nz.size:
            continue
        tgt = np.clip((src[nz] * (ref / p)).astype(int), 0, nbins - 1)
        np.add.at(out, tgt, h[nz])
        used += 1
    return out, used


def _smooth(h, k=9):
    return np.convolve(h, np.ones(k) / k, mode="same")


def peak_and_fwhm(h, lo=None):
    """Position and FWHM of the largest maximum above `lo`, on a smoothed copy."""
    sm = _smooth(np.asarray(h, float))
    lo = int(lo if lo is not None else 0.25 * len(sm))
    if lo >= len(sm) - 2 or not np.any(sm[lo:]):
        return np.nan, np.nan, sm
    p = int(np.argmax(sm[lo:])) + lo
    half = sm[p] / 2.0
    l = p
    while l > lo and sm[l] > half:
        l -= 1
    r = p
    while r < len(sm) - 1 and sm[r] > half:
        r += 1
    return float(p), float(r - l), sm


def escape_evidence(sm, peak):
    """Is there a bump at 0.50 x the main peak, above the valley between them?

    The Ar K-alpha (2.96 keV) escapes the gas for a fraction of conversions,
    leaving a second line at (5.90-2.96)/5.90 = 0.498 of the main charge.
    Resolving it is the standard proof the main peak is the photopeak and not a
    threshold turn-on, so the test is the ratio of the escape position to the
    valley halfway between -- above 1 means a real bump, not a shoulder.
    """
    if not np.isfinite(peak) or peak <= 0:
        return None
    e = int(round(fm.ESCAPE_FRACTION * peak))
    v = int(round(0.5 * (e + peak)))
    if e < 1 or v >= len(sm):
        return None
    return {"escape_adc": e, "valley_adc": v,
            "counts_escape": float(sm[e]), "counts_valley": float(sm[v]),
            "counts_main": float(sm[int(peak)]),
            "escape_over_valley": float(sm[e] / max(sm[v], 1.0)),
            "resolved": bool(sm[e] > 1.15 * sm[v])}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_corrected(raw, corr, base, out_dir, title, n_used, esc):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    x = np.arange(fm.ADC_BINS)
    p_raw, f_raw, _ = peak_and_fwhm(raw, lo=160)
    ax = axes[0]
    ax.step(x, raw, where="mid", color="0.3", lw=1.0)
    ax.set_xlabel("ADC")
    ax.set_ylabel("hits")
    ax.set_title(f"RAW sum of all pads — peak {p_raw:.0f}, "
                 f"FWHM/peak {100 * f_raw / p_raw:.0f} %\n"
                 "dominated by the pad-to-pad gain spread, NOT the resolution",
                 fontsize=9)
    ax.set_xlim(0, fm.ADC_BINS)

    xc = np.arange(len(corr))
    p_c, f_c, sm = peak_and_fwhm(corr, lo=int(0.35 * REF_ADC))
    ax = axes[1]
    ax.step(xc, corr, where="mid", color="0.3", lw=1.0, label="gain-corrected")
    ax.plot(xc, sm, "-", color="crimson", lw=1.2, alpha=0.8, label="smoothed")
    ax.axvline(p_c, color="crimson", ls="--", lw=1.2)
    if esc:
        ax.axvline(esc["escape_adc"], color="darkorange", ls=":", lw=1.6,
                   label=f"Ar escape (0.50x) at {esc['escape_adc']}")
        ax.axvline(esc["valley_adc"], color="steelblue", ls=":", lw=1.0,
                   label="valley")
    ax.set_xlim(0, min(len(corr), int(2.0 * REF_ADC)))
    ax.set_xlabel(f"ADC rescaled so each pad's peak = {REF_ADC:.0f}")
    ax.set_ylabel("hits")
    t2 = (f"GAIN-CORRECTED sum, {n_used} pads — peak {p_c:.0f}, "
          f"FWHM/peak {100 * f_c / p_c:.0f} %")
    if esc:
        t2 += ("\nescape/valley = %.2f  %s"
               % (esc["escape_over_valley"],
                  "-> Ar escape peak RESOLVED: this is the 5.9 keV line"
                  if esc["resolved"] else "-> escape peak not resolved "
                  "(likely below threshold at this gain)"))
    ax.set_title(t2, fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle(title + " — photopeak, before and after removing the "
                         "pad-to-pad gain spread", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out_dir, f"{base}_padspec_corrected.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p, p_c, f_c


def plot_grid(a, sel, base, out_dir, title, fit_min):
    """Individual spectra of the selected pads — the per-pad evidence itself."""
    n = len(sel)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.0 * nrow),
                             squeeze=False)
    x = np.arange(fm.ADC_BINS)
    key = dc.pad_key(sel["vmm"].to_numpy(), sel["ch"].to_numpy())
    for i, ((_, r), k) in enumerate(zip(sel.iterrows(), key)):
        ax = axes[i // ncol][i % ncol]
        h = a[k]
        ax.step(x, h, where="mid", color="0.35", lw=0.8)
        ax.plot(x, _smooth(h, 15), "-", color="crimson", lw=1.0)
        if np.isfinite(r["pad_peak_adc"]):
            ax.axvline(r["pad_peak_adc"], color="crimson", ls="--", lw=1.0)
            e = fm.ESCAPE_FRACTION * r["pad_peak_adc"]
            ax.axvline(e, color="darkorange", ls=":", lw=1.0)
        ax.axvline(fit_min, color="steelblue", ls="-.", lw=0.8)
        ax.set_xlim(0, min(fm.ADC_BINS, int(2.6 * np.nanmax([r["pad_peak_adc"], 100]))))
        ax.set_title(f"pad {int(r['channel_id'])}  c{int(r['connector_N'])}\n"
                     f"{int(r['n_hits'])} hits, peak {r['pad_peak_adc']:.0f}, "
                     f"{r['pad_res_pct']:.0f} %", fontsize=6.5)
        ax.tick_params(labelsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title + " — individual pad spectra (red dash = pad peak, "
                         "orange dot = expected Ar escape, blue = ADC cut)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out_dir, f"{base}_padspec_grid.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_checks(t, xs, ys, base, out_dir, title):
    """The null test plus the resolution and gain distributions."""
    g = t.dropna(subset=["pad_peak_adc"])
    rho = np.hypot(g["pad_cx"] - xs, g["pad_cy"] - ys)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.plot(rho, g["pad_peak_adc"], ".", ms=3, color="0.35")
    r = float(np.corrcoef(rho, g["pad_peak_adc"])[0, 1])
    med = float(g["pad_peak_adc"].median())
    ax.axhline(med, color="crimson", ls="--", lw=1.2,
               label=f"median {med:.0f}")
    ax.set_xlabel("distance from the source axis [mm]")
    ax.set_ylabel("pad photopeak [ADC]")
    ax.set_title("NULL TEST — a photopeak position depends on local GAIN,\n"
                 f"never on flux.  correlation r = {r:+.3f}", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    res = g["pad_res_pct"].dropna()
    ax.hist(res, bins=40, range=(0, 80), color="0.4")
    ax.axvline(res.median(), color="crimson", ls="--", lw=1.2,
               label=f"median {res.median():.0f} %")
    ax.set_xlabel("per-pad FWHM / peak [%]")
    ax.set_ylabel("pads")
    ax.set_title("per-pad energy resolution\n(the summed spectrum is far "
                 "broader — that is gain spread)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[2]
    rel = g["pad_peak_adc"] / med
    ax.hist(rel, bins=40, color="0.4")
    ax.set_xlabel("pad peak / detector median  =  relative gain")
    ax.set_ylabel("pads")
    ax.set_title(f"pad-to-pad GAIN spread — std/median = "
                 f"{g['pad_peak_adc'].std() / med:.3f}\n"
                 f"contributes {2.355 * g['pad_peak_adc'].std() / med * 100:.0f} %"
                 " FWHM to any summed spectrum", fontsize=9)

    fig.suptitle(title + " — per-pad photopeak checks", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out_dir, f"{base}_padspec_checks.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p, r


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-pad Fe55 photopeak check, from the pads that see the "
                    "source most directly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--name", default="fe55_padspec")
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--max-packets", type=int, default=None)
    ap.add_argument("--fit-min", type=int, default=150,
                    help="ignore ADC below this when locating a pad's peak")
    ap.add_argument("--min-hits", type=int, default=500,
                    help="pads with fewer hits get no per-pad peak")
    ap.add_argument("--n-pads", type=int, default=24,
                    help="how many of the brightest pads to draw individually")
    ap.add_argument("--region-mm", type=float, default=None,
                    help="restrict the corrected sum to pads within this "
                         "distance of the source axis (default: all pads)")
    ap.add_argument("--keep-saturation", action="store_true",
                    help="do NOT zero the ADC-1023 bin (it is pile-up, not a "
                         "spectral feature; zeroing is the default)")
    args = ap.parse_args(argv)

    paths = fm.resolve_inputs(args.inputs)
    print(f"decoding {len(paths)} capture(s)")
    hists, _ = fm.accumulate(paths, max_packets=args.max_packets)
    live = dc.live_time(paths)
    tab = dc.build_pad_table()
    t = fm.per_pad_table(hists, tab, max(live["live_s"], 1e-9), min_events=30)
    t = fm.mask_hot_pads(t)
    t = t[~t["hot"] & (t["n_hits"] >= args.min_hits)].reset_index(drop=True)
    print(f"  live {live['live_s']:.1f} s, {len(t)} pads with "
          f">= {args.min_hits} hits")

    t, a = per_pad_peaks(hists.adc, t, fit_min=args.fit_min,
                         drop_saturation=not args.keep_saturation)

    # Source axis: the rate-weighted centroid, which is the estimator that
    # survives a non-Gaussian spot (see vmm_fe55_map.move_advice).
    w = (t["rate_hz"] / t["pad_area"]).to_numpy()
    xs = float((w * t["pad_cx"]).sum() / w.sum())
    ys = float((w * t["pad_cy"]).sum() / w.sum())
    t["rho_mm"] = np.hypot(t["pad_cx"] - xs, t["pad_cy"] - ys)

    sel = t
    if args.region_mm:
        sel = t[t["rho_mm"] <= args.region_mm]
        print(f"  region cut: {len(sel)} pads within {args.region_mm:.0f} mm "
              "of the source axis")
    key = dc.pad_key(sel["vmm"].to_numpy(), sel["ch"].to_numpy())
    raw = a[key].sum(axis=0)
    corr, n_used = gain_corrected_sum(a, key, sel["pad_peak_adc"].to_numpy())

    p_c, f_c, sm = peak_and_fwhm(corr, lo=int(0.35 * REF_ADC))
    esc = escape_evidence(sm, p_c)

    title = f"det1 Fe55 — {len(paths)} capture(s)"
    if args.conditions:
        title += f", {args.conditions}"
    os.makedirs(args.out_dir, exist_ok=True)

    f1, p_corr, fw_corr = plot_corrected(raw, corr, args.name, args.out_dir,
                                         title, n_used, esc)
    bright = t.nlargest(args.n_pads, "n_hits")
    f2 = plot_grid(a, bright, args.name, args.out_dir, title, args.fit_min)
    f3, r_null = plot_checks(t, xs, ys, args.name, args.out_dir, title)

    cols = ["channel_id", "vmm", "ch", "connector_N", "half", "pad_cx",
            "pad_cy", "radius", "rho_mm", "n_hits", "rate_hz",
            "pad_peak_adc", "pad_fwhm_adc", "pad_res_pct", "sat_frac"]
    t[cols].to_csv(os.path.join(args.out_dir, f"{args.name}_padspec.csv"),
                   index=False)

    g = t.dropna(subset=["pad_peak_adc"])
    med = float(g["pad_peak_adc"].median())
    summary = {
        "conditions": args.conditions,
        "inputs": [os.path.basename(p) for p in paths],
        "live_s": live["live_s"],
        "n_pads_with_peak": int(len(g)),
        "source_axis_mm": [xs, ys],
        "pad_peak_adc": {"median": med,
                         "p5": float(np.percentile(g["pad_peak_adc"], 5)),
                         "p95": float(np.percentile(g["pad_peak_adc"], 95)),
                         "std_over_median": float(g["pad_peak_adc"].std() / med)},
        "per_pad_resolution_fwhm_pct": {
            "median": float(g["pad_res_pct"].median()),
            "p10": float(np.nanpercentile(g["pad_res_pct"], 10)),
            "best": float(np.nanmin(g["pad_res_pct"]))},
        "summed_raw": dict(zip(("peak_adc", "fwhm_adc"),
                               [float(v) for v in peak_and_fwhm(raw, lo=160)[:2]])),
        "gain_corrected": {"peak_adc": float(p_corr), "fwhm_adc": float(fw_corr),
                           "resolution_fwhm_pct": float(100 * fw_corr / p_corr),
                           "n_pads": int(n_used), "reference_adc": REF_ADC},
        "escape_peak": esc,
        "null_test_corr_peak_vs_distance": float(r_null),
        "figures": [os.path.basename(x) for x in (f1, f2, f3)],
    }
    with open(os.path.join(args.out_dir, f"{args.name}_padspec.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"PER-PAD  peak median {med:.0f} ADC (p5-p95 "
          f"{summary['pad_peak_adc']['p5']:.0f}-"
          f"{summary['pad_peak_adc']['p95']:.0f}), gain spread std/med "
          f"{summary['pad_peak_adc']['std_over_median']:.3f}")
    print(f"         resolution median {summary['per_pad_resolution_fwhm_pct']['median']:.1f} %"
          f", best {summary['per_pad_resolution_fwhm_pct']['best']:.1f} %")
    print(f"SUMMED   raw FWHM/peak "
          f"{100 * summary['summed_raw']['fwhm_adc'] / summary['summed_raw']['peak_adc']:.0f} %"
          f"  ->  gain-corrected {summary['gain_corrected']['resolution_fwhm_pct']:.0f} %"
          f"  ({n_used} pads)")
    if esc:
        print(f"ESCAPE   {'RESOLVED' if esc['resolved'] else 'not resolved'}: "
              f"escape/valley = {esc['escape_over_valley']:.2f} at ADC "
              f"{esc['escape_adc']} (0.50 x peak {p_corr:.0f})")
    print(f"NULL     corr(pad peak, distance from source) = {r_null:+.3f} "
          f"{'-- consistent with a photopeak' if abs(r_null) < 0.15 else '-- SUSPICIOUS'}")
    print(f"{'=' * 70}\nwrote {args.out_dir}/{args.name}_padspec*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
