#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_fe55_gain_map.py — the det1 gain map: per-pad Fe55 photopeak on the real pad
tiles, as a RELATIVE gain map and (with a stated calibration) an ABSOLUTE one.

Method, and why it is this one
------------------------------
This is the standard Fe55 gain-map recipe used for Micromegas readout planes
(T2K TPC modules, PandaX-III, the resistive-Micromegas characterisations):

  1. Per pad, histogram the ADC of every hit on that pad alone.
  2. Locate the 5.9 keV photopeak in that histogram. The peak POSITION is the
     observable -- not the rate, not the mean.
  3. relative gain = peak(pad) / median(peak over all pads). Plot it on the pad
     geometry. Quote non-uniformity as RMS/mean of the peak distribution.
  4. For an ABSOLUTE gain, convert the peak position to collected charge and
     divide by the primary charge:

         G = Q_peak / (N_primary * e)

     with N_primary = E_gamma / W the number of primary electrons liberated by
     one 5.9 keV photon in the gas.

Step 3 needs NO electronics calibration and is what the literature figures
almost always show. Step 4 needs the ADC -> charge conversion of the front end,
and on this chain that is the one genuinely uncertain link (see below).

Why the peak position and not the mean or the rate
--------------------------------------------------
  * The RATE map measures how many photons arrive -- source geometry, not gain.
  * The MEAN ADC is pulled by the escape peak, the noise tail and any
    saturation pile-up, all of which vary with gain and threshold.
  * The PEAK POSITION is proportional to the charge of a fixed-energy line, so
    it tracks the gas gain and nothing else. The null test in
    vmm_fe55_pad_spectra.py confirms this on our data: correlation between a
    pad's peak and its distance from the source is |r| < 0.13, i.e. the peak
    does not care about flux, exactly as a photopeak should not.

Absolute gain: the calibration chain, and its weak link
-------------------------------------------------------
    ADC counts --(a)--> mV --(b)--> fC --(c)--> gain

  (a) ADC_FULL_SCALE_MV / 1023 mV per count.
  (b) divide by the VMM3a preamp gain setting in mV/fC (0.5, 1, 3, 4.5, 6, 9,
      12, 16 are the eight selectable values).
  (c) divide by N_primary * e.

(b) and (c) are documented numbers. (a) IS NOT PINNED DOWN by any source I
could reach: the VMM3a papers quote the gain settings and the 10-bit ADC but
not the full-scale voltage, so ADC_FULL_SCALE_MV below is a NOMINAL value and
every absolute gain printed here inherits its error directly and linearly.

    ==> The relative map is solid. The absolute scale is provisional until a
        VMM3a internal test-pulse scan is taken: inject a known charge through
        the on-chip calibration capacitor, read the ADC, and the (a)x(b)
        product is measured rather than assumed. Pass the measured value with
        --fc-per-adc and the caveat goes away.

Usage
-----
  python3 vmm_fe55_gain_map.py <captures...> --out-dir DIR --name NAME
        [--vmm-gain 4.5] [--gas ar_co2_93_7] [--fc-per-adc 0.217]
        [--conditions "..."]

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
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_det1_cabling as dc
import vmm_fe55_map as fm
import vmm_fe55_pad_spectra as ps

E_FE55_KEV = 5.90            # Mn K-alpha
ELEMENTARY_CHARGE_FC = 1.602176634e-4

# Primary electrons liberated by one 5.9 keV photon. Literature values, not
# derived here -- the mixture W is not simply the volume-weighted mean, so a
# measured/simulated number is preferred where one exists.
#   ar_co2_90_10 : 211 e-, measured (Garfield histograms, MPGD characterisations)
#   ar_co2_93_7  : W ~ 26.4 eV -> 5900/26.4 = 223 e-  (PandaX-III Micromegas)
#   ar_co2_iso_93_5_2 : the SPS beam gas; W taken as the Ar/CO2 value, isobutane
#                       at 2 % shifts it by well under the spread of the map
GASES = {
    "ar_co2_90_10":      {"label": "Ar/CO2 90/10",         "n_primary": 211.0},
    "ar_co2_93_7":       {"label": "Ar/CO2 93/7",          "n_primary": 223.0},
    "ar_co2_iso_93_5_2": {"label": "Ar/CO2/iC4H10 93/5/2", "n_primary": 224.0},
    "ar_iso_95_5":       {"label": "Ar/iC4H10 95/5",       "n_primary": 218.0},
}
# CONFIRMED 2026-09-11 (Alexandra): the det1 bench source runs use Ar/iC4H10
# 95/5 and a VMM3a preamp gain of 3 mV/fC.
DEFAULT_GAS = "ar_iso_95_5"
DEFAULT_VMM_GAIN = 3.0

# NOMINAL peak-detector full-scale swing. See the module docstring: this is the
# unverified link in the absolute chain. Override with --fc-per-adc once a
# test-pulse scan exists.
ADC_FULL_SCALE_MV = 1000.0
ADC_MAX_COUNT = 1023.0
VMM_GAIN_SETTINGS = (0.5, 1.0, 3.0, 4.5, 6.0, 9.0, 12.0, 16.0)


def fc_per_adc(vmm_gain_mv_per_fc, full_scale_mv=ADC_FULL_SCALE_MV):
    """Charge in fC corresponding to one ADC count."""
    return (full_scale_mv / ADC_MAX_COUNT) / float(vmm_gain_mv_per_fc)


def absolute_gain(peak_adc, fc_adc, n_primary):
    """Gas gain from a photopeak position."""
    q_fc = np.asarray(peak_adc, dtype=float) * fc_adc
    return q_fc / (n_primary * ELEMENTARY_CHARGE_FC)


def linearity_from_escape(a, key, peaks, thr_adc, min_pads=10, margin=1.6):
    """Two-point per-pad energy calibration using the Ar escape peak.

    The escape line at 2.94 keV and the main line at 5.90 keV give two points on
    the same pad. If the chain is linear THROUGH THE ORIGIN -- which is what the
    absolute-gain formula assumes -- their ADC ratio must equal the energy ratio
    0.498. A ratio systematically above it means a positive ADC offset (pedestal
    not subtracted), which would inflate every absolute gain.

    Only pads whose expected escape peak sits clearly above the threshold can
    say anything; the rest are skipped rather than contributing a biased value.
    """
    out = []
    for k, p in zip(key, peaks):
        if not np.isfinite(p) or p <= 0:
            continue
        e_exp = fm.ESCAPE_FRACTION * p
        # The escape line must sit CLEARLY above the threshold, not merely
        # above it. A line whose low side is clipped has its measured position
        # pushed upward, which fakes exactly the "ratio too high -> pedestal"
        # signature this function exists to detect. Measured on our low-gain
        # runs: a 1.25x margin still returned 0.55 (i.e. "non-linear") purely
        # from clipping, while the high-gain run, where the line is far clear
        # of threshold, gives 0.509 -- consistent with the true 0.498.
        if e_exp < margin * thr_adc:
            continue
        h = ps._smooth(a[k], 11)
        lo = max(int(1.05 * thr_adc), int(0.30 * p))
        hi = int(0.78 * p)
        if hi <= lo + 3:
            continue
        out.append((int(np.argmax(h[lo:hi])) + lo) / p)
    if len(out) < min_pads:
        return {"n_pads": len(out), "usable": False,
                "note": f"only {len(out)} pads have the escape line clearly "
                        f"({margin:g}x) above the ADC-{thr_adc:.0f} threshold; "
                        "raise the gain to make this test possible"}
    r = np.array(out)
    med = float(np.median(r))
    # offset implied if the chain were linear with a pedestal:
    #   (E_esc/E_main) = (peak_esc - off)/(peak_main - off)
    #   -> off = (r*P - f*P)/(r - f) for measured ratio r, true fraction f
    f = fm.ESCAPE_FRACTION
    return {"n_pads": int(len(r)), "usable": True,
            "escape_over_main_median": med,
            "escape_over_main_std": float(r.std()),
            "expected": float(f),
            "deviation_pct": float(100.0 * (med - f) / f),
            "implied_adc_offset_frac_of_peak": float((med - f) / (1.0 - f))
            if med > f else 0.0,
            "linear_through_origin": bool(abs(med - f) < 0.05)}


def gas_or_electronics(t):
    """Is the measured non-uniformity the GAS GAIN or the readout electronics?

    The decisive test, and it is available for free because the strip -> pad
    mapping is an interleaved permutation: pads that are NEIGHBOURS IN SPACE are
    mostly NOT adjacent channels on the same VMM, and vice versa.

      * a real gas-gain variation (mesh planarity, pillar density, amplification
        gap) is a smooth function of POSITION -> neighbouring pads correlate,
        adjacent channel numbers do not;
      * a per-channel electronics spread is a property of the CHANNEL ->
        neither correlates spatially, and the map would look like salt-and-
        pepper noise on the tiles.

    Measured on det1: spatial r = +0.59 to +0.89, adjacent-channel r = 0.00 to
    -0.08. The non-uniformity is the detector, not the readout.
    """
    g = t[t["valid"]].dropna(subset=["rel_gain"])
    if len(g) < 50:
        return {"usable": False, "note": "too few valid pads"}
    cx = g["pad_cx"].to_numpy(float); cy = g["pad_cy"].to_numpy(float)
    v = g["rel_gain"].to_numpy(float)
    d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    np.fill_diagonal(d, np.inf)
    nb = np.argsort(d, axis=1)[:, :6]
    r_space = float(np.corrcoef(v, v[nb].mean(axis=1))[0, 1])
    rr = []
    for _, grp in g.sort_values(["vmm", "ch"]).groupby("vmm"):
        a = grp["rel_gain"].to_numpy(float)
        if len(a) >= 20:
            rr.append(np.corrcoef(a[:-1], a[1:])[0, 1])
    r_chan = float(np.mean(rr)) if rr else float("nan")
    gm = g.groupby("vmm")["rel_gain"].mean()
    return {"usable": True,
            "spatial_neighbour_corr": r_space,
            "adjacent_channel_corr": r_chan,
            "between_vmm_spread": float(gm.std()),
            "within_vmm_spread": float(
                g.groupby("vmm")["rel_gain"].transform(lambda s: s - s.mean()).std()),
            "verdict": ("gas gain — varies smoothly in space, not with channel "
                        "number" if r_space > 0.3 and abs(r_chan) < 0.25 else
                        "inconclusive / possible electronics contribution")}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _tiles(ax, verts_bg):
    ax.add_collection(PolyCollection(verts_bg, facecolors="0.93",
                                     edgecolors="0.82", linewidths=0.25))
    ax.set_xlabel("pad_cx [mm]")
    ax.set_ylabel("pad_cy [mm]")
    ax.set_aspect("equal")


def plot_gain_map(t, full, base, out_dir, title, gain_col, label, cmap,
                  suffix, vmin=None, vmax=None, note=""):
    """The map itself, plus the distribution and the radial trend."""
    verts_bg = dc.pad_polygons(full)
    verts = dc.pad_polygons(t)
    v = t[gain_col].to_numpy(float)
    ok = np.isfinite(v)

    fig = plt.figure(figsize=(15.5, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    _tiles(ax, verts_bg)
    if ok.any():
        pc = PolyCollection(verts[ok], array=v[ok], cmap=cmap,
                            edgecolors="face", linewidths=0.2)
        if vmin is not None:
            pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, label=label, fraction=0.046, pad=0.03)
    ax.autoscale_view()
    ax.set_title(f"{int(ok.sum())}/{len(t)} pads   grey = not read out",
                 fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    if ok.any():
        ax.hist(v[ok], bins=45, color="0.4")
        m = float(np.mean(v[ok]))
        ax.axvline(m, color="crimson", ls="--", lw=1.3, label=f"mean {m:.3g}")
        ax.legend(fontsize=8)
    ax.set_xlabel(label)
    ax.set_ylabel("pads")
    rms = float(np.std(v[ok]) / np.mean(v[ok])) if ok.sum() > 1 else np.nan
    ax.set_title(f"non-uniformity RMS/mean = {100 * rms:.1f} %", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    if ok.any():
        r = t["radius"].to_numpy(float)[ok]
        bins = np.linspace(np.nanmin(r), np.nanmax(r), 16)
        idx = np.digitize(r, bins) - 1
        xs, ys, es = [], [], []
        for i in range(len(bins) - 1):
            m = idx == i
            if m.sum() < 5:
                continue
            xs.append(0.5 * (bins[i] + bins[i + 1]))
            ys.append(np.mean(v[ok][m]))
            es.append(np.std(v[ok][m]) / np.sqrt(m.sum()))
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color="0.25", ms=4, lw=1.2)
    ax.set_xlabel("pad radius [mm]")
    ax.set_ylabel(label)
    ax.set_title("radial trend — a real gas-gain gradient shows here\n"
                 "(mesh-to-strip distance varies across the fan)", fontsize=9)

    fig.suptitle(title + f" — {label}", fontsize=11)
    if note:
        # As a footer, not in the suptitle: a four-line note above three axes
        # overlaps their own titles and makes the figure unreadable.
        fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8,
                 color="0.25")
    fig.subplots_adjust(top=0.88, bottom=0.26 if note else 0.12)
    p = os.path.join(out_dir, f"{base}_{suffix}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p, rms


def plot_quality(t, full, base, out_dir, title, min_hits):
    """Is the map trustworthy where it is drawn? Statistics, resolution, and
    the pads whose peak could not be measured."""
    verts_bg = dc.pad_polygons(full)
    verts = dc.pad_polygons(t)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    from matplotlib.colors import LogNorm
    ax = axes[0]
    _tiles(ax, verts_bg)
    n = t["n_hits"].to_numpy(float)
    m = n > 0
    pc = PolyCollection(verts[m], array=n[m], cmap="inferno",
                        norm=LogNorm(vmin=max(n[m].min(), 1)),
                        edgecolors="face", linewidths=0.2)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, label="hits per pad (log)", fraction=0.046, pad=0.03)
    ax.autoscale_view()
    ax.set_title("statistics behind each pad's peak", fontsize=9)

    ax = axes[1]
    _tiles(ax, verts_bg)
    res = t["pad_res_pct"].to_numpy(float)
    ok = np.isfinite(res)
    pc = PolyCollection(verts[ok], array=res[ok], cmap="magma",
                        edgecolors="face", linewidths=0.2)
    pc.set_clim(np.nanpercentile(res[ok], 2), np.nanpercentile(res[ok], 98))
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, label="FWHM/peak [%]", fraction=0.046, pad=0.03)
    ax.autoscale_view()
    ax.set_title(f"energy resolution — median "
                 f"{np.nanmedian(res):.0f} %", fontsize=9)

    ax = axes[2]
    _tiles(ax, verts_bg)
    bad = ~np.isfinite(t["pad_peak_adc"].to_numpy(float))
    if bad.any():
        ax.add_collection(PolyCollection(verts[bad], facecolors="#c62828",
                                         edgecolors="face", linewidths=0.2))
    ax.add_collection(PolyCollection(verts[~bad], facecolors="#2f7d32",
                                     edgecolors="face", linewidths=0.2))
    ax.autoscale_view()
    ax.set_title(f"green = peak measured, red = not "
                 f"({int(bad.sum())} pads, < {min_hits} hits or no peak)",
                 fontsize=9)

    fig.suptitle(title + " — gain-map quality", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out_dir, f"{base}_gainmap_quality.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="det1 Fe55 gain map — relative and (provisionally) absolute.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--name", default="fe55_gainmap")
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--max-packets", type=int, default=None)
    ap.add_argument("--fit-min", type=int, default=150)
    ap.add_argument("--min-hits", type=int, default=500,
                    help="pads with fewer hits get no gain measurement")
    ap.add_argument("--threshold-adc", type=float, default=128.0,
                    help="VMM threshold turn-on in ADC counts")
    ap.add_argument("--peak-over-threshold", type=float, default=2.0,
                    help="a pad's photopeak must exceed this multiple of the "
                         "threshold, else its low side is cut and the 'peak' "
                         "is the threshold edge")
    ap.add_argument("--max-sat-frac", type=float, default=0.10,
                    help="drop pads with more than this fraction of hits at "
                         "ADC 1023 (high side clipped)")
    g = ap.add_argument_group("absolute gain (provisional — see --fc-per-adc)")
    g.add_argument("--vmm-gain", type=float, default=DEFAULT_VMM_GAIN,
                   choices=VMM_GAIN_SETTINGS,
                   help="VMM3a preamp gain setting [mV/fC]")
    g.add_argument("--gas", default=DEFAULT_GAS, choices=sorted(GASES),
                   help="gas mixture, sets the primary-electron count")
    g.add_argument("--fc-per-adc", type=float, default=None,
                   help="MEASURED fC per ADC count from a VMM3a test-pulse "
                        "scan. Supersedes --vmm-gain and removes the nominal "
                        "full-scale assumption.")
    args = ap.parse_args(argv)

    paths = fm.resolve_inputs(args.inputs)
    print(f"decoding {len(paths)} capture(s)")
    hists, _ = fm.accumulate(paths, max_packets=args.max_packets)
    live = dc.live_time(paths)
    tab = dc.build_pad_table()
    t = fm.per_pad_table(hists, tab, max(live["live_s"], 1e-9), min_events=30)
    t = fm.mask_hot_pads(t)
    full = dc.full_pad_map()

    keep = (~t["hot"]) & (t["n_hits"] >= args.min_hits)
    t = t[keep].reset_index(drop=True)
    t, a = ps.per_pad_peaks(hists.adc, t, fit_min=args.fit_min)
    key = dc.pad_key(t["vmm"].to_numpy(), t["ch"].to_numpy())

    pk = t["pad_peak_adc"].to_numpy(float)

    # VALIDITY. A pad's "peak" is only a photopeak if the line is actually
    # inside the measurable window at both ends:
    #
    #  * too close to the THRESHOLD and the low side of the line is cut, so the
    #    estimator returns the threshold edge instead of the peak. Measured on
    #    the 720/390 V run: the lowest-gain pads have their ENTIRE ADC
    #    distribution inside 133-171 counts against a 128 threshold -- those are
    #    not low-gain pads, they are pads whose photopeak has fallen below
    #    threshold and is invisible. Left in, they pile up at the threshold and
    #    fake a low-gain tail.
    #  * too much SATURATION and the high side is clipped instead, pulling the
    #    peak down.
    #
    # Both are censoring, not measurement, so those pads are excluded from the
    # map and from the non-uniformity rather than being quietly averaged in.
    censored = pk < args.peak_over_threshold * args.threshold_adc
    saturated = t["sat_frac"].to_numpy(float) > args.max_sat_frac
    t["valid"] = np.isfinite(pk) & ~censored & ~saturated
    t["censored_by_threshold"] = censored
    t["clipped_by_saturation"] = saturated

    ok = t["valid"].to_numpy()
    if ok.sum() < 20:
        print("  FATAL: fewer than 20 pads have a measurable photopeak; the "
              "gain is too close to the threshold for a gain map at all.")
    med = float(np.median(pk[ok])) if ok.any() else np.nan
    t["rel_gain"] = np.where(ok, pk / med, np.nan)
    print(f"  live {live['live_s']:.1f} s, {int(ok.sum())} pads with a peak; "
          f"median peak {med:.1f} ADC")

    # linearity / pedestal check from the escape line
    lin = linearity_from_escape(a, key, pk, args.threshold_adc)
    origin = gas_or_electronics(t)

    title = f"det1 Fe55 gain map — {len(paths)} capture(s)"
    if args.conditions:
        title += f", {args.conditions}"
    os.makedirs(args.out_dir, exist_ok=True)

    f_rel, rms_rel = plot_gain_map(
        t, full, args.name, args.out_dir, title, "rel_gain",
        "relative gain  (pad peak / detector median)", "coolwarm",
        "gainmap_relative", vmin=0.5, vmax=1.5,
        note="needs no electronics calibration — this is the robust product")
    t["peak_adc_valid"] = np.where(ok, pk, np.nan)
    f_pk, _ = plot_gain_map(
        t, full, args.name, args.out_dir, title, "peak_adc_valid",
        "Fe55 photopeak [ADC]", "viridis", "gainmap_peak_adc")
    f_q = plot_quality(t, full, args.name, args.out_dir, title, args.min_hits)
    figs = [f_rel, f_pk, f_q]

    absolute = None
    if args.fc_per_adc or args.vmm_gain:
        gas = GASES[args.gas]
        if args.fc_per_adc:
            k = float(args.fc_per_adc)
            prov = f"MEASURED {k:.4g} fC/ADC (test-pulse)"
            nominal = False
        else:
            k = fc_per_adc(args.vmm_gain)
            prov = (f"NOMINAL: {ADC_FULL_SCALE_MV:.0f} mV full scale / 1023 / "
                    f"{args.vmm_gain:g} mV/fC = {k:.4g} fC/ADC")
            nominal = True
        t["gain"] = absolute_gain(pk, k, gas["n_primary"])
        gmed = float(np.nanmedian(t["gain"]))
        note = (f"{gas['label']}, N_primary = {gas['n_primary']:.0f} e-  |  "
                f"{prov}"
                + ("\nABSOLUTE SCALE IS PROVISIONAL — the ADC full-scale voltage is nominal, so every gain moves in proportion with it."
                   "\nThe SHAPE of the map is unaffected. Fix the scale with a VMM3a test-pulse scan and --fc-per-adc."
                   if nominal else ""))
        f_abs, rms_abs = plot_gain_map(
            t, full, args.name, args.out_dir, title, "gain",
            "absolute gas gain", "viridis", "gainmap_absolute", note=note)
        figs.append(f_abs)
        absolute = {
            "gas": gas["label"], "n_primary_electrons": gas["n_primary"],
            "fc_per_adc": k, "fc_per_adc_provenance": prov,
            "nominal_full_scale_assumed": nominal,
            "vmm_gain_mv_per_fc": args.vmm_gain,
            "gain_median": gmed,
            "gain_p5": float(np.nanpercentile(t["gain"], 5)),
            "gain_p95": float(np.nanpercentile(t["gain"], 95)),
            "charge_at_peak_fC": float(med * k),
        }

    cols = ["channel_id", "vmm", "ch", "connector_N", "half", "pad_cx",
            "pad_cy", "radius", "phi", "n_hits", "pad_peak_adc",
            "pad_fwhm_adc", "pad_res_pct", "rel_gain", "valid",
            "censored_by_threshold", "clipped_by_saturation", "sat_frac"]
    if absolute:
        cols.append("gain")
    t[cols].to_csv(os.path.join(args.out_dir, f"{args.name}_gainmap.csv"),
                   index=False)

    summary = {
        "conditions": args.conditions,
        "inputs": [os.path.basename(p) for p in paths],
        "live_s": live["live_s"],
        "method": "per-pad Fe55 photopeak position; relative gain = peak / "
                  "median(peak). Absolute gain = Q_peak / (N_primary * e).",
        "n_pads_measured": int(ok.sum()),
        "validity": {
            "peak_over_threshold_required": args.peak_over_threshold,
            "threshold_adc": args.threshold_adc,
            "max_sat_frac": args.max_sat_frac,
            "n_censored_by_threshold": int(t["censored_by_threshold"].sum()),
            "n_clipped_by_saturation": int(t["clipped_by_saturation"].sum()),
            "frac_valid": float(ok.mean()),
            "median_peak_over_threshold": float(med / args.threshold_adc)
            if np.isfinite(med) else None},
        "n_pads_instrumented": int(full["instrumented"].sum()),
        "peak_adc": {"median": med,
                     "p5": float(np.nanpercentile(pk, 5)),
                     "p95": float(np.nanpercentile(pk, 95))},
        "non_uniformity_rms_over_mean": rms_rel,
        "resolution_fwhm_pct_median": float(np.nanmedian(t["pad_res_pct"])),
        "linearity_escape_check": lin,
        "gain_origin_test": origin,
        "absolute": absolute,
        "figures": [os.path.basename(x) for x in figs],
    }
    with open(os.path.join(args.out_dir, f"{args.name}_gainmap.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"GAIN MAP  {int(ok.sum())}/{int(full['instrumented'].sum())} pads   "
          f"peak {med:.0f} ADC (p5-p95 {summary['peak_adc']['p5']:.0f}-"
          f"{summary['peak_adc']['p95']:.0f})")
    nc = int(t["censored_by_threshold"].sum())
    ns = int(t["clipped_by_saturation"].sum())
    print(f"VALID     {int(ok.sum())}/{len(t)} pads "
          f"({100 * ok.mean():.0f} %)  |  {nc} censored by the ADC-"
          f"{args.threshold_adc:.0f} threshold, {ns} clipped by saturation")
    print(f"NON-UNIF  RMS/mean = {100 * rms_rel:.1f} %   "
          f"(literature: 3-13 % for a good Micromegas plane)")
    if ok.mean() < 0.85:
        print(f"          ^ WARNING: only {100 * ok.mean():.0f} % of pads are "
              "usable, so this map covers part of the detector and the "
              "non-uniformity is measured on a biased subset.")
    print(f"RESOLUT.  median FWHM/peak {summary['resolution_fwhm_pct_median']:.1f} %")
    if lin.get("usable"):
        print(f"LINEARITY escape/main = {lin['escape_over_main_median']:.3f} "
              f"+- {lin['escape_over_main_std']:.3f} vs expected "
              f"{lin['expected']:.3f} ({lin['deviation_pct']:+.1f} %) on "
              f"{lin['n_pads']} pads"
              + ("  -> linear through the origin"
                 if lin["linear_through_origin"] else
                 "  -> NOT linear through the origin; an ADC offset is implied "
                 "and the absolute gain is biased high"))
    else:
        print(f"LINEARITY not testable: {lin.get('note')}")
    if origin.get("usable"):
        print(f"ORIGIN    spatial-neighbour r = {origin['spatial_neighbour_corr']:+.2f}, "
              f"adjacent-channel r = {origin['adjacent_channel_corr']:+.2f}"
              f"  ->  {origin['verdict']}")
    if absolute:
        print(f"ABSOLUTE  gain {absolute['gain_median']:.3g} "
              f"(p5-p95 {absolute['gain_p5']:.3g}-{absolute['gain_p95']:.3g}), "
              f"Q_peak {absolute['charge_at_peak_fC']:.1f} fC")
        if absolute["nominal_full_scale_assumed"]:
            print("          ^ PROVISIONAL: scales linearly with the assumed "
                  "ADC full scale; take a test-pulse scan to fix it.")
    print(f"{'=' * 70}\nwrote {args.out_dir}/{args.name}_gainmap*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
