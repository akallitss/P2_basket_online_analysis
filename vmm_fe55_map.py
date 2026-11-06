#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_fe55_map.py — Fe55 rate map, source localisation and per-pad gain map for
one P2 basket detector read out by nine VMM hybrids.

What this answers, in order of how urgently you want it at the bench:

  1. WHERE IS THE SOURCE?   Per-pad rate in Hz and Hz/cm^2 on the real fan
     tiles, a 2D Gaussian fit to the spot, and the vector you have to move the
     source by to put it where you want it.
  2. HOW MUCH OF THE SPOT AM I SEEING?  The fraction of the fitted spot that
     lands on instrumented pads, so a spot half off the readout is not read as
     a spot that is merely dim.
  3. WHAT IS THE GAIN THERE?  Per-pad Fe55 photopeak in ADC, and the same
     normalised to the detector median -> relative gain map.
  4. IS THE DATA TRUSTWORTHY?  Cabling cross-check, saturation, threshold bias,
     hot pads, live-time provenance.

Why there is no event building
------------------------------
The pads are ~12 x 11.6 mm (130-144 mm^2). An Fe55 photon converts at 5.9 keV
in a spot far smaller than that, so essentially all of its charge lands on ONE
pad and the single-pad ADC spectrum IS the Fe55 spectrum. That matters more
than convenience: vmm_stations documents that `offset` is stuck in this
firmware, so chip time is only unambiguous within 92.16 us of an SRS marker and
time-clustering across VMMs would be built on sand. Counting hits per pad and
histogramming their ADC needs no timestamps at all.

The residual effect that single-pad spectra DO miss is charge sharing at a pad
boundary: a conversion landing on the seam splits between two pads and appears
in both spectra below the photopeak. It shows up as a low-ADC shoulder, not as
a shift of the peak, and the windowed-median peak estimator is built to ignore
exactly that. Recovering those events would need time-clustering across pads,
which is not implemented here precisely because of the marker caveat above --
so the pad-boundary events are dropped from the gain estimate rather than
mis-measured.

Memory
------
Nothing holds the hits. Each decode chunk is folded straight into a
(32*64, 1024) ADC histogram (8 MB), so a multi-GB capture costs the same as a
small one. This is the same reasoning as vmm_reduce.py's counts.npz.

Usage
-----
  # one capture, or a sub-run / run directory, or a glob
  python3 vmm_fe55_map.py <file.pcapng | dir | 'runs/run_3/**/*.pcapng'>

  # tell it where you want the spot, get the move vector
  python3 vmm_fe55_map.py <input> --target 300 250

  # subtract a source-out run
  python3 vmm_fe55_map.py <input> --background <source_out_dir>

  # confirm the within-half wiring on the first real file
  python3 vmm_fe55_map.py <input> --scan-mapping

  # keep re-rendering as captures land during the run
  python3 vmm_fe55_map.py <run_dir> --watch

@author: ak271430 Alexandra Kallitsopoulou
"""

import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_det1_cabling as dc
import vmm_decode as vd

ADC_BINS = 1024
ADC_MAX = ADC_BINS - 1          # VMM3a ADC is 10 bit; 1023 is saturation
NKEY = 32 * 64                  # (vmm, ch) key space, vmm id is 5 bits

# Fe55 emits 5.90 keV Mn K-alpha. In an argon-based gas a fraction of the
# conversions lose the Ar K-alpha (2.96 keV) out of the gas, leaving a second
# peak at 5.90 - 2.96 = 2.94 keV, i.e. at 0.50 of the main peak's charge.
# Resolving it is the standard proof that the main peak really is the photopeak
# and not a threshold turn-on.
ESCAPE_FRACTION = (5.90 - 2.96) / 5.90


# ==========================================================================
# Accumulation
# ==========================================================================
class PadHistograms:
    """Per-(vmm, ch) hit counts and ADC spectra, summed over any number of
    captures. Additive by construction, so merging runs is an array add."""

    def __init__(self):
        self.adc = np.zeros((NKEY, ADC_BINS), dtype=np.int64)
        self.n_raw = np.zeros(NKEY, dtype=np.int64)      # before ADC/OT cuts
        self.n_ot = np.zeros(NKEY, dtype=np.int64)       # over_threshold set
        self.srs_lo = None
        self.srs_hi = None
        self.n_hits_seen = 0

    def add_chunk(self, cols, min_adc=0, require_ot=False):
        vmm = np.asarray(cols["vmm"], dtype=np.int64)
        ch = np.asarray(cols["ch"], dtype=np.int64)
        adc = np.asarray(cols["adc"], dtype=np.int64)
        ot = np.asarray(cols["over_threshold"], dtype=bool)
        self.n_hits_seen += vmm.size

        key = vmm * 64 + ch
        valid = (key >= 0) & (key < NKEY) & (adc >= 0) & (adc < ADC_BINS)
        self.n_raw += np.bincount(key[valid], minlength=NKEY)
        self.n_ot += np.bincount(key[valid & ot], minlength=NKEY)

        sel = valid & (adc >= min_adc)
        if require_ot:
            sel &= ot
        if sel.any():
            flat = np.bincount(key[sel] * ADC_BINS + adc[sel],
                               minlength=NKEY * ADC_BINS)
            self.adc += flat.reshape(NKEY, ADC_BINS)

        srs = np.asarray(cols["srs_timestamp"], dtype=np.float64)
        srs = srs[srs > 0]
        if srs.size:
            lo, hi = float(srs.min()), float(srs.max())
            self.srs_lo = lo if self.srs_lo is None else min(self.srs_lo, lo)
            self.srs_hi = hi if self.srs_hi is None else max(self.srs_hi, hi)

    @property
    def counts(self):
        """Hits per key after the ADC / over-threshold cuts."""
        return self.adc.sum(axis=1)

    def srs_span_s(self):
        if self.srs_lo is None or self.srs_hi is None:
            return None
        return (self.srs_hi - self.srs_lo) * 25e-9      # nominal 25 ns tick

    def vmm_totals(self):
        return {int(v): int(c) for v, c in
                enumerate(self.n_raw.reshape(32, 64).sum(axis=1)) if c}


def resolve_inputs(patterns):
    """Expand files / directories / globs into a sorted list of pcapng paths.

    A directory is searched recursively, which makes `<run_dir>` work directly:
    the DAQ writes <run>/<sub_run>/raw_daq_data/*.pcapng.
    """
    out = []
    for pat in patterns:
        if os.path.isdir(pat):
            out += glob.glob(os.path.join(pat, "**", "*.pcapng"), recursive=True)
        elif any(c in pat for c in "*?["):
            out += glob.glob(pat, recursive=True)
        elif os.path.isfile(pat):
            out.append(pat)
        else:
            raise FileNotFoundError(pat)
    # A file listed twice would be decoded twice and double every count, while
    # live_time() correctly counts its interval once -> a doubled rate.
    return sorted(set(os.path.abspath(p) for p in out))


def accumulate(paths, min_adc=0, require_ot=False, max_packets=None,
               data_format="SRS", progress=True):
    """Fold every capture into one PadHistograms. Returns (hists, metas)."""
    h = PadHistograms()
    metas = []
    for i, p in enumerate(paths, 1):
        meta = {}
        try:
            for cols in vd.iter_chunks(p, data_format=data_format,
                                       max_packets=max_packets, meta=meta):
                h.add_chunk(cols, min_adc=min_adc, require_ot=require_ot)
        except Exception as e:
            # A capture still being written by dumpcap has a truncated tail.
            # Decoding what is there and moving on beats losing the whole run.
            meta["error"] = f"{type(e).__name__}: {e}"
        meta["path"] = p
        metas.append(meta)
        if progress:
            print(f"  [{i}/{len(paths)}] {os.path.basename(p)}: "
                  f"{meta.get('n_hits', 0):,} hits"
                  + (f"  ({meta['error']})" if "error" in meta else ""))
    return h, metas


# ==========================================================================
# Per-pad estimators, all computed from the ADC histogram
# ==========================================================================
def _hist_median(h, lo=0, hi=ADC_BINS):
    """Median ADC of one pad's spectrum restricted to bins [lo, hi)."""
    seg = h[lo:hi]
    tot = seg.sum()
    if tot <= 0:
        return np.nan
    c = np.cumsum(seg)
    return float(lo + np.searchsorted(c, 0.5 * tot))


def windowed_median_peak(h, fit_min=0, n_iter=4, window=0.4):
    """Robust photopeak of one pad, from its ADC histogram.

    Median above fit_min, then re-median inside +-`window` around the running
    estimate. This is the estimator 18_fe55_spectra.py settled on for the DREAM
    Fe55 scans: it tracks the photopeak through the escape-peak and noise tail
    at the ~50-100 events/pad a short run gives, where a per-pad Gaussian fit
    simply does not converge.
    """
    if h[fit_min:].sum() <= 0:
        return np.nan
    m = _hist_median(h, fit_min)
    for _ in range(n_iter):
        lo = max(fit_min, int((1.0 - window) * m))
        hi = min(ADC_BINS, int((1.0 + window) * m) + 1)
        if hi <= lo or h[lo:hi].sum() <= 0:
            break
        m_new = _hist_median(h, lo, hi)
        if not np.isfinite(m_new) or abs(m_new - m) < 0.5:
            m = m_new
            break
        m = m_new
    return float(m)


def hist_fwhm(h, peak, window=0.6):
    """FWHM in ADC around `peak`, by half-maximum crossings of a smoothed
    spectrum. Returns nan if either side does not cross inside the window."""
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    lo = max(0, int((1.0 - window) * peak))
    hi = min(ADC_BINS, int((1.0 + window) * peak) + 1)
    seg = h[lo:hi].astype(float)
    if seg.sum() < 30 or seg.size < 5:
        return np.nan
    k = max(3, (seg.size // 20) * 2 + 1)                 # odd smoothing kernel
    sm = np.convolve(seg, np.ones(k) / k, mode="same")
    ip = int(round(peak)) - lo
    ip = int(np.clip(ip, 0, sm.size - 1))
    half = 0.5 * sm[ip]
    if half <= 0:
        return np.nan

    def _cross(idx_range):
        prev = ip
        for i in idx_range:
            if sm[i] <= half:
                # linear interpolation between the bracketing bins
                d = sm[prev] - sm[i]
                f = (sm[prev] - half) / d if d else 0.0
                return prev + f * (i - prev)
            prev = i
        return None

    left = _cross(range(ip - 1, -1, -1))
    right = _cross(range(ip + 1, sm.size))
    if left is None or right is None:
        return np.nan
    return float(right - left)


def per_pad_table(hists, tab, live_s, min_events=30, fit_min=0):
    """Per-pad rate and gain, one row per instrumented pad.

    Columns added to a copy of `tab`: n_hits, n_raw, rate_hz, rate_hz_cm2,
    peak_adc, fwhm_adc, res_fwhm_pct, sat_frac, mean_adc, ot_frac.
    """
    t = tab[tab["mapped"]].copy().reset_index(drop=True)
    key = dc.pad_key(t["vmm"].to_numpy(), t["ch"].to_numpy())

    adc = hists.adc[key]                                  # (n_pads, ADC_BINS)
    n = adc.sum(axis=1).astype(float)
    t["n_hits"] = n.astype(np.int64)
    t["n_raw"] = hists.n_raw[key]
    t["ot_frac"] = np.divide(hists.n_ot[key], np.maximum(hists.n_raw[key], 1),
                             dtype=float)

    t["rate_hz"] = n / live_s if live_s > 0 else np.nan
    area_cm2 = t["pad_area"].to_numpy(float) / 100.0      # mm^2 -> cm^2
    t["rate_hz_cm2"] = t["rate_hz"] / area_cm2

    bins = np.arange(ADC_BINS, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        t["mean_adc"] = (adc * bins).sum(axis=1) / n
    t["sat_frac"] = adc[:, ADC_MAX] / np.maximum(n, 1)

    # Plain median of each channel's ADC distribution. Distinct from both
    # mean_adc (which the ADC-1023 pile-up drags upward) and peak_adc (the
    # windowed-median photopeak estimator, which deliberately ignores
    # everything outside the peak window). The median is the robust one-number
    # summary of "how much charge is this channel seeing", and it is defined
    # even when there is no photopeak to find -- which is the situation on
    # every source run so far.
    med = np.full(len(t), np.nan)
    enough_med = n >= max(min_events // 3, 10)
    for i in np.flatnonzero(enough_med):
        med[i] = _hist_median(adc[i])
    t["median_adc"] = med

    peak = np.full(len(t), np.nan)
    fwhm = np.full(len(t), np.nan)
    enough = n >= min_events
    for i in np.flatnonzero(enough):
        peak[i] = windowed_median_peak(adc[i], fit_min=fit_min)
        fwhm[i] = hist_fwhm(adc[i], peak[i])
    t["peak_adc"] = peak
    t["fwhm_adc"] = fwhm
    with np.errstate(invalid="ignore", divide="ignore"):
        t["res_fwhm_pct"] = 100.0 * fwhm / peak

    ref = np.nanmedian(peak) if np.isfinite(peak).any() else np.nan
    t["rel_gain"] = peak / ref
    t.attrs["peak_ref_adc"] = float(ref) if np.isfinite(ref) else None
    return t


def mask_hot_pads(t, ratio=6.0, min_hits=50, k=6):
    """Flag pads far above their SPATIAL neighbours -> boolean column `hot`.

    Same argument as vmm_stations.auto_hot_pads: occupancy varies a lot across
    the fan, so a per-VMM median hides a pad that is hot relative to the pads
    actually next to it. Under a collimated source the variation is far larger
    still -- the point of the run is a bright spot on a dark detector -- so the
    comparison MUST be local. The 6x threshold is the one calibrated on run_25.
    """
    cx = t["pad_cx"].to_numpy(float)
    cy = t["pad_cy"].to_numpy(float)
    n = t["n_hits"].to_numpy(float)
    d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    np.fill_diagonal(d, np.inf)
    nb = np.argsort(d, axis=1)[:, :k]
    med = np.median(n[nb], axis=1)
    t["nb_median"] = med
    t["hot"] = (n >= min_hits) & (n > ratio * np.maximum(med, 1.0))
    return t


def map_roughness(t, subset=None, min_pads=10):
    """Median |n - neighbour median| / (n + med + 1) over illuminated pads.

    The metric vmm_map_rough.py uses to find a scrambled within-half ordering:
    a correct map is locally smooth, a scrambled one is speckled. Restricted to
    pads whose neighbourhood is lit, because with a collimated source most of
    the detector is legitimately empty and would swamp the statistic.

    `subset` (a boolean mask over t's rows) restricts WHICH pads are scored,
    while the neighbour medians still come from the whole map. That distinction
    is what makes a per-half decision possible: scored globally, permuting one
    64-channel half moves a median taken over ~900 pads by ~0.005 and the
    decision drowns in the pads it did not touch. Scored over the half itself,
    the same permutation moves it by ~0.5.
    """
    n = t["n_hits"].to_numpy(float)
    med = t["nb_median"].to_numpy(float)
    in_scope = (np.ones(len(t), bool) if subset is None
                else np.asarray(subset, dtype=bool))
    if not in_scope.any() or np.max(med[in_scope]) <= 0:
        return float("nan")
    # "Lit" is judged against the brightest pad IN SCOPE, not the brightest on
    # the detector. Scaled globally, a half sitting in the tail of a collimated
    # spot has every pad below 5 % of the spot core and nothing left to score,
    # so the scan reports no preference for a half it could perfectly well have
    # judged. A half holding only noise still gets scored -- and gives the same
    # roughness for every ordering, which is exactly the signature the caller's
    # margin test uses to call it undetermined.
    live = in_scope & (med > 0.05 * np.max(med[in_scope])) & (med > 0)
    if live.sum() < min_pads:
        return float("nan")
    return float(np.median(np.abs(n[live] - med[live]) /
                           (n[live] + med[live] + 1.0)))


# ==========================================================================
# Where is the source?
# ==========================================================================
def _gauss2d_counts(p, x, y, area):
    """Predicted COUNTS per pad for a rotated 2D Gaussian plus flat background.

    Counts, not density: the pads differ in area by ~10 %, and fitting the
    density would quietly weight a big pad the same as a small one while its
    Poisson error says otherwise.
    """
    amp, x0, y0, sx, sy, th, bg = p
    ct, st = np.cos(th), np.sin(th)
    dx, dy = x - x0, y - y0
    u = dx * ct + dy * st
    v = -dx * st + dy * ct
    g = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    return area * (amp * g + bg)


def fit_source(t, use="n_hits"):
    """2D Gaussian fit to the spot. Returns a dict of results (never raises).

    Falls back to the intensity-weighted centroid when the fit will not
    converge, which is the honest answer for a spot of only a few pads.
    """
    from scipy.optimize import least_squares

    hot = (t["hot"].to_numpy(bool) if "hot" in t
           else np.zeros(len(t), dtype=bool))
    sub = t[t["mapped"].to_numpy(bool) & ~hot]
    n = sub[use].to_numpy(float)
    x = sub["pad_cx"].to_numpy(float)
    y = sub["pad_cy"].to_numpy(float)
    area = sub["pad_area"].to_numpy(float)
    out = {"n_pads_used": int(len(sub)), "n_hits_used": float(n.sum())}

    if n.sum() <= 0:
        out["error"] = "no hits on instrumented pads"
        return out

    # Density-weighted centroid: area-corrected, so a wider pad does not pull
    # the centroid simply by collecting more.
    w = n / area
    cx = float((w * x).sum() / w.sum())
    cy = float((w * y).sum() / w.sum())
    out["centroid_x_mm"] = cx
    out["centroid_y_mm"] = cy
    out["rms_x_mm"] = float(np.sqrt(max((w * (x - cx) ** 2).sum() / w.sum(), 0)))
    out["rms_y_mm"] = float(np.sqrt(max((w * (y - cy) ** 2).sum() / w.sum(), 0)))

    if len(sub) < 12 or np.count_nonzero(n) < 6:
        out["fit"] = None
        out["fit_note"] = "too few lit pads to fit; centroid only"
        return out

    dens = n / area
    p0 = [max(dens.max() - np.median(dens), 1e-6), cx, cy,
          max(out["rms_x_mm"], 6.0), max(out["rms_y_mm"], 6.0), 0.0,
          max(float(np.median(dens)), 0.0)]
    lo = [0.0, x.min() - 100, y.min() - 100, 1.0, 1.0, -np.pi / 2, 0.0]
    hi = [np.inf, x.max() + 100, y.max() + 100, 1e4, 1e4, np.pi / 2, np.inf]
    sigma = np.sqrt(np.maximum(n, 1.0))

    try:
        r = least_squares(
            lambda p: (_gauss2d_counts(p, x, y, area) - n) / sigma,
            p0, bounds=(lo, hi), max_nfev=4000)
        amp, x0, y0, sx, sy, th, bg = r.x
        ndf = max(len(n) - len(p0), 1)
        out["fit"] = {
            "x0_mm": float(x0), "y0_mm": float(y0),
            "sigma_major_mm": float(max(sx, sy)),
            "sigma_minor_mm": float(min(sx, sy)),
            "theta_deg": float(np.degrees(th)),
            "amplitude_hits_per_mm2": float(amp),
            "background_hits_per_mm2": float(bg),
            "chi2_per_ndf": float(2.0 * r.cost / ndf),
            "converged": bool(r.success),
        }
        out["fit"]["fwhm_major_mm"] = 2.355 * out["fit"]["sigma_major_mm"]
        out["fit"]["fwhm_minor_mm"] = 2.355 * out["fit"]["sigma_minor_mm"]
        out["containment"] = _containment(x, y, area, (amp, x0, y0, sx, sy, th))
    except Exception as e:
        out["fit"] = None
        out["fit_note"] = f"fit failed: {type(e).__name__}: {e}"
    return out


def _containment(x, y, area, gp, n_sigma=3.0, n_grid=201):
    """Fraction of the fitted Gaussian's volume landing on instrumented pads.

    The number that distinguishes a dim source from a source half off the
    readout -- and therefore the number that tells you whether the fitted
    centre can be believed at all. A spot straddling the edge of the
    instrumented sectors gets its centre pulled inwards, and only this flags it.
    """
    amp, x0, y0, sx, sy, th = gp
    if amp <= 0:
        return None
    rad = n_sigma * max(sx, sy)
    gx = np.linspace(x0 - rad, x0 + rad, n_grid)
    gy = np.linspace(y0 - rad, y0 + rad, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    ct, st = np.cos(th), np.sin(th)
    dx, dy = GX - x0, GY - y0
    u = dx * ct + dy * st
    v = -dx * st + dy * ct
    g = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))

    # "On an instrumented pad" ~ within half a pad pitch of a pad centre. Exact
    # tile containment is not worth it: the tiles tile the plane, so the two
    # agree everywhere except within a fraction of a mm of a pad edge.
    pitch = float(np.sqrt(np.median(area)))
    d2 = ((GX[..., None] - x[None, None, :]) ** 2 +
          (GY[..., None] - y[None, None, :]) ** 2)
    inside = d2.min(axis=2) <= (0.5 * pitch * 1.15) ** 2
    tot = g.sum()
    return float(g[inside].sum() / tot) if tot > 0 else None


def move_advice(src, target_xy, t):
    """Vector from the measured spot to where you want it, in x/y and r/phi.

    Polar as well as Cartesian because this detector is a fan: on a wedge, "out
    by 20 mm and round by 2 degrees" is what the stage actually does, and a
    pure dx/dy hides how far round the fan you are asking to go.
    """
    # MOVE is the one number that gets acted on, so it must come from the
    # estimator that is actually valid here. A 2D Gaussian fit is the better
    # estimator for a compact spot, but it returns a confident centre for
    # distributions it does not describe: on the first real capture it gave
    # chi2/ndf = 35500 and a centre 166 mm away from the centroid. Above a
    # chi2/ndf of 10 the fit is not describing the data, and the
    # rate-weighted centroid -- which needs no shape assumption -- wins.
    fit = src.get("fit")
    chi2 = (fit or {}).get("chi2_per_ndf")
    use_fit = bool(fit) and (chi2 is None or chi2 <= 10)
    x = fit["x0_mm"] if use_fit else src.get("centroid_x_mm")
    y = fit["y0_mm"] if use_fit else src.get("centroid_y_mm")
    if x is None:
        return None
    if target_xy is None:
        live = t[t["mapped"]]
        w = live["pad_area"].to_numpy(float)
        tx = float((live["pad_cx"] * w).sum() / w.sum())
        ty = float((live["pad_cy"] * w).sum() / w.sum())
        target_name = "centre of the instrumented area"
    else:
        tx, ty = float(target_xy[0]), float(target_xy[1])
        target_name = "requested target"

    r0, r1 = np.hypot(x, y), np.hypot(tx, ty)
    p0, p1 = np.arctan2(y, x), np.arctan2(ty, tx)
    return {
        "source_x_mm": float(x), "source_y_mm": float(y),
        "source_r_mm": float(r0), "source_phi_deg": float(np.degrees(p0)),
        "target": target_name,
        "target_x_mm": tx, "target_y_mm": ty,
        "dx_mm": float(tx - x), "dy_mm": float(ty - y),
        "distance_mm": float(np.hypot(tx - x, ty - y)),
        "dr_mm": float(r1 - r0),
        "dphi_deg": float(np.degrees(p1 - p0)),
        "estimator": "gaussian fit" if use_fit else (
            "rate-weighted centroid (Gaussian fit rejected: "
            f"chi2/ndf = {chi2:.0f})" if chi2 is not None and chi2 > 10
            else "rate-weighted centroid"),
    }


# ==========================================================================
# Spectrum fit (whole detector / one region)
# ==========================================================================
def fit_fe55_spectrum(h, fit_min=0):
    """Main photopeak + Ar escape peak + exponential background.

    Returns a dict with peak_adc, sigma_adc, resolution_fwhm_pct, the escape
    peak, and the model curve for drawing. Resolving the escape peak at the
    expected 0.50 x charge ratio is the check that the main peak really is the
    5.9 keV photopeak rather than a threshold turn-on, so it is fitted rather
    than assumed.
    """
    from scipy.optimize import curve_fit

    x = np.arange(ADC_BINS, dtype=float)
    y = np.asarray(h, dtype=float)
    m = x >= fit_min
    if y[m].sum() < 500:
        return {"error": f"only {int(y[m].sum())} hits above ADC {fit_min}"}

    seed = windowed_median_peak(y.astype(np.int64), fit_min=fit_min)
    if not np.isfinite(seed) or seed <= 0:
        return {"error": "no photopeak seed"}

    # Where the VMM threshold cuts in: the lowest ADC carrying a substantial
    # fraction of the peak. Everything below it is not measured, it is absent.
    #
    # The top bin is EXCLUDED from the peak height. ADC 1023 is saturation
    # pile-up, not a spectral feature: on the 750/420 V run it held 14 % of all
    # hits and was 100x the continuum, so including it put the "20 % of peak"
    # level above every real bin and drove turn_on to 1023, collapsing the fit
    # window. Saturation is reported separately; it must not define the
    # threshold.
    k0 = max(fit_min, 1)
    seg = y[k0:ADC_MAX]                       # drop the saturation bin
    peak_h = float(seg.max()) if seg.size else 0.0
    above = np.flatnonzero(seg >= 0.2 * peak_h) if peak_h > 0 else np.array([])
    turn_on = float(above[0] + k0) if above.size else float(fit_min)

    # If the escape peak would land below the threshold it is not in the data,
    # and fitting a Gaussian to a region the hardware never recorded leaves the
    # model unconstrained -- which is exactly why this failed to converge on the
    # 720/390 V run, where the peak fell to ~205 ADC with the turn-on at ~135.
    esc_visible = ESCAPE_FRACTION * seed > 1.05 * turn_on

    def model_full(xx, a1, mu, sig, a2, sig2, b, tau):
        main = a1 * np.exp(-0.5 * ((xx - mu) / sig) ** 2)
        esc = a2 * np.exp(-0.5 * ((xx - ESCAPE_FRACTION * mu) / sig2) ** 2)
        return main + esc + b * np.exp(-xx / max(tau, 1.0))

    def model_noesc(xx, a1, mu, sig, b, tau):
        return (a1 * np.exp(-0.5 * ((xx - mu) / sig) ** 2)
                + b * np.exp(-xx / max(tau, 1.0)))

    # Never fit below the turn-on: those bins are empty by construction and
    # would drag the background term.
    lo = int(max(fit_min, 0.35 * seed, turn_on))
    hi = min(ADC_BINS, int(1.6 * seed) + 1)
    if hi - lo < 10:
        return {"error": "fit window collapsed against the threshold",
                "seed_adc": float(seed), "turn_on_adc": turn_on}
    xf, yf = x[lo:hi], y[lo:hi]
    if yf.sum() < 300:
        return {"error": "too few hits in the fit window",
                "seed_adc": float(seed), "turn_on_adc": turn_on}

    peak_amp = float(yf.max())
    sw = np.sqrt(np.maximum(yf, 1.0))
    attempts = []
    if esc_visible:
        attempts.append(("main+escape+bkg", model_full,
                         [peak_amp, seed, 0.12 * seed, 0.15 * peak_amp,
                          0.15 * seed, peak_amp, max(seed / 2.0, 1.0)],
                         ([0, 0.5 * seed, 1.0, 0, 1.0, 0, 1.0],
                          [np.inf, 1.6 * seed, 0.6 * seed, np.inf,
                           0.6 * seed, np.inf, np.inf])))
    attempts.append(("main+bkg (escape below threshold)", model_noesc,
                     [peak_amp, seed, 0.12 * seed, peak_amp,
                      max(seed / 2.0, 1.0)],
                     ([0, 0.5 * seed, 1.0, 0, 1.0],
                      [np.inf, 1.6 * seed, 0.6 * seed, np.inf, np.inf])))

    popt = which = None
    last_err = None
    for name, mdl, p0, bounds in attempts:
        try:
            popt, _ = curve_fit(mdl, xf, yf, p0=p0, bounds=bounds, sigma=sw,
                                maxfev=40000)
            which, model = name, mdl
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    if popt is None:
        return {"error": last_err, "seed_adc": float(seed),
                "turn_on_adc": turn_on}

    if len(popt) == 7:
        a1, mu, sig, a2, sig2, b, tau = popt
    else:
        a1, mu, sig, b, tau = popt
        a2 = sig2 = 0.0
    esc_area = a2 * sig2
    main_area = a1 * sig

    # IS IT ACTUALLY A PEAK? A Gaussian will fit a flat continuum perfectly
    # happily and report a confident "photopeak" in the middle of it -- which is
    # exactly what this returned on the first real capture (a flat spectrum from
    # the threshold to saturation gave "peak 575 ADC"). Prominence is the
    # fitted Gaussian's height at mu against the total there, so a bump sitting
    # on its own background scores high and a fit to featureless continuum
    # scores near zero.
    bkg_at_mu = float(b * np.exp(-mu / max(tau, 1.0)))
    prominence = float(a1 / (a1 + bkg_at_mu)) if (a1 + bkg_at_mu) > 0 else 0.0

    return {
        "peak_adc": float(mu),
        "sigma_adc": float(sig),
        "resolution_fwhm_pct": float(100.0 * 2.355 * sig / mu) if mu else None,
        "escape_peak_adc": float(ESCAPE_FRACTION * mu),
        "escape_to_main_area": (float(esc_area / main_area)
                                if (main_area and esc_visible) else None),
        "escape_visible": bool(esc_visible),
        "peak_prominence": prominence,
        "model": which,
        "turn_on_adc": turn_on,
        # How close the peak sits to the threshold. Below ~2 the low-energy
        # side of the line is being cut by the discriminator, so the fitted
        # width is a lower limit and the peak position is pulled upward.
        "peak_over_turn_on": float(mu / turn_on) if turn_on > 0 else None,
        "fit_range_adc": [int(lo), int(hi)],
        "seed_adc": float(seed),
        "_curve": (xf, model(xf, *popt)),
    }


# ==========================================================================
# Plots
# ==========================================================================
def _tile_axes(ax, verts_bg):
    ax.add_collection(PolyCollection(verts_bg, facecolors="0.93",
                                     edgecolors="0.82", linewidths=0.25))
    ax.set_xlabel("pad_cx [mm]")
    ax.set_ylabel("pad_cy [mm]")
    ax.set_aspect("equal")


def _draw_map(fig, ax, verts, values, label, cmap="inferno", log=False,
              vmin=None):
    ok = np.isfinite(values) & (values > 0 if log else np.isfinite(values))
    if not ok.any():
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return None
    norm = LogNorm(vmin=vmin or max(values[ok].min(), 1e-6)) if log else None
    pc = PolyCollection(verts[ok], array=values[ok], cmap=cmap, norm=norm,
                        edgecolors="face", linewidths=0.2)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, label=label, fraction=0.046, pad=0.03)
    return pc


def _overlay_source(ax, src):
    fit = src.get("fit")
    if fit:
        ax.add_patch(Ellipse((fit["x0_mm"], fit["y0_mm"]),
                             2 * 2.355 * fit["sigma_major_mm"],
                             2 * 2.355 * fit["sigma_minor_mm"],
                             angle=fit["theta_deg"], fill=False,
                             edgecolor="cyan", lw=1.6, ls="--"))
        ax.plot(fit["x0_mm"], fit["y0_mm"], "x", color="cyan", ms=11, mew=2.2,
                label="Gaussian centre")
    if "centroid_x_mm" in src:
        ax.plot(src["centroid_x_mm"], src["centroid_y_mm"], "+", color="lime",
                ms=13, mew=2.0, label="centroid")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)


def plot_rate_map(t, full, src, move, live, base, out_dir, title):
    """Rate map in Hz/cm^2, linear and log, with the fitted spot overlaid."""
    verts_bg = dc.pad_polygons(full)
    verts = dc.pad_polygons(t)
    rate = t["rate_hz_cm2"].to_numpy(float).copy()
    hot = t["hot"].to_numpy(bool) if "hot" in t else np.zeros(len(t), bool)
    rate_masked = rate.copy()
    rate_masked[hot] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    for ax, log in ((axes[0], False), (axes[1], True)):
        _tile_axes(ax, verts_bg)
        _draw_map(fig, ax, verts, rate_masked,
                  "rate [Hz/cm$^2$]" + (" (log)" if log else ""), log=log)
        if hot.any():
            ax.add_collection(PolyCollection(verts[hot], facecolors="none",
                                             edgecolors="red", linewidths=0.9))
        _overlay_source(ax, src)
        ax.autoscale_view()
    axes[0].set_title(f"linear — grey = not read out"
                      + (f", red = {int(hot.sum())} hot pads masked"
                         if hot.any() else ""))
    axes[1].set_title("log")

    lines = [f"live time {live['live_s']:.1f} s  ({live['method']})",
             f"total {int(t['n_hits'].sum()):,} hits  "
             f"= {t['rate_hz'].sum():.1f} Hz on {int((t['n_hits'] > 0).sum())}"
             f"/{len(t)} instrumented pads"]
    if move:
        lines.append(f"spot at ({move['source_x_mm']:.1f}, "
                     f"{move['source_y_mm']:.1f}) mm  "
                     f"= (r {move['source_r_mm']:.1f} mm, "
                     f"phi {move['source_phi_deg']:.2f} deg)")
        lines.append(f"MOVE dx={move['dx_mm']:+.1f}  dy={move['dy_mm']:+.1f} mm"
                     f"   (dr={move['dr_mm']:+.1f} mm, "
                     f"dphi={move['dphi_deg']:+.2f} deg)  "
                     f"to reach the {move['target']}")
    c = src.get("containment")
    if c is not None:
        lines.append(f"{100 * c:.0f} % of the fitted spot lands on instrumented"
                     " pads" + ("  <-- SPOT IS CLIPPED" if c < 0.9 else ""))
    fig.suptitle(title + "\n" + "\n".join(lines), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = os.path.join(out_dir, f"{base}_rate_map.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_source_profile(t, src, move, base, out_dir, title):
    """x / y and r / phi projections of the rate DENSITY, with the centre marked.

    Each bin is sum(rate) / sum(pad area), not the sum of the pad rates. The fan
    puts far more pads at large radius than small, so a plain sum plots the
    pad-count distribution with the source modulating it faintly on top -- the
    radial turnover that is actually in the data vanishes entirely. Dividing by
    the area actually read out in each bin removes the geometry and leaves the
    illumination.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    hot = t["hot"].to_numpy(bool) if "hot" in t else np.zeros(len(t), bool)
    keep = ~hot
    rate = t["rate_hz"].to_numpy(float)[keep]
    area_cm2 = t["pad_area"].to_numpy(float)[keep] / 100.0
    fit = src.get("fit")

    specs = [
        ("pad_cx", "x [mm]", src.get("centroid_x_mm")),
        ("pad_cy", "y [mm]", src.get("centroid_y_mm")),
        ("radius", "r [mm]", None),
        (None, "phi [deg]", None),
    ]
    for ax, (col, lab, mark) in zip(axes.ravel(), specs):
        v = (np.degrees(t["phi"].to_numpy(float)) if col is None
             else t[col].to_numpy(float))[keep]
        if not np.isfinite(v).any() or rate.sum() <= 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue
        bins = np.linspace(np.nanmin(v), np.nanmax(v), 40)
        num, _ = np.histogram(v, bins=bins, weights=rate)
        den, _ = np.histogram(v, bins=bins, weights=area_cm2)
        dens = np.divide(num, den, out=np.full_like(num, np.nan),
                         where=den > 0)
        ctr = 0.5 * (bins[1:] + bins[:-1])
        ax.step(ctr, dens, where="mid", color="0.25", lw=1.4)
        ax.fill_between(ctr, 0, dens, step="mid", color="0.75")
        # Rate-weighted mean of this projection: the estimator that survives a
        # non-Gaussian spot, which the fitted centre does not.
        ok = np.isfinite(dens) & (den > 0)
        if ok.any() and num[ok].sum() > 0:
            mean_v = float((ctr[ok] * num[ok]).sum() / num[ok].sum())
            ax.axvline(mean_v, color="seagreen", ls="-", lw=1.5,
                       label=f"rate-weighted mean {mean_v:.1f}")
        if mark is not None:
            ax.axvline(mark, color="crimson", ls="--", lw=1.4,
                       label=f"centroid {mark:.1f}")
        ax.legend(fontsize=7)
        ax.set_xlabel(lab)
        ax.set_ylabel("rate density [Hz/cm$^2$]")
        ax.set_ylim(bottom=0)
    fig.suptitle(title + " — source projections", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out_dir, f"{base}_source_profile.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_gain_map(t, full, base, out_dir, title, min_events):
    """Per-pad Fe55 photopeak, relative gain, statistics and resolution."""
    verts_bg = dc.pad_polygons(full)
    verts = dc.pad_polygons(t)
    peak = t["peak_adc"].to_numpy(float)
    rel = t["rel_gain"].to_numpy(float)
    nev = t["n_hits"].to_numpy(float)
    res = t["res_fwhm_pct"].to_numpy(float)

    ok = np.isfinite(peak)
    spread = (float(np.nanstd(peak[ok]) / np.nanmean(peak[ok]))
              if ok.sum() > 1 else np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12))
    panels = [
        (axes[0, 0], peak, "Fe55 photopeak [ADC]", "viridis", False,
         f"per-pad photopeak — grey = <{min_events} hits\n"
         f"{int(ok.sum())}/{len(t)} pads, spread std/mean = {spread:.3f}"),
        (axes[0, 1], rel, "relative gain (peak / detector median)", "coolwarm",
         False, "relative gain"),
        (axes[1, 0], nev, "hits per pad (log)", "inferno", True,
         "per-pad statistics — this is what limits the gain map"),
        (axes[1, 1], res, "FWHM / peak [%]", "magma", False,
         "energy resolution"),
    ]
    for ax, vals, lab, cmap, log, ttl in panels:
        _tile_axes(ax, verts_bg)
        _draw_map(fig, ax, verts, vals, lab, cmap=cmap, log=log)
        ax.set_title(ttl, fontsize=9)
        ax.autoscale_view()
    fig.suptitle(title + " — Fe55 gain map", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out_dir, f"{base}_gain_map.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def channel_census(t, low_frac=0.10):
    """Channels reading far below the detector median, grouped into runs.

    Grouped into CONTIGUOUS runs on purpose: scattered single channels are
    normal attrition, but a block of consecutive channels on one VMM is a
    connector or ribbon fault, and the difference is the whole diagnosis. This
    is invisible in pad space -- the strip -> pad permutation scatters a
    contiguous electronics block across the fan.
    """
    med = float(t["rate_hz"].median())
    if med <= 0:
        return {"detector_median_rate_hz": med, "low_channels": {}}
    low = t[t["rate_hz"] < low_frac * med]
    out, blocks = {}, []
    for v, g in low.groupby("vmm"):
        chans = sorted(int(c) for c in g["ch"])
        runs, start, prev = [], chans[0], chans[0]
        for c in chans[1:]:
            if c == prev + 1:
                prev = c
            else:
                runs.append((start, prev))
                start = prev = c
        runs.append((start, prev))
        conn = int(g["connector_N"].iloc[0])
        half = str(g["half"].iloc[0])
        out[int(v)] = {"connector_N": conn, "half": half,
                       "n_low": len(chans), "channels": chans,
                       "runs": [[a, b] for a, b in runs]}
        for a, b in runs:
            if b - a + 1 >= 4:          # a block, not attrition
                blocks.append({"vmm": int(v), "connector_N": conn,
                               "half": half, "ch_from": a, "ch_to": b,
                               "n_channels": b - a + 1})
    return {"detector_median_rate_hz": med,
            "low_threshold_frac_of_median": low_frac,
            "n_low_channels": int(len(low)),
            "n_channels": int(len(t)),
            "low_channels": out,
            "contiguous_blocks": blocks}


def plot_per_channel(t, base, out_dir, title, column="rate_hz",
                     label="rate [Hz]", suffix="rate_vs_channel",
                     log_heatmap=True, hlines=None, min_hits=1):
    """A per-channel quantity in ELECTRONICS space: one panel per connector,
    plus a VMM x channel heatmap of the whole readout.

    The pad-space maps answer "where is the source"; this answers "is any
    channel misbehaving", and the two cannot substitute for each other. A dead
    channel, a hot one, a threshold step between the two VMMs of a hybrid, or a
    ribbon half reading at the wrong level are all obvious here and nearly
    invisible on the fan, because the strip -> pad mapping is an interleaved
    permutation that scatters an electronics-ordered defect across the tiles.

    x is the VMM channel (0-63) rather than the strip, so a feature lines up
    with the chip pin that causes it. The two VMMs of each connector are drawn
    separately: bottom reads strips 1-64, top reads 65-128.

    `column` selects what is plotted, so rate and median ADC share one
    implementation and therefore one layout -- the two figures can be laid side
    by side and read against each other channel for channel.
    """
    conns = sorted(t["connector_N"].unique())
    hot = t["hot"].to_numpy(bool) if "hot" in t else np.zeros(len(t), bool)
    vals_all = t[column].to_numpy(float).copy()
    # A quantity like median ADC is meaningless on a channel with a handful of
    # hits; blank it rather than plotting noise as if it were a measurement.
    vals_all[t["n_hits"].to_numpy() < min_hits] = np.nan

    n_cols = 2
    n_rows = int(np.ceil(len(conns) / n_cols))
    fig = plt.figure(figsize=(14.5, 3.6 + 2.5 * n_rows))
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1.5] + [1] * n_rows,
                          hspace=0.55, wspace=0.18)

    # --- VMM x channel heatmap: the whole readout on one axis ---
    ax = fig.add_subplot(gs[0, :])
    vmms = sorted(t["vmm"].unique())
    grid = np.full((len(vmms), 64), np.nan)
    vmm_arr = t["vmm"].to_numpy()
    ch_arr = t["ch"].to_numpy(int)
    for i, v in enumerate(vmms):
        m = vmm_arr == v
        grid[i, ch_arr[m]] = vals_all[m]
    finite = np.isfinite(grid) & (grid > 0)
    norm = None
    if finite.any() and log_heatmap:
        norm = LogNorm(vmin=max(grid[finite].min(), 1e-3),
                       vmax=grid[finite].max())
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="inferno",
                   interpolation="nearest", norm=norm,
                   extent=(-0.5, 63.5, -0.5, len(vmms) - 0.5))
    fig.colorbar(im, ax=ax, label=label + (" (log)" if log_heatmap else ""),
                 fraction=0.03, pad=0.01)
    ax.set_yticks(range(len(vmms)))
    ax.set_yticklabels([f"VMM {v} (c{int(t.loc[t.vmm == v, 'connector_N'].iloc[0])}"
                        f"{t.loc[t.vmm == v, 'half'].iloc[0][0]})" for v in vmms],
                       fontsize=6.5)
    ax.set_xlabel("VMM channel")
    ax.set_title(f"{label} per channel, whole readout — white = no measurement",
                 fontsize=9)

    # Shared y-scale, so a connector reading low is visible as such rather than
    # being autoscaled up to look like all the others.
    good = vals_all[~hot & np.isfinite(vals_all)]
    ymax = float(np.nanpercentile(good, 99.5)) * 1.25 if good.size else 1.0

    for k, n in enumerate(conns):
        ax = fig.add_subplot(gs[1 + k // n_cols, k % n_cols])
        sub_m = t["connector_N"].to_numpy() == n
        for half, colour in (("bot", "#1f77b4"), ("top", "#d62728")):
            m = sub_m & (t["half"].to_numpy() == half)
            if not m.any():
                continue
            order = np.argsort(ch_arr[m])
            cc = ch_arr[m][order]
            vv = vals_all[m][order]
            vmm_id = int(vmm_arr[m][0])
            ax.step(cc, vv, where="mid", color=colour, lw=1.1,
                    label=f"VMM {vmm_id} ({half}, strips "
                          f"{'1-64' if half == 'bot' else '65-128'})")
            blank = cc[~np.isfinite(vv)]
            if blank.size:
                ax.plot(blank, np.zeros(blank.size), "v", color=colour, ms=4,
                        mew=0)
        hm = sub_m & hot
        if hm.any():
            ax.plot(ch_arr[hm], vals_all[hm], "o", mfc="none", mec="red",
                    ms=7, mew=1.2, label="hot")
        for hy, hlab, hcol in (hlines or []):
            if hy is not None:
                ax.axhline(hy, color=hcol, ls="--", lw=1.0, label=hlab)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-1, 64)
        ax.set_xlabel("VMM channel")
        ax.set_ylabel(label)
        v_conn = vals_all[sub_m]
        med = float(np.nanmedian(v_conn)) if np.isfinite(v_conn).any() else np.nan
        n_blank = int((~np.isfinite(v_conn)).sum())
        ax.set_title(f"connector {int(n)} — median {med:.1f}"
                     + (f", {n_blank} channel(s) without a measurement"
                        if n_blank else ""), fontsize=8.5)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.85)

    fig.suptitle(f"{title} — {label} per channel (electronics space)",
                 fontsize=11)
    p = os.path.join(out_dir, f"{base}_{suffix}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_rate_vs_channel(t, base, out_dir, title):
    """Per-channel RATE — see plot_per_channel."""
    return plot_per_channel(t, base, out_dir, title, column="rate_hz",
                            label="rate [Hz]", suffix="rate_vs_channel",
                            log_heatmap=True)


def plot_adc_vs_channel(t, base, out_dir, title, spec=None):
    """Per-channel MEDIAN ADC — see plot_per_channel.

    The companion to the rate view: rate says how often a channel fires, this
    says how big its signals are. A channel low here with a normal rate is a
    gain or threshold problem, not a connection problem, and the two figures
    share a layout so they can be read against each other channel for channel.

    The threshold turn-on from the spectrum fit is drawn when available: a
    channel whose median sits on it is being cut by the discriminator, and its
    value is then a property of the threshold rather than of the gas gain.
    """
    hl = []
    if spec and spec.get("turn_on_adc"):
        hl.append((spec["turn_on_adc"],
                   f"threshold turn-on {spec['turn_on_adc']:.0f}", "steelblue"))
    if spec and spec.get("peak_adc"):
        hl.append((spec["peak_adc"],
                   f"detector photopeak {spec['peak_adc']:.0f}", "seagreen"))
    return plot_per_channel(t, base, out_dir, title, column="median_adc",
                            label="median ADC", suffix="adc_vs_channel",
                            log_heatmap=False, hlines=hl, min_hits=20)


def plot_spectra(hists, t, spec, base, out_dir, title, fit_min):
    """Whole-detector Fe55 spectrum with the fit, plus one curve per connector
    and the saturation / threshold diagnostics."""
    key = dc.pad_key(t["vmm"].to_numpy(), t["ch"].to_numpy())
    lit = t["n_hits"].to_numpy() > 0
    hot = t["hot"].to_numpy(bool) if "hot" in t else np.zeros(len(t), bool)
    total = hists.adc[key[lit & ~hot]].sum(axis=0)
    x = np.arange(ADC_BINS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    ax.step(x, total, where="mid", color="0.25", lw=1.0, label="all pads")
    if spec and "peak_adc" in spec:
        xf, yf = spec["_curve"]
        ax.plot(xf, yf, "r-", lw=1.6, label=spec.get("model", "fit"))
        ax.axvline(spec["peak_adc"], color="crimson", ls="--", lw=1.2)
        esc = spec.get("escape_to_main_area")
        if esc is not None:
            ax.axvline(spec["escape_peak_adc"], color="darkorange", ls=":",
                       lw=1.2)
        if spec.get("turn_on_adc"):
            ax.axvline(spec["turn_on_adc"], color="steelblue", ls="-.", lw=1.2,
                       label=f"threshold turn-on {spec['turn_on_adc']:.0f}")
        ttl = (f"peak {spec['peak_adc']:.1f} ADC, "
               f"FWHM/peak {spec['resolution_fwhm_pct']:.1f} %, ")
        ttl += (f"escape/main area {esc:.2f}" if esc is not None
                else f"escape peak (ADC {spec['escape_peak_adc']:.0f}) is "
                     "BELOW THRESHOLD — cannot confirm the line")
        ax.set_title(ttl, fontsize=9)
    elif spec:
        ax.set_title(f"no photopeak fit: {spec.get('error')}", fontsize=9)
    if fit_min > 0:
        ax.axvspan(0, fit_min, color="0.85", label=f"excluded (<{fit_min})")
    ax.set_yscale("log")
    # The fitted Gaussian falls to ~1e-7 in its tails; left alone that stretches
    # a log axis over nine decades and squashes the peak into a line.
    if total.max() > 0:
        ax.set_ylim(0.5, 3.0 * total.max())
    ax.set_xlim(0, max(int(1.3 * spec["peak_adc"]) if spec and "peak_adc" in spec
                       else ADC_BINS, 100))
    ax.set_xlabel("ADC")
    ax.set_ylabel("hits")
    ax.legend(fontsize=8)

    ax = axes[1]
    for n, grp in t[lit & ~hot].groupby("connector_N"):
        hh = hists.adc[dc.pad_key(grp["vmm"].to_numpy(), grp["ch"].to_numpy())]
        s = hh.sum(axis=0)
        if s.sum() > 0:
            ax.step(x, s / s.sum(), where="mid", lw=1.0,
                    label=f"connector {int(n)} ({int(s.sum()):,})")
    ax.set_yscale("log")
    ax.set_xlabel("ADC")
    ax.set_ylabel("normalised")
    ax.set_title("per connector — a shifted peak is a gain difference,\n"
                 "a shifted turn-on is a threshold difference", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle(title + " — Fe55 spectra", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out_dir, f"{base}_spectra.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


# ==========================================================================
# Mapping scan
# ==========================================================================
def scan_mapping(paths, args, strategies=("linear", "reverse", "pairswap"),
                 min_hits_per_half=2000, min_gain=0.02):
    """Greedy per-half within-ordering scan, judged by speckle.

    Mirrors vmm_map_rough.py: fix everything else at the current best, try each
    strategy on one half, keep the winner. Two passes so an early pick can be
    revised. Decoding is done ONCE and the histograms reused, because only the
    (vmm,ch) -> pad assignment changes between trials, not the data.

    A COLLIMATED SOURCE ONLY LIGHTS A FEW CONNECTORS, and speckle cannot judge
    a half that has no hits: the greedy step would still return a winner, and
    it would be noise. So every half is classified as determined or not, on two
    counts -- enough hits to speak, and a large enough roughness gap between
    the best and second-best ordering -- and only determined halves produce an
    override. An undetermined half is reported as such rather than silently
    given the default.
    """
    hists, _ = accumulate(paths, min_adc=args.min_adc,
                          require_ot=args.require_ot,
                          max_packets=args.max_packets)
    live = dc.live_time(paths, capture_duration_s=args.capture_duration)
    best, scores_of = {}, {}

    def rough(over, half=None):
        tab = dc.build_pad_table(strategy=args.strategy, overrides=over,
                                 disconnected=tuple(args.disconnected))
        t = per_pad_table(hists, tab, max(live["live_s"], 1e-9),
                          min_events=10 ** 9)      # peaks not needed here
        t = mask_hot_pads(t, ratio=args.hot_ratio)
        sub = None
        if half is not None:
            sub = ((t["connector_N"] == half[0]) & (t["half"] == half[1])
                   ).to_numpy()
        return map_roughness(t, subset=sub)

    halves_map = dc.vmm_half_map(tuple(args.disconnected))
    halves = sorted({(n, h) for (n, h) in halves_map.values()})
    hits_of = {}
    for vmm, (n, h) in halves_map.items():
        lo = vmm * 64
        hits_of[(n, h)] = int(hists.counts[lo:lo + 64].sum())

    for _ in range(2):
        for nh in halves:
            sc = {}
            for s in strategies:
                trial = dict(best)
                trial[nh] = s
                sc[s] = rough(trial, half=nh)
            if not any(np.isfinite(v) for v in sc.values()):
                best[nh] = args.strategy      # nothing to judge on; keep default
            else:
                best[nh] = min(sc, key=lambda k: (np.inf if np.isnan(sc[k])
                                                  else sc[k]))
            scores_of[nh] = sc

    base_rough = rough({})
    final_rough = rough(best)
    print(f"\nmapping scan: roughness {base_rough:.3f} -> {final_rough:.3f}\n")

    determined, undetermined = {}, {}
    print(f"{'half':>12} {'hits':>10}  {'pick':>8}  {'margin':>7}  verdict")
    for nh in halves:
        sc = scores_of[nh]
        vals = sorted(v for v in sc.values() if np.isfinite(v))
        margin = (vals[1] - vals[0]) if len(vals) > 1 else 0.0
        n = hits_of[nh]
        ok = n >= min_hits_per_half and margin >= min_gain
        (determined if ok else undetermined)[nh] = best[nh]
        why = "" if ok else (f"only {n} hits" if n < min_hits_per_half
                             else f"orderings differ by {margin:.3f} < {min_gain}")
        print(f"  c{nh[0]}_{nh[1]:<8} {n:>10,}  {best[nh]:>8}  {margin:7.3f}  "
              + ("determined" if ok else f"UNDETERMINED ({why})"))

    changed = {k: v for k, v in sorted(determined.items()) if v != args.strategy}
    print()
    if changed:
        print("Paste into vmm_det1_cabling.STRATEGY_OVERRIDES:")
        print("STRATEGY_OVERRIDES = {")
        for k, v in changed.items():
            print(f"    {k!r}: {v!r},")
        print("}")
    elif determined:
        print(f"every determined half agrees with WITHIN_STRATEGY = "
              f"{args.strategy!r}; no overrides needed.")
    if undetermined:
        print(f"\n{len(undetermined)}/{len(halves)} halves are UNDETERMINED and "
              "were left at the default. Speckle cannot judge a half the source\n"
              "never lit. Move the collimated source over each connector in "
              "turn and re-run the scan on the merged set of captures\n"
              "(vmm_fe55_map.py <all captures> --scan-mapping) before trusting "
              "the mapping across the whole fan.")
    return determined


# ==========================================================================
# Driver
# ==========================================================================
def analyse(paths, args, out_dir, base, title):
    print(f"decoding {len(paths)} capture(s)")
    hists, metas = accumulate(paths, min_adc=args.min_adc,
                              require_ot=args.require_ot,
                              max_packets=args.max_packets)
    live = dc.live_time(paths, capture_duration_s=args.capture_duration,
                        srs_span_s=hists.srs_span_s())
    for w in live["warnings"]:
        print(f"  WARNING: {w}")
    print(f"  live time {live['live_s']:.2f} s ({live['method']})")

    cab = dc.verify_cabling(hists.vmm_totals(),
                            disconnected=tuple(args.disconnected))
    print(dc.format_cabling_report(cab))

    tab = dc.build_pad_table(strategy=args.strategy,
                             disconnected=tuple(args.disconnected))
    full = dc.full_pad_map(strategy=args.strategy,
                           disconnected=tuple(args.disconnected))
    t = per_pad_table(hists, tab, max(live["live_s"], 1e-9),
                      min_events=args.min_events, fit_min=args.fit_min)
    t = mask_hot_pads(t, ratio=args.hot_ratio)

    if args.background:
        bg_paths = resolve_inputs(args.background)
        print(f"background: {len(bg_paths)} capture(s)")
        bh, _ = accumulate(bg_paths, min_adc=args.min_adc,
                           require_ot=args.require_ot,
                           max_packets=args.max_packets, progress=False)
        blive = dc.live_time(bg_paths, capture_duration_s=args.capture_duration)
        bt = per_pad_table(bh, tab, max(blive["live_s"], 1e-9),
                           min_events=10 ** 9)
        # Rate subtraction only. The gain map is deliberately NOT background
        # subtracted: a photopeak is a position in ADC, not a count, and
        # subtracting spectra would distort it.
        t["rate_hz_bkg"] = bt["rate_hz"].to_numpy()
        t["rate_hz_cm2_bkg"] = bt["rate_hz_cm2"].to_numpy()
        t["rate_hz"] = t["rate_hz"] - t["rate_hz_bkg"]
        t["rate_hz_cm2"] = t["rate_hz_cm2"] - t["rate_hz_cm2_bkg"]
        print(f"  background live {blive['live_s']:.1f} s, "
              f"{bt['rate_hz'].sum():.2f} Hz subtracted")

    census = channel_census(t)
    src = fit_source(t)
    move = move_advice(src, args.target, t)
    spec = fit_fe55_spectrum(
        hists.adc[dc.pad_key(t.loc[~t["hot"], "vmm"].to_numpy(),
                             t.loc[~t["hot"], "ch"].to_numpy())].sum(axis=0),
        fit_min=args.fit_min)

    figs = [
        plot_rate_map(t, full, src, move, live, base, out_dir, title),
        plot_source_profile(t, src, move, base, out_dir, title),
        plot_gain_map(t, full, base, out_dir, title, args.min_events),
        plot_spectra(hists, t, spec, base, out_dir, title, args.fit_min),
        plot_rate_vs_channel(t, base, out_dir, title),
        plot_adc_vs_channel(t, base, out_dir, title, spec),
    ]

    csv_cols = ["channel_id", "vmm", "ch", "hybrid", "connector_N", "half",
                "sector", "strip", "pad_cx", "pad_cy", "radius", "phi",
                "pad_area", "n_hits", "n_raw", "rate_hz", "rate_hz_cm2",
                "peak_adc", "fwhm_adc", "res_fwhm_pct", "rel_gain",
                "median_adc", "mean_adc", "sat_frac", "ot_frac", "hot"]
    csv_cols += [c for c in ("rate_hz_bkg", "rate_hz_cm2_bkg") if c in t]
    csv_path = os.path.join(out_dir, f"{base}_pads.csv")
    t[csv_cols].to_csv(csv_path, index=False)

    lit = int((t["n_hits"] > 0).sum())
    with_peak = int(np.isfinite(t["peak_adc"]).sum())
    summary = {
        "title": title,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": [os.path.basename(p) for p in paths],
        "n_captures": len(paths),
        "live_time": {k: v for k, v in live.items() if k != "per_file"},
        "cabling_check": cab,
        "conditions": args.conditions,
        "cabling": {"strategy": args.strategy,
                    "disconnected_connectors": list(args.disconnected),
                    "vmm_base": dc.VMM_BASE,
                    "bottom_vmm_half": dc.BOTTOM_VMM_HALF},
        "cuts": {"min_adc": args.min_adc, "require_over_threshold":
                 args.require_ot, "fit_min": args.fit_min,
                 "min_events_per_pad": args.min_events,
                 "hot_pad_ratio": args.hot_ratio},
        "rate": {
            "total_hz": float(t["rate_hz"].sum()),
            "instrumented_pads": int(len(t)),
            "pads_with_hits": lit,
            "coverage_frac": lit / len(t) if len(t) else 0.0,
            "peak_pad_rate_hz_cm2": float(np.nanmax(t["rate_hz_cm2"]))
            if len(t) else None,
            "n_hot_pads_masked": int(t["hot"].sum()),
        },
        "source": {k: v for k, v in src.items() if not k.startswith("_")},
        "move": move,
        "gain": {
            "pads_with_peak": with_peak,
            "gain_map_coverage_frac": with_peak / len(t) if len(t) else 0.0,
            "peak_ref_adc": t.attrs.get("peak_ref_adc"),
            "peak_spread_std_over_mean":
                float(np.nanstd(t["peak_adc"]) / np.nanmean(t["peak_adc"]))
                if with_peak > 1 else None,
            "median_resolution_fwhm_pct":
                float(np.nanmedian(t["res_fwhm_pct"])) if with_peak else None,
            "saturated_hit_frac": float(
                (t["sat_frac"] * t["n_hits"]).sum() / max(t["n_hits"].sum(), 1)),
        },
        "channels": census,
        "spectrum": {k: v for k, v in (spec or {}).items()
                     if not k.startswith("_")},
        "figures": [os.path.basename(f) for f in figs],
        "pads_csv": os.path.basename(csv_path),
        "warnings": _warnings(t, src, spec, live, cab, args, census),
    }
    json_path = os.path.join(out_dir, f"{base}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 68}")
    if move:
        print(f"SOURCE at ({move['source_x_mm']:.1f}, {move['source_y_mm']:.1f})"
              f" mm  =  r {move['source_r_mm']:.1f} mm, "
              f"phi {move['source_phi_deg']:.2f} deg   [{move['estimator']}]")
        print(f"MOVE   dx {move['dx_mm']:+.1f} mm, dy {move['dy_mm']:+.1f} mm"
              f"   (dr {move['dr_mm']:+.1f} mm, "
              f"dphi {move['dphi_deg']:+.2f} deg)  -> {move['target']}")
    print(f"RATE   {t['rate_hz'].sum():.1f} Hz total, "
          f"{lit}/{len(t)} pads lit, "
          f"peak {np.nanmax(t['rate_hz_cm2']):.2f} Hz/cm2")
    if spec and "peak_adc" in spec:
        print(f"GAIN   photopeak {spec['peak_adc']:.1f} ADC, "
              f"FWHM/peak {spec['resolution_fwhm_pct']:.1f} %, "
              f"gain map on {with_peak}/{len(t)} pads")
    for w in summary["warnings"]:
        print(f"WARN   {w}")
    print(f"{'=' * 68}\nwrote {out_dir}/{base}_*.png, _pads.csv, _summary.json")
    return summary


def _warnings(t, src, spec, live, cab, args, census=None):
    """Everything that would make a reader mistrust these numbers, said out
    loud rather than left for them to notice."""
    w = []
    w += live.get("warnings", [])
    if not cab["consistent"]:
        w.append("cabling check failed — see cabling_check; the pad map may be "
                 "rotated by a sector")
    lit = int((t["n_hits"] > 0).sum())
    if len(t) and lit / len(t) < 0.10:
        w.append(f"only {lit}/{len(t)} instrumented pads have any hits: this is "
                 "a local map, not a map of the whole detector — move the "
                 "source and merge with vmm_fe55_merge.py")

    # The opposite failure, and the one the first real capture hit: the whole
    # detector lit at a comparable rate. A collimated source is a bright spot on
    # a dark chamber, so low contrast means the illumination is broad and the
    # fitted centre is only weakly constrained -- it is NOT a claim that there
    # is no structure, which the fit quality below is what actually settles.
    r = t.loc[t["n_hits"] > 0, "rate_hz_cm2"].to_numpy(float)
    if r.size > 50:
        med = float(np.median(r))
        contrast = float(np.percentile(r, 99) / med) if med > 0 else np.inf
        if contrast < 5.0 and lit > 0.8 * len(t):
            w.append(f"illumination is BROAD and low-contrast ({lit}/{len(t)} "
                     f"pads lit, p99/median rate = {contrast:.1f}): the source "
                     "covers essentially the whole readout, so the fitted "
                     "centre is weakly constrained — consistent with an "
                     "uncollimated or distant source")

    # A 2D Gaussian will return a confident centre for a distribution it does
    # not describe at all. On the first real capture it converged with
    # chi2/ndf = 511 and still printed a position to 0.1 mm. Poisson errors on
    # a per-pad count make this a meaningful test, so use it.
    fit = (src or {}).get("fit") or {}
    chi2 = fit.get("chi2_per_ndf")
    if chi2 is not None and chi2 > 10:
        w.append(f"the 2D Gaussian fit is POOR (chi2/ndf = {chi2:.0f}): the "
                 "spot is not Gaussian and the fitted centre should not be "
                 "trusted to better than the pad scale — prefer the centroid, "
                 "and read the radial/phi projections instead")
    # Containment is computed FROM the Gaussian, so it is only meaningful when
    # the Gaussian is. Reporting "the spot is clipped" off a fit that has
    # already been rejected for MOVE would be quoting the same bad number twice.
    fit0 = (src or {}).get("fit") or {}
    chi2_0 = fit0.get("chi2_per_ndf")
    fit_ok = bool(fit0) and (chi2_0 is None or chi2_0 <= 10)
    c = src.get("containment")
    if fit_ok and c is not None and c < 0.9:
        w.append(f"only {100 * c:.0f}% of the fitted spot lands on instrumented "
                 "pads: the spot is clipped and its fitted centre is pulled "
                 "inwards")
    sat = float((t["sat_frac"] * t["n_hits"]).sum() / max(t["n_hits"].sum(), 1))
    if sat > 0.01:
        w.append(f"{100 * sat:.1f}% of hits are at ADC {ADC_MAX} (saturated): "
                 "the photopeak is clipped, lower the VMM gain")
    if spec and "peak_adc" in spec:
        prom = spec.get("peak_prominence")
        if prom is not None and prom < 0.30:
            w.append(f"the 'photopeak' has prominence {prom:.2f} over its own "
                     "background: this spectrum has NO REAL PEAK and the "
                     "quoted peak_adc is meaningless — the gain map built from "
                     "it is not a gain map")
        res = spec.get("resolution_fwhm_pct")
        if res is not None and res > 40:
            w.append(f"FWHM/peak = {res:.0f}% is far broader than an Fe55 "
                     "photopeak (~20%): this is not a 5.9 keV line")
        if spec["peak_adc"] < 3 * max(args.fit_min, 1):
            w.append("photopeak sits close to the ADC cut: what is mapped may "
                     "be the threshold, not the gain")
        e = spec.get("escape_to_main_area")
        if e is not None and e < 0.02:
            w.append("no Ar escape peak resolved — the main peak may be a "
                     "threshold turn-on rather than the 5.9 keV photopeak")
        elif spec.get("escape_visible") is False:
            w.append(f"the Ar escape peak would fall at ADC "
                     f"{spec['escape_peak_adc']:.0f}, below the threshold "
                     f"turn-on at {spec['turn_on_adc']:.0f} — it cannot be used "
                     "to confirm the photopeak; raise the gain or lower the "
                     "threshold to bring it into range")
        pot = spec.get("peak_over_turn_on")
        if pot is not None and pot < 2.0:
            w.append(f"the peak sits only {pot:.1f}x above the threshold "
                     f"turn-on (ADC {spec['turn_on_adc']:.0f}): its low-energy "
                     "side is cut by the discriminator, so the width is a lower "
                     "limit and the peak position is biased upward")
    elif spec:
        w.append(f"no photopeak fit: {spec.get('error')}")
    for b in (census or {}).get("contiguous_blocks", []):
        w.append(f"VMM {b['vmm']} (connector {b['connector_N']}, {b['half']}) "
                 f"has {b['n_channels']} CONSECUTIVE dead/low channels "
                 f"{b['ch_from']}-{b['ch_to']}: a contiguous block is a "
                 "connector or ribbon fault, not channel attrition")
    if t["hot"].sum() > 0.05 * max(len(t), 1):
        w.append(f"{int(t['hot'].sum())} pads flagged hot — check the "
                 "thresholds before trusting the rate map")
    return w


def build_parser():
    p = argparse.ArgumentParser(
        description="Fe55 rate map, source localisation and per-pad gain map "
                    "for the P2 basket read out by 9 VMM hybrids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="pcapng file(s), a directory (searched recursively), "
                        "or a glob")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: alongside the first input)")
    p.add_argument("--name", default=None, help="basename for the products")
    p.add_argument("--max-packets", type=int, default=None,
                   help="decode at most N packets per capture (quick look)")
    p.add_argument("--data-format", default="SRS", choices=["SRS", "TRG"])
    p.add_argument("--capture-duration", type=float, default=None,
                   help="configured seconds per capture "
                        "(run_config_beam.CAPTURE_DURATION_S); used when the "
                        "packet-clock span undercounts")

    g = p.add_argument_group("cuts")
    g.add_argument("--min-adc", type=int, default=0,
                   help="drop hits below this ADC before anything else")
    g.add_argument("--require-ot", action="store_true",
                   help="keep only hits with the VMM over-threshold flag set")
    g.add_argument("--fit-min", type=int, default=0,
                   help="ignore ADC below this when locating a photopeak")
    g.add_argument("--min-events", type=int, default=30,
                   help="pads with fewer hits get no gain measurement")
    g.add_argument("--hot-ratio", type=float, default=6.0,
                   help="pad is hot above this x its 6 spatial neighbours")
    g.add_argument("--background", nargs="+", default=None,
                   help="source-out capture(s) whose rate is subtracted")

    g = p.add_argument_group("cabling")
    g.add_argument("--strategy", default=dc.WITHIN_STRATEGY,
                   choices=["linear", "reverse", "pairswap"],
                   help="within-half channel ordering")
    g.add_argument("--disconnected", type=int, nargs="*",
                   default=list(dc.DISCONNECTED),
                   help="connector numbers with no hybrid")
    g.add_argument("--scan-mapping", action="store_true",
                   help="scan within-half orderings and exit")

    g = p.add_argument_group("source")
    g.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"),
                   default=None,
                   help="where you want the spot [mm]; default is the centre "
                        "of the instrumented area")

    p.add_argument("--conditions", default=None,
                   help="run conditions for the figure titles and the summary, "
                        "e.g. \"drift 720 V / mesh 390 V, SFM on\". Recorded "
                        "verbatim; nothing is inferred from it.")
    p.add_argument("--watch", action="store_true",
                   help="re-run whenever new captures appear")
    p.add_argument("--watch-interval", type=float, default=20.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = resolve_inputs(args.inputs)
    if not paths:
        print("no pcapng found", file=sys.stderr)
        return 2

    out_dir = args.out_dir or os.path.dirname(paths[0]) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = args.name or ("fe55_" + os.path.splitext(os.path.basename(paths[0]))[0]
                         if len(paths) == 1 else "fe55_combined")
    # The run conditions are NOT hardcoded: drift/mesh and the chip settings
    # change between source runs, and a figure archived under the wrong
    # voltages is worse than one with no voltages on it at all.
    title = f"det1 Fe55 — {len(paths)} capture(s)"
    if args.conditions:
        title += f", {args.conditions}"

    if args.scan_mapping:
        scan_mapping(paths, args)
        return 0

    if not args.watch:
        analyse(paths, args, out_dir, base, title)
        return 0

    seen = set()
    print(f"watching {args.inputs} every {args.watch_interval:.0f} s "
          "(Ctrl-C to stop)")
    try:
        while True:
            paths = resolve_inputs(args.inputs)
            # Only re-run when a capture is COMPLETE: a file dumpcap is still
            # appending to would be decoded now and again later, and its hits
            # would be counted at two different live times.
            ready = [p for p in paths
                     if time.time() - os.path.getmtime(p) > 5.0]
            new = [p for p in ready if p not in seen]
            if new:
                print(f"\n--- {len(new)} new capture(s), "
                      f"{len(ready)} total ---")
                analyse(ready, args, out_dir, base, title)
                seen.update(ready)
            time.sleep(args.watch_interval)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
