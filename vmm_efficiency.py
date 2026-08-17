#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_efficiency.py — trigger-referenced efficiency of the P2 telescope stations
from the VMM data stream.

Method
------
The external trigger is digitised on hybrid 0 in the SAME stream as the
detectors, so no external reference is needed:

  1. trigger times = the trigger channel's hits (VMM 0 ch 44)
  2. dt            = t_station - t_trigger
  3. fit           = Gaussian peak on a flat accidental background
                     -> signal window [mu - n_sigma*sigma, mu + n_sigma*sigma]
                     -> sideband of equal width, offset past the peak
  4. efficiency    = P(>=1 station hit in signal window)
                     - P(same in sideband), accidental-subtracted

A station counts as hit if ANY of its six VMMs fired, which is the quantity
that matters for tracking.

Why the timing is not simply abs_time_ns
----------------------------------------
This firmware leaves `offset` — the BCID rollover counter — stuck at a constant
(see VMM_ONLINE_EFFICIENCY_PLAN.md §2c), and SRS markers arrive only every
1.6384 ms while BCID wraps every 4096 * 22.5 = 92.16 us. So a hit's absolute
time is ambiguous by an unknown multiple of 92.16 us, and abs_time_ns is not
even monotonic. Coincidence-hunting on it directly returns noise (measured:
sigma ~ 300 ns, i.e. the whole search window).

What IS unambiguous is the phase within the BCID cycle, and which marker
interval a hit belongs to. So this module:

  * pairs a trigger only with station hits carrying the SAME srs_timestamp
    (markers are broadcast — 2799 of 2808 values are shared across VMMs), and
  * compares BCID phases, wrapped into +/- 46.08 us.

Measured on run_24/nominal_05: this lifts the coincidence peak from 1.01x the
accidental background (ungrouped) to 8.6-17.1x, with per-station latencies of
+180 / +112 / +112 ns for IN / MID / OUT.

Residual ambiguity: ~17.8 BCID cycles fit in one marker interval, so a
trigger's true partner is one of ~18 candidates. That is what the sideband
subtracts. Restoring `offset`, or raising the marker rate above ~11 kHz, would
remove the ambiguity entirely and sharpen every number here — see the plan doc.

Honest caveat, printed on every plot
------------------------------------
This is efficiency relative to the TRIGGER acceptance — the scintillator
defines the denominator — and is not corrected for the geometric overlap of the
trigger with each station. Station-to-station comparison is meaningful; the
absolute value is a lower bound.

No scipy in the DAQ venv, so the fit and the binomial interval have numpy
fallbacks; scipy is used automatically when present (offline).

Usage
-----
  python3 vmm_efficiency.py <file.pcapng> [--out-dir DIR] [--json]
                            [--max-packets N] [--n-sigma 3] [--window 2000]
                            [--no-mask] [--find-trigger]

@author: ak271430 Alexandra Kallitsopoulou
"""

import os
import sys
import json
import argparse
import datetime

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_stations as vs

try:
    from scipy.optimize import curve_fit
    from scipy.stats import beta as _beta
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

BCID_PERIOD_NS = 4096 * vs.CLOCK_PERIOD_NS   # 92 160 ns

# Colour per station, fixed by entity so a station keeps its colour in every
# figure regardless of how many are plotted or how they rank.
STATION_COLOR = {'P2_IN': '#2C6FB5', 'P2_MID': '#E08214', 'P2_OUT': '#6A3D9A'}

# Cap on the trigger x hit outer product evaluated at once, to bound memory
# when a marker interval is unusually busy.
_PAIR_CHUNK = 2_000_000

# The trigger channel re-fires: on run_24 the inter-arrival distribution of
# VMM 0 ch 44 peaks at 500-600 ns, and every station shows a matching secondary
# coincidence ~500 ns before the main one. Those repeats are the SAME particle,
# so counting them as separate triggers inflates the denominator several-fold
# and drives the apparent efficiency down. Collapse anything within this window
# into one trigger. Must exceed the ~600 ns repeat spacing; well under the
# ~20 us mean spacing between real particles at 50 kHz, so the loss is small.
TRIGGER_DEADTIME_NS = 1500.0


def dedup_triggers(pt, dead_ns=TRIGGER_DEADTIME_NS):
    """Collapse trigger repeats: keep a hit only if it is more than dead_ns
    after the last kept one."""
    if dead_ns <= 0 or pt.size < 2:
        return pt
    pt = np.sort(pt)
    keep = np.empty(pt.size, dtype=bool)
    keep[0] = True
    last = pt[0]
    for i in range(1, pt.size):
        if pt[i] - last > dead_ns:
            keep[i] = True
            last = pt[i]
        else:
            keep[i] = False
    return pt[keep]


def wrap_phase(d, period=BCID_PERIOD_NS):
    """Wrap a phase difference into [-period/2, +period/2)."""
    return (d + period / 2.0) % period - period / 2.0


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def binomial_interval(k, n, cl=0.6827):
    """68.27% interval on k/n. Clopper-Pearson with scipy, Wilson without.

    At these trigger counts (n ~ 1e5) the two agree to far better than 0.1%,
    so the fallback costs nothing; the exact interval is used offline, matching
    the Dream pipeline's convention.
    """
    if n == 0:
        return 0.0, 1.0
    if HAVE_SCIPY:
        a = 1.0 - cl
        lo = _beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
        hi = _beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
        return float(lo), float(hi)
    z = 0.9944578832097535   # normal quantile for a two-sided 68.27% CL
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _gaussian_plus_bg(x, amplitude, mu, sigma, bg):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bg


def _robust_peak(centers, hist):
    """Lock onto the DOMINANT peak and measure it locally.

    Deliberately local: the mixing-subtracted spectrum carries a secondary
    coincidence ~600 ns before the main one (see find_secondary_peak), and a
    global percentile width straddles both, returning sigma ~ 300 ns for a peak
    that is really ~20 ns wide. Half-maximum walk first, then weighted moments
    inside +/-3 sigma of that.
    """
    n = len(hist)
    if n < 5 or hist.max() <= 0:
        return None
    binw = float(centers[1] - centers[0])
    pk = int(np.argmax(hist))
    half = 0.5 * hist[pk]

    lo = pk
    while lo > 0 and hist[lo - 1] > half:
        lo -= 1
    hi = pk
    while hi < n - 1 and hist[hi + 1] > half:
        hi += 1
    sigma0 = max(centers[hi] - centers[lo], binw) / 2.3548

    sel = np.abs(centers - centers[pk]) <= 3.0 * sigma0
    w = np.clip(hist[sel], 0, None)
    if w.sum() <= 0:
        return None
    mu = float(np.sum(w * centers[sel]) / w.sum())
    var = float(np.sum(w * (centers[sel] - mu) ** 2) / w.sum())
    sigma = float(max(np.sqrt(max(var, 0.0)), binw / 2.0))
    bg = float(np.median(hist))          # ~0 once mixing is subtracted
    return float(hist[pk] - bg), mu, sigma, max(bg, 0.0)


def find_secondary_peak(centers, hist, mu, sigma, min_frac=0.05):
    """Largest peak outside +/-5 sigma of the main one, if it is >= min_frac of it.

    Reported, never fitted. On run_24 every station shows one ~600 ns before the
    main coincidence -- unexplained, and worth understanding before these
    efficiencies are quoted anywhere.
    """
    far = np.abs(centers - mu) > 5.0 * sigma
    if not np.any(far) or hist.max() <= 0:
        return None
    j = int(np.argmax(np.where(far, hist, -np.inf)))
    if hist[j] < min_frac * hist.max():
        return None
    return {'dt_ns': float(centers[j]), 'height_frac': float(hist[j] / hist.max())}


def fit_dt_peak(dt, window_ns, n_bins=200, n_sigma=3.0, dt_mixed=None):
    """Locate the coincidence peak and derive signal + sideband windows.

    With dt_mixed (event-mixed pairs) the accidental shape is subtracted before
    fitting, so the peak sits on a genuinely flat residual.
    """
    edges = np.linspace(-window_ns, window_ns, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.histogram(dt, bins=edges)[0].astype(float)
    hist_raw = hist.copy()

    if len(dt) < 50:
        return {'success': False, 'bin_centers': centers, 'hist': hist}

    mixed = None
    if dt_mixed is not None and len(dt_mixed) >= 50:
        mixed = np.histogram(dt_mixed, bins=edges)[0].astype(float)
        # Normalise the mixed sample on the wings, where signal is absent.
        n = len(hist)
        w = np.r_[np.arange(n // 4), np.arange(3 * n // 4, n)]
        scale = hist[w].sum() / mixed[w].sum() if mixed[w].sum() > 0 else 1.0
        mixed *= scale
        hist = hist - mixed

    guess = _robust_peak(centers, hist)
    if guess is None:
        return {'success': False, 'bin_centers': centers, 'hist': hist}

    A, mu, sigma, bg = guess
    method = 'robust'
    if HAVE_SCIPY:
        try:
            # Fit only the neighbourhood of the main peak, for the same reason
            # _robust_peak measures locally.
            m = np.abs(centers - mu) <= 6.0 * sigma
            popt, _ = curve_fit(
                _gaussian_plus_bg, centers[m], hist[m], p0=[A, mu, sigma, bg],
                bounds=([0, mu - 6 * sigma, 1.0, 0],
                        [np.inf, mu + 6 * sigma, 6 * sigma, np.inf]), maxfev=5000)
            A, mu, sigma, bg = (float(v) for v in popt)
            method = 'gaussian+bg'
        except (RuntimeError, ValueError):
            pass

    half = n_sigma * sigma
    dt_min, dt_max = mu - half, mu + half
    sb_min = dt_max + sigma           # one-sigma gap past the signal window
    sb_max = sb_min + 2 * half        # same width as the signal window
    if sb_max > window_ns:            # no room on the right — mirror to the left
        sb_max, sb_min = dt_min - sigma, dt_min - sigma - 2 * half

    return {'success': True, 'method': method, 'amplitude': A, 'mu': mu,
            'sigma': sigma, 'bg': bg,
            'contrast': _contrast(hist_raw, mixed, centers, mu),
            'secondary_peak': find_secondary_peak(centers, hist, mu, sigma),
            'dt_min': dt_min, 'dt_max': dt_max,
            'sb_min': min(sb_min, sb_max), 'sb_max': max(sb_min, sb_max),
            'bin_centers': centers, 'hist': hist, 'hist_raw': hist_raw,
            'mixed': mixed}


def _contrast(hist_raw, mixed, centers, mu):
    """Peak height over the accidental level underneath it."""
    j = int(np.argmin(np.abs(centers - mu)))
    if mixed is None or mixed[j] <= 0:
        return float('inf')
    return float(hist_raw[j] / mixed[j])


# --------------------------------------------------------------------------
# Marker-interval grouped coincidences
# --------------------------------------------------------------------------
def build_groups(hits, station_vmms, trigger=None,
                 dead_ns=TRIGGER_DEADTIME_NS):
    """Split into marker intervals: [(trigger_phases, station_phases), ...].

    Grouping on srs_timestamp is what makes the measurement possible — see the
    module docstring.
    """
    trg = trigger or vs.TRIGGER
    vmm = hits['vmm'].to_numpy()
    ch = hits['ch'].to_numpy()
    srs = hits['srs_timestamp'].to_numpy()
    phase = np.mod(hits['timestamp_ns'].to_numpy(), BCID_PERIOD_NS)

    is_trg = (vmm == trg['vmm']) & (ch == trg['channel'])
    is_det = np.isin(vmm, station_vmms)

    groups = []
    order = np.argsort(srs, kind='stable')
    srs_s, phase_s, trg_s, det_s = srs[order], phase[order], is_trg[order], is_det[order]
    bounds = np.flatnonzero(np.diff(srs_s)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, srs_s.size]):
        pt = phase_s[lo:hi][trg_s[lo:hi]]
        pd_ = phase_s[lo:hi][det_s[lo:hi]]
        if pt.size and pd_.size:
            groups.append((dedup_triggers(np.unique(pt), dead_ns), pd_))
    return groups


def group_dt(groups, max_pairs=8_000_000, mixed=False):
    """All wrapped phase differences, for the Delta-t histogram.

    mixed=True correlates each interval's triggers with the NEXT interval's
    station hits — event mixing. Those pairs cannot be true coincidences, so
    the result is the accidental background WITH its real shape, including the
    non-uniform BCID phase structure that a flat-background fit gets wrong.
    """
    out, n = [], 0
    for i, (pt, _) in enumerate(groups):
        if n >= max_pairs:
            break
        pd_ = groups[(i + 1) % len(groups)][1] if mixed else groups[i][1]
        d = wrap_phase(pd_[None, :] - pt[:, None]).ravel()
        out.append(d)
        n += d.size
    return np.concatenate(out) if out else np.array([])


def group_efficiency(groups, fit):
    """Per-trigger counting: does any station hit fall in the window?"""
    sig_lo, sig_hi = fit['dt_min'], fit['dt_max']
    sb_lo, sb_hi = fit['sb_min'], fit['sb_max']
    n_trig = n_sig = n_side = 0

    for pt, pd_ in groups:
        n_trig += pt.size
        # Chunk the outer product so a busy interval cannot blow up memory.
        step = max(1, int(_PAIR_CHUNK // max(pd_.size, 1)))
        for i in range(0, pt.size, step):
            d = wrap_phase(pd_[None, :] - pt[i:i + step, None])
            n_sig += int(np.count_nonzero(((d >= sig_lo) & (d <= sig_hi)).any(axis=1)))
            n_side += int(np.count_nonzero(((d >= sb_lo) & (d <= sb_hi)).any(axis=1)))

    sig_w = sig_hi - sig_lo
    side_w = sb_hi - sb_lo
    scale = sig_w / side_w if side_w > 0 else 1.0

    raw = n_sig / n_trig if n_trig else 0.0
    acc = (n_side / n_trig * scale) if n_trig else 0.0
    lo, hi = binomial_interval(n_sig, n_trig)

    return {
        'n_triggers': n_trig, 'n_signal': n_sig, 'n_sideband': n_side,
        'raw_efficiency': raw, 'accidental_efficiency': acc,
        # Subtractive, matching vmm_detector_efficiency.py.
        'efficiency': raw - acc,
        # Probabilistic: signal and accidentals independent, raw = 1-(1-e)(1-a).
        'efficiency_corrected': ((raw - acc) / (1 - acc)) if acc < 1 else float('nan'),
        # Statistical interval on the raw count, shifted by the accidental
        # estimate; excludes the sideband's own uncertainty.
        'efficiency_lo': max(0.0, lo - acc), 'efficiency_hi': min(1.0, hi - acc),
    }


def coincident_mask(hits, station_vmms, fit, trigger=None,
                    dead_ns=TRIGGER_DEADTIME_NS):
    """Boolean over `hits`: True for station hits in the signal window of some
    trigger, in the same marker interval.

    This is the in-time hit selection -- the input to a per-pad efficiency map,
    and the clean sample for validating the pad wiring (noise hits are spread
    over the whole fan and wash the beam spot out).
    """
    trg = trigger or vs.TRIGGER
    vmm = hits['vmm'].to_numpy()
    ch = hits['ch'].to_numpy()
    srs = hits['srs_timestamp'].to_numpy()
    phase = np.mod(hits['timestamp_ns'].to_numpy(), BCID_PERIOD_NS)
    is_trg = (vmm == trg['vmm']) & (ch == trg['channel'])
    is_det = np.isin(vmm, station_vmms)

    keep = np.zeros(len(hits), dtype=bool)
    order = np.argsort(srs, kind='stable')
    srs_s = srs[order]
    bounds = np.flatnonzero(np.diff(srs_s)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, srs_s.size]):
        idx = order[lo:hi]
        pt = dedup_triggers(np.unique(phase[idx[is_trg[idx]]]), dead_ns)
        det_idx = idx[is_det[idx]]
        if pt.size == 0 or det_idx.size == 0:
            continue
        d = wrap_phase(phase[det_idx][None, :] - pt[:, None])
        hit = ((d >= fit['dt_min']) & (d <= fit['dt_max'])).any(axis=0)
        keep[det_idx[hit]] = True
    return keep


def station_fits(hits, window_ns=1000.0, n_sigma=3.0,
                 dead_ns=TRIGGER_DEADTIME_NS):
    """{station: fit} — the coincidence windows, without the efficiency pass."""
    out = {}
    for name, vmms in vs.STATION_VMMS.items():
        g = build_groups(hits, vmms, dead_ns=dead_ns)
        if not g:
            continue
        f = fit_dt_peak(group_dt(g), window_ns, n_sigma=n_sigma,
                        dt_mixed=group_dt(g, mixed=True))
        if f['success']:
            out[name] = f
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
CAVEAT = ('efficiency relative to the trigger acceptance, accidental-subtracted; '
          'BCID-phase coincidence within one SRS marker interval')


def plot_dt(results, base, out_dir, window_ns):
    names = [n for n in vs.STATIONS if n in results]
    if not names:
        return None
    fig, axes = plt.subplots(2, len(names), figsize=(5.6 * len(names), 8.4),
                             squeeze=False, sharex=True)
    for col, name in enumerate(names):
        r = results[name]
        fit, colour = r['fit'], STATION_COLOR[name]
        for row, scale in enumerate(('linear', 'log')):
            ax = axes[row][col]
            c, h = fit['bin_centers'], fit['hist']
            ax.bar(c, h, width=c[1] - c[0], color=colour, alpha=0.75,
                   label='event-mixing subtracted' if row == 0 else None)
            if fit['success']:
                ax.axvspan(fit['dt_min'], fit['dt_max'], color='0.35', alpha=0.16,
                           lw=0, label='signal' if row == 0 else None)
                ax.axvspan(fit['sb_min'], fit['sb_max'], color='0.55', alpha=0.10,
                           lw=0, hatch='///', label='sideband' if row == 0 else None)
                xs = np.linspace(-window_ns, window_ns, 600)
                ax.plot(xs, _gaussian_plus_bg(xs, fit['amplitude'], fit['mu'],
                                              fit['sigma'], fit['bg']),
                        color='0.15', lw=1.4, label='fit' if row == 0 else None)
            ax.axvline(0, color='0.2', ls=':', lw=1.0)
            ax.set_yscale(scale)
            if scale == 'log':
                # Mixing subtraction leaves bins at ~0 (and slightly negative),
                # which would drag a log axis to 1e-300. Floor it at 1 pair.
                ax.set_ylim(1.0, max(h.max(), 10.0) * 2.0)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            if row == 0:
                e = r['eff']
                t = f"{name}   z = {vs.STATIONS[name]['z_mm']:.0f} mm"
                if fit['success']:
                    t += (f"\n$\\mu$={fit['mu']:+.1f} ns   $\\sigma$={fit['sigma']:.1f} ns"
                          f"   peak/bg={fit['contrast']:.1f}"
                          f"\n$\\epsilon$={e['efficiency']:.3f}"
                          f" (raw {e['raw_efficiency']:.3f} − acc {e['accidental_efficiency']:.3f})")
                ax.set_title(t, fontsize=10.5, color=colour)
                ax.legend(fontsize=8, loc='upper right')
                ax.set_ylabel('trigger–hit pairs')
            else:
                ax.set_ylabel('trigger–hit pairs (log)')
                ax.set_xlabel(r'wrapped $\Delta t = t_{\rm station} - t_{\rm trigger}$  [ns]')
    fig.suptitle(f'Trigger-referenced coincidence — {base}\n{CAVEAT}', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, f'{base}_trigger_dt.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_efficiency(results, base, out_dir):
    names = [n for n in vs.STATIONS if n in results]
    if not names:
        return None
    vals = [results[n]['eff']['efficiency'] for n in names]
    lo = [max(0.0, v - results[n]['eff']['efficiency_lo']) for v, n in zip(vals, names)]
    hi = [max(0.0, results[n]['eff']['efficiency_hi'] - v) for v, n in zip(vals, names)]

    fig, ax = plt.subplots(figsize=(2.0 * len(names) + 3.6, 5.0))
    ax.bar(names, vals, color=[STATION_COLOR[n] for n in names], alpha=0.9, width=0.55)
    ax.errorbar(names, vals, yerr=np.array([lo, hi]), fmt='none', ecolor='0.2',
                elinewidth=1.3, capsize=4)
    for x, (n, v) in enumerate(zip(names, vals)):
        ax.text(x, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, color='0.15')
        ax.text(x, 0.015, f"z={vs.STATIONS[n]['z_mm']:.0f} mm\n"
                          f"N={results[n]['eff']['n_triggers']:,}",
                ha='center', fontsize=8, color='white')
    ax.set_ylim(0, min(1.0, (max(vals) if vals else 0.5) * 1.3 + 0.06))
    ax.set_ylabel('trigger-referenced efficiency')
    ax.grid(True, axis='y', alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f'Station efficiency — {base}\n{CAVEAT}', fontsize=10.5)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{base}_trigger_efficiency.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def analyse(hits, window_ns=2000.0, n_sigma=3.0, mask=True, trigger=None,
            dead_ns=TRIGGER_DEADTIME_NS):
    """Stage-1 analysis on an already-parsed hits DataFrame.

    Returns (summary_dict, per_station_arrays). Safe to call from
    vmm_pcapng_qa.py with the DataFrame it has already built.
    """
    applied, n_masked = {}, 0
    if mask:
        # Three sources: the hand-recorded list, the per-VMM median test, and
        # the neighbour test (needs the pad map — much the most sensitive).
        applied = vs.merge_masks(vs.NOISY_CHANNELS,
                                 vs.auto_hot_channels(hits),
                                 vs.auto_hot_pads(hits, vs.build_pad_table()))
        hits, n_masked = vs.apply_channel_mask(hits, applied)

    out = {'mask': {int(k): v for k, v in applied.items()},
           'n_masked_hits': int(n_masked),
           'window_ns': window_ns, 'n_sigma': n_sigma,
           'trigger_deadtime_ns': dead_ns,
           'bcid_period_ns': BCID_PERIOD_NS, 'stations': {}}
    results = {}

    for name, vmms in vs.STATION_VMMS.items():
        groups = build_groups(hits, vmms, trigger, dead_ns=dead_ns)
        if not groups:
            continue
        dt = group_dt(groups)
        dt_mix = group_dt(groups, mixed=True)
        fit = fit_dt_peak(dt, window_ns, n_sigma=n_sigma, dt_mixed=dt_mix)
        if not fit['success']:
            continue
        eff = group_efficiency(groups, fit)
        results[name] = {'dt': dt, 'fit': fit, 'eff': eff}
        out['stations'][name] = {
            'z_mm': vs.STATIONS[name]['z_mm'],
            'n_marker_intervals': len(groups),
            'mu_ns': fit['mu'], 'sigma_ns': fit['sigma'],
            'contrast': fit['contrast'], 'fit_method': fit['method'],
            'secondary_peak': fit.get('secondary_peak'),
            'window_ns': [fit['dt_min'], fit['dt_max']],
            'sideband_ns': [fit['sb_min'], fit['sb_max']],
            **eff,
        }
    out['n_triggers'] = max((s['n_triggers'] for s in out['stations'].values()),
                            default=0)
    return out, results


def main():
    ap = argparse.ArgumentParser(description='Trigger-referenced VMM station efficiency')
    ap.add_argument('pcap_file')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--max-packets', type=int, default=None)
    ap.add_argument('--window', type=float, default=2000.0,
                    help='Delta-t histogram half-width in ns (default 2000)')
    ap.add_argument('--n-sigma', type=float, default=3.0)
    ap.add_argument('--trigger-deadtime', type=float, default=TRIGGER_DEADTIME_NS,
                    help='Collapse trigger repeats within this many ns '
                         f'(default {TRIGGER_DEADTIME_NS:.0f}; 0 disables)')
    ap.add_argument('--no-mask', action='store_true', help='Skip hot-channel masking')
    ap.add_argument('--find-trigger', action='store_true',
                    help='Report the empirical trigger channel and exit')
    ap.add_argument('--json', action='store_true', help='Write efficiency.json')
    args = ap.parse_args()

    if not os.path.isfile(args.pcap_file):
        sys.exit(f'File not found: {args.pcap_file}')

    base = os.path.splitext(os.path.basename(args.pcap_file))[0]
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else \
        os.path.join(os.path.dirname(os.path.abspath(args.pcap_file)), 'qa_plots', base)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Reading: {args.pcap_file}')
    hits, meta = vs.parse_pcapng(args.pcap_file, max_packets=args.max_packets,
                                 progress=True)
    print(f"  {meta['n_packets']:,} packets | {meta['n_hits']:,} hits")
    if meta['bad_vmm_ids']:
        n_bad = sum(meta['bad_vmm_ids'].values())
        print(f"  corrupt VMM ids: {meta['bad_vmm_ids']} "
              f"({n_bad} hits, {n_bad / max(meta['n_hits'], 1):.4%})")
    if meta.get('marker_gap_ns_p50') is not None:
        print(f"  SRS marker gap p50={meta['marker_gap_ns_p50']:,.0f} ns  "
              f"(BCID rollover {meta['bcid_rollover_ns']:,.0f} ns) — "
              f"{meta['marker_gap_ns_p50'] / meta['bcid_rollover_ns']:.1f} cycles "
              f"per interval, resolved by phase grouping")

    if args.find_trigger:
        print(f'  empirical trigger channel: {vs.find_trigger_channel(hits)}  '
              f'(configured: {vs.TRIGGER})')
        return

    out, results = analyse(hits, window_ns=args.window, n_sigma=args.n_sigma,
                           mask=not args.no_mask, dead_ns=args.trigger_deadtime)
    print(f"\n  masked {out['n_masked_hits']:,} hits from {out['mask']}")
    print(f"  triggers: {out['n_triggers']:,}\n")
    if not out['stations']:
        print('  No station coincidence found — check --find-trigger.')
        return

    for name, s in out['stations'].items():
        print(f"  {name:8s} mu={s['mu_ns']:+7.1f} ns  sigma={s['sigma_ns']:6.1f} ns  "
              f"peak/bg={s['contrast']:5.1f}  raw={s['raw_efficiency']:.4f}  "
              f"acc={s['accidental_efficiency']:.4f}  eff={s['efficiency']:.4f} "
              f"[{s['efficiency_lo']:.4f}, {s['efficiency_hi']:.4f}]")

    p1 = plot_dt(results, base, out_dir, args.window)
    p2 = plot_efficiency(results, base, out_dir)
    print(f'\nWritten: {p1}\n         {p2}')

    if args.json:
        out['pcap_file'] = os.path.abspath(args.pcap_file)
        out['meta'] = meta
        out['processed_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        jpath = os.path.join(out_dir, 'efficiency.json')
        with open(jpath, 'w') as f:
            json.dump(out, f, indent=2, default=float)
        print(f'         {jpath}')


if __name__ == '__main__':
    main()
