#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_beam_profile.py — hit maps and beam profiles in real detector geometry.

The SPS stage-23 analogue for the VMM readout. Uses the validated
(vmm, ch) -> pad wiring from vmm_stations.build_pad_table(), so everything here
is in millimetres on the actual pad tiles, not in channel space.

Products, per capture file:
  <base>_hitmap.png        pad-tile occupancy per station.
                           Top row  = all hits (includes noise).
                           Bottom row = trigger-coincident hits = the beam.
  <base>_beam_profile.png  x and y projections of the in-time hits per station,
                           with centroid and RMS.

Only the instrumented connectors (c4-c6) are drawn -- the rest of each P2
detector is not read out by the VMM chain and is simply absent, which is worth
remembering when looking at how much of the chamber the beam actually covers.

Usage:
  python3 vmm_beam_profile.py <file.pcapng> [--out-dir DIR] [--max-packets N]

@author: ak271430 Alexandra Kallitsopoulou
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_stations as vs
import vmm_efficiency as ve

STATION_COLOR = ve.STATION_COLOR


def pad_polygons(tab):
    """(N,4,2) array of pad corners from centre + size + rotation.

    The P2 pads are fan-shaped tiles, so each carries its own angle; drawing
    them as rotated rectangles reproduces the fan instead of a square grid.
    """
    cx = tab['pad_cx'].to_numpy(float)
    cy = tab['pad_cy'].to_numpy(float)
    w = tab['pad_w'].to_numpy(float) if 'pad_w' in tab else np.full(len(tab), 5.0)
    h = tab['pad_h'].to_numpy(float) if 'pad_h' in tab else np.full(len(tab), 5.0)
    ang = np.deg2rad(tab['pad_angle'].to_numpy(float)) if 'pad_angle' in tab \
        else np.zeros(len(tab))
    w = np.nan_to_num(w, nan=5.0)
    h = np.nan_to_num(h, nan=5.0)
    ang = np.nan_to_num(ang, nan=0.0)

    dx = np.array([-0.5, 0.5, 0.5, -0.5])
    dy = np.array([-0.5, -0.5, 0.5, 0.5])
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    ox = (dx[None, :] * w[:, None])
    oy = (dy[None, :] * h[:, None])
    xs = cx[:, None] + ox * ca - oy * sa
    ys = cy[:, None] + ox * sa + oy * ca
    return np.stack([xs, ys], axis=-1)


def _counts_per_pad(hits, tab, station):
    """Hits per instrumented pad of one station, aligned to tab's row order."""
    sub = tab[tab['station'] == station].reset_index(drop=True)
    key = sub['vmm'].to_numpy().astype(np.int64) * 64 + sub['ch'].to_numpy()
    lut = {int(k): i for i, k in enumerate(key)}
    counts = np.zeros(len(sub))
    if len(hits):
        hk = hits['vmm'].to_numpy().astype(np.int64) * 64 + hits['ch'].to_numpy()
        uk, uc = np.unique(hk, return_counts=True)
        for k, c in zip(uk, uc):
            i = lut.get(int(k))
            if i is not None:
                counts[i] = c
    return sub, counts


def plot_hitmap(hits, in_time, tab, base, out_dir):
    stations = [s for s in vs.STATIONS]
    fig, axes = plt.subplots(2, len(stations),
                             figsize=(4.6 * len(stations), 8.6), squeeze=False)
    for col, st in enumerate(stations):
        for row, (sel, label) in enumerate(
                ((hits, 'all hits'), (in_time, 'trigger-coincident (beam)'))):
            ax = axes[row][col]
            sub, counts = _counts_per_pad(
                sel[np.isin(sel['vmm'].to_numpy(), vs.STATION_VMMS[st])]
                if len(sel) else sel, tab, st)
            good = sub['mapped'].astype(bool).to_numpy()
            polys = pad_polygons(sub)[good]
            c = counts[good]
            pc = PolyCollection(polys, array=c, cmap='viridis',
                                norm=LogNorm(vmin=max(c[c > 0].min(), 1) if np.any(c > 0) else 1,
                                             vmax=max(c.max(), 2)),
                                edgecolors='none')
            ax.add_collection(pc)
            ax.autoscale_view()
            ax.set_aspect('equal')
            fig.colorbar(pc, ax=ax, label='hits', fraction=0.046, pad=0.03)
            n = int(c.sum())
            ax.set_title(f"{st}  z={vs.STATIONS[st]['z_mm']:.0f} mm\n{label} — "
                         f"{n:,} hits", fontsize=10,
                         color=STATION_COLOR[st] if row == 1 else '0.25')
            ax.set_xlabel('pad x [mm]')
            if col == 0:
                ax.set_ylabel('pad y [mm]')
    fig.suptitle(f'Pad hit maps — {base}\n'
                 'only the instrumented connectors (c4–c6) are read out; '
                 'the rest of each chamber is absent by construction',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, f'{base}_hitmap.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_profile(in_time, tab, base, out_dir):
    stations = [s for s in vs.STATIONS]
    fig, axes = plt.subplots(2, len(stations),
                             figsize=(4.4 * len(stations), 6.6), squeeze=False)
    summary = {}
    h = vs.attach_pads(in_time, tab)
    h = h[h['mapped'].astype(bool)]
    for col, st in enumerate(stations):
        d = h[h['station'] == st]
        summary[st] = {}
        for row, axis in enumerate(('pad_cx', 'pad_cy')):
            ax = axes[row][col]
            if len(d) < 20:
                ax.text(0.5, 0.5, 'too few hits', transform=ax.transAxes,
                        ha='center')
                continue
            v = d[axis].to_numpy()
            ax.hist(v, bins=60, color=STATION_COLOR[st], alpha=0.8)
            mu, sd = float(np.mean(v)), float(np.std(v))
            ax.axvline(mu, color='0.15', ls='--', lw=1.2,
                       label=f'mean {mu:.1f} mm\nRMS {sd:.1f} mm')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            ax.set_xlabel(f'{"x" if axis == "pad_cx" else "y"} [mm]')
            if col == 0:
                ax.set_ylabel('in-time hits')
            if row == 0:
                ax.set_title(f"{st}  z={vs.STATIONS[st]['z_mm']:.0f} mm",
                             fontsize=10, color=STATION_COLOR[st])
            summary[st]['x_mean_mm' if row == 0 else 'y_mean_mm'] = mu
            summary[st]['x_rms_mm' if row == 0 else 'y_rms_mm'] = sd
        summary[st]['n_in_time'] = int(len(d))
    fig.suptitle(f'Beam profile, trigger-coincident hits — {base}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out_dir, f'{base}_beam_profile.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p, summary


def make(hits, fits, base, out_dir, tab=None, mask=True):
    """Build both products. `fits` is {station: fit} from vmm_efficiency.

    Applies the SAME hot-channel mask the efficiency uses. Without this the hit
    maps show noisy pads that the efficiency has already thrown away, so the two
    products disagree about what the detector looks like.
    """
    tab = tab if tab is not None else vs.build_pad_table()
    if mask:
        applied = vs.merge_masks(vs.NOISY_CHANNELS, vs.auto_hot_channels(hits),
                                 vs.auto_hot_pads(hits, tab))
        hits, n_masked = vs.apply_channel_mask(hits, applied)
        print(f'  beam profile: masked {n_masked:,} hits from {applied}')
    keep = np.zeros(len(hits), dtype=bool)
    for name, f in (fits or {}).items():
        keep |= ve.coincident_mask(hits, vs.STATION_VMMS[name], f)
    det = hits[hits['vmm'].to_numpy() != vs.TRIGGER['vmm']]
    in_time = hits[keep]
    p1 = plot_hitmap(det, in_time, tab, base, out_dir)
    p2, summary = plot_profile(in_time, tab, base, out_dir)
    return [p1, p2], summary


def main():
    ap = argparse.ArgumentParser(description='VMM pad hit maps and beam profiles')
    ap.add_argument('pcap_file')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--max-packets', type=int, default=None)
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.pcap_file))[0]
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else \
        os.path.join(os.path.dirname(os.path.abspath(args.pcap_file)),
                     'qa_plots', base)
    os.makedirs(out_dir, exist_ok=True)

    hits, _ = vs.parse_pcapng(args.pcap_file, max_packets=args.max_packets,
                              progress=True)
    fits = ve.station_fits(hits)
    paths, summary = make(hits, fits, base, out_dir)
    for st, s in summary.items():
        print(f"  {st:8s} n={s['n_in_time']:>8,}  "
              f"x={s.get('x_mean_mm', float('nan')):7.1f} +/- {s.get('x_rms_mm', float('nan')):5.1f} mm   "
              f"y={s.get('y_mean_mm', float('nan')):7.1f} +/- {s.get('y_rms_mm', float('nan')):5.1f} mm")
    print('\n' + '\n'.join(f'Written: {p}' for p in paths))


if __name__ == '__main__':
    main()
