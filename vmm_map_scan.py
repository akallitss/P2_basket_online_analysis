#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_map_scan.py — pick the (vmm, ch) -> pad wiring by looking at the beam spot.

Two links in the chain are not yet validated for the VMM hybrid (see
vmm_stations.py): the within-half channel ordering, and whether the hybrid's
bottom VMM reads the 'bot' or 'top' half of its connector. This scans all
3 x 2 combinations and ranks them by how COMPACT the beam spot comes out in pad
space, per station.

The argument: the H4 beam is a small spot, so the correct wiring concentrates
hits; a wrong within-half ordering interleaves strips and smears them across the
fan. This is the same test that settled the det1 mapping and the run-67
orientation scan -- with the caveat recorded there, that a WIDE beam
discriminates only weakly. Check the rendered panels, do not trust the number
alone.

Usage:
  python3 vmm_map_scan.py <file.pcapng> [--max-packets N] [--out-dir DIR]
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_stations as vs
import vmm_efficiency as ve

STRATEGIES = ['linear', 'reverse', 'pairswap']
HALVES = ['bot', 'top']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pcap_file')
    ap.add_argument('--max-packets', type=int, default=400)
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--all-hits', action='store_true',
                    help='Use every hit instead of only trigger-coincident '
                         'ones (noise washes the beam spot out)')
    args = ap.parse_args()

    hits, _ = vs.parse_pcapng(args.pcap_file, max_packets=args.max_packets)
    mask = vs.merge_masks(vs.NOISY_CHANNELS, vs.auto_hot_channels(hits))
    hits, n_masked = vs.apply_channel_mask(hits, mask)
    n_all = int((hits['vmm'] != vs.TRIGGER['vmm']).sum())
    if args.all_hits:
        hits = hits[hits['vmm'] != vs.TRIGGER['vmm']]
    else:
        fits = ve.station_fits(hits)
        keep = np.zeros(len(hits), dtype=bool)
        for name, f in fits.items():
            keep |= ve.coincident_mask(hits, vs.STATION_VMMS[name], f)
        hits = hits[keep]
    print(f'{len(hits):,} detector hits used of {n_all:,} '
          f'({"all" if args.all_hits else "trigger-coincident"}; '
          f'masked {n_masked:,})\n')

    stations = list(vs.STATIONS)
    combos = [(s, h) for s in STRATEGIES for h in HALVES]
    print(f'{"strategy":10s} {"botVMM=":8s} ' +
          ' '.join(f'{n:>12s}' for n in stations) + f' {"mean":>9s}')
    scores = {}
    for strat, half in combos:
        tab = vs.build_pad_table(strategy=strat, bottom_half=half)
        sp = [vs.beam_spread(hits, tab, st)[0] for st in stations]
        scores[(strat, half)] = sp
        print(f'{strat:10s} {half:8s} ' +
              ' '.join(f'{v:12.1f}' for v in sp) +
              f' {np.nanmean(sp):9.1f}')

    best = min(scores, key=lambda k: np.nanmean(scores[k]))
    print(f'\nMost compact: strategy={best[0]!r}  bottom_half={best[1]!r}'
          f'  (mean RMS {np.nanmean(scores[best]):.1f} mm)')
    print('Confirm visually before adopting — see the rendered panels.')

    # Render every combination, one row per station.
    fig, axes = plt.subplots(len(stations), len(combos),
                             figsize=(3.1 * len(combos), 3.0 * len(stations)),
                             squeeze=False)
    for j, (strat, half) in enumerate(combos):
        tab = vs.build_pad_table(strategy=strat, bottom_half=half)
        h = vs.attach_pads(hits, tab)
        h = h[h['mapped'].astype(bool)]
        for i, st in enumerate(stations):
            ax = axes[i][j]
            d = h[h['station'] == st]
            if len(d):
                ax.hexbin(d['pad_cx'], d['pad_cy'], gridsize=45, cmap='viridis',
                          bins='log', linewidths=0)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f'{strat}\nbotVMM={half}', fontsize=9,
                             color='crimson' if (strat, half) == best else '0.2')
            if j == 0:
                ax.set_ylabel(st, fontsize=9)
    fig.suptitle('Beam spot vs assumed (vmm, ch) -> pad wiring — '
                 'the correct one is the most compact', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(args.out_dir, 'vmm_map_scan.png')
    fig.savefig(p, dpi=140)
    print(f'\nWritten: {p}')


if __name__ == '__main__':
    main()
