#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-connector-half wiring scan: which halves are cabled differently?

The global scan (vmm_map_scan.py) picks one ordering for the whole telescope.
A single ribbon can still be flipped -- the DREAM readout needed exactly that on
P2_IN (c_5_top, commit 659dfbb). This finds it per half.

Method: take the station's beam centroid from the trigger-coincident hits of the
halves that agree with the global choice, then for each half pick the strategy
that puts ITS hits closest to that centroid. A correctly-ordered half sits on
the beam spot; a wrongly-ordered one is thrown across the fan.

Usage: python3 vmm_map_perhalf.py <file.pcapng> [--max-packets N]
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_stations as vs
import vmm_efficiency as ve

STRATEGIES = ['linear', 'reverse', 'pairswap']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pcap_file')
    ap.add_argument('--max-packets', type=int, default=400)
    ap.add_argument('--bottom-half', default='bot')
    ap.add_argument('--base', default='reverse')
    args = ap.parse_args()

    hits, _ = vs.parse_pcapng(args.pcap_file, max_packets=args.max_packets)
    hits, _ = vs.apply_channel_mask(
        hits, vs.merge_masks(vs.NOISY_CHANNELS, vs.auto_hot_channels(hits)))
    fits = ve.station_fits(hits)
    keep = np.zeros(len(hits), dtype=bool)
    for name, f in fits.items():
        keep |= ve.coincident_mask(hits, vs.STATION_VMMS[name], f)
    hits = hits[keep]
    print(f'{len(hits):,} trigger-coincident detector hits\n')

    halves = vs.vmm_half_map(args.bottom_half)
    best = {}
    for station in vs.STATIONS:
        # Station centroid under the global (base) mapping.
        tab = vs.build_pad_table(strategy=args.base, bottom_half=args.bottom_half,
                                 overrides={})
        h = vs.attach_pads(hits, tab)
        h = h[h['mapped'].astype(bool) & (h['station'] == station)]
        if len(h) < 200:
            print(f'{station}: too few hits'); continue
        cx = np.median(h['pad_cx']); cy = np.median(h['pad_cy'])

        print(f'{station}  centroid=({cx:.0f}, {cy:.0f}) mm')
        for vmm, (st, n, half) in sorted(halves.items()):
            if st != station:
                continue
            sub = hits[hits['vmm'] == vmm]
            if len(sub) < 50:
                print(f'   c{n} {half:3s} (VMM {vmm:2d}): only {len(sub)} hits — skipped')
                continue
            d = {}
            for strat in STRATEGIES:
                t = vs.build_pad_table(strategy=args.base,
                                       bottom_half=args.bottom_half,
                                       overrides={(station, n, half): strat})
                hh = vs.attach_pads(sub, t)
                hh = hh[hh['mapped'].astype(bool)]
                d[strat] = float(np.median(np.hypot(hh['pad_cx'] - cx,
                                                    hh['pad_cy'] - cy)))
            pick = min(d, key=d.get)
            flag = '' if pick == args.base else '   <== OVERRIDE'
            print(f'   c{n} {half:3s} (VMM {vmm:2d}): ' +
                  '  '.join(f'{k}={v:6.1f}' for k, v in d.items()) +
                  f'   -> {pick}{flag}')
            if pick != args.base:
                best[(station, n, half)] = pick

    print('\nSTRATEGY_OVERRIDES = {')
    for k, v in best.items():
        print(f'    {k!r}: {v!r},')
    print('}')


if __name__ == '__main__':
    main()
