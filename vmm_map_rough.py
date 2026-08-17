#!/usr/bin/env python3
"""Per-half wiring scan judged by SPECKLE, not compactness.

vmm_map_scan.py minimises the beam-spot RMS, which is nearly blind to a
permutation inside a 64-strip half. This minimises vmm_stations.map_roughness()
instead -- dead pads next to bright ones -- which is what actually shows a
scrambled ordering on a hit map.

Greedy per half: fix everything else at the current best, try each strategy for
one half, keep the winner, move on. Two passes, so an early choice can be
revised once its neighbours are settled.
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_stations as vs
import vmm_efficiency as ve

STRATEGIES = ['linear', 'reverse', 'pairswap']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pcap_file')
    ap.add_argument('--max-packets', type=int, default=600)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--all-hits', action='store_true')
    args = ap.parse_args()

    hits, _ = vs.parse_pcapng(args.pcap_file, max_packets=args.max_packets)
    hits, _ = vs.apply_channel_mask(
        hits, vs.merge_masks(vs.NOISY_CHANNELS, vs.auto_hot_channels(hits)))
    if not args.all_hits:
        fits = ve.station_fits(hits)
        keep = np.zeros(len(hits), dtype=bool)
        for nm, f in fits.items():
            keep |= ve.coincident_mask(hits, vs.STATION_VMMS[nm], f)
        hits = hits[keep]
    print(f'{len(hits):,} hits used\n')

    halves = vs.vmm_half_map()
    best = {}   # (station, N, half) -> strategy

    def rough(station, over):
        tab = vs.build_pad_table(overrides=over)
        return vs.map_roughness(hits, tab, station)

    for station in vs.STATIONS:
        mine = [(n, h) for v, (st, n, h) in sorted(halves.items())
                if st == station]
        for _ in range(args.passes):
            for (n, h) in mine:
                scores = {}
                for s in STRATEGIES:
                    trial = dict(best); trial[(station, n, h)] = s
                    scores[s] = rough(station, trial)
                pick = min(scores, key=lambda k: (np.inf if np.isnan(scores[k])
                                                  else scores[k]))
                best[(station, n, h)] = pick
        base = rough(station, {k: v for k, v in best.items()
                               if k[0] != station})
        final = rough(station, best)
        picks = ' '.join(f'c{n}{h[0]}={best[(station, n, h)][:4]}'
                         for (n, h) in mine)
        print(f'{station}: roughness {base:.3f} -> {final:.3f}   {picks}')

    print('\nSTRATEGY_OVERRIDES = {')
    for k, v in sorted(best.items()):
        if v != vs.WITHIN_STRATEGY:
            print(f'    {k!r}: {v!r},')
    print('}')


if __name__ == '__main__':
    main()
