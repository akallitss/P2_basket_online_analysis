#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_scan_curves.py — scan-level aggregation: efficiency and timing vs HV.

The SPS stage-22/28 analogue. Walks a run's analysis tree, reads the per-file
`events.json` written by the online QA, joins each sub-run to its HV setpoints
from the DAQ's run_config.json, and plots the curves.

Because it reads the JSONs the QA already wrote, this is a cheap scan -- no
pcapng is re-parsed. Run it any time during or after a scan.

HV note: the CAEN crate is driven from banco, not from this machine, so there is
no hv_monitor.csv on the VMM side. The x axis is therefore the SETPOINT from
run_config.json, not a readback. For anything final, cross-check against the
Dream side's hv_monitor.csv, which is the ground truth for what the HV actually
did (trips, sags).

Each station carries its own mesh voltage in a stepped-together scan (P2_IN runs
lower), so every station is plotted against its OWN voltage.

Usage:
  python3 vmm_scan_curves.py <run_name> [--analysis-root DIR] [--runs-root DIR]

@author: ak271430 Alexandra Kallitsopoulou
"""

import os
import sys
import csv
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_stations as vs
import vmm_efficiency as ve

DATA_ROOT = '/local/p2/p2data/TB_July26_H4'

# (card, channel) of each station's supplies — mirrors run_config_beam.P2_HV.
P2_HV = {
    'P2_IN':  {'drift': ('8', '0'), 'mesh': ('8', '1')},
    'P2_MID': {'drift': ('8', '2'), 'mesh': ('8', '3')},
    'P2_OUT': {'drift': ('8', '4'), 'mesh': ('8', '5')},
}


def hv_per_subrun(run_config_path):
    """{sub_run: {station: {'mesh': V, 'drift': V}}} from the DAQ run config."""
    with open(run_config_path) as f:
        cfg = json.load(f)
    out = {}
    for sr in cfg.get('sub_runs', []):
        hvs = sr.get('hvs', {})
        row = {}
        for st, chans in P2_HV.items():
            d = {}
            for what, (card, ch) in chans.items():
                try:
                    d[what] = float(hvs[card][ch])
                except (KeyError, TypeError, ValueError):
                    pass
            if d:
                row[st] = d
        out[sr['sub_run_name']] = row
    return out


def collect(analysis_run_dir):
    """{sub_run: {station: {...efficiency fields...}}} from every events.json."""
    out = {}
    for p in sorted(glob.glob(os.path.join(analysis_run_dir, '*', '*',
                                           'events.json'))):
        sub_run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        eff = d.get('efficiency')
        if not eff:
            continue
        # If a sub-run has several processed files, keep the one with the most
        # triggers -- the best-measured point, not an arbitrary one.
        prev = out.get(sub_run)
        if prev and prev.get('_n_trig', 0) >= d.get('n_triggers_dedup', 0):
            continue
        rec = dict(eff)
        rec['_n_trig'] = d.get('n_triggers_dedup', 0)
        rec['_file'] = os.path.basename(os.path.dirname(p))
        out[sub_run] = rec
    return out


def _plot(rows, xkey, xlabel, title, out_png, ykey, yerr=True,
          ylabel='trigger-referenced efficiency'):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    any_pt = False
    for st in vs.STATIONS:
        pts = sorted(((r['hv'][st][xkey], r['eff'][st]) for r in rows
                      if st in r['hv'] and xkey in r['hv'][st]
                      and st in r['eff']), key=lambda t: t[0])
        if not pts:
            continue
        any_pt = True
        x = np.array([p[0] for p in pts])
        y = np.array([p[1][ykey] for p in pts])
        kw = dict(color=ve.STATION_COLOR[st], marker='o', ms=6, lw=1.8,
                  label=st)
        if yerr and ykey == 'efficiency':
            lo = y - np.array([p[1]['efficiency_lo'] for p in pts])
            hi = np.array([p[1]['efficiency_hi'] for p in pts]) - y
            ax.errorbar(x, y, yerr=np.clip([lo, hi], 0, None), capsize=3, **kw)
        else:
            ax.plot(x, y, **kw)
        # Direct label at the last point so identity is never colour-alone.
        ax.annotate(st, xy=(x[-1], y[-1]), xytext=(6, 0),
                    textcoords='offset points', fontsize=9,
                    color=ve.STATION_COLOR[st], va='center')
    if not any_pt:
        plt.close(fig)
        return None
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='best')
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+',
                    help='One or more runs to merge (an interrupted scan and '
                         'its retake belong on the same curve)')
    ap.add_argument('--data-root', default=DATA_ROOT)
    args = ap.parse_args()

    out_dir = os.path.join(args.data_root, 'analysis', args.runs[0], 'scan')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for run in args.runs:
        hv = hv_per_subrun(os.path.join(args.data_root, 'runs', run,
                                        'run_config.json'))
        eff = collect(os.path.join(args.data_root, 'analysis', run))
        print(f'{run}: {len(eff)} sub-runs with efficiency, '
              f'{len(hv)} in the run config')
        for sr in sorted(eff):
            if hv.get(sr):
                rows.append({'sub_run': sr, 'run': run, 'hv': hv[sr],
                             'eff': eff[sr]})
    # A retake supersedes the interrupted attempt: keep the better-measured
    # copy of any sub-run that appears in more than one run.
    bysr = {}
    for r in rows:
        k = r['sub_run']
        if k not in bysr or r['eff'].get('_n_trig', 0) > bysr[k]['eff'].get('_n_trig', 0):
            bysr[k] = r
    rows = sorted(bysr.values(), key=lambda r: r['sub_run'])
    eff = {r['sub_run']: r['eff'] for r in rows}


    mesh = [r for r in rows if r['sub_run'].startswith('meshscan')]
    drift = [r for r in rows if r['sub_run'].startswith('driftscan')]
    print(f'  mesh points: {len(mesh)}   drift points: {len(drift)}')

    made = []
    for rws, key, lbl, tag in ((mesh, 'mesh', 'mesh voltage [V]', 'mesh'),
                               (drift, 'drift', 'drift voltage [V]', 'drift')):
        if not rws:
            continue
        made += [p for p in (
            _plot(rws, key, lbl,
                  f'{" + ".join(args.runs)} — efficiency vs {tag} (trigger-referenced)',
                  os.path.join(out_dir, f'efficiency_vs_{tag}.png'),
                  'efficiency'),
            _plot(rws, key, lbl,
                  f'{" + ".join(args.runs)} — coincidence width vs {tag}',
                  os.path.join(out_dir, f'timing_vs_{tag}.png'),
                  'sigma_ns', yerr=False,
                  ylabel=r'coincidence $\sigma$ [ns]'),
            _plot(rws, key, lbl,
                  f'{" + ".join(args.runs)} — coincidence latency vs {tag}',
                  os.path.join(out_dir, f'latency_vs_{tag}.png'),
                  'mu_ns', yerr=False,
                  ylabel=r'coincidence $\mu$ [ns]'),
        ) if p]

    csv_path = os.path.join(out_dir, 'scan_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sub_run', 'run', 'file', 'station', 'mesh_V', 'drift_V',
                    'n_triggers', 'efficiency', 'eff_lo', 'eff_hi',
                    'raw', 'accidental', 'mu_ns', 'sigma_ns', 'contrast'])
        for r in rows:
            for st in vs.STATIONS:
                if st not in r['eff'] or st not in r['hv']:
                    continue
                e = r['eff'][st]
                w.writerow([r['sub_run'], r.get('run', ''),
                            r['eff'].get('_file', ''), st,
                            r['hv'][st].get('mesh', ''),
                            r['hv'][st].get('drift', ''),
                            r['eff'].get('_n_trig', ''),
                            f"{e['efficiency']:.4f}",
                            f"{e['efficiency_lo']:.4f}",
                            f"{e['efficiency_hi']:.4f}",
                            f"{e['raw_efficiency']:.4f}",
                            f"{e['accidental_efficiency']:.4f}",
                            f"{e['mu_ns']:.1f}", f"{e['sigma_ns']:.1f}",
                            f"{e['contrast']:.1f}"])
    print('\n' + '\n'.join(f'Written: {p}' for p in made + [csv_path]))

    for rws, key, tag in ((mesh, 'mesh', 'MESH'), (drift, 'drift', 'DRIFT')):
        if not rws:
            continue
        print(f'\n{tag} scan:')
        print(f'  {"sub_run":22s} ' +
              ' '.join(f'{st:>22s}' for st in vs.STATIONS))
        for r in sorted(rws, key=lambda r: r['hv'].get('P2_MID', {}).get(key, 0)):
            cells = []
            for st in vs.STATIONS:
                if st in r['eff'] and st in r['hv']:
                    cells.append(f"{r['hv'][st].get(key, 0):5.0f}V "
                                 f"{r['eff'][st]['efficiency']:.3f}")
                else:
                    cells.append('—')
            print(f"  {r['sub_run']:22s} " + ' '.join(f'{c:>22s}' for c in cells))


if __name__ == '__main__':
    main()
