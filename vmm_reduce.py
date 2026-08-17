#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 1: reduce one decoded capture to counts + scalars.

This is the online product. Rendering 36 PNGs per 45 s capture costs ~43 s and
produces ~2900 images an hour that nobody opens; worse, four of the twenty VMMs
carry so few hits per file that their per-file 1024-bin ADC "spectrum" is under
one entry per bin. So the watcher reduces instead of drawing:

  counts.npz   ~0.1 MB   per-VMM histograms, ADDITIVE across files
  scalars.json ~2 kB     the numbers that go on the trend dashboard

Because counts are additive, merging N files is a sum of arrays. That is how the
dashboard gets a real ADC spectrum for the quiet VMMs (382 hits/file becomes
~8000 over 20 files) while doing less work than drawing one bad plot per file.
Any of the existing QA plots can be rendered from counts.npz later, on demand.

Used as a library by the dashboard, and as the processor watcher's unit of work:

    python vmm_reduce.py <capture.pcapng> --store-dir <dir>
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NV = 32                     # VMM id space (5 bits)
ADC_BINS = 1024
CH_BINS = 64
BCID_BINS = 4096
TDC_BINS = 256
OFFSET_BINS = 32            # signed offset -16..+15, stored shifted by +16
ADC_CH_ADC_BINS = 128       # adc >> 3 for the 2D adc-vs-channel map
TIME_BINS = 200             # frame-counter profile, for the rate-vs-time plot

CLOCK_PERIOD_NS = 22.5


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------

def _per_vmm(vmm, vals, nbins):
    """Per-VMM histogram via one bincount on vmm*nbins + value."""
    idx = vmm.astype(np.int64) * nbins + vals.astype(np.int64)
    return np.bincount(idx, minlength=NV * nbins).reshape(NV, nbins).astype(np.uint32)


def build_counts(cols) -> dict:
    """Per-VMM histograms of every quantity the QA plots use.

    `cols` is a dict of arrays (from vmm_decode.load_hits or decode). Only the
    raw columns are touched, so this works straight off a memory-mapped store.
    """
    vmm = np.asarray(cols["vmm"])
    ch = np.asarray(cols["ch"])
    adc = np.asarray(cols["adc"])
    tdc = np.asarray(cols["tdc"])
    bcid = np.asarray(cols["bcid"])
    offset = np.asarray(cols["offset"])
    ot = np.asarray(cols["over_threshold"]).astype(np.uint8)
    tcol = np.asarray(cols["time"])

    out = {
        "adc": _per_vmm(vmm, adc, ADC_BINS),
        "ch": _per_vmm(vmm, ch, CH_BINS),
        "bcid": _per_vmm(vmm, bcid, BCID_BINS),
        "tdc": _per_vmm(vmm, tdc, TDC_BINS),
        "offset": _per_vmm(vmm, offset.astype(np.int16) + 16, OFFSET_BINS),
        "ot": _per_vmm(vmm, ot, 2),
        "hits_per_vmm": np.bincount(vmm, minlength=NV).astype(np.uint64),
    }

    # ADC split by the over-threshold flag (the adc_ot plot)
    for name, sel in (("adc_ot0", ot == 0), ("adc_ot1", ot == 1)):
        out[name] = (_per_vmm(vmm[sel], adc[sel], ADC_BINS) if sel.any()
                     else np.zeros((NV, ADC_BINS), np.uint32))

    # 2D ADC vs channel, ADC coarsened 8x to keep the array small
    idx2 = ((vmm.astype(np.int64) * CH_BINS + ch) * ADC_CH_ADC_BINS
            + (adc.astype(np.int64) >> 3))
    out["adc_vs_ch"] = np.bincount(
        idx2, minlength=NV * CH_BINS * ADC_CH_ADC_BINS
    ).reshape(NV, CH_BINS, ADC_CH_ADC_BINS).astype(np.uint32)

    # rate vs frame counter. Edges are stored so files with different spans
    # can still be summed by the dashboard after re-binning.
    if len(tcol):
        t0, t1 = int(tcol.min()), int(tcol.max())
        span = max(t1 - t0, 1)
        tb = np.clip(((tcol.astype(np.int64) - t0) * TIME_BINS) // span,
                     0, TIME_BINS - 1)
        out["time_profile"] = _per_vmm(vmm, tb.astype(np.int64), TIME_BINS)
        out["time_range"] = np.array([t0, t1], dtype=np.int64)
    else:
        out["time_profile"] = np.zeros((NV, TIME_BINS), np.uint32)
        out["time_range"] = np.array([0, 0], dtype=np.int64)
    return out


def merge_counts(list_of_counts) -> dict:
    """Sum counts across files. The whole point of storing counts."""
    it = iter(list_of_counts)
    total = {k: np.array(v) for k, v in next(it).items()}
    for c in it:
        for k, v in c.items():
            if k == "time_range":
                total[k] = np.array([min(total[k][0], v[0]),
                                     max(total[k][1], v[1])], dtype=np.int64)
            elif k in total and total[k].shape == np.shape(v):
                total[k] = total[k] + v
    return total


# ---------------------------------------------------------------------------
# scalars — what lands on the trend dashboard
# ---------------------------------------------------------------------------

def _pct_from_counts(hist, q):
    """Percentile of a 1D histogram over bin index, without the raw data."""
    tot = hist.sum()
    if tot == 0:
        return None
    return float(np.searchsorted(np.cumsum(hist), q * tot))


def build_scalars(cols, counts, meta, station_map=None) -> dict:
    """The per-file trend record.

    Deliberately picks the quantities that have already caught real problems on
    this setup: occupancy (dead/hot hybrids), live channel count (P2_MID's ~39
    dead channels), ADC percentiles (P2_IN's compressed spectrum, i.e. the gain
    story), invalid offsets and unknown VMM ids (firmware corruption), and the
    SRS marker cadence (the 1.6384 ms aliasing).
    """
    vmm = np.asarray(cols["vmm"])
    hits_per_vmm = counts["hits_per_vmm"]
    ch_counts = counts["ch"]

    live = (ch_counts > 0).sum(axis=1)          # channels that ever fired
    adc_p50 = [_pct_from_counts(counts["adc"][v], 0.50) for v in range(NV)]
    adc_p90 = [_pct_from_counts(counts["adc"][v], 0.90) for v in range(NV)]

    # frame counter spans the capture; it is a proxy for time, not seconds
    t0, t1 = (int(counts["time_range"][0]), int(counts["time_range"][1]))

    scal = {
        "n_packets": int(meta.get("n_packets", 0)),
        "n_vm3_packets": int(meta.get("n_vm3_packets", 0)),
        "n_hits": int(meta.get("n_hits", len(vmm))),
        "n_invalid_offset": int(meta.get("n_invalid_offset", 0)),
        "offset_constant": bool(meta.get("offset_constant", False)),
        "frame_first": t0,
        "frame_last": t1,
        "hits_per_vmm": {str(v): int(hits_per_vmm[v]) for v in range(NV)
                         if hits_per_vmm[v]},
        "live_channels_per_vmm": {str(v): int(live[v]) for v in range(NV)
                                  if hits_per_vmm[v]},
        "adc_p50_per_vmm": {str(v): adc_p50[v] for v in range(NV)
                            if hits_per_vmm[v]},
        "adc_p90_per_vmm": {str(v): adc_p90[v] for v in range(NV)
                            if hits_per_vmm[v]},
        "ot_fraction": (float(counts["ot"][:, 1].sum()) /
                        max(int(counts["ot"].sum()), 1)),
    }

    # VMM ids outside the cabling are firmware corruption, not physics
    if station_map:
        known = set(station_map)
        bad = {str(v): int(hits_per_vmm[v]) for v in range(NV)
               if hits_per_vmm[v] and v not in known}
        scal["unknown_vmm_hits"] = bad
    return scal


def add_station_scalars(scal, hits, vs_mod):
    """Fold per-station aggregates in, using STATION_VMMS from vmm_stations."""
    table = getattr(vs_mod, "STATION_VMMS", None)
    if not table:
        return scal
    per_station = {}
    for station, vmms in table.items():
        sel = np.isin(np.asarray(hits["vmm"]), list(vmms))
        n = int(sel.sum())
        if not n:
            continue
        adc = np.asarray(hits["adc"])[sel]
        per_station[station] = {
            "n_hits": n,
            "adc_p50": float(np.percentile(adc, 50)),
            "adc_p90": float(np.percentile(adc, 90)),
        }
    if per_station:
        scal["per_station"] = per_station
    return scal


def station_vmms():
    """{station: {vmm ids}} from vmm_stations, or {} if it is unavailable."""
    try:
        import vmm_stations as vs
        return {str(k): set(v) for k, v in vs.STATION_VMMS.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# worker entry point
# ---------------------------------------------------------------------------

# Every column the efficiency stage actually reads, traced through
# vmm_efficiency.analyse -> build_groups/group_dt (vmm, ch, srs_timestamp,
# timestamp_ns), vmm_stations (abs_time_ns) and add_station_scalars (adc).
# Loading the full 16-column table instead costs 53 B/hit against 28 B/hit here,
# and pandas consolidates the dict into resident blocks, so the extra ten
# columns are copied out of the mmap and never read. At ~161k hits per MB of
# capture that is the difference between fitting in RAM and being memory-killed.
EFF_COLUMNS = ["vmm", "ch", "adc", "srs_timestamp", "timestamp_ns", "abs_time_ns"]


def process_file(pcap, store_dir, data_format="SRS", calibration=None,
                 do_efficiency=True, keep_store=True, eff_window=1000.0,
                 write_root=False):
    """pcapng -> column store + counts.npz + scalars.json (+ counts.root).

    The store is built under <store_dir>.partial and renamed into place only
    once everything succeeded, so a directory appearing at <store_dir> is
    complete by construction. That atomic rename is the handoff to qa_watcher:
    it never has to guess whether a store is still being written.

    write_root additionally writes counts.root, the same histograms as TH1D/TH2D
    (see vmm_root_io). Off by default: the DAQ venv at the beam has no ROOT, and
    counts.npz is what the online path reads.
    """
    import vmm_decode as vd

    t0 = time.time()
    partial = str(store_dir).rstrip("/") + ".partial"
    if os.path.isdir(partial):
        import shutil
        shutil.rmtree(partial, ignore_errors=True)

    meta = vd.decode_to(pcap, partial, data_format=data_format,
                        calibration=calibration)
    t_decode = time.time() - t0

    cols, _ = vd.load_hits(partial, derive_cols=False)
    counts = build_counts(cols)
    np.savez_compressed(os.path.join(partial, "counts.npz"), **counts)

    known = set().union(*station_vmms().values()) if station_vmms() else None
    scal = build_scalars(cols, counts, meta, known)
    scal["source"] = os.path.abspath(str(pcap))
    scal["capture"] = os.path.basename(str(pcap))
    scal["mtime"] = os.path.getmtime(pcap)
    scal["decode_s"] = round(t_decode, 2)

    # Done with the raw column mmaps; drop them before the efficiency stage
    # allocates, so their pages are reclaimable under memory pressure.
    del cols

    if do_efficiency:
        try:
            import vmm_efficiency as ve
            full, _ = vd.load_hits(partial, columns=EFF_COLUMNS)
            import pandas as pd
            hits = pd.DataFrame({c: np.asarray(full[c]) for c in EFF_COLUMNS})
            # 1000 ns matches vmm_pcapng_qa.py's --eff-window default; analyse()
            # itself defaults to 2000, which would put a different number on the
            # trend than the QA plots show for the same file.
            summary, _res = ve.analyse(hits, window_ns=eff_window)
            scal["efficiency"] = {
                k: {kk: vv for kk, vv in v.items()
                    if isinstance(vv, (int, float))}
                for k, v in (summary.get("stations") or summary).items()
                if isinstance(v, dict)
            }
            scal["n_masked_hits"] = int(summary.get("n_masked_hits", 0))
            try:
                import vmm_stations as vs
                add_station_scalars(scal, hits, vs)
            except Exception:
                pass
            del hits, full
        except Exception as e:
            # Efficiency is the most fragile stage; never let it lose the decode
            scal["efficiency_error"] = f"{type(e).__name__}: {e}"

    scal["reduce_s"] = round(time.time() - t0 - t_decode, 2)
    with open(os.path.join(partial, "scalars.json"), "w") as f:
        json.dump(scal, f, indent=2, default=str)

    if write_root:
        # Imported here, not at module level, so the online path never touches
        # ROOT. Treated like the efficiency stage: a failure must not lose the
        # decode, since counts.npz already holds the same numbers.
        try:
            import vmm_root_io
            t_root = time.time()
            vmm_root_io.write_counts_root(
                counts, os.path.join(partial, "counts.root"), scalars=scal)
            scal["root_s"] = round(time.time() - t_root, 2)
        except Exception as e:
            scal["root_error"] = f"{type(e).__name__}: {e}"
        with open(os.path.join(partial, "scalars.json"), "w") as f:
            json.dump(scal, f, indent=2, default=str)

    if os.path.isdir(str(store_dir)):
        import shutil
        shutil.rmtree(str(store_dir), ignore_errors=True)
    os.rename(partial, str(store_dir))          # <- the atomic handoff

    if not keep_store:
        # counts + scalars are the archive; the columns are regenerable
        for f in os.listdir(store_dir):
            if f.endswith(".npy"):
                os.unlink(os.path.join(store_dir, f))

    scal["total_s"] = round(time.time() - t0, 2)
    return scal


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap_file")
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--format", default="SRS", choices=["SRS", "TRG"])
    ap.add_argument("--no-efficiency", action="store_true")
    ap.add_argument("--eff-window", type=float, default=1000.0, metavar="NS")
    ap.add_argument("--drop-columns", action="store_true",
                    help="keep only counts.npz + scalars.json (columns are "
                         "regenerable from the pcapng in ~0.5 s)")
    ap.add_argument("--root", action="store_true",
                    help="also write counts.root (TH1D/TH2D of the same "
                         "histograms; merge with hadd). Needs ROOT, so it is "
                         "off on the beam machine.")
    args = ap.parse_args()

    scal = process_file(args.pcap_file, args.store_dir,
                        data_format=args.format,
                        do_efficiency=not args.no_efficiency,
                        keep_store=not args.drop_columns,
                        eff_window=args.eff_window,
                        write_root=args.root)
    print(f"{os.path.basename(args.pcap_file)}: {scal['n_hits']:,} hits  "
          f"decode={scal['decode_s']}s  reduce={scal['reduce_s']}s  "
          f"total={scal['total_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
