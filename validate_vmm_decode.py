#!/usr/bin/env python3
"""Validate vmm_decode.decode() against the live vmm_pcapng_qa.py parser.

The reference DataFrame is not reimplemented here. Instead the real QA script
is exec'd with pandas.DataFrame monkeypatched, so the moment it builds the hits
table we grab that exact object and abort before it spends 60 s plotting. What
is compared is therefore the code that is actually running at the beam.

Usage:
    python validate_vmm_decode.py <file.pcapng> [more.pcapng ...] [--format SRS]
"""

import argparse
import io
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

QA_SCRIPT = "/local/p2/DAQ_Control_VMM_Beam/vmm_qa/vmm_pcapng_qa.py"

COLUMNS = [
    "fec", "vmm", "time", "ch", "adc", "adc_calibrated", "over_threshold",
    "offset", "bcid", "tdc", "timestamp_ns", "srs_timestamp", "abs_time_ns",
    "trigger_time", "trigger_counter", "hit_valid",
]


class _Stop(Exception):
    """Raised once the reference hits DataFrame has been captured."""


def reference_hits(pcap, data_format="SRS", max_packets=None, qa_script=QA_SCRIPT):
    """Run the live QA script far enough to build its hits DataFrame, then stop."""
    captured = {}
    real_df = pd.DataFrame

    def spy(*a, **k):
        df = real_df(*a, **k)
        try:
            cols = set(df.columns)
        except Exception:
            return df
        if {"srs_timestamp", "hit_valid", "bcid"} <= cols:
            captured["hits"] = df.copy()
            raise _Stop()
        return df

    argv = [qa_script, pcap, "--format", data_format,
            "--out-dir", "/tmp/_vmm_validate_unused"]
    if max_packets:
        argv += ["--max-packets", str(max_packets)]

    old_argv, old_stdout = sys.argv, sys.stdout
    src = open(qa_script).read()
    g = {"__name__": "__main__", "__file__": qa_script}
    sys.argv = argv
    sys.stdout = io.StringIO()
    pd.DataFrame = spy
    t0 = time.time()
    try:
        exec(compile(src, qa_script, "exec"), g)
    except _Stop:
        pass
    except SystemExit as e:
        if "hits" not in captured:
            raise RuntimeError(f"QA script exited early ({e}) before building hits")
    finally:
        pd.DataFrame = real_df
        sys.argv, sys.stdout = old_argv, old_stdout
    if "hits" not in captured:
        raise RuntimeError("never captured the reference DataFrame")
    return captured["hits"], time.time() - t0


def compare(ref, new):
    """Compare every column. Returns (n_bad, list of report lines)."""
    lines = []
    bad = 0

    if len(ref) != len(new):
        return 1, [f"  ROW COUNT differs: reference {len(ref):,} vs new {len(new):,}"]
    lines.append(f"  rows: {len(ref):,} (match)")

    missing = [c for c in COLUMNS if c not in new.columns]
    extra = [c for c in new.columns if c not in COLUMNS]
    if missing:
        bad += 1
        lines.append(f"  MISSING COLUMNS: {missing}")
    if extra:
        lines.append(f"  (extra columns, not an error: {extra})")

    for c in COLUMNS:
        if c not in ref.columns or c not in new.columns:
            continue
        a, b = ref[c].to_numpy(), new[c].to_numpy()
        dt_ok = a.dtype == b.dtype
        if a.dtype.kind == "f":
            same = np.array_equal(a, b, equal_nan=True)
            if same:
                note = "exact"
            else:
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                worst = float(np.nanmax(d)) if len(d) else 0.0
                n_diff = int((d > 0).sum())
                same = worst == 0.0
                note = f"max|delta|={worst:.6g} on {n_diff:,} rows"
        else:
            same = np.array_equal(a, b)
            n_diff = int((a != b).sum()) if not same else 0
            note = "exact" if same else f"{n_diff:,} rows differ"

        status = "OK  " if (same and dt_ok) else "FAIL"
        if not (same and dt_ok):
            bad += 1
        dt_note = "" if dt_ok else f"  DTYPE {a.dtype} vs {b.dtype}"
        lines.append(f"  [{status}] {c:16s} {note}{dt_note}")
    return bad, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--format", default="SRS", choices=["SRS", "TRG"])
    ap.add_argument("--max-packets", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vmm_decode

    total_bad = 0
    for pcap in args.files:
        print(f"\n=== {os.path.basename(pcap)} "
              f"({os.path.getsize(pcap)/1e6:.1f} MB, format={args.format}) ===")
        try:
            ref, t_ref = reference_hits(pcap, args.format, args.max_packets)
        except Exception as e:
            print(f"  reference parser FAILED: {type(e).__name__}: {e}")
            total_bad += 1
            continue

        t0 = time.time()
        new, meta = vmm_decode.decode(pcap, data_format=args.format,
                                      max_packets=args.max_packets)
        t_new = time.time() - t0

        bad, lines = compare(ref, new)
        for ln in lines:
            print(ln)
        speed = t_ref / t_new if t_new > 0 else float("inf")
        print(f"  reference {t_ref:6.2f} s   new {t_new:5.2f} s   speedup {speed:5.1f}x")

        # Chunking must not change a single value: force many chunks and
        # re-compare. This is what exercises the cross-chunk marker carry.
        del ref
        small = max(1000, meta["n_words"] // 7)
        chunked, cmeta = vmm_decode.decode(pcap, data_format=args.format,
                                           max_packets=args.max_packets,
                                           max_words=small)
        cbad, _ = compare(new, chunked)
        print(f"  [{'OK  ' if cbad == 0 else 'FAIL'}] chunk invariance   "
              f"{cmeta['n_chunks']} chunks vs {meta['n_chunks']} "
              f"({'identical' if cbad == 0 else f'{cbad} column(s) differ'})")
        bad += cbad
        del chunked

        # Streaming round-trip: decode_to writes only the raw columns and
        # load_frame recomputes the derived ones, so this checks the store
        # AND that derive() reproduces what the in-memory path produced.
        store = tempfile.mkdtemp(prefix="vmmstore_")
        try:
            smeta = vmm_decode.decode_to(pcap, store, data_format=args.format,
                                         max_packets=args.max_packets,
                                         max_words=small)
            back, _ = vmm_decode.load_frame(store)
            sbad, slines = compare(new, back)
            nbytes = sum(os.path.getsize(os.path.join(store, f))
                         for f in os.listdir(store))
            print(f"  [{'OK  ' if sbad == 0 else 'FAIL'}] store round-trip   "
                  f"{smeta['n_hits']:,} hits, {nbytes/1e6:.1f} MB on disk "
                  f"({'identical' if sbad == 0 else f'{sbad} column(s) differ'})")
            if sbad:
                for ln in slines:
                    if "FAIL" in ln:
                        print(f"    {ln.strip()}")
            bad += sbad
            del back
        finally:
            shutil.rmtree(store, ignore_errors=True)
        del new

        print(f"  verdict: {'ALL COLUMNS IDENTICAL' if bad == 0 else f'{bad} MISMATCH(ES)'}")
        total_bad += bad

    print(f"\n{'='*60}\n{'PASS' if total_bad == 0 else f'FAIL ({total_bad} problems)'}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
