#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROOT writer/reader for the Tier-1 reduce product.

vmm_reduce builds every QA quantity as a per-VMM numpy histogram (see
build_counts). This module is the thin adapter that puts those same arrays into
a TFile of TH1D/TH2D, and reads them back into the identical dict, so the
plotting half can consume either format:

    counts.npz   numpy, additive by summing arrays      (no ROOT needed)
    counts.root  TH1D/TH2D, additive by `hadd`          (needs ROOT)

Deliberately NOT imported by vmm_reduce at module level. The DAQ venv at the
beam has no ROOT -- that is why vmm_pcapng_qa.py is PNG-only -- so `import ROOT`
must never happen on the online path. vmm_reduce imports this lazily, only when
--root is passed.

Binning mirrors vmm_reduce's constants exactly, so a round trip through ROOT
returns arrays identical to what went in:

    per VMM v with hits, directory "vmm_<v>":
        h_adc, h_adc_ot0, h_adc_ot1   1024 bins, [0, 1024)
        h_ch                            64 bins, [0, 64)
        h_bcid                        4096 bins, [0, 4096)
        h_tdc                          256 bins, [0, 256)
        h_offset                        32 bins, [-16, +16)   (unshifted)
        h_ot                             2 bins, [0, 2)
        h_time_profile                 200 bins, [t0, t1]
        h_adc_vs_ch                  64 x 128, channel vs adc>>3
    top level:
        h_hits_per_vmm                  32 bins, [0, 32)
        time_range                      TVectorD(2), the frame-counter span
        scalars                         TNamed, scalars.json verbatim

Merging. The histograms are plain bin counts, so `hadd` sums them correctly and
is the fast path:

    hadd merged.root <store>/*/counts.root

with one caveat. `time_range` is a TVectorD and `hadd` has no Merge() for it, so
a hadd-ed file carries the FIRST input's span, not the union; and `hadd` can only
sum h_time_profile when every input shares the same frame-counter axis. Both are
moot while the frame counter is constant (it is, on the July 2026 data -- every
capture reports [20, 20]), but neither is guaranteed. Use merge_root() when you
need the same semantics as vmm_reduce.merge_counts.

Usage:
    python vmm_reduce.py <capture.pcapng> --store-dir <dir> --root
"""

import json

import numpy as np

# Kept in sync with vmm_reduce; imported rather than duplicated where possible.
NV = 32
ADC_BINS = 1024
CH_BINS = 64
BCID_BINS = 4096
TDC_BINS = 256
OFFSET_BINS = 32
ADC_CH_ADC_BINS = 128
TIME_BINS = 200

# name -> (nbins, xlow, xhigh). The offset axis is shifted back to signed here:
# vmm_reduce stores it as index+16 so it can use bincount.
_H1 = {
    "adc":       (ADC_BINS, 0.0, float(ADC_BINS)),
    "adc_ot0":   (ADC_BINS, 0.0, float(ADC_BINS)),
    "adc_ot1":   (ADC_BINS, 0.0, float(ADC_BINS)),
    "ch":        (CH_BINS, 0.0, float(CH_BINS)),
    "bcid":      (BCID_BINS, 0.0, float(BCID_BINS)),
    "tdc":       (TDC_BINS, 0.0, float(TDC_BINS)),
    "offset":    (OFFSET_BINS, -16.0, 16.0),
    "ot":        (2, 0.0, 2.0),
}

_AXIS_TITLE = {
    "adc": "ADC", "adc_ot0": "ADC (over_threshold = 0)",
    "adc_ot1": "ADC (over_threshold = 1)", "ch": "channel",
    "bcid": "BCID", "tdc": "TDC", "offset": "offset",
    "ot": "over_threshold",
}


def _pad1(vals):
    """numpy bin contents -> the (nbins+2) buffer TH1::SetContent expects.

    ROOT's internal array carries underflow at 0 and overflow at nbins+1; the
    reduce histograms are exhaustive over their range, so both stay empty.
    """
    buf = np.zeros(len(vals) + 2, dtype=np.float64)
    buf[1:-1] = vals
    return buf


def _pad2(mat):
    """(nx, ny) bin contents -> the TH2::SetContent buffer.

    ROOT flattens a 2D histogram as bin = binx + (nbinsx + 2) * biny, so the
    padded array is transposed relative to the natural numpy layout.
    """
    nx, ny = mat.shape
    buf = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    buf[1:-1, 1:-1] = np.asarray(mat, dtype=np.float64).T
    return buf.ravel()


def write_counts_root(counts, path, scalars=None):
    """Write the reduce counts dict to `path` as a TFile of TH1D/TH2D.

    `counts` is exactly what vmm_reduce.build_counts returned (or what
    merge_counts summed). Only VMMs that saw hits get a directory, which keeps
    a quiet capture's file small and makes `hadd` cheap.
    """
    import ROOT
    ROOT.gROOT.SetBatch(True)

    hits_per_vmm = np.asarray(counts["hits_per_vmm"])
    t0, t1 = (int(counts["time_range"][0]), int(counts["time_range"][1]))
    # A zero-width axis makes ROOT complain; give a degenerate span one unit.
    t_hi = float(t1) if t1 > t0 else float(t0 + 1)

    f = ROOT.TFile(str(path), "RECREATE")
    try:
        # ROOT owns histograms created while a TFile is open; detaching them
        # would mean managing the lifetime by hand, so just let the file own it.
        h_tot = ROOT.TH1D("h_hits_per_vmm", "hits per VMM;VMM;hits", NV, 0, NV)
        h_tot.SetContent(_pad1(hits_per_vmm.astype(np.float64)))
        h_tot.SetEntries(float(hits_per_vmm.sum()))

        tr = ROOT.TVectorD(2)
        tr[0], tr[1] = float(t0), float(t1)
        tr.Write("time_range")

        if scalars is not None:
            ROOT.TNamed("scalars", json.dumps(scalars, default=str)).Write()

        for v in range(NV):
            if not hits_per_vmm[v]:
                continue
            d = f.mkdir(f"vmm_{v}")
            d.cd()

            for key, (nb, lo, hi) in _H1.items():
                vals = np.asarray(counts[key][v], dtype=np.float64)
                h = ROOT.TH1D(f"h_{key}", f"VMM {v} {key};"
                                          f"{_AXIS_TITLE[key]};hits", nb, lo, hi)
                h.SetContent(_pad1(vals))
                h.SetEntries(float(vals.sum()))

            tp = np.asarray(counts["time_profile"][v], dtype=np.float64)
            h = ROOT.TH1D("h_time_profile",
                          f"VMM {v} rate vs frame counter;frame counter;hits",
                          TIME_BINS, float(t0), t_hi)
            h.SetContent(_pad1(tp))
            h.SetEntries(float(tp.sum()))

            # adc_vs_ch is stored with ADC coarsened 8x (adc >> 3), so the y
            # axis spans the full 0-1024 ADC range in 128 bins.
            m = np.asarray(counts["adc_vs_ch"][v], dtype=np.float64)
            h2 = ROOT.TH2D("h_adc_vs_ch", f"VMM {v} ADC vs channel;channel;ADC",
                           CH_BINS, 0, CH_BINS, ADC_CH_ADC_BINS, 0, ADC_BINS)
            h2.SetContent(_pad2(m))
            h2.SetEntries(float(m.sum()))

            f.cd()

        f.Write()
    finally:
        f.Close()
    return str(path)


def read_counts_root(path):
    """Read a counts.root back into the dict shape vmm_reduce produces.

    The inverse of write_counts_root, so plotting code can take either format:
        counts = (read_counts_root(p) if p.endswith('.root')
                  else dict(np.load(p)))
    Works on a `hadd`-merged file too -- summing TH1s and summing the numpy
    arrays give the same result.
    """
    import ROOT
    ROOT.gROOT.SetBatch(True)

    f = ROOT.TFile(str(path), "READ")
    if f.IsZombie():
        raise IOError(f"cannot open {path}")
    try:
        out = {k: np.zeros((NV, nb), dtype=np.uint32) for k, (nb, _, _) in _H1.items()}
        out["time_profile"] = np.zeros((NV, TIME_BINS), dtype=np.uint32)
        out["adc_vs_ch"] = np.zeros((NV, CH_BINS, ADC_CH_ADC_BINS), dtype=np.uint32)

        h_tot = f.Get("h_hits_per_vmm")
        hits_per_vmm = np.array(
            [h_tot.GetBinContent(i + 1) for i in range(NV)] if h_tot
            else [0] * NV, dtype=np.uint64)
        out["hits_per_vmm"] = hits_per_vmm

        tr = f.Get("time_range")
        out["time_range"] = (np.array([int(tr[0]), int(tr[1])], dtype=np.int64)
                             if tr else np.array([0, 0], dtype=np.int64))

        for v in range(NV):
            d = f.Get(f"vmm_{v}")
            if not d:
                continue
            for key, (nb, _, _) in _H1.items():
                h = d.Get(f"h_{key}")
                if h:
                    out[key][v] = [h.GetBinContent(i + 1) for i in range(nb)]
            h = d.Get("h_time_profile")
            if h:
                out["time_profile"][v] = [h.GetBinContent(i + 1)
                                          for i in range(TIME_BINS)]
            h2 = d.Get("h_adc_vs_ch")
            if h2:
                out["adc_vs_ch"][v] = [
                    [h2.GetBinContent(ix + 1, iy + 1)
                     for iy in range(ADC_CH_ADC_BINS)]
                    for ix in range(CH_BINS)]
        return out
    finally:
        f.Close()


def merge_root(paths, out_path, scalars=None):
    """Merge counts.root files with vmm_reduce.merge_counts semantics.

    `hadd` is faster and equivalent for the histograms, but mishandles
    time_range (see the module docstring). This route reads each file back to
    numpy, merges with the same function the npz path uses, and rewrites -- so
    ROOT and npz merges are guaranteed to agree.
    """
    import vmm_reduce
    paths = list(paths)
    if not paths:
        raise ValueError("nothing to merge")
    merged = vmm_reduce.merge_counts(read_counts_root(p) for p in paths)
    return write_counts_root(merged, out_path, scalars=scalars)


def read_scalars_root(path):
    """The scalars.json dict stored alongside the histograms, or None."""
    import ROOT
    f = ROOT.TFile(str(path), "READ")
    try:
        obj = f.Get("scalars")
        return json.loads(obj.GetTitle()) if obj else None
    finally:
        f.Close()
