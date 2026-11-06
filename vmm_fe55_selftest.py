#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_fe55_selftest.py — write a synthetic det1 Fe55 capture with a KNOWN source
position and a KNOWN gain field, run vmm_fe55_map.py on it, and check that both
come back.

Why bother
----------
Every estimator in vmm_fe55_map.py can run happily on real data and be wrong:
a within-half ordering off by one scatters the spot, a mis-parsed `if_tsresol`
scales every rate by 1000, a centroid biased by the fan edge points the source
in the wrong direction. None of that shows up as an exception. The only way to
know the chain works before the real Fe55 data exists is to feed it data whose
answer is known.

The generator writes a real pcapng -- SHB/IDB/EPB blocks, Ethernet/IPv4/UDP
frames, SRS VM3 payloads with the same bit layout vmm_decode.py unpacks -- so
the test exercises the actual decoder, not a mock of it.

Truth model
-----------
  * a 2D Gaussian source spot on the instrumented pads;
  * a linear gain gradient across the detector, so the recovered gain map has
    structure to reproduce rather than a single flat number;
  * an Fe55 spectrum per pad: the 5.9 keV photopeak plus the Ar escape peak at
    0.50 x charge, both Gaussian, at a set relative resolution;
  * a uniform low-ADC noise floor on every instrumented pad, so the photopeak
    finder has to cope with a background the way it will on real data.

Usage
-----
  python3 vmm_fe55_selftest.py                     # generate, analyse, check
  python3 vmm_fe55_selftest.py --keep --out-dir /tmp/fe55   # keep the products
  python3 vmm_fe55_selftest.py --generate-only out.pcapng   # just the file

@author: ak271430 Alexandra Kallitsopoulou
"""

import os
import sys
import json
import struct
import argparse
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmm_det1_cabling as dc
import vmm_fe55_map as fm

# --------------------------------------------------------------------------
# pcapng / Ethernet / UDP framing
# --------------------------------------------------------------------------
_SHB = 0x0A0D0D0A
_IDB = 0x00000001
_EPB = 0x00000006
_BOM = 0x1A2B3C4D

FEC_IP = "192.168.1.13"          # what the DAQ's FEC actually uses
HOST_IP = "192.168.1.10"
FEC_ID = 0


def _pad4(b):
    return b + b"\x00" * ((-len(b)) % 4)


def _shb():
    body = struct.pack("<IHHq", _BOM, 1, 0, -1) + struct.pack("<HH", 0, 0)
    blen = 12 + len(body)
    return struct.pack("<II", _SHB, blen) + body + struct.pack("<I", blen)


def _idb(tsresol=9):
    # linktype 1 (Ethernet), snaplen 0 (unlimited), if_tsresol = 10^-tsresol.
    # tsresol is written explicitly because the pcapng DEFAULT is microseconds
    # while dumpcap writes nanoseconds -- a live-time factor of 1000, and the
    # exact bug this test exists to catch.
    body = struct.pack("<HHI", 1, 0, 0)
    body += struct.pack("<HH", 9, 1) + _pad4(bytes([tsresol]))
    body += struct.pack("<HH", 0, 0)
    blen = 12 + len(body)
    return struct.pack("<II", _IDB, blen) + body + struct.pack("<I", blen)


def _epb(data, t_ns):
    d = _pad4(data)
    body = struct.pack("<IIIII", 0, (t_ns >> 32) & 0xFFFFFFFF,
                       t_ns & 0xFFFFFFFF, len(data), len(data)) + d
    blen = 12 + len(body)
    return struct.pack("<II", _EPB, blen) + body + struct.pack("<I", blen)


def _ip4(s):
    return bytes(int(x) for x in s.split("."))


def _udp_frame(payload, src_ip=FEC_IP, dst_ip=HOST_IP, sport=6006, dport=6006):
    udp = struct.pack(">HHHH", sport, dport, 8 + len(payload), 0) + payload
    total = 20 + len(udp)
    ip = (struct.pack(">BBHHHBBH", 0x45, 0, total, 0, 0, 64, 17, 0)
          + _ip4(src_ip) + _ip4(dst_ip))
    eth = b"\x00\x11\x22\x33\x44\x55" + b"\x66\x77\x88\x99\xaa\xbb" \
        + struct.pack(">H", 0x0800)
    return eth + ip + udp


def _bin2gray(v):
    v = np.asarray(v, dtype=np.uint32)
    return (v ^ (v >> 1)).astype(np.uint32)


def _srs_payload(frame_counter, vmm, ch, adc, bcid, tdc, over_threshold=1,
                 marker_vmms=(), marker_value=0):
    """One VM3 UDP payload: 16-byte header, then 6-byte words.

    Bit layout, matching vmm_decode.iter_chunks exactly:
      d1 (32b): [31:27] offset  [26:22] vmm  [21:12] adc  [11:0] bcid (gray)
      d2 (16b): [15] 1=hit  [14] over_threshold  [13:8] ch  [7:0] tdc
      marker  : d2[15]=0, d2[14:10]=vmm id, srs_ts = (d1 << 10) | d2[9:0]
    """
    words = []
    for v in marker_vmms:
        d1 = (marker_value >> 10) & 0xFFFFFFFF
        d2 = ((v & 0x1F) << 10) | (marker_value & 0x3FF)
        words.append(struct.pack(">IH", d1, d2))
    g = _bin2gray(np.asarray(bcid))
    for i in range(len(vmm)):
        d1 = ((0 & 0x1F) << 27) | ((int(vmm[i]) & 0x1F) << 22) \
            | ((int(adc[i]) & 0x3FF) << 12) | (int(g[i]) & 0xFFF)
        d2 = 0x8000 | ((over_threshold & 1) << 14) \
            | ((int(ch[i]) & 0x3F) << 8) | (int(tdc[i]) & 0xFF)
        words.append(struct.pack(">IH", d1, d2))

    # plen - 22 must round up to exactly len(words) 6-byte words, so the
    # payload is a 16-byte header + the words + 6 trailing bytes.
    header = struct.pack(">I", frame_counter & 0xFFFFFFFF) \
        + b"VM3" + bytes([(FEC_ID & 0x0F) << 4]) \
        + struct.pack(">II", 0, 0)
    return header + b"".join(words) + b"\x00" * 6


def write_pcapng(path, vmm, ch, adc, t_start=1.757e9, duration_s=60.0,
                 hits_per_packet=400):
    """Write the hits as a valid pcapng, in time order."""
    n = len(vmm)
    order = np.argsort(np.random.rand(n))          # arrival order is arbitrary
    vmm, ch, adc = vmm[order], ch[order], adc[order]
    bcid = np.random.randint(0, 4096, n)
    tdc = np.random.randint(0, 256, n)

    n_pkt = max(1, int(np.ceil(n / hits_per_packet)))
    t0_ns = int(t_start * 1e9)
    dt_ns = int(duration_s * 1e9 / max(n_pkt - 1, 1))

    with open(path, "wb") as f:
        f.write(_shb())
        f.write(_idb(tsresol=9))
        for p in range(n_pkt):
            lo, hi = p * hits_per_packet, min((p + 1) * hits_per_packet, n)
            # One SRS marker per packet, advancing at the nominal 25 ns tick,
            # so the SRS cross-check in live_time() has something to compare.
            mval = int((p * dt_ns) / 25)
            payload = _srs_payload(p, vmm[lo:hi], ch[lo:hi], adc[lo:hi],
                                   bcid[lo:hi], tdc[lo:hi],
                                   marker_vmms=sorted(set(int(v) for v
                                                          in vmm[lo:hi])),
                                   marker_value=mval)
            f.write(_epb(_udp_frame(payload), t0_ns + p * dt_ns))
    return path


# --------------------------------------------------------------------------
# Truth model
# --------------------------------------------------------------------------
class Truth:
    """The answer the analysis is supposed to recover."""

    def __init__(self, tab, x0=None, y0=None, sigma_mm=18.0, n_photons=250_000,
                 peak_adc=520.0, resolution_fwhm=0.20, escape_frac=0.12,
                 noise_frac=0.06, gain_grad_per_100mm=0.10, seed=1234,
                 gain_ref_x=None):
        live = tab[tab["mapped"]].reset_index(drop=True)
        self.pads = live
        self.rng = np.random.default_rng(seed)
        w = live["pad_area"].to_numpy(float)
        cx = live["pad_cx"].to_numpy(float)
        cy = live["pad_cy"].to_numpy(float)
        # Default the spot to the area centroid so it is fully contained and
        # the recovered position is not fighting the fan edge.
        self.x0 = float((cx * w).sum() / w.sum()) if x0 is None else float(x0)
        self.y0 = float((cy * w).sum() / w.sum()) if y0 is None else float(y0)
        self.sigma = float(sigma_mm)
        self.n_photons = int(n_photons)
        self.peak_adc = float(peak_adc)
        self.resolution_fwhm = float(resolution_fwhm)
        self.escape_frac = float(escape_frac)
        self.noise_frac = float(noise_frac)
        self.grad = float(gain_grad_per_100mm)

        # Relative gain: linear in x. The reference x defaults to the spot, but
        # a MULTI-POSITION test must pin it to the same place for every run --
        # otherwise each run gets its own gain field and there is nothing
        # common for the stitch to recover.
        self.gain_ref_x = self.x0 if gain_ref_x is None else float(gain_ref_x)
        self.rel_gain = 1.0 + self.grad * (cx - self.gain_ref_x) / 100.0
        self.pad_peak = self.peak_adc * self.rel_gain

    def sample(self):
        """Return (vmm, ch, adc) arrays for the synthetic capture."""
        cx = self.pads["pad_cx"].to_numpy(float)
        cy = self.pads["pad_cy"].to_numpy(float)
        area = self.pads["pad_area"].to_numpy(float)
        p = area * np.exp(-0.5 * (((cx - self.x0) ** 2 + (cy - self.y0) ** 2)
                                  / self.sigma ** 2))
        p = p / p.sum()

        idx = self.rng.choice(len(self.pads), size=self.n_photons, p=p)
        peak = self.pad_peak[idx]
        sig = peak * self.resolution_fwhm / 2.355
        adc = self.rng.normal(peak, sig)
        # Ar escape: a fraction of conversions lose 2.96 keV out of the gas.
        esc = self.rng.random(self.n_photons) < self.escape_frac
        adc[esc] = self.rng.normal(peak[esc] * fm.ESCAPE_FRACTION,
                                   sig[esc] * np.sqrt(fm.ESCAPE_FRACTION))

        # Uniform noise floor on every instrumented pad, low ADC.
        n_noise = int(self.noise_frac * self.n_photons)
        nidx = self.rng.integers(0, len(self.pads), n_noise)
        nadc = self.rng.exponential(30.0, n_noise) + 5.0

        idx = np.concatenate([idx, nidx])
        adc = np.concatenate([adc, nadc])
        adc = np.clip(np.round(adc), 0, fm.ADC_MAX).astype(np.int64)
        return (self.pads["vmm"].to_numpy()[idx].astype(np.int64),
                self.pads["ch"].to_numpy()[idx].astype(np.int64),
                adc)

    def as_dict(self):
        return {"x0_mm": self.x0, "y0_mm": self.y0, "sigma_mm": self.sigma,
                "n_photons": self.n_photons, "peak_adc": self.peak_adc,
                "resolution_fwhm_pct": 100 * self.resolution_fwhm,
                "escape_frac": self.escape_frac,
                "gain_gradient_per_100mm": self.grad}


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check(truth, summary, pads_csv, duration_s, tol):
    """Compare recovered against truth. Returns (ok, list of result lines)."""
    import pandas as pd
    out = []
    ok = True

    def rec(name, got, want, tolerance, unit=""):
        nonlocal ok
        if got is None or not np.isfinite(got):
            ok = False
            out.append(f"  FAIL {name}: not measured (truth {want:.3g}{unit})")
            return
        d = abs(got - want)
        good = d <= tolerance
        ok = ok and good
        out.append(f"  {'ok  ' if good else 'FAIL'} {name}: got {got:.4g}{unit}"
                   f", truth {want:.4g}{unit}, |diff| {d:.3g} "
                   f"(tol {tolerance:g})")

    fit = (summary.get("source") or {}).get("fit") or {}
    rec("source x0", fit.get("x0_mm"), truth.x0, tol["pos_mm"], " mm")
    rec("source y0", fit.get("y0_mm"), truth.y0, tol["pos_mm"], " mm")
    rec("spot sigma", fit.get("sigma_major_mm"), truth.sigma,
        tol["sigma_mm"], " mm")

    cen = summary.get("source") or {}
    rec("centroid x", cen.get("centroid_x_mm"), truth.x0, tol["centroid_mm"],
        " mm")
    rec("centroid y", cen.get("centroid_y_mm"), truth.y0, tol["centroid_mm"],
        " mm")

    spec = summary.get("spectrum") or {}
    rec("photopeak", spec.get("peak_adc"), truth.peak_adc, tol["peak_adc"],
        " ADC")
    rec("resolution FWHM", spec.get("resolution_fwhm_pct"),
        100 * truth.resolution_fwhm, tol["res_pct"], " %")

    total_expected = (truth.n_photons * (1 + truth.noise_frac)) / duration_s
    rec("total rate", (summary.get("rate") or {}).get("total_hz"),
        total_expected, tol["rate_frac"] * total_expected, " Hz")

    # The gain map is the point of the exercise: does the recovered per-pad
    # peak reproduce the injected gradient, or only its average?
    t = pd.read_csv(pads_csv).set_index("channel_id")
    truth_gain = dict(zip(truth.pads["channel_id"], truth.rel_gain))
    m = t["rel_gain"].notna()
    if m.sum() < 10:
        ok = False
        out.append(f"  FAIL gain map: only {int(m.sum())} pads measured")
    else:
        got = t.loc[m, "rel_gain"].to_numpy()
        want = np.array([truth_gain[i] for i in t.index[m]])
        want = want / np.median(want)            # both are median-normalised
        resid = np.std(got - want)
        r = np.corrcoef(got, want)[0, 1] if len(got) > 2 else np.nan
        good = resid <= tol["gain_rms"] and r >= tol["gain_corr"]
        ok = ok and good
        out.append(f"  {'ok  ' if good else 'FAIL'} gain map over "
                   f"{int(m.sum())} pads: rms(recovered - truth) = {resid:.4f} "
                   f"(tol {tol['gain_rms']}), correlation r = {r:.3f} "
                   f"(tol {tol['gain_corr']})")

    cab = summary.get("cabling_check") or {}
    good = bool(cab.get("consistent"))
    ok = ok and good
    out.append(f"  {'ok  ' if good else 'FAIL'} cabling check consistent")

    lt = summary.get("live_time") or {}
    rec("live time", lt.get("live_s"), duration_s, tol["live_s"], " s")
    return ok, out


DEFAULT_TOL = {
    "pos_mm": 2.0,          # a pad is 12 mm; 2 mm is well inside one
    "centroid_mm": 4.0,     # the centroid is the cruder of the two estimators
    "sigma_mm": 3.0,
    "peak_adc": 25.0,
    "res_pct": 4.0,
    "rate_frac": 0.05,
    "gain_rms": 0.05,       # 5 % per-pad scatter on a +-10 %/100 mm gradient
    "gain_corr": 0.80,
    "live_s": 2.0,
}


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
def _generate(tab, out_dir, name, args, x0=None, y0=None, uniform=False,
              **kw):
    truth = Truth(tab, x0=x0, y0=y0, sigma_mm=args.sigma,
                  n_photons=args.photons, seed=args.seed, **kw)
    if uniform:
        truth.sigma = 1e6            # flat illumination over the live pads
    vmm, ch, adc = truth.sample()
    pcap = os.path.join(out_dir, f"{name}.pcapng")
    write_pcapng(pcap, vmm, ch, adc, duration_s=args.duration)
    return truth, pcap


def scenario_nominal(args, out_dir):
    """Spot in the middle of the instrumented area: everything should be
    recovered to within a fraction of a pad."""
    tab = dc.build_pad_table()
    truth, pcap = _generate(tab, out_dir, "nominal", args,
                            x0=args.x0, y0=args.y0)
    print(f"truth: {json.dumps(truth.as_dict(), indent=2)}\n")
    if fm.main([pcap, "--out-dir", out_dir, "--name", "nominal",
                "--min-events", "40", "--fit-min", "120"]):
        return False, ["  FAIL analysis returned non-zero"]
    with open(os.path.join(out_dir, "nominal_summary.json")) as f:
        summary = json.load(f)
    return check(truth, summary, os.path.join(out_dir, "nominal_pads.csv"),
                 args.duration, DEFAULT_TOL)


def scenario_clipped(args, out_dir):
    """Spot on the edge of the live sectors. The position is EXPECTED to be
    wrong -- what must work is that the analysis says so, instead of reporting
    a confident centre that would send the source the wrong way."""
    tab = dc.build_pad_table()
    live = tab[tab["mapped"]]
    edge = live.loc[live["phi"].idxmax()]         # against the dead sector 8
    truth, pcap = _generate(tab, out_dir, "clipped", args,
                            x0=float(edge["pad_cx"]), y0=float(edge["pad_cy"]))
    if fm.main([pcap, "--out-dir", out_dir, "--name", "clipped",
                "--min-events", "40", "--fit-min", "120"]):
        return False, ["  FAIL analysis returned non-zero"]
    with open(os.path.join(out_dir, "clipped_summary.json")) as f:
        s = json.load(f)
    c = (s.get("source") or {}).get("containment")
    warned = any("clipped" in w for w in s.get("warnings", []))
    ok = c is not None and c < 0.9 and warned
    return ok, [f"  {'ok  ' if ok else 'FAIL'} clipped spot flagged: "
                f"containment {c if c is None else round(c, 3)}, "
                f"warning {'raised' if warned else 'MISSING'}"]


def scenario_wrong_mapping(args, out_dir):
    """Generate through 'reverse', scan starting from 'linear'. Every half the
    scan calls determined must come back 'reverse', and none may be a false
    positive on a half the source never lit."""
    tab = dc.build_pad_table(strategy="reverse")
    _, pcap = _generate(tab, out_dir, "wrongmap", args)
    ns = fm.build_parser().parse_args([pcap, "--strategy", "linear",
                                       "--scan-mapping"])
    determined = fm.scan_mapping([pcap], ns)
    wrong = {k: v for k, v in determined.items() if v != "reverse"}
    ok = bool(determined) and not wrong
    return ok, [f"  {'ok  ' if ok else 'FAIL'} mapping scan: "
                f"{len(determined)} halves determined, "
                f"{len(wrong)} disagree with the injected 'reverse'"
                + (f" {wrong}" if wrong else "")]


def scenario_cabling(args, out_dir):
    """Disconnect a DIFFERENT pair of connectors than the analysis declares.

    This is the check that makes the connector-numbering convention verifiable
    from data rather than taken on trust: each convention predicts a different
    pair of silent VMMs, so a mismatch must be reported, not absorbed.
    """
    tab = dc.build_pad_table(disconnected=(2, 8))
    _, pcap = _generate(tab, out_dir, "cabling", args, uniform=True)
    if fm.main([pcap, "--out-dir", out_dir, "--name", "cabling",
                "--min-events", "40"]):
        return False, ["  FAIL analysis returned non-zero"]
    with open(os.path.join(out_dir, "cabling_summary.json")) as f:
        s = json.load(f)
    cab = s["cabling_check"]
    silent = set(cab["silent_but_expected"])
    loud = set(int(k) for k in cab["firing_but_declared_disconnected"])
    # Derive the expectation from the cabling module rather than hardcoding VMM
    # ids: this assertion was written against VMM_BASE = 0 and silently went
    # stale the moment the real data corrected the base to 2. The test must
    # follow the module, or it tests last week's cabling.
    want_silent = set(sum((list(dc.hybrid_vmms(n)) for n in (2, 8)), []))
    want_loud = set(sum((list(dc.hybrid_vmms(n)) for n in (1, 9)), []))
    ok = (not cab["consistent"] and want_silent <= silent
          and want_loud <= loud)
    return ok, [f"  {'ok  ' if ok else 'FAIL'} wrong disconnected list "
                f"detected: silent {sorted(silent)}, firing {sorted(loud)}"]


def scenario_merge(args, out_dir):
    """Four overlapping source positions, four DIFFERENT run scales, one common
    gain field. vmm_fe55_merge.py must recover the gain field over the union of
    the positions and absorb the run-to-run differences into the scales -- not
    the other way round, which would print the changing conditions onto the
    detector as a gain step at every seam."""
    import vmm_fe55_merge as mg

    tab = dc.build_pad_table()
    ref_x = float(tab.loc[tab["mapped"], "pad_cx"].mean())
    spots = [(300.0, 150.0), (350.0, 195.0), (400.0, 240.0), (450.0, 285.0)]
    peaks = [520.0, 585.0, 470.0, 610.0]

    csvs, truth_gain = [], None
    for i, ((x0, y0), pk) in enumerate(zip(spots, peaks)):
        tr = Truth(tab, x0=x0, y0=y0, sigma_mm=26.0,
                   n_photons=max(args.photons // 2, 60_000), peak_adc=pk,
                   seed=args.seed + 100 + i, gain_ref_x=ref_x)
        vmm, ch, adc = tr.sample()
        pcap = os.path.join(out_dir, f"pos{i}.pcapng")
        write_pcapng(pcap, vmm, ch, adc, duration_s=args.duration)
        if fm.main([pcap, "--out-dir", out_dir, "--name", f"pos{i}",
                    "--min-events", "40", "--fit-min", "120"]):
            return False, [f"  FAIL analysis of position {i} returned non-zero"]
        csvs.append(os.path.join(out_dir, f"pos{i}_pads.csv"))
        if truth_gain is None:
            truth_gain = dict(zip(tr.pads["channel_id"], tr.rel_gain))

    if mg.main(csvs + ["--out-dir", out_dir, "--name", "merged"]):
        return False, ["  FAIL merge returned non-zero"]

    import pandas as pd
    m = pd.read_csv(os.path.join(out_dir, "merged_merged_pads.csv"))
    want = np.array([truth_gain[i] for i in m["channel_id"]])
    want = want / np.median(want)
    got = m["rel_gain"].to_numpy()
    resid = float(np.std(got - want))
    r = float(np.corrcoef(got, want)[0, 1])

    with open(os.path.join(out_dir, "merged_summary.json")) as f:
        fitted = json.load(f)["runs"]
    ts = np.array(peaks) / np.exp(np.mean(np.log(peaks)))
    gs = np.array([fitted[f"pos{i}"] for i in range(len(peaks))])
    gs = gs / np.exp(np.mean(np.log(gs)))
    scale_err = float(np.max(np.abs(gs / ts - 1.0)))

    # The union must be meaningfully bigger than any one position, or the test
    # is not exercising the stitch at all.
    single = max(len(pd.read_csv(c)[lambda d: d["peak_adc"].notna()])
                 for c in csvs)
    ok = (resid < 0.05 and r > 0.85 and scale_err < 0.05
          and len(m) > 1.8 * single)
    return ok, [
        f"  {'ok  ' if ok else 'FAIL'} merged {len(csvs)} positions -> "
        f"{len(m)} pads (best single position: {single})",
        f"       rms(merged - truth) = {resid:.4f} (tol 0.05), "
        f"correlation r = {r:.4f} (tol 0.85)",
        f"       max run-scale error = {100 * scale_err:.2f} % (tol 5 %)",
    ]


SCENARIOS = {
    "nominal": scenario_nominal,
    "clipped": scenario_clipped,
    "wrongmap": scenario_wrong_mapping,
    "cabling": scenario_cabling,
    "merge": scenario_merge,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None,
                    help="where to put the pcapng and the products "
                         "(default: a temporary directory)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temporary directory")
    ap.add_argument("--scenario", default="all",
                    choices=["all"] + list(SCENARIOS),
                    help="which scenario(s) to run")
    ap.add_argument("--generate-only", metavar="PCAPNG", default=None,
                    help="only write a synthetic capture, do not analyse")
    ap.add_argument("--photons", type=int, default=250_000)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--sigma", type=float, default=18.0)
    ap.add_argument("--x0", type=float, default=None)
    ap.add_argument("--y0", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    if args.generate_only:
        tab = dc.build_pad_table()
        truth = Truth(tab, x0=args.x0, y0=args.y0, sigma_mm=args.sigma,
                      n_photons=args.photons, seed=args.seed)
        vmm, ch, adc = truth.sample()
        write_pcapng(args.generate_only, vmm, ch, adc,
                     duration_s=args.duration)
        print(f"wrote {args.generate_only}: {len(vmm):,} hits, "
              f"truth {json.dumps(truth.as_dict())}")
        return 0

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="fe55_selftest_")
    os.makedirs(out_dir, exist_ok=True)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    results = {}
    for name in names:
        print(f"\n{'#' * 70}\n# scenario: {name}\n{'#' * 70}")
        sub = os.path.join(out_dir, name)
        os.makedirs(sub, exist_ok=True)
        try:
            results[name] = SCENARIOS[name](args, sub)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = (False, [f"  FAIL raised {type(e).__name__}: {e}"])

    print(f"\n{'=' * 70}\nSELF-TEST SUMMARY\n{'=' * 70}")
    all_ok = True
    for name, (ok, lines) in results.items():
        all_ok &= ok
        print(f"\n{name}: {'PASS' if ok else 'FAIL'}")
        print("\n".join(lines))
    print("\n" + ("ALL SCENARIOS PASSED" if all_ok else "SELF-TEST FAILED"))
    if args.keep or args.out_dir:
        print(f"products kept in {out_dir}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
