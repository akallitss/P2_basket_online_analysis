#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_det1_cabling.py — (vmm, ch) -> pad for ONE P2 basket detector read out by
nine VMM hybrids, one per connector.

Why this exists next to vmm_stations.py
---------------------------------------
`vmm_stations.build_pad_table()` is hardwired to the SPS telescope: three P2
stations, connectors c4-c6 each, VMMs 2-19, hybrid 0 reserved for the external
trigger. That geometry is what the live beam QA runs on and must not be
disturbed. The Fe55 bench setup is a different animal:

  * ONE detector (det1), so there is no station axis at all;
  * NINE hybrids on connectors 1-9, i.e. most of the fan is instrumented
    instead of three connectors of it;
  * hybrid n carries VMMs 2n and 2n+1 (the telescope's convention, kept), but
    hybrid 0 is not populated, so VMM 0/1 -- the telescope's trigger digitiser
    -- are simply absent from the data;
  * no external trigger: the VMMs self-trigger on the source.

Rather than bolt a second mode onto vmm_stations, the cabling lives here and
the two files stay independent. Everything downstream of the pad map --
`P2_BASKET_mapping.csv`, the strip/sector arithmetic, the within-half
strategies -- is the same detector and is deliberately kept bit-identical to
vmm_stations / cosmic_bench_analysis.p2_mapping so a pad number means the same
thing in all three.

Chain
-----
    (vmm, ch)
      -> hybrid, connector_N (1..9), half           HYBRID_CONNECTOR + halves
      -> within = ch (0..63)
      -> strip  = half_base + f(within)             WITHIN_STRATEGY
      -> sector = connector_N - 1
      -> channel_id = sector*128 + (strip - 1)
      -> pad_cx, pad_cy, ...                        P2_BASKET_mapping.csv

CABLING AS OF 2026-09-09 (Alexandra, Fe55 bench, det1, drift 750 V / mesh 420 V)
  hybrid 1 -> connector 1 -> VMM 2 (bot) / 3 (top)
  hybrid 2 -> connector 2 -> VMM 4 (bot) / 5 (top)
  ...
  hybrid 9 -> connector 9 -> VMM 18 (bot) / 19 (top)
  connectors 1 and 9 DISCONNECTED  -> sectors 0 and 8 read out by nobody
  connector 10 (sector 9) is not cabled in this setup at all

So the live pads are sectors 1-7 (connectors 2-8), VMMs 4-17: 7*128 = 896 of
the detector's 1280 pads.

@author: ak271430 Alexandra Kallitsopoulou
"""

import os
import struct

import numpy as np
import pandas as pd

__all__ = ["build_pad_table", "full_pad_map", "attach_pads", "pad_polygons",
           "vmm_half_map", "expected_vmms", "verify_cabling", "live_time",
           "MAP_CSV"]

# --------------------------------------------------------------------------
# Detector constants — identical to vmm_stations / p2_mapping
# --------------------------------------------------------------------------
STRIPS_PER_CONN = 128       # one physical connector = 128 strips = 2 halves
CH_PER_HALF = 64            # one VMM = 64 channels = one half
N_CONNECTORS = 10           # the PCB has 10 connectors / 1280 pads
N_PADS = N_CONNECTORS * STRIPS_PER_CONN

MAP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "P2_BASKET_mapping.csv")

# --------------------------------------------------------------------------
# Cabling
# --------------------------------------------------------------------------
# Hybrids present, in connector order. Hybrid n sits on connector n.
HYBRIDS = tuple(range(1, 10))

# Global VMM id of the BOTTOM chip of hybrid n.
#
# MEASURED 2026-09-09 on the first real Fe55 capture
# (source_tests/enp4s0f1_20260909-150423_0.pcapng, 611,119 hits). Per-VMM
# occupancy:
#     VMM 0,1     0 hits            no hybrid 0 (no trigger hybrid, as expected
#                                   for a self-triggered source run)
#     VMM 2,3     32, 18  (0.01 %)  connector 1 -- DISCONNECTED
#     VMM 4-17    34k-55k each      connectors 2-8, the seven live hybrids
#     VMM 18,19   27, 154 (0.03 %)  connector 9 -- DISCONNECTED
#     VMM 20-31   0 hits
#
# So hybrid n carries VMM 2n / 2n+1: hybrid 1 is VMM 2/3, hybrid 9 is VMM
# 18/19. This is the SAME convention as run_config_beam.P2_VMM_CABLING, kept
# even though hybrid 0 is not populated here.
#
# The initial guess was base = 0 ("vmm0 to the bottom connector"), which turns
# out to describe the LOCAL chip numbering on the hybrid, not the global VMM
# id. That guess shifted every connector by one sector -- the exact failure
# this check exists to catch, and the two disconnected connectors are what
# caught it: base = 0 predicts VMM 0,1 and 16,17 silent, base = 2 predicts
# VMM 2,3 and 18,19 silent, and the data shows the latter.
VMM_BASE = 2


def hybrid_vmms(hybrid, base=VMM_BASE):
    """(bottom_vmm, top_vmm) global ids for a hybrid."""
    return base + 2 * (hybrid - 1), base + 2 * (hybrid - 1) + 1


# Connectors with no hybrid plugged in. Their channels are omitted from the
# table entirely, so their hits (there should be none) never reach a pad and
# their pads never enter a gain/rate map as "dead pads".
DISCONNECTED = (1, 9)

# Which detector half the hybrid's BOTTOM VMM reads. 'bot' = strips 1-64,
# which the map CSV labels mec8_connector 0. Swappable, like in vmm_stations:
# it is a property of how the hybrid seats on the MEC8, not of the PCB.
BOTTOM_VMM_HALF = "bot"

# Order of `within` (0..63) inside a 64-strip half. 'reverse' is validated for
# the DREAM/K59V readout (p2_mapping.py, M3 track correlation r=0.93/0.88 on
# run p2_det1_long_run_6-30-26) and is what the VMM telescope scan also settled
# on for every half but two. It is NOT independently validated for this bench
# cabling -- run `vmm_fe55_map.py <file> --scan-mapping` on the first Fe55 file
# to confirm, exactly as vmm_map_rough.py does for the beam.
WITHIN_STRATEGY = "reverse"

# Per-(connector_N, half) exceptions, for a ribbon cabled the other way round.
# The telescope needed two of these on P2_IN (flipped c5/c6 top ribbons). Empty
# until --scan-mapping says otherwise on THIS cabling.
STRATEGY_OVERRIDES = {}

# Fe55 is a 5.9 keV photon: the conversion is a point on the scale of a 12 mm
# pad, so charge is contained in one pad and the single-pad ADC spectrum IS the
# Fe55 spectrum. Recorded here because the number drives the gain estimator.
PAD_PITCH_MM = 12.0


def _within_to_strip(within, half, strategy=WITHIN_STRATEGY):
    """within (0..63) + half -> strip (1..128). Same forms as p2_mapping.py."""
    base = 65 if half == "top" else 1
    if strategy == "linear":
        off = within
    elif strategy == "reverse":
        off = 63 - within
    elif strategy == "pairswap":
        off = within ^ 1
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return base + off


def vmm_half_map(disconnected=DISCONNECTED, bottom_half=BOTTOM_VMM_HALF,
                 base=VMM_BASE, hybrids=HYBRIDS):
    """{vmm_id: (connector_N, half)} for every instrumented VMM."""
    top_half = "top" if bottom_half == "bot" else "bot"
    drop = {int(c) for c in (disconnected or ())}
    out = {}
    for n in hybrids:
        if n in drop:
            continue
        bot_vmm, top_vmm = hybrid_vmms(n, base)
        out[int(bot_vmm)] = (int(n), bottom_half)
        out[int(top_vmm)] = (int(n), top_half)
    return out


def expected_vmms(**kw):
    """Sorted list of VMM ids that should appear in the data."""
    return sorted(vmm_half_map(**kw))


# --------------------------------------------------------------------------
# Pad map
# --------------------------------------------------------------------------
_GEOM_COLS = ("pad_cx", "pad_cy", "radius", "phi", "delta_phi",
              "pad_w", "pad_h", "pad_angle", "pad_area", "mec8_connector")


def load_pad_map(map_csv=None):
    """The Gerber pad map indexed by channel_id (0..1279)."""
    m = pd.read_csv(map_csv or MAP_CSV)
    if "channel_id" not in m.columns:
        raise ValueError(f"{map_csv or MAP_CSV} has no channel_id column")
    m = m.set_index("channel_id").sort_index()
    if len(m) != N_PADS:
        print(f"[det1_cabling] WARNING: map has {len(m)} rows, expected {N_PADS}")
    return m


def build_pad_table(map_csv=None, strategy=WITHIN_STRATEGY,
                    bottom_half=BOTTOM_VMM_HALF, disconnected=DISCONNECTED,
                    overrides=None, base=VMM_BASE, hybrids=HYBRIDS):
    """One row per instrumented (vmm, ch), joined to pad geometry.

    Columns: vmm, ch, hybrid, connector_N, half, sector, strip, channel_id,
             pad_cx, pad_cy, radius, phi, pad_w, pad_h, pad_angle, pad_area,
             mec8_connector, mapped.
    """
    pad = load_pad_map(map_csv)
    halves = vmm_half_map(disconnected, bottom_half, base, hybrids)
    over = dict(STRATEGY_OVERRIDES if overrides is None else overrides)

    rows = []
    for vmm, (n, half) in sorted(halves.items()):
        sector = n - 1
        strat = over.get((n, half), strategy)
        for within in range(CH_PER_HALF):
            strip = int(_within_to_strip(within, half, strat))
            rows.append((vmm, within, n, n, half, sector, strip,
                         sector * STRIPS_PER_CONN + (strip - 1)))

    tab = pd.DataFrame(rows, columns=["vmm", "ch", "hybrid", "connector_N",
                                      "half", "sector", "strip", "channel_id"])
    have = [c for c in _GEOM_COLS if c in pad.columns]
    tab = tab.merge(pad[have], left_on="channel_id", right_index=True, how="left")
    tab["mapped"] = tab["pad_cx"].notna()

    # A scrambled strategy would collide two channels onto one pad; a silent
    # duplicate would double-count that pad's rate. Cheap, so always checked.
    dup = tab.loc[tab["mapped"], "channel_id"].duplicated().sum()
    if dup:
        raise ValueError(f"cabling produces {dup} duplicated channel_id -- "
                         f"strategy={strategy!r} overrides={over!r}")

    tab.attrs.update(strategy=strategy, bottom_half=bottom_half,
                     overrides=dict(over), base=base,
                     disconnected=sorted(int(c) for c in (disconnected or ())),
                     hybrids=list(hybrids))
    return tab


def full_pad_map(map_csv=None, **kw):
    """All 1280 pads with an `instrumented` flag, for drawing the whole fan.

    The uninstrumented parts of the detector are real estate that exists and is
    simply not read out. Drawing them grey rather than omitting them is what
    makes a rate map honest about how much of the chamber the source covers.
    """
    pad = load_pad_map(map_csv).reset_index()
    tab = build_pad_table(map_csv=map_csv, **kw)
    live = tab.loc[tab["mapped"], ["channel_id", "vmm", "ch", "connector_N",
                                   "half", "hybrid"]]
    out = pad.merge(live, on="channel_id", how="left", suffixes=("", "_tab"))
    out["instrumented"] = out["vmm"].notna()
    return out


def attach_pads(hits, tab):
    """Left-join hits to the pad table on (vmm, ch).

    Only for small hit samples: the join materialises every pad column on every
    hit (~1.5 GB on a 5 M hit capture, as vmm_stations.auto_hot_pads documents).
    The per-pad products in vmm_fe55_map.py histogram on (vmm, ch) instead and
    attach geometry per PAD, which is 64x32 rows rather than millions.
    """
    return hits.merge(tab, on=["vmm", "ch"], how="left", copy=False)


def pad_key(vmm, ch):
    """Flat key vmm*64 + ch, the index used by every per-pad histogram here."""
    return np.asarray(vmm, dtype=np.int64) * 64 + np.asarray(ch, dtype=np.int64)


def counts_per_pad(vmm, ch, tab, weights=None):
    """Total (or weight-summed) hits per row of `tab`, aligned to tab's order.

    One bincount over the 32*64 key space; never joins geometry onto hits.
    """
    n_key = 32 * 64
    k = pad_key(vmm, ch)
    ok = (k >= 0) & (k < n_key)
    hist = np.bincount(k[ok], weights=None if weights is None
                       else np.asarray(weights, dtype=np.float64)[ok],
                       minlength=n_key)
    return hist[pad_key(tab["vmm"].to_numpy(), tab["ch"].to_numpy())]


def pad_polygons(tab):
    """(N,4,2) pad corners from centre + size + rotation.

    The P2 pads are fan tiles, each with its own angle; drawing them as rotated
    rectangles reproduces the fan instead of a square grid. Same construction as
    vmm_beam_profile.pad_polygons -- kept here so this module stands alone.
    """
    cx = tab["pad_cx"].to_numpy(float)
    cy = tab["pad_cy"].to_numpy(float)
    w = np.nan_to_num(tab["pad_w"].to_numpy(float), nan=PAD_PITCH_MM) \
        if "pad_w" in tab else np.full(len(tab), PAD_PITCH_MM)
    h = np.nan_to_num(tab["pad_h"].to_numpy(float), nan=PAD_PITCH_MM) \
        if "pad_h" in tab else np.full(len(tab), PAD_PITCH_MM)
    ang = np.nan_to_num(np.deg2rad(tab["pad_angle"].to_numpy(float)), nan=0.0) \
        if "pad_angle" in tab else np.zeros(len(tab))

    dx = np.array([-0.5, 0.5, 0.5, -0.5])
    dy = np.array([-0.5, -0.5, 0.5, 0.5])
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    ox = dx[None, :] * w[:, None]
    oy = dy[None, :] * h[:, None]
    return np.stack([cx[:, None] + ox * ca - oy * sa,
                     cy[:, None] + ox * sa + oy * ca], axis=-1)


# --------------------------------------------------------------------------
# Does the data agree with the cabling above?
# --------------------------------------------------------------------------
def verify_cabling(vmm_ids, tab=None, disconnected=DISCONNECTED, base=VMM_BASE,
                   hybrids=HYBRIDS, quiet_frac=0.002):
    """Compare the VMM ids present in the data against the declared cabling.

    `vmm_ids` is a {vmm: n_hits} mapping (or any Series/dict-like).

    The two disconnected connectors are a free calibration of the whole
    numbering convention: each candidate convention predicts a DIFFERENT pair of
    silent VMM ids, so one file settles it. Anything that fires on a connector
    declared disconnected, or is silent on one declared live, is reported.

    Returns a dict; nothing is raised, because at the bench a surprising answer
    is information, not a reason to stop.
    """
    counts = {int(k): int(v) for k, v in dict(vmm_ids).items()}
    total = sum(counts.values()) or 1
    live = vmm_half_map(disconnected, BOTTOM_VMM_HALF, base, hybrids)
    dead = {}
    for n in hybrids:
        if n in set(disconnected or ()):
            for v in hybrid_vmms(n, base):
                dead[int(v)] = int(n)

    silent = [v for v in sorted(live) if counts.get(v, 0) / total < quiet_frac]
    loud_dead = {v: counts[v] for v in sorted(dead)
                 if counts.get(v, 0) / total >= quiet_frac}
    unknown = {v: c for v, c in sorted(counts.items())
               if v not in live and v not in dead and c / total >= quiet_frac}

    return {
        "expected_vmms": sorted(live),
        "declared_disconnected_connectors": sorted(disconnected or ()),
        "silent_but_expected": silent,
        "firing_but_declared_disconnected": loud_dead,
        "unexpected_vmm_ids": unknown,
        "n_hits_total": int(total),
        "consistent": not (silent or loud_dead or unknown),
    }


def format_cabling_report(rep):
    """One-screen summary of verify_cabling(), for the console and the JSON."""
    lines = []
    ok = "OK" if rep["consistent"] else "MISMATCH"
    lines.append(f"cabling check: {ok}   ({rep['n_hits_total']:,} hits)")
    lines.append(f"  expected VMMs: {rep['expected_vmms']}")
    if rep["silent_but_expected"]:
        lines.append(f"  SILENT but expected live : {rep['silent_but_expected']}"
                     "  <- dead hybrid, unpowered, or wrong VMM_BASE")
    if rep["firing_but_declared_disconnected"]:
        lines.append("  FIRING but declared disconnected : "
                     f"{rep['firing_but_declared_disconnected']}"
                     "  <- DISCONNECTED list or numbering convention is wrong")
    if rep["unexpected_vmm_ids"]:
        lines.append(f"  unexpected VMM ids : {rep['unexpected_vmm_ids']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Live time
# --------------------------------------------------------------------------
# Every rate in Hz comes from the pcapng's own capture clock: the Enhanced
# Packet Block timestamps, which dumpcap writes from the system clock and whose
# resolution is declared in the Interface Description Block. Getting
# `if_tsresol` wrong is a silent factor-1000 on every rate, so it is parsed
# rather than assumed (the DAQ's dumpcap writes nanoseconds, not the pcapng
# default of microseconds).
#
# The SRS marker span is kept only as a CROSS-CHECK, never as the live time.
# Measured 2026-09-09 on data/hybrid_performance_data, the two agree to ~10 %
# (packet clock 19.9 s vs SRS 22.0 s at the nominal 25 ns tick on the same
# file), which is close enough to catch a gross error and far too loose to
# normalise a rate map with. It also cannot be trusted in principle:
# vmm_stations documents that `offset` is stuck in this firmware, so there is
# no BCID rollover counter and chip time is only unambiguous within 92.16 us
# of a marker. The wall clock has no such caveat.
_BT_SHB = 0x0A0D0D0A
_BT_IDB = 0x00000001
_BT_EPB = 0x00000006
_BT_SPB = 0x00000003
_BOM = 0x1A2B3C4D
_DEFAULT_TSRESOL = 1e-6      # pcapng default when if_tsresol is absent


def _tsresol_from_idb(body, endian):
    """Seconds per timestamp tick from an IDB's if_tsresol option (code 9)."""
    # body: linktype u16, reserved u16, snaplen u32, then (code u16, len u16,
    # value, padded to 4) options terminated by code 0.
    pos = 8
    while pos + 4 <= len(body):
        code, olen = struct.unpack_from(endian + "HH", body, pos)
        pos += 4
        if code == 0:
            break
        if code == 9 and olen >= 1:
            v = body[pos]
            return 2.0 ** -(v & 0x7F) if v & 0x80 else 10.0 ** -v
        pos += (olen + 3) & ~3
    return _DEFAULT_TSRESOL


def capture_window(path):
    """(t_first, t_last, n_packets, tsresol) from a pcapng's packet clock.

    Unix epoch seconds. Walks the block chain reading only headers, so the cost
    is one iteration per packet (~1e3 for a 45 s capture) and the packet data is
    never touched. Simple Packet Blocks carry no timestamp and are counted only.
    """
    size = os.path.getsize(path)
    if size < 12:
        raise ValueError(f"{path}: too small to be a pcapng")
    u8 = np.memmap(path, dtype=np.uint8, mode="r")
    head = bytes(u8[:12])
    if struct.unpack_from("<I", head, 0)[0] != _BT_SHB:
        raise ValueError(f"{path}: not a pcapng")
    endian = "<" if struct.unpack_from("<I", head, 8)[0] == _BOM else ">"

    tsresol = _DEFAULT_TSRESOL
    t_first = t_last = None
    n_pkt = 0
    pos = 0
    while pos + 12 <= size:
        btype, blen = struct.unpack_from(endian + "II", u8, pos)
        if blen < 12 or blen % 4 or pos + blen > size:
            break                       # truncated tail: dumpcap mid-write
        if btype == _BT_EPB and blen >= 32:
            hi, lo = struct.unpack_from(endian + "II", u8, pos + 12)
            t = ((int(hi) << 32) | int(lo)) * tsresol
            if t_first is None:
                t_first = t
            t_last = t
            n_pkt += 1
        elif btype == _BT_IDB and blen >= 20:
            tsresol = _tsresol_from_idb(bytes(u8[pos + 8:pos + blen - 4]), endian)
        elif btype == _BT_SPB:
            n_pkt += 1
        pos += blen
    del u8
    return t_first, t_last, n_pkt, tsresol


def live_time(paths, capture_duration_s=None, srs_span_s=None):
    """Live time in seconds over one or more captures, plus how it was got.

    The number reported is the UNION of the files' [t_first, t_last] intervals.
    Union, not a sum of spans: captures handed in twice, or overlapping ring
    segments, must not each be credited with their own live time. Duplicated
    files are common enough at a bench (a copied directory, a re-listed glob)
    that silently doubling the denominator would halve every rate.

    A first-to-last-packet span always UNDERCOUNTS by the dead interval before
    the first packet and after the last -- at a few Hz of Fe55 with packets
    arriving sparsely, that is not negligible. So when the DAQ's configured
    capture length is known (run_config_beam.CAPTURE_DURATION_S), the nominal
    n_files * capture_duration_s is reported alongside and preferred when it is
    larger, with the ratio kept so the caller can see how much was inferred.

    `srs_span_s` is an optional cross-check from the SRS marker span; it is
    never used for the value, only compared (see the note above).

    Returns dict(live_s, span_s, nominal_s, t_start, t_end, n_files,
                 n_packets, method, warnings, per_file).
    """
    paths = [str(p) for p in ([paths] if isinstance(paths, (str, bytes))
                              else list(paths))]
    per, warnings = [], []
    for p in paths:
        try:
            t0, t1, n, res = capture_window(p)
        except Exception as e:                      # a half-written file
            per.append({"path": p, "error": f"{type(e).__name__}: {e}"})
            continue
        if t0 is None:
            per.append({"path": p, "error": "no packets"})
            continue
        per.append({"path": p, "t_first": t0, "t_last": t1,
                    "span_s": t1 - t0, "n_packets": n, "tsresol": res})

    good = [d for d in per if "t_first" in d]
    if not good:
        return {"live_s": 0.0, "span_s": 0.0, "nominal_s": None,
                "t_start": None, "t_end": None, "n_files": 0, "n_packets": 0,
                "method": "no usable capture",
                "warnings": ["no capture yielded a packet timestamp"],
                "per_file": per}

    good.sort(key=lambda d: d["t_first"])

    # Union of the covered intervals.
    merged = []
    for d in good:
        if merged and d["t_first"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], d["t_last"])
        else:
            merged.append([d["t_first"], d["t_last"]])
    span = float(sum(b - a for a, b in merged))

    n_distinct = len(merged)
    if n_distinct < len(good):
        warnings.append(
            f"{len(good) - n_distinct} of {len(good)} captures overlap in time "
            "(duplicated or ring-overlapping files); counted once")

    live, method = span, f"packet clock, union of {n_distinct} interval(s)"
    nominal = None
    if capture_duration_s:
        nominal = float(capture_duration_s) * n_distinct
        if nominal > span:
            live, method = nominal, (
                f"configured capture duration {capture_duration_s:.1f} s x "
                f"{n_distinct} (packet-clock union was {span:.2f} s)")

    if srs_span_s and span > 0:
        ratio = srs_span_s / span
        if not 0.5 < ratio < 2.0:
            warnings.append(
                f"SRS marker span {srs_span_s:.1f} s disagrees with the packet "
                f"clock {span:.1f} s by {ratio:.1f}x -- rates use the packet "
                "clock; check the capture is not truncated")

    return {"live_s": float(live), "span_s": span, "nominal_s": nominal,
            "t_start": float(merged[0][0]), "t_end": float(merged[-1][1]),
            "n_files": n_distinct, "n_packets": sum(d["n_packets"] for d in good),
            "method": method, "warnings": warnings, "per_file": per}


if __name__ == "__main__":
    tab = build_pad_table()
    print(f"instrumented (vmm,ch): {len(tab)}   mapped pads: {tab.mapped.sum()}")
    print(f"VMMs: {sorted(tab.vmm.unique())}")
    print(f"connectors: {sorted(tab.connector_N.unique())}  "
          f"sectors: {sorted(tab.sector.unique())}")
    full = full_pad_map()
    print(f"detector pads: {len(full)}  instrumented: {int(full.instrumented.sum())}")
    live = tab[tab.mapped]
    print(f"active area: x {live.pad_cx.min():.0f}..{live.pad_cx.max():.0f} mm, "
          f"y {live.pad_cy.min():.0f}..{live.pad_cy.max():.0f} mm, "
          f"pad area {live.pad_area.min():.1f}..{live.pad_area.max():.1f} mm^2")
