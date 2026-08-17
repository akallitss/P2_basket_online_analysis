#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmm_stations.py — telescope geometry, trigger, channel masking, and a pcapng
parser shared by the VMM online QA and the efficiency analysis.

Why this file exists
--------------------
`vmm_pcapng_qa.py` is a top-level script (argparse runs at import), so nothing
can import its parser. Rather than have the efficiency analysis re-implement the
bit unpacking a second time, the parse lives here and both use it.

TODO (tracked in VMM_ONLINE_EFFICIENCY_PLAN.md): make vmm_pcapng_qa.py import
parse_pcapng() from here instead of keeping its own copy. Deliberately NOT done
while beam data is being taken — that script is what the live qa_watcher runs.

Station cabling mirrors run_config_beam.P2_VMM_CABLING. It is duplicated here
(not imported) so this module works standalone, e.g. on lxplus/EOS where the DAQ
config is not checked out. If the cabling ever changes, change both.

@author: ak271430 Alexandra Kallitsopoulou
"""

import os
import struct
import array

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Telescope
# --------------------------------------------------------------------------
# connector -> (hybrid, bottom VMM, top VMM); hybrid H<n> carries VMMs 2n/2n+1.
STATIONS = {
    'P2_IN':  {'z_mm': 320.0,
               'connectors': {'c4': (1, 2, 3), 'c5': (2, 4, 5), 'c6': (3, 6, 7)}},
    'P2_MID': {'z_mm': 630.0,
               'connectors': {'c4': (4, 8, 9), 'c5': (5, 10, 11), 'c6': (6, 12, 13)}},
    'P2_OUT': {'z_mm': 940.0,
               'connectors': {'c4': (7, 14, 15), 'c5': (8, 16, 17), 'c6': (9, 18, 19)}},
}

STATION_VMMS = {
    name: sorted(v for _, b, t in cfg['connectors'].values() for v in (b, t))
    for name, cfg in STATIONS.items()
}
VMM_TO_STATION = {v: name for name, vs in STATION_VMMS.items() for v in vs}
DETECTOR_VMMS = sorted(VMM_TO_STATION)

# --------------------------------------------------------------------------
# Trigger
# --------------------------------------------------------------------------
# MEASURED 2026-07-30 from run_24/nominal_05: VMM 0 channel 44 carries 100.0%
# of that VMM's hits (94,223/94,223 in a 200-packet sample) and VMM 1 never
# appears in the data at all.
#
# run_config_beam.TRIGGER_VMM records {'vmm': 1, 'channel': 60} from colleagues'
# July notes, which predates the 2026-07-29 cabling. The data disagrees; the
# data wins. Every trigger-referenced efficiency number depends on this, so if
# the hybrid-0 cabling is ever changed, re-run:
#     python3 vmm_efficiency.py <file.pcapng> --find-trigger
TRIGGER = {'vmm': 0, 'channel': 44}

# --------------------------------------------------------------------------
# Channel masking
# --------------------------------------------------------------------------
# Known-hot channels, confirmed 2026-07-30 and known to the group beforehand.
# They are recorded in the data on purpose and masked here, in analysis, rather
# than being suppressed at the DAQ.
#
#   VMM 7 ch 62  — P2_IN c6 (top), 63.3% of that VMM's hits
#   VMM 4 ch 58  — P2_IN c5 (bottom), 27.4% (ch 59/61 also elevated)
#
# Every healthy VMM's busiest channel sits at 3-6%. Left unmasked these drag
# P2_IN cluster centroids, which is what poisoned the cosmic-bench det3
# analysis (pad 510 took 55% of signal hits).
NOISY_CHANNELS = {
    7: [62],
    4: [58],
}

# A channel is auto-flagged when it holds more than this many times its own
# VMM's median channel occupancy. 8x is comfortably above the ~2-3x structure
# of a real beam spot and catches ch 61/59 on VMM 4.
HOT_RATIO = 8.0

CLOCK_PERIOD_NS = 22.5   # ns per BCID count (44.44 MHz)
TAC_SLOPE_NS = 60.0      # ns full scale of the TDC TAC ramp
TDC_RANGE = 255


def auto_hot_channels(hits, ratio=HOT_RATIO, min_hits=200):
    """Return {vmm: [ch, ...]} for channels far above their VMM's median.

    Purely data-driven, so it also catches channels that go hot mid-campaign.
    VMMs with fewer than min_hits are skipped (median is meaningless there).
    """
    found = {}
    for vmm, grp in hits.groupby('vmm', observed=True):
        if len(grp) < min_hits:
            continue
        counts = np.bincount(grp['ch'].to_numpy(), minlength=64).astype(float)
        med = np.median(counts[counts > 0]) if np.any(counts > 0) else 0.0
        if med <= 0:
            continue
        hot = np.flatnonzero(counts > ratio * med)
        if hot.size:
            found[int(vmm)] = [int(c) for c in hot]
    return found


def merge_masks(*masks):
    """Union of several {vmm: [ch,...]} dicts."""
    out = {}
    for m in masks:
        for vmm, chans in (m or {}).items():
            out.setdefault(int(vmm), set()).update(int(c) for c in chans)
    return {v: sorted(cs) for v, cs in sorted(out.items())}


def apply_channel_mask(hits, mask):
    """Drop masked (vmm, ch) pairs. Trigger hits are never masked."""
    if not mask:
        return hits, 0
    t_vmm, t_ch = TRIGGER['vmm'], TRIGGER['channel']
    drop = np.zeros(len(hits), dtype=bool)
    vmm = hits['vmm'].to_numpy()
    ch = hits['ch'].to_numpy()
    for v, chans in mask.items():
        if v == t_vmm:
            chans = [c for c in chans if c != t_ch]
        if chans:
            drop |= (vmm == v) & np.isin(ch, chans)
    return hits.loc[~drop], int(drop.sum())


def trigger_times(hits, trigger=None):
    """Sorted, de-duplicated trigger timestamps in ns."""
    trg = trigger or TRIGGER
    sel = (hits['vmm'] == trg['vmm']) & (hits['ch'] == trg['channel'])
    t = np.unique(hits.loc[sel, 'abs_time_ns'].to_numpy())
    return np.sort(t)


def find_trigger_channel(hits, min_share=0.80):
    """Identify the trigger channel empirically: the (vmm, ch) that holds
    >= min_share of its VMM's hits. Returns (vmm, ch, share) or None.

    Use this after any hybrid-0 recabling rather than trusting TRIGGER.
    """
    best = None
    for vmm, grp in hits.groupby('vmm', observed=True):
        counts = np.bincount(grp['ch'].to_numpy(), minlength=64)
        ch = int(np.argmax(counts))
        share = counts[ch] / max(len(grp), 1)
        if share >= min_share and (best is None or share > best[2]):
            best = (int(vmm), ch, float(share))
    return best


# --------------------------------------------------------------------------
# pcapng parsing  (lifted from vmm_pcapng_qa.py — keep the two in step)
# --------------------------------------------------------------------------
_DEFAULT_MARKER = [0, 0, 0]


def gray2bin_np(arr):
    arr = arr.astype(np.uint32)
    arr ^= arr >> 16
    arr ^= arr >> 8
    arr ^= arr >> 4
    arr ^= arr >> 2
    arr ^= arr >> 1
    return arr.astype(np.uint16)


def _parse_block(block, frame_counter, fec_id, markers, data_format,
                 vmm_buf, ch_buf, adc_buf, ot_buf, offset_buf, bcid_buf,
                 tdc_buf, srs_ts_buf, bad_vmm):
    if len(block) < 22 or block[4:7] != b'VM3':
        return
    for i in range(0, len(block) - 22, 6):
        d1, d2 = struct.unpack_from('>IH', block, i + 16)
        if d2 & 0x8000:
            vmm_id = (d1 >> 22) & 0x1F
            if vmm_id not in VMM_TO_STATION and vmm_id != TRIGGER['vmm']:
                bad_vmm[vmm_id] = bad_vmm.get(vmm_id, 0) + 1
            m = markers.get((fec_id, vmm_id), _DEFAULT_MARKER)
            vmm_buf.append(vmm_id)
            ch_buf.append((d2 >> 8) & 0x3F)
            adc_buf.append((d1 >> 12) & 0x3FF)
            ot_buf.append((d2 >> 14) & 0x1)
            raw_off = (d1 >> 27) & 0x1F
            offset_buf.append(raw_off if raw_off < 16 else raw_off - 32)
            bcid_buf.append(d1 & 0xFFF)
            tdc_buf.append(d2 & 0xFF)
            srs_ts_buf.append(m[0])
        else:
            vmmid_marker = (d2 >> 10) & 0x1F
            srs_ts = (d1 << 10) | (d2 & 0x3FF)
            # NOTE: vmm-sdat's ParserSRS (and vmm_pcapng_qa.py, which copies it)
            # gates this on `vmmid_marker < 16`, because upstream a FEC hosts at
            # most 16 VMMs and ids >= 16 encode TRG trigger words instead. THIS
            # FEC HOSTS 20 (hybrids 0-9), so that test silently threw away every
            # marker for VMMs 16-19 -- they ended up with srs_timestamp = 0 for
            # every hit, no absolute time, and therefore ZERO trigger
            # coincidences. That is 4 of P2_OUT's 6 VMMs (37% of all detector
            # hits). Measured 2026-07-30 on run_24.
            # In SRS mode there are no TRG markers, so every marker word is a
            # VMM marker and the id must be taken at face value.
            if data_format == 'SRS':
                markers.setdefault((fec_id, vmmid_marker), [0, 0, 0])[0] = srs_ts
            elif vmmid_marker < 16 and srs_ts < 4096:
                markers.setdefault((fec_id, vmmid_marker), [0, 0, 0])[0] = srs_ts


def parse_pcapng(path, max_packets=None, data_format='SRS', src_ips=None,
                 progress=False):
    """Parse a VMM pcapng into a hits DataFrame.

    Returns (hits, meta). hits columns: vmm, ch, adc, over_threshold, offset,
    bcid, tdc, timestamp_ns, srs_timestamp, abs_time_ns.

    meta carries n_packets, n_vm3_packets, bad_vmm_ids (counts of VMM ids that
    are neither a station VMM nor the trigger — corrupt words) and
    marker_gap_ns_p50/p99, the SRS marker cadence per VMM. The cadence matters:
    `offset` is stuck at a constant in this firmware (see the plan doc), so the
    BCID rollover counter is unavailable and chip time is only unambiguous
    within 4096 * 22.5 ns = 92.16 us of a marker.
    """
    from scapy.all import PcapReader, UDP, IP

    if src_ips is None:
        src_ips = _detect_fec_ips(path)
    markers = {}
    bad_vmm = {}
    vmm_buf = array.array('B'); ch_buf = array.array('B')
    adc_buf = array.array('H'); ot_buf = array.array('B')
    offset_buf = array.array('b'); bcid_buf = array.array('H')
    tdc_buf = array.array('B'); srs_ts_buf = array.array('Q')

    n_pkt = n_vm3 = 0
    with PcapReader(path) as reader:
        for pkt in reader:
            if UDP in pkt and IP in pkt and pkt[IP].src in src_ips:
                payload = bytes(pkt[UDP].payload)
                fc = struct.unpack_from('>I', payload)[0]
                fec_id = (struct.unpack_from('>I', payload, 4)[0] >> 4) & 0x0F
                before = len(vmm_buf)
                _parse_block(payload, fc, fec_id, markers, data_format,
                             vmm_buf, ch_buf, adc_buf, ot_buf, offset_buf,
                             bcid_buf, tdc_buf, srs_ts_buf, bad_vmm)
                if len(vmm_buf) > before:
                    n_vm3 += 1
            n_pkt += 1
            if progress and n_pkt % 10000 == 0:
                print(f'  {n_pkt} packets | {len(vmm_buf):,} hits')
            if max_packets and n_pkt >= max_packets:
                break

    offset = np.frombuffer(offset_buf, dtype=np.int8).copy()
    bcid = gray2bin_np(np.frombuffer(bcid_buf, dtype=np.uint16).copy())
    tdc = np.frombuffer(tdc_buf, dtype=np.uint8).copy()
    srs_ts = np.frombuffer(srs_ts_buf, dtype=np.uint64).copy()

    t_coarse = (offset.astype(np.float64) * 4096 + bcid.astype(np.float64)) * CLOCK_PERIOD_NS
    t_fine = CLOCK_PERIOD_NS - tdc.astype(np.float64) * TAC_SLOPE_NS / TDC_RANGE
    timestamp_ns = t_coarse + t_fine

    hits = pd.DataFrame({
        'vmm': np.frombuffer(vmm_buf, dtype=np.uint8).copy(),
        'ch': np.frombuffer(ch_buf, dtype=np.uint8).copy(),
        'adc': np.frombuffer(adc_buf, dtype=np.uint16).copy(),
        'over_threshold': np.frombuffer(ot_buf, dtype=np.uint8).astype(bool),
        'offset': offset,
        'bcid': bcid,
        'tdc': tdc,
        'timestamp_ns': timestamp_ns,
        'srs_timestamp': srs_ts,
        'abs_time_ns': srs_ts.astype(np.float64) * 25.0 + timestamp_ns,
    })

    meta = {
        'n_packets': n_pkt,
        'n_vm3_packets': n_vm3,
        'n_hits': len(hits),
        'fec_ips': sorted(src_ips),
        'bad_vmm_ids': {int(k): int(v) for k, v in sorted(bad_vmm.items())},
        'offset_constant': bool(len(np.unique(offset)) == 1),
    }
    meta.update(_marker_cadence(hits))
    return hits, meta


def _marker_cadence(hits):
    """Median/99th-percentile spacing between distinct SRS marker values, in ns.

    Compared against the 92.16 us BCID rollover: if p99 exceeds it, chip time
    can alias and abs_time_ns is not safe for coincidences.
    """
    gaps = []
    for _, grp in hits.groupby('vmm', observed=True):
        ts = np.unique(grp['srs_timestamp'].to_numpy())
        if ts.size > 2:
            gaps.append(np.diff(ts.astype(np.float64)) * 25.0)
    if not gaps:
        return {'marker_gap_ns_p50': None, 'marker_gap_ns_p99': None}
    allg = np.concatenate(gaps)
    return {'marker_gap_ns_p50': float(np.percentile(allg, 50)),
            'marker_gap_ns_p99': float(np.percentile(allg, 99)),
            'bcid_rollover_ns': 4096 * CLOCK_PERIOD_NS}


def _detect_fec_ips(path, n_probe=500):
    from scapy.all import PcapReader, UDP, IP
    found = set()
    with PcapReader(path) as r:
        for i, pkt in enumerate(r):
            if UDP in pkt and IP in pkt:
                payload = bytes(pkt[UDP].payload)
                if len(payload) > 6 and payload[4:7] == b'VM3':
                    found.add(pkt[IP].src)
            if i >= n_probe:
                break
    return found


# --------------------------------------------------------------------------
# (vmm, ch) -> pad
# --------------------------------------------------------------------------
# Chain, mirroring cosmic_bench_analysis/p2_mapping.py (which is M3-validated
# for the DREAM/K59V readout):
#
#   (vmm, ch)
#     -> station, connector N, half          (P2_VMM_CABLING: bottom/top VMM)
#     -> within = ch (0..63)
#     -> strip   = half_base + f(within)     (WITHIN_STRATEGY, see below)
#     -> sector  = N - 1
#     -> channel_id = sector*128 + (strip-1)
#     -> pad_cx, pad_cy                      (P2_BASKET_mapping.csv)
#
# All three stations are identical P2 detectors, so the same 1280-pad map
# applies to each; they differ only in which connectors are instrumented
# (c4-c6) and in z.
#
# TWO LINKS ARE NOT YET VALIDATED FOR THE VMM HYBRID and are therefore
# swappable, exactly as p2_mapping.py keeps the DREAM one swappable:
#
#   WITHIN_STRATEGY  'reverse' is validated for DREAM/K59V, where the ordering
#                    is a property of that adapter. The VMM hybrid plugs in
#                    differently, so it must be re-validated here.
#   BOTTOM_VMM_HALF  whether the hybrid's bottom VMM reads the 'bot' half
#                    (strips 1-64) or the 'top' half (65-128).
#
# Validate with:  python3 vmm_map_scan.py <file.pcapng>
# which renders the beam spot under every combination and ranks them by
# compactness -- the same argument that settled the run-67 orientation and the
# det1 mapping. A wrong ordering scatters the spot across the fan.
STRIPS_PER_CONN = 128
CH_PER_HALF = 64
N_CONNECTORS = 10

WITHIN_STRATEGY = 'reverse'
BOTTOM_VMM_HALF = 'bot'

# Per-(station, connector_N, half) exceptions to WITHIN_STRATEGY, for ribbons
# cabled differently from the rest of the detector. The DREAM readout needed
# exactly this on P2_IN (flipped c_5_top ribbon, commit 659dfbb), and the VMM
# beam-spot scan finds P2_IN anomalous too. Filled in by vmm_map_scan.py
# --per-half; empty means every half uses WITHIN_STRATEGY.
#
# MEASURED 2026-07-30 (run_24/nominal_05, trigger-coincident hits): every half
# is 'reverse' except P2_IN c5 top, which is 'linear' — a flipped ribbon. This
# is INDEPENDENT confirmation of the DREAM-side finding (commit 659dfbb,
# "P2_IN mapping fix (flipped c_5_top ribbon)"): the same ribbon, recovered
# through a completely different readout chain. Median distance from the beam
# centroid for that half: linear 59.7 mm vs reverse 103.8 mm.
# Found by vmm_map_rough.py, which minimises SPECKLE (dead pads next to bright
# ones) rather than beam-spot compactness. Compactness is nearly blind to a
# permutation inside a 64-strip half -- it was the speckled texture on the hit
# maps that gave these away.
#   P2_IN c5 top  — also found by the compactness scan, and independently by
#                   the Dream side (commit 659dfbb, "flipped c_5_top ribbon")
#   P2_IN c6 top  — only the speckle metric sees this one
# Roughness after: P2_OUT 0.057, P2_MID 0.105, P2_IN 0.239 -> 0.184.
# P2_MID and P2_OUT are not improved by any ordering, i.e. they are already
# correct; P2_IN stays roughest even after the fix (see the plan doc).
STRATEGY_OVERRIDES = {
    ('P2_IN', 5, 'top'): 'linear',
    ('P2_IN', 6, 'top'): 'linear',
}

MAP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'P2_BASKET_mapping.csv')


def _within_to_strip(within, half, strategy=WITHIN_STRATEGY):
    """within (0..63) + half -> strip (1..128). Same forms as p2_mapping.py."""
    base = 65 if half == 'top' else 1
    if strategy == 'linear':
        off = within
    elif strategy == 'reverse':
        off = 63 - within
    elif strategy == 'pairswap':
        off = within ^ 1
    else:
        raise ValueError(f'unknown strategy {strategy!r}')
    return base + off


def vmm_half_map(bottom_half=BOTTOM_VMM_HALF):
    """{vmm_id: (station, connector_N, half)} from P2_VMM_CABLING."""
    top_half = 'top' if bottom_half == 'bot' else 'bot'
    out = {}
    for station, cfg in STATIONS.items():
        for cname, (_hyb, bot_vmm, top_vmm) in cfg['connectors'].items():
            n = int(cname.lstrip('c'))
            out[int(bot_vmm)] = (station, n, bottom_half)
            out[int(top_vmm)] = (station, n, top_half)
    return out


def build_pad_table(map_csv=None, strategy=WITHIN_STRATEGY,
                    bottom_half=BOTTOM_VMM_HALF, overrides=None):
    """One row per instrumented (vmm, ch), joined to pad geometry.

    Columns: vmm, ch, station, connector_N, half, sector, strip, channel_id,
             pad_cx, pad_cy, radius, phi, pad_w, pad_h, pad_angle, mapped.
    """
    pad = pd.read_csv(map_csv or MAP_CSV).set_index('channel_id').sort_index()
    halves = vmm_half_map(bottom_half)

    over = dict(STRATEGY_OVERRIDES if overrides is None else overrides)
    rows = []
    for vmm, (station, n, half) in sorted(halves.items()):
        sector = n - 1
        strat = over.get((station, n, half), strategy)
        for within in range(CH_PER_HALF):
            strip = int(_within_to_strip(within, half, strat))
            rows.append((vmm, within, station, n, half, sector, strip,
                         sector * STRIPS_PER_CONN + (strip - 1)))

    tab = pd.DataFrame(rows, columns=['vmm', 'ch', 'station', 'connector_N',
                                      'half', 'sector', 'strip', 'channel_id'])
    cols = [c for c in ('pad_cx', 'pad_cy', 'radius', 'phi', 'pad_w', 'pad_h',
                        'pad_angle', 'delta_phi') if c in pad.columns]
    tab = tab.merge(pad[cols], left_on='channel_id', right_index=True, how='left')
    tab['mapped'] = tab['pad_cx'].notna()
    tab.attrs.update(strategy=strategy, bottom_half=bottom_half,
                     overrides=dict(over))
    return tab


def attach_pads(hits, tab):
    """Left-join hits to the pad table on (vmm, ch)."""
    return hits.merge(tab, on=['vmm', 'ch'], how='left', copy=False)


def beam_spread(hits, tab, station=None):
    """RMS radius (mm) of the hit distribution in pad space, and the hit count.

    The mapping-quality metric: the beam is a compact spot, so the correct
    (strategy, bottom_half) combination minimises this. A wrong within-half
    ordering smears hits across the fan and inflates it.
    """
    h = attach_pads(hits, tab)
    h = h[h['mapped'].astype(bool)]
    if station is not None:
        h = h[h['station'] == station]
    if len(h) < 100:
        return float('nan'), len(h)
    x = h['pad_cx'].to_numpy(); y = h['pad_cy'].to_numpy()
    return float(np.sqrt(np.var(x) + np.var(y))), len(h)


def _neighbours(cx, cy, k=6):
    """Indices of the k nearest pads for each pad (brute force; ~400 pads)."""
    d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def map_roughness(hits, tab, station, min_frac=0.05):
    """How SPECKLED the occupancy is — the metric that actually finds a
    scrambled channel ordering.

    beam_spread() (compactness) is nearly blind to this: permuting channels
    inside a 64-strip half leaves the hits in the same part of the fan, so the
    RMS hardly moves. What a wrong ordering DOES produce is high-frequency
    texture -- dead pads sitting next to bright ones, and stripes -- which is
    what the eye picks up immediately on a hit map.

    So: for every pad in the beam region, compare its count to the median of
    its 6 nearest neighbours. A correct map is locally smooth (small values);
    a scrambled one is speckled (large). Returned value is the median over
    pads of |n - med| / (n + med + 1), in [0, 1].
    """
    h = attach_pads(hits, tab)
    h = h[h['mapped'].astype(bool) & (h['station'] == station)]
    sub = tab[(tab['station'] == station) & tab['mapped'].astype(bool)].reset_index(drop=True)
    if len(h) < 500 or len(sub) < 20:
        return float('nan')

    key = sub['vmm'].to_numpy().astype(np.int64) * 64 + sub['ch'].to_numpy()
    lut = {int(k): i for i, k in enumerate(key)}
    n = np.zeros(len(sub))
    hk = h['vmm'].to_numpy().astype(np.int64) * 64 + h['ch'].to_numpy()
    uk, uc = np.unique(hk, return_counts=True)
    for k, c in zip(uk, uc):
        i = lut.get(int(k))
        if i is not None:
            n[i] = c

    cx = sub['pad_cx'].to_numpy(float)
    cy = sub['pad_cy'].to_numpy(float)
    nb = _neighbours(cx, cy)
    med = np.median(n[nb], axis=1)
    # Only judge the illuminated region: empty fringes are genuinely empty.
    live = med > min_frac * np.max(med) if np.max(med) > 0 else np.zeros(len(n), bool)
    if live.sum() < 10:
        return float('nan')
    return float(np.median(np.abs(n[live] - med[live]) / (n[live] + med[live] + 1.0)))


# A pad is flagged hot when it holds more than this many times the median of
# its 6 spatial neighbours. Calibrated on run_25: the worst pad on the healthy
# P2_OUT is 4.7x, so 6x flags nothing there while catching the genuine outliers
# on IN (13-30x) and MID (8-986x).
HOT_PAD_RATIO = 6.0


def auto_hot_pads(hits, tab, ratio=HOT_PAD_RATIO, min_hits=50):
    """Flag pads far above their SPATIAL neighbours -> {vmm: [ch, ...]}.

    Strictly better than auto_hot_channels() on this detector: occupancy varies
    a lot across the fan, so a per-VMM median hides a channel that is hot
    relative to the pads actually next to it. Needs the pad map, so it only
    became possible once the mapping was validated.

    Pads whose neighbours are all dead give a large ratio by construction. They
    are flagged too -- a lone firing pad in a dead neighbourhood is not usable
    for tracking either way -- but the caller should log what was masked.
    """
    # Occupancy per (vmm, ch), counted once on the raw arrays. The pad geometry
    # is only ever needed per PAD, so joining it onto every hit is waste: on a
    # 5.5 M hit capture attach_pads() materialised ~1.5 GB (17 pad columns x
    # every hit) and was the single largest allocation in the whole reduce.
    # There are at most 32*64 keys, so the per-station work below indexes a
    # small histogram instead. The station/mapped filters that used to be
    # applied to the hits are redundant — `sub` already contains only mapped
    # pads of one station, so a key absent from it is simply never looked up.
    hk = hits['vmm'].to_numpy().astype(np.int64) * 64 + hits['ch'].to_numpy()
    uk, uc = np.unique(hk, return_counts=True)
    occ = dict(zip(uk.tolist(), uc.tolist()))

    found = {}
    for st in STATIONS:
        sub = tab[(tab['station'] == st) & tab['mapped'].astype(bool)].reset_index(drop=True)
        if len(sub) < 20:
            continue
        key = sub['vmm'].to_numpy().astype(np.int64) * 64 + sub['ch'].to_numpy()
        n = np.array([occ.get(int(k), 0) for k in key], dtype=float)
        if not n.any():
            continue
        nb = _neighbours(sub['pad_cx'].to_numpy(float), sub['pad_cy'].to_numpy(float))
        med = np.median(n[nb], axis=1)
        hot = np.flatnonzero((n >= min_hits) & (n > ratio * np.maximum(med, 1.0)))
        for i in hot:
            found.setdefault(int(sub['vmm'][i]), []).append(int(sub['ch'][i]))
    return {v: sorted(set(c)) for v, c in found.items()}
