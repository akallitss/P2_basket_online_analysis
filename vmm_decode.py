#!/usr/bin/env python3
"""Tier 0: vectorised VMM3a pcapng decoder.

Reproduces, bit for bit, the hits DataFrame that `vmm_qa/vmm_pcapng_qa.py`
builds with its per-word Python loop, but without scapy and without a Python
loop over hits. Two changes carry the speedup:

  * the pcapng is walked as raw blocks (only the block headers are touched in
    Python; the link layer is parsed with numpy over all packets at once),
  * every 6-byte VMM data word in the file is decoded in a single set of numpy
    expressions rather than one `struct.unpack_from` per word.

The one genuinely sequential piece is the marker state machine: each hit
carries the most recent marker values for its own (fec, vmm), so the value is
a forward-fill along the global word order. That is done per (fec, vmm) key
with `searchsorted` -- see `_last_value_before`.

Output columns (identical names, dtypes and values to the reference parser):
    fec, vmm, time, ch, adc, adc_calibrated, over_threshold, offset, bcid,
    tdc, timestamp_ns, srs_timestamp, abs_time_ns, trigger_time,
    trigger_counter, hit_valid

Usage:
    from vmm_decode import decode
    hits, meta = decode("capture.pcapng")
"""

import json
import os
import struct

import numpy as np
import pandas as pd

__all__ = ["decode", "decode_to", "load_hits", "load_frame", "derive",
           "iter_chunks", "read_packets", "detect_fec_ips", "COLUMNS",
           "RAW_COLUMNS", "DERIVED_COLUMNS", "CLOCK_PERIOD_NS"]

# the hits table, in order. dtypes are fixed by the reference parser.
COLUMNS = [
    "fec", "vmm", "time", "ch", "adc", "adc_calibrated", "over_threshold",
    "offset", "bcid", "tdc", "timestamp_ns", "srs_timestamp", "abs_time_ns",
    "trigger_time", "trigger_counter", "hit_valid",
]

# What the hardware actually sends. These are stored.
RAW_COLUMNS = [
    "fec", "vmm", "time", "ch", "adc", "over_threshold", "offset", "bcid",
    "tdc", "srs_timestamp", "trigger_time", "trigger_counter",
]

# Pure functions of RAW_COLUMNS (plus a calibration). Recomputed on load rather
# than stored: they are 27 MB per 11 MB capture, and storing them would freeze
# one calibration into the file, forcing a re-decode whenever it is retuned.
DERIVED_COLUMNS = ["adc_calibrated", "timestamp_ns", "abs_time_ns", "hit_valid"]

COLUMN_DTYPES = {
    "fec": np.uint8, "vmm": np.uint8, "time": np.uint32, "ch": np.uint8,
    "adc": np.uint16, "adc_calibrated": np.float32, "over_threshold": np.bool_,
    "offset": np.int8, "bcid": np.uint16, "tdc": np.uint8,
    "timestamp_ns": np.float64, "srs_timestamp": np.uint64,
    "abs_time_ns": np.float64, "trigger_time": np.uint64,
    "trigger_counter": np.uint16, "hit_valid": np.bool_,
}

# --- physics/format constants: must track vmm_pcapng_qa.py ---
CLOCK_PERIOD_NS = 22.5   # ns per BCID count (44.44 MHz clock)
TAC_SLOPE_NS = 60.0      # ns full-scale of the TDC TAC ramp
TDC_RANGE = 255          # TDC full-scale bin count (vmm-sdat SRSTime::tdc_range)
PROBE_PACKETS = 500      # packets sampled by detect_fec_ips

_VM3 = (ord("V") << 16) | (ord("M") << 8) | ord("3")

# marker state is keyed by (fec_id, vmm_id) with fec 0-15 and vmm 0-31
_NKEYS = 16 * 32

# words decoded per chunk; ~6 B raw + ~24 B of decoded columns each, so 5e6
# keeps a chunk's transient footprint around 150 MB
_MAX_WORDS = 5_000_000

# pcapng header words held in RAM at a time (64 MB); module-level so tests can
# shrink it to exercise the slab-reload path
_SLAB_WORDS = 1 << 24

# pcapng block types
_BT_SHB = 0x0A0D0D0A
_BT_IDB = 0x00000001
_BT_EPB = 0x00000006
_BT_SPB = 0x00000003

_BOM = 0x1A2B3C4D
_LINKTYPE_ETHERNET = 1


# ---------------------------------------------------------------------------
# pcapng container
# ---------------------------------------------------------------------------

def _walk_blocks(path, max_packets=None):
    """Return (u8, pkt_off, pkt_caplen, linktypes).

    Walks the pcapng block chain to find every Enhanced/Simple Packet Block.
    Only block headers are read in Python -- roughly one iteration per packet,
    which is ~1e3 for a 45 s capture and ~2e5 for a 2 GB file.

    The file is memory-mapped rather than read: a 2 GB capture must not be
    pulled into a 7 GB machine that kills QA at 80 % RAM. Block headers are
    served from a 64 MB in-RAM slab rather than read from the memmap one scalar
    at a time -- memmap.__getitem__ costs ~75 us per scalar, which on a 2 GB
    file turns the walk alone into ~35 s.

    `max_packets` stops the walk early; without it a --max-packets=3000 run on
    a 2 GB capture still pays for chasing all 230k blocks.
    """
    size = os.path.getsize(path)
    if size < 12:
        raise ValueError(f"{path}: too small to be a pcapng")

    u8 = np.memmap(path, dtype=np.uint8, mode="r")

    head = bytes(u8[:12])
    btype0 = struct.unpack_from("<I", head, 0)[0]
    if btype0 != _BT_SHB:
        # A classic .pcap starts with 0xa1b2c3d4 / 0xd4c3b2a1 instead.
        if struct.unpack_from("<I", head, 0)[0] in (0xA1B2C3D4, 0xD4C3B2A1):
            raise ValueError(
                f"{path}: this is a classic .pcap, not a .pcapng. "
                "Only pcapng is produced by the DAQ's dumpcap; convert with editcap."
            )
        raise ValueError(f"{path}: not a pcapng (first block type 0x{btype0:08x})")

    bom = struct.unpack_from("<I", head, 8)[0]
    endian = "<" if bom == _BOM else ">"
    u32 = np.memmap(path, dtype=(endian + "u4"), mode="r",
                    shape=(size // 4,))

    pkt_off = []
    pkt_len = []
    linktypes = []

    n_words = size // 4
    slab_words = _SLAB_WORDS
    slab = None
    slab_lo = slab_hi = -1

    pos = 0
    n = size
    while pos + 12 <= n:
        w = pos >> 2
        # Reload when w falls outside the slab, or when there is not enough
        # headroom left for a block header -- unless the slab already reaches
        # EOF, in which case no reload can help and we must not loop forever.
        if not (slab_lo <= w < slab_hi) or (w + 8 >= slab_hi < n_words):
            slab_lo = w
            slab_hi = min(w + slab_words, n_words)
            if slab_hi - slab_lo < 2:
                break
            slab = np.array(u32[slab_lo:slab_hi])   # one sequential copy
        i = w - slab_lo
        avail = slab_hi - slab_lo          # words readable at slab[i + k]
        btype = int(slab[i])
        blen = int(slab[i + 1])
        if blen < 12 or blen % 4 or pos + blen > n:
            # Truncated tail: dumpcap was mid-write, or the file was cut.
            break
        if btype == _BT_EPB:
            #  interface_id, ts_hi, ts_lo, caplen, origlen, data...
            if blen < 32 or i + 5 >= avail:
                break
            pkt_off.append(pos + 28)
            pkt_len.append(int(slab[i + 5]))
        elif btype == _BT_SPB:
            # Simple Packet Block has no caplen; the block length bounds it.
            if blen < 16 or i + 2 >= avail:
                break
            pkt_off.append(pos + 12)
            pkt_len.append(min(int(slab[i + 2]), blen - 16))
        elif btype == _BT_IDB:
            if blen < 20 or i + 2 >= avail:
                break
            linktypes.append(int(slab[i + 2]) & 0xFFFF)
        elif btype == _BT_SHB:
            if blen < 28 or i + 2 >= avail:
                break
            if int(slab[i + 2]) not in (_BOM, 0x4D3C2B1A):
                break
        pos += blen
        if max_packets is not None and len(pkt_off) >= max_packets:
            break

    del slab
    return (u8,
            np.asarray(pkt_off, dtype=np.int64),
            np.asarray(pkt_len, dtype=np.int64),
            linktypes)


def read_packets(path, max_packets=None):
    """Locate every UDP payload in a pcapng, vectorised over packets.

    Returns (u8, payload_start, payload_len, src_ip_u32, n_packets).
    `n_packets` counts every captured packet, matching the reference parser's
    `pkt_count` (which drives --max-packets), not just the UDP ones.
    """
    u8, off, caplen, linktypes = _walk_blocks(path, max_packets=max_packets)

    if linktypes and any(lt != _LINKTYPE_ETHERNET for lt in linktypes):
        raise ValueError(
            f"{path}: link type(s) {sorted(set(linktypes))} -- only Ethernet (1) "
            "is supported; the DAQ captures Ethernet."
        )

    n_packets = len(off)
    if max_packets is not None and n_packets > max_packets:
        off = off[:max_packets]
        caplen = caplen[:max_packets]
        n_packets = len(off)

    if n_packets == 0:
        empty = np.zeros(0, dtype=np.int64)
        return u8, empty, empty, np.zeros(0, dtype=np.uint32), 0

    # --- Ethernet, with any number of VLAN tags (802.1Q / 802.1ad) ---
    ok = caplen >= 14
    ethertype = np.zeros(n_packets, dtype=np.uint32)
    l3 = off + 14
    idx = np.flatnonzero(ok)
    ethertype[idx] = (u8[off[idx] + 12].astype(np.uint32) << 8 |
                      u8[off[idx] + 13].astype(np.uint32))
    for _ in range(3):  # QinQ nests at most a couple deep in practice
        vlan = ok & ((ethertype == 0x8100) | (ethertype == 0x88A8) |
                     (ethertype == 0x9100))
        if not vlan.any():
            break
        vi = np.flatnonzero(vlan)
        l3[vi] += 4
        good = vi[(l3[vi] + 2) <= (off[vi] + caplen[vi])]
        ethertype[vi] = 0
        ethertype[good] = (u8[l3[good] - 2].astype(np.uint32) << 8 |
                           u8[l3[good] - 1].astype(np.uint32))

    # --- IPv4 + UDP ---
    ok &= ethertype == 0x0800
    ok &= (l3 + 20) <= (off + caplen)
    idx = np.flatnonzero(ok)

    ihl = np.zeros(n_packets, dtype=np.int64)
    proto = np.zeros(n_packets, dtype=np.uint8)
    src = np.zeros(n_packets, dtype=np.uint32)
    if len(idx):
        vers_ihl = u8[l3[idx]]
        ihl[idx] = (vers_ihl & 0x0F).astype(np.int64) * 4
        proto[idx] = u8[l3[idx] + 9]
        src[idx] = (u8[l3[idx] + 12].astype(np.uint32) << 24 |
                    u8[l3[idx] + 13].astype(np.uint32) << 16 |
                    u8[l3[idx] + 14].astype(np.uint32) << 8 |
                    u8[l3[idx] + 15].astype(np.uint32))
        ok[idx] &= ((vers_ihl >> 4) == 4) & (ihl[idx] >= 20) & (proto[idx] == 17)

    l4 = l3 + ihl
    pstart = l4 + 8
    # UDP length field bounds the payload; fall back to the capture length.
    plen = np.zeros(n_packets, dtype=np.int64)
    idx = np.flatnonzero(ok & ((l4 + 8) <= (off + caplen)))
    if len(idx):
        udp_len = (u8[l4[idx] + 4].astype(np.int64) << 8 |
                   u8[l4[idx] + 5].astype(np.int64))
        avail = off[idx] + caplen[idx] - pstart[idx]
        plen[idx] = np.minimum(np.maximum(udp_len - 8, 0), avail)
    keep = plen > 0

    return u8, pstart[keep], plen[keep], src[keep], n_packets


def _ip_str(v):
    return f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{(v >> 8) & 0xFF}.{v & 0xFF}"


def _ip_u32(s):
    a, b, c, d = (int(x) for x in s.split("."))
    return np.uint32((a << 24) | (b << 16) | (c << 8) | d)


def detect_fec_ips(path, n_probe=PROBE_PACKETS):
    """Source IPs carrying VM3 payloads, sampled over the first n_probe packets."""
    u8, pstart, plen, src, _ = read_packets(path, max_packets=n_probe)
    if len(pstart) == 0:
        return {}
    m = plen > 6
    ps, sr = pstart[m], src[m]
    if len(ps) == 0:
        return {}
    magic = (u8[ps + 4].astype(np.uint32) << 16 |
             u8[ps + 5].astype(np.uint32) << 8 |
             u8[ps + 6].astype(np.uint32))
    sr = sr[magic == _VM3]
    vals, counts = np.unique(sr, return_counts=True)
    return {_ip_str(int(v)): int(c) for v, c in zip(vals, counts)}


# ---------------------------------------------------------------------------
# marker forward-fill
# ---------------------------------------------------------------------------

def _last_value_before(upd_pos, upd_key, upd_val, q_pos, q_key, n_hits):
    """For each query, the most recent update value with the same key.

    This is the vectorised form of the reference parser's `markers` dict: a hit
    takes whatever was last written to markers[(fec, vmm)] before it in the
    word stream, or 0 if that key was never written.

    Both sides are collapsed to a single sort key `key * STRIDE + pos`, so one
    global searchsorted answers every hit at once. Because `pos` is bounded by
    STRIDE, ordering by the combined key orders first by key and then by
    position, which is exactly the "previous marker for my own (fec, vmm)"
    lookup. A per-key Python loop with boolean masks costs ~5 passes over the
    full hit array per key; this is one O(n log m) pass in total.
    """
    out = np.zeros(n_hits, dtype=np.uint64)
    if len(upd_pos) == 0 or n_hits == 0:
        return out

    stride = np.int64(max(int(q_pos[-1]) if n_hits else 0,
                          int(upd_pos[-1])) + 2)
    u_comb = upd_key.astype(np.int64) * stride + upd_pos
    order = np.argsort(u_comb, kind="stable")
    u_comb = u_comb[order]
    u_key = upd_key.astype(np.int64)[order]
    u_val = upd_val[order]

    q_comb = q_key.astype(np.int64) * stride + q_pos
    j = np.searchsorted(u_comb, q_comb, side="left") - 1
    # j < 0, or the update found belongs to a different key -> never written
    ok = j >= 0
    jc = np.clip(j, 0, None)
    ok &= u_key[jc] == q_key.astype(np.int64)
    np.copyto(out, u_val[jc], where=ok)
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def iter_chunks(path, data_format="SRS", src_ips=None, max_packets=None,
                calibration=None, max_words=_MAX_WORDS, meta=None):
    """Yield the final 16 hit columns one chunk at a time.

    This is the streaming core shared by `decode` (which concatenates) and
    `decode_to` (which writes each chunk straight to disk and drops it). Both
    therefore produce identical values by construction.

    `meta`, if given, is a dict filled in once the generator is exhausted.
    """
    if data_format not in ("SRS", "TRG"):
        raise ValueError(f"data_format must be 'SRS' or 'TRG', got {data_format!r}")
    if meta is None:
        meta = {}

    u8, pstart, plen, src, n_packets = read_packets(path, max_packets=max_packets)

    if src_ips is None:
        found = detect_fec_ips(path)
        if not found:
            raise ValueError(
                f"{path}: no VM3 packets in the first {PROBE_PACKETS} packets."
            )
        src_ips = set(found)
    want = np.array([_ip_u32(s) for s in sorted(src_ips)], dtype=np.uint32)

    # --- select VM3 packets from the wanted FEC(s) ---
    m = (plen > 22) & np.isin(src, want)
    pstart, plen = pstart[m], plen[m]
    if len(pstart):
        magic = (u8[pstart + 4].astype(np.uint32) << 16 |
                 u8[pstart + 5].astype(np.uint32) << 8 |
                 u8[pstart + 6].astype(np.uint32))
        m2 = magic == _VM3
        pstart, plen = pstart[m2], plen[m2]

    if len(pstart) == 0:
        meta.update(_empty_meta(n_packets, src_ips, data_format))
        return

    # --- per-packet header: frame counter and FEC id ---
    frame_counter = (u8[pstart].astype(np.uint32) << 24 |
                     u8[pstart + 1].astype(np.uint32) << 16 |
                     u8[pstart + 2].astype(np.uint32) << 8 |
                     u8[pstart + 3].astype(np.uint32))
    # reference: (u32_at_byte4 >> 4) & 0x0F, i.e. the upper nibble of byte 7
    fec_id = ((u8[pstart + 7].astype(np.uint32) >> 4) & 0x0F).astype(np.uint8)

    # --- flatten every 6-byte data word, in file order ---
    # reference: for i in range(0, len(block) - 22, 6): read 6 bytes at i + 16
    span = plen - 22
    nwords = np.where(span > 0, (span + 5) // 6, 0)
    total = int(nwords.sum())
    if total == 0:
        meta.update(_empty_meta(n_packets, src_ips, data_format))
        return

    # Split the packet list so no chunk decodes more than max_words words at
    # once. A 2 GB capture holds ~3e8 words; materialising those in one go
    # would blow the 80 % RAM kill the watcher enforces. Chunks always break on
    # packet boundaries, and marker state is carried across them.
    cum = np.cumsum(nwords)
    bounds = [0]
    while bounds[-1] < len(pstart):
        done = int(cum[bounds[-1] - 1]) if bounds[-1] > 0 else 0
        nxt = int(np.searchsorted(cum, done + max_words, side="left")) + 1
        bounds.append(min(max(nxt, bounds[-1] + 1), len(pstart)))

    carry = np.zeros((3, _NKEYS), dtype=np.uint64)
    n_vm3 = 0
    n_markers = 0
    n_hits = 0
    n_invalid = 0
    off_seen = set()

    for ci in range(len(bounds) - 1):
        lo, hi = bounds[ci], bounds[ci + 1]
        c_start, c_len = pstart[lo:hi], nwords[lo:hi]
        c_total = int(c_len.sum())
        if c_total == 0:
            continue

        pkt_of_word = np.repeat(np.arange(hi - lo, dtype=np.int64), c_len)
        starts = np.cumsum(c_len) - c_len

        # Copy each packet's word region into one contiguous buffer with
        # sequential slice reads. Fancy-indexing the memmap instead costs
        # ~3.5 s/GB, because every gather round-trips memmap.__getitem__.
        wbuf = np.empty(c_total * 6, dtype=np.uint8)
        for i in range(hi - lo):
            n = int(c_len[i])
            if n:
                s = int(c_start[i]) + 16
                o = int(starts[i]) * 6
                wbuf[o:o + n * 6] = u8[s:s + n * 6]
        w6 = wbuf.reshape(c_total, 6)
        d1 = w6[:, :4].copy().view(">u4").ravel().astype(np.uint32)
        d2 = w6[:, 4:6].copy().view(">u2").ravel().astype(np.uint32)
        del wbuf, w6

        is_hit = (d2 & 0x8000) != 0
        hit_pos = np.flatnonzero(is_hit)
        mrk_pos = np.flatnonzero(~is_hit)
        n_markers += len(mrk_pos)

        c_fec = fec_id[lo:hi]
        h1 = d1[hit_pos]
        h2 = d2[hit_pos]
        hit_fec = c_fec[pkt_of_word[hit_pos]]
        vmm = ((h1 >> 22) & 0x1F).astype(np.uint8)
        hit_key = hit_fec.astype(np.int64) * 32 + vmm.astype(np.int64)

        srs_ts, trg_time, trg_ctr, carry = _decode_markers(
            d1, d2, mrk_pos, c_fec, pkt_of_word, data_format,
            hit_pos, hit_key, carry)
        del d1, d2

        ch = ((h2 >> 8) & 0x3F).astype(np.uint8)
        adc = ((h1 >> 12) & 0x3FF).astype(np.uint16)
        ot = ((h2 >> 14) & 0x1).astype(np.uint8)
        raw_off = ((h1 >> 27) & 0x1F).astype(np.int16)
        bcid_raw = (h1 & 0xFFF).astype(np.uint16)
        tdc = (h2 & 0xFF).astype(np.uint8)
        time_col = frame_counter[lo:hi][pkt_of_word[hit_pos]]
        offset = np.where(raw_off < 16, raw_off, raw_off - 32).astype(np.int8)

        n_hits += len(hit_pos)
        n_invalid += int((offset < np.int8(-1)).sum())
        if len(offset):
            off_seen.update(np.unique(offset).tolist())
        # packets yielding >=1 hit; pkt_of_word is non-decreasing and chunks
        # never split a packet, so distinct count == transitions + 1
        if len(hit_pos):
            pw = pkt_of_word[hit_pos]
            n_vm3 += int(np.count_nonzero(np.diff(pw))) + 1

        yield {
            "fec": hit_fec,
            "vmm": vmm,
            "time": time_col,
            "ch": ch,
            "adc": adc,
            "over_threshold": ot.astype(bool),
            "offset": offset,
            "bcid": _gray2bin(bcid_raw),
            "tdc": tdc,
            "srs_timestamp": srs_ts,
            "trigger_time": trg_time,
            "trigger_counter": trg_ctr.astype(np.uint16),
        }

    meta.update({
        "n_packets": int(n_packets),
        # counts packets yielding >= 1 hit, matching the reference's
        # `if len(fec_buf) > n_before` test (marker-only packets do not count)
        "n_vm3_packets": int(n_vm3),
        "n_hits": int(n_hits),
        "fec_ips": sorted(src_ips),
        "data_format": data_format,
        "n_words": int(total),
        "n_markers": int(n_markers),
        "n_chunks": len(bounds) - 1,
        "n_invalid_offset": int(n_invalid),
        "offset_constant": bool(len(off_seen) == 1),
    })


def decode(path, data_format="SRS", src_ips=None, max_packets=None,
           calibration=None, max_words=_MAX_WORDS):
    """Decode a VMM pcapng into the hits DataFrame.

    calibration: optional 4-tuple (time_offset, time_slope, adc_offset,
    adc_slope), each (32, 64) float64, as returned by
    vmm_pcapng_qa.load_calibration.
    max_words: decode at most this many VMM words per chunk, bounding peak
    memory on multi-GB captures. Results are independent of the value.

    Holds every hit in memory. For multi-GB captures use `decode_to`, which
    streams to disk instead -- a full 2 GB run is ~3e8 hits, which is ~18 GB
    as a DataFrame.
    """
    meta = {}
    parts = list(iter_chunks(path, data_format, src_ips, max_packets,
                             calibration, max_words, meta))
    if not parts:
        return _empty_hits(), meta

    cols = {}
    for name in RAW_COLUMNS:
        cols[name] = (np.concatenate([p[name] for p in parts]) if len(parts) > 1
                      else parts[0][name])
        for p in parts:
            del p[name]
    del parts
    derive(cols, calibration)
    return pd.DataFrame({c: cols[c] for c in COLUMNS}), meta


def derive(cols, calibration=None):
    """Add the derived columns in place, from the raw ones.

    The single source of these formulas (lifted from vmm_pcapng_qa.py) --
    `decode` and `load_hits` both route through here, so an in-memory decode
    and a round-trip through the column store cannot drift apart.
    """
    vmm, ch, adc = cols["vmm"], cols["ch"], cols["adc"]
    offset, bcid, tdc = cols["offset"], cols["bcid"], cols["tdc"]
    srs_ts = cols["srs_timestamp"]

    if calibration is None:
        t_off, t_slp, a_off, a_slp = 0.0, 1.0, 0.0, 1.0
    else:
        cal_to, cal_ts, cal_ao, cal_as = calibration
        vi = np.asarray(vmm, dtype=np.intp)
        ci = np.asarray(ch, dtype=np.intp)
        t_off, t_slp = cal_to[vi, ci], cal_ts[vi, ci]
        a_off, a_slp = cal_ao[vi, ci], cal_as[vi, ci]

    t_coarse = (np.asarray(offset, dtype=np.float64) * 4096 +
                np.asarray(bcid, dtype=np.float64)) * CLOCK_PERIOD_NS
    t_fine = (CLOCK_PERIOD_NS -
              np.asarray(tdc, dtype=np.float64) * TAC_SLOPE_NS / TDC_RANGE
              - t_off) * t_slp
    timestamp_ns = t_coarse + t_fine

    cols["adc_calibrated"] = (a_slp * np.asarray(adc, dtype=np.float64)
                              + a_off).astype(np.float32)
    cols["timestamp_ns"] = timestamp_ns
    cols["abs_time_ns"] = np.asarray(srs_ts, dtype=np.float64) * 25.0 + timestamp_ns
    cols["hit_valid"] = np.asarray(offset) >= np.int8(-1)
    return cols



def _resolve(upd_pos, upd_key, upd_val, hit_pos, hit_key, n_hits, carry_row):
    """One marker slot: seed with the carry-in state, resolve, return new carry.

    The carry is injected as a synthetic update at position -1 for every key, so
    a hit that precedes any marker in this chunk still sees whatever the
    previous chunk left in markers[(fec, vmm)] -- exactly what the reference's
    persistent dict does. Keys never written anywhere carry 0, which is the
    reference's _DEFAULT_MARKER.
    """
    seed_key = np.arange(_NKEYS, dtype=np.int64)
    seed_pos = np.full(_NKEYS, -1, dtype=np.int64)
    all_pos = np.concatenate([seed_pos, upd_pos.astype(np.int64)])
    all_key = np.concatenate([seed_key, upd_key.astype(np.int64)])
    all_val = np.concatenate([carry_row, upd_val.astype(np.uint64)])

    out = _last_value_before(all_pos, all_key, all_val, hit_pos, hit_key, n_hits)

    # new carry = last update per key in this chunk, else the old carry
    new_carry = carry_row.copy()
    if len(upd_pos):
        k = upd_key.astype(np.int64)
        # updates arrive in position order, so the last occurrence of each key
        # is the first occurrence when scanned backwards
        ku, first_rev = np.unique(k[::-1], return_index=True)
        new_carry[ku] = upd_val.astype(np.uint64)[len(k) - 1 - first_rev]
    return out, new_carry


def _decode_markers(d1, d2, mrk_pos, fec_id, pkt_of_word, data_format,
                    hit_pos, hit_key, carry):
    """Resolve srs_timestamp / trigger_time / trigger_counter for every hit.

    `carry` is a (3, _NKEYS) uint64 array holding the marker state left by the
    previous chunk; the updated state is returned alongside the hit columns.
    """
    n_hits = len(hit_pos)
    empty_pos = np.zeros(0, dtype=np.int64)
    empty_val = np.zeros(0, dtype=np.uint64)

    if len(mrk_pos):
        m1 = d1[mrk_pos]
        m2 = d2[mrk_pos]
        m_fec = fec_id[pkt_of_word[mrk_pos]].astype(np.int64)
        vmmid = ((m2 >> 10) & 0x1F).astype(np.int64)
        # 42-bit value: d1 is 32 bits, so promote before shifting
        val42 = ((m1.astype(np.uint64) << np.uint64(10)) |
                 (m2 & 0x3FF).astype(np.uint64))
    else:
        m1 = m2 = m_fec = vmmid = val42 = None

    new_carry = np.empty_like(carry)

    if data_format == "SRS":
        # Reference takes vmmid at face value in SRS mode: this FEC hosts 20
        # VMMs, and gating on `vmmid < 16` silently dropped every marker for
        # VMMs 16-19 (see plan doc 2A.1).
        if len(mrk_pos):
            up, uk, uv = mrk_pos, m_fec * 32 + vmmid, val42
        else:
            up, uk, uv = empty_pos, empty_pos, empty_val
        srs, new_carry[0] = _resolve(up, uk, uv, hit_pos, hit_key, n_hits, carry[0])
        z64 = np.zeros(n_hits, dtype=np.uint64)
        new_carry[1] = carry[1]
        new_carry[2] = carry[2]
        return srs, z64, z64.copy(), new_carry

    # --- TRG ---
    if len(mrk_pos):
        is_vmm = vmmid < 16
        # VMM markers: only stored when the value fits in a BCID (< 4096)
        sel = is_vmm & (val42 < np.uint64(4096))
        trg = ~is_vmm
        trg_key = m_fec * 32 + (vmmid % 16)
        flag = (m1 >> 28) & 0x0F
        is_ctr = trg & (flag == 0xF)
        is_time = trg & (flag != 0xF)
        ctr_val = ((m1 & 0x3F).astype(np.uint64) * np.uint64(1024) +
                   (m2 & 0x3FF).astype(np.uint64))
        time_val = m1.astype(np.uint64) << np.uint64(10)
        srs_args = (mrk_pos[sel], (m_fec * 32 + vmmid)[sel], val42[sel])
        ctr_args = (mrk_pos[is_ctr], trg_key[is_ctr], ctr_val[is_ctr])
        tim_args = (mrk_pos[is_time], trg_key[is_time], time_val[is_time])
    else:
        srs_args = ctr_args = tim_args = (empty_pos, empty_pos, empty_val)

    srs, new_carry[0] = _resolve(*srs_args, hit_pos, hit_key, n_hits, carry[0])
    trigger_time, new_carry[1] = _resolve(*tim_args, hit_pos, hit_key, n_hits, carry[1])
    counter, new_carry[2] = _resolve(*ctr_args, hit_pos, hit_key, n_hits, carry[2])
    return srs, trigger_time, counter, new_carry


def _gray2bin(arr):
    """Gray -> binary, matching vmm_pcapng_qa.gray2bin_np / Lua gray2bin32."""
    a = arr.astype(np.uint32)
    a = a ^ (a >> 16)
    a = a ^ (a >> 8)
    a = a ^ (a >> 4)
    a = a ^ (a >> 2)
    a = a ^ (a >> 1)
    return a.astype(np.uint16)


# ---------------------------------------------------------------------------
# streaming column store
# ---------------------------------------------------------------------------
#
# One .npy per column in an output directory, plus meta.json. Chosen over
# parquet/HDF5 because the DAQ venv has none of pyarrow, fastparquet, h5py or
# pytables, and adding a dependency to the machine the beam depends on is not
# worth it. The format costs nothing: Tier 2 opens columns with
# np.load(mmap_mode='r') and pages in only what it touches, so a per-VMM ADC
# histogram never reads the timestamp columns at all.
#
# .npy stores its length in a header, which is not known until the last chunk
# is written. The header is therefore written with shape (0,) and a fixed
# padded length, then rewritten in place at close.

_NPY_HDR_LEN = 118          # 10-byte prefix + 118 = 128, a multiple of 64


def _npy_header(dtype, n):
    from numpy.lib import format as npf
    descr = npf.dtype_to_descr(np.dtype(dtype))
    s = "{'descr': %r, 'fortran_order': False, 'shape': (%d,), }" % (descr, n)
    pad = _NPY_HDR_LEN - len(s) - 1
    if pad < 0:
        raise ValueError(f"npy header too long for {dtype!r}")
    body = (s + " " * pad + "\n").encode("latin1")
    return b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", _NPY_HDR_LEN) + body


class HitWriter:
    """Append hit columns to a directory of .npy files without buffering them."""

    def __init__(self, out_dir, columns=RAW_COLUMNS, dtypes=COLUMN_DTYPES,
                 constants=None):
        self.dir = out_dir
        self.columns = list(columns)
        self.dtypes = dict(dtypes)
        # Columns known to hold one value for every hit are recorded in
        # meta.json instead of being written out. In SRS mode trigger_time and
        # trigger_counter are provably 0 -- only the TRG marker branch ever
        # writes them -- and they are 10 of the 32 bytes per hit.
        self.constants = dict(constants or {})
        self.stored = [c for c in self.columns if c not in self.constants]
        os.makedirs(out_dir, exist_ok=True)
        self.n = 0
        self._f = {}
        for c in self.stored:
            f = open(os.path.join(out_dir, f"{c}.npy"), "wb")
            f.write(_npy_header(self.dtypes[c], 0))
            self._f[c] = f

    def append(self, cols):
        n_new = None
        for c in self.columns:
            a = np.ascontiguousarray(cols[c], dtype=self.dtypes[c])
            if n_new is None:
                n_new = len(a)
            elif len(a) != n_new:
                raise ValueError(f"column {c} has {len(a)} rows, expected {n_new}")
            if c in self.constants:
                v = self.constants[c]
                if a.size and not bool((a == v).all()):
                    raise ValueError(
                        f"column {c} was declared constant {v!r} but this chunk "
                        "contains other values")
                continue
            self._f[c].write(a.tobytes())
        self.n += n_new or 0

    def close(self, meta=None):
        for c in self.stored:
            f = self._f[c]
            f.flush()
            f.seek(0)
            f.write(_npy_header(self.dtypes[c], self.n))   # same length, in place
            f.close()
        self._f.clear()
        if meta is not None:
            with open(os.path.join(self.dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2, sort_keys=True, default=str)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._f:
            self.close()


def decode_to(path, out_dir, data_format="SRS", src_ips=None, max_packets=None,
              calibration=None, max_words=_MAX_WORDS):
    """Decode a pcapng straight to a column store, at flat memory cost.

    Peak memory is one chunk, not one file, so a 2 GB capture (~3e8 hits, which
    is ~18 GB as a DataFrame) decodes in bounded RAM. Returns the meta dict,
    which is also written to <out_dir>/meta.json.
    """
    meta = {}
    # In SRS mode the trigger slots are never written by the marker decoder,
    # so they are constant 0 and need not hit the disk (31 % of the store).
    constants = ({"trigger_time": 0, "trigger_counter": 0}
                 if data_format == "SRS" else {})
    writer = HitWriter(out_dir, constants=constants)
    try:
        for cols in iter_chunks(path, data_format, src_ips, max_packets,
                                calibration, max_words, meta):
            writer.append(cols)
            del cols
    except BaseException:
        writer.close()
        raise
    meta["source"] = os.path.abspath(path)
    meta["columns"] = list(RAW_COLUMNS)
    meta["derived_columns"] = list(DERIVED_COLUMNS)
    meta["constant_columns"] = {k: int(v) for k, v in constants.items()}
    writer.close(meta)
    return meta


def load_hits(out_dir, columns=None, mmap=True, derive_cols=True,
              calibration=None):
    """Read a column store back. Returns (dict of arrays, meta).

    Memory-mapped by default, so reading one column of a 2 GB store pages in
    only that column. `derive_cols` recomputes the derived columns via
    `derive()`; that materialises float64 arrays, so pass False (or a narrow
    `columns` list) when you only need raw ones.
    """
    meta_path = os.path.join(out_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    stored = meta.get("columns", RAW_COLUMNS)

    if columns is None:
        names = list(stored)
    else:
        # pull in whatever raw columns the requested derived ones need
        names = [c for c in columns if c in stored]
        if derive_cols and any(c in DERIVED_COLUMNS for c in columns):
            names = list(dict.fromkeys(names + list(stored)))

    const = meta.get("constant_columns", {})
    n = int(meta.get("n_hits", 0))
    out = {}
    for c in names:
        if c in const:
            out[c] = np.full(n, const[c], dtype=COLUMN_DTYPES[c])
        else:
            out[c] = np.load(os.path.join(out_dir, f"{c}.npy"),
                             mmap_mode="r" if mmap else None)
    if derive_cols and set(RAW_COLUMNS) <= set(out):
        derive(out, calibration)
    if columns is not None:
        out = {c: out[c] for c in columns if c in out}
    return out, meta


def load_frame(out_dir, calibration=None):
    """The column store as the same 16-column DataFrame `decode` returns."""
    cols, meta = load_hits(out_dir, calibration=calibration, mmap=False)
    return pd.DataFrame({c: cols[c] for c in COLUMNS}), meta


def _empty_meta(n_packets, src_ips, data_format):
    return {
        "n_packets": int(n_packets), "n_vm3_packets": 0, "n_hits": 0,
        "fec_ips": sorted(src_ips), "data_format": data_format,
        "n_words": 0, "n_markers": 0, "n_chunks": 0,
        "n_invalid_offset": 0, "offset_constant": False,
    }


def _empty_hits():
    return pd.DataFrame({
        "fec": np.zeros(0, np.uint8), "vmm": np.zeros(0, np.uint8),
        "time": np.zeros(0, np.uint32), "ch": np.zeros(0, np.uint8),
        "adc": np.zeros(0, np.uint16), "adc_calibrated": np.zeros(0, np.float32),
        "over_threshold": np.zeros(0, bool), "offset": np.zeros(0, np.int8),
        "bcid": np.zeros(0, np.uint16), "tdc": np.zeros(0, np.uint8),
        "timestamp_ns": np.zeros(0, np.float64),
        "srs_timestamp": np.zeros(0, np.uint64),
        "abs_time_ns": np.zeros(0, np.float64),
        "trigger_time": np.zeros(0, np.uint64),
        "trigger_counter": np.zeros(0, np.uint16),
        "hit_valid": np.zeros(0, bool),
    })


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap_file")
    ap.add_argument("--format", choices=["SRS", "TRG"], default="SRS")
    ap.add_argument("--max-packets", type=int, default=None)
    ap.add_argument("--out-dir", metavar="DIR",
                    help="stream hits to a column store here instead of "
                         "holding them in memory (required for multi-GB files)")
    ap.add_argument("--max-words", type=int, default=_MAX_WORDS,
                    help=f"VMM words decoded per chunk (default {_MAX_WORDS:,})")
    args = ap.parse_args()

    t0 = time.time()
    if args.out_dir:
        meta = decode_to(args.pcap_file, args.out_dir, data_format=args.format,
                         max_packets=args.max_packets, max_words=args.max_words)
        hits = None
    else:
        hits, meta = decode(args.pcap_file, data_format=args.format,
                            max_packets=args.max_packets,
                            max_words=args.max_words)
    dt = time.time() - t0
    print(f"{args.pcap_file}")
    for k, v in meta.items():
        print(f"  {k:18s} {v}")
    print(f"  {'decode_s':18s} {dt:.2f}")
    if args.out_dir:
        tot = sum(os.path.getsize(os.path.join(args.out_dir, f))
                  for f in os.listdir(args.out_dir))
        print(f"  {'store_MB':18s} {tot/1e6:.1f}  ({args.out_dir})")
    else:
        print(hits.head())
