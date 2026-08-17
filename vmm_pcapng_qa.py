#!/usr/bin/env python3
"""
VMM hybrid QA script for the SPS beam test DAQ (DAQ_Control_VMM_Beam).
Reads a pcapng file and produces per-VMM ADC and channel occupancy plots.

PNG-only adaptation of P2_basket_online_analysis/vmm_hybrid_pcapng_monitoring.py:
the PyROOT self-re-exec and the ROOT histogram/TTree output were removed so the
script runs in the DAQ venv with only scapy/numpy/pandas/matplotlib. It also
writes an events.json summary per input file (--events-json) that
get_run_events.py sums for the GUI event counter.

@author: ak271430 Alexandra Kallitsopoulou (alexandra.kallitsopoulou@cea.fr)

Usage:
    python3 vmm_pcapng_qa.py <pcap_file> [--out-dir DIR] [--events-json]
                             [--format SRS|TRG] [--calibration JSON]
                             [--max-packets N] [--live]

Output PNGs are saved in --out-dir (default: qa_plots/<basename>/ next to the
input file).

hits DataFrame columns:
    fec            : FEC ID (from SRS dataId byte 7 upper nibble, 1-based)
    vmm            : VMM ID (0-31, from packet data)
    time           : frame counter (SRS header bytes 0-3)
    # NOTE: udp_timestamp (bytes 8-11) and overflow (bytes 12-15) are deprecated SRS header fields —
    # explicitly marked "will vanish soon" in vmm-sdat ParserSRS.cpp; not populated by current FEC
    # firmware (always 0 and a fixed boot-time value respectively). Excluded from output.
    ch             : channel number (0-63)
    adc            : raw ADC value (0-1023)
    adc_calibrated : adc_slope * adc + adc_offset  (float; equals adc without --calibration)
    over_threshold : over-threshold flag (bool)
    offset         : 5-bit signed offset, #BCID overflows since last SRS marker (-16 to +15)
    bcid           : bunch-crossing ID (0-4095, 12-bit, Gray-decoded)
    tdc            : TDC fine-timing value (0-255, 8-bit)
    timestamp_ns   : chip_time = (offset*4096 + bcid)*22.5 + (22.5 - tdc*60/255 - time_offset)*time_slope
                     (relative to most recent SRS marker; time_offset/slope default to 0/1 without --calibration)
    srs_timestamp  : 42-bit SRS marker FEC timestamp for this VMM (25 ns ticks; 0 before first marker)
    abs_time_ns    : absolute time = srs_timestamp * 25 + timestamp_ns (ns)
    trigger_time   : 42-bit external trigger timestamp (TRG format only; 25 ns ticks; 0 in SRS mode)
    trigger_counter: trigger event counter (TRG format only; 0 in SRS mode)
    hit_valid      : True if offset is within the valid range for the selected format
                     SRS/TRG (new SRS format): valid offsets are -1 to +15; anything outside is flagged False
                     Old SRS format (raw 0-31 unsigned) is not yet a separate mode — see --format
"""

import sys
import os
import struct
import json
import array

# Redirect per-user cache dirs before any import touches /tmp/.cache on shared machines
import getpass
_user = getpass.getuser()  # os.getlogin() fails without a controlling tty (watcher subprocess)
os.environ.setdefault('XDG_CACHE_HOME', f'/tmp/.cache-{_user}')
os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{_user}')

import time
import numpy as np
import pandas as pd
import argparse
import datetime
import matplotlib
# Live mode needs an interactive backend; batch mode uses the non-display Agg backend.
matplotlib.use('TkAgg' if '--live' in sys.argv else 'Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scapy.all import PcapReader, UDP, IP

#########################################
# AUTO-DETECT FEC SOURCE IPs
# Scans the first PROBE_PACKETS packets and
# returns all source IPs that send VM3 data.
# Pass SRC_IP_OVERRIDE to skip auto-detection.
#########################################
PROBE_PACKETS = 500
SRC_IP_OVERRIDE = None  # e.g. "192.168.1.13" to force a specific IP
CLOCK_PERIOD_NS = 22.5  # ns per BCID count (44.44 MHz clock)
TAC_SLOPE_NS    = 60.0  # ns full-scale of the TDC TAC ramp (default; tunable per VMM)
TDC_RANGE       = 255   # TDC full-scale bin count (matches vmm-sdat SRSTime::tdc_range)

# matplotlib's default histtype='bar' emits one Rectangle patch per bin, so a
# 20-VMM x 1024-bin figure builds ~20k artists and takes ~19 s. 'step' draws
# the identical bin contents as a single line in ~2.5 s.
#
# 'stepfilled' keeps the filled look and is still ~6x faster than 'bar', but
# measured end-to-end it lands at 44.3-44.8 s against the 44.4 s dumpcap
# rotation, i.e. no margin -- so 'step' is the default. Bin contents are the
# same for all three; only the rendering differs.
HIST_TYPE = "step"


def gray2bin_np(arr):
    """Convert a numpy array of Gray-coded integers to binary (matches Lua gray2bin32)."""
    arr = arr.astype(np.uint32)
    arr ^= arr >> 16
    arr ^= arr >> 8
    arr ^= arr >> 4
    arr ^= arr >> 2
    arr ^= arr >> 1
    return arr.astype(np.uint16)


def load_calibration(json_path):
    """Parse a vmm-sdat calibration JSON and return per-(vmm,ch) correction arrays.

    Supports the real vmm-sdat file format (vmmID + hybridID, arrays of 64 floats).
    Also accepts the C++ ParserSRS format (fecID + vmmID) — fecID is ignored; calibration
    is applied by vmmID only, which is correct for single-FEC setups.

    Returns four numpy arrays shaped (32, 64):
        time_offset  — subtracted from the TAC term before applying time_slope (ns)
        time_slope   — multiplies (bc_factor - tdc_corr - time_offset)
        adc_offset   — applied as:  adc_calibrated = adc_slope * adc + adc_offset
        adc_slope    — (see above)

    Timewalk coefficients (timewalk_a/b/c/d) are loaded and printed but not yet applied
    since the application formula is not part of the base ParserSRS.
    """
    import json as _json
    with open(json_path) as _f:
        data = _json.load(_f)

    time_offset = np.zeros((32, 64), dtype=np.float64)
    time_slope  = np.ones( (32, 64), dtype=np.float64)
    adc_offset  = np.zeros((32, 64), dtype=np.float64)
    adc_slope   = np.ones( (32, 64), dtype=np.float64)

    has_time, has_adc, has_tw = False, False, False
    for entry in data.get('vmm_calibration', []):
        vid = int(entry['vmmID'])
        if vid >= 32:
            continue
        def _load(key, arr, row):
            vals = entry.get(key, [])
            if vals:
                arr[row, :min(len(vals), 64)] = vals[:64]
        _load('time_offsets', time_offset, vid); has_time = has_time or bool(entry.get('time_offsets'))
        _load('time_slopes',  time_slope,  vid)
        _load('adc_offsets',  adc_offset,  vid); has_adc = has_adc or bool(entry.get('adc_offsets'))
        _load('adc_slopes',   adc_slope,   vid)
        if any(entry.get(k) for k in ('timewalk_a','timewalk_b','timewalk_c','timewalk_d')):
            has_tw = True

    loaded = []
    if has_time: loaded.append('time_offset/slope')
    if has_adc:  loaded.append('adc_offset/slope')
    if has_tw:   loaded.append('timewalk (loaded, not applied)')
    print(f"  Calibration arrays loaded: {', '.join(loaded) if loaded else 'none found'}")
    return time_offset, time_slope, adc_offset, adc_slope


def detect_fec_ips(filename, n_probe=PROBE_PACKETS):
    found = {}
    with PcapReader(filename) as r:
        for i, pkt in enumerate(r):
            if UDP in pkt and IP in pkt:
                payload = bytes(pkt[UDP].payload)
                if len(payload) > 6 and payload[4:7] == b'VM3':
                    ip = pkt[IP].src
                    found[ip] = found.get(ip, 0) + 1
            if i >= n_probe:
                break
    return found


#########################################
# PARSE ONE UDP PAYLOAD BLOCK
#########################################
_DEFAULT_MARKER = [0, 0, 0]  # [srs_ts, trigger_time, trigger_counter]

def parse_block(block: bytes, frame_counter: int, fec_id: int,
                markers: dict, data_format: str,
                fec_buf, vmm_buf, time_buf,
                ch_buf, adc_buf, ot_buf, offset_buf, bcid_buf, tdc_buf,
                srs_ts_buf, trg_time_buf, trg_ctr_buf):
    if len(block) < 22 or block[4:7] != b'VM3':
        return
    for i in range(0, len(block) - 22, 6):
        d1, d2 = struct.unpack_from('>IH', block, i + 16)
        if d2 & 0x8000:          # hit word: MSB of d2 set
            vmm_id  = (d1 >> 22) & 0x1F
            m       = markers.get((fec_id, vmm_id), _DEFAULT_MARKER)
            fec_buf.append(fec_id)
            vmm_buf.append(vmm_id)
            time_buf.append(frame_counter)
            ch_buf.append((d2 >> 8) & 0x3F)     # bits 13-8 of d2
            adc_buf.append((d1 >> 12) & 0x3FF)  # bits 21-12 of d1
            ot_buf.append((d2 >> 14) & 0x1)     # bit 14 of d2
            raw_off = (d1 >> 27) & 0x1F          # bits 31-27 of d1 (5-bit signed)
            offset_buf.append(raw_off if raw_off < 16 else raw_off - 32)
            bcid_buf.append(d1 & 0xFFF)          # bits 11-0 of d1 — raw Gray code; decoded below
            tdc_buf.append(d2 & 0xFF)            # bits 7-0 of d2
            srs_ts_buf.append(m[0])
            trg_time_buf.append(m[1])
            trg_ctr_buf.append(m[2])
        else:                    # marker word: MSB of d2 clear
            vmmid_marker = (d2 >> 10) & 0x1F
            # `vmmid_marker < 16` comes from vmm-sdat's ParserSRS, where a FEC
            # hosts at most 16 VMMs and ids >= 16 encode TRG trigger words.
            # THIS FEC HOSTS 20 (hybrids 0-9), so in SRS mode that test threw
            # away every marker for VMMs 16-19: they had srs_timestamp = 0 for
            # every hit, hence a meaningless abs_time_ns and zero trigger
            # coincidences (4 of P2_OUT's 6 VMMs, 37% of detector hits).
            # In SRS mode there are no TRG markers, so take the id at face
            # value. TRG mode keeps the original behaviour untouched.
            if vmmid_marker < 16 or data_format == 'SRS':
                # Normal VMM marker — 42-bit SRS FEC timestamp (25 ns ticks)
                srs_ts = (d1 << 10) | (d2 & 0x3FF)
                key = (fec_id, vmmid_marker)
                if data_format == 'SRS':
                    if key not in markers:
                        markers[key] = [0, 0, 0]
                    markers[key][0] = srs_ts
                else:
                    # TRG mode: VMM markers carry a relative BCID timestamp (< 4096)
                    if srs_ts < 4096:
                        if key not in markers:
                            markers[key] = [0, 0, 0]
                        markers[key][0] = srs_ts
            elif data_format == 'TRG':
                # TRG marker (vmmid >= 16): carries external trigger info
                # Maps to the same VMM slot as vmmid % 16
                key = (fec_id, vmmid_marker % 16)
                if key not in markers:
                    markers[key] = [0, 0, 0]
                trigger_flag = (d1 >> 28) & 0x0F
                if trigger_flag == 0xF:
                    # Trigger counter word (16-bit: 6 high bits from d1, 10 low from d2)
                    markers[key][2] = (d1 & 0x3F) * 1024 + (d2 & 0x3FF)
                else:
                    # Trigger timestamp: upper 32 bits only (lower 10 unused — matches C++)
                    markers[key][1] = d1 << 10

#########################################
# LIVE-MONITORING HELPERS
#########################################
def follow_pcap(filename):
    """Yield packets from a growing pcapng file indefinitely (tail -f style).

    PcapReader requires the file's global header at position 0, so we reopen
    from the start on every poll and skip packets we have already yielded.
    """
    pkts_yielded = 0
    while True:
        try:
            with PcapReader(filename) as reader:
                new_count = 0
                for new_count, pkt in enumerate(reader, 1):
                    if new_count > pkts_yielded:
                        yield pkt
                pkts_yielded = new_count  # only advance after a clean full read
        except FileNotFoundError:
            pass
        except Exception:
            pass  # Scapy_Exception when file has no new data yet; retry next poll
        time.sleep(0.1)


def run_live(pcap_file, src_ips, update_interval=2.0):
    """
    Continuously follow a growing pcapng file and display live hit-rate-per-channel
    plots for every active VMM. Press Ctrl-C to stop.

    Uses a poll-then-draw loop: read all currently available packets, update the
    plot, sleep for update_interval, repeat. This ensures the plot appears even
    when all hits arrive in a single burst (e.g. replaying a static file).
    """
    vmm_counts = {}   # vmm_id -> np.ndarray(64, int64) — cumulative channel hits
    vmm_hits   = {}   # vmm_id -> total hits for per-VMM rate label
    total_hits = 0
    t_start    = time.time()
    pkts_seen  = 0    # total packets already processed from previous polls

    fig        = None
    axes_map   = {}
    n_vmm_last = 0

    plt.ion()
    print(f"Live monitoring: {pcap_file}")
    print(f"Update interval: {update_interval}s   |   Press Ctrl-C to stop\n")

    try:
        while True:
            # --- read every packet in the file, skip ones already processed ---
            try:
                with PcapReader(pcap_file) as reader:
                    new_total = 0
                    for new_total, pkt in enumerate(reader, 1):
                        if new_total <= pkts_seen:
                            continue
                        if UDP not in pkt or IP not in pkt or pkt[IP].src not in src_ips:
                            continue
                        payload = bytes(pkt[UDP].payload)
                        if len(payload) < 22 or payload[4:7] != b'VM3':
                            continue
                        for j in range(0, len(payload) - 22, 6):
                            d1, d2 = struct.unpack_from('>IH', payload, j + 16)
                            if d2 & 0x8000:
                                vmm = (d1 >> 22) & 0x1F
                                ch  = (d2 >> 8) & 0x3F
                                if vmm not in vmm_counts:
                                    vmm_counts[vmm] = np.zeros(64, dtype=np.int64)
                                    vmm_hits[vmm]   = 0
                                vmm_counts[vmm][ch] += 1
                                vmm_hits[vmm]       += 1
                                total_hits          += 1
                    pkts_seen = new_total
            except FileNotFoundError:
                pass
            except Exception:
                pass

            # --- update plot whenever we have data ---
            if total_hits > 0:
                elapsed  = time.time() - t_start
                vmm_list = sorted(vmm_counts.keys())
                n        = len(vmm_list)

                # Rebuild figure layout if a new VMM appeared
                if n != n_vmm_last:
                    if fig is not None:
                        plt.close(fig)
                    ncols = min(4, n)
                    nrows = (n + ncols - 1) // ncols
                    fig, ax_arr = plt.subplots(nrows, ncols,
                                               figsize=(5 * ncols, 4 * nrows),
                                               squeeze=False)
                    axes_map = {v: ax_arr[i // ncols][i % ncols]
                                for i, v in enumerate(vmm_list)}
                    for i in range(n, nrows * ncols):
                        ax_arr[i // ncols][i % ncols].set_visible(False)
                    n_vmm_last = n

                rate = total_hits / elapsed if elapsed > 0 else 0
                fig.suptitle(
                    f"VMM Online Monitoring  |  {os.path.basename(pcap_file)}\n"
                    f"Total: {total_hits:,} hits  |  Rate: {rate:.1f} hits/s  |  "
                    f"Elapsed: {elapsed:.0f}s",
                    fontsize=11,
                )
                for v in vmm_list:
                    ax     = axes_map[v]
                    counts = vmm_counts[v]
                    ax.cla()
                    ax.bar(range(64), counts, color='steelblue', alpha=0.8, width=1.0)
                    v_rate = vmm_hits[v] / elapsed if elapsed > 0 else 0
                    ax.set_title(f"VMM {v}  |  {vmm_hits[v]:,} hits  |  {v_rate:.1f} hits/s",
                                 fontsize=9)
                    ax.set_xlabel("Channel")
                    ax.set_ylabel("Hits")
                    ax.set_xlim(-0.5, 63.5)
                fig.tight_layout()
                plt.pause(0.05)

            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        if fig is not None:
            plt.ioff()
            plt.show()


#########################################
# MAIN
#########################################

_ap = argparse.ArgumentParser(
    description="VMM3 hybrid diagnostic: parse a pcapng capture and produce QA plots (PNG only)."
)
_ap.add_argument("pcap_file", help="Input .pcapng file")
_ap.add_argument("--out-dir", metavar="DIR", default=None,
                 help="Directory for output PNGs and events.json "
                      "(default: qa_plots/<basename>/ next to the input file)")
_ap.add_argument("--events-json", action="store_true",
                 help="Write an events.json summary (n_packets, n_hits, hits_per_vmm, ...) "
                      "into the output directory — summed by get_run_events.py for the GUI")
_ap.add_argument("--max-packets", type=int, default=None, metavar="N",
                 help="Stop reading after N packets (guard against pathological files)")
_ap.add_argument("--calibration", metavar="JSON",
                 help="vmm-sdat calibration JSON file with per-channel time and ADC corrections "
                      "(time_offsets, time_slopes, adc_offsets, adc_slopes arrays of 64 floats per VMM)")
_ap.add_argument("--format", choices=["SRS", "TRG"], default="SRS",
                 help="SRS data format: SRS (continuous readout, default) or "
                      "TRG (external trigger — parses trigger_counter and trigger_time from marker words)")
_ap.add_argument("--no-efficiency", action="store_true",
                 help="Skip the trigger-referenced efficiency stage")
_ap.add_argument("--eff-window", type=float, default=1000.0, metavar="NS",
                 help="Delta-t histogram half-width for the efficiency stage "
                      "(default 1000 ns)")
_ap.add_argument("--live", action="store_true",
                 help="Live monitoring mode: follow a growing pcapng and display "
                      "hit-rate-per-channel for each active VMM (no ROOT output)")
_ap.add_argument("--update-interval", type=float, default=2.0, metavar="SECONDS",
                 help="Plot refresh interval in seconds for --live mode (default: 2.0)")
_ap.add_argument("--legacy-parser", action="store_true",
                 help="Use the original in-file scapy/parse_block loop instead of "
                      "vmm_decode. Slower; kept as a fallback and for A/B checks.")
_args = _ap.parse_args()

pcap_file      = _args.pcap_file
data_format    = _args.format

if _args.calibration:
    print(f"Loading calibration: {_args.calibration}")
    _cal_to, _cal_ts, _cal_ao, _cal_as = load_calibration(_args.calibration)
    _has_calibration = True
else:
    _cal_to = np.zeros((32, 64), dtype=np.float64)
    _cal_ts = np.ones( (32, 64), dtype=np.float64)
    _cal_ao = np.zeros((32, 64), dtype=np.float64)
    _cal_as = np.ones( (32, 64), dtype=np.float64)
    _has_calibration = False

# The positional argument may be a capture OR a column store already decoded
# by vmm_processor_watcher. In store mode nothing re-reads the pcapng: the
# decode happened once, in the processor, and this script only renders. That is
# the whole point of the split -- qa_watcher never touches raw captures.
_from_store = (os.path.isdir(pcap_file)
               and os.path.isfile(os.path.join(pcap_file, 'meta.json')))
_source_capture = pcap_file    # overwritten from the store's meta below

src_ips = None
if _from_store:
    print(f"Reading decoded store: {pcap_file}")
elif not os.path.isfile(pcap_file):
    print(f"File not found: {pcap_file}")
    sys.exit(1)
elif SRC_IP_OVERRIDE:
    src_ips = {SRC_IP_OVERRIDE}
    print(f"Using forced IP: {SRC_IP_OVERRIDE}")
else:
    detected = detect_fec_ips(pcap_file)
    if not detected:
        print("ERROR: no VM3 packets found in the first "
              f"{PROBE_PACKETS} packets. Check the file or set SRC_IP_OVERRIDE.")
        sys.exit(1)
    src_ips = set(detected.keys())
    print(f"Auto-detected FEC IP(s): {', '.join(sorted(src_ips))}")

# ---- Live monitoring mode ------------------------------------------------
if _args.live and not _from_store:
    run_live(pcap_file, src_ips, update_interval=_args.update_interval)
    print("\nProceeding with batch QA analysis on the same file...\n")
    # Switch matplotlib back to the non-interactive batch backend
    plt.switch_backend('Agg')
# --------------------------------------------------------------------------

# Per-VMM marker state: (fec_id, vmm_id) → [srs_ts, trigger_time, trigger_counter]
# ---- Parse ---------------------------------------------------------------
# Fast path: vmm_decode is a vectorised decoder (no scapy, no per-word Python
# loop) validated to reproduce this script's hits DataFrame bit for bit --
# every column, every dtype -- across SRS and TRG on the beam campaign data.
# It is ~5-12x faster and uses less memory.
#
# The legacy loop below is kept deliberately: if vmm_decode is missing or
# raises for any reason, QA degrades to its previous speed instead of failing.
# --legacy-parser forces it, which is also how to A/B the two.
hits = None
if _from_store:
    import vmm_decode as _vd
    _t_parse = time.time()
    hits, _smeta = _vd.load_frame(
        pcap_file,
        calibration=((_cal_to, _cal_ts, _cal_ao, _cal_as) if _has_calibration else None))
    pkt_count = _smeta.get('n_packets', 0)
    vm3_count = _smeta.get('n_vm3_packets', 0)
    # the store records which FEC(s) it was decoded from; events.json needs it
    src_ips = set(_smeta.get('fec_ips') or [])
    _source_capture = _smeta.get('source') or pcap_file
    print(f"Loaded {len(hits):,} hits from store  ({time.time() - _t_parse:.1f}s)")

if hits is None and not _args.legacy_parser:
    try:
        import vmm_decode as _vd
        _t_parse = time.time()
        print(f"Reading: {pcap_file}  (vmm_decode)")
        hits, _dmeta = _vd.decode(
            pcap_file,
            data_format=data_format,
            src_ips=src_ips,
            max_packets=_args.max_packets,
            calibration=((_cal_to, _cal_ts, _cal_ao, _cal_as)
                         if _has_calibration else None),
        )
        pkt_count = _dmeta["n_packets"]
        vm3_count = _dmeta["n_vm3_packets"]
        print(f"Done: {pkt_count} packets | {vm3_count} VM3 packets | "
              f"{len(hits):,} total hits  ({time.time() - _t_parse:.1f}s)")
    except Exception as _e:
        print(f"WARNING: vmm_decode failed ({type(_e).__name__}: {_e}) -- "
              f"falling back to the legacy parser.")
        hits = None

if hits is None:

    markers = {}

    # Memory-efficient typed arrays (no Python object overhead)
    fec_buf      = array.array('B')   # uint8
    vmm_buf      = array.array('B')   # uint8
    time_buf     = array.array('I')   # uint32 — frame counter (SRS header bytes 0-3)
    ch_buf       = array.array('B')   # uint8
    adc_buf      = array.array('H')   # uint16
    ot_buf       = array.array('B')   # uint8 (0/1)
    offset_buf   = array.array('b')   # int8  (-16 to +15, 5-bit signed)
    bcid_buf     = array.array('H')   # uint16 (0-4095)
    tdc_buf      = array.array('B')   # uint8  (0-255)
    srs_ts_buf   = array.array('Q')   # uint64 — 42-bit SRS marker FEC timestamp (25 ns ticks)
    trg_time_buf = array.array('Q')   # uint64 — 42-bit external trigger timestamp (TRG mode)
    trg_ctr_buf  = array.array('H')   # uint16 — trigger counter (TRG mode)

    print(f"Reading: {pcap_file}")
    pkt_count = 0
    vm3_count = 0

    with PcapReader(pcap_file) as reader:
        for pkt in reader:
            if UDP in pkt and IP in pkt and pkt[IP].src in src_ips:
                payload = bytes(pkt[UDP].payload)
                fc     = struct.unpack_from('>I', payload)[0]
                fec_id = (struct.unpack_from('>I', payload, 4)[0] >> 4) & 0x0F
                n_before = len(fec_buf)
                parse_block(payload, fc, fec_id,
                            markers, data_format,
                            fec_buf, vmm_buf, time_buf,
                            ch_buf, adc_buf, ot_buf, offset_buf, bcid_buf, tdc_buf,
                            srs_ts_buf, trg_time_buf, trg_ctr_buf)
                if len(fec_buf) > n_before:
                    vm3_count += 1
            pkt_count += 1
            if pkt_count % 10000 == 0:
                print(f"  {pkt_count} packets | {len(fec_buf):,} hits so far...")
            if _args.max_packets and pkt_count >= _args.max_packets:
                print(f"  --max-packets={_args.max_packets} reached, stopping read.")
                break

    print(f"\nDone: {pkt_count} packets | {vm3_count} VM3 packets | {len(fec_buf):,} total hits")

    # Build DataFrame with compact dtypes
    _bcid_raw = np.frombuffer(bcid_buf,    dtype=np.uint16).copy()
    _offset   = np.frombuffer(offset_buf,  dtype=np.int8).copy()
    _tdc      = np.frombuffer(tdc_buf,     dtype=np.uint8).copy()
    _adc_raw  = np.frombuffer(adc_buf,     dtype=np.uint16).copy()
    _vmm      = np.frombuffer(vmm_buf,     dtype=np.uint8).copy()
    _ch       = np.frombuffer(ch_buf,      dtype=np.uint8).copy()
    _bcid     = gray2bin_np(_bcid_raw)     # Gray → binary, matching vmm-sdat gray2bin32

    # Per-hit calibration lookup (vectorised over vmm/ch index pairs)
    _t_off = _cal_to[_vmm.astype(np.intp), _ch.astype(np.intp)]
    _t_slp = _cal_ts[_vmm.astype(np.intp), _ch.astype(np.intp)]
    _a_off = _cal_ao[_vmm.astype(np.intp), _ch.astype(np.intp)]
    _a_slp = _cal_as[_vmm.astype(np.intp), _ch.astype(np.intp)]

    # chip_time formula matching vmm-sdat SRSTime::chip_time_ns (calibrated):
    #   t_coarse = (offset*4096 + bcid) * bc_factor
    #   t_fine   = (bc_factor - tdc * (tac/255) - time_offset) * time_slope
    #   timestamp_ns = t_coarse + t_fine
    # Without calibration (time_offset=0, time_slope=1):
    #   = (offset*4096 + bcid + 1) * 22.5 - tdc * 60/255
    _t_coarse = (_offset.astype(np.float64) * 4096 + _bcid.astype(np.float64)) * CLOCK_PERIOD_NS
    _t_fine   = (CLOCK_PERIOD_NS - _tdc.astype(np.float64) * TAC_SLOPE_NS / TDC_RANGE - _t_off) * _t_slp
    _timestamp_ns = _t_coarse + _t_fine

    _srs_ts   = np.frombuffer(srs_ts_buf,   dtype=np.uint64).copy()
    _trg_time = np.frombuffer(trg_time_buf, dtype=np.uint64).copy()
    _trg_ctr  = np.frombuffer(trg_ctr_buf,  dtype=np.uint16).copy()
    _abs_time_ns = _srs_ts.astype(np.float64) * 25.0 + _timestamp_ns
    _adc_cal  = (_a_slp * _adc_raw.astype(np.float64) + _a_off).astype(np.float32)

    # Offset validity flag — new SRS format (and TRG) valid range is -1 to +15.
    # Anything with offset < -1 (raw 16-30) is outside the firmware specification
    # and indicates abnormal FEC behaviour. All values are kept in the DataFrame;
    # hit_valid=False lets the user apply or inspect the cut downstream.
    _hit_valid = (_offset >= np.int8(-1))

    hits = pd.DataFrame({
        'fec':             np.frombuffer(fec_buf,      dtype=np.uint8).copy(),
        'vmm':             _vmm,
        'time':            np.frombuffer(time_buf,     dtype=np.uint32).copy(),
        'ch':              _ch,
        'adc':             _adc_raw,
        'adc_calibrated':  _adc_cal,
        'over_threshold':  np.frombuffer(ot_buf,       dtype=np.uint8).astype(bool).copy(),
        'offset':          _offset,
        'bcid':            _bcid,
        'tdc':             _tdc,
        'timestamp_ns':    _timestamp_ns,
        'srs_timestamp':   _srs_ts,
        'abs_time_ns':     _abs_time_ns,
        'trigger_time':    _trg_time,
        'trigger_counter': _trg_ctr,
        'hit_valid':       _hit_valid,
    })
    del (fec_buf, vmm_buf, time_buf, ch_buf, adc_buf, ot_buf,
         offset_buf, bcid_buf, tdc_buf, srs_ts_buf, trg_time_buf, trg_ctr_buf)

mem_mb = hits.memory_usage(deep=True).sum() / 1e6
print(f"DataFrame memory: {mem_mb:.1f} MB")

# Report anomalous hits
_n_invalid = int((~hits['hit_valid']).sum())
if _n_invalid > 0:
    print(f"\nWARNING: {_n_invalid} hits ({_n_invalid/len(hits):.3%}) have offset outside the valid "
          f"range [-1, +15] for {data_format} format (vmm-sdat README: valid offsets -1 to 15).")
    print("  Breakdown by offset value (raw bits → signed):")
    for off_val, cnt in hits.loc[~hits['hit_valid'], 'offset'].value_counts().sort_index().items():
        raw = int(off_val) + 32
        print(f"    offset={int(off_val):4d}  (raw={raw:2d} = {raw:05b}b):  {cnt} hits")
    print("  These hits are flagged hit_valid=False. They are retained in the DataFrame")
    print("  but you should exclude them with:  hits = hits[hits.hit_valid]")
else:
    print("Offset validity: all hits within valid range [-1, +15].")

print(f"\nVMM IDs found: {sorted(hits.vmm.unique())}")
for v in sorted(hits.vmm.unique()):
    n      = int((hits.vmm == v).sum())
    n_bad  = int(((hits.vmm == v) & ~hits['hit_valid']).sum())
    flag   = f"  *** {n_bad} anomalous ***" if n_bad else ""
    print(f"  VMM {v:2d}: {n:,} hits{flag}")

#########################################
# HISTOGRAM PARAMETERS
#########################################
ADC_BINS      = 100; ADC_MIN      = 0;   ADC_MAX      = 1024
CH_BINS       = 64;  CH_MIN       = 0;   CH_MAX       = 64
TIME_BINS     = 200
BCID_BINS     = 100; BCID_MIN     = 0;   BCID_MAX     = 4096
TDC_BINS      = 64;  TDC_MIN      = 0;   TDC_MAX      = 256
OFFSET_BINS   = 32;  OFFSET_MIN   = -16; OFFSET_MAX   = 16

#########################################
# PLOTS
#########################################
vmm_ids = sorted(hits.vmm.unique())
n_vmm   = len(vmm_ids)
ncols   = min(4, n_vmm)
nrows   = (n_vmm + ncols - 1) // ncols

# --- ADC histograms ---
fig_adc, axes_adc = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_adc.suptitle("ADC distributions per VMM", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax   = axes_adc[idx // ncols][idx % ncols]
    data = hits.loc[hits.vmm == v, 'adc']
    ax.hist(data, bins=ADC_BINS, range=(ADC_MIN, ADC_MAX), color='steelblue', alpha=0.8, histtype=HIST_TYPE)
    ax.set_title(f"VMM {v}  ({len(data):,} hits)")
    ax.set_xlabel("ADC")
    ax.set_ylabel("Counts")
for idx in range(n_vmm, nrows * ncols):
    axes_adc[idx // ncols][idx % ncols].set_visible(False)
fig_adc.tight_layout()

# --- ADC distribution split by over-threshold flag ---
fig_adc_ot, axes_adc_ot = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_adc_ot.suptitle("ADC distributions per VMM — split by over-threshold flag", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax      = axes_adc_ot[idx // ncols][idx % ncols]
    vmm_hit = hits.loc[hits.vmm == v]
    adc_off = vmm_hit.loc[~vmm_hit.over_threshold, 'adc']
    adc_on  = vmm_hit.loc[ vmm_hit.over_threshold, 'adc']
    ax.hist(adc_off, bins=ADC_BINS, range=(ADC_MIN, ADC_MAX), color='steelblue', alpha=0.6, label=f'Not OT ({len(adc_off):,})', histtype=HIST_TYPE)
    ax.hist(adc_on,  bins=ADC_BINS, range=(ADC_MIN, ADC_MAX), color='tomato',    alpha=0.6, label=f'OT ({len(adc_on):,})', histtype=HIST_TYPE)
    ax.set_title(f"VMM {v}")
    ax.set_xlabel("ADC")
    ax.set_ylabel("Counts")
    ax.legend(fontsize=7)
for idx in range(n_vmm, nrows * ncols):
    axes_adc_ot[idx // ncols][idx % ncols].set_visible(False)
fig_adc_ot.tight_layout()

# --- Over-threshold flag distribution ---
fig_ot, axes_ot = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_ot.suptitle("Over-threshold flag distribution per VMM", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax    = axes_ot[idx // ncols][idx % ncols]
    grp   = hits.loc[hits.vmm == v, 'over_threshold']
    n_off = int((~grp).sum())
    n_on  = int(grp.sum())
    ax.bar([0, 1], [n_off, n_on], color=['steelblue', 'tomato'], alpha=0.8, width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not OT', 'OT'])
    ax.set_title(f"VMM {v}  (OT frac: {n_on / max(n_off + n_on, 1):.2%})")
    ax.set_ylabel("Counts")
for idx in range(n_vmm, nrows * ncols):
    axes_ot[idx // ncols][idx % ncols].set_visible(False)
fig_ot.tight_layout()

# --- Channel occupancy ---
fig_ch, axes_ch = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_ch.suptitle("Channel occupancy per VMM", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax   = axes_ch[idx // ncols][idx % ncols]
    data = hits.loc[hits.vmm == v, 'ch']
    ax.hist(data, bins=CH_BINS, range=(CH_MIN, CH_MAX), color='tomato', alpha=0.8, histtype=HIST_TYPE)
    ax.set_title(f"VMM {v}  ({len(data):,} hits)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")
    ax.set_xlim(0, 63)
for idx in range(n_vmm, nrows * ncols):
    axes_ch[idx // ncols][idx % ncols].set_visible(False)
fig_ch.tight_layout()

# --- ADC vs channel 2D histogram ---
fig_adc2d, axes_adc2d = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_adc2d.suptitle("ADC vs channel (2-D, log colour scale) per VMM", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax    = axes_adc2d[idx // ncols][idx % ncols]
    vdata = hits.loc[hits.vmm == v]
    _, _, _, img = ax.hist2d(
        vdata['ch'], vdata['adc'],
        bins=[CH_BINS, ADC_BINS], range=[[CH_MIN, CH_MAX], [ADC_MIN, ADC_MAX]],
        cmap='viridis', norm=LogNorm(vmin=1),
    )
    plt.colorbar(img, ax=ax, label='Hits')
    ax.set_title(f"VMM {v}  ({len(vdata):,} hits)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("ADC")
for idx in range(n_vmm, nrows * ncols):
    axes_adc2d[idx // ncols][idx % ncols].set_visible(False)
fig_adc2d.tight_layout()

# --- Total hits per VMM ---
fig_sum, ax_sum = plt.subplots(figsize=(max(6, n_vmm), 4))
ax_sum.bar([str(v) for v in vmm_ids],
           [int((hits.vmm == v).sum()) for v in vmm_ids],
           color='mediumpurple', alpha=0.8)
ax_sum.set_title("Total hits per VMM")
ax_sum.set_xlabel("VMM ID")
ax_sum.set_ylabel("Hits")
fig_sum.tight_layout()

# --- Save all PNGs in --out-dir (default: qa_plots/<base>/ next to the input) ---
base = os.path.splitext(os.path.basename(pcap_file))[0]
if _args.out_dir:
    out_dir = os.path.abspath(_args.out_dir)
else:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(pcap_file)), "qa_plots", base)
os.makedirs(out_dir, exist_ok=True)

saved = []

def _save(fig, fname):
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    saved.append(fname)

_save(fig_adc,    f"{base}_adc.png")
_save(fig_adc2d,  f"{base}_adc_vs_ch.png")
_save(fig_adc_ot, f"{base}_adc_ot.png")
_save(fig_ot,     f"{base}_ot.png")
_save(fig_ch,     f"{base}_chno.png")
_save(fig_sum,    f"{base}_hits_per_vmm.png")

# --- BCID distribution ---
fig_bcid, axes_bcid = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_bcid.suptitle("BCID distribution per VMM", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax   = axes_bcid[idx // ncols][idx % ncols]
    data = hits.loc[hits.vmm == v, 'bcid']
    ax.hist(data, bins=BCID_BINS, range=(BCID_MIN, BCID_MAX), color='teal', alpha=0.8, histtype=HIST_TYPE)
    ax.set_title(f"VMM {v}  ({len(data):,} hits)")
    ax.set_xlabel("BCID")
    ax.set_ylabel("Counts")
for idx in range(n_vmm, nrows * ncols):
    axes_bcid[idx // ncols][idx % ncols].set_visible(False)
fig_bcid.tight_layout()
_save(fig_bcid, f"{base}_bcid.png")

# --- TDC distribution ---
fig_tdc, axes_tdc = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_tdc.suptitle("TDC distribution per VMM", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax   = axes_tdc[idx // ncols][idx % ncols]
    data = hits.loc[hits.vmm == v, 'tdc']
    ax.hist(data, bins=TDC_BINS, range=(TDC_MIN, TDC_MAX), color='darkcyan', alpha=0.8, histtype=HIST_TYPE)
    ax.set_title(f"VMM {v}  ({len(data):,} hits)")
    ax.set_xlabel("TDC")
    ax.set_ylabel("Counts")
for idx in range(n_vmm, nrows * ncols):
    axes_tdc[idx // ncols][idx % ncols].set_visible(False)
fig_tdc.tight_layout()
_save(fig_tdc, f"{base}_tdc.png")

# --- Offset distribution (valid hits in brown, anomalous in red) ---
fig_offset, axes_offset = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_offset.suptitle("Offset distribution per VMM (5-bit signed)\n"
                    "brown = valid [-1,+15]   red = anomalous (outside SRS spec)", fontsize=13)
for idx, v in enumerate(vmm_ids):
    ax      = axes_offset[idx // ncols][idx % ncols]
    vmm_sel = hits.vmm == v
    valid   = hits.loc[vmm_sel &  hits['hit_valid'], 'offset']
    bad     = hits.loc[vmm_sel & ~hits['hit_valid'], 'offset']
    ax.hist(valid, bins=OFFSET_BINS, range=(OFFSET_MIN, OFFSET_MAX),
            color='saddlebrown', alpha=0.8, label=f'valid ({len(valid):,})', histtype=HIST_TYPE)
    if len(bad):
        ax.hist(bad, bins=OFFSET_BINS, range=(OFFSET_MIN, OFFSET_MAX),
                color='red', alpha=0.9, label=f'anomalous ({len(bad):,})', histtype=HIST_TYPE)
        ax.legend(fontsize=7)
    ax.set_title(f"VMM {v}  ({len(valid)+len(bad):,} hits)")
    ax.set_xlabel("Offset (signed)")
    ax.set_ylabel("Counts")
for idx in range(n_vmm, nrows * ncols):
    axes_offset[idx // ncols][idx % ncols].set_visible(False)
fig_offset.tight_layout()
_save(fig_offset, f"{base}_offset.png")

# --- Full timestamp distribution ---
TS_BINS = 200
ts_min_ns = float(hits['timestamp_ns'].min())
ts_max_ns = float(hits['timestamp_ns'].max())
fig_ts, axes_ts = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_ts.suptitle("Hit timestamp distribution per VMM (ns)", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax   = axes_ts[idx // ncols][idx % ncols]
    data = hits.loc[hits.vmm == v, 'timestamp_ns']
    ax.hist(data, bins=TS_BINS, range=(ts_min_ns, ts_max_ns), color='darkviolet', alpha=0.8, histtype=HIST_TYPE)
    ax.set_title(f"VMM {v}  ({len(data):,} hits)")
    ax.set_xlabel("Timestamp (ns)")
    ax.set_ylabel("Counts")
for idx in range(n_vmm, nrows * ncols):
    axes_ts[idx // ncols][idx % ncols].set_visible(False)
fig_ts.tight_layout()
_save(fig_ts, f"{base}_timestamp_ns.png")

# --- Absolute time distribution ---
abs_min_ns = float(hits['abs_time_ns'].min())
abs_max_ns = float(hits['abs_time_ns'].max())
fig_abs, axes_abs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_abs.suptitle("Absolute hit time per VMM (SRS marker + chip_time, ns)", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax   = axes_abs[idx // ncols][idx % ncols]
    data = hits.loc[hits.vmm == v, 'abs_time_ns']
    ax.hist(data, bins=TS_BINS, range=(abs_min_ns, abs_max_ns), color='mediumseagreen', alpha=0.8, histtype=HIST_TYPE)
    ax.set_title(f"VMM {v}  ({len(data):,} hits)")
    ax.set_xlabel("Absolute time (ns)")
    ax.set_ylabel("Counts")
for idx in range(n_vmm, nrows * ncols):
    axes_abs[idx // ncols][idx % ncols].set_visible(False)
fig_abs.tight_layout()
_save(fig_abs, f"{base}_abs_time_ns.png")

# --- SRS marker timestamp distribution ---
srs_min = int(hits['srs_timestamp'].min())
srs_max = int(hits['srs_timestamp'].max())
if srs_max > srs_min:
    fig_srs, axes_srs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig_srs.suptitle("SRS marker timestamp per VMM (42-bit, 25 ns ticks)", fontsize=14)
    for idx, v in enumerate(vmm_ids):
        ax   = axes_srs[idx // ncols][idx % ncols]
        data = hits.loc[hits.vmm == v, 'srs_timestamp']
        ax.hist(data.astype(np.float64), bins=200, range=(srs_min, srs_max),
                color='cadetblue', alpha=0.8, histtype=HIST_TYPE)
        ax.set_title(f"VMM {v}  ({len(data):,} hits)")
        ax.set_xlabel("SRS timestamp (25 ns ticks)")
        ax.set_ylabel("Counts")
    for idx in range(n_vmm, nrows * ncols):
        axes_srs[idx // ncols][idx % ncols].set_visible(False)
    fig_srs.tight_layout()
    _save(fig_srs, f"{base}_srs_timestamp.png")

# --- Calibrated ADC distribution (only if a calibration file was given) ---
if _has_calibration:
    adc_cal_min = float(hits['adc_calibrated'].min())
    adc_cal_max = float(hits['adc_calibrated'].max())
    fig_adccal, axes_adccal = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig_adccal.suptitle("Calibrated ADC per VMM", fontsize=14)
    for idx, v in enumerate(vmm_ids):
        ax   = axes_adccal[idx // ncols][idx % ncols]
        data = hits.loc[hits.vmm == v, 'adc_calibrated']
        ax.hist(data, bins=ADC_BINS, range=(adc_cal_min, adc_cal_max), color='darkorange', alpha=0.8, histtype=HIST_TYPE)
        ax.set_title(f"VMM {v}  ({len(data):,} hits)")
        ax.set_xlabel("Calibrated ADC")
        ax.set_ylabel("Counts")
    for idx in range(n_vmm, nrows * ncols):
        axes_adccal[idx // ncols][idx % ncols].set_visible(False)
    fig_adccal.tight_layout()
    _save(fig_adccal, f"{base}_adc_calibrated.png")

# --- TRG trigger counter and trigger time (TRG format only) ---
if data_format == 'TRG' and hits['trigger_counter'].max() > 0:
    fig_tc, axes_tc = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig_tc.suptitle("Trigger counter per VMM (TRG mode)", fontsize=14)
    tc_min = int(hits['trigger_counter'].min())
    tc_max = int(hits['trigger_counter'].max())
    for idx, v in enumerate(vmm_ids):
        ax   = axes_tc[idx // ncols][idx % ncols]
        data = hits.loc[hits.vmm == v, 'trigger_counter']
        ax.hist(data, bins=min(200, tc_max - tc_min + 1), range=(tc_min, tc_max + 1),
                color='firebrick', alpha=0.8, histtype=HIST_TYPE)
        ax.set_title(f"VMM {v}  ({len(data):,} hits)")
        ax.set_xlabel("Trigger counter")
        ax.set_ylabel("Counts")
    for idx in range(n_vmm, nrows * ncols):
        axes_tc[idx // ncols][idx % ncols].set_visible(False)
    fig_tc.tight_layout()
    _save(fig_tc, f"{base}_trigger_counter.png")

    tt_min = int(hits['trigger_time'].min())
    tt_max = int(hits['trigger_time'].max())
    if tt_max > tt_min:
        fig_tt, axes_tt = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig_tt.suptitle("Trigger time per VMM (TRG mode, 25 ns ticks)", fontsize=14)
        for idx, v in enumerate(vmm_ids):
            ax   = axes_tt[idx // ncols][idx % ncols]
            data = hits.loc[hits.vmm == v, 'trigger_time']
            ax.hist(data.astype(np.float64), bins=200, range=(tt_min, tt_max),
                    color='tomato', alpha=0.8, histtype=HIST_TYPE)
            ax.set_title(f"VMM {v}  ({len(data):,} hits)")
            ax.set_xlabel("Trigger time (25 ns ticks)")
            ax.set_ylabel("Counts")
        for idx in range(n_vmm, nrows * ncols):
            axes_tt[idx // ncols][idx % ncols].set_visible(False)
        fig_tt.tight_layout()
        _save(fig_tt, f"{base}_trigger_time.png")

# --- Time distribution: one PNG per VMM (built and saved immediately to avoid accumulation) ---
t_min, t_max = int(hits.time.min()), int(hits.time.max())
for v in vmm_ids:
    fig_t, ax_t = plt.subplots(figsize=(12, 5))
    data = hits.loc[hits.vmm == v, 'time']
    ax_t.hist(data, bins=TIME_BINS, range=(t_min, t_max), color='darkorange', alpha=0.8, histtype=HIST_TYPE)
    ax_t.set_title(f"Hit rate over time — VMM {v}  ({len(data):,} hits)")
    ax_t.set_xlabel("Frame counter (proxy for time)")
    ax_t.set_ylabel("Hits per bin")
    fig_t.tight_layout()
    _save(fig_t, f"{base}_time_vmm{v}.png")

print(f"\nSaved {len(saved)} files in: {out_dir}")
for name in saved:
    print(f"  {name}")

#########################################
# TRIGGER-REFERENCED EFFICIENCY (vmm_efficiency.py)
#########################################
# Runs on the `hits` DataFrame already built above -- the pcapng is NOT parsed
# a second time. Products land in the same out_dir, so the flask Online QA
# gallery and Analysis browser pick them up with no GUI change.
#
# Wrapped in a broad try/except on purpose: QA is the beam-time monitoring tool
# and must keep producing its plots even if the efficiency stage fails.
_eff_summary = None
_prof = None
if not _args.no_efficiency:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import vmm_efficiency as _ve
        print('\nTrigger-referenced efficiency...')
        _eff_summary, _eff_res = _ve.analyse(hits, window_ns=_args.eff_window)
        if _eff_summary['stations']:
            _ve.plot_dt(_eff_res, base, out_dir, _args.eff_window)
            _ve.plot_efficiency(_eff_res, base, out_dir)
            for _n, _s in _eff_summary['stations'].items():
                print(f"  {_n:8s} mu={_s['mu_ns']:+7.1f} ns  "
                      f"sigma={_s['sigma_ns']:6.1f} ns  "
                      f"eff={_s['efficiency']:.4f}")
            saved.append(f'{base}_trigger_dt.png')
            saved.append(f'{base}_trigger_efficiency.png')

            # Pad hit maps + beam profiles, in real geometry. Reuses the
            # coincidence windows just fitted, so the "beam" panels are the
            # in-time hits only.
            try:
                import vmm_beam_profile as _vbp
                _fits = {_n: _r['fit'] for _n, _r in _eff_res.items()}
                _paths, _prof = _vbp.make(hits, _fits, base, out_dir)
                for _p in _paths:
                    saved.append(os.path.basename(_p))
                for _n, _v in _prof.items():
                    if 'x_mean_mm' in _v:
                        print(f"  {_n:8s} beam x={_v['x_mean_mm']:7.1f} +/- "
                              f"{_v['x_rms_mm']:5.1f} mm   "
                              f"y={_v['y_mean_mm']:7.1f} +/- {_v['y_rms_mm']:5.1f} mm")
            except Exception as _exc2:
                print(f'  beam-profile stage failed '
                      f'({_exc2.__class__.__name__}: {_exc2})')
                _prof = None
        else:
            print('  no station coincidence found — check the trigger channel')
    except Exception as _exc:
        print(f'  efficiency stage failed ({_exc.__class__.__name__}: {_exc}) '
              f'— QA plots are unaffected')
        _eff_summary = None

#########################################
# EVENTS SUMMARY (events.json)
#########################################
if _args.events_json:
    # iface/seq from the capture file name (dumpcap or loop_daq naming) via the
    # DAQ repo's shared parser; fall back to nulls when run on arbitrary files.
    _iface, _seq = None, None
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from common_functions import parse_pcapng_name
        # In store mode the positional is a directory whose name has no
        # .pcapng suffix, so parse the original capture path the store
        # recorded when it was decoded.
        _parsed = parse_pcapng_name(_source_capture)
        if _parsed:
            _iface, _seq, _ = _parsed
    except ImportError:
        pass

    events = {
        'pcap_file':       os.path.abspath(_source_capture),
        'iface':           _iface,
        'seq':             _seq,
        'n_packets':       int(pkt_count),
        'n_vm3_packets':   int(vm3_count),
        'n_hits':          int(len(hits)),
        'n_hits_valid':    int(hits['hit_valid'].sum()),
        'hits_per_vmm':    {str(v): int((hits.vmm == v).sum()) for v in vmm_ids},
        't_first_ns':      float(hits['abs_time_ns'].min()) if len(hits) else None,
        't_last_ns':       float(hits['abs_time_ns'].max()) if len(hits) else None,
        'fec_ips':         sorted(src_ips or []),
        'data_format':     data_format,
        'processed_at':    datetime.datetime.now().isoformat(timespec='seconds'),
    }
    # Per-file efficiency scalars, so sub-run/run aggregation (efficiency vs HV)
    # is a cheap JSON scan instead of a re-parse.
    if _eff_summary and _eff_summary.get('stations'):
        events['n_triggers_dedup'] = _eff_summary['n_triggers']
        events['n_masked_hits'] = _eff_summary['n_masked_hits']
        events['efficiency'] = {
            _n: {k: _s[k] for k in ('efficiency', 'efficiency_lo',
                                    'efficiency_hi', 'raw_efficiency',
                                    'accidental_efficiency', 'mu_ns',
                                    'sigma_ns', 'contrast')}
            for _n, _s in _eff_summary['stations'].items()
        }
    if _prof:
        events['beam_profile'] = _prof
    events_path = os.path.join(out_dir, 'events.json')
    with open(events_path, 'w') as f:
        json.dump(events, f, indent=2)
    print(f"Events summary: {events_path}  (n_hits={events['n_hits']:,})")


