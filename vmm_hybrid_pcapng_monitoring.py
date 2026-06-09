#!/usr/bin/env python3.12
"""
VMM hybrid diagnostic script for irradiation measurements.
Reads a pcapng file and produces per-VMM ADC and channel occupancy plots.
@author: ak271430 Alexandra Kallitsopoulou (alexandra.kallitsopoulou@cea.fr)

Usage:
    python3 vmm_hybrid_pcapng_monitoring.py <pcap_file>

Output PNGs are saved in the same directory as the input file.

hits DataFrame columns:
    fec            : FEC ID (last octet of source IP)
    vmm            : VMM ID (0-31, from packet data)
    time           : frame counter (proxy for time; multiply by frame period for real time)
    ch             : channel number (0-63)
    adc            : ADC value (0-1023)
    over_threshold : over-threshold flag (bool)
    offset         : 5-bit signed offset (-16 to +15)
    bcid           : bunch-crossing ID (0-4095, 12-bit)
    tdc            : TDC fine-timing value (0-255, 8-bit)
"""

import sys
import os
import struct

# Self-re-exec: ROOT was built against Python 3.12; if we're running under any
# other interpreter, replace this process with the right one (sourcing thisroot.sh
# first so PyROOT finds its shared libraries).
_ROOT312  = "/local/home/ak271430/miniconda3/envs/root312/bin/python3.12"
_THISROOT = "/local/home/ak271430/Software/root/bin/thisroot.sh"
if sys.executable != _ROOT312:
    os.execvp("bash", [
        "bash", "-c",
        f'source "{_THISROOT}" && exec "{_ROOT312}" "$@"',
        "python", *sys.argv,
    ])

import array

# Redirect per-user cache dirs before any import touches /tmp/.cache on shared machines
_user = os.getlogin()
os.environ.setdefault('XDG_CACHE_HOME', f'/tmp/.cache-{_user}')
os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{_user}')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scapy.all import PcapReader, UDP, IP
import ROOT
ROOT.gROOT.SetBatch(True)

#########################################
# AUTO-DETECT FEC SOURCE IPs
# Scans the first PROBE_PACKETS packets and
# returns all source IPs that send VM3 data.
# Pass SRC_IP_OVERRIDE to skip auto-detection.
#########################################
PROBE_PACKETS = 500
SRC_IP_OVERRIDE = None  # e.g. "192.168.1.13" to force a specific IP


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
def parse_block(block: bytes, frame_counter: int, fec_id: int,
                fec_buf, vmm_buf, time_buf, ch_buf, adc_buf, ot_buf,
                offset_buf, bcid_buf, tdc_buf):
    if len(block) < 22 or block[4:7] != b'VM3':
        return
    for i in range(0, len(block) - 22, 6):
        d1, d2 = struct.unpack_from('>IH', block, i + 16)
        if d2 & 0x8000:          # valid-hit flag = MSB of d2
            fec_buf.append(fec_id)
            vmm_buf.append((d1 >> 22) & 0x1F)   # bits 26-22 of d1
            time_buf.append(frame_counter)
            ch_buf.append((d2 >> 8) & 0x3F)     # bits 13-8 of d2
            adc_buf.append((d1 >> 12) & 0x3FF)  # bits 21-12 of d1
            ot_buf.append((d2 >> 14) & 0x1)     # bit 14 of d2
            raw_off = (d1 >> 27) & 0x1F          # bits 31-27 of d1 (5-bit signed)
            offset_buf.append(raw_off if raw_off < 16 else raw_off - 32)
            bcid_buf.append(d1 & 0xFFF)          # bits 11-0 of d1
            tdc_buf.append(d2 & 0xFF)            # bits 7-0 of d2

#########################################
# MAIN
#########################################
if len(sys.argv) < 2:
    print("Usage: python3 vmm_hybrid_pcapng_monitoring.py <pcap_file>")
    sys.exit(1)

pcap_file = sys.argv[1]
if not os.path.isfile(pcap_file):
    print(f"File not found: {pcap_file}")
    sys.exit(1)

# Memory-efficient typed arrays (no Python object overhead)
fec_buf    = array.array('B')   # uint8
vmm_buf    = array.array('B')   # uint8
time_buf   = array.array('I')   # uint32 ('I'=unsigned int, always 4 bytes; 'L' is 8 bytes on 64-bit Linux)
ch_buf     = array.array('B')   # uint8
adc_buf    = array.array('H')   # uint16
ot_buf     = array.array('B')   # uint8 (0/1)
offset_buf = array.array('b')   # int8  (-16 to +15, 5-bit signed)
bcid_buf   = array.array('H')   # uint16 (0-4095)
tdc_buf    = array.array('B')   # uint8  (0-255)

if SRC_IP_OVERRIDE:
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

print(f"Reading: {pcap_file}")
pkt_count = 0
vm3_count = 0

with PcapReader(pcap_file) as reader:
    for pkt in reader:
        if UDP in pkt and IP in pkt and pkt[IP].src in src_ips:
            payload = bytes(pkt[UDP].payload)
            fc     = struct.unpack_from('>I', payload)[0]
            fec_id = int(pkt[IP].src.split('.')[-1])
            n_before = len(fec_buf)
            parse_block(payload, fc, fec_id,
                        fec_buf, vmm_buf, time_buf, ch_buf, adc_buf, ot_buf,
                        offset_buf, bcid_buf, tdc_buf)
            if len(fec_buf) > n_before:
                vm3_count += 1
        pkt_count += 1
        if pkt_count % 10000 == 0:
            print(f"  {pkt_count} packets | {len(fec_buf):,} hits so far...")

print(f"\nDone: {pkt_count} packets | {vm3_count} VM3 packets | {len(fec_buf):,} total hits")

# Build DataFrame with compact dtypes
hits = pd.DataFrame({
    'fec':            np.frombuffer(fec_buf,    dtype=np.uint8).copy(),
    'vmm':            np.frombuffer(vmm_buf,    dtype=np.uint8).copy(),
    'time':           np.frombuffer(time_buf,   dtype=np.uint32).copy(),
    'ch':             np.frombuffer(ch_buf,     dtype=np.uint8).copy(),
    'adc':            np.frombuffer(adc_buf,    dtype=np.uint16).copy(),
    'over_threshold': np.frombuffer(ot_buf,     dtype=np.uint8).astype(bool).copy(),
    'offset':         np.frombuffer(offset_buf, dtype=np.int8).copy(),
    'bcid':           np.frombuffer(bcid_buf,   dtype=np.uint16).copy(),
    'tdc':            np.frombuffer(tdc_buf,    dtype=np.uint8).copy(),
})
del fec_buf, vmm_buf, time_buf, ch_buf, adc_buf, ot_buf, offset_buf, bcid_buf, tdc_buf

mem_mb = hits.memory_usage(deep=True).sum() / 1e6
print(f"DataFrame memory: {mem_mb:.1f} MB")
print(f"\nVMM IDs found: {sorted(hits.vmm.unique())}")
for v in sorted(hits.vmm.unique()):
    n = int((hits.vmm == v).sum())
    print(f"  VMM {v:2d}: {n:,} hits")

#########################################
# HISTOGRAM PARAMETERS
#########################################
ADC_BINS    = 100; ADC_MIN    = 0;   ADC_MAX    = 1024
CH_BINS     = 64;  CH_MIN     = 0;   CH_MAX     = 64
TIME_BINS   = 200
BCID_BINS   = 100; BCID_MIN   = 0;   BCID_MAX   = 4096
TDC_BINS    = 64;  TDC_MIN    = 0;   TDC_MAX    = 256
OFFSET_BINS = 32;  OFFSET_MIN = -16; OFFSET_MAX = 16

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
    ax.hist(data, bins=ADC_BINS, range=(ADC_MIN, ADC_MAX), color='steelblue', alpha=0.8)
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
    ax.hist(adc_off, bins=ADC_BINS, range=(ADC_MIN, ADC_MAX), color='steelblue', alpha=0.6, label=f'Not OT ({len(adc_off):,})')
    ax.hist(adc_on,  bins=ADC_BINS, range=(ADC_MIN, ADC_MAX), color='tomato',    alpha=0.6, label=f'OT ({len(adc_on):,})')
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
    ax.hist(data, bins=CH_BINS, range=(CH_MIN, CH_MAX), color='tomato', alpha=0.8)
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

# --- Save all PNGs in qa_plots/<base>/ ---
base     = os.path.splitext(os.path.basename(pcap_file))[0]
out_dir  = os.path.join(os.path.dirname(os.path.abspath(pcap_file)), "qa_plots", base)
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
    ax.hist(data, bins=BCID_BINS, range=(BCID_MIN, BCID_MAX), color='teal', alpha=0.8)
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
    ax.hist(data, bins=TDC_BINS, range=(TDC_MIN, TDC_MAX), color='darkcyan', alpha=0.8)
    ax.set_title(f"VMM {v}  ({len(data):,} hits)")
    ax.set_xlabel("TDC")
    ax.set_ylabel("Counts")
for idx in range(n_vmm, nrows * ncols):
    axes_tdc[idx // ncols][idx % ncols].set_visible(False)
fig_tdc.tight_layout()
_save(fig_tdc, f"{base}_tdc.png")

# --- Offset distribution ---
fig_offset, axes_offset = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
fig_offset.suptitle("Offset distribution per VMM (5-bit signed)", fontsize=14)
for idx, v in enumerate(vmm_ids):
    ax   = axes_offset[idx // ncols][idx % ncols]
    data = hits.loc[hits.vmm == v, 'offset']
    ax.hist(data, bins=OFFSET_BINS, range=(OFFSET_MIN, OFFSET_MAX), color='saddlebrown', alpha=0.8)
    ax.set_title(f"VMM {v}  ({len(data):,} hits)")
    ax.set_xlabel("Offset (signed)")
    ax.set_ylabel("Counts")
for idx in range(n_vmm, nrows * ncols):
    axes_offset[idx // ncols][idx % ncols].set_visible(False)
fig_offset.tight_layout()
_save(fig_offset, f"{base}_offset.png")

# --- Time distribution: one PNG per VMM (built and saved immediately to avoid accumulation) ---
t_min, t_max = int(hits.time.min()), int(hits.time.max())
for v in vmm_ids:
    fig_t, ax_t = plt.subplots(figsize=(12, 5))
    data = hits.loc[hits.vmm == v, 'time']
    ax_t.hist(data, bins=TIME_BINS, range=(t_min, t_max), color='darkorange', alpha=0.8)
    ax_t.set_title(f"Hit rate over time — VMM {v}  ({len(data):,} hits)")
    ax_t.set_xlabel("Frame counter (proxy for time)")
    ax_t.set_ylabel("Hits per bin")
    fig_t.tight_layout()
    _save(fig_t, f"{base}_time_vmm{v}.png")

print(f"\nSaved {len(saved)} files in: {out_dir}")
for name in saved:
    print(f"  {name}")

#########################################
# ROOT OUTPUT
#########################################
_CHUNK   = 500_000
_w_chunk = np.ones(_CHUNK, dtype=np.float64)  # reusable weights buffer

def _fill1d(h, arr):
    for start in range(0, len(arr), _CHUNK):
        a = np.ascontiguousarray(arr[start:start + _CHUNK], dtype=np.float64)
        h.FillN(len(a), a, _w_chunk[:len(a)])

# Cling-compiled C++ filler for the per-hit TTree — avoids a slow Python loop
ROOT.gInterpreter.Declare("""
void _vmm_fill_hits(TTree* t,
                    const unsigned char*  fec_a,
                    const unsigned char*  vmm_a,
                    const unsigned char*  ch_a,
                    const unsigned short* adc_a,
                    const unsigned char*  ot_a,
                    const unsigned int*   time_a,
                    const unsigned char*  off_a,
                    const unsigned short* bcid_a,
                    const unsigned char*  tdc_a,
                    long long n)
{
    unsigned char  fec = 0, vmm = 0, ch = 0, ot = 0, tdc = 0;
    unsigned short adc = 0, bcid = 0;
    unsigned int   ts  = 0;
    signed char    off = 0;
    t->Branch("fec",            &fec,  "fec/b");
    t->Branch("vmm",            &vmm,  "vmm/b");
    t->Branch("ch",             &ch,   "ch/b");
    t->Branch("adc",            &adc,  "adc/s");
    t->Branch("over_threshold", &ot,   "over_threshold/b");
    t->Branch("time",           &ts,   "time/i");
    t->Branch("offset",         &off,  "offset/B");
    t->Branch("bcid",           &bcid, "bcid/s");
    t->Branch("tdc",            &tdc,  "tdc/b");
    for (long long i = 0; i < n; ++i) {
        fec  = fec_a[i];  vmm  = vmm_a[i]; ch   = ch_a[i];
        adc  = adc_a[i];  ot   = ot_a[i];  ts   = time_a[i];
        off  = (signed char)off_a[i]; bcid = bcid_a[i]; tdc  = tdc_a[i];
        t->Fill();
    }
}
""")

import datetime
import ctypes

root_path = os.path.join(out_dir, f"{base}.root")
rf = ROOT.TFile(root_path, "RECREATE")

# --- hits_parameters: one-entry TTree storing run parameters and statistics ---
t_params = ROOT.TTree("hits_parameters", "Run parameters and statistics")

_sbuf = {}  # string buffers must stay alive until Fill()
_ibuf = {}  # int buffers must stay alive until Fill()

def _sbranch(name, val, maxlen=512):
    buf = ctypes.create_string_buffer(str(val).encode(), maxlen)
    _sbuf[name] = buf
    t_params.Branch(name, buf, f"{name}/C")

def _ibranch(name, val):
    buf = array.array('i', [int(val)])
    _ibuf[name] = buf
    t_params.Branch(name, buf, f"{name}/I")

_sbranch("input_file",        os.path.abspath(pcap_file))
_sbranch("created",           datetime.datetime.now().isoformat(timespec='seconds'))
_sbranch("fec_ips",           ", ".join(sorted(src_ips)))
_sbranch("src_ip_override",   str(SRC_IP_OVERRIDE))
_ibranch("probe_packets",     PROBE_PACKETS)
_ibranch("n_packets",         pkt_count)
_ibranch("n_vm3_packets",     vm3_count)
_ibranch("n_hits_total",      len(hits))
_ibranch("n_vmm",             n_vmm)
_sbranch("vmm_ids",           ", ".join(str(v) for v in vmm_ids))
_ibranch("frame_counter_min", t_min)
_ibranch("frame_counter_max", t_max)
_ibranch("adc_bins",          ADC_BINS)
_ibranch("adc_min",           ADC_MIN)
_ibranch("adc_max",           ADC_MAX)
_ibranch("ch_bins",           CH_BINS)
_ibranch("ch_min",            CH_MIN)
_ibranch("ch_max",            CH_MAX)
_ibranch("time_bins",         TIME_BINS)
_ibranch("bcid_bins",         BCID_BINS)
_ibranch("bcid_min",          BCID_MIN)
_ibranch("bcid_max",          BCID_MAX)
_ibranch("tdc_bins",          TDC_BINS)
_ibranch("tdc_min",           TDC_MIN)
_ibranch("tdc_max",           TDC_MAX)
_ibranch("offset_bins",       OFFSET_BINS)
_ibranch("offset_min",        OFFSET_MIN)
_ibranch("offset_max",        OFFSET_MAX)

# hits_per_vmm[32]: indexed directly by VMM ID; -1 means VMM not present in this run
_hpv = array.array('i', [-1] * 32)
for v in vmm_ids:
    _hpv[v] = int((hits.vmm == v).sum())
t_params.Branch("hits_per_vmm", _hpv, "hits_per_vmm[32]/I")

t_params.Fill()
t_params.Write()

# --- Summary: total hits per VMM (one bin per VMM, labelled) ---
h_hits = ROOT.TH1F("hits_per_vmm", "Total hits per VMM;VMM ID;Hits",
                   n_vmm, -0.5, n_vmm - 0.5)
for i, v in enumerate(vmm_ids):
    h_hits.SetBinContent(i + 1, int((hits.vmm == v).sum()))
    h_hits.GetXaxis().SetBinLabel(i + 1, str(v))
h_hits.Write()

for v in vmm_ids:
    vdir = rf.mkdir(f"vmm{v:02d}")
    vdir.cd()
    vdata  = hits.loc[hits.vmm == v]
    adc_all    = vdata['adc'].values
    adc_ot     = vdata.loc[ vdata.over_threshold, 'adc'].values
    adc_not_ot = vdata.loc[~vdata.over_threshold, 'adc'].values

    h_adc = ROOT.TH1F("adc", f"ADC — VMM {v};ADC;Counts", ADC_BINS, ADC_MIN, ADC_MAX)
    _fill1d(h_adc, adc_all)
    h_adc.Write()

    h_adc_ot = ROOT.TH1F("adc_ot", f"ADC (OT) — VMM {v};ADC;Counts", ADC_BINS, ADC_MIN, ADC_MAX)
    _fill1d(h_adc_ot, adc_ot)
    h_adc_ot.Write()

    h_adc_not_ot = ROOT.TH1F("adc_not_ot", f"ADC (not OT) — VMM {v};ADC;Counts", ADC_BINS, ADC_MIN, ADC_MAX)
    _fill1d(h_adc_not_ot, adc_not_ot)
    h_adc_not_ot.Write()

    h_ot = ROOT.TH1F("ot_flag", f"OT flag — VMM {v};;Counts", 2, 0, 2)
    h_ot.SetBinContent(1, int((~vdata.over_threshold).sum()))
    h_ot.SetBinContent(2, int(vdata.over_threshold.sum()))
    h_ot.GetXaxis().SetBinLabel(1, "Not OT")
    h_ot.GetXaxis().SetBinLabel(2, "OT")
    h_ot.Write()

    h_ch = ROOT.TH1F("ch_occ", f"Channel occupancy — VMM {v};Channel;Counts", CH_BINS, CH_MIN, CH_MAX)
    _fill1d(h_ch, vdata['ch'].values.astype(np.float64))
    h_ch.Write()

    h_time = ROOT.TH1F("time", f"Hit rate over time — VMM {v};Frame counter;Hits per bin",
                       TIME_BINS, t_min, t_max)
    _fill1d(h_time, vdata['time'].values.astype(np.float64))
    h_time.Write()

    h_adc_ch = ROOT.TH2F("adc_vs_ch", f"ADC vs channel — VMM {v};Channel;ADC;Hits",
                          CH_BINS, CH_MIN, CH_MAX, ADC_BINS, ADC_MIN, ADC_MAX)
    h2, _, _ = np.histogram2d(
        vdata['ch'].values.astype(np.float64),
        vdata['adc'].values.astype(np.float64),
        bins=[CH_BINS, ADC_BINS], range=[[CH_MIN, CH_MAX], [ADC_MIN, ADC_MAX]],
    )
    for ix in range(64):
        for iy in range(100):
            if h2[ix, iy] > 0:
                h_adc_ch.SetBinContent(ix + 1, iy + 1, h2[ix, iy])
    h_adc_ch.Write()

    h_bcid = ROOT.TH1F("bcid", f"BCID — VMM {v};BCID;Counts", BCID_BINS, BCID_MIN, BCID_MAX)
    _fill1d(h_bcid, vdata['bcid'].values.astype(np.float64))
    h_bcid.Write()

    h_tdc = ROOT.TH1F("tdc", f"TDC — VMM {v};TDC;Counts", TDC_BINS, TDC_MIN, TDC_MAX)
    _fill1d(h_tdc, vdata['tdc'].values.astype(np.float64))
    h_tdc.Write()

    h_offset = ROOT.TH1F("offset", f"Offset — VMM {v};Offset (signed);Counts",
                          OFFSET_BINS, OFFSET_MIN, OFFSET_MAX)
    _fill1d(h_offset, vdata['offset'].values.astype(np.float64))
    h_offset.Write()

    del vdata, adc_all, adc_ot, adc_not_ot

# --- hits TTree: one row per hit, written into rf via Cling-compiled C++ filler ---
rf.cd()
_hits_tree = ROOT.TTree("hits", "Per-hit data")
ROOT._vmm_fill_hits(
    _hits_tree,
    np.ascontiguousarray(hits['fec'].values,            dtype=np.uint8),
    np.ascontiguousarray(hits['vmm'].values,            dtype=np.uint8),
    np.ascontiguousarray(hits['ch'].values,             dtype=np.uint8),
    np.ascontiguousarray(hits['adc'].values,            dtype=np.uint16),
    np.ascontiguousarray(hits['over_threshold'].values, dtype=np.uint8),
    np.ascontiguousarray(hits['time'].values,           dtype=np.uint32),
    np.ascontiguousarray(hits['offset'].values.view(np.uint8)),
    np.ascontiguousarray(hits['bcid'].values,           dtype=np.uint16),
    np.ascontiguousarray(hits['tdc'].values,            dtype=np.uint8),
    len(hits),
)
_hits_tree.Write()

rf.Close()
print(f"ROOT file: {root_path}")
