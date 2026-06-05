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
"""

#THIS IS A TEST
import sys
import os
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
                fec_buf, vmm_buf, time_buf, ch_buf, adc_buf, ot_buf):
    if len(block) < 22 or block[4:7] != b'VM3':
        return
    for i in range(0, len(block) - 22, 6):
        d1 = "{:032b}".format(int(block[i+16:i+20].hex(), 16))
        d2 = "{:016b}".format(int(block[i+20:i+22].hex(), 16))
        if d2[0] == '1':
            fec_buf.append(fec_id)
            vmm_buf.append(int(d1[5:10], 2))
            time_buf.append(frame_counter)
            ch_buf.append(int(d2[2:8], 2))
            adc_buf.append(int(d1[10:20], 2))
            ot_buf.append(int(d2[1], 2))

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
fec_buf  = array.array('B')   # uint8
vmm_buf  = array.array('B')   # uint8
time_buf = array.array('I')   # uint32 ('I'=unsigned int, always 4 bytes; 'L' is 8 bytes on 64-bit Linux)
ch_buf   = array.array('B')   # uint8
adc_buf  = array.array('H')   # uint16
ot_buf   = array.array('B')   # uint8 (0/1)

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
            fc     = int(payload[0:4].hex(), 16)
            fec_id = int(pkt[IP].src.split('.')[-1])
            n_before = len(fec_buf)
            parse_block(payload, fc, fec_id,
                        fec_buf, vmm_buf, time_buf, ch_buf, adc_buf, ot_buf)
            if len(fec_buf) > n_before:
                vm3_count += 1
        pkt_count += 1
        if pkt_count % 10000 == 0:
            print(f"  {pkt_count} packets | {len(fec_buf):,} hits so far...")

print(f"\nDone: {pkt_count} packets | {vm3_count} VM3 packets | {len(fec_buf):,} total hits")

# Build DataFrame with compact dtypes
hits = pd.DataFrame({
    'fec':            np.frombuffer(fec_buf,  dtype=np.uint8).copy(),
    'vmm':            np.frombuffer(vmm_buf,  dtype=np.uint8).copy(),
    'time':           np.frombuffer(time_buf, dtype=np.uint32).copy(),
    'ch':             np.frombuffer(ch_buf,   dtype=np.uint8).copy(),
    'adc':            np.frombuffer(adc_buf,  dtype=np.uint16).copy(),
    'over_threshold': np.frombuffer(ot_buf,   dtype=np.uint8).astype(bool).copy(),
})

mem_mb = hits.memory_usage(deep=True).sum() / 1e6
print(f"DataFrame memory: {mem_mb:.1f} MB")
print(f"\nVMM IDs found: {sorted(hits.vmm.unique())}")
for v in sorted(hits.vmm.unique()):
    n = int((hits.vmm == v).sum())
    print(f"  VMM {v:2d}: {n:,} hits")

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
    ax.hist(data, bins=100, range=(0, 1024), color='steelblue', alpha=0.8)
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
    ax.hist(adc_off, bins=100, range=(0, 1024), color='steelblue', alpha=0.6, label=f'Not OT ({len(adc_off):,})')
    ax.hist(adc_on,  bins=100, range=(0, 1024), color='tomato',    alpha=0.6, label=f'OT ({len(adc_on):,})')
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
    ax.hist(data, bins=64, range=(0, 64), color='tomato', alpha=0.8)
    ax.set_title(f"VMM {v}  ({len(data):,} hits)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")
    ax.set_xlim(0, 63)
for idx in range(n_vmm, nrows * ncols):
    axes_ch[idx // ncols][idx % ncols].set_visible(False)
fig_ch.tight_layout()

# --- Total hits per VMM ---
fig_sum, ax_sum = plt.subplots(figsize=(max(6, n_vmm), 4))
ax_sum.bar([str(v) for v in vmm_ids],
           [int((hits.vmm == v).sum()) for v in vmm_ids],
           color='mediumpurple', alpha=0.8)
ax_sum.set_title("Total hits per VMM")
ax_sum.set_xlabel("VMM ID")
ax_sum.set_ylabel("Hits")
fig_sum.tight_layout()

# --- Time distribution: one PNG per VMM ---
n_time_bins = 200
t_min, t_max = int(hits.time.min()), int(hits.time.max())
time_figs = {}
for v in vmm_ids:
    fig_t, ax_t = plt.subplots(figsize=(12, 5))
    data = hits.loc[hits.vmm == v, 'time']
    ax_t.hist(data, bins=n_time_bins, range=(t_min, t_max), color='darkorange', alpha=0.8)
    ax_t.set_title(f"Hit rate over time — VMM {v}  ({len(data):,} hits)")
    ax_t.set_xlabel("Frame counter (proxy for time)")
    ax_t.set_ylabel("Hits per bin")
    fig_t.tight_layout()
    time_figs[v] = fig_t

# --- Save all PNGs in qa_plots/<base>/ ---
base     = os.path.splitext(os.path.basename(pcap_file))[0]
out_dir  = os.path.join(os.path.dirname(os.path.abspath(pcap_file)), "qa_plots", base)
os.makedirs(out_dir, exist_ok=True)

saved = []

fig_adc.savefig(os.path.join(out_dir,    f"{base}_adc.png"),        dpi=150); saved.append("_adc.png")
fig_adc_ot.savefig(os.path.join(out_dir, f"{base}_adc_ot.png"),    dpi=150); saved.append("_adc_ot.png")
fig_ot.savefig(os.path.join(out_dir,    f"{base}_ot.png"),          dpi=150); saved.append("_ot.png")
fig_ch.savefig(os.path.join(out_dir,  f"{base}_chno.png"),         dpi=150); saved.append("_chno.png")
fig_sum.savefig(os.path.join(out_dir, f"{base}_hits_per_vmm.png"), dpi=150); saved.append("_hits_per_vmm.png")
for v, fig_t in time_figs.items():
    fname = f"{base}_time_vmm{v}.png"
    fig_t.savefig(os.path.join(out_dir, fname), dpi=150)
    saved.append(fname)

print(f"\nSaved {len(saved)} files in: {out_dir}")
for name in saved:
    print(f"  {name}")

#########################################
# ROOT OUTPUT
#########################################
def _fill1d(h, arr):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    h.FillN(len(a), a, np.ones(len(a), dtype=np.float64))

root_path = os.path.join(out_dir, f"{base}.root")
rf = ROOT.TFile(root_path, "RECREATE")

# Summary: total hits per VMM (one bin per VMM, labelled)
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

    h_adc = ROOT.TH1F("adc", f"ADC — VMM {v};ADC;Counts", 100, 0, 1024)
    _fill1d(h_adc, adc_all)
    h_adc.Write()

    h_adc_ot = ROOT.TH1F("adc_ot", f"ADC (OT) — VMM {v};ADC;Counts", 100, 0, 1024)
    _fill1d(h_adc_ot, adc_ot)
    h_adc_ot.Write()

    h_adc_not_ot = ROOT.TH1F("adc_not_ot", f"ADC (not OT) — VMM {v};ADC;Counts", 100, 0, 1024)
    _fill1d(h_adc_not_ot, adc_not_ot)
    h_adc_not_ot.Write()

    h_ot = ROOT.TH1F("ot_flag", f"OT flag — VMM {v};;Counts", 2, 0, 2)
    h_ot.SetBinContent(1, int((~vdata.over_threshold).sum()))
    h_ot.SetBinContent(2, int(vdata.over_threshold.sum()))
    h_ot.GetXaxis().SetBinLabel(1, "Not OT")
    h_ot.GetXaxis().SetBinLabel(2, "OT")
    h_ot.Write()

    h_ch = ROOT.TH1F("ch_occ", f"Channel occupancy — VMM {v};Channel;Counts", 64, 0, 64)
    _fill1d(h_ch, vdata['ch'].values.astype(np.float64))
    h_ch.Write()

    h_time = ROOT.TH1F("time", f"Hit rate over time — VMM {v};Frame counter;Hits per bin",
                       200, t_min, t_max)
    _fill1d(h_time, vdata['time'].values.astype(np.float64))
    h_time.Write()

rf.Close()
print(f"ROOT file: {root_path}")
