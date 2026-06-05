# XYMegas — VMM Hybrid Diagnostic Tools

Scripts for reading and analysing binary pcapng files produced by the VMM3 ASIC readout chain (FEC over UDP). Developed for irradiation measurements with VMM hybrids (no detector involved).

---

## Scripts

### `vmm_hybrid_analysis.py` — offline QA analysis
Reads a pcapng file, parses all VMM3 hits, and saves diagnostic plots as PNGs.

**Usage:**
```bash
python3 vmm_hybrid_pcapng_monitoring.py <path/to/file.pcapng>
```

**Output** — saved in `qa_plots/<filename>/` next to the input file:

| File | Content |
|---|---|
| `<name>_adc.png` | ADC distribution per VMM (subplots) |
| `<name>_chno.png` | Channel occupancy per VMM (subplots) |
| `<name>_hits_per_vmm.png` | Total hits per VMM (bar chart) |
| `<name>_time_vmm<id>.png` | Hit rate vs frame counter, one file per VMM |

**Adapting to your setup:**
- Edit `SRC_IP` at the top of the script to match your FEC source IP.
  - CERN SPS data: `192.168.1.13`
  - Saclay (Pagure) data: `192.168.0.12`
- To find the IP of an unknown file: inspect it with `tshark -r file.pcapng` or use the scapy snippet in the notes below.

**hits DataFrame columns** (available after parsing):

| Column | Type | Description |
|---|---|---|
| `fec` | uint8 | FEC ID (last octet of source IP) |
| `vmm` | uint8 | VMM ID (0–31, from packet data) |
| `time` | uint32 | Frame counter (proxy for time) |
| `ch` | uint8 | Channel number (0–63) |
| `adc` | uint16 | ADC value (0–1023) |
| `over_threshold` | bool | Over-threshold flag |

---

### `FEC_Datadescrambler_Default.py` — live monitoring
Follows a pcapng file in real time and displays live-updating ADC and channel histograms. Assumes 4 VMMs (IDs 0–3). Written by colleagues.

```bash
python3 FEC_Datadescrambler_Default.py <path/to/file.pcapng>
```

### `FEC_Datadescrambler.py` — live monitoring + auto-capture
Same as above but also launches `dumpcap` to capture from the network interface. Requires `dumpcap` installed and the correct interface name (`enp2s0` hardcoded). Written by colleagues.

### `FEC_Analysis.py` — offline analysis (Windows, with detector mapping)
Full offline analysis including strip remapping from an Excel pinout file and 2D hit map. Written by D. Baudin. **Not adapted for hybrid-only use.**

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** Use the miniconda Python interpreter, not the system Python.
> On this machine: `/local/home/ak271430/miniconda3/bin/python3`

---

## Finding the source IP of an unknown pcapng file

```python
from scapy.all import PcapReader, UDP, IP
ips = {}
with PcapReader("yourfile.pcapng") as r:
    for pkt in r:
        if IP in pkt and UDP in pkt:
            k = (pkt[IP].src, pkt[IP].dst)
            ips[k] = ips.get(k, 0) + 1
        if sum(ips.values()) > 5000:
            break
for k, v in sorted(ips.items(), key=lambda x: -x[1]):
    print(v, k)
```

---

## Notes on timing

The `time` field stores the **frame counter** from the UDP payload header — it is a proxy for time, not an absolute timestamp. To convert to real time, multiply by the frame period (depends on your FEC clock configuration).
