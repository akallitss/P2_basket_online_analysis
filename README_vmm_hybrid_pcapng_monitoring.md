# vmm_hybrid_pcapng_monitoring.py

Diagnostic script for VMM3 hybrid readout data captured as `.pcapng` network traces.
It parses UDP packets from one or more FEC boards, extracts hit data, and saves
a set of QA plots (ADC, occupancy, hit rate) without any intermediate file writing.

---

## How it works

### 1. FEC auto-detection

On startup the script probes the first 500 packets of the capture file and
identifies every source IP that sends packets with the `VM3` magic header.
Those IPs are treated as FEC boards for the rest of the run.

To skip auto-detection and force a specific IP, set `SRC_IP_OVERRIDE` at the
top of the script:

```python
SRC_IP_OVERRIDE = "192.168.1.13"
```

### 2. Packet parsing

For every UDP packet from a detected FEC IP the script reads:

| field | bits | description |
|---|---|---|
| `fec` | — | last octet of the source IP (e.g. `13` for `192.168.1.13`) |
| `vmm` | 5 bits | VMM ID (0–31) |
| `ch` | 6 bits | channel number (0–63) |
| `adc` | 10 bits | ADC value (0–1023) |
| `over_threshold` | 1 bit | over-threshold flag |
| `time` | 32 bits | frame counter at the packet header (proxy for time) |

Each UDP payload has the following structure:

```
payload bytes
 ├─ bytes  0– 3   frame counter (32-bit, used as time proxy)
 ├─ bytes  4– 6   "VM3" magic word (packet skipped if absent)
 ├─ bytes  7–15   header (skipped)
 └─ bytes 16+     hits, 6 bytes each, packed at the bit level
      ├─ d1  (bytes +0 to +3, 32 bits)
      │    bits  5– 9  →  VMM ID  (0–31)
      │    bits 10–19  →  ADC     (0–1023)
      └─ d2  (bytes +4 to +5, 16 bits)
           bit   0     →  valid flag  (hit ignored if 0)
           bit   1     →  over-threshold flag
           bits  2– 7  →  channel     (0–63)
```

One packet can carry many hits back-to-back; each valid hit becomes one row
in the output DataFrame.

Hits are accumulated in compact typed arrays (`array.array`) to minimise
memory overhead before being converted to a pandas DataFrame at the end.

### 3. Plots produced

All plots are saved under `qa_plots/<input_basename>/` next to the input file.

| filename | content |
|---|---|
| `<base>_adc.png` | ADC value distribution (0–1023), one panel per VMM |
| `<base>_adc_ot.png` | ADC distribution split by over-threshold flag (Not OT vs OT overlaid), one panel per VMM |
| `<base>_ot.png` | Over-threshold flag distribution — bar chart of Not OT vs OT hit counts with OT fraction in title, one panel per VMM |
| `<base>_chno.png` | Channel occupancy (0–63), one panel per VMM |
| `<base>_hits_per_vmm.png` | Bar chart of total hits per VMM |
| `<base>_time_vmm<N>.png` | Hit rate vs frame counter for VMM N (one file per VMM) |

---

## Requirements

```
python >= 3.8
numpy
pandas
matplotlib
scapy
```

Install with:

```bash
pip install numpy pandas matplotlib scapy
```

---

## Usage

```bash
python3 vmm_hybrid_pcapng_monitoring.py <path/to/capture.pcapng>
```

### Example

```bash
python3 vmm_hybrid_pcapng_monitoring.py /data/run042/hybrid_run042.pcapng
```

Console output:

```
Auto-detected FEC IP(s): 192.168.1.13
Reading: /data/run042/hybrid_run042.pcapng
  10000 packets | 45,231 hits so far...
  20000 packets | 91,804 hits so far...

Done: 23417 packets | 18632 VM3 packets | 102,558 total hits
DataFrame memory: 2.1 MB

VMM IDs found: [0, 1, 2, 3]
  VMM  0: 25,412 hits
  VMM  1: 26,891 hits
  VMM  2: 24,103 hits
  VMM  3: 26,152 hits

Saved 10 files in: /data/run042/qa_plots/hybrid_run042/
```

---

## Output location

Plots are always written to:

```
<directory of input file>/qa_plots/<input_basename>/
```

For example, input `/data/run042/hybrid_run042.pcapng` produces output in
`/data/run042/qa_plots/hybrid_run042/`.

---

## Configuration options

Both options are edited directly at the top of the script:

| variable | default | effect |
|---|---|---|
| `PROBE_PACKETS` | `500` | number of packets scanned during FEC auto-detection |
| `SRC_IP_OVERRIDE` | `None` | if set, skips auto-detection and uses this IP only |

---

## DataFrame columns

After parsing, hits are stored in a `pd.DataFrame` with the following columns:

| column | dtype | description |
|---|---|---|
| `fec` | uint8 | FEC board ID (last IP octet) |
| `vmm` | uint8 | VMM chip ID (0–31) |
| `time` | uint32 | frame counter (multiply by frame period for real time) |
| `ch` | uint8 | channel number (0–63) |
| `adc` | uint16 | ADC value (0–1023) |
| `over_threshold` | bool | over-threshold flag |

This DataFrame is available in memory after the script finishes, so the script
can be imported and extended for custom analysis.