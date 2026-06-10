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
| `offset` | 5 bits (signed) | timing offset, two's-complement (−16 to +15) |
| `bcid` | 12 bits | bunch-crossing ID (0–4095) |
| `tdc` | 8 bits | TDC fine-timing value (0–255) |

Each UDP payload has the following structure:

```
payload bytes
 ├─ bytes  0– 3   frame counter (32-bit, used as time proxy)
 ├─ bytes  4– 6   "VM3" magic word (packet skipped if absent)
 ├─ bytes  7–15   header (skipped)
 └─ bytes 16+     hits, 6 bytes each, packed at the bit level
      ├─ d1  (bytes +0 to +3, 32 bits)
      │    bits  0– 4  →  offset  (5-bit signed, two's-complement)
      │    bits  5– 9  →  VMM ID  (0–31)
      │    bits 10–19  →  ADC     (0–1023)
      │    bits 20–31  →  BCID    (0–4095)
      └─ d2  (bytes +4 to +5, 16 bits)
           bit   0     →  valid flag  (hit ignored if 0)
           bit   1     →  over-threshold flag
           bits  2– 7  →  channel     (0–63)
           bits  8–15  →  TDC         (0–255)
```

One packet can carry many hits back-to-back; each valid hit becomes one row
in the output DataFrame.

Hits are accumulated in compact typed arrays (`array.array`) to minimise
memory overhead before being converted to a pandas DataFrame at the end.

### 3. Outputs produced

All outputs are saved under `qa_plots/<input_basename>/` next to the input file.

#### PNG plots

| filename | content |
|---|---|
| `<base>_adc.png` | ADC value distribution (0–1023), one panel per VMM |
| `<base>_adc_ot.png` | ADC distribution split by over-threshold flag (Not OT vs OT overlaid), one panel per VMM |
| `<base>_adc_vs_ch.png` | **2-D histogram: ADC (y) vs channel (x), log colour scale (hits).** Reveals per-channel pedestal bands, noisy or dead channels, and ADC saturation at a glance. One panel per VMM. |
| `<base>_ot.png` | Over-threshold flag distribution — bar chart of Not OT vs OT hit counts with OT fraction in title, one panel per VMM |
| `<base>_chno.png` | Channel occupancy (0–63), one panel per VMM |
| `<base>_hits_per_vmm.png` | Bar chart of total hits per VMM |
| `<base>_time_vmm<N>.png` | Hit rate vs frame counter for VMM N (one file per VMM) |
| `<base>_bcid.png` | BCID distribution (0–4095), one panel per VMM |
| `<base>_tdc.png` | TDC distribution (0–255), one panel per VMM |
| `<base>_offset.png` | Offset distribution (−16 to +15, 5-bit signed), one panel per VMM |

#### ROOT file

A single `<base>.root` file is written alongside the PNGs with the following
top-level structure:

```
hits_parameters   TTree          — 1 entry: run config + histogram binning (see below)
hits              TTree          — 1 entry per hit: full raw hit data (see below)
hits_per_vmm      TH1F           — total hits per VMM
vmm00/            TDirectoryFile — per-VMM histograms
vmm01/            TDirectoryFile
…
```

##### `hits_parameters` TTree (1 entry)

Stores everything needed to reproduce the run configuration and re-use the
same histogram binning in downstream analysis:

| branch | type | content |
|---|---|---|
| `input_file` | `Char_t[]` | absolute path of the source pcapng |
| `created` | `Char_t[]` | ISO-8601 timestamp of when the file was produced |
| `fec_ips` | `Char_t[]` | comma-separated list of detected FEC IPs |
| `src_ip_override` | `Char_t[]` | value of `SRC_IP_OVERRIDE` (`None` if auto-detected) |
| `probe_packets` | `Int_t` | number of packets scanned for FEC auto-detection |
| `n_packets` | `Int_t` | total packets in the capture |
| `n_vm3_packets` | `Int_t` | packets that contained VM3 hit data |
| `n_hits_total` | `Int_t` | total hits parsed |
| `n_vmm` | `Int_t` | number of distinct VMMs seen |
| `vmm_ids` | `Char_t[]` | comma-separated list of active VMM IDs |
| `frame_counter_min` | `Int_t` | minimum frame counter value |
| `frame_counter_max` | `Int_t` | maximum frame counter value |
| `adc_bins` | `Int_t` | number of ADC histogram bins |
| `adc_min` | `Int_t` | ADC histogram lower edge |
| `adc_max` | `Int_t` | ADC histogram upper edge |
| `ch_bins` | `Int_t` | number of channel histogram bins |
| `ch_min` | `Int_t` | channel histogram lower edge |
| `ch_max` | `Int_t` | channel histogram upper edge |
| `time_bins` | `Int_t` | number of time histogram bins |
| `bcid_bins` | `Int_t` | number of BCID histogram bins |
| `bcid_min` | `Int_t` | BCID histogram lower edge |
| `bcid_max` | `Int_t` | BCID histogram upper edge |
| `tdc_bins` | `Int_t` | number of TDC histogram bins |
| `tdc_min` | `Int_t` | TDC histogram lower edge |
| `tdc_max` | `Int_t` | TDC histogram upper edge |
| `offset_bins` | `Int_t` | number of offset histogram bins |
| `offset_min` | `Int_t` | offset histogram lower edge |
| `offset_max` | `Int_t` | offset histogram upper edge |
| `hits_per_vmm[32]` | `Int_t[32]` | hits per VMM slot; `-1` if VMM not present |

##### `hits` TTree (1 entry per hit)

Contains the full raw hit data for every parsed hit — use this for custom
analysis without re-parsing the pcapng:

| branch | ROOT type | range | description |
|---|---|---|---|
| `fec` | `UChar_t` | 1–15 | FEC board ID (upper nibble of SRS header `dataId`, byte 7) |
| `vmm` | `UChar_t` | 0–31 | VMM chip ID |
| `ch` | `UChar_t` | 0–63 | channel number |
| `adc` | `UShort_t` | 0–1023 | raw ADC value |
| `adc_calibrated` | `Double_t` | — | calibrated ADC (`adc_slope × adc + adc_offset`); equals `adc` without `--calibration` |
| `over_threshold` | `UChar_t` | 0/1 | over-threshold flag |
| `time` | `UInt_t` | 0–2³²−1 | frame counter (SRS header bytes 0–3) |
| `offset` | `Char_t` | −16–+15 | 5-bit signed timing offset (BCID sweeps since last SRS marker) |
| `bcid` | `UShort_t` | 0–4095 | bunch-crossing ID (Gray-decoded) |
| `tdc` | `UChar_t` | 0–255 | TDC fine-timing value |
| `timestamp_ns` | `Double_t` | — | chip_time relative to most recent SRS marker (ns) |
| `srs_timestamp` | `Double_t` | — | 42-bit SRS marker FEC timestamp for this VMM (25 ns ticks; 0 before first marker) |
| `abs_time_ns` | `Double_t` | — | absolute hit time = `srs_timestamp × 25 + timestamp_ns` (ns) |
| `trigger_time` | `Double_t` | — | external trigger timestamp (TRG format only; 0 in SRS mode) |
| `trigger_counter` | `UShort_t` | — | trigger event counter (TRG format only; 0 in SRS mode) |
| `hit_valid` | `UChar_t` | 0/1 | 1 if offset is in the valid range [−1, +15] for the new SRS format; 0 otherwise |

> **Deprecated SRS header fields — not present in output.**
> The SRS header contains two additional 32-bit fields: `udpTimeStamp` (bytes 8–11,
> intended as a per-packet FEC hardware timestamp) and `offsetOverflow` (bytes 12–15,
> intended as a running BCID rollover counter). Both are explicitly marked as
> *"will vanish soon and be replaced by a timestamp for each VMM"* in the vmm-sdat
> `ParserSRS.cpp` source (ESS detector group). Current FEC firmware does not populate
> either field: `udpTimeStamp` is always 0 and `offsetOverflow` is set once at
> boot and never updated. They carry no timing information and are excluded from
> the DataFrame and ROOT output.

Example usage in a ROOT macro:

```cpp
TFile *f = TFile::Open("run.root");

// Reproduce a histogram with the saved binning
TTree *p = (TTree*)f->Get("hits_parameters");
int adc_bins, adc_min, adc_max;
p->SetBranchAddress("adc_bins", &adc_bins);
p->SetBranchAddress("adc_min",  &adc_min);
p->SetBranchAddress("adc_max",  &adc_max);
p->GetEntry(0);

// Draw ADC for over-threshold hits on VMM 3
TTree *hits = (TTree*)f->Get("hits");
hits->Draw(Form("adc>>h(%d,%d,%d)", adc_bins, adc_min, adc_max),
           "over_threshold==1 && vmm==3");
```

##### Per-VMM directories (`vmm<N>/`)

| object | type | content |
|---|---|---|
| `adc` | `TH1F` | ADC distribution, all hits |
| `adc_ot` | `TH1F` | ADC distribution, OT hits only |
| `adc_not_ot` | `TH1F` | ADC distribution, non-OT hits only |
| `ot_flag` | `TH1F` | Not-OT vs OT hit counts |
| `ch_occ` | `TH1F` | channel occupancy |
| `time` | `TH1F` | hit rate vs frame counter |
| `adc_vs_ch` | `TH2F` | ADC (y) vs channel (x) — same data as the 2-D PNG |
| `bcid` | `TH1F` | BCID distribution |
| `tdc` | `TH1F` | TDC distribution |
| `offset` | `TH1F` | offset distribution (signed) |

---

## Memory efficiency

The script is designed to handle large captures (1 GB+, ~100 M hits) without
blowing up RAM. Five specific measures keep the footprint under control:

### 1. Bit parsing via `struct` + bitwise operations

`parse_block` unpacks the 6-byte hit words with `struct.unpack_from('>IH', ...)`
and extracts fields with bitwise shifts and masks, e.g.:

```python
d1, d2 = struct.unpack_from('>IH', block, i + 16)
if d2 & 0x8000:                      # valid-hit flag
    vmm = (d1 >> 22) & 0x1F
    adc = (d1 >> 12) & 0x3FF
    ch  = (d2 >>  8) & 0x3F
    ot  = (d2 >> 14) & 0x1
```

The previous implementation formatted each word as a binary string
(`"{:032b}".format(...)`), creating two temporary Python string objects per hit.
At 100 M hits that is 200 M short-lived strings generating constant GC pressure.
The bitwise approach allocates nothing.

### 2. Array buffers freed after DataFrame construction

Hits are accumulated during parsing into six compact `array.array` buffers
(1–4 bytes per element, no Python object overhead). Once the pandas DataFrame
is built from those buffers, the originals are immediately released:

```python
hits = pd.DataFrame({...})
del fec_buf, vmm_buf, time_buf, ch_buf, adc_buf, ot_buf, offset_buf, bcid_buf, tdc_buf
```

This frees roughly one full copy of the hit data (~1 GB at 100 M hits) as soon
as the DataFrame exists.

### 3. Matplotlib figures closed after saving

Each figure is saved and immediately closed with `plt.close(fig)` via a small
helper:

```python
def _save(fig, fname):
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
```

The per-VMM time-rate figures are also built, saved, and closed one at a time
rather than accumulated in a dict. For 32 VMMs, this prevents dozens of large
canvas objects from coexisting in memory.

### 4. Chunked ROOT filling + per-VMM data released

`_fill1d` fills ROOT histograms in chunks of 500 000 entries, capping the
temporary `float64` working array at ~4 MB regardless of how many hits a VMM has:

```python
_CHUNK   = 500_000
_w_chunk = np.ones(_CHUNK, dtype=np.float64)   # reused across all calls

def _fill1d(h, arr):
    for start in range(0, len(arr), _CHUNK):
        a = np.ascontiguousarray(arr[start:start + _CHUNK], dtype=np.float64)
        h.FillN(len(a), a, _w_chunk[:len(a)])
```

After all histograms for a VMM are written, the per-VMM DataFrame slice and its
derived arrays are explicitly freed:

```python
del vdata, adc_all, adc_ot, adc_not_ot
```

### 5. Cling-compiled C++ filler for the `hits` TTree

Writing one row per hit into a ROOT TTree from Python would require a Python
`for` loop calling `tree.Fill()` at every iteration — roughly 10–60 minutes for
100 M hits. Instead, the script declares a small C++ function at startup via
ROOT's Cling JIT compiler:

```python
ROOT.gInterpreter.Declare("""
void _vmm_fill_hits(TTree* t,
                    const unsigned char*  fec_a, ...,
                    long long n)
{
    // branch setup ...
    for (long long i = 0; i < n; ++i) { ...; t->Fill(); }
}
""")
```

The numpy arrays (already in compact dtypes: `uint8`, `uint16`, `uint32`) are
passed directly as typed C++ pointers — no upcasting, no copies. The loop runs
as compiled C++, so 100 M hits fill in seconds rather than hours.

---

## Requirements

The script uses PyROOT to write a `.root` output file. The ROOT installation at
`/local/home/ak271430/Software/root` was built against **Python 3.12**, so the
script must be run with Python 3.12 — not the system default 3.13.

A dedicated conda environment (`root312`) is set up with all required packages:

```
python = 3.12
numpy
pandas
matplotlib
scapy
ROOT  (via sourcing thisroot.sh, see Usage below)
```

To recreate the environment from scratch:

```bash
conda create -n root312 python=3.12 pandas numpy matplotlib scapy -y
```

---

## Usage

Because ROOT must be on the Python path, always source `thisroot.sh` and invoke
the `root312` conda environment explicitly:

```bash
source /local/home/ak271430/Software/root/bin/thisroot.sh && \
/local/home/ak271430/miniconda3/envs/root312/bin/python3.12 \
    vmm_hybrid_pcapng_monitoring.py <path/to/capture.pcapng>
```

### Example

```bash
source /local/home/ak271430/Software/root/bin/thisroot.sh && \
/local/home/ak271430/miniconda3/envs/root312/bin/python3.12 \
    vmm_hybrid_pcapng_monitoring.py /data/run042/hybrid_run042.pcapng
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

Saved 11 files in: /data/run042/qa_plots/hybrid_run042/
ROOT file: /data/run042/qa_plots/hybrid_run042/hybrid_run042.root
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
| `offset` | int8 | 5-bit signed timing offset (−16 to +15) |
| `bcid` | uint16 | bunch-crossing ID (0–4095) |
| `tdc` | uint8 | TDC fine-timing value (0–255) |

This DataFrame is available in memory after the script finishes, so the script
can be imported and extended for custom analysis.