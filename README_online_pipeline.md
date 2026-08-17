# Online pipeline — processing split from QA plotting

Rendering QA plots straight off every capture is what runs the beam machine out
of memory: a 45 s capture holds ~2.3 M hits, the per-hit table is ~3.4x the
pcapng, and drawing 36 PNGs per file costs ~43 s to produce images nobody opens.

So the work is split in two. **Processing** decodes a capture once and throws the
per-hit data away, keeping only additive histograms. **Plotting** reads those and
never touches a pcapng.

```
raw_daq_data/<capture>.pcapng
        |
        |  vmm_reduce.py          <- processing (Tier 1)
        v
hits_store/<capture>/
        counts.npz     ~0.2 MB    per-VMM histograms, ADDITIVE across files
        counts.root    ~0.4 MB    the same histograms as TH1D/TH2D  (--root)
        scalars.json   ~4 kB      the numbers the trend dashboard plots
        *.npy                     per-hit columns, optional (--drop-columns)
        |
        |  vmm_pcapng_qa.py       <- plotting
        v
36 PNGs + efficiency + beam profile
```

Because the histograms are additive, merging N captures is a sum of arrays. That
is how a quiet VMM gets a usable ADC spectrum (a few hundred hits per file
becomes thousands over a sub-run) for less work than drawing one bad plot per
file.

## Where the code lives

Everything in this repo; `DAQ_Control_VMM_Beam` imports it rather than vendoring
a copy. On the beam machine the link is:

```
P2_basket_online_analysis  (this repo)
  -> cloned to /local/p2/p2soft/vmm-analysis
     -> .venv/lib/python3.12/site-packages/vmm_analysis.pth puts it on sys.path
        -> DAQ_Control_VMM_Beam/{vmm_processor_watcher,qa_watcher}.py
```

The watchers stay in the DAQ repo — they are orchestration (polling, run
filtering, memory kill, atomic handoff). Everything they invoke lives here:

| Module | Role |
|---|---|
| `vmm_decode.py` | vectorised pcapng decoder; no scapy, no per-word Python loop |
| `vmm_reduce.py` | capture -> counts + scalars (the processing half) |
| `vmm_root_io.py` | counts <-> ROOT TH1D/TH2D (`--root`) |
| `vmm_pcapng_qa.py` | store -> PNGs (the plotting half) |
| `vmm_stations.py` | station cabling, channel masking, `P2_BASKET_mapping.csv` |
| `vmm_efficiency.py` | trigger-referenced efficiency |
| `vmm_trend.py` | scalars of every capture vs time |
| `vmm_beam_profile.py` | beam position/width per station |
| `vmm_hybrid_pcapng_monitoring.py` | the original all-in-one ROOT + PNG QA |

## Running it

```bash
# processing — one capture
python vmm_reduce.py <capture.pcapng> --store-dir <store>/<capture>

# ... also writing ROOT histograms (needs ROOT; off by default)
python vmm_reduce.py <capture.pcapng> --store-dir <store>/<capture> --root

# ... keeping only the histograms, not the per-hit columns
python vmm_reduce.py <capture.pcapng> --store-dir <store>/<capture> --drop-columns

# plotting — from the store, never from the pcapng
python vmm_pcapng_qa.py <store>/<capture> --out-dir <plots>/<capture>
```

`--root` is off by default on purpose: the DAQ venv at the beam has no ROOT,
which is why `vmm_pcapng_qa.py` is PNG-only. `vmm_root_io` is imported lazily so
`import ROOT` never happens on the online path, and a ROOT failure is caught and
recorded in `scalars.json` as `root_error` rather than losing the decode.

### Merging

```bash
hadd merged.root <store>/*/counts.root          # fast path
```

`hadd` sums the histograms correctly. It does **not** merge `time_range` (a
TVectorD with no `Merge()`, so the first input's span wins) and it can only sum
`h_time_profile` when every input shares the same frame-counter axis. When those
matter, use the numpy semantics instead:

```python
import vmm_root_io
vmm_root_io.merge_root(glob.glob("store/*/counts.root"), "merged.root")
```

## Environment

The online path needs only numpy / pandas / matplotlib / scapy. The `--root`
path additionally needs a ROOT whose Python version matches the interpreter —
`root-config --python3-version`. On `dphppcj15` that is ROOT 6.36 built against
python 3.12, so:

```bash
python3.12 -m venv --system-site-packages .venv312
./.venv312/bin/pip install pandas matplotlib scapy
```

## Known issue: the frame counter is constant

`time` (SRS header bytes 0-3, decoded as the frame counter) reads **20 on every
hit of every capture** in the July 2026 H4 data. `time_range` is therefore
`[20, 20]`, and `time_profile` — the rate-vs-frame-counter plot — puts every hit
in one bin.

This is not a regression from the split: the reduce output here is bit-identical
to what the beam machine produced for the same capture, `time_profile` included.
But it means the rate-vs-time plots carry no information on this data, and
whether the field is misinterpreted or simply not populated by the firmware is
still open.
