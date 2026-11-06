# Fe55 rate map, source localisation and gain map — det1, 9 VMM hybrids

Answers, from one or more pcapng captures:

1. **Where is the source?** per-pad rate in Hz and Hz/cm² on the real fan tiles,
   a 2D Gaussian fit to the spot, and the vector to move the source by.
2. **How much of the spot am I seeing?** the fraction of the fitted spot landing
   on instrumented pads, so a clipped spot is not read as a dim one.
3. **What is the gain there?** per-pad Fe55 photopeak in ADC and relative gain.
4. **Is the data trustworthy?** cabling cross-check, saturation, threshold bias,
   hot pads, live-time provenance.

## Files

| File | What it is |
|---|---|
| `vmm_det1_cabling.py` | `(vmm, ch) → pad` for this bench cabling, plus live time from the pcapng packet clock |
| `vmm_fe55_map.py` | the analysis — rate map, source fit, gain map, spectra |
| `vmm_fe55_merge.py` | stitches per-pad gains from several source positions into one map |
| `vmm_fe55_selftest.py` | synthetic data with a known answer; five scenarios |

They sit beside `vmm_stations.py` and share `P2_BASKET_mapping.csv`, but are
independent of it: `vmm_stations` is the three-station SPS telescope that the
live beam QA runs on and is deliberately left untouched.

## Cabling this assumes

Recorded in `vmm_det1_cabling.py`, confirmed 2026-09-09:

```
hybrid n  ->  connector n  ->  VMM 2n bottom / 2n+1 top
   hybrid 1 -> connector 1 -> VMM 2 / 3
   ...
   hybrid 9 -> connector 9 -> VMM 18 / 19
connectors 1 and 9 DISCONNECTED   -> sectors 0 and 8 not read out
connector 10 (sector 9)           -> not cabled in this setup
hybrid 0 (VMM 0/1)                -> not populated; absent from the data
```

Live: **sectors 1–7, connectors 2–8, VMM 4–17, 896 of the 1280 pads.**

MEASURED on the first real capture, not assumed. The initial guess was
`VMM_BASE = 0` ("vmm0 to the bottom connector") — that phrase turns out to
describe the *local* chip index on the hybrid, not the global VMM id, and taking
it literally shifted every connector by one sector. The two disconnected
connectors are what caught it: base 0 predicts VMM 0,1 and 16,17 silent, base 2
predicts VMM 2,3 and 18,19 silent, and the data shows the latter
(VMM 2,3 = 0.01 % of hits; VMM 18,19 = 0.03 %; VMM 4–17 = 34k–55k each).

`sector = connector_N - 1` and `channel_id = sector*128 + (strip-1)`, identical
to `vmm_stations.py` and `cosmic_bench_analysis/p2_mapping.py`, so a pad number
means the same thing in all three.

The **two disconnected connectors are a free calibration of the numbering
convention**: each candidate convention predicts a different pair of silent VMM
ids, so the first file settles it. The cabling check runs on every analysis and
prints `OK` or `MISMATCH` with the offending VMM ids.

## Running it

```bash
# one capture, a sub-run / run directory (searched recursively), or a glob
python3 vmm_fe55_map.py <file.pcapng | dir | 'runs/run_3/**/*.pcapng'>

# tell it where you want the spot; get the move vector in x/y and r/phi
python3 vmm_fe55_map.py <input> --target 300 250

# subtract a source-out run (rate only — never the gain map, see below)
python3 vmm_fe55_map.py <input> --background <source_out_dir>

# confirm the within-half wiring on the first real file
python3 vmm_fe55_map.py <input> --scan-mapping

# keep re-rendering as captures land, while you move the source
python3 vmm_fe55_map.py <run_dir> --watch
```

Products per run: `_rate_map.png`, `_source_profile.png`, `_gain_map.png`,
`_spectra.png`, `_rate_vs_channel.png`, `_pads.csv` (one row per pad),
`_summary.json`.

`_rate_vs_channel.png` is the ELECTRONICS-space companion to the rate map: a
VMM x channel heatmap of the whole readout, then one panel per connector with
the two VMMs (bottom = strips 1-64, top = 65-128) overlaid on a shared y-scale.
Dead channels are marked, hot ones circled. The two views are not
interchangeable — the strip-to-pad mapping is an interleaved permutation, so a
defect that is contiguous in channel number is scattered across the fan and
essentially invisible there. `_summary.json` carries the same information as a
`channels` census, with consecutive low channels grouped into `contiguous_blocks`
and warned about separately: scattered single channels are attrition, a block is
a connector or ribbon fault.

Stitching several source positions:

```bash
python3 vmm_fe55_merge.py pos*_pads.csv --out-dir merged --name det1_gain
```

`_coverage.png` from the merge is the practical one: green = measured,
**red = still to do**, i.e. where to put the source next.

## Design decisions worth knowing

**No event building, and none wanted.** Pads are ~12 × 11.6 mm (130–144 mm²)
and a 5.9 keV conversion is a point on that scale, so essentially all the charge
lands on one pad: the single-pad ADC spectrum *is* the Fe55 spectrum. This is
not just convenient — `vmm_stations.py` documents that `offset` is stuck in this
firmware, so chip time is only unambiguous within 92.16 µs of an SRS marker, and
time-clustering across VMMs would be built on sand. Counting hits per pad and
histogramming their ADC needs no timestamps at all.

What single-pad spectra do miss is charge sharing on a pad seam: such an event
splits between two pads and lands in both spectra *below* the photopeak. That
is a low-ADC shoulder, not a shift of the peak, and the windowed-median
estimator ignores it by construction — so those events are dropped from the
gain estimate rather than mis-measured. Recovering them would need exactly the
cross-pad time clustering the marker caveat rules out, so it is not implemented.

**Live time comes from the pcapng packet clock**, with `if_tsresol` parsed from
the Interface Description Block rather than assumed — the DAQ's dumpcap writes
nanoseconds while the pcapng default is microseconds, which would be a silent
factor-1000 on every rate. The SRS marker span is kept only as a cross-check;
measured on `data/hybrid_performance_data`, the two agree to ~10 % (19.9 s vs
22.0 s on the same file), close enough to catch a gross error and far too loose
to normalise with.

Files handed in twice are counted **once** — the live time is the union of the
capture intervals, not the sum of their spans, so a duplicated glob cannot halve
every rate.

**The gain map is never background-subtracted.** `--background` subtracts rate
only. A photopeak is a *position* in ADC, not a count; subtracting spectra would
distort it.

**Hot pads are judged against their spatial neighbours**, not against a per-VMM
median. Under a collimated source the whole point of the run is a bright spot on
a dark detector, so a global comparison would flag the signal. Same 6× threshold
as `vmm_stations.auto_hot_pads`.

**The mapping scan reports what it could not determine.** Speckle cannot judge a
half the source never lit — the greedy step would still return a winner and it
would be noise. Every half is classified on two counts (enough hits, and a large
enough roughness gap between best and second-best ordering), and only determined
halves produce an override. To settle the whole fan, move the source over each
connector in turn and run `--scan-mapping` on the merged set of captures.

**Stitching solves for run scales and pad gains together.** Modelling
`peak(r,p) = s_r · g_p` and taking logs makes it a least-squares problem tying
runs together through the pads they *share* — so overlap between neighbouring
source positions is not optional, it is the only thing fixing their relative
scale. Runs sharing nothing with the rest are reported, not silently floated.

## Validation

`vmm_fe55_selftest.py` writes a real pcapng (SHB/IDB/EPB, Ethernet/IPv4/UDP,
SRS VM3 payloads with the bit layout `vmm_decode.py` unpacks), so it exercises
the actual decoder rather than a mock, and checks five scenarios:

```bash
python3 vmm_fe55_selftest.py              # all scenarios
python3 vmm_fe55_selftest.py --scenario nominal
```

| Scenario | What it proves |
|---|---|
| `nominal` | position, spot size, photopeak, resolution, rate, live time and the injected +10 %/100 mm gain gradient all come back |
| `clipped` | a spot on the edge of the live sectors is *flagged*, not reported as a confident centre |
| `wrongmap` | generated through `reverse`, scanned from `linear` — the scan recovers `reverse` with no false positives on unlit halves |
| `cabling` | a different pair of disconnected connectors than declared is detected |
| `merge` | four positions with four different run scales → the gain field is recovered over their union and the conditions absorbed into the scales |

Last full run: **all five passed** — position to 0.07 mm, photopeak to 0.01 ADC,
gain map r = 0.96, stitch over 379 pads (best single position 154) at
r = 0.994 with run scales to 0.07 %.

## What the first real data changed

Synthetic data cannot invent a failure nobody thought of. The 2026-09-09 Fe55
capture (54 files, 32.8 M hits, 3239.6 s) drove four fixes:

* **Projections are area-normalised.** They summed pad rates per bin, so the
  fan's pad-count distribution swamped the source and the radial turnover that
  is plainly in the data was invisible. This one changed a conclusion, not just
  a plot.
* **Fit quality is checked.** The 2D Gaussian converged at chi2/ndf = 35500 and
  still reported a centre to 0.1 mm — 166 mm from the centroid, and unstable at
  the 100 mm level between one file and 54. Above chi2/ndf = 10 the fit is
  rejected and **the MOVE vector falls back to the rate-weighted centroid**,
  which agrees to 0.4 mm between one file and all 54. Containment, being derived
  from the same Gaussian, is suppressed when the fit is rejected.
* **"Is it actually a peak?"** A Gaussian fits a flat continuum happily and
  reported "peak 575 ADC" on a spectrum with no peak at all. Now warned via
  FWHM > 40 % (an Fe55 line is ~20 %), with a prominence metric recorded too.
* **Broad vs absent illumination** are distinguished. The first wording claimed
  there was no source when there was one — just very broad.

## Deploying to the DAQ machine

The beam DAQ runs its **own copy** under `DAQ_Control_VMM_Beam/vmm_qa/` — that
directory is not this checkout. To use these there, copy all four modules plus
`P2_BASKET_mapping.csv` into it and run with that repo's venv:

```bash
cp vmm_det1_cabling.py vmm_fe55_map.py vmm_fe55_merge.py vmm_fe55_selftest.py \
   ../DAQ_Control_VMM_Beam/vmm_qa/
../DAQ_Control_VMM_Beam/.venv/bin/python \
   ../DAQ_Control_VMM_Beam/vmm_qa/vmm_fe55_map.py <run_dir> --watch
```

Nothing here touches `vmm_pcapng_qa.py`, `vmm_stations.py` or `qa_watcher.py`,
so the live beam QA is unaffected either way.

## First real file — suggested order

1. `vmm_fe55_map.py <capture> --max-packets 2000` — quick look. Check the
   cabling line says `OK`. If it says `MISMATCH`, the numbering convention or
   the `DISCONNECTED` list is wrong; the message names the VMM ids.
2. Look at `_spectra.png`. If there is no Ar escape peak at 0.50 × the main
   peak, what you have is probably a threshold turn-on, not the photopeak —
   raise the mesh or lower the threshold before mapping anything.
3. `--scan-mapping` to confirm the within-half ordering on the lit connectors.
4. Full run without `--max-packets`, with `--target` if you have one in mind.
