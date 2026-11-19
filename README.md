# P2_basket_online_analysis — retired

This repository is **archived**. It held the first online/offline analysis of
VMM3a/SRS pcapng captures for the P2 basket (decoder, beam QA, Fe55 tools). That
work has been split into dedicated repositories, and nothing reads from here
any more.

**The last full state is the tag [`legacy-final`](https://github.com/akallitss/P2_basket_online_analysis/tree/legacy-final)**
(commit `0f3043e`, 2026-11-06). Check it out to get any of the old files:

```bash
git clone https://github.com/akallitss/P2_basket_online_analysis.git
cd P2_basket_online_analysis && git checkout legacy-final
```

## Where everything went

| Was here | Now |
|---|---|
| `vmm_decode.py` — pcapng → hits | [p2-decode](https://github.com/akallitss/p2-decode) (`legacy_word_count=True` reproduces this decoder bit for bit) |
| `vmm_stations.py`, `P2_BASKET_mapping.csv` — (vmm, ch) → pad | [p2-mapping](https://github.com/akallitss/p2-mapping) (setups + the same pad geometry) |
| `vmm_reduce.py`, `vmm_root_io.py`, `vmm_pcapng_qa.py`, `vmm_trend.py` — the online processing / QA split | [p2-monitor](https://github.com/akallitss/p2-monitor) (`p2-monitor-reduce`, `p2-monitor-qa`, `p2-monitor live`) |
| `vmm_efficiency.py`, `vmm_beam_profile.py`, `vmm_map_*.py`, `vmm_scan_curves.py` — beam QA plots | [DAQ_Control_VMM_Beam](https://github.com/akallitss/DAQ_Control_VMM_Beam) `vmm_qa/` (newer copies) |
| `vmm_fe55_*.py`, `vmm_det1_cabling.py`, `README_fe55_gain_map.md` — Fe55 rate / gain maps | [P2_basket_analysis](https://github.com/akallitss/P2_basket_analysis) `fe55_analysis/`, ported to p2-decode + p2-mapping |
| `FEC_*.py` — colleagues' first readers | [P2_basket_analysis](https://github.com/akallitss/P2_basket_analysis) `fec_colleague_scripts/` (reference only) |
| `chd_toy_montecarlo.py` | [P2_basket_analysis](https://github.com/akallitss/P2_basket_analysis) `chd_toy_mc/` |
| `vmm_srs_dataflow_schematic.py` + schematic PNGs | [cern-site](https://github.com/akallitss/cern-site) note 2026-10-07 (VMM3a/SRS data-flow schematics) |
| `vmm_hybrid_pcapng_monitoring.py` — all-in-one offline ROOT + PNG QA | only in `legacy-final`; anything similar should be rebuilt on p2-decode / p2-mapping / p2-monitor |
| `validate_vmm_decode.py` | superseded by p2-decode's `tests/test_legacy.py` |

The p2-* packages' `tests/test_legacy.py` compare against the files in
`legacy-final` (`P2_DECODE_LEGACY`, `P2_MONITOR_LEGACY` point at a checkout of
that tag).
