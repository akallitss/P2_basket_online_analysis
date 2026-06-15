#!/usr/bin/env python3
"""
VMM3a / SRS readout system — data-flow schematic.
Saves four separate high-resolution PNGs:
  schematic_1_hardware_chain.png
  schematic_2_vmm3a_hit_processing.png
  schematic_3_clock_timeline.png
  schematic_4_timestamp_reconstruction.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ── palette ───────────────────────────────────────────────────────────────────
BLU = '#1D4ED8'   # hardware
VIO = '#6D28D9'   # offset / BCID cycles
AMB = '#B45309'   # clock / BCID
RED = '#B91C1C'   # hit
GRN = '#047857'   # marker
CYN = '#0E7490'   # abs time / packet
PIN = '#9D174D'   # TDC
DRK = '#1E293B'   # dark boxes
GRY = '#475569'   # arrows
LBG = '#F1F5F9'   # background
WHT = 'white'
DPI = 220

# ── shared helpers ────────────────────────────────────────────────────────────
def new_fig(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=LBG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(LBG)
    fig.text(0.5, 0.97, title, ha='center', va='top',
             fontsize=14, fontweight='bold', color='#0F172A')
    return fig, ax

def rbox(ax, cx, cy, w, h, line1, line2='', line3='',
         color=BLU, fs=10, lh=0.22):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle='round,pad=0.018',
                       facecolor=color, edgecolor=WHT,
                       linewidth=2.5, zorder=3)
    ax.add_patch(r)
    lines = [l for l in [line1, line2, line3] if l]
    n = len(lines)
    for i, txt in enumerate(lines):
        yo = (n - 1) / 2 * lh - i * lh
        ax.text(cx, cy + yo * h, txt,
                ha='center', va='center',
                fontsize=fs, color=WHT, fontweight='bold', zorder=4)

def arrow(ax, x0, y0, x1, y1, color=GRY, lw=2.0, rad=0.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))

def brace_h(ax, x0, x1, y, label, color=VIO, fs=9):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='<->', color=color, lw=2.0))
    ax.text((x0 + x1) / 2, y - 0.03, label,
            ha='center', va='top', fontsize=fs, color=color, fontweight='bold')

OUT = '/local/home/ak271430/Documents/PostDocSaclay/P2_basket_online_analysis/'

# ══════════════════════════════════════════════════════════════════════════════
# 1 · HARDWARE CHAIN
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = new_fig(22, 6,
    '1 · Hardware chain — from detector strip to data file')

items = [
    (0.07,  'Detector\nstrip',   '',                  BLU),
    (0.21,  'VMM3a\nASIC × N',  '64 channels / chip', VIO),
    (0.35,  'Hybrid\nPCB',      'strips → chips',     BLU),
    (0.50,  'DVM\ncard',        'adapter board',      BLU),
    (0.65,  'FEC  FPGA',        'Spartan7 · 44.44 MHz', GRN),
    (0.80,  'UDP packet',       'port 6006 · ~9 kB',  CYN),
    (0.93,  'PC\nanalysis',     '.pcapng file',       DRK),
]
for cx, t1, t2, c in items:
    rbox(ax, cx, 0.52, 0.115, 0.56, t1, t2, color=c, fs=10)
for i in range(len(items) - 1):
    arrow(ax, items[i][0] + 0.064, 0.52,
               items[i+1][0] - 0.064, 0.52, color=GRY, lw=2.5)

# role annotations
roles = [
    (0.07,  'Generates charge\nwhen particle passes'),
    (0.21,  'Measures time & charge\nof every hit'),
    (0.35,  'Physical support\n& routing'),
    (0.50,  'Links hybrid\nto FEC'),
    (0.65,  'Concentrates all data\npackages & timestamps'),
    (0.80,  'One packet\nper readout frame'),
    (0.93,  'Stores raw packets\nfor offline analysis'),
]
for cx, txt in roles:
    ax.text(cx, 0.14, txt, ha='center', va='center',
            fontsize=8, color='#334155', style='italic')

ax.text(0.5, 0.03,
        'The FEC adds a 16-byte SRS header to each packet: frame counter · FEC ID · '
        'UDP timestamp (deprecated) · overflow count (deprecated)',
        ha='center', va='bottom', fontsize=8.5, color='#64748B', style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT + 'schematic_1_hardware_chain.png',
            dpi=DPI, bbox_inches='tight', facecolor=LBG)
print('Saved schematic_1_hardware_chain.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 2 · VMM3a HIT PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = new_fig(22, 11,
    '2 · Inside the VMM3a ASIC — what happens when a hit arrives')

# top signal chain
chain = [
    (0.07,  0.80, 'Signal\non strip',    '',                      BLU),
    (0.21,  0.80, 'Amplifier\n+ Shaper', 'smooth pulse',          VIO),
    (0.37,  0.80, 'Discriminator',       'threshold\ncrossing',   VIO),
    (0.55,  0.80, 'Peak Finder',         'detects pulse\nmaximum',VIO),
    (0.71,  0.80, 'ADC',                 'peak height\n= charge', GRN),
    (0.88,  0.80, '6-byte\nhit word',    'buffered in VMM3a\nuntil FEC reads', RED),
]
for cx, cy, t1, t2, c in chain:
    rbox(ax, cx, cy, 0.12, 0.24, t1, t2, color=c, fs=10)
for i in range(len(chain) - 1):
    arrow(ax, chain[i][0] + 0.066, 0.80,
               chain[i+1][0] - 0.066, 0.80, lw=2.2)

# ── branch: Discriminator → Latch BCID ───────────────────────────────────────
arrow(ax, 0.37, 0.68, 0.37, 0.52, color=AMB, lw=2.2)
rbox(ax, 0.37, 0.38, 0.18, 0.25,
     'Latch BCID',
     '→ coarse time',
     '(22.5 ns / bin)',
     color=AMB, fs=10)
ax.text(0.37, 0.20,
        'Captures which 22.5 ns clock bin\nthe particle arrived in\n(12-bit counter, 0 – 4095)',
        ha='center', va='center', fontsize=9, color=AMB, style='italic')

# ── branch: Peak Finder → TAC → TDC ─────────────────────────────────────────
arrow(ax, 0.55, 0.68, 0.55, 0.52, color=PIN, lw=2.2)
rbox(ax, 0.55, 0.38, 0.18, 0.25,
     'TAC → TDC',
     'start: peak found',
     'stop: next clock edge',
     color=PIN, fs=10)
ax.text(0.55, 0.20,
        'Measures fine time WITHIN the clock bin\n8-bit value (0–255)\n0.23 ns per count · full range = 60 ns',
        ha='center', va='center', fontsize=9, color=PIN, style='italic')

# ── arrows into hit word ──────────────────────────────────────────────────────
arrow(ax, 0.46, 0.38, 0.81, 0.68, color='#94A3B8', lw=1.5, rad=-0.22)
arrow(ax, 0.64, 0.38, 0.83, 0.68, color='#94A3B8', lw=1.5, rad=-0.12)
arrow(ax, 0.71, 0.68, 0.82, 0.68, color='#94A3B8', lw=2.0)

# ── hit word contents ─────────────────────────────────────────────────────────
ax.text(0.88, 0.49,
        'Hit word contents:\n'
        '━━━━━━━━━━━━━━━━━━━━━\n'
        'VMM ID        (5-bit)\n'
        'Channel       (6-bit, 0–63)\n'
        'BCID          (12-bit, Gray coded)\n'
        'ADC           (10-bit, 0–1023)\n'
        'Over-threshold flag   (1-bit)\n'
        'TDC           (8-bit, 0–255)\n'
        'Offset        (5-bit signed)',
        ha='center', va='center', fontsize=9.5, color=WHT, zorder=5,
        linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=RED,
                  alpha=0.92, edgecolor=WHT, lw=1.5))

# ── note on offset ────────────────────────────────────────────────────────────
ax.text(0.5, 0.045,
        'The offset field counts how many full BCID cycles (≈ 92 µs each) have passed since the last SRS marker '
        'sent by the FEC to this VMM.\n'
        'Valid range for new SRS format: −1 to +15.  '
        'Offset = −1 means the hit was latched just before a BCID rollover (normal pipeline effect, not an error).',
        ha='center', va='bottom', fontsize=9, color='#334155', style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT + 'schematic_2_vmm3a_hit_processing.png',
            dpi=DPI, bbox_inches='tight', facecolor=LBG)
print('Saved schematic_2_vmm3a_hit_processing.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 3 · CLOCK / MARKER / OFFSET / TDC TIMELINE  (2-panel: overview + zoom)
# ══════════════════════════════════════════════════════════════════════════════
BCID_EX      = 2000
T_BC         = 22.5                    # ns/tick at 44.44 MHz
bcid_ns      = BCID_EX * T_BC         # 45 000 ns

# ── Realistic zoom example ─────────────────────────────────────────────────────
# Shaping time = 90 ns = 4 × T_BC (chip ST register, range 25–200 ns).
# Threshold crosses at Z_hit ns into BCID=2000 → BCID=2000 latched.
# Peak arrives shaping_ns later → in BCID=2004.
# TAC starts at peak, stops at BC edge closing BCID=2004.
# Formula chip_time=(BCID_thr+1)×T_BC−TDC×60/256 gives t_threshold exactly
# here because shaping_ns is an exact multiple of T_BC (ε=0); general error < T_BC.
shaping_ns   = 90.0                   # ns  (= 4 × T_BC)
Z_hit        = 5.0                    # threshold at 5 ns into BCID=2000
Z_pk         = Z_hit + shaping_ns     # 95.0 ns — peak in BCID=2004's window
BCID_peak    = BCID_EX + int(Z_pk / T_BC)      # 2004
Z_stop       = (int(Z_pk / T_BC) + 1) * T_BC   # 112.5 ns — end of BCID=2004
Z_tdc_ns     = Z_stop - Z_pk          # 17.5 ns
TDC_EX       = round(Z_tdc_ns * 256 / 60)      # 75
tdc_ns       = TDC_EX * 60 / 256      # 17.578 ns (rounded)
chip_ns      = (1 * 4096 + BCID_EX) * T_BC - tdc_ns   # full chip_time (with offset) — top panel
chip_ns_bcid = (BCID_EX + 1) * T_BC - tdc_ns          # BCID-only term — zoom formula box

fig3 = plt.figure(figsize=(24, 23), facecolor=LBG)
fig3.text(0.5, 0.993, '3 · The clock, SRS markers, offset, and TDC — all on one timeline',
          ha='center', va='top', fontsize=14, fontweight='bold', color='#0F172A')
gs = GridSpec(2, 1, figure=fig3, height_ratios=[1.35, 1.0],
              hspace=0.04, top=0.985, bottom=0.01, left=0.01, right=0.99)
ax  = fig3.add_subplot(gs[0])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.set_facecolor(LBG)
ax2 = fig3.add_subplot(gs[1])
ax2.set_facecolor('#F8FAFC')
for sp in ax2.spines.values(): sp.set_color('#CBD5E1')

# placeholder so the rest of the top-panel code still refers to (ax, T_BC, …)

# ── time axis ─────────────────────────────────────────────────────────────────
t_y = 0.80
ax.annotate('', xy=(0.97, t_y), xytext=(0.02, t_y),
            arrowprops=dict(arrowstyle='->', color='#1E293B', lw=3))
ax.text(0.985, t_y, 't', ha='left', va='center',
        fontsize=14, color='#1E293B', fontweight='bold')

# ── three BCID cycles ─────────────────────────────────────────────────────────
B = [0.04, 0.34, 0.64, 0.90]
cyc_col = ['#DBEAFE', '#EDE9FE', '#DBEAFE']
for i in range(3):
    x0, x1 = B[i], B[i+1]
    rect = mpatches.Rectangle((x0, 0.657), x1 - x0, 0.12,
                               facecolor=cyc_col[i], edgecolor='#93C5FD',
                               linewidth=1.5, zorder=1)
    ax.add_patch(rect)
    ax.text((x0 + x1) / 2, 0.717, 'BCID  0 → 4095\n(one cycle ≈ 92 µs)',
            ha='center', va='center', fontsize=9, color='#1E40AF')

# ── BCID rollover dashed lines — label strictly LEFT of line ──────────────────
for xr in [B[1], B[2]]:
    ax.plot([xr, xr], [0.632, 0.865], color='#93C5FD',
            lw=1.8, ls='--', zorder=2)
    ax.text(xr - 0.016, 0.625, 'BCID rollover\n(counter → 0)',
            ha='right', va='top', fontsize=7.5, color='#3B82F6')

# ── SRS Markers — "offset → 0" ABOVE the time axis ───────────────────────────
for mx in [B[0], B[1], B[2]]:
    ax.plot([mx, mx], [t_y, 0.945], color=GRN, lw=3, zorder=4)
    ax.plot(mx, 0.945, 'v', color=GRN, ms=12, zorder=5)
    ax.text(mx, 0.952,
            'SRS Marker\nsrs_timestamp\nrecorded by FEC',
            ha='center', va='bottom', fontsize=8.5,
            color=GRN, fontweight='bold')
    ax.text(mx + 0.012, t_y + 0.020, 'offset → 0',
            ha='left', va='bottom', fontsize=8,
            color=GRN, style='italic', fontweight='bold')

# ── Offset counter braces ─────────────────────────────────────────────────────
brace_h(ax, B[0]+0.005, B[1]-0.005, 0.535, 'offset = 0', color=VIO, fs=10)
brace_h(ax, B[1]+0.005, B[2]-0.005, 0.535, 'offset = 1', color=VIO, fs=10)
brace_h(ax, B[2]+0.005, B[3]-0.005, 0.535, 'offset = 2', color=VIO, fs=10)

# ── The hit (BCID=2000, ~48.8% into cycle 1 — clear of B[2]) ─────────────────
hit_x = B[1] + (BCID_EX / 4096) * (B[2] - B[1])   # ≈ 0.487
hit_y = t_y

ax.plot(hit_x, hit_y, '*', color=RED, ms=24, zorder=6)
ax.text(hit_x, hit_y + 0.065,
        f'Hit arrives  (threshold crossed)\noffset = 1  ·  BCID = {BCID_EX} latched',
        ha='center', va='bottom', fontsize=10, color=RED, fontweight='bold')
ax.text(hit_x, hit_y - 0.025,
        f'BCID = {BCID_EX}  ({bcid_ns:,.0f} ns into cycle)',
        ha='center', va='top', fontsize=8.5, color=RED,
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#FEF2F2',
                  edgecolor=RED, lw=1, alpha=0.92))

# ── peak finder fires — label above-left of diamond ──────────────────────────
pk_x  = hit_x + 0.038   # ≈ 0.525
ax.plot(pk_x, hit_y, 'D', color=PIN, ms=12, zorder=6)
ax.text(pk_x - 0.007, t_y + 0.022, 'peak found\nTAC starts',
        ha='right', va='bottom', fontsize=8.5, color=PIN, fontweight='bold')

# ── next BC clock edge — label above-right of tick ───────────────────────────
nxt_x = min(pk_x + 0.075, B[2] - 0.055)   # ≈ 0.600
ax.plot([nxt_x, nxt_x], [t_y - 0.022, t_y + 0.022], color=AMB, lw=3, zorder=4)
ax.text(nxt_x + 0.007, t_y + 0.022, 'BC edge\nTAC stops',
        ha='left', va='bottom', fontsize=8.5, color=AMB, fontweight='bold')

# TDC brace — dotted drop lines + double-headed arrow + value label
tdc_y = 0.615
ax.plot([pk_x,  pk_x],  [t_y - 0.04, tdc_y + 0.005], color=PIN, lw=1, ls=':', alpha=0.6)
ax.plot([nxt_x, nxt_x], [t_y - 0.04, tdc_y + 0.005], color=AMB, lw=1, ls=':', alpha=0.6)
ax.annotate('', xy=(nxt_x, tdc_y), xytext=(pk_x, tdc_y),
            arrowprops=dict(arrowstyle='<->', color=PIN, lw=2))
ax.text((pk_x + nxt_x) / 2, tdc_y - 0.022,
        f'TDC = {TDC_EX}  →  {TDC_EX} × 60 ns / 256 = {tdc_ns:.1f} ns',
        ha='center', va='top', fontsize=9, color=PIN, fontweight='bold')

# ── BCID brace ────────────────────────────────────────────────────────────────
brace_h(ax, B[1]+0.005, hit_x, 0.437,
        f'BCID = {BCID_EX}  →  {BCID_EX} × 22.5 ns = {bcid_ns:,.0f} ns',
        color=AMB, fs=9.5)

# ── full coarse-time brace (first marker → hit) ───────────────────────────────
ax.annotate('', xy=(hit_x, 0.365), xytext=(B[0], 0.365),
            arrowprops=dict(arrowstyle='<->', color=VIO, lw=2.2))
ax.text((B[0]+hit_x)/2, 0.335,
        f'offset=1  →  1×4096×22.5 ns + {BCID_EX}×22.5 ns  =  coarse time since last marker',
        ha='center', va='top', fontsize=9.5, color=VIO, fontweight='bold')

# ── chip_time calculation box (top panel, centred) ───────────────────────────
ax.text(0.50, 0.240,
        f'chip_time = (offset × 4096 + BCID) × 22.5 ns  −  TDC × 60 ns / 256\n'
        f'Example (BCID={BCID_EX}, TDC={TDC_EX}):  (1×4096 + {BCID_EX})×22.5 − {tdc_ns:.1f}'
        f'  =  {chip_ns:,.0f} ns  ≈  {chip_ns/1000:.2f} µs',
        ha='center', va='center', fontsize=10, color=DRK,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#E0F2FE', edgecolor=CYN, lw=1.5))

# "see zoom" note near the hit TDC annotation
ax.text(hit_x + (nxt_x - hit_x)/2, t_y + 0.112,
        '↓  see zoom panel below', ha='center', va='bottom',
        fontsize=8, color='#475569', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM PANEL (ax2) — zoom of the top-panel hit region
# Shows 5 consecutive steps from threshold crossing to chip_time formula,
# all at the correct physical timescale (5 BC bins = 112.5 ns window)
# ══════════════════════════════════════════════════════════════════════════════
N_per = 5
x_end = N_per * T_BC   # 112.5 ns — covers threshold through TAC stop

ax2.set_xlim(-18, x_end + 52)   # extends right to show TAC full-scale register bar
ax2.set_ylim(0, 1)
ax2.set_yticks([])
ax2.tick_params(bottom=False, labelbottom=False)
ax2.text(0.5, 0.985,
         'Zoom  ·  5 × 22.5 ns BC clock periods  ·  '
         'steps in order of appearance:  ① threshold  ② shaping  ③ peak  ④ TAC stop  ⑤ chip_time',
         ha='center', va='top', transform=ax2.transAxes,
         fontsize=10.5, fontweight='bold', color='#0F172A',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CBD5E1', lw=1))

# coloured BCID bands — 5 bins, violet tones
bc_band_cols = ['#F5F3FF', '#EDE9FE', '#F5F3FF', '#EDE9FE', '#F5F3FF']
for i in range(N_per):
    x0, x1 = i * T_BC, (i+1) * T_BC
    ax2.axvspan(x0, x1, alpha=0.85, color=bc_band_cols[i], zorder=0)
    ax2.text((x0+x1)/2, 0.955, f'BCID = {BCID_EX + i}',
             ha='center', va='top', fontsize=10, color=VIO, fontweight='bold')
ax2.text(0.99, 0.985, 'all within  offset = 1',
         ha='right', va='top', transform=ax2.transAxes,
         fontsize=9, color=VIO, style='italic', fontweight='bold')

t_axis_y = 0.70

# ── SRS marker — the marker that opened offset=1, ~45 µs before this window ──
srs_ann_x = -11
ax2.plot([srs_ann_x, srs_ann_x], [t_axis_y, t_axis_y + 0.16], color=GRN, lw=2.5, zorder=4)
ax2.plot(srs_ann_x, t_axis_y + 0.16, 'v', color=GRN, ms=11, zorder=5)
ax2.text(srs_ann_x, t_axis_y + 0.175,
         'SRS Marker\nsrs_timestamp\nrecorded by FEC',
         ha='center', va='bottom', fontsize=8, color=GRN, fontweight='bold')
ax2.text(srs_ann_x + 0.4, t_axis_y + 0.022, 'offset → 0',
         ha='left', va='bottom', fontsize=8, color=GRN, style='italic', fontweight='bold')
ax2.text(srs_ann_x, t_axis_y - 0.065,
         '(~45 µs before\nthis window)',
         ha='center', va='top', fontsize=7.5, color=GRN, style='italic')

# BC clock edges — dashed lines + amber ticks + ns labels
for i in range(N_per + 1):
    xe = i * T_BC
    ax2.axvline(xe, color='#60A5FA', lw=1.8, ls='--', alpha=0.9, zorder=2)
    ax2.plot(xe, t_axis_y, '|', color=AMB, ms=20, mew=3, zorder=5)
    ax2.text(xe, t_axis_y - 0.065, f'{xe:.1f} ns', ha='center', va='top',
             fontsize=8, color=AMB)
    ax2.plot(xe, 0.895, 'v', color='#3B82F6', ms=8, zorder=3)

# "22.5 ns per BCID tick" brace
ax2.annotate('', xy=(T_BC, 0.540), xytext=(0.0, 0.540),
             arrowprops=dict(arrowstyle='<->', color=AMB, lw=1.8))
ax2.text(T_BC / 2, 0.515, '22.5 ns = 1 BCID tick  (44.44 MHz)',
         ha='center', va='top', fontsize=8.5, color=AMB, style='italic')

# time axis line
ax2.axhline(t_axis_y, xmin=0.02, xmax=0.97, color=DRK, lw=2.5, zorder=3)
ax2.annotate('', xy=(x_end+51, t_axis_y), xytext=(x_end+49, t_axis_y),
             arrowprops=dict(arrowstyle='->', color=DRK, lw=2))
ax2.text(x_end+53, t_axis_y, 't', ha='left', va='center',
         fontsize=13, color=DRK, fontweight='bold')

# ── STEP 1 · Threshold crossing ★ — BCID latched ─────────────────────────────
ax2.plot(Z_hit, t_axis_y, '*', color=RED, ms=22, zorder=7)
ax2.text(Z_hit + 0.8, t_axis_y + 0.09,
         f'① threshold crossed  →  BCID = {BCID_EX} latched',
         ha='left', va='bottom', fontsize=9.5, color=RED, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.22', facecolor='#FEF2F2',
                   edgecolor=RED, lw=1, alpha=0.95))

# ── STEP 2 · Shaping time span — from threshold to peak ───────────────────────
shaping_y = t_axis_y + 0.200
ax2.plot([Z_hit, Z_hit], [t_axis_y + 0.005, shaping_y],
         color='#475569', lw=1.3, ls=':', zorder=3)
ax2.plot([Z_pk,  Z_pk],  [t_axis_y + 0.005, shaping_y],
         color='#475569', lw=1.3, ls=':', zorder=3)
ax2.annotate('', xy=(Z_pk, shaping_y), xytext=(Z_hit, shaping_y),
             arrowprops=dict(arrowstyle='<->', color='#475569', lw=2.2))
ax2.text((Z_hit + Z_pk) / 2, shaping_y + 0.012,
         f'② shaping time = {shaping_ns:.0f} ns  =  {int(shaping_ns/T_BC)} BC bins'
         '  ·  chip ST register  ·  range: 25–200 ns  ·  not recorded per-hit',
         ha='center', va='bottom', fontsize=9, color='#334155', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.18', facecolor='#F8FAFC',
                   edgecolor='#475569', lw=1, alpha=0.92))

# ── STEP 3 · Peak found ◆ — TAC starts ────────────────────────────────────────
ax2.plot(Z_pk, t_axis_y, 'D', color=PIN, ms=14, zorder=7)
ax2.text(Z_pk, t_axis_y - 0.090,
         f'③ peak found  →  TAC starts',
         ha='center', va='top', fontsize=9.5, color=PIN, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.22', facecolor='#FDF2F8',
                   edgecolor=PIN, lw=1, alpha=0.95))

# ── STEP 4 · BC edge — TAC stops, TDC captured ────────────────────────────────
ax2.plot(Z_stop, t_axis_y, '|', color=AMB, ms=28, mew=5, zorder=8)
ax2.text(Z_stop, t_axis_y + 0.09,
         f'④ BC edge  ·  TAC stops\n   BCID {BCID_peak} → {BCID_peak + 1}',
         ha='center', va='bottom', fontsize=9, color=AMB, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.22', facecolor='#FEF3C7',
                   edgecolor=AMB, lw=1, alpha=0.95))

# TDC double-headed arrow + step-by-step derivation
tdc_y_z = 0.310
ax2.plot([Z_pk,   Z_pk],   [t_axis_y - 0.01, 0.132], color=PIN, lw=1, ls=':', zorder=2)
ax2.plot([Z_stop, Z_stop], [t_axis_y - 0.01, 0.132], color=AMB, lw=1, ls=':', zorder=2)
ax2.annotate('', xy=(Z_stop, tdc_y_z), xytext=(Z_pk, tdc_y_z),
             arrowprops=dict(arrowstyle='<->', color=PIN, lw=2.8))
mid_tdc = (Z_pk + Z_stop) / 2
ax2.text(mid_tdc, tdc_y_z + 0.044,
         f'TDC = {TDC_EX}',
         ha='center', va='bottom', fontsize=12, color=PIN, fontweight='bold')
ax2.text(mid_tdc, tdc_y_z - 0.010,
         'Δt  =  TDC  ×  60 ns / 256',
         ha='center', va='top', fontsize=9.5, color=PIN, fontweight='bold')
ax2.text(mid_tdc, tdc_y_z - 0.055,
         f'=  {TDC_EX}  ×  60 ns / 256  =  {Z_tdc_ns:.1f} ns',
         ha='center', va='top', fontsize=9.5, color=PIN)
ax2.text(mid_tdc, tdc_y_z - 0.098,
         '60 ns = TAC full-scale range (VMM3a hardware)    |    256 = 8-bit digitisation',
         ha='center', va='top', fontsize=8, color='#64748B', style='italic')
ax2.text(mid_tdc, tdc_y_z - 0.135,
         'resolution  =  60 ns / 256  =  0.234 ns / count',
         ha='center', va='top', fontsize=8, color=PIN, style='italic')

# TAC count register bar — x-axis aligned with the timeline
bar_y0, bar_y1 = 0.042, 0.130
bar_x0    = Z_pk            # bin 0  — TAC starts (peak)
bar_xstop = Z_stop          # bin TDC_EX — TAC stops
bar_x255  = Z_pk + 60.0     # bin 255 — 60 ns full scale
n_empty   = 256 - TDC_EX
ns_empty  = bar_x255 - bar_xstop
ax2.add_patch(mpatches.Rectangle(
    (bar_x0, bar_y0), bar_xstop - bar_x0, bar_y1 - bar_y0,
    facecolor='#FBCFE8', edgecolor='none', zorder=3))
ax2.add_patch(mpatches.Rectangle(
    (bar_xstop, bar_y0), bar_x255 - bar_xstop, bar_y1 - bar_y0,
    facecolor='#F1F5F9', edgecolor='none', zorder=3))
ax2.add_patch(mpatches.Rectangle(
    (bar_x0, bar_y0), bar_x255 - bar_x0, bar_y1 - bar_y0,
    facecolor='none', edgecolor='#94A3B8', lw=1.5, zorder=4))
ax2.plot([bar_xstop, bar_xstop], [bar_y0, bar_y1], color=AMB, lw=2.5, zorder=5)
ax2.text(bar_x0,    bar_y0 - 0.013, 'bin 0\n(TAC start)',
         ha='center', va='top', fontsize=7.5, color=PIN, fontweight='bold')
ax2.text(bar_xstop, bar_y0 - 0.013, f'bin {TDC_EX}\n(TAC stops)',
         ha='center', va='top', fontsize=7.5, color=AMB, fontweight='bold')
ax2.text(bar_x255,  bar_y0 - 0.013, 'bin 255\n(full scale)',
         ha='center', va='top', fontsize=7.5, color='#64748B')
mid_bar_y = (bar_y0 + bar_y1) / 2
ax2.text((bar_x0 + bar_xstop)/2, mid_bar_y,
         f'{TDC_EX} bins  →  {Z_tdc_ns:.1f} ns',
         ha='center', va='center', fontsize=9, color=PIN, fontweight='bold')
ax2.text((bar_xstop + bar_x255)/2, mid_bar_y,
         f'{n_empty} bins / {ns_empty:.1f} ns\n(not reached)',
         ha='center', va='center', fontsize=8, color='#94A3B8')
ax2.text((bar_x0 + bar_x255)/2, bar_y1 + 0.013,
         'TAC register  ·  8-bit  ·  bin 0 = peak  ·  bin 255 = 60 ns full scale',
         ha='center', va='bottom', fontsize=8.5, color=DRK, fontweight='bold')

# ── STEP 5 · chip_time formula — recovers t_threshold ─────────────────────────
ax2.text((Z_hit + Z_pk) / 2, 0.30,
         f'⑤  chip_time  =  (BCID_thr + 1) × T_BC  −  TDC × 60 ns / 256\n'
         f'   =  (2000 + 1) × 22.5  −  {TDC_EX} × 60/256\n'
         f'   =  45 022.5  −  {tdc_ns:.2f}  =  {chip_ns_bcid:.2f} ns\n'
         f'   ≈  t_threshold  =  {BCID_EX} × 22.5 + {Z_hit:.0f}  =  45 005.0 ns  ✓',
         ha='center', va='top', fontsize=9.5, color=DRK, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.40', facecolor='#F0FDF4', edgecolor=GRN, lw=1.5))

# Resolution comparison
ax2.text(x_end + 50, 0.30,
         'Without TDC:  22.5 ns\nWith TDC:   0.234 ns',
         ha='right', va='top', fontsize=10, color=DRK,
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#F0FDF4', edgecolor=GRN, lw=1.3))

# ── time axis label ───────────────────────────────────────────────────────────
ax2.set_xlabel('Time (ns)', fontsize=10, color='#334155', labelpad=3)

fig3.savefig(OUT + 'schematic_3_clock_timeline.png',
             dpi=DPI, bbox_inches='tight', facecolor=LBG)
print('Saved schematic_3_clock_timeline.png')
plt.close(fig3)

# ══════════════════════════════════════════════════════════════════════════════
# 4 · TIMESTAMP RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = new_fig(22, 11,
    '4 · Timestamp reconstruction — from raw numbers to absolute hit time')

# ── srs_timestamp ─────────────────────────────────────────────────────────────
rbox(ax, 0.17, 0.73, 0.28, 0.36,
     'srs_timestamp',
     '42-bit FEC free-running counter',
     'value at moment of last SRS marker',
     color=GRN, fs=10)
ax.text(0.17, 0.51,
        'Absolute reference:\ncounts 25 ns ticks\nsince FEC was powered on',
        ha='center', va='top', fontsize=9, color=GRN, style='italic')

# ── chip_time components ──────────────────────────────────────────────────────
rbox(ax, 0.52, 0.82, 0.155, 0.26,
     'offset × 4096',
     '× 22.5 ns',
     '= full BCID cycles elapsed',
     color=VIO, fs=9.5)
rbox(ax, 0.685, 0.82, 0.12, 0.26,
     'BCID',
     '× 22.5 ns',
     'position in cycle',
     color=AMB, fs=9.5)
rbox(ax, 0.815, 0.82, 0.12, 0.26,
     '− TDC',
     '× 60 ns / 256',
     'fine-time correction',
     color=PIN, fs=9.5)

# ── chip_time sum box ─────────────────────────────────────────────────────────
rbox(ax, 0.685, 0.49, 0.48, 0.24,
     'chip_time  (ns)',
     'time of hit relative to last SRS marker',
     color=DRK, fs=11)
for cx, tx in [(0.52, 0.59), (0.685, 0.685), (0.815, 0.775)]:
    arrow(ax, cx, 0.69, tx, 0.615, color='#94A3B8', lw=1.8)

# ── plus sign ─────────────────────────────────────────────────────────────────
ax.text(0.435, 0.49, '+', ha='center', va='center',
        fontsize=38, color='#334155', fontweight='bold')

# ── abs_time_ns ───────────────────────────────────────────────────────────────
rbox(ax, 0.50, 0.17, 0.88, 0.24,
     'abs_time_ns  =  srs_timestamp × 25 ns  +  chip_time',
     'Absolute hit time (ns) since FEC boot — use this to correlate hits across VMMs or with external trigger',
     color=CYN, fs=11.5)

arrow(ax, 0.17, 0.545, 0.17, 0.30, color='#94A3B8', lw=2.0)
arrow(ax, 0.685, 0.375, 0.685, 0.30, color='#94A3B8', lw=2.0)
arrow(ax, 0.17,  0.29, 0.40, 0.22, color='#94A3B8', lw=1.8, rad=-0.15)
arrow(ax, 0.685, 0.29, 0.60, 0.22, color='#94A3B8', lw=1.8, rad=0.15)

# ── formula note ──────────────────────────────────────────────────────────────
ax.text(0.5, 0.036,
        'Full formula:   abs_time_ns  =  srs_timestamp × 25 ns  '
        '+  (offset × 4096 + BCID) × 22.5 ns  −  TDC × 60 ns / 256',
        ha='center', va='bottom', fontsize=10, color='#1E293B',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E2E8F0',
                  edgecolor='#94A3B8', lw=1))

ax.text(0.5, 0.005,
        'Deprecated fields (excluded from output): udpTimeStamp (always 0) and '
        'offsetOverflow (fixed boot value) — not populated by current FEC firmware.',
        ha='center', va='bottom', fontsize=8, color='#94A3B8', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig(OUT + 'schematic_4_timestamp_reconstruction.png',
            dpi=DPI, bbox_inches='tight', facecolor=LBG)
print('Saved schematic_4_timestamp_reconstruction.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT  —  all four schematics in one scrollable page
# ══════════════════════════════════════════════════════════════════════════════
panels = [
    ('schematic_1_hardware_chain.png',
     '1 &middot; Hardware chain',
     'Signal path from detector strip through VMM3a ASIC and FEC FPGA to UDP packet on the network. '
     'Key numbers: 64 channels per VMM chip, Spartan7 FPGA at 44.44 MHz (22.5 ns clock), '
     '9 010-byte UDP datagrams on port 6006.'),
    ('schematic_2_vmm3a_hit_processing.png',
     '2 &middot; VMM3a hit processing',
     'Inside the VMM3a ASIC: discriminator fires on threshold crossing, BCID is latched, '
     'the pulse shaper reaches its peak, the peak finder triggers the TAC, and the ADC digitises '
     'the amplitude. Output is a 6-byte hit word containing channel, BCID, TDC, ADC, and offset.'),
    ('schematic_3_clock_timeline.png',
     '3 &middot; Clock, markers, offset, and TDC',
     '<b>Top panel</b>: overview timeline — SRS markers reset the offset counter every ~92&nbsp;µs '
     '(one 4096-tick BCID cycle). Hit offset + BCID give the coarse time; TDC gives the '
     'sub-22.5&nbsp;ns fine-time correction. '
     '<b>Bottom panel</b>: zoom into 5 BC bins (0–112.5&nbsp;ns) showing the five steps in order. '
     '① Threshold at t&nbsp;=&nbsp;5&nbsp;ns latches BCID&nbsp;=&nbsp;2000. '
     '② Shaping time&nbsp;=&nbsp;90&nbsp;ns (4&nbsp;BC bins, chip ST register, range 25–200&nbsp;ns, '
     'not recorded per-hit) carries the pulse from threshold to peak. '
     '③ Peak found at t&nbsp;=&nbsp;95&nbsp;ns in BCID&nbsp;=&nbsp;2004 — TAC starts. '
     '④ BC edge at t&nbsp;=&nbsp;112.5&nbsp;ns — TAC stops; TDC&nbsp;=&nbsp;75&nbsp;→&nbsp;17.5&nbsp;ns. '
     '⑤ <code>chip_time = (BCID_thr+1)×T_BC − TDC×60/256 ≈ t_threshold</code>; '
     'resolution 0.234&nbsp;ns/count.'),
    ('schematic_4_timestamp_reconstruction.png',
     '4 &middot; Timestamp reconstruction',
     'Combining all fields into an absolute hit time: '
     '<code>abs_time_ns = srs_timestamp &times; 25&nbsp;ns '
     '+ (offset &times; 4096 + BCID) &times; 22.5&nbsp;ns '
     '&minus; TDC &times; 60&nbsp;ns / 256</code>. '
     'The srs_timestamp (42-bit FEC free-running counter at 40&nbsp;MHz) provides the '
     'long-range absolute reference; chip_time gives the fine-grained hit position within '
     'the ~92&nbsp;µs BCID cycle.'),
]

html = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VMM3a / SRS Data Flow — Schematics</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #F1F5F9;
           color: #1E293B; padding: 28px 32px; max-width: 1800px; margin: 0 auto; }
    header { margin-bottom: 24px; }
    header h1 { font-size: 1.65rem; color: #0F172A; }
    header p  { color: #64748B; font-size: 0.9rem; margin-top: 4px; }
    nav { background: #E0F2FE; border: 1px solid #BAE6FD; border-radius: 8px;
          padding: 10px 20px; margin-bottom: 32px; display: flex; flex-wrap: wrap; gap: 14px; }
    nav a { color: #1D4ED8; text-decoration: none; font-weight: 600; font-size: 0.9rem; }
    nav a:hover { text-decoration: underline; }
    .panel { background: white; border-radius: 10px; padding: 26px 28px;
             margin-bottom: 36px; box-shadow: 0 2px 10px #CBD5E160; }
    .panel h2 { color: #1D4ED8; font-size: 1.15rem; margin-bottom: 10px; }
    .panel .desc { color: #475569; font-size: 0.88rem; line-height: 1.65;
                   margin-bottom: 18px; }
    .panel .desc code { background: #F1F5F9; padding: 1px 5px; border-radius: 4px;
                        font-size: 0.85em; color: #0E7490; }
    .panel img { width: 100%; height: auto; border-radius: 6px;
                 border: 1px solid #E2E8F0; display: block; }
  </style>
</head>
<body>
  <header>
    <h1>VMM3a / SRS Readout System &#8212; Data Flow Schematics</h1>
    <p>P2 basket hybrid detector &nbsp;&middot;&nbsp; CEA Saclay
       &nbsp;&middot;&nbsp; Reference for VMM3a/SRS pcapng analysis</p>
  </header>
  <nav>\n'''

for fname, title, _ in panels:
    anchor = fname.replace('.png', '').replace('_', '-')
    html += f'    <a href="#{anchor}">{title}</a>\n'
html += '  </nav>\n'

for fname, title, desc in panels:
    anchor = fname.replace('.png', '').replace('_', '-')
    html += (f'  <div class="panel" id="{anchor}">\n'
             f'    <h2>{title}</h2>\n'
             f'    <p class="desc">{desc}</p>\n'
             f'    <img src="{fname}" alt="{title}">\n'
             f'  </div>\n')

html += '</body>\n</html>\n'

html_path = OUT + 'vmm_srs_schematics.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Saved vmm_srs_schematics.html')
