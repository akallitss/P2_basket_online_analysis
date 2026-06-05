#!/usr/bin/env python3
import sys
import time
from collections import deque
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from scapy.all import PcapReader, UDP, IP

#Dans le Bashrc: 
    
# vmm_acq_online() {
#     outfile="$(date +%F_%H-%M-%S).pcapng"
#     echo "📁 Fichier : $outfile"
#     filter="dst host 10.0.0.3"

#     # Lancer dumpcap en foreground, Python dans un autre terminal ou via screen/tmux
#     dumpcap -i enp2s0 -f "$filter" -w "$outfile" &
#     DUMPCAP_PID=$!

#     # Ctrl+C dans le shell arrête Python, ensuite on kill dumpcap
#     trap "echo '🛑 Arrêt demandé'; kill $DUMPCAP_PID; wait $DUMPCAP_PID; exit" SIGINT SIGTERM

#     python3 ~/scripts/FEC_Datadescrambler.py "$outfile"
# }

#########################################
# ARGUMENT : fichier pcapng
#########################################
if len(sys.argv) < 2:
    print("Usage: python3 vmm_live_stable.py <pcap_file>")
    sys.exit(1)

pcap_file = sys.argv[1]

#########################################
# CONFIG
#########################################
window_size = 100000        # nombre max de hits en mémoire
update_every_hits = 1000    # fréquence update graphique

all_hits = deque(maxlen=window_size)

#########################################
# FOLLOW PCAP : lecture en continu
#########################################
def follow_pcap(filename):
    offset = 0
    while True:
        try:
            with open(filename, "rb") as f:
                f.seek(offset)
                reader = PcapReader(f)
                for pkt in reader:
                    yield pkt
                offset = f.tell()
        except FileNotFoundError:
            pass
        time.sleep(0.1)

#########################################
# PARSING HITS VM3
#########################################
def parse_block(block, frame):
    hits = []
    if block[4:7] == b'VM3':
        for i in range(0, len(block)-16, 6):
            d1 = "{:032b}".format(int(block[i+16:i+20].hex(),16))
            d2 = "{:016b}".format(int(block[i+20:i+22].hex(),16))
            if d2[0] == "1":
                hits.append({
                    "frame": frame,
                    "vmmid": int(d1[5:10],2),
                    "offset": int(d1[0:5],2),
                    "adc": int(d1[10:20],2),
                    "bcid": int(d1[20:32],2),
                    "overthreshold": int(d2[1],2),
                    "chno": int(d2[2:8],2),
                    "tdc": int(d2[8:16],2)
                })
    return hits

#########################################
# SETUP MATPLOTLIB
#########################################
plt.ion()
fig = plt.figure(figsize=(16,8))
gs = fig.add_gridspec(4,4)

ax_adc_1 = fig.add_subplot(gs[0,0])
ax_adc_2 = fig.add_subplot(gs[0,1])
ax_adc_3 = fig.add_subplot(gs[0,2])
ax_adc_4 = fig.add_subplot(gs[0,3])

ax_vmm0 = fig.add_subplot(gs[1,0])
ax_vmm1 = fig.add_subplot(gs[1,1])
ax_vmm2 = fig.add_subplot(gs[1,2])
ax_vmm3 = fig.add_subplot(gs[1,3])

# init axes
for ax in [ax_adc_1, ax_adc_2,ax_adc_3,ax_adc_4]:
    ax.set_xlim(0,1000)
    ax.set_xlabel("ADC")
    ax.set_ylabel("Fréquence")

for ax in [ax_vmm0, ax_vmm1, ax_vmm2, ax_vmm3]:
    ax.set_xlim(0,63)
    ax.set_xlabel("Canal")
    ax.set_ylabel("Fréquence")

# histogrammes et textes init
adc_hist_1 = adc_hist_2 = adc_hist_3 = adc_hist_4 = None
hist_0 = hist_1 = hist_2 = hist_3 = None
text0 = text1 = text2 = text3 = None

#########################################
# FONCTION DE PLOTTING
#########################################
def plot_histograms(df):
    global adc_hist_1, adc_hist_2, adc_hist_3, adc_hist_4, hist_0, hist_1, hist_2, hist_3
    global text0, text1, text2, text3

    mask_1 = df['vmmid'].isin([0])
    mask_2 = df['vmmid'].isin([1])
    mask_3 = df['vmmid'].isin([2])
    mask_4 = df['vmmid'].isin([3])

    # supprimer anciens histogrammes
    for hist in [adc_hist_1, adc_hist_2, adc_hist_3, adc_hist_4, hist_0, hist_1, hist_2, hist_3]:
        if hist:
            for bar in hist[2]:
                bar.remove()

    adc_hist_1 = ax_adc_1.hist(df[mask_1]["adc"], bins=20, color='red', alpha=0.7)
    adc_hist_2 = ax_adc_2.hist(df[mask_2]["adc"], bins=20, color='red', alpha=0.7)
    adc_hist_3 = ax_adc_3.hist(df[mask_3]["adc"], bins=20, color='red', alpha=0.7)
    adc_hist_4 = ax_adc_4.hist(df[mask_4]["adc"], bins=20, color='red', alpha=0.7)

    hist_0 = ax_vmm0.hist(df[df['vmmid']==0]["chno"], bins=range(0,65), color='blue', alpha=0.7)
    hist_1 = ax_vmm1.hist(df[df['vmmid']==1]["chno"], bins=range(0,65), color='blue', alpha=0.7)
    hist_2 = ax_vmm2.hist(df[df['vmmid']==2]["chno"], bins=range(0,65), color='blue', alpha=0.7)
    hist_3 = ax_vmm3.hist(df[df['vmmid']==3]["chno"], bins=range(0,65), color='blue', alpha=0.7)

    # supprimer anciens textes
    for t in [text0,text1,text2,text3]:
        if t: Artist.remove(t)

    # ajouter textes mis à jour
    text0 = ax_vmm0.text(30, max(hist_0[0])*0.9, f"Total hits: {len(all_hits)}", ha='center')
    text1 = ax_vmm1.text(30, max(hist_1[0])*0.9, f"Total hits: {len(all_hits)}", ha='center')
    text2 = ax_vmm2.text(30, max(hist_2[0])*0.9, f"Total hits: {len(all_hits)}", ha='center')
    text3 = ax_vmm3.text(30, max(hist_3[0])*0.9, f"Total hits: {len(all_hits)}", ha='center')

    plt.pause(0.1)

#########################################
# MAIN LOOP
#########################################
plt.show()

try:
    for pkt in follow_pcap(pcap_file):
        if UDP in pkt and IP in pkt and pkt[IP].src=="10.0.0.2":
            payload = bytes(pkt[UDP].payload)
            frame = int(payload[:4].hex(),16)
            hits = parse_block(payload, frame)
            all_hits.extend(hits)

        if len(all_hits) and len(all_hits) % update_every_hits == 0:
            df = pd.DataFrame(list(all_hits))
            plot_histograms(df)

except KeyboardInterrupt:
    print("\n🛑 Arrêt demandé par l'utilisateur. Fermeture…")
