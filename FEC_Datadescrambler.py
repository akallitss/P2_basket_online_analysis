import time
from scapy.all import PcapReader, UDP, IP
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import subprocess
from datetime import datetime



iface = "enp2s0"
bpf = "dst host 10.0.0.3"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"{timestamp}.pcapng"

cmd = [
    "dumpcap",
    "-i", iface,
    "-f", bpf,
    "-w", outfile
]

ret = subprocess.run(cmd)


file_path = outfile
parquet_file = "hits_data.parquet"
window_size = 100000


all_hits = []

plt.ion()
fig = plt.figure()
gs = fig.add_gridspec(2,4)
ax_adc_X = fig.add_subplot(gs[0,0:2])
ax_adc_Y = fig.add_subplot(gs[0,2:4])
ax_vmm0 = fig.add_subplot(gs[1,0])
ax_vmm1 = fig.add_subplot(gs[1,1])
ax_vmm2 = fig.add_subplot(gs[1,2])
ax_vmm3 = fig.add_subplot(gs[1,3])
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
fig.tight_layout(pad=1)



ax_adc_X.set_title(f"Histogramme ADC (derniers {window_size} hits)")
ax_adc_X.set_xlabel("ADC")
ax_adc_X.set_xlim(xmin=0, xmax=1000)
ax_adc_X.set_ylabel("Fréquence")
ax_adc_Y.set_title(f"Histogramme ADC (derniers {window_size} hits)")
ax_adc_Y.set_xlabel("ADC")
ax_adc_Y.set_xlim(xmin=0, xmax=1000)
ax_adc_Y.set_ylabel("Fréquence")

ax_vmm0.set_title("Répartition des canaux (CHNO)")
ax_vmm0.set_xlabel("Canal")
ax_vmm0.set_ylabel("Fréquence") 
ax_vmm0.set_xlim(xmin=0, xmax=63)
ax_vmm1.set_title("Répartition des canaux (CHNO)")
ax_vmm1.set_xlabel("Canal")
ax_vmm1.set_ylabel("Fréquence") 
ax_vmm1.set_xlim(xmin=0, xmax=63)
ax_vmm2.set_title("Répartition des canaux (CHNO)")
ax_vmm2.set_xlabel("Canal")
ax_vmm2.set_ylabel("Fréquence")
ax_vmm2.set_xlim(xmin=0, xmax=63) 
ax_vmm3.set_title("Répartition des canaux (CHNO)")
ax_vmm3.set_xlabel("Canal")
ax_vmm3.set_ylabel("Fréquence")  
ax_vmm3.set_xlim(xmin=0, xmax=63)
    

hist_X = None
hist_Y = None  
hist_0 = None  
hist_1 = None  
hist_2 = None  
hist_3 = None  
text0 = None
text1 = None
text2 = None
text3 = None


def plot_histograms(df):
    global hist_X, hist_Y, hist_0, hist_1, hist_2, hist_3, text0, text1, text2, text3
    mask_X = (df['vmmid'] == 0) | (df['vmmid'] == 1)
    mask_Y = (df['vmmid'] == 2) | (df['vmmid'] == 3)

    if hist_X:
        for bar in hist_X[2]:
            bar.remove()
    if hist_Y:
        for bar in hist_Y[2]:
            bar.remove()
    if hist_0:
        for bar in hist_0[2]:
            bar.remove()
    if hist_1:
        for bar in hist_1[2]:
            bar.remove()                        
    if hist_2:
        for bar in hist_2[2]:
            bar.remove()
    if hist_3:
        for bar in hist_3[2]:
            bar.remove()
    hist_X = ax_adc_X.hist(df[mask_X]["adc"], bins=5, color='red', alpha=0.7)
    hist_Y = ax_adc_Y.hist(df[mask_Y]["adc"], bins=5, color='red', alpha=0.7)
    
    
    
    hist_0 = ax_vmm0.hist(df[df['vmmid'] == 0]["chno"], bins=range(min(df["chno"]), max(df["chno"]) + 2), color='blue', alpha=0.7)
    hist_1  = ax_vmm1.hist(df[df['vmmid'] == 1]["chno"], bins=range(min(df["chno"]), max(df["chno"]) + 2), color='blue', alpha=0.7)
    hist_2  = ax_vmm2.hist(df[df['vmmid'] == 2]["chno"], bins=range(min(df["chno"]), max(df["chno"]) + 2), color='blue', alpha=0.7)
    hist_3  = ax_vmm3.hist(df[df['vmmid'] == 3]["chno"], bins=range(min(df["chno"]), max(df["chno"]) + 2), color='blue', alpha=0.7)   
    if(text0 != None):
        Artist.remove(text0)
    if(text1 != None):
        Artist.remove(text1)
    if(text2 != None):
        Artist.remove(text2)
    if(text3 != None):
        Artist.remove(text3)
    text0 = ax_vmm0.text(30, max(hist_0[0])*0.9, f"Total hits : {len(all_hits)}", fontsize=16, ha='center', va='center')
    text1 = ax_vmm1.text(30, max(hist_1[0])*0.9, f"Total hits : {len(all_hits)}", fontsize=16, ha='center', va='center')
    text2 = ax_vmm2.text(30, max(hist_2[0])*0.9, f"Total hits : {len(all_hits)}", fontsize=16, ha='center', va='center')
    text3 = ax_vmm3.text(30, max(hist_3[0])*0.9, f"Total hits : {len(all_hits)}", fontsize=16, ha='center', va='center')
    plt.pause(0.2)

def follow_pcap(filename):
    offset = 0
    while True:
        try:
            f = open(filename, "rb")
            f.seek(offset)
            pcap_reader = PcapReader(f)
            for pkt in pcap_reader:
                yield pkt
            offset = f.tell()
            f.close()
        except FileNotFoundError:
            pass
        time.sleep(0.1)

def parse_block(block: bytes, frameCounter):
    hits = []
    if block[4:6+1] == b'VM3':
        for i in range(0, len(block) - 16, 6):
            d1 = "{:032b}".format(int(block[i+16:i+20].hex(), 16))
            d2 = "{:016b}".format(int(block[i+20:i+22].hex(), 16))
            if d2[0] == '1':
                hit = {
                    "frame": frameCounter,
                    "vmmid": int(d1[5:10], 2),
                    "offset": int(d1[0:5], 2),
                    "adc": int(d1[10:20], 2),
                    "bcid": int(d1[20:32], 2),
                    "overthreshold": int(d2[1], 2),
                    "chno": int(d2[2:8], 2),
                    "tdc": int(d2[8:16], 2)
                }
                hits.append(hit)
    return hits

plt.show()

for pkt in follow_pcap(file_path):
    if UDP in pkt and IP in pkt and pkt[IP].src == "10.0.0.2":
        payload = bytes(pkt[UDP].payload)
        frameCounter = int(payload[0:4].hex(), 16)
        hits = parse_block(payload, frameCounter)
        all_hits.extend(hits)

    # Mise à jour du dashboard toutes les 1000 hits
    if len(all_hits) % 1000 == 0 and all_hits:
        df = pd.DataFrame(all_hits[-window_size:])
        plot_histograms(df)