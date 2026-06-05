# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 17:36:02 2025

@author: dbaudin
"""

import time
from scapy.all import PcapReader, UDP, IP
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import subprocess
from datetime import datetime
import numpy as np
import subprocess 
from tqdm import tqdm
import matplotlib.animation as animation
import openpyxl

#Parsing des données, fonction permettant de récupérer un dataframe
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

#Lecture du fichier, peu être long si le fichier est lourd



all_hits = []
Config = r"C:\Users\dbaudin\Documents_Local\Projet\NTOF\Implementation\Hardware\VMM3_HVBoard/Pinout_DetecteurASIC.xlsx"
tshark_path = r"C:\Program Files\Wireshark\tshark.exe"  # <- chemin complet vers tshark.exe
#input_file = "20251118_153029.pcapng" #Milieu
#input_file = "20251118_142925.pcapng" #Haut droite
#input_file = "20251118_145245.pcapng" #Haut gauche
input_file =  "20251118_124625.pcapng"# bas
#input_file =  "20251118_103851.pcapng" #Movement
output_file = "udp_data.txt"

cmd = [
       tshark_path,
       "-r", input_file,
       "-T", "fields",
       "-e", "frame.number",
       ]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = proc.communicate()
total_packets = len(out.strip().splitlines())
print(f"Total paquets : {total_packets}")
start = 0

cmd = [
    tshark_path,
    "-r", input_file,
    "-Y", "ip.src == 10.0.0.2 && udp",
    "-T", "fields",
    "-e", "data.data",
    "-Y", f"frame.number >= {start}",
    "-c" "1000"
]


proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

all_hits = []
# barre de progression indéterminée (juste animation)
for line in tqdm(proc.stdout, desc="Parsing paquets", unit="pkt"):
    try: 
        payload = bytes.fromhex(line.strip())
        frameCounter = int(payload[0:4].hex(), 16)
        hits = parse_block(payload, frameCounter)
        all_hits.extend(hits)
    except: 
        print("End of File")

proc.stdout.close()
proc.wait()

df = pd.DataFrame(all_hits)

#%%Traitement
#Dans un premier temps on affiche l'histogramme des X et Y et une carte de coups (image)
#Ensuite on cherche les évenement dans un pattern pour sommer toute l'énergie distribuée sur plusieurs strips

fig = plt.figure()
ax_adc_X = fig.add_subplot()
ax_adc_X.set_title(f"Histogramme ADC")
ax_adc_X.set_xlabel("ADC")
ax_adc_X.set_xlim(xmin=0, xmax=1000)
ax_adc_X.set_ylabel("Nb_Counts")
mask_X = (df['vmmid'] == 0) | (df['vmmid'] == 1)
hist_X = ax_adc_X.hist(df[mask_X]["adc"], bins=int((max(df[mask_X]["adc"])-min(df[mask_X]["adc"]))/5), color='red', alpha=0.7)

fig = plt.figure()
ax_adc_Y = fig.add_subplot()
ax_adc_Y.set_title(f"Histogramme ADC")
ax_adc_Y.set_xlabel("ADC")
ax_adc_Y.set_xlim(xmin=0, xmax=1000)
ax_adc_Y.set_ylabel("Nb_Counts")
mask_Y = (df['vmmid'] == 0) | (df['vmmid'] == 1)
hist_Y = ax_adc_Y.hist(df[mask_Y]["adc"], bins=int((max(df[mask_Y]["adc"])-min(df[mask_Y]["adc"]))/5), color='blue', alpha=0.7)


#%% histogram of one specific pixel
fig = plt.figure()
ax_adc_X = fig.add_subplot()
ax_adc_X.set_title(f"Histogramme ADC")
ax_adc_X.set_xlabel("ADC")
ax_adc_X.set_xlim(xmin=0, xmax=1000)
ax_adc_X.set_ylabel("Nb_Counts")
mask_X = (df['vmmid'] == 0) & (df['chno'] == 28)
hist_X = ax_adc_X.hist(df[mask_X]["adc"], bins=int((max(df[mask_X]["adc"])-min(df[mask_X]["adc"]))/10), color='red', alpha=0.7)

#%% histograme of simple frames + bcid (mono event)
# Exemple: ton DataFrame s'appelle df

# 1️⃣ Compter les occurrences de chaque combinaison frame + bcid
counts = df.groupby(["frame", "bcid"]).size()

# 2️⃣ Sélectionner uniquement celles qui apparaissent une seule fois
unique_combos = counts[counts == 1].index

# 3️⃣ Filtrer le DataFrame pour ne garder que ces combinaisons uniques
df_filtered = df.set_index(["frame", "bcid"]).loc[unique_combos].reset_index()

# 4️⃣ Tracer l’histogramme des adc
plt.hist(df_filtered["adc"], bins=200)
plt.xlabel("ADC")
plt.ylabel("Nombre d’occurrences")
plt.title("Histogramme des ADC pour les combinaisons frame+bcid uniques")
plt.show()

#Tracer l'histograme pour une seule strip
# fig = plt.figure()
# Strip = 40
# mask = (df_filtered['vmmid'] == 0) & (df_filtered['chno'] == Strip)
# Histo = np.histogram(df_filtered[mask]["adc"], bins=int((max(df_filtered[mask]["adc"])-min(df_filtered[mask]["adc"]))/10))
# plt.plot(Histo[1][0:-1],Histo[0],linewidth = 1, markersize = 2, marker = "o", drawstyle='steps')   

#%% Repining maps
# The design follows a "detector pin" and a "ASIC" pin map given by an excel file on Strip X and Y
# We will compute the channel number and change it for each count to the known map given by excel file
# Lecture et tri
df_Config_X = pd.read_excel(Config, 'Strips X').sort_values("Pin Number VMM3").reset_index(drop=True)
df_Config_Y = pd.read_excel(Config, 'Strips Y').sort_values("Pin Number VMM3").reset_index(drop=True)

# Création de séries pour les correspondances
map_X_0 = df_Config_X["Pin Number Detecteur"]
map_X_1 = df_Config_X["Pin Number Detecteur"].shift(-64)
map_Y_2 = df_Config_Y["Pin Number Detecteur"]
map_Y_3 = df_Config_Y["Pin Number Detecteur"].shift(-64)

# Copie du DataFrame pour ne pas écraser l’original
df_new = df.copy()

# Application vectorisée selon vmmid
mask0 = df_new["vmmid"] == 0
mask1 = df_new["vmmid"] == 1
mask2 = df_new["vmmid"] == 2
mask3 = df_new["vmmid"] == 3

df_new.loc[mask0, "chno"] = df_new.loc[mask0, "chno"].map(map_Y_2)
df_new.loc[mask1, "chno"] = (df_new.loc[mask1, "chno"] + 64).map(df_Config_Y["Pin Number Detecteur"])
df_new.loc[mask2, "chno"] = df_new.loc[mask2, "chno"].map(map_X_0)
df_new.loc[mask3, "chno"] = (df_new.loc[mask3, "chno"] + 64).map(df_Config_X["Pin Number Detecteur"])
        


#%%animation


events_per_frame = 10000
total_events = len(df_new)
n_frames = total_events // events_per_frame

size = 124  # taille fixe

# --- Fonction pour construire X et Y à partir d'un batch ---
def build_XY(df_batch):
    # X = vmmid = 0 ou 1
    dfX = df_batch[df_batch['vmmid'].isin([0,1])]["chno"].value_counts().reset_index()
    dfX.columns = ["chno", "Nbcoups"]
    dfX = dfX.sort_values("chno")

    # Y = vmmid = 2 ou 3
    dfY = df_batch[df_batch['vmmid'].isin([2,3])]["chno"].value_counts().reset_index()
    dfY.columns = ["chno", "Nbcoups"]
    dfY = dfY.sort_values("chno")

    X = np.zeros(size, dtype=float)
    for _, row in dfX.iterrows():
        if row["chno"] < size:
            X[row["chno"]] = row["Nbcoups"]

    Y = np.zeros(size, dtype=float)
    for _, row in dfY.iterrows():
        if row["chno"] < size:
            Y[row["chno"]] = row["Nbcoups"]

    return X, Y


# --- Figure et image initiale ---
fig, ax = plt.subplots(figsize=(7,7))

# batch initial
batch0 = df_new.iloc[:events_per_frame]
X0, Y0 = build_XY(batch0)
image0 = Y0[:, None] + X0[None, :]

im = ax.imshow(image0, cmap="hot", origin="lower", interpolation="nearest")
ax.invert_yaxis()

plt.colorbar(im, ax=ax, label="Intensité (Nbcoups X + Nbcoups Y)")
ax.set_xlabel("Canal Y (chno)")
ax.set_ylabel("Canal X (chno)")
ax.set_title("Animation des événements")


# --- Fonction d'update ---
def update(frame):
    start = frame * events_per_frame
    end = min(start + events_per_frame, total_events)

    df_batch = df_new.iloc[start:end]
    X, Y = build_XY(df_batch)
    image = Y[:, None] + X[None, :]

    im.set_data(image)
    ax.set_title(f"Frame {frame} : événements {start}-{end}")

    return [im]


# --- Animation ---
ani = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=50, blit=False, repeat=False
)

plt.show()

#%%Image
mask = (df_new['vmmid'] == 0 )| (df_new['vmmid'] == 1)
dfX = df_new[mask]["chno"].value_counts().reset_index()
dfX.columns = ["chno", "Nbcoups"]
dfX = dfX.sort_values("chno").reset_index(drop=True)
mask = (df_new['vmmid'] == 2 )| (df_new['vmmid'] == 3)
dfY = df_new[mask]["chno"].value_counts().reset_index()
dfY.columns = ["chno", "Nbcoups"]
dfY = dfY.sort_values("chno").reset_index(drop=True)

size = 124  # taille fixe

# --- Construction de tableaux X et Y de taille fixe ---
X = np.zeros(size)
for _, row in dfX.iterrows():
    if row["chno"] < size:
        X[row["chno"]] = row["Nbcoups"]

Y = np.zeros(size)
for _, row in dfY.iterrows():
    if row["chno"] < size:
        Y[row["chno"]] = row["Nbcoups"]

# Création de l’image 128×128
image = Y[:, None] + X[None, :]

# --- Affichage ---
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap="hot", origin="lower", interpolation="nearest")

plt.gca().invert_yaxis()   # X=0 en haut
plt.colorbar(label="Intensité (Nbcoups_X + Nbcoups_Y)")
plt.xlabel("Canal Y (chno)")
plt.ylabel("Canal X (chno)")
plt.title("Image X–Y superposée (grille 128×128 fixe)")
plt.show()


#%%
#première constante: vmm0,1 c'est le Y et vmm2,3 c'est le X

# df_new = df.copy()
# mask = (df_new["vmmid"] == 1)
# df_new.loc[mask,"chno"] = df_new.loc[mask,"chno"] + 64
mask = (df_new['vmmid'] == 0 )| (df_new['vmmid'] == 1)
dfY = df_new[mask]["chno"].value_counts().reset_index()
dfY.columns = ["chno", "Nbcoups"]
dfY = dfY.sort_values("chno").reset_index(drop=True)
# mask = df_new["vmmid"] == 3
# df_new.loc[mask,"chno"] = df_new.loc[mask,"chno"] + 64
mask = (df_new['vmmid'] == 2 )| (df_new['vmmid'] == 3)
dfX = df_new[mask]["chno"].value_counts().reset_index()
dfX.columns = ["chno", "Nbcoups"]
dfX = dfX.sort_values("chno").reset_index(drop=True)

plt.figure()
plt.plot(dfX["chno"], dfX["Nbcoups"], color='b')
plt.plot(dfY["chno"],  dfY["Nbcoups"], color='r')
#%%

# plt.plot(df[df['vmmid'] == 0]["frame"], df[df['vmmid'] == 0]["chno"], marker = 'o', markersize = 1, linewidth = 0)
# plt.plot(df[df['vmmid'] == 1]["frame"], df[df['vmmid'] == 1]["chno"], marker = 'o', markersize = 1, linewidth = 0)
# plt.plot(df[df['vmmid'] == 2]["frame"], df[df['vmmid'] == 2]["chno"], marker = 'o', markersize = 1, linewidth = 0)
# plt.plot(df[df['vmmid'] == 3]["frame"], df[df['vmmid'] == 3]["chno"], marker = 'o', markersize = 1, linewidth = 0)
