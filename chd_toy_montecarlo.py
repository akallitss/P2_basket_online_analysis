#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6/15/26 4:55 PM
Created in PyCharm
Created as chd_toy_montecarlo.py

@author: ak271430
"""

"""
Toy Monte Carlo example of CHD (Compress Hash Displace)
for the P2-Basket detector lookup table.

Simulates:
  - A small detector with 3 planes of 64 channels each (instead of 640)
  - A Monte Carlo that generates "physical" triplets
  - The offline CHD construction
  - The online FPGA lookup at runtime
"""

import random

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
N_CHANNELS = 640    # channels per plane  (real: 640)
N_PHYSICAL = 41_581   # physical triplets   (real: 41581)
N_BUCKETS  = 10_000    # ~ N_PHYSICAL / 4

random.seed(42)

# ─────────────────────────────────────────────
# Triplet <-> integer key packing
# (defined up front so STEP 0 can sample keys directly,
#  instead of materializing every possible triplet)
# ─────────────────────────────────────────────
def triplet_to_key(ch1, ch2, ch3):
    return ch1 + ch2 * N_CHANNELS + ch3 * N_CHANNELS * N_CHANNELS

def key_to_triplet(key):
    ch3 = key // (N_CHANNELS ** 2)
    ch2 = (key % (N_CHANNELS ** 2)) // N_CHANNELS
    ch1 = key % N_CHANNELS
    return (ch1, ch2, ch3)

# ─────────────────────────────────────────────
# STEP 0 — Monte Carlo: generate physical triplets
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0 — generating physical triplets (Monte Carlo)")
print("=" * 60)

# random.sample() on a range object never materializes the population
# (640^3 = 262M tuples would be ~tens of GB if built as a list first).
N_TOTAL       = N_CHANNELS ** 3
physical_keys = random.sample(range(N_TOTAL), N_PHYSICAL)
physical_triplets = [key_to_triplet(k) for k in physical_keys]
physical_set      = set(physical_triplets)

print(f"  Total possible triplets : {N_CHANNELS}^3 = {N_TOTAL:,}")
print(f"  Physical triplets found : {N_PHYSICAL}")
print(f"  First 5: {physical_triplets[:5]}")

# ─────────────────────────────────────────────
# STEP 1 — convert triplets -> integer keys
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — convert triplets to integer keys")
print("=" * 60)

print(f"  Example : {physical_triplets[0]} -> key = {physical_keys[0]}")
print(f"  Reverse : key {physical_keys[0]} -> {key_to_triplet(physical_keys[0])}")

# ─────────────────────────────────────────────
# Bit-mixing hash functions
# These avoid the modular arithmetic problems of simple multiply-mod
# The mix() function scrambles bits thoroughly so any change in
# input (even just d increasing by 1) completely changes the output
# ─────────────────────────────────────────────
MASK32 = 0xFFFFFFFF

def mix32(x):
    """Thomas Wang 32-bit integer hash — thoroughly mixes all bits."""
    x = x & MASK32
    x = ((x >> 16) ^ x) * 0x45d9f3b & MASK32
    x = ((x >> 16) ^ x) * 0x45d9f3b & MASK32
    x = ((x >> 16) ^ x) & MASK32
    return x

A0 = 2654435761   # for bucket assignment

def assign_bucket(key):
    return mix32(key ^ A0) % N_BUCKETS

def compute_addr(key, d):
    # XOR the key with a scrambled version of d, then mix
    # Each new d value completely re-scrambles where the key lands
    return mix32(key ^ mix32(d)) % N_PHYSICAL

print(f"\n  Hash constants:")
print(f"    A0  = {A0}  (for bucket assignment)")
print(f"    mix32() = Wang 32-bit mixing function")

# ─────────────────────────────────────────────
# STEP 2 — assign keys to buckets
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — assign keys to buckets")
print("=" * 60)

buckets = [[] for _ in range(N_BUCKETS)]
for key in physical_keys:
    buckets[assign_bucket(key)].append(key)

sizes = [len(b) for b in buckets]
print(f"  Number of buckets      : {N_BUCKETS}")
print(f"  Bucket size min/avg/max: {min(sizes)} / {N_PHYSICAL/N_BUCKETS:.1f} / {max(sizes)}")

# ─────────────────────────────────────────────
# STEP 3 — offline: find displacement per bucket
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — offline: find displacement for each bucket")
print("=" * 60)

# Process largest buckets first
bucket_order = sorted(range(N_BUCKETS), key=lambda i: -len(buckets[i]))

lookup_table = [None]  * N_PHYSICAL
displacement = [0]     * N_BUCKETS
slot_taken   = [False] * N_PHYSICAL

failed = 0
max_d  = 0
total_tries = 0

for bi in bucket_order:
    keys_in_bucket = buckets[bi]
    if not keys_in_bucket:
        displacement[bi] = 0
        continue

    placed = False
    for d in range(200_000):
        total_tries += 1
        slots = [compute_addr(key, d) for key in keys_in_bucket]

        # no collision within the bucket
        if len(set(slots)) < len(slots):
            continue

        # no collision with already-placed buckets
        if any(slot_taken[s] for s in slots):
            continue

        # commit
        displacement[bi] = d
        for key, s in zip(keys_in_bucket, slots):
            lookup_table[s] = key
            slot_taken[s]   = True
        max_d  = max(max_d, d)
        placed = True
        break

    if not placed:
        failed += 1
        print(f"  WARNING: could not place bucket {bi} (size {len(keys_in_bucket)})")

print(f"  Failed buckets         : {failed}  (should be 0)")
print(f"  Max displacement used  : {max_d}")
print(f"  Slots filled           : {sum(slot_taken)} / {N_PHYSICAL}")

# ─────────────────────────────────────────────
# STEP 4 — parameters burned into FPGA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — parameters burned into FPGA")
print("=" * 60)

size_const = 1 * 4             # A0
size_disp  = N_BUCKETS  * 4   # displacement table
size_table = N_PHYSICAL * 4   # lookup table
total      = size_const + size_disp + size_table

print(f"  A0                            : {A0}")
print(f"  displacement[0:5]             : {displacement[:5]}")
print(f"  lookup_table[0:5]             : {lookup_table[:5]}")
print()
print(f"  Constant A0                   : {size_const} bytes")
print(f"  Displacement table ({N_BUCKETS} entries): {size_disp} bytes")
print(f"  Lookup table ({N_PHYSICAL} entries)     : {size_table} bytes")
print(f"  ─────────────────────────────────────────")
print(f"  TOTAL                         : {total} bytes  ({total/1024:.1f} KB)")
print(f"  Naive 640^3 table             : ~268 million entries ≈ 1 GB")

# ─────────────────────────────────────────────
# Runtime lookup — exactly what the FPGA does
# ─────────────────────────────────────────────
def lookup(ch1, ch2, ch3):
    key    = triplet_to_key(ch1, ch2, ch3)  # pack triplet -> key
    bucket = assign_bucket(key)              # which bucket? (1 mix)
    d      = displacement[bucket]            # read displacement (1 table read)
    addr   = compute_addr(key, d)            # final address (1 mix)
    return lookup_table[addr] == key         # physical? (1 compare)

# ─────────────────────────────────────────────
# STEP 5 — Monte Carlo validation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Monte Carlo validation")
print("=" * 60)

errors = sum(1 for t in physical_triplets if not lookup(*t))
print(f"  Physical triplets tested      : {N_PHYSICAL}")
print(f"  Correctly identified    (✓)   : {N_PHYSICAL - errors}")
print(f"  Errors (should be 0)          : {errors}")

N_TEST    = 10_000
false_pos = 0
n_tested  = 0
for _ in range(N_TEST):
    t = (random.randint(0, N_CHANNELS-1),
         random.randint(0, N_CHANNELS-1),
         random.randint(0, N_CHANNELS-1))
    if t not in physical_set:
        n_tested += 1
        if lookup(*t):
            false_pos += 1

print(f"\n  Non-physical triplets tested  : {n_tested}")
print(f"  Correctly rejected      (✓)   : {n_tested - false_pos}")
print(f"  False positives (should be 0) : {false_pos}")

# ─────────────────────────────────────────────
# STEP 6 — step-by-step individual examples
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — step-by-step lookup examples")
print("=" * 60)

def show(ch1, ch2, ch3, label=""):
    key    = triplet_to_key(ch1, ch2, ch3)
    bucket = assign_bucket(key)
    d      = displacement[bucket]
    addr   = compute_addr(key, d)
    stored = lookup_table[addr]
    ok     = (stored == key)
    tag    = "PHYSICAL ✓" if ok else "NOT PHYSICAL ✗"
    print(f"  {label:12s} {str((ch1,ch2,ch3)):16s}"
          f"key={key:6d}  bucket={bucket:2d}  d={d:5d}  "
          f"addr={addr:3d}  stored={str(stored):8}  → {tag}")

print("  [physical]")
for t in physical_triplets[:4]:
    show(*t, label="physical")

print("  [non-physical]")
for t in [(0,0,0),(1,1,1),(63,63,63),(10,20,30)]:
    if t not in physical_set:
        show(*t, label="non-phys")

print("\nDone.")

def main():
    print('bonzo')


if __name__ == '__main__':
    main()
