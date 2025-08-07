# matcher.py
import numpy as np
import json
import os

def hamming_distance(bits1, bits2):
    return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))

def load_reference_hashes(path='hashes'):
    database = {}
    for file in os.listdir(path):
        if file.endswith(".json"):
            name = file.replace(".json", "")
            with open(os.path.join(path, file)) as f:
                database[name] = json.load(f)
    return database

def match_probe_hash(probe_bits, db, threshold=15):
    matches = []
    for name, ref_bits in db.items():
        dist = hamming_distance(probe_bits, ref_bits)
        if dist <= threshold:
            matches.append((name, dist))
    return sorted(matches, key=lambda x: x[1])
