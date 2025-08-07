# hash_group_utils.py

import json
import os

GROUPS_FILE = "./db/blinded_hash_groups.json"

def load_hash_groups():
    return json.load(open(GROUPS_FILE, "r")) if os.path.exists(GROUPS_FILE) else {}

def save_hash_groups(groups):
    with open(GROUPS_FILE, "w") as f:
        json.dump(groups, f, indent=2)

def hamming_distance(b1, b2):
    return sum(a != b for a, b in zip(b1, b2))

def find_or_assign_group(blinded_bits, groups, threshold=10):
    for group_id, hashes in groups.items():
        for stored_bits in hashes:
            if hamming_distance(blinded_bits, stored_bits) <= threshold:
                return group_id
    # new group
    new_id = f"group_{len(groups)+1}"
    groups[new_id] = [blinded_bits]
    save_hash_groups(groups)
    return new_id
