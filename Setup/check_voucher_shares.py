# check_voucher_shares.py

import os
import json
from collections import defaultdict

VOUCHER_DIR = "./vouchers"

def load_vouchers():
    voucher_groups = defaultdict(list)

    for fname in os.listdir(VOUCHER_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(VOUCHER_DIR, fname), "r") as f:
            voucher = json.load(f)
        bh = voucher["blinded_hash"]
        x = voucher["payload"]["share"]["x"]
        voucher_groups[bh].append((fname, x))

    return voucher_groups

def check_duplicates(voucher_groups):
    for bh, entries in voucher_groups.items():
        seen_x = set()
        print(f"\n🔍 Checking vouchers for blinded_hash: {bh}")
        for fname, x in entries:
            if x in seen_x:
                print(f"  ❌ Duplicate x={x} found in {fname}")
            else:
                print(f"  ✅ x={x} OK in {fname}")
                seen_x.add(x)
        print(f"  → Total unique x-values: {len(seen_x)} / {len(entries)}")

if __name__ == "__main__":
    groups = load_vouchers()
    check_duplicates(groups)
