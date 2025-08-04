# reconstruct_and_decrypt.py

import os
import json
from collections import defaultdict
from Crypto.Protocol.SecretSharing import Shamir
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

VOUCHER_DIR = "./vouchers"
THRESHOLD = 3  # Minimum shares required

def collect_vouchers():
    share_groups = defaultdict(list)

    for filename in os.listdir(VOUCHER_DIR):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(VOUCHER_DIR, filename)
        with open(path, "r") as f:
            voucher = json.load(f)

        blinded_hash = voucher["blinded_hash"]
        share_data = voucher["payload"]["share"]

        x = share_data["x"]
        y_hex = share_data["y"]
        y = int(y_hex, 16)

        share_groups[blinded_hash].append((x, y, voucher["payload"]["inner_encrypted"], voucher["header"]))

    return share_groups

def reconstruct_and_decrypt(blinded_hash, shares):
    print(f"\n🔐 Reconstructing for hash: {blinded_hash}")

    if len(shares) < THRESHOLD:
        print(f"⚠️ Not enough shares: found {len(shares)}, need {THRESHOLD}")
        return

    subset = shares[:THRESHOLD]
    share_tuples = [(x, y) for (x, y, _, _) in subset]

    # Reconstruct the AES key
    key_int = Shamir.combine(share_tuples)
    key_bytes = key_int.to_bytes(16, 'big')  # AES-128

    iv = bytes.fromhex(subset[0][2]["iv"])
    ciphertext = bytes.fromhex(subset[0][2]["ciphertext"])

    cipher = AES.new(key_bytes, AES.MODE_CBC, iv)
    try:
        decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
        result = json.loads(decrypted)

        # Optional: integrity check
        header = subset[0][3]
        from hashlib import sha256
        if sha256(bytes.fromhex(result["neural_hash"])).hexdigest() != header:
            print("❌ Integrity check failed!")
        else:
            print("✅ Decrypted metadata:")
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"❌ Decryption failed: {e}")

if __name__ == "__main__":
    all_groups = collect_vouchers()
    for blinded_hash, shares in all_groups.items():
        reconstruct_and_decrypt(blinded_hash, shares)
