# admin_reconstruct.py
import os
import sys
import json
from secretsharing import PlaintextToHexSecretSharer

SHARES_DIR = './db/accounts'

def reconstruct_key(account_id):
    account_dir = os.path.join(SHARES_DIR, account_id)
    if not os.path.exists(account_dir):
        raise FileNotFoundError(f"No account directory found for '{account_id}' in {SHARES_DIR}")

    shares_file = os.path.join(account_dir, 'shares.json')
    if not os.path.exists(shares_file):
        raise FileNotFoundError(f"No shares.json found for '{account_id}'")

    with open(shares_file, 'r') as f:
        shares_data = json.load(f)

    # shares_data should be a list of share strings
    shares = shares_data.get("shares")
    if not shares or not isinstance(shares, list):
        raise ValueError(f"Invalid shares.json format for '{account_id}'")

    print(f"[INFO] Loaded {len(shares)} shares for account '{account_id}'")

    # Recombine shares into the original hex key
    reconstructed_hex = PlaintextToHexSecretSharer.recover_secret(shares)
    print(f"[SUCCESS] Reconstructed hex key: {reconstructed_hex}")

    return reconstructed_hex

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python admin_reconstruct.py <account_id>")
        sys.exit(1)

    account_id = sys.argv[1]
    try:
        reconstruct_key(account_id)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
