# setup_account_shares.py
import os
import json
import argparse
from secretsharing import PlaintextToHexSecretSharer

parser = argparse.ArgumentParser()
parser.add_argument('--account', required=True)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--n', type=int, default=5)
args = parser.parse_args()

account_id = args.account
k = args.k
n = args.n

os.makedirs('db/accounts', exist_ok=True)
os.makedirs('device_keys', exist_ok=True)

# Generate K_inner for the *device* (demo only: the device stores this key locally)
import os
K_inner = os.urandom(32)
hexkey = K_inner.hex()

# Split into shares
shares = PlaintextToHexSecretSharer.split_secret(hexkey, k, n)

# Save shares pool (server-side which we will pop for each voucher in the demo)
with open(f'db/accounts/{account_id}_shares.json', 'w') as f:
    json.dump({'account_id': account_id, 'k': k, 'n': n, 'shares': shares}, f, indent=2)

# Save device key locally (DEMO ONLY) — in production, devices generate and keep this secret and shares are distributed securely
with open(f'device_keys/{account_id}.key', 'wb') as f:
    f.write(K_inner)

print(f"Wrote shares pool to db/accounts/{account_id}_shares.json and device key to device_keys/{account_id}.key")