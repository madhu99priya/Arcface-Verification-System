import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from secretsharing import PlaintextToHexSecretSharer

THRESHOLD = 3
received_shares = []

def generate_secret_key():
    return "SuperSecretKeyForFaceData"

def split_secret(secret, n=5, t=3):
    return PlaintextToHexSecretSharer.split_secret(secret, t, n)

def reconstruct_secret(shares):
    return PlaintextToHexSecretSharer.recover_secret(shares)

def encrypt_inner_data(data, key):
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data.encode('utf-8'), AES.block_size))
    return {
        'iv': cipher.iv.hex(),
        'ciphertext': ct_bytes.hex()
    }

def generate_safety_voucher(hash_hex, secret_share, encryption_key):
    encrypted_data = encrypt_inner_data(hash_hex, encryption_key)
    return json.dumps({
        'secret_share': secret_share,
        'encrypted_hash': encrypted_data
    })
