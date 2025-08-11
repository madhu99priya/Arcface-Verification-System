# voucher.py (helpers)
import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from hashlib import sha256


def create_inner_ciphertext(K_inner: bytes, neuralhash_bytes: bytes, visual_bytes: bytes=b'') -> bytes:
    aes = AESGCM(K_inner)
    nonce = os.urandom(12)
    plaintext = neuralhash_bytes + b'--VIS--' + visual_bytes
    ct = aes.encrypt(nonce, plaintext, None)
    return nonce + ct


def derive_k_outer_from_hash_hex(hash_hex: str) -> bytes:
    return sha256(bytes.fromhex(hash_hex)).digest()


def create_outer_ciphertext(K_outer: bytes, inner_ct: bytes, share_str: str, metadata: bytes=b'') -> bytes:
    aes = AESGCM(K_outer)
    nonce = os.urandom(12)
    payload = inner_ct + b'--SHARE--' + share_str.encode() + b'--META--' + metadata
    ct = aes.encrypt(nonce, payload, None)
    return nonce + ct


def encode_b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def decode_b64(s: str) -> bytes:
    return base64.b64decode(s)