from camera import capture_frame
from neural_hash import generate_neuralhash, bits_to_hex, load_pca_model, load_hyperplanes
from psi_utils import load_blinded_hashes, blind_hash_simulated, psi_check
from voucher import generate_secret_key, split_secret, reconstruct_secret, generate_safety_voucher, received_shares
import config


def run_pipeline():
    load_blinded_hashes()

    print("[1] Capturing image...")
    image_path = capture_frame()

    print("[2] Loading models...")
    pca = load_pca_model(config.pca_path)
    hyperplanes = load_hyperplanes(config.hyperplane_path)

    print("[3] Generating NeuralHash...")
    hash_bits = generate_neuralhash(image_path, pca, hyperplanes)
    hash_hex = bits_to_hex(hash_bits)

    print("[4] Performing PSI...")
    blinded = blind_hash_simulated(hash_bits)
    if psi_check(blinded):
        print("[✓] Match found. Generating voucher...")
        key = generate_secret_key()
        shares = split_secret(key)
        received_shares.append(shares[len(received_shares)])

        voucher = generate_safety_voucher(hash_hex, received_shares[-1], key)
        print("[✓] Voucher:", voucher)

        if len(received_shares) >= 3:
            print("[✓] Threshold met. Reconstructing key...")
            recovered = reconstruct_secret(received_shares)
            print("[✓] Decrypted Key:", recovered)
    else:
        print("[!] No match found.")

if __name__ == '__main__':
    run_pipeline()