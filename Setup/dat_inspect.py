import numpy as np
import os

def try_load_and_inspect(file_path, dtype, shape_guess=None, max_elements=10):
    with open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=dtype)

    print(f"Trying dtype={dtype}: total elements = {len(data)}")

    if shape_guess and len(data) == np.prod(shape_guess):
        matrix = data.reshape(shape_guess)
        print(f"Reshaped to {shape_guess}:\n", matrix[:max_elements])
    else:
        print(f"First {max_elements} elements:\n", data[:max_elements])

    print("-" * 40)
    return data

file_path = "./models/neuralhash_128x96_seed1.dat"

# Try interpreting as float32
float32_data = try_load_and_inspect(file_path, dtype=np.float32, shape_guess=(96, 128))

# Try interpreting as int8
int8_data = try_load_and_inspect(file_path, dtype=np.int8, shape_guess=(96, 128))

# Optional: check actual file size
file_size = os.path.getsize(file_path)
print(f"File size in bytes: {file_size}")
