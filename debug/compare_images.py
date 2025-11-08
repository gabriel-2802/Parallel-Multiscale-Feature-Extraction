#!/usr/bin/env python3
import sys
from PIL import Image
import numpy as np
import subprocess

def image_to_matrix(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.int32)

def save_matrix(matrix, filename):
    np.savetxt(filename, matrix, fmt="%d")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compare_images_diff.py <image1> <image2>")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]

    mat1 = image_to_matrix(img1_path)
    mat2 = image_to_matrix(img2_path)

    save_matrix(mat1, "image1_matrix.txt")
    save_matrix(mat2, "image2_matrix.txt")

    print("Matrices saved to image1_matrix.txt and image2_matrix.txt")

    # Use Ubuntu diff command
    try:
        print("\n--- Diff between the two matrices ---")
        subprocess.run(["diff", "-u", "image1_matrix.txt", "image2_matrix.txt"])
    except FileNotFoundError:
        print("Error: 'diff' command not found. Make sure you are on Ubuntu/Linux.")

if __name__ == "__main__":
    main()
