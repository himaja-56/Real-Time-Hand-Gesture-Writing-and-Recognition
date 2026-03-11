import pandas as pd
import numpy as np
import cv2
import os

# Load CSV
data = pd.read_csv("archive/A_Z Handwritten Data.csv")

# Create main folder
os.makedirs("dataset/train", exist_ok=True)

for index, row in data.iterrows():
    label = chr(row[0] + 65)  # Convert 0-25 to A-Z
    pixels = row[1:].values.reshape(28, 28).astype(np.uint8)

    folder_path = f"dataset/train/{label}"
    os.makedirs(folder_path, exist_ok=True)

    cv2.imwrite(f"{folder_path}/{index}.png", pixels)

print("Done converting CSV to images!")