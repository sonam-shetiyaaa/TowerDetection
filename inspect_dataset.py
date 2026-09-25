import glob
import os
import cv2
import numpy as np

sample_dir = r"C:\Users\Poonam Shetiya\Downloads\sample\sample"
files = [f for f in glob.glob(os.path.join(sample_dir, "*")) if os.path.isfile(f)]
print(f"Total files found in raw sample: {len(files)}")

resolutions = {}
corrupted = 0
for f in files:
    img = cv2.imread(f)
    if img is None:
        corrupted += 1
        print(f"Corrupted: {os.path.basename(f)}")
    else:
        h, w, c = img.shape
        resolutions[(w, h)] = resolutions.get((w, h), 0) + 1

print(f"Valid images: {len(files) - corrupted}, Corrupted: {corrupted}")
print("Top resolutions:", sorted(resolutions.items(), key=lambda x: x[1], reverse=True)[:5])
