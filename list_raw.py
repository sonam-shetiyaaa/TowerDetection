import glob
import os

sample_dir = r"C:\Users\Poonam Shetiya\Downloads\sample\sample"
files = sorted([f for f in glob.glob(os.path.join(sample_dir, "*")) if os.path.isfile(f)])
print(f"Total images: {len(files)}")
for f in files[:20]:
    sz = os.path.getsize(f) / (1024 * 1024)
    print(f"{os.path.basename(f)}: {sz:.2f} MB")
