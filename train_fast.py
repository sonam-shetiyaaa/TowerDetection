import os
import shutil
from ultralytics import YOLO

print("Starting fast fine-tuning on custom dataset...")
model = YOLO("yolov8n.pt")
results = model.train(
    data="d:/college/TowerDetection/dataset/data.yaml",
    epochs=5,
    imgsz=320,
    batch=8,
    project="d:/college/TowerDetection/runs",
    name="tower_model",
    exist_ok=True,
    verbose=True
)

dest = "d:/college/TowerDetection/model/best.pt"
src = "d:/college/TowerDetection/runs/tower_model/weights/best.pt"
if os.path.exists(src):
    shutil.copy2(src, dest)
    print(f"Copied best weights to {dest}")
else:
    src_last = "d:/college/TowerDetection/runs/tower_model/weights/last.pt"
    if os.path.exists(src_last):
        shutil.copy2(src_last, dest)
        print(f"Copied last weights to {dest}")

# Copy artifact plots for judges
exp_dir = "d:/college/TowerDetection/runs/tower_model"
for plot_file in ["confusion_matrix.png", "results.png", "PR_curve.png", "F1_curve.png"]:
    s = os.path.join(exp_dir, plot_file)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join("d:/college/TowerDetection/model", plot_file))
        print(f"Copied artifact: {plot_file}")

print("Fast model training & evaluation completed!")
