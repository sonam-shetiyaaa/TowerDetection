import os
import glob
from ultralytics import YOLOWorld

print("Initializing YOLOWorld model...")
model = YOLOWorld("yolov8s-worldv2.pt")

classes = [
    "lattice transmission tower, supporting tower, pylon, steel lattice tower",
    "monopole tower, cell monopole, single tubular pole tower, telecom pole"
]
model.set_classes(classes)
print("Custom classes configured!")

sample_dir = r"C:\Users\Poonam Shetiya\Downloads\sample\sample"
files = glob.glob(os.path.join(sample_dir, "*.jpg")) + glob.glob(os.path.join(sample_dir, "*.JPG"))

if files:
    test_img = files[0]
    print(f"Testing on {os.path.basename(test_img)}...")
    results = model.predict(test_img, conf=0.15, imgsz=640)
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} boxes.")
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            xywhn = b.xywhn[0].tolist()
            label_name = "supporting_tower" if cls_id == 0 else "monopole_tower"
            print(f"  Class: {label_name} ({cls_id}), Conf: {conf:.3f}, Box (xywhn): {[round(x, 4) for x in xywhn]}")
