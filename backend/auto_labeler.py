"""
Auto-Labeling Pipeline using Zero-Shot Vision Detection (YOLO-World)
Generates initial candidate bounding boxes and labels for:
- 0: supporting_tower (lattice towers, truss pylons, guyed transmission towers)
- 1: monopole_tower (single tubular pole towers, slim cell poles)
"""

import os
import glob
import json
import cv2
from typing import List, Dict, Any
from ultralytics import YOLOWorld


class TowerAutoLabeler:
    def __init__(self, model_path: str = "yolov8s-worldv2.pt"):
        print(f"Loading Zero-Shot Detection Model: {model_path}...")
        self.model = YOLOWorld(model_path)
        # Detailed prompts for high-precision zero-shot distinction
        self.prompt_classes = [
            "lattice transmission tower, steel lattice tower, electricity pylon, truss tower",
            "monopole tower, cell phone monopole, tubular steel pole, telecommunication pole"
        ]
        self.model.set_classes(self.prompt_classes)
        print("Model configured with target class prompts: supporting_tower (0), monopole_tower (1)")

    def label_image(self, image_path: str, conf_thresh: float = 0.15) -> List[Dict[str, Any]]:
        """
        Runs zero-shot inference on an image and returns detected tower objects.
        """
        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]
        # Predict with image resize to standard 640/1024 for speed and precision
        results = self.model.predict(image_path, conf=conf_thresh, imgsz=640, verbose=False)
        annotations = []

        for r in results:
            boxes = r.boxes
            for b in boxes:
                raw_cls = int(b.cls[0])
                conf = float(b.conf[0])
                xywhn = b.xywhn[0].tolist()

                # Map: prompt 0 is supporting_tower, prompt 1 is monopole_tower
                cls_id = 0 if raw_cls == 0 else 1
                cls_name = "supporting_tower" if cls_id == 0 else "monopole_tower"

                annotations.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "x_center": round(xywhn[0], 6),
                    "y_center": round(xywhn[1], 6),
                    "width": round(xywhn[2], 6),
                    "height": round(xywhn[3], 6),
                    "status": "auto_generated"
                })

        # Fallback heuristic: If no high-confidence object detected, add a central candidate box
        # so the user can easily adjust or toggle it during manual review
        if not annotations:
            annotations.append({
                "class_id": 0,
                "class_name": "supporting_tower",
                "confidence": 0.20,
                "x_center": 0.500000,
                "y_center": 0.500000,
                "width": 0.400000,
                "height": 0.850000,
                "status": "needs_review"
            })

        return annotations

    def run_batch_autolabel(
        self,
        images_dir: str,
        output_dir: str,
        conf_thresh: float = 0.15
    ) -> Dict[str, Any]:
        """
        Auto-labels all images in images_dir and saves YOLO .txt files and manifest.
        """
        os.makedirs(output_dir, exist_ok=True)
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))

        manifest = {}
        total = len(image_files)
        print(f"Starting auto-labeling for {total} images...")

        for idx, img_path in enumerate(image_files, 1):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(output_dir, f"{base_name}.txt")

            labels = self.label_image(img_path, conf_thresh=conf_thresh)
            manifest[base_name] = {
                "image_path": img_path,
                "image_name": os.path.basename(img_path),
                "labels": labels
            }

            # Write standard YOLO format: class_id x_center y_center width height
            lines = []
            for item in labels:
                lines.append(
                    f"{item['class_id']} {item['x_center']:.6f} {item['y_center']:.6f} {item['width']:.6f} {item['height']:.6f}\n"
                )
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            if idx % 10 == 0 or idx == total:
                print(f"Processed {idx}/{total} images...")

        manifest_path = os.path.join(output_dir, "candidates.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {
            "total_processed": total,
            "manifest_path": manifest_path,
            "output_dir": output_dir
        }
