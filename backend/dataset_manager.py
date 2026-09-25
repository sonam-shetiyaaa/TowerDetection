"""
Dataset Preparation and Management Module for Tower Detection
Handles:
1. Structuring raw images into YOLO format (train / val)
2. Generating data.yaml
3. Dataset verification, class balancing, and statistical analysis
"""

import os
import shutil
import glob
import random
import yaml
from typing import Dict, Any, List


class DatasetManager:
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            d_path = "d:/college/TowerDetection/dataset"
            local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
            root_dir = d_path if os.path.exists(d_path) else local_path
        self.root_dir = os.path.abspath(root_dir)
        self.images_train_dir = os.path.join(self.root_dir, "images", "train")
        self.images_val_dir = os.path.join(self.root_dir, "images", "val")
        self.labels_train_dir = os.path.join(self.root_dir, "labels", "train")
        self.labels_val_dir = os.path.join(self.root_dir, "labels", "val")
        self.raw_images_dir = os.path.join(self.root_dir, "raw_images")
        self.annotations_dir = os.path.join(self.root_dir, "annotations")
        self.yaml_path = os.path.join(self.root_dir, "data.yaml")

        self.class_names = {
            0: "supporting_tower",
            1: "monopole_tower"
        }

        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [
            self.images_train_dir, self.images_val_dir,
            self.labels_train_dir, self.labels_val_dir,
            self.raw_images_dir, self.annotations_dir
        ]:
            os.makedirs(d, exist_ok=True)

    def import_raw_dataset(self, source_dir: str) -> int:
        """Copies images from source folder into raw_images_dir."""
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        copied = 0
        for ext in extensions:
            for filepath in glob.glob(os.path.join(source_dir, ext)):
                dest = os.path.join(self.raw_images_dir, os.path.basename(filepath))
                if not os.path.exists(dest):
                    shutil.copy2(filepath, dest)
                copied += 1
        return copied

    def save_annotation(self, filename_base: str, labels: List[Dict[str, Any]]):
        """
        Saves labels in YOLO format: <class_id> <x_center> <y_center> <width> <height>
        """
        txt_path = os.path.join(self.annotations_dir, f"{filename_base}.txt")
        lines = []
        for item in labels:
            cls_id = int(item["class_id"])
            x = float(item["x_center"])
            y = float(item["y_center"])
            w = float(item["width"])
            h = float(item["height"])
            lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def get_annotation(self, filename_base: str) -> List[Dict[str, Any]]:
        txt_path = os.path.join(self.annotations_dir, f"{filename_base}.txt")
        if not os.path.exists(txt_path):
            return []
        items = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    items.append({
                        "class_id": cls_id,
                        "class_name": self.class_names.get(cls_id, "unknown"),
                        "x_center": float(parts[1]),
                        "y_center": float(parts[2]),
                        "width": float(parts[3]),
                        "height": float(parts[4]),
                    })
        return items

    def split_and_build_dataset(self, val_ratio: float = 0.2, seed: int = 42) -> Dict[str, Any]:
        """
        Splits annotated images into train & val sets, generating YOLO folder structure and data.yaml.
        """
        random.seed(seed)
        # Clear existing train/val directories
        for d in [self.images_train_dir, self.images_val_dir, self.labels_train_dir, self.labels_val_dir]:
            for f in glob.glob(os.path.join(d, "*")):
                os.remove(f)

        # Find all raw images that have annotations
        all_imgs = [
            f for f in glob.glob(os.path.join(self.raw_images_dir, "*"))
            if os.path.isfile(f)
        ]

        paired = []
        for img_path in all_imgs:
            base = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.annotations_dir, f"{base}.txt")
            if os.path.exists(txt_path):
                paired.append((img_path, txt_path))

        random.shuffle(paired)
        val_count = max(1, int(len(paired) * val_ratio))
        val_set = paired[:val_count]
        train_set = paired[val_count:]

        for img_path, txt_path in train_set:
            shutil.copy2(img_path, os.path.join(self.images_train_dir, os.path.basename(img_path)))
            shutil.copy2(txt_path, os.path.join(self.labels_train_dir, os.path.basename(txt_path)))

        for img_path, txt_path in val_set:
            shutil.copy2(img_path, os.path.join(self.images_val_dir, os.path.basename(img_path)))
            shutil.copy2(txt_path, os.path.join(self.labels_val_dir, os.path.basename(txt_path)))

        # Create data.yaml
        yaml_content = {
            "path": self.root_dir.replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "supporting_tower",
                1: "monopole_tower"
            }
        }
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        return {
            "total_annotated": len(paired),
            "train_count": len(train_set),
            "val_count": len(val_set),
            "yaml_path": self.yaml_path
        }

    def compute_statistics(self) -> Dict[str, Any]:
        """Audits dataset labels and counts instances per class."""
        class_counts = {0: 0, 1: 0}
        total_images = len(glob.glob(os.path.join(self.raw_images_dir, "*")))
        annotated_files = glob.glob(os.path.join(self.annotations_dir, "*.txt"))
        total_boxes = 0

        for txt_path in annotated_files:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                        total_boxes += 1

        train_imgs = len(glob.glob(os.path.join(self.images_train_dir, "*")))
        val_imgs = len(glob.glob(os.path.join(self.images_val_dir, "*")))

        return {
            "total_raw_images": total_images,
            "total_annotated_images": len(annotated_files),
            "unannotated_images": total_images - len(annotated_files),
            "total_bounding_boxes": total_boxes,
            "classes": {
                "supporting_tower": {
                    "id": 0,
                    "count": class_counts.get(0, 0)
                },
                "monopole_tower": {
                    "id": 1,
                    "count": class_counts.get(1, 0)
                }
            },
            "split": {
                "train_images": train_imgs,
                "val_images": val_imgs
            }
        }


if __name__ == "__main__":
    dm = DatasetManager()
    source = r"C:\Users\Poonam Shetiya\Downloads\sample\sample"
    count = dm.import_raw_dataset(source)
    print(f"Imported {count} images into {dm.raw_images_dir}")
    stats = dm.compute_statistics()
    print("Dataset Stats:", stats)
