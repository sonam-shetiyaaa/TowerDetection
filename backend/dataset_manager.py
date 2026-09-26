"""
Dataset Preparation and Management Module for Tower Detection
Handles:
1. Pascal VOC XML & YOLO format reading, conversion, and synchronization
2. Structuring raw images into YOLO format (train / val)
3. Generating data.yaml
4. Dataset verification, class balancing, and statistical analysis
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
        self.labeled_images_dir = os.path.join(self.root_dir, "labeled_images")
        self.annotations_dir = os.path.join(self.root_dir, "annotations")
        self.yaml_path = os.path.join(self.root_dir, "data.yaml")

        self.class_names = {
            0: "supporting_tower",
            1: "monopole_tower"
        }

        self._ensure_dirs()
        # Automatically sync any Pascal VOC XML annotations found into YOLO format
        self.sync_all_xml_annotations()

    def _ensure_dirs(self):
        for d in [
            self.images_train_dir, self.images_val_dir,
            self.labels_train_dir, self.labels_val_dir,
            self.raw_images_dir, self.labeled_images_dir, self.annotations_dir
        ]:
            os.makedirs(d, exist_ok=True)

    def import_raw_dataset(self, source_dir: str) -> int:
        """Copies images and annotations from source folder into raw_images_dir."""
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.webp")
        copied = 0
        for ext in extensions:
            for filepath in glob.glob(os.path.join(source_dir, ext)):
                dest = os.path.join(self.raw_images_dir, os.path.basename(filepath))
                if not os.path.exists(dest):
                    shutil.copy2(filepath, dest)
                copied += 1

        # Also copy XMLs if present
        for xmlpath in glob.glob(os.path.join(source_dir, "*.xml")):
            dest = os.path.join(self.labeled_images_dir, os.path.basename(xmlpath))
            shutil.copy2(xmlpath, dest)

        self.sync_all_xml_annotations()
        return copied

    def _map_class_name_to_id(self, name_str: str) -> int:
        """Normalizes various annotation class name strings to standard class IDs (0 or 1)."""
        name_clean = str(name_str).lower().strip().replace("-", "_").replace(" ", "_")
        if any(k in name_clean for k in ["monopole", "pole", "single_pole", "telecom_pole"]):
            return 1  # monopole_tower
        return 0  # supporting_tower / lattice default

    def parse_voc_xml(self, xml_path: str) -> List[Dict[str, Any]]:
        """
        Parses Pascal VOC XML format annotation and converts to normalized YOLO coordinates.
        """
        import xml.etree.ElementTree as ET
        if not os.path.exists(xml_path):
            return []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            return []

        # Find image width and height
        img_w, img_h = 0, 0
        size_elem = root.find("size")
        if size_elem is not None:
            w_e = size_elem.find("width")
            h_e = size_elem.find("height")
            if w_e is not None and h_e is not None:
                try:
                    img_w = float(w_e.text)
                    img_h = float(h_e.text)
                except (ValueError, TypeError):
                    img_w, img_h = 0, 0

        # If size is missing from XML, attempt to read actual image from raw_images
        base = os.path.splitext(os.path.basename(xml_path))[0]
        if img_w <= 0 or img_h <= 0:
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp"]:
                candidate = os.path.join(self.raw_images_dir, f"{base}{ext}")
                if os.path.exists(candidate):
                    try:
                        from PIL import Image
                        with Image.open(candidate) as im:
                            img_w, img_h = float(im.width), float(im.height)
                        break
                    except Exception:
                        pass

        if img_w <= 0 or img_h <= 0:
            img_w, img_h = 1000.0, 1000.0  # Safe fallback to prevent div-by-zero

        labels = []
        for obj in root.findall("object"):
            name_e = obj.find("name")
            name = name_e.text if name_e is not None else "supporting_tower"
            cls_id = self._map_class_name_to_id(name)

            bnd = obj.find("bndbox")
            if bnd is None:
                continue

            try:
                xmin = float(bnd.find("xmin").text)
                ymin = float(bnd.find("ymin").text)
                xmax = float(bnd.find("xmax").text)
                ymax = float(bnd.find("ymax").text)
            except Exception:
                continue

            # Ensure valid bounds
            xmin = max(0.0, min(xmin, img_w))
            ymin = max(0.0, min(ymin, img_h))
            xmax = max(0.0, min(xmax, img_w))
            ymax = max(0.0, min(ymax, img_h))

            box_w = max(1.0, xmax - xmin)
            box_h = max(1.0, ymax - ymin)
            x_center = (xmin + xmax) / (2.0 * img_w)
            y_center = (ymin + ymax) / (2.0 * img_h)
            w_norm = box_w / img_w
            h_norm = box_h / img_h

            labels.append({
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, "unknown"),
                "x_center": round(min(1.0, max(0.0, x_center)), 6),
                "y_center": round(min(1.0, max(0.0, y_center)), 6),
                "width": round(min(1.0, max(0.0, w_norm)), 6),
                "height": round(min(1.0, max(0.0, h_norm)), 6),
                "bbox_raw": [int(xmin), int(ymin), int(xmax), int(ymax)]
            })

        return labels

    def sync_all_xml_annotations(self) -> int:
        """
        Scans labeled_images/, annotations/, and raw_images/ for Pascal VOC XML files,
        automatically converting them into normalized YOLO .txt annotations.
        """
        search_dirs = [
            self.labeled_images_dir,
            self.annotations_dir,
            self.raw_images_dir
        ]
        synced_count = 0
        for sdir in search_dirs:
            if not os.path.exists(sdir):
                continue
            for xml_file in glob.glob(os.path.join(sdir, "*.xml")):
                base = os.path.splitext(os.path.basename(xml_file))[0]
                labels = self.parse_voc_xml(xml_file)
                if labels:
                    self.save_annotation(base, labels)
                    synced_count += 1

        return synced_count

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
        
        # If txt doesn't exist, check if XML exists and convert on the fly
        if not os.path.exists(txt_path):
            xml_candidates = [
                os.path.join(self.labeled_images_dir, f"{filename_base}.xml"),
                os.path.join(self.annotations_dir, f"{filename_base}.xml"),
                os.path.join(self.raw_images_dir, f"{filename_base}.xml")
            ]
            for xc in xml_candidates:
                if os.path.exists(xc):
                    labels = self.parse_voc_xml(xc)
                    if labels:
                        self.save_annotation(filename_base, labels)
                        return labels
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
        self.sync_all_xml_annotations()
        random.seed(seed)
        
        # Clear existing train/val directories
        for d in [self.images_train_dir, self.images_val_dir, self.labels_train_dir, self.labels_val_dir]:
            for f in glob.glob(os.path.join(d, "*")):
                os.remove(f)

        # Find all raw images that have annotations
        all_imgs = [
            f for f in glob.glob(os.path.join(self.raw_images_dir, "*"))
            if os.path.isfile(f) and not f.endswith(".txt") and not f.endswith(".xml") and not f.endswith(".json")
        ]

        paired = []
        for img_path in all_imgs:
            base = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.annotations_dir, f"{base}.txt")
            if not os.path.exists(txt_path):
                # Check XML
                xml_path = os.path.join(self.labeled_images_dir, f"{base}.xml")
                if os.path.exists(xml_path):
                    lbls = self.parse_voc_xml(xml_path)
                    if lbls:
                        self.save_annotation(base, lbls)

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
        self.sync_all_xml_annotations()

        class_counts = {0: 0, 1: 0}
        img_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
        raw_imgs = set()
        for ext in img_exts:
            for p in glob.glob(os.path.join(self.raw_images_dir, ext)):
                raw_imgs.add(os.path.normcase(os.path.abspath(p)))
        total_images = len(raw_imgs)

        annotated_files = [
            f for f in glob.glob(os.path.join(self.annotations_dir, "*.txt"))
            if os.path.basename(f) != "classes.txt"
        ]
        total_boxes = 0

        for txt_path in annotated_files:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                        total_boxes += 1

        train_imgs = set()
        for ext in img_exts:
            for p in glob.glob(os.path.join(self.images_train_dir, ext)):
                train_imgs.add(os.path.normcase(os.path.abspath(p)))

        val_imgs = set()
        for ext in img_exts:
            for p in glob.glob(os.path.join(self.images_val_dir, ext)):
                val_imgs.add(os.path.normcase(os.path.abspath(p)))

        return {
            "total_raw_images": total_images,
            "total_annotated_images": len(annotated_files),
            "unannotated_images": max(0, total_images - len(annotated_files)),
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
                "train_images": len(train_imgs),
                "val_images": len(val_imgs)
            }
        }


if __name__ == "__main__":
    dm = DatasetManager()
    stats = dm.compute_statistics()
    print("Dataset Stats:", stats)
