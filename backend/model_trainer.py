"""
Model Training and Evaluation Pipeline for Tower Component Detection
Trains YOLOv8 on the prepared dataset and generates evaluation metrics:
- Precision, Recall, mAP50, mAP50-95
- Confusion Matrix
- Loss Curves & PR Curves
"""

import os
import shutil
import json
from typing import Dict, Any
from ultralytics import YOLO


class TowerModelTrainer:
    def __init__(
        self,
        yaml_path: str = "d:/college/TowerDetection/dataset/data.yaml",
        output_model_dir: str = "d:/college/TowerDetection/model",
        base_weights: str = "yolov8n.pt"
    ):
        self.yaml_path = os.path.abspath(yaml_path)
        self.output_model_dir = os.path.abspath(output_model_dir)
        self.base_weights = base_weights
        os.makedirs(self.output_model_dir, exist_ok=True)

    def train(
        self,
        epochs: int = 15,
        imgsz: int = 640,
        batch_size: int = 4,
        patience: int = 8
    ) -> Dict[str, Any]:
        """
        Executes YOLO fine-tuning on the custom tower dataset.
        """
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"data.yaml not found at {self.yaml_path}. Build dataset first.")

        print(f"Initializing YOLO model from {self.base_weights}...")
        model = YOLO(self.base_weights)

        runs_dir = "d:/college/TowerDetection/runs"
        print(f"Starting training for {epochs} epochs (imgsz={imgsz}, batch={batch_size})...")

        results = model.train(
            data=self.yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            project=runs_dir,
            name="train_exp",
            exist_ok=True,
            verbose=True
        )

        # Path to best model
        exp_dir = os.path.join(runs_dir, "train_exp")
        weights_best = os.path.join(exp_dir, "weights", "best.pt")
        dest_best = os.path.join(self.output_model_dir, "best.pt")

        if os.path.exists(weights_best):
            shutil.copy2(weights_best, dest_best)
            print(f"Saved best model weights to {dest_best}")
        else:
            weights_last = os.path.join(exp_dir, "weights", "last.pt")
            if os.path.exists(weights_last):
                shutil.copy2(weights_last, dest_best)

        # Validate trained model
        val_metrics = model.val(data=self.yaml_path)

        metrics_data = {
            "mAP50": round(float(val_metrics.box.map50), 4),
            "mAP50_95": round(float(val_metrics.box.map), 4),
            "precision": round(float(val_metrics.box.mp), 4),
            "recall": round(float(val_metrics.box.mr), 4),
            "classes": ["supporting_tower", "monopole_tower"],
            "model_path": dest_best,
            "exp_dir": exp_dir
        }

        # Check for generated plot artifacts
        artifacts = {}
        for plot_name in ["confusion_matrix.png", "results.png", "PR_curve.png", "F1_curve.png"]:
            src = os.path.join(exp_dir, plot_name)
            if os.path.exists(src):
                dest = os.path.join(self.output_model_dir, plot_name)
                shutil.copy2(src, dest)
                artifacts[plot_name] = dest

        metrics_data["artifacts"] = artifacts

        summary_file = os.path.join(self.output_model_dir, "evaluation_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        return metrics_data


if __name__ == "__main__":
    trainer = TowerModelTrainer()
    metrics = trainer.train(epochs=10, batch_size=4)
    print("Training finished with metrics:", metrics)
