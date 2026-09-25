import os
import shutil
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("PHASE 2: YOLOv8 Training & Optimization for Tower Detection")
    print("Classes: 0 -> supporting_tower | 1 -> monopole_tower")
    print("=" * 60)

    # Load pre-trained nano backbone
    model = YOLO("yolov8n.pt")

    # Train on prepared Phase 1 dataset
    results = model.train(
        data="d:/college/TowerDetection/dataset/data.yaml",
        epochs=15,
        imgsz=384,
        batch=16,
        workers=2,
        project="d:/college/TowerDetection/runs",
        name="phase2_tower_model",
        exist_ok=True,
        verbose=True,
        patience=10,
        save=True
    )

    exp_dir = "d:/college/TowerDetection/runs/phase2_tower_model"
    os.makedirs("d:/college/TowerDetection/model", exist_ok=True)

    # Copy best weights
    best_src = os.path.join(exp_dir, "weights", "best.pt")
    best_dest = "d:/college/TowerDetection/model/best.pt"
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dest)
        print(f"[SUCCESS] Exported best model weights to: {best_dest}")

    # Copy evaluation artifacts
    eval_artifacts = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg"
    ]
    for art in eval_artifacts:
        src = os.path.join(exp_dir, art)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join("d:/college/TowerDetection/model", art))
            print(f"[ARTIFACT] Preserved: {art}")

    # Evaluate validation metrics
    val_model = YOLO(best_dest)
    metrics = val_model.val(data="d:/college/TowerDetection/dataset/data.yaml", split="val", imgsz=384)
    print("\n" + "=" * 60)
    print("PHASE 2 VALIDATION METRICS:")
    print(f"  Precision: {metrics.box.mp * 100:.2f}%")
    print(f"  Recall:    {metrics.box.mr * 100:.2f}%")
    print(f"  mAP50:     {metrics.box.map50 * 100:.2f}%")
    print(f"  mAP50-95:  {metrics.box.map * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
