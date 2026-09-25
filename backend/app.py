"""
Flask REST API and Web Server for AI-Based Tower Component Detection and Visualization
Problem Statement 1 - ELECTROHACK 4.0 (KSIT & Nemilink Technologies)
"""

import os
import glob
import json
import threading
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

from backend.quality_filter import ImageQualityFilter
from backend.dataset_manager import DatasetManager
from backend.inference_engine import TowerInferenceEngine
from backend.model_trainer import TowerModelTrainer

app = Flask(__name__, static_folder="../frontend", static_url_path="")

UPLOAD_FOLDER = os.path.abspath("d:/college/TowerDetection/test_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max

# Initialize core services
dataset_mgr = DatasetManager()
quality_filter = ImageQualityFilter()
inference_engine = TowerInferenceEngine()

# Training state tracker
training_state = {
    "is_training": False,
    "progress": 0,
    "message": "Idle",
    "metrics": None,
    "error": None
}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# -------------------------------------------------------------------------
# INFERENCE & DASHBOARD ENDPOINTS (Phase 2 & Phase 3)
# -------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Handles image upload and saves to test_images directory."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in upload"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Read image to obtain dimensions and preview
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({"error": "Uploaded file is not a valid image"}), 400

    h, w = img.shape[:2]
    return jsonify({
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "width": w,
        "height": h,
        "size_kb": round(os.path.getsize(filepath) / 1024, 1)
    })


@app.route("/api/execute", methods=["POST"])
def execute_pipeline():
    """
    Executes the required flow:
    Upload -> Image Quality Check -> Object Detection -> Average Confidence Scores -> Annotated Image
    """
    data = request.json or {}
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "No filename specified for execution"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        # Also check raw_images
        filepath = os.path.join(dataset_mgr.raw_images_dir, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": f"Image file {filename} not found"}), 404

    conf_thresh = float(data.get("confidence_threshold", 0.20))
    result = inference_engine.process_image(filepath, conf_thresh=conf_thresh)

    return jsonify(result)


# -------------------------------------------------------------------------
# DATASET & AUTO-LABEL REVIEW ENDPOINTS (Phase 1)
# -------------------------------------------------------------------------

@app.route("/api/dataset/stats", methods=["GET"])
def get_dataset_stats():
    stats = dataset_mgr.compute_statistics()
    return jsonify(stats)


@app.route("/api/dataset/import", methods=["POST"])
def import_dataset():
    data = request.json or {}
    source = data.get("source_dir", r"C:\Users\Poonam Shetiya\Downloads\sample\sample")
    if not os.path.exists(source):
        return jsonify({"error": f"Source directory {source} does not exist"}), 400

    count = dataset_mgr.import_raw_dataset(source)
    stats = dataset_mgr.compute_statistics()
    return jsonify({
        "success": True,
        "imported_count": count,
        "stats": stats
    })


@app.route("/api/review/list", methods=["GET"])
def list_review_images():
    """Lists raw images and their current annotation status."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(dataset_mgr.raw_images_dir, ext)))

    files.sort(key=lambda x: os.path.basename(x))
    candidates_path = os.path.join(dataset_mgr.annotations_dir, "candidates.json")
    candidates = {}
    if os.path.exists(candidates_path):
        try:
            with open(candidates_path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
        except Exception:
            candidates = {}

    items = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        txt_path = os.path.join(dataset_mgr.annotations_dir, f"{base}.txt")
        has_annotation = os.path.exists(txt_path)

        meta = candidates.get(base, {})
        labels = dataset_mgr.get_annotation(base) if has_annotation else meta.get("labels", [])

        items.append({
            "filename": os.path.basename(f),
            "base_name": base,
            "has_annotation": has_annotation,
            "label_count": len(labels),
            "labels": labels
        })

    return jsonify({
        "total": len(items),
        "annotated_count": sum(1 for i in items if i["has_annotation"]),
        "images": items
    })


@app.route("/api/raw-image/<path:filename>")
def serve_raw_image(filename):
    return send_from_directory(dataset_mgr.raw_images_dir, filename)


@app.route("/api/test-image/<path:filename>")
def serve_test_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/review/save", methods=["POST"])
def save_review_label():
    data = request.json or {}
    base_name = data.get("base_name")
    labels = data.get("labels", [])

    if not base_name:
        return jsonify({"error": "Missing base_name"}), 400

    dataset_mgr.save_annotation(base_name, labels)
    return jsonify({
        "success": True,
        "base_name": base_name,
        "saved_labels": labels
    })


@app.route("/api/dataset/build", methods=["POST"])
def build_dataset_split():
    data = request.json or {}
    val_ratio = float(data.get("val_ratio", 0.2))
    res = dataset_mgr.split_and_build_dataset(val_ratio=val_ratio)
    stats = dataset_mgr.compute_statistics()
    return jsonify({
        "success": True,
        "split_summary": res,
        "stats": stats
    })


# -------------------------------------------------------------------------
# MODEL TRAINING & EVALUATION ENDPOINTS (Phase 2)
# -------------------------------------------------------------------------

def _run_training_background(epochs, batch_size):
    global training_state
    try:
        training_state["is_training"] = True
        training_state["progress"] = 10
        training_state["message"] = "Initializing YOLOv8 training architecture..."
        trainer = TowerModelTrainer()

        training_state["progress"] = 30
        training_state["message"] = f"Training YOLOv8 for {epochs} epochs..."
        metrics = trainer.train(epochs=epochs, batch_size=batch_size)

        training_state["progress"] = 100
        training_state["message"] = "Training and evaluation completed successfully!"
        training_state["metrics"] = metrics
        training_state["is_training"] = False

        # Reload inference engine with the freshly trained model
        inference_engine.reload_model(metrics["model_path"])
    except Exception as e:
        training_state["is_training"] = False
        training_state["error"] = str(e)
        training_state["message"] = f"Training failed: {str(e)}"


@app.route("/api/model/train", methods=["POST"])
def trigger_training():
    global training_state
    if training_state["is_training"]:
        return jsonify({"error": "Training is already in progress"}), 400

    data = request.json or {}
    epochs = int(data.get("epochs", 15))
    batch_size = int(data.get("batch_size", 4))

    training_state["is_training"] = True
    training_state["progress"] = 0
    training_state["message"] = "Starting training thread..."
    training_state["error"] = None
    training_state["metrics"] = None

    thread = threading.Thread(target=_run_training_background, args=(epochs, batch_size), daemon=True)
    thread.start()

    return jsonify({"success": True, "message": "Training started in background"})


@app.route("/api/model/status", methods=["GET"])
def get_model_status():
    summary_path = "d:/college/TowerDetection/model/evaluation_summary.json"
    cached_metrics = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                cached_metrics = json.load(f)
        except Exception:
            pass

    return jsonify({
        "training_state": training_state,
        "is_model_loaded": inference_engine.model is not None,
        "is_custom_trained": inference_engine.is_custom,
        "cached_metrics": cached_metrics
    })


@app.route("/api/model/artifacts/<path:filename>")
def serve_model_artifact(filename):
    model_dir = os.path.abspath("d:/college/TowerDetection/model")
    return send_from_directory(model_dir, filename)


if __name__ == "__main__":
    # Import sample dataset automatically if not already populated
    dm = DatasetManager()
    sample_src = r"C:\Users\Poonam Shetiya\Downloads\sample\sample"
    if os.path.exists(sample_src) and len(glob.glob(os.path.join(dm.raw_images_dir, "*"))) == 0:
        dm.import_raw_dataset(sample_src)

    print("Starting Flask Web Server on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
