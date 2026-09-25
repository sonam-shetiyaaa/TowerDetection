# AI-Based Tower Component Detection and Visualization

**ELECTROHACK 4.0 — Problem Statement 1**  
*Organized by KSIT & Nemilink Technologies*

An end-to-end Computer Vision system and Web Dashboard for detecting, classifying, and visualizing telecom towers:
- **`supporting_tower`**: Self-supporting steel lattice/framework structures with cross-bracing (triangular or square profile).
- **`monopole_tower`**: Single vertical pole structures with antenna mounts on top and no lattice framework.

---

## Key Features

1. **Phase 1: Dataset Pipeline & YOLO Annotation**
   - 124 telecom tower images prepared with normalized YOLO bounding boxes.
   - Train/Validation dataset split (80/20) with verified `data.yaml`.
   - Automated annotation verification with visual bounding box overlays.

2. **Phase 2: Quality Filtering & Model Training**
   - **Pre-Inference Quality Filter**: Evaluates image clarity (Laplacian variance blur score $\ge 100$) and exposure (flags underexposed/overexposed captures).
   - **Trained YOLOv8 Object Detection Model**: Custom-trained model with saved checkpoints (`model/best.pt`), confusion matrix, and precision-recall metrics.

3. **Phase 3: Interactive Web Dashboard & Multi-Strategy Detection**
   - Responsive dark-mode dashboard with drag-and-drop upload and progress indicator.
   - Dual-engine fallback: YOLOv8 local detector + Zero-shot heuristic/aspect ratio analysis for edge-case coverage.
   - Real-time bounding box visualization with class tags and average confidence score computation.
   - Image quality diagnostics overlay (Laplacian variance blur index, exposure check).

4. **Batch Classifier CLI & CSV Export**
   - Standalone utility (`classify_towers.py`) to process an entire folder of tower images.
   - Automatically sorts photos into `supporting_tower/`, `monopole_tower/`, and `unsure/`.
   - Exports formatted `classification_results.csv` with predictions and confidence metrics.

---

## Project Structure

```text
TowerDetection/
├── backend/
│   ├── app.py                 # Flask REST API server
│   ├── inference_engine.py    # YOLOv8 + fallback tower detection engine
│   └── quality_filter.py      # Laplacian blur and exposure assessment
├── frontend/
│   ├── index.html             # Dashboard UI
│   ├── style.css              # Cyber-industrial dark UI styling
│   └── app.js                 # Upload handling, canvas annotation & metrics
├── dataset/
│   ├── data.yaml              # YOLO dataset configuration
│   ├── images/train & val     # Split training/validation images
│   ├── labels/train & val     # Normalized YOLO bounding boxes
│   ├── raw_images/            # Original 124 dataset images
│   └── verification_previews/ # Visual verification samples
├── model/
│   ├── best.pt                # Trained YOLOv8 model weights
│   ├── confusion_matrix.png   # Model evaluation confusion matrix
│   └── results.png            # Training curves and loss progression
├── classify_towers.py         # Batch classification script
├── prepare_phase1_dataset.py  # Dataset split and label generation script
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/sonam-shetiyaaa/TowerDetection.git
cd TowerDetection
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launching the Web Dashboard
Start the Flask backend:
```bash
python -m backend.app
```
Open `http://localhost:5000` in any modern web browser to interact with the dashboard.

### 3. Batch Classification Script
To classify a folder of tower images and export results to CSV:
```bash
python classify_towers.py --input test_images --output classified_towers --csv classification_results.csv
```

---

## API Endpoints

- **`POST /api/detect`**: Accepts `multipart/form-data` with an `image` file and optional `min_conf` parameter. Returns quality filter metrics, bounding boxes, classifications, and average confidence score.
- **`GET /api/health`**: Health check returning model status, device (CPU/CUDA), and supported classes.

---

## Model Evaluation

| Metric | Target | Status |
|---|---|---|
| Classification Precision | Supporting & Monopole | High Recall & Precision |
| Quality Filter | Laplacian Variance $\ge 100$ | Passed |
| Real-Time Inference | Low latency ($< 150\text{ ms}$) | Enabled |

---

## Authors & Acknowledgments
- Developed for **ELECTROHACK 4.0** (KSIT & Nemilink Technologies).
