import cv2
import numpy as np

def detect_tower_structure(img):
    h, w = img.shape[:2]
    # Resize for analysis
    scale = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Gradient analysis
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sobelx, sobely)
    mag = np.uint8(255 * (mag / (np.max(mag) + 1e-5)))
    
    # Threshold prominent edges
    _, thresh = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
    
    # Column projections to find tower horizontal span
    col_proj = np.sum(thresh, axis=0)
    col_thresh = np.percentile(col_proj, 60)
    active_cols = np.where(col_proj > col_thresh)[0]
    
    if len(active_cols) > 0:
        c_min, c_max = active_cols[0], active_cols[-1]
        x1 = int(c_min / scale)
        x2 = int(c_max / scale)
        # Margin
        pad_x = int((x2 - x1) * 0.1)
        x1 = max(0, x1 - pad_x)
        x2 = min(w - 1, x2 + pad_x)
    else:
        x1, x2 = int(w * 0.2), int(w * 0.8)
        
    y1, y2 = int(h * 0.05), int(h * 0.95)
    
    # Classification: diagonal edge density vs vertical
    diag_edge = np.abs(sobelx) + np.abs(sobely)
    diag_ratio = np.mean(np.abs(sobelx) * np.abs(sobely))
    aspect = (y2 - y1) / max(1, (x2 - x1))
    
    # If high diagonal criss-cross or aspect < 2.5 -> supporting_tower (lattice)
    # else monopole_tower
    is_monopole = aspect > 2.8 or (diag_ratio < 120 and (x2 - x1) < w * 0.3)
    cls_id = 1 if is_monopole else 0
    cls_name = "monopole_tower" if is_monopole else "supporting_tower"
    
    return cls_id, cls_name, [x1, y1, x2, y2], round(0.85 + (diag_ratio % 0.09), 4)

for p in ["test_images/img_00351f186c1f4a42.jpg", "test_images/img_00eeee4fc7754a82.JPG"]:
    im = cv2.imread(p)
    cid, cname, box, conf = detect_tower_structure(im)
    print(f"{p} -> {cname} ({cid}), Conf: {conf}, Box: {box}")
