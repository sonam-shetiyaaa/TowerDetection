"""
Image Quality Filtering Module for Tower Detection System
Performs automated pre-inference checks to reject unsuitable images:
1. Blurred images (Laplacian variance check)
2. Overexposed images (Saturated pixels and mean luminance check)
3. Underexposed images (Crushed shadow pixels and low luminance check)
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


class ImageQualityFilter:
    def __init__(
        self,
        blur_threshold: float = 45.0,
        overexposed_ratio_threshold: float = 0.85,
        overexposed_luminance_threshold: float = 248.0,
        underexposed_ratio_threshold: float = 0.70,
        underexposed_luminance_threshold: float = 25.0,
    ):
        """
        Initialize quality thresholds.
        Args:
            blur_threshold: Minimum Laplacian variance. Lower values indicate blur.
            overexposed_ratio_threshold: Maximum ratio of saturated pixels (>245).
            overexposed_luminance_threshold: Mean luminance threshold for overexposure.
            underexposed_ratio_threshold: Maximum ratio of dark pixels (<25).
            underexposed_luminance_threshold: Minimum mean luminance threshold for underexposure.
        """
        self.blur_threshold = blur_threshold
        self.overexposed_ratio_threshold = overexposed_ratio_threshold
        self.overexposed_luminance_threshold = overexposed_luminance_threshold
        self.underexposed_ratio_threshold = underexposed_ratio_threshold
        self.underexposed_luminance_threshold = underexposed_luminance_threshold

    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate image quality.
        Args:
            image: BGR numpy array from cv2.imread or decoded buffer.
        Returns:
            Dictionary containing evaluation decision, detailed metrics, and error reasons.
        """
        if image is None or image.size == 0:
            return {
                "passed": False,
                "status": "REJECTED",
                "reason": "Corrupted or empty image provided.",
                "metrics": {}
            }

        # Convert to grayscale for intensity and edge analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape
        total_pixels = h * w

        # 1. Blur Detection using Laplacian Variance
        # Downscale for very large images so variance doesn't explode artificially with megapixel count
        scale = min(1.0, 1024.0 / max(h, w))
        if scale < 1.0:
            resized_gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            resized_gray = gray

        laplacian_var = float(cv2.Laplacian(resized_gray, cv2.CV_64F).var())
        is_blurred = laplacian_var < self.blur_threshold

        # 2. Luminance & Exposure Metrics
        mean_luminance = float(np.mean(gray))

        # Overexposure: Ratio of pixels near pure white (>= 245)
        bright_pixels = int(np.sum(gray >= 245))
        overexposed_ratio = float(bright_pixels / total_pixels)
        is_overexposed = (
            overexposed_ratio > self.overexposed_ratio_threshold
            or mean_luminance > self.overexposed_luminance_threshold
        )

        # Underexposure: Ratio of pixels near pitch black (<= 25)
        dark_pixels = int(np.sum(gray <= 25))
        underexposed_ratio = float(dark_pixels / total_pixels)
        is_underexposed = (
            underexposed_ratio > self.underexposed_ratio_threshold
            or mean_luminance < self.underexposed_luminance_threshold
        )

        # Compile decision
        rejection_reasons = []
        if is_blurred:
            rejection_reasons.append(
                f"Image is blurred (Laplacian variance: {laplacian_var:.1f} < threshold: {self.blur_threshold})"
            )
        if is_overexposed:
            rejection_reasons.append(
                f"Image is overexposed (Highlight ratio: {overexposed_ratio*100:.1f}%, Mean luminance: {mean_luminance:.1f})"
            )
        if is_underexposed:
            rejection_reasons.append(
                f"Image is underexposed (Shadow ratio: {underexposed_ratio*100:.1f}%, Mean luminance: {mean_luminance:.1f})"
            )

        passed = len(rejection_reasons) == 0

        return {
            "passed": passed,
            "status": "ACCEPTED" if passed else "REJECTED",
            "reason": "Image passed all quality verification checks." if passed else " | ".join(rejection_reasons),
            "metrics": {
                "blur_score": round(laplacian_var, 2),
                "blur_threshold": self.blur_threshold,
                "is_blurred": is_blurred,
                "mean_luminance": round(mean_luminance, 2),
                "overexposed_ratio_pct": round(overexposed_ratio * 100, 2),
                "is_overexposed": is_overexposed,
                "underexposed_ratio_pct": round(underexposed_ratio * 100, 2),
                "is_underexposed": is_underexposed,
                "resolution": f"{w}x{h}"
            }
        }
