"""
Telecom Tower Image Classification using OpenAI CLIP
Problem Statement: AI-Based Tower Component Classification (ELECTROHACK 4.0)

Classes:
1. supporting_tower: Self-supporting lattice telecom tower with steel cross-bracing framework.
2. monopole_tower: Single vertical tubular steel pole with antennas mounted on top.
3. unsure: Low-confidence or ambiguous images.

Outputs:
- Copies images into categorized folders:
    classified_towers/supporting_tower/
    classified_towers/monopole_tower/
    classified_towers/unsure/
- Generates classification_results.csv (image_name, predicted_class, confidence)
"""

import os
import glob
import shutil
import csv
import argparse
import torch
import clip
from PIL import Image
from tqdm import tqdm


def get_text_features(model, device):
    """
    Creates ensemble text prompt embeddings for supporting_tower and monopole_tower.
    """
    supporting_prompts = [
        "a photo of a self-supporting steel lattice telecom tower with cross-bracing framework",
        "a steel lattice telecommunication tower with triangular or square metal truss structure",
        "an electricity transmission lattice tower pylon",
        "a lattice framework antenna tower with open diagonal metal girders"
    ]

    monopole_prompts = [
        "a photo of a monopole telecom tower, single vertical tubular pole with antennas on top",
        "a single steel pole antenna tower without lattice framework",
        "a cylindrical metal pole telecommunication tower",
        "a slim vertical utility pole cell tower"
    ]

    with torch.no_grad():
        tokens_sup = clip.tokenize(supporting_prompts).to(device)
        feats_sup = model.encode_text(tokens_sup)
        feats_sup = feats_sup / feats_sup.norm(dim=-1, keepdim=True)
        mean_sup = feats_sup.mean(dim=0, keepdim=True)
        mean_sup = mean_sup / mean_sup.norm(dim=-1, keepdim=True)

        tokens_mono = clip.tokenize(monopole_prompts).to(device)
        feats_mono = model.encode_text(tokens_mono)
        feats_mono = feats_mono / feats_mono.norm(dim=-1, keepdim=True)
        mean_mono = feats_mono.mean(dim=0, keepdim=True)
        mean_mono = mean_mono / mean_mono.norm(dim=-1, keepdim=True)

        # Shape: (2, feature_dim)
        text_features = torch.cat([mean_sup, mean_mono], dim=0)

    return text_features


def classify_images(
    input_dir: str,
    output_dir: str,
    csv_file: str,
    confidence_thresh: float = 0.60,
    model_name: str = "ViT-B/32"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running on device: {device.upper()}")
    print(f"[*] Loading CLIP model ({model_name})...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    # Precompute class text embeddings
    text_features = get_text_features(model, device)
    classes = ["supporting_tower", "monopole_tower"]

    # Prepare destination directories
    folders = {
        "supporting_tower": os.path.join(output_dir, "supporting_tower"),
        "monopole_tower": os.path.join(output_dir, "monopole_tower"),
        "unsure": os.path.join(output_dir, "unsure")
    }
    for p in folders.values():
        os.makedirs(p, exist_ok=True)

    # Gather images
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.webp")
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(image_paths)

    total_images = len(image_paths)
    if total_images == 0:
        print(f"[!] No images found in: {input_dir}")
        return

    print(f"[*] Found {total_images} images to process in '{input_dir}'")
    print(f"[*] Output directory: '{output_dir}'")
    print(f"[*] Confidence threshold for 'unsure': {confidence_thresh * 100:.1f}%\n")

    results_data = []
    stats = {"supporting_tower": 0, "monopole_tower": 0, "unsure": 0}

    print(f"{'IMAGE NAME':<32} | {'PREDICTED CLASS':<18} | {'CONFIDENCE':<10}")
    print("-" * 68)

    for img_path in tqdm(image_paths, desc="Classifying", unit="image"):
        img_name = os.path.basename(img_path)
        try:
            pil_image = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                img_features = model.encode_image(img_tensor)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

                # Cosine similarity scaled by 100 (CLIP standard temperature)
                logits = (100.0 * img_features @ text_features.T).softmax(dim=-1)
                probs = logits[0].cpu().numpy()

            sup_prob = float(probs[0])
            mono_prob = float(probs[1])

            # Classify
            if sup_prob >= mono_prob:
                top_class = "supporting_tower"
                confidence = sup_prob
            else:
                top_class = "monopole_tower"
                confidence = mono_prob

            # Ambiguity / low confidence check
            margin = abs(sup_prob - mono_prob)
            if confidence < confidence_thresh or margin < 0.12:
                final_class = "unsure"
            else:
                final_class = top_class

            stats[final_class] += 1

            # Copy image to target category folder
            dest_path = os.path.join(folders[final_class], img_name)
            shutil.copy2(img_path, dest_path)

            results_data.append({
                "image_name": img_name,
                "predicted_class": final_class,
                "confidence": f"{confidence:.4f}"
            })

            # Print single-line progress
            tqdm.write(f"{img_name[:30]:<32} | {final_class:<18} | {confidence*100:6.2f}%")

        except Exception as e:
            tqdm.write(f"[ERROR] Failed to process {img_name}: {str(e)}")
            results_data.append({
                "image_name": img_name,
                "predicted_class": "unsure",
                "confidence": "0.0000"
            })
            stats["unsure"] += 1

    # Write CSV
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "predicted_class", "confidence"])
        writer.writeheader()
        writer.writerows(results_data)

    print("\n" + "=" * 68)
    print("                     CLASSIFICATION SUMMARY")
    print("=" * 68)
    print(f"Total Images Processed : {total_images}")
    print(f" - supporting_tower    : {stats['supporting_tower']} ({(stats['supporting_tower']/total_images)*100:.1f}%)")
    print(f" - monopole_tower      : {stats['monopole_tower']} ({(stats['monopole_tower']/total_images)*100:.1f}%)")
    print(f" - unsure              : {stats['unsure']} ({(stats['unsure']/total_images)*100:.1f}%)")
    print("-" * 68)
    print(f"[✓] CSV Results saved to : {os.path.abspath(csv_file)}")
    print(f"[✓] Sorted images saved to: {os.path.abspath(output_dir)}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom Tower Image Classification using Vision Model (CLIP)")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="dataset/raw_images",
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="classified_towers",
        help="Folder to save sorted images"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="classification_results.csv",
        help="Path for output CSV file"
    )
    parser.add_argument(
        "--confidence_thresh",
        type=float,
        default=0.60,
        help="Confidence threshold below which images are classified as 'unsure'"
    )
    args = parser.parse_args()

    classify_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_file=args.csv_file,
        confidence_thresh=args.confidence_thresh
    )
