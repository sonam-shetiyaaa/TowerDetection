"""
LabelImg Launcher for Telecom Tower Component Dataset
Launches LabelImg with predefined directories and YOLO classes:
  0: supporting_tower (Lattice / Cross-Bracing / Supporting Pole Tower)
  1: monopole_tower   (Single Vertical Cylindrical Pole)
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw_images")
CLASS_FILE = os.path.join(IMAGE_DIR, "classes.txt")
SAVE_DIR = IMAGE_DIR

LABELIMG_EXE = os.path.join(PROJECT_ROOT, "venv", "Scripts", "labelImg.exe")
PYTHON_EXE = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")

def main():
    print("=" * 68)
    print("        LAUNCHING LABELIMG FOR TOWER COMPONENT ANNOTATION")
    print("=" * 68)
    print(f"Image Directory : {IMAGE_DIR}")
    print(f"Classes File    : {CLASS_FILE}")
    print(f"Save Directory  : {SAVE_DIR}")
    print("\nPre-configured Classes:")
    print("  [0] supporting_tower (Lattice / framework cross-bracing tower)")
    print("  [1] monopole_tower   (Single vertical tubular pole tower)")
    print("=" * 68)
    print("Quick Shortcuts in LabelImg:")
    print("  • W       : Create a new bounding box")
    print("  • D       : Next image")
    print("  • A       : Previous image")
    print("  • Del     : Delete selected bounding box")
    print("  • Ctrl + S: Save annotation")
    print("=" * 68)
    print("\nStarting LabelImg GUI...")

    if os.path.exists(LABELIMG_EXE):
        cmd = [LABELIMG_EXE, IMAGE_DIR, CLASS_FILE, SAVE_DIR]
    else:
        cmd = [PYTHON_EXE, "-m", "labelImg", IMAGE_DIR, CLASS_FILE, SAVE_DIR]

    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error launching labelImg: {e}")

if __name__ == "__main__":
    main()
