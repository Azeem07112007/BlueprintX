"""
train.py
Blueprint-X — Milestone 2

Script to train the YOLO11 segmentation model for blueprint understanding.
"""
import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # 1. Initialize YOLO11 segmentation model (nano version for speed, can use yolo11s-seg.pt or larger)
    model = YOLO("yolo11n-seg.pt")
    
    # 2. Train the model
    yaml_path = Path(__file__).parent / "blueprint_yolo.yaml"
    
    results = model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=1024,
        batch=8,
        device="cpu",  # change to 0 if CUDA is available
        project="runs",
        name="blueprint_seg",
        patience=20,
    )
    
    print("Training complete. Exporting model...")
    # 3. Export as ONNX or CoreML if needed
    model.export(format="onnx")

    # The resulting best weights will be in runs/blueprint_seg/weights/best.pt
    print("Best weights are saved in 'runs/blueprint_seg/weights/best.pt'.")
    print("Copy 'best.pt' to the 'training/weights/' directory for the Blueprint-X pipeline to use it.")

if __name__ == "__main__":
    main()
