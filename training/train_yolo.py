#!/usr/bin/env python3
"""
YOLOv8n-OBB Training Script for MLBB Hero Detection

Usage:
    python training/train_yolo.py --data training/dataset.yaml --epochs 100
    python training/train_yolo.py --data training/dataset.yaml --epochs 50 --batch 8
    python training/train_yolo.py --data training/dataset.yaml --validate-only
"""

import argparse
from pathlib import Path

def train(data_yaml: str, epochs: int = 100, batch_size: int = 16, img_size: int = 640, project: str = "runs/train"):
    """Train YOLOv8n-OBB model."""
    from ultralytics import YOLO
    
    model = YOLO("yolov8n-obb.pt")  # Load pretrained YOLOv8n-OBB
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name="mlbb_hero_detect",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
    )
    
    print(f"\nTraining complete! Results saved to {project}/mlbb_hero_detect/")
    return results

def validate(data_yaml: str, model_path: str = None, img_size: int = 640):
    """Validate trained model."""
    from ultralytics import YOLO
    
    if model_path is None:
        model_path = "runs/train/mlbb_hero_detect/weights/best.pt"
    
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=img_size)
    
    print(f"\nValidation Results:")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    
    return metrics

def export_model(model_path: str = None, format: str = "onnx"):
    """Export trained model to various formats."""
    from ultralytics import YOLO
    
    if model_path is None:
        model_path = "runs/train/mlbb_hero_detect/weights/best.pt"
    
    model = YOLO(model_path)
    model.export(format=format)
    
    print(f"\nModel exported to {format} format")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n-OBB Training for MLBB Hero Detection")
    parser.add_argument("--data", type=str, default="training/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory")
    parser.add_argument("--validate-only", action="store_true", help="Run validation only")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    parser.add_argument("--export-format", type=str, default="onnx", choices=["onnx", "torchscript", "tflite"], help="Export format")
    parser.add_argument("--model", type=str, default=None, help="Model path for validation/export")
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate(args.data, args.model, args.img_size)
    else:
        train(args.data, args.epochs, args.batch, args.img_size, args.project)
        
        if args.export:
            export_model(args.model, args.export_format)

if __name__ == "__main__":
    main()
