#!/usr/bin/env python3
"""
Synthetic Hero Icon Image Generator for YOLOv8n-OBB Training

Generates 5,000 synthetic training images with hero icons placed at random positions.
Each image includes YOLO OBB labels for oriented bounding box detection.

Augmentations applied:
- Random rotation (±15°)
- Gaussian noise
- Transparency shifts (simulating "selection glow")
- Random brightness/contrast

Usage:
    python training/generate_images.py
    python training/generate_images.py --count 5000 --output training/data/synthetic_images
"""

import argparse
import json
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2


# Role to class index mapping
ROLE_CLASSES = {
    "Fighter": 0,
    "Assassin": 1,
    "Mage": 2,
    "Tank": 3,
    "Support": 4,
    "Marksman": 5,
}

# Colors for each role (BGR)
ROLE_COLORS = {
    "Fighter": (0, 0, 255),      # Red
    "Assassin": (200, 0, 200),   # Purple
    "Mage": (255, 100, 0),       # Blue-ish
    "Tank": (0, 150, 0),         # Green
    "Support": (0, 200, 255),    # Yellow
    "Marksman": (255, 0, 0),     # Blue
}

ROLE_NAMES = list(ROLE_CLASSES.keys())


def load_heroes(hero_meta_path: str) -> List[Dict]:
    """Load hero metadata."""
    with open(hero_meta_path) as f:
        data = json.load(f)
    # Handle both {"heroes": [...]} and [...] formats
    if isinstance(data, dict) and "heroes" in data:
        return data["heroes"]
    return data


def create_hero_icon(size: int, role: str, hero_name: str, glow: bool = False) -> np.ndarray:
    """Create a synthetic hero icon image.
    
    Args:
        size: Icon size in pixels
        role: Hero role for color coding
        hero_name: Name to render on icon
        glow: Whether to add selection glow effect
    
    Returns:
        RGBA icon image
    """
    icon = np.zeros((size, size, 4), dtype=np.uint8)
    color = ROLE_COLORS.get(role, (128, 128, 128))
    
    # Draw rounded rectangle background
    margin = size // 8
    cv2.rectangle(icon, (margin, margin), (size - margin, size - margin), (*color, 220), -1)
    cv2.rectangle(icon, (margin, margin), (size - margin, size - margin), (255, 255, 255, 180), 2)
    
    # Draw role circle
    circle_r = size // 6
    circle_center = (size // 2, size // 3)
    cv2.circle(icon, circle_center, circle_r, (*color, 255), -1)
    cv2.circle(icon, circle_center, circle_r, (255, 255, 255, 200), 2)
    
    # Draw hero name text
    text = hero_name[:8]  # Truncate long names
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = size / 200.0
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    text_x = (size - text_size[0]) // 2
    text_y = size - margin // 2
    cv2.putText(icon, text, (text_x, text_y), font, font_scale, (255, 255, 255, 255), 2)
    
    # Add selection glow effect
    if glow:
        glow_layer = np.zeros_like(icon)
        glow_color = (*color, 60)
        cv2.rectangle(glow_layer, (0, 0), (size, size), glow_color, -1)
        icon = cv2.add(icon, glow_layer)
    
    return icon


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees with border padding."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    return cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def add_gaussian_noise(image: np.ndarray, intensity: int = 25) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def compute_obb_corners(x: float, y: float, w: float, h: float, angle: float) -> List[float]:
    """Compute 4 corners of oriented bounding box.
    
    Args:
        x, y: Top-left corner (normalized)
        w, h: Width and height (normalized)
        angle: Rotation angle in degrees
    
    Returns:
        List of 8 values: x1 y1 x2 y2 x3 y3 x4 y4 (clockwise from top-left)
    """
    cx = x + w / 2
    cy = y + h / 2
    
    # Half dimensions
    hw, hh = w / 2, h / 2
    
    # Corners relative to center (before rotation)
    corners = [
        (-hw, -hh),  # top-left
        (hw, -hh),   # top-right
        (hw, hh),    # bottom-right
        (-hw, hh),   # bottom-left
    ]
    
    # Rotate corners
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    
    rotated = []
    for px, py in corners:
        rx = px * cos_a - py * sin_a + cx
        ry = px * sin_a + py * cos_a + cy
        rotated.extend([rx, ry])
    
    return rotated


def generate_image(
    img_size: int,
    heroes: List[Dict],
    num_icons: int,
    output_dir: Path,
    img_idx: int,
    split: str,
) -> None:
    """Generate a single training image with labels."""
    # Create blank canvas (dark background like MLBB client)
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 40)  # Dark background
    
    labels = []
    placed = []
    
    for _ in range(num_icons):
        # Pick random hero
        hero = random.choice(heroes)
        role = hero.get("role", random.choice(ROLE_NAMES))
        if role not in ROLE_CLASSES:
            role = random.choice(ROLE_NAMES)
        class_id = ROLE_CLASSES[role]
        
        # Icon size (20-40% of canvas)
        icon_size = random.randint(img_size // 5, int(img_size // 2.5))
        
        # Random position (ensure within bounds)
        max_x = img_size - icon_size
        max_y = img_size - icon_size
        if max_x <= 0 or max_y <= 0:
            continue
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # Check overlap with existing icons (simple IoU check)
        overlap = False
        for px, py, ps in placed:
            ix1 = max(x, px)
            iy1 = max(y, py)
            ix2 = min(x + icon_size, px + ps)
            iy2 = min(y + icon_size, py + ps)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = icon_size * icon_size + ps * ps - inter
            if union > 0 and inter / union > 0.3:
                overlap = True
                break
        if overlap:
            continue
        
        # Create hero icon with random glow
        glow = random.random() < 0.3
        icon = create_hero_icon(icon_size, role, hero["name"], glow=glow)
        
        # Random rotation ±15°
        angle = random.uniform(-15, 15)
        if abs(angle) > 1:
            icon = rotate_image(icon, angle)
        
        # Get actual rotated icon size
        ih, iw = icon.shape[:2]
        
        # Adjust position for rotated size
        paste_x = max(0, min(x, img_size - iw))
        paste_y = max(0, min(y, img_size - ih))
        
        # Paste icon onto canvas (alpha composite)
        roi = canvas[paste_y:paste_y + ih, paste_x:paste_x + iw]
        if icon.shape[2] == 4:
            alpha = icon[:, :, 3:] / 255.0
            bgr = icon[:, :, :3]
            if roi.shape == bgr.shape:
                canvas[paste_y:paste_y + ih, paste_x:paste_x + iw] = (
                    bgr * alpha + roi * (1 - alpha)
                ).astype(np.uint8)
        
        # Compute OBB label (normalized)
        obb = compute_obb_corners(
            paste_x / img_size,
            paste_y / img_size,
            iw / img_size,
            ih / img_size,
            angle
        )
        labels.append(f"{class_id} {' '.join(f'{v:.6f}' for v in obb)}")
        placed.append((paste_x, paste_y, icon_size))
    
    # Add Gaussian noise (augmentation)
    if random.random() < 0.5:
        canvas = add_gaussian_noise(canvas, intensity=random.randint(10, 30))
    
    # Save image
    img_path = output_dir / "images" / split / f"img_{img_idx:06d}.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), canvas)
    
    # Save label
    lbl_path = output_dir / "labels" / split / f"img_{img_idx:06d}.txt"
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w") as f:
        f.write("\n".join(labels))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic hero icon training images")
    parser.add_argument("--count", type=int, default=5000, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="training/data/synthetic_images", help="Output directory")
    parser.add_argument("--img-size", type=int, default=640, help="Image size (square)")
    parser.add_argument("--hero-meta", type=str, default="shared/hero_meta.json", help="Hero metadata path")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()
    
    heroes = load_heroes(args.hero_meta)
    print(f"Loaded {len(heroes)} heroes")
    
    output_dir = Path(args.output)
    val_count = int(args.count * args.val_split)
    train_count = args.count - val_count
    
    print(f"Generating {train_count} train + {val_count} val = {args.count} images")
    print(f"Output: {output_dir}/")
    
    for i in range(args.count):
        split = "val" if i < val_count else "train"
        num_icons = random.choices([1, 2, 3, 4, 5], weights=[10, 25, 30, 25, 10])[0]
        generate_image(args.img_size, heroes, num_icons, output_dir, i, split)
        
        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{args.count}] generated")
    
    print(f"\nGeneration complete!")
    print(f"  Train: {train_count} images in {output_dir}/images/train/")
    print(f"  Val:   {val_count} images in {output_dir}/images/val/")
    print(f"  Labels: {output_dir}/labels/")


if __name__ == "__main__":
    main()
