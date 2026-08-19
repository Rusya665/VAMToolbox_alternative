#!/usr/bin/env python3
"""
Generate Optical Calibration Targets for OpenCAL + Optoma ML1080 (1080x1920 Portrait)
Produces:
  1. calibration_axis_and_grid.png  - Axis of Rotation (AoR), Vial Boundaries & Keystone Grid
  2. calibration_siemens_star.png   - High-frequency Siemens Star for precision focusing
  3. calibration_aruco_checker.png   - Vision-based auto-keystone / homography target
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "calibration_targets")
os.makedirs(OUT_DIR, exist_ok=True)

WIDTH  = 1080
HEIGHT = 1920

# ── 1. Axis of Rotation (AoR), Grid & Vial Boundaries Target ─────────────────
def generate_axis_grid_target(vial_width_px=600):
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = WIDTH // 2, HEIGHT // 2
    half_vial = vial_width_px // 2

    # 1. Outer Keystone Box (Screen perimeter)
    draw.rectangle([0, 0, WIDTH - 1, HEIGHT - 1], outline=(0, 255, 255), width=3)
    draw.rectangle([20, 20, WIDTH - 21, HEIGHT - 21], outline=(0, 100, 100), width=1)

    # 2. Grid lines every 100 px
    for x in range(100, WIDTH, 100):
        draw.line([(x, 0), (x, HEIGHT)], fill=(30, 30, 30), width=1)
    for y in range(100, HEIGHT, 100):
        draw.line([(0, y), (WIDTH, y)], fill=(30, 30, 30), width=1)

    # 3. Vial Boundary Lines (Left & Right limits of the resin vial)
    draw.line([(cx - half_vial, 0), (cx - half_vial, HEIGHT)], fill=(0, 200, 0), width=2)
    draw.line([(cx + half_vial, 0), (cx + half_vial, HEIGHT)], fill=(0, 200, 0), width=2)

    # 4. Center Axis of Rotation (AoR) - Bright Red/White Centerline
    draw.line([(cx, 0), (cx, HEIGHT)], fill=(255, 0, 0), width=2)
    draw.line([(cx, cy - 200), (cx, cy + 200)], fill=(255, 255, 255), width=3)
    draw.line([(0, cy), (WIDTH, cy)], fill=(255, 0, 0), width=2)

    # 5. Concentric Circles at Center (Radius: 50, 100, 150, 200, 250, 300 px)
    for r in range(50, half_vial + 50, 50):
        color = (255, 255, 0) if r == half_vial else (100, 100, 100)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=1)

    # 6. Vernier Scale / Tick marks along the centerline
    for y in range(100, HEIGHT, 50):
        tick_len = 15 if y % 100 == 0 else 8
        draw.line([(cx - tick_len, y), (cx + tick_len, y)], fill=(255, 255, 255), width=1)

    # Labels
    draw.text((cx + 10, 40), "AXIS OF ROTATION (AoR)", fill=(255, 255, 255))
    draw.text((cx - half_vial + 10, 80), "VIAL LEFT BOUND", fill=(0, 255, 0))
    draw.text((cx + half_vial - 160, 80), "VIAL RIGHT BOUND", fill=(0, 255, 0))
    draw.text((40, HEIGHT - 60), f"Optoma ML1080 1080x1920 Portrait Grid (Vial Width: {vial_width_px}px)", fill=(0, 255, 255))

    path = os.path.join(OUT_DIR, "calibration_axis_and_grid.png")
    img.save(path)
    print(f"Generated: {path}")

# ── 2. Siemens Star Focus Target ─────────────────────────────────────────────
def generate_siemens_star_target(spokes=36, radius=350):
    img = Image.new("L", (WIDTH, HEIGHT), 0)
    cx, cy = WIDTH // 2, HEIGHT // 2

    Y, X = np.ogrid[:HEIGHT, :WIDTH]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    angles = np.arctan2(Y - cy, X - cx)

    # Siemens star equation: alternating black and white sectors
    star = ((np.sin(spokes * angles) > 0) & (dist_from_center <= radius)).astype(np.uint8) * 255

    # Center target dot
    star[dist_from_center <= 8] = 128

    star_img = Image.fromarray(star, mode="L").convert("RGB")
    draw = ImageDraw.Draw(star_img)

    # Alignment crosshair
    draw.line([(cx, cy - radius - 50), (cx, cy + radius + 50)], fill=(255, 0, 0), width=1)
    draw.line([(cx - radius - 50, cy), (cx + radius + 50, cy)], fill=(255, 0, 0), width=1)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(0, 255, 255), width=2)
    draw.text((cx - 120, cy + radius + 60), "FOCUS TO SHARPEST CENTER POINT", fill=(255, 255, 255))

    path = os.path.join(OUT_DIR, "calibration_siemens_star.png")
    star_img.save(path)
    print(f"Generated: {path}")

# ── 3. Checkerboard / Dot Grid Target (Computer Vision Auto-Keystone) ─────────
def generate_checkerboard_target(rows=11, cols=7, square_size=100):
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    start_x = (WIDTH - (cols * square_size)) // 2
    start_y = (HEIGHT - (rows * square_size)) // 2

    # White backdrop for checkerboard
    draw.rectangle([start_x - 20, start_y - 20, start_x + cols * square_size + 20, start_y + rows * square_size + 20], fill=(255, 255, 255))

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 1:
                x0 = start_x + c * square_size
                y0 = start_y + r * square_size
                draw.rectangle([x0, y0, x0 + square_size, y0 + square_size], fill=(0, 0, 0))

    # Center marker
    cx, cy = WIDTH // 2, HEIGHT // 2
    draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(255, 0, 0))

    draw.text((40, 40), "OpenCV Calibration Pattern (7x11 Corners @ 100px)", fill=(0, 255, 255))
    path = os.path.join(OUT_DIR, "calibration_checkerboard.png")
    img.save(path)
    print(f"Generated: {path}")

if __name__ == "__main__":
    generate_axis_grid_target()
    generate_siemens_star_target()
    generate_checkerboard_target()
