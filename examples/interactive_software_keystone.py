#!/usr/bin/env python3
"""
Interactive Software Keystone & Perspective Calibration Tool for Raspberry Pi / OpenCAL
Controls keystone directly from Python on the Pi without touching projector menus.

Usage:
  1. Clamps your 30x60mm acrylic blade into the lower vial chuck.
  2. Runs fullscreen on the projector display.
  3. Use keyboard to adjust the 4 corners (TL, TR, BR, BL) to match the physical blade.
  4. Press 'S' to save the calibrated Homography Matrix to JSON.
"""

import sys
import os
import json
import cv2
import numpy as np

WIDTH = 1080
HEIGHT = 1920
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "calibration_targets", "keystone_matrix.json")

# Initial 4-corner coordinates (centered 30x60mm aspect ratio window)
# [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
DEFAULT_MARGIN_X = 250
DEFAULT_MARGIN_Y = 400

corners = np.array([
    [DEFAULT_MARGIN_X, DEFAULT_MARGIN_Y],                 # Top-Left (0)
    [WIDTH - DEFAULT_MARGIN_X, DEFAULT_MARGIN_Y],         # Top-Right (1)
    [WIDTH - DEFAULT_MARGIN_X, HEIGHT - DEFAULT_MARGIN_Y], # Bottom-Right (2)
    [DEFAULT_MARGIN_X, HEIGHT - DEFAULT_MARGIN_Y]          # Bottom-Left (3)
], dtype=np.float32)

src_corners = np.array([
    [0, 0],
    [WIDTH - 1, 0],
    [WIDTH - 1, HEIGHT - 1],
    [0, HEIGHT - 1]
], dtype=np.float32)

selected_corner = 0
step = 5

def draw_calibration_pattern(dst_corners, active_idx):
    # 1. Base unwarped image (Grid + Siemens Star + Crosshair)
    base = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Draw internal grid
    for x in range(0, WIDTH, 80):
        cv2.line(base, (x, 0), (x, HEIGHT), (40, 40, 40), 1)
    for y in range(0, HEIGHT, 80):
        cv2.line(base, (0, y), (WIDTH, y), (40, 40, 40), 1)
        
    # Centerline (Axis of Rotation)
    cx, cy = WIDTH // 2, HEIGHT // 2
    cv2.line(base, (cx, 0), (cx, HEIGHT), (0, 0, 255), 2)
    cv2.line(base, (0, cy), (WIDTH, cy), (0, 0, 255), 2)
    
    # High-contrast center target for focus checking
    cv2.circle(base, (cx, cy), 120, (0, 255, 255), 2)
    cv2.circle(base, (cx, cy), 60, (0, 255, 255), 2)
    cv2.circle(base, (cx, cy), 6, (255, 255, 255), -1)
    
    # Outer border
    cv2.rectangle(base, (0, 0), (WIDTH - 1, HEIGHT - 1), (0, 255, 0), 4)

    # 2. Compute Homography matrix from standard canvas -> adjusted 4-corners
    H = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    # 3. Warp the pattern into the physical target area
    warped = cv2.warpPerspective(base, H, (WIDTH, HEIGHT))
    
    # Draw corner control handles on top
    for i, pt in enumerate(dst_corners):
        px, py = int(round(pt[0])), int(round(pt[1]))
        color = (0, 0, 255) if i == active_idx else (255, 255, 0)
        radius = 12 if i == active_idx else 8
        cv2.circle(warped, (px, py), radius, color, -1)
        cv2.putText(warped, f"C{i+1}", (px + 15, py + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # On-screen instructions
    info = [
        f"Active Corner: C{active_idx+1} (Pos: {int(dst_corners[active_idx][0])}, {int(dst_corners[active_idx][1])})",
        "Keys: [TAB] Next Corner | [Arrows / WASD] Move Corner",
        "[+] / [-] Step Size | [S] Save & Exit | [Q] Quit"
    ]
    for idx, text in enumerate(info):
        cv2.putText(warped, text, (30, 50 + idx * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return warped, H

def main():
    global selected_corner, step, corners
    win_name = "OpenCAL Software Keystone Calibrator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIDTH, HEIGHT)

    print("=" * 60)
    print("  OPENCAL SOFTWARE KEYSTONE CALIBRATOR")
    print("  Controls:")
    print("    TAB       : Switch active corner (TL -> TR -> BR -> BL)")
    print("    Arrows/WASD: Move selected corner")
    print("    +/-       : Increase / decrease step size (fine/coarse)")
    print("    S         : Save calibration to JSON and exit")
    print("    Q / ESC   : Quit without saving")
    print("=" * 60)

    while True:
        display_img, H = draw_calibration_pattern(corners, selected_corner)
        cv2.imshow(win_name, display_img)
        
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            print("Exited without saving.")
            break
        elif key == 9: # TAB
            selected_corner = (selected_corner + 1) % 4
        elif key == ord('1'):
            selected_corner = 0
        elif key == ord('2'):
            selected_corner = 1
        elif key == ord('3'):
            selected_corner = 2
        elif key == ord('4'):
            selected_corner = 3
        elif key in [ord('w'), 82]: # Up arrow or W
            corners[selected_corner][1] = max(0, corners[selected_corner][1] - step)
        elif key in [ord('s'), 84]: # Down arrow or S (note: hold shift or press Enter to save)
            corners[selected_corner][1] = min(HEIGHT - 1, corners[selected_corner][1] + step)
        elif key in [ord('a'), 81]: # Left arrow or A
            corners[selected_corner][0] = max(0, corners[selected_corner][0] - step)
        elif key in [ord('d'), 83]: # Right arrow or D
            corners[selected_corner][0] = min(WIDTH - 1, corners[selected_corner][0] + step)
        elif key == ord('+') or key == ord('='):
            step = min(50, step + 1)
            print(f"Step size: {step}px")
        elif key == ord('-') or key == ord('_'):
            step = max(1, step - 1)
            print(f"Step size: {step}px")
        elif key == 13 or key == ord('k'): # Enter or K to save
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            data = {
                "screen_width": WIDTH,
                "screen_height": HEIGHT,
                "dst_corners": corners.tolist(),
                "homography_matrix": H.tolist()
            }
            with open(CONFIG_PATH, "w") as f:
                json.dump(data, f, indent=4)
            print(f"\n[SUCCESS] Saved Keystone Matrix to: {CONFIG_PATH}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
