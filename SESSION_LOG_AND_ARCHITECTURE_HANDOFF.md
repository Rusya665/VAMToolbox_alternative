# OpenCAL & VAMToolbox: Complete Engineering Session Log & AI Handoff
**Date of Session:** August 19–20, 2026  
**Project Workspace:** `C:\Users\runiza\GoogleProjects\VAMToolbox_alternative`  
**Related Controller Workspace:** `C:\Users\runiza\GoogleProjects\OpenCAL-alternative`  
**Hardware Configuration:** OpenCAL 3D Printer (Raspberry Pi 5) + Optoma ML1080 (RGB Triple Laser, 1080p Portrait, ~1200 Lumens) + Collimator Optic + 30mm x 60mm Glass Resin Vial.

---

## 1. Executive Context for Future AI Agents

This document is a complete, chronological record of the engineering investigation, mathematical findings, hardware calibrations, software solutions, and tools developed during this session. Any future AI agent resuming this task should read this document first to understand the exact state of the project, why specific decisions were made, and where all artifacts are located.

---

## 2. Chronological Timeline & Technical Milestones

### [2026-08-19 08:40:04] Phase 1: Problem Definition & Discord Analysis
* **User Problem**:
  * Attempted to use the **TOMO GUI** (`Tomo Setup 1.0.0.exe`) on a Cube and Benchy model.
  * Observed massive optimization errors even after 150 cycles of OSMO:
    * Volumetric Voxel Error: **$18.11\%$**
    * Process Window ($PW$): **$-0.82$** (negative means background void overlaps in-part dose)
    * In-part Min Dose: **$0.82$** vs Out-part Max Dose: **$1.64$** (void received $2\times$ the target gel dose).
* **Discord Community Insights (OpenCAL Core Developers)**:
  * **Taylor W** (Lead author of OpenCAL / Tomo / VAMToolbox) confirmed that **Tomo is an early experimental GUI wrapper**, and the raw Python package [`VAMToolbox`](https://github.com/computed-axial-lithography/VAMToolbox) is the true, recommended production pipeline.
  * Raspberry Pi 5 motor timing under CPU load required switching from STEP/DIR to driver UART / Pololu Tic controllers with strict **$9.000 \pm 0.001\text{ RPM}$** calibration.
  * Projector orientation is **$90^\circ$ counter-clockwise** ($1080 \times 1920$ portrait mode).
  * System credentials: user `opencal`, password `OpenCAL1!`, fix password via `sudo passwd opencal`, ensure `I2C` and `SPI` are enabled.

---

### [2026-08-19 08:44:34] Phase 2: User's Physical Hardware Setup Identified
* **Controller**: Custom controller repository located at `C:\Users\runiza\GoogleProjects\OpenCAL-alternative`.
* **Projector**: **Optoma ML1080** (Triple RGB Laser, $1080 \times 1920$ native portrait, ~1200 ANSI/Laser lumens, high contrast, manual focus and keystone).
* **Optics & Setup**: High-intensity laser projection; manual optical alignment performed.

---

### [2026-08-19 08:47:31] Phase 3: Physics Breakdown & Core Guide Creation
* **Theoretical Investigations**:
  1. **Why Solid Cubes Fail in VAM**:
     * Radon transform non-negativity constraint ($I \ge 0$).
     * The diagonal of a square is $\sqrt{2} \approx 1.414\times$ thicker than the flats.
     * High-intensity rays targeting sharp corners intersect through the center from 360°, creating an over-cured central fireball and spraying "wing/horn" streak artifacts into the liquid void.
     * **Remedy**: Solid objects must be **shelled/hollowed** ($1.5 - 2.5\text{ mm}$ walls) or given internal lattice cores.
  2. **Why Benchy Fails Standing Up & Why Tilting Fixes It**:
     * A vertical Benchy has a flat cabin roof/chimney directly over an open cabin void in the same horizontal $Z$-slices. Horizontal rays cure the open cabin.
     * **Remedy**: Tilting Benchy $30^\circ - 45^\circ$ breaks coplanar slice alignment, distributing optical density smoothly across the vertical axis.
  3. **OSMO vs. BCLP Optimizers**:
     * **OSMO** (Object-Space Model Optimization) is an unconstrained heuristic (clips void, boosts gel). On geometries with impossible non-negative Radon inversions, it oscillates and stalls after 15–20 iterations. 150 cycles is useless.
     * **BCLP** (Band-Constrained Linear Programming) is a formal $L_p$-norm optimizer with relaxation bands ($\varepsilon$), hard non-negativity, and material response models (oxygen inhibition threshold $D_0$).
  4. **Why TOMO Fails**:
     * Hardcoded unconstrained OSMO defaults, lack of pre-flight geometry checks for solid infill, linear error metric ignoring resin chemical thresholding, and fixed video export formats.
* **Artifact Created**:
  * [`VAM_EXPLAINED_AND_TROUBLESHOOTING.md`](file:///c:/Users/runiza/GoogleProjects/VAMToolbox_alternative/VAM_EXPLAINED_AND_TROUBLESHOOTING.md) (26+ KB deep-dive guide).

---

### [2026-08-19 08:55:49] Phase 4: Exposure Window & RPM Drift Analysis
* **User Print Result**:
  * In a previous test, the cube became clearly visible and solidified after **$1.0 - 1.5\text{ minutes}$**, but running for **$5.0\text{ minutes}$** over-cured the entire vial into a solid block of plastic.
* **Root Cause & Physics**:
  1. **The VAM Exposure Window**:
     * Gelation is a race: Part dose rate $R_{part} > R_{void}$.
     * At $1.0\text{ min}$, part dose reaches $D_{th} = 1.0$ (gels). Void dose is $0.6$ (remains liquid). **This is the print window.**
     * At $5.0\text{ min}$ (45 rotations), void dose reaches $3.0$ ($3\times$ gel threshold). The liquid surrounding the cube solidifies.
     * Exposure must **stop immediately** after the part gels.
  2. **Angular Drift**:
     * If motor RPM drifts by even $0.05\text{ RPM}$, by revolution 10 the angular error is $20^\circ$, smearing corners and curing the void.
* **Solutions Added to Guide (Section 9)**:
  * OpenCAL 20-rotation stopwatch calibration method: $\text{CORRECTION\_FACTOR} = \text{MEASURED\_RPM} / 9.000$.
  * 50-rotation verification test: target time $= 333.3 \pm 0.3\text{ seconds}$.
  * Generation of finite MP4 videos (e.g. 9 loops $= 60\text{ s}$ @ 9 RPM) with a 2-second pure black tail to halt curing automatically.

---

### [2026-08-19 09:22:58] Phase 5: Optical Calibration Targets Generated
* **Target Generator Tool Created**:
  * Script: [`examples/generate_calibration_patterns.py`](file:///c:/Users/runiza/GoogleProjects/VAMToolbox_alternative/examples/generate_calibration_patterns.py)
* **Generated $1080 \times 1920$ Portrait Targets**:
  1. [`calibration_targets/calibration_axis_and_grid.png`](file:///c:/Users/runiza/GoogleProjects/VAMToolbox_alternative/calibration_targets/calibration_axis_and_grid.png): Axis of Rotation (AoR, $X=540\text{ px}$), vial boundaries, 100px keystone grid.
  2. [`calibration_targets/calibration_siemens_star.png`](file:///c:/Users/runiza/GoogleProjects/VAMToolbox_alternative/calibration_targets/calibration_siemens_star.png): 36-spoke radial Siemens star for visual/camera autofocus.
  3. [`calibration_targets/calibration_checkerboard.png`](file:///c:/Users/runiza/GoogleProjects/VAMToolbox_alternative/calibration_targets/calibration_checkerboard.png): $7 \times 11$ OpenCV chessboard for vision-based homography.

---

### [2026-08-19 09:28:38] Phase 6: Projector Control & Laser-Cut Target Dummy
* **Optoma ML1080 Hardware Controls**:
  * Mini-USB RS232 port (ASCII commands, 9600 baud, 8N1).
  * HDMI-CEC from Raspberry Pi: Auto power-on (`echo 'on 0' | cec-client -s -d 1`) and standby.
  * **Critical Operating Rule**: Must **LOCK / DISABLE continuous auto-focus and auto-keystone during printing**. The rotating liquid resin confuses the ToF infrared sensor mid-print.
* **Laser-Cut Acrylic Blade Guidelines**:
  * 3mm Matte/Cast White Acrylic (prevents glare, maximum contrast).
  * Engraved centerline ($U=0$) and vertical millimeter ruler ticks ($Z$-scale).

---

### [2026-08-19 10:01:26] Phase 7: CAD Analysis & Tight Space Constraints
* **Physical CAD Rig Review**:
  * 3-Axis adjustable Optoma ML1080 stage + green collimating cone optic + rotary vial chuck ($30\text{ mm}$ diameter $\times 60\text{ mm}$ height).
  * The projected light cone naturally overshoots the physical vial.
  * **Software Resolution**: The active sinogram is rendered only inside the central $30\text{ mm} \times 60\text{ mm}$ pixel box; all outer overshoot pixels are **pure black ($0, 0, 0$)**, emitting zero photons from the RGB laser.
* **Physical Constraint Identified**:
  * A wide calibration plate cannot fit in the physical apparatus.
  * The calibration target must mount directly into the **lower vial holder chuck** ($30\text{ mm} \times 60\text{ mm}$ blade) to eliminate mechanical holder offset errors.

---

### [2026-08-19 10:06:38 – 10:09:02] Phase 8: Pi-Driven Software Keystoning
* **Software Solution Implemented**:
  * Developed an interactive Python tool running on the Raspberry Pi / PC:
    📁 [`examples/interactive_software_keystone.py`](file:///c:/Users/runiza/GoogleProjects/VAMToolbox_alternative/examples/interactive_software_keystone.py)
* **How It Works**:
  1. Clamps the $30\text{ mm} \times 60\text{ mm}$ acrylic blade into the lower vial chuck.
  2. Runs fullscreen on the Optoma ML1080 via HDMI.
  3. User moves 4 corner control pins ($C_1, C_2, C_3, C_4$) using keyboard (`TAB`, Arrow keys / `WASD`) until the projected grid matches the physical edges of the blade.
  4. Computes the $3 \times 3$ perspective homography matrix $H = \text{cv2.getPerspectiveTransform()}$ and saves it to [`calibration_targets/keystone_matrix.json`](file:///c:/Users/runiza/GoogleProjects/VAMToolbox_alternative/calibration_targets/keystone_matrix.json).
  5. Print video generation scripts apply `cv2.warpPerspective(frame, H, (1080, 1920))` to warp every frame with sub-pixel precision.
* **Clarification on Terminology**:
  * `cv2.moveWindow` is **HDMI display routing** (telling the OS which monitor to show the window on).
  * `cv2.warpPerspective` is **Digital Keystoning** (mathematically pre-distorting the image to cancel optical keystone distortion).

---

## 3. Directory Structure & Key Files Created

```
C:\Users\runiza\GoogleProjects\VAMToolbox_alternative\
├── VAM_EXPLAINED_AND_TROUBLESHOOTING.md    # [NEW] Master guide: physics, math, BCLP, RPM, and printing checklist.
├── SESSION_LOG_AND_ARCHITECTURE_HANDOFF.md # [NEW] This comprehensive session log.
├── calibration_targets/                   # [NEW] Generated 1080x1920 calibration patterns
│   ├── calibration_axis_and_grid.png      # AoR, vial boundaries, 100px keystone grid
│   ├── calibration_siemens_star.png       # 36-spoke radial star for precision focusing
│   ├── calibration_checkerboard.png       # 7x11 OpenCV chessboard
│   └── keystone_matrix.json               # Saved 3x3 homography matrix output
├── examples/
│   ├── generate_calibration_patterns.py   # [NEW] Script that created the calibration PNGs
│   ├── interactive_software_keystone.py   # [NEW] Real-time 4-corner software keystone tool
│   ├── voxelize_and_optimize.py           # Headless STL -> Voxel -> BCLP -> MP4 pipeline
│   └── ... (other standard VAMToolbox examples)
└── vamtoolbox/                            # Core VAM mathematical optimization library
    ├── optimizer/
    │   ├── BCLP.py                        # Band-Constrained Linear Programming optimizer
    │   ├── OSMO.py                        # Object-Space Model Optimization
    │   ├── CAL.py                         # Gradient descent with sigmoid response
    │   └── ...
    └── ...
```

---

## 4. Key Mathematical Formulas & Parameters for Reference

| Parameter / Formula | Value / Definition | Purpose |
| :--- | :--- | :--- |
| **Projector Resolution** | $1080 \times 1920$ (Portrait) | Native Optoma ML1080 aspect ratio |
| **OpenCAL Target Speed** | $9.000\text{ RPM}$ | Calibrated motor rotational velocity |
| **Rotation Duration ($T_{rev}$)** | $60 / 9.0 = 6.6667\text{ seconds}$ | Time per full $360^\circ$ revolution |
| **Frames per Revolution** | $200\text{ frames}$ @ $30\text{ FPS}$ | Angular projection synchronization |
| **RPM Correction Formula** | $\text{CORRECTION\_FACTOR} = \frac{\text{MEASURED\_RPM}}{\text{TARGET\_RPM}}$ | Applied in `_rpm_to_vactual` |
| **Process Window ($PW$)** | $PW = D_{gel}^{min} - D_{void}^{max}$ | Must be $> 0$ for clean thresholding |
| **Exposure Timing Formula** | $T_{exposure} = N_{loops} \times \frac{60}{\text{RPM}}$ | Sets total curing exposure before black cutoff |
| **Cube Exposure Duration** | $8 - 10\text{ loops}$ ($53 - 66\text{ seconds}$) | Avoids 5-minute over-curing block |

---

## 5. Next Steps for Next Session / AI Agent

1. **Laser-cut the $30\text{ mm} \times 60\text{ mm}$ acrylic blade** (3mm matte white cast acrylic with engraved center line).
2. **Mount the blade in the lower holder** and run `python examples/interactive_software_keystone.py` on the Pi to align the 4 corners and save `keystone_matrix.json`.
3. **Lock Optoma ML1080 focus** onto the center target and ensure continuous auto-focus/keystone is switched to **Manual**.
4. **Run BCLP optimization** on hollowed/tilted 3D models (using the template in [Section 7 of `VAM_EXPLAINED_AND_TROUBLESHOOTING.md`](file:///c:/Users/runiza/GoogleProjects/VAMToolbox_alternative/VAM_EXPLAINED_AND_TROUBLESHOOTING.md#L286-L377)).
5. **Print with finite loops (8–10 loops / ~1 min)** to achieve sharp, un-melted parts.
