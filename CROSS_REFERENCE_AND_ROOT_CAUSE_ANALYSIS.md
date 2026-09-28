# Comprehensive Cross-Reference, Video Pipeline & Root Cause Investigation Report

**Date:** August 21, 2026  
**Repositories Analyzed:**
1. **Tomo GUI / Backend:** [`c:/Users/runiza/GoogleProjects/tomo-alternative`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative)
2. **OpenCAL Controller (RPi 5):** [`C:/Users/runiza/GoogleProjects/OpenCAL-alternative`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative) (Custom Newhaven LCD branch)
3. **VAMToolbox Core & Tuning Studio:** [`C:/Users/runiza/GoogleProjects/VAMToolbox_alternative`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative)
4. **Hardware Rig:** Optoma ML1080 (Triple RGB Laser Projector, mounted on its left side) + Raspberry Pi 5 + Glass Resin Vial.

---

## Table of Contents
1. [Executive Summary & Quick Reference Table](#1-executive-summary--quick-reference-table)
2. [Physical Setup & Coordinate Transformations](#2-physical-setup--coordinate-transformations)
3. [How OpenCAL on Raspberry Pi Operates](#3-how-opencal-on-raspberry-pi-operates)
   - [Display Server & Orientation Management](#display-server--orientation-management)
   - [How OpenCAL "Knows" Where the Projector Is (Calibration Architecture)](#how-opencal-knows-where-the-projector-is-calibration-architecture)
   - [Video Playback Pipeline During Printing](#video-playback-pipeline-during-printing)
4. [How Tomo Generates Videos](#4-how-tomo-generates-videos)
   - [Mesh Slicing & Voxel Coordinates](#mesh-slicing--voxel-coordinates)
   - [Why Benchy Appeared Normal in VLC but Upside Down on the Pi](#why-benchy-appeared-normal-in-vlc-but-upside-down-on-the-pi)
   - [The `np.flipud()` Optical Correction](#the-npflipud-optical-correction)
5. [Root Cause Analysis: Why VAMToolbox Videos Never Look the Same as Tomo](#5-root-cause-analysis-why-vamtoolbox-videos-never-look-the-same-as-tomo)
   - [Root Cause 1: Missing Vertical Flip (`np.flipud`) in VAMToolbox `saveAsVideo`](#root-cause-1-missing-vertical-flip-npflipud-in-vamtoolbox-saveasvideo)
   - [Root Cause 2: Z-Inversion & Axis Transposition Mismatch During Voxelization](#root-cause-2-z-inversion--axis-transposition-mismatch-during-voxelization)
   - [Root Cause 3: The Double-Scaling Bug in VAMToolbox `pipeline.py`](#root-cause-3-the-double-scaling-bug-in-vamtoolbox-pipelinepy)
   - [Root Cause 4: Baseline Subtraction Distortion in Tuning Studio Engine](#root-cause-4-baseline-subtraction-distortion-in-tuning-studio-engine)
   - [Root Cause 5: Field of View (FOV) & Pixel Pitch Scale Mismatch](#root-cause-5-field-of-view-fov--pixel-pitch-scale-mismatch)
   - [Root Cause 6: Video Encoding, Color Planes & Angular Sub-Frame Sampling](#root-cause-6-video-encoding-color-planes--angular-sub-frame-sampling)
   - [Root Cause 7: Optimization Math & Threshold Differences](#root-cause-7-optimization-math--threshold-differences)
6. [Cross-Repository Code Location Matrix](#6-cross-repository-code-location-matrix)
7. [Actionable Recommendations & Unified Solution](#7-actionable-recommendations--unified-solution)

---

## 1. Executive Summary & Quick Reference Table

| Aspect | Tomo Backend ([`VAM_Ob.py`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py)) | OpenCAL RPi ([`projector_controller.py`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/hardware/projector_controller.py)) | VAMToolbox Raw ([`pipeline.py`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/pipeline.py) / [`imagesequence.py`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/imagesequence.py)) |
| :--- | :--- | :--- | :--- |
| **Canvas Geometry** | $1080 \times 1920$ (Portrait) | $1080 \times 1920$ (via Wayland `wlr-randr`) | $1080 \times 1920$ or $1920 \times 1080$ |
| **Voxel $Z$-Axis** | Reverses $Z$ (`arr[:, :, ::-1]`) & transposes `(1,0,2)` | N/A (Receives pre-rendered MP4) | Keeps raw OpenGL buffer (Z top-at-0) |
| **Frame Vertical Flip** | **YES**: `np.flipud(image_seq.images[idx])` | N/A (Plays video fullscreen via `cvlc`) | **NO**: Directly writes unflipped array |
| **FOV / mm per px** | $108.0\text{ mm} \rightarrow 0.100\text{ mm/px}$ | $200\text{ px}$ vial width setting | Default $76.0\text{ mm} \rightarrow 0.0704\text{ mm/px}$ ($42\%$ smaller) |
| **Rebin Size Scaling** | Clamps `size_scale = 1.0` if rebinned | Centers with `croppadd` filter | Passes `size_scale=rp["size_scale"]` $\rightarrow$ **Double Scales!** |
| **Background Clipping** | Preserves raw sinogram intensity | Projects raw frames / color patches | Subtracted `np.min(arr)` in `tomo_video_engine.py` |
| **Video Encoder** | Bundled `imageio-ffmpeg` (libx264/libx265, 3-ch RGB `yuv420p`) | Plays using `/usr/bin/cvlc` | OpenCV `avc1` (`isColor=False`, 1-ch grayscale) |
| **Angular Interpolation** | Computes `deg_per_frame = rpm * 6.0 / fps` | Stepper turns CCW at calibrated RPM | Discrete angle frame step |

---

## 2. Physical Setup & Coordinate Transformations

### The Physical Setup
1. **Optoma ML1080 Projector**: Mounted on its **left side** (rotated $90^\circ$ counter-clockwise relative to landscape).
2. **Resulting Coordinate System**:
   - In standard landscape, projector resolution is $1920\text{ (width)} \times 1080\text{ (height)}$.
   - On its left side, the physical vertical axis (along the tall resin vial) is the **1920-pixel axis**, and the horizontal axis (vial diameter) is the **1080-pixel axis**.
   - If an unrotated landscape video is played, human heads and model tops point to the **left**.
3. **Software Compensation**:
   - On the Raspberry Pi, Wayland / Wayfire rotates the display output (`wlr-randr --transform 90` or `270`).
   - Slicers generate $1080 \times 1920$ portrait frames.

```
       [ Landscape Projector Normal ]                 [ Optoma ML1080 on Left Side ]
       +----------------------------+                  +------------------+
       |                            |                  |        TOP       | (Vial Top / Chimney)
       |   (1920 px Width)          |                  |  (1080 px Wide)  |
       |  x 1080 px Height          |                  |                  |
       |                            |                  | (1920 px Height) |
       +----------------------------+                  |                  |
                                                       |      BOTTOM      | (Vial Bottom / Base)
                                                       +------------------+
```

---

## 3. How OpenCAL on Raspberry Pi Operates

### Display Server & Orientation Management
OpenCAL runs headless on a Raspberry Pi 5 under Debian Bookworm with native Wayland via Wayfire (no X11 / Xwayland).
- Display control is managed in [`opencal/hardware/projector_controller.py`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/hardware/projector_controller.py#L15-L46):
```python
# Lines 15-46 in projector_controller.py
class ProjectorOrientation(Enum):
    NORMAL = "normal"
    LEFT = "left"      # wlr-randr "90"
    RIGHT = "right"    # wlr-randr "270"
    FLIPPED = "flipped"# wlr-randr "180"
```
- Querying and setting display orientation uses `wlr-randr`:
  ```bash
  wlr-randr --output HDMI-A-1 --transform 90
  ```
  See [`projector_controller.py:118-149`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/hardware/projector_controller.py#L118-L149).

### How OpenCAL "Knows" Where the Projector Is (Calibration Architecture)
OpenCAL does **not** perform automatic camera-based homography during normal print execution. Instead, it relies on three explicit calibration subsystems:

1. **Vial Width Calibration ([`vial_width.py`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/gui/modes/vial_width.py#L14-L74))**:
   - Launches an interactive Pygame mode projecting a solid white vertical bar.
   - User rotates the rotary encoder to expand/contract the bar until it matches the physical glass vial walls.
   - Persists `vial_width_px` (e.g. `200`) into [`opencal/utils/config.json`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/utils/config.json#L63).
2. **Optical Center & Strut Alignment ([`alignment.py`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/gui/modes/alignment.py#L17-L73) & [`generate_alignment_image.py`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/utils/calibration/generate_alignment_image.py#L34-L205))**:
   - `generate_alignment_image.py` converts the CAD DXF drawing of the physical cross-strut alignment tool into a $1080 \times 1920$ image at $80.1\text{ }\mu\text{m/pixel}$.
   - OpenCAL projects this pattern with 4 alignment slots, center fiducials, and numbered corners `1, 2, 3, 4`.
   - The user adjusts the vertical offset using the encoder knob (`alignment_y_offset_px`), saving it to `config.json`.
3. **Motor Angular Velocity & Jitter Auto-Calibration ([`motor_calibrator.py`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/utils/calibration/motor_calibrator.py#L93-L150))**:
   - Uses the RPi camera (`picamera2`) to track a high-contrast reflective marker on the rotating vial chuck.
   - Computes microsecond-accurate zero-crossings to measure exact RPM and jitter, calculating `correction_factor = target_rpm / measured_rpm`.
4. **Digital Software Keystoning ([`interactive_software_keystone.py`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/examples/interactive_software_keystone.py))**:
   - Clamps a $30\text{ mm} \times 60\text{ mm}$ acrylic calibration blade into the lower vial chuck.
   - User pins 4 corners to calculate a $3 \times 3$ perspective homography matrix ($H = \text{cv2.getPerspectiveTransform}$), saved in `keystone_matrix.json` for `cv2.warpPerspective()`.

### Video Playback Pipeline During Printing
When a user launches a print job from the LCD menu:
1. [`opencal/hardware/print_controller.py:32-47`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/hardware/print_controller.py#L32-L47):
   - Motor starts rotating **CCW** (`self.hardware.stepper.start_rotation("CCW")`).
   - LED ring turns on (`(0, 240, 0, 0)`).
   - Projector triggers video playback via `self.hardware.projector.play_video_with_vlc(video_file)`.
2. [`opencal/hardware/projector_controller.py:183-226`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/hardware/projector_controller.py#L183-L226):
   - Measures video dimensions using `ffprobe`.
   - Calculates centering crops based on configured print scale:
     ```python
     crop_x = int(orig_width / 2) - new_width / 2
     crop_y = int(orig_height / 2) - new_height / 2
     ```
   - Spawns `cvlc` in fullscreen with Wayland display environment:
     ```bash
     /usr/bin/cvlc --fullscreen --loop --no-video-title-show \
       --video-filter=croppadd \
       --croppadd-cropleft={crop_x} --croppadd-cropright={crop_x} \
       --croppadd-croptop={crop_y} --croppadd-cropbottom={crop_y} \
       <actual_video_path>
     ```

---

## 4. How Tomo Generates Videos

### Mesh Slicing & Voxel Coordinates
In [`tomo-alternative/UIMain/Python_Backend/VAM_Ob.py`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py):
1. **Loading & Voxelization ([`VAM_Ob.py:146-157`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L146-L157))**:
   ```python
   arr, _, _ = vamtb.voxelize.voxelizeTargetOpenGL(stl_path, n_layers)
   arr = (_np.asarray(arr) > 0).astype('uint8')         # (nY, nX, nZ)
   arr = arr[:, :, ::-1]                                # undo OpenGL Z-inversion (top was at z=0)
   arr = _np.ascontiguousarray(arr.transpose(1, 0, 2))   # -> (nX, nY, nZ)
   print_body = self._embed_offset(arr, off_mm, pitch)
   ```
2. **Projector Scale Computation ([`VAM_Ob.py:480-507`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L480-L507))**:
   - `self.proj_width = 108.0` mm, `self.proj_px_w = 1080`, `self.proj_px_h = 1920`.
   - Scale per voxel: `true_scale = (self.res * self.proj_px_w) / self.proj_width`.
   - If rebinned, `true_scale = 1.0` (sinogram is already scaled to projector pixels).

### Why Benchy Appeared Normal in VLC but Upside Down on the Pi
This discrepancy arises from three compounding optical and graphics conventions:

1. **OpenGL / Pyglet vs. Video Codec Origin**:
   - **OpenGL/Pyglet (DLP Projector Engine)**: Texture origin $(0,0)$ is at the **bottom-left** ($Y$ increases upwards).
   - **Video Codecs (MP4, VLC, OpenCV, FFmpeg)**: Pixel origin $(0,0)$ is at the **top-left** ($Y$ increases downwards).
2. **Projector Physical Mounting ($90^\circ$ on its Left Side)**:
   - When the projector is turned on its left side, the projector's internal optical scanline origin $(0,0)$ rotates $90^\circ$.
   - When Tomo originally rendered frames directly through `ImageSeq` without vertical inversion, row 0 was encoded at the top of the MP4 file.
   - When VLC played this on a standard monitor, Benchy's chimney pointed to the top of the monitor.
   - However, when projected through the Optoma ML1080 on its side into the physical resin vial, the optical projection was upside down relative to gravity—meaning the chimney pointed to the bottom of the vial!

### The `np.flipud()` Optical Correction
To fix this, Tomo added an explicit vertical flip in [`VAM_Ob.py:610-614`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L610-L614):
```python
def _frame(k):
    angle = (k * deg_per_frame) % 360.0
    idx = int(angle / 360.0 * n_images) % n_images
    # np.flipud to match vamtoolbox's DLP projector (dlp/players.py) — the
    # bottom-origin pyglet path flips each frame; keep the saved video in the
    # same orientation as the real print (and as the live preview above).
    g = np.flipud(image_seq.images[idx])
    return np.repeat(g[:, :, None], 3, axis=2)   # grey -> (H, W, 3) RGB
```

---

## 5. Root Cause Analysis: Why VAMToolbox Videos Never Look the Same as Tomo

When users run VAMToolbox (or scripts in `VAMToolbox_alternative`) to find optimal slicing parameters, the resulting videos look radically different from Tomo's output. The investigation identified **7 distinct root causes**:

### Root Cause 1: Missing Vertical Flip (`np.flipud`) in VAMToolbox `saveAsVideo`
- **In Tomo ([`VAM_Ob.py:613`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L613))**:
  Every frame undergoes `np.flipud(image_seq.images[idx])`.
- **In VAMToolbox ([`vamtoolbox/imagesequence.py:267-268`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/imagesequence.py#L267-L268))**:
  `saveAsVideo()` writes raw `self.images[k]` directly to `cv2.VideoWriter` without calling `np.flipud()`.
- **Impact**: VAMToolbox videos are **vertically inverted (upside-down)** relative to Tomo videos.

### Root Cause 2: Z-Inversion & Axis Transposition Mismatch During Voxelization
- **In Tomo ([`VAM_Ob.py:153-155`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L153-L155))**:
  ```python
  arr = arr[:, :, ::-1]                              # Inverts Z
  arr = _np.ascontiguousarray(arr.transpose(1, 0, 2)) # Transposes (nY, nX, nZ) -> (nX, nY, nZ)
  ```
- **In Raw VAMToolbox ([`geometry.py:576-581`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/geometry.py#L576-L581))**:
  `TargetGeometry(stlfilename=...)` passes the raw OpenGL voxel array through without reversing $Z$ and without transposing $(1,0,2)$.
- **Impact**: The 3D radon projection is computed on a coordinate frame rotated by $90^\circ$ in $XY$ and inverted in $Z$.

### Root Cause 3: The Double-Scaling Bug in VAMToolbox `pipeline.py`
- In [`vamtoolbox/pipeline.py:377-386`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/pipeline.py#L377-L386):
  `rebin()` calculates zoom factor $f = \text{vial\_width\_px} / \text{cur\_vial\_px}$ and resamples the sinogram:
  `up = zoom(sino.array, (f, 1.0, f))`.
- Then in [`pipeline.py:406-409`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/pipeline.py#L406-L409):
  `save_video()` passes `size_scale=rp["size_scale"]` to `ImageConfig`.
  Inside `ImageSeq`, `_scaleSize()` zooms the sinogram **a second time** by `size_scale`!
- **In Tomo ([`VAM_Ob.py:490-493`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L490-L493))**:
  Tomo prevents this by setting `true_scale = 1.0` whenever `_sino_is_rebinned` is True.
- **Impact**: In VAMToolbox, rebinned models are magnified twice, overflowing the projector canvas and getting cropped or blown out.

### Root Cause 4: Baseline Subtraction Distortion in Tuning Studio Engine
- In [`vam_tuning_studio/tomo_video_engine.py:82-84`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vam_tuning_studio/tomo_video_engine.py#L82-L84):
  ```python
  bg = float(np.min(arr))
  if bg > 0.0:
      arr = np.maximum(0.0, arr - bg)
  ```
- **In Tomo ([`VAM_Ob.py:583-588`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L583-L588))**:
  Tomo never subtracts `np.min(arr)`. It normalizes to the 99.9th percentile directly.
- **Impact**: Subtracting the minimum shifts the entire non-zero dose baseline, eliminating low-dose peripheral rays, over-contrasting the sinogram, and causing missing features on thin walls.

### Root Cause 5: Field of View (FOV) & Pixel Pitch Scale Mismatch
- **Tomo Defaults ([`VAM_Ob.py:51-53`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L51-L53))**:
  - `proj_width = 108.0 mm`
  - `proj_px_w = 1080`
  - Resulting scale: $108.0 / 1080 = \mathbf{0.100\text{ mm/px}}$.
- **VAMToolbox Pipeline Defaults ([`pipeline.py:91`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/pipeline.py#L91))**:
  - `mm_per_pix = 76.0 / 1080 = \mathbf{0.07037\text{ mm/px}}$.
- **Impact**: VAMToolbox renders models at $1.421\times$ ($42\%$) different physical zoom compared to Tomo.

### Root Cause 6: Video Encoding, Color Planes & Angular Sub-Frame Sampling
- **Tomo ([`VAM_Ob.py:597-658`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L597-L658))**:
  - Uses bundled `imageio-ffmpeg` writing 3-channel RGB `yuv420p` (`libx264`/`libx265`).
  - Computes `deg_per_frame = (rpm * 6.0) / fps` (continuous sub-frame angular sampling across 360°).
  - Uses fast stream-looping (`-stream_loop`) for exact rotation loops.
- **Raw VAMToolbox ([`imagesequence.py:256-276`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/imagesequence.py#L256-L276))**:
  - Uses OpenCV `cv2.VideoWriter(..., isColor=False)` writing single-channel 8-bit `avc1`.
  - Many hardware players / VLC decoders fail or display corrupted luminance on single-channel MP4s.

### Root Cause 7: Optimization Math & Threshold Differences
- **Tomo ([`VAM_Ob.py:20-25`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L20-L25))**:
  - Defaults to **OSMO** with $d_h = 0.60, d_l = 0.50, n_{iter} = 5$, `filter="hamming"`.
- **VAMToolbox Pipeline ([`pipeline.py:61-68`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/pipeline.py#L61-L68))**:
  - Defaults to $d_h = 0.85, d_l = 0.65, n_{iter} = 10$, `filter="hanning"`.
- **Impact**: The higher dose threshold bounds ($0.85 / 0.65$) in VAMToolbox produce dramatically steeper sinogram gradients compared to Tomo's smoother $0.60 / 0.50$ distribution.

---

## 6. Cross-Repository Code Location Matrix

| Functionality | Repository | File Path & Exact Line References |
| :--- | :--- | :--- |
| **Wayland Projector Display Rotation** | `OpenCAL-alternative` | [`opencal/hardware/projector_controller.py:15-46`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/hardware/projector_controller.py#L15-L46) & [`:118-149`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/hardware/projector_controller.py#L118-L149) |
| **VLC Video Playback & Crop Scaling** | `OpenCAL-alternative` | [`opencal/hardware/projector_controller.py:183-226`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/hardware/projector_controller.py#L183-L226) |
| **Vial Width Interactive Calibration** | `OpenCAL-alternative` | [`opencal/gui/modes/vial_width.py:14-74`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/gui/modes/vial_width.py#L14-L74) |
| **Cross-Strut Optical Alignment Tool** | `OpenCAL-alternative` | [`opencal/gui/modes/alignment.py:17-73`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/gui/modes/alignment.py#L17-L73) & [`opencal/utils/calibration/generate_alignment_image.py:34-205`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/utils/calibration/generate_alignment_image.py#L34-L205) |
| **Motor Speed & Jitter Calibration** | `OpenCAL-alternative` | [`opencal/utils/calibration/motor_calibrator.py:93-150`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/utils/calibration/motor_calibrator.py#L93-L150) |
| **Tomo Voxelization & $Z$-Fix** | `tomo-alternative` | [`UIMain/Python_Backend/VAM_Ob.py:126-189`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L126-L189) |
| **Tomo Video Scale & Offset** | `tomo-alternative` | [`UIMain/Python_Backend/VAM_Ob.py:480-520`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L480-L520) |
| **Tomo Vertical Flip & MP4 Export** | `tomo-alternative` | [`UIMain/Python_Backend/VAM_Ob.py:558-685`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L558-L685) |
| **VAMToolbox Fan-Beam Rebinning** | `VAMToolbox_alternative` | [`vamtoolbox/geometry.py:1031-1265`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/geometry.py#L1031-L1265) |
| **VAMToolbox Image Sequence Builder** | `VAMToolbox_alternative` | [`vamtoolbox/imagesequence.py:85-175`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/imagesequence.py#L85-L175) |
| **VAMToolbox Unflipped Video Export** | `VAMToolbox_alternative` | [`vamtoolbox/imagesequence.py:194-277`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/imagesequence.py#L194-L277) |
| **Tuning Studio 1:1 Video Engine** | `VAMToolbox_alternative` | [`vam_tuning_studio/tomo_video_engine.py:27-250`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vam_tuning_studio/tomo_video_engine.py#L27-L250) |

---

## 7. Actionable Recommendations & Unified Solution

To ensure that any optimization or video generation in `VAMToolbox_alternative` matches Tomo bit-for-bit, apply the following 5 fixes:

1. **Adopt Tomo's Voxel Coordinate Mapping**:
   Invert the $Z$-axis (`arr[:, :, ::-1]`) and transpose $(1, 0, 2)$ immediately after calling `voxelizeTargetOpenGL()`.
2. **Standardize on `np.flipud()` for all Video Outputs**:
   Always apply `np.flipud()` to every image in `ImageSeq.images` before feeding frames to the video encoder.
3. **Fix the Double-Scaling Bug in `pipeline.py`**:
   If the sinogram has already been rebinned via `rebinFanBeam()`, set `size_scale = 1.0` in `ImageConfig`.
4. **Remove Baseline Subtraction (`bg = np.min(arr)`)**:
   Do not subtract `np.min(arr)` before normalization. Retain raw percentile normalization (`99.9%`).
5. **Synchronize Optical Constants**:
   Use `proj_width = 108.0 mm` ($0.100\text{ mm/px}$) and $1080 \times 1920$ portrait canvas across all scripts.

---

## 8. OpenCAL Architectural Boundary & Calibration Image Overhaul

### Strict Architectural Role: OpenCAL as Controller & Video Player
To eliminate all coordinate confusion, OpenCAL on the Raspberry Pi must maintain a clean, single-responsibility architecture:
- **OpenCAL is strictly a hardware controller and video player**.
- It controls the stepper motor (CW/CCW RPM), LED ring, camera recording, and plays pre-rendered MP4 video files fullscreen via `cvlc`.
- **OpenCAL must NEVER attempt real-time video transformations (flipping, rotating, or perspective warping) during print playback**.
- All video preparation (slicing, Z-inversion, fan-beam refraction correction, and optical `np.flipud` vertical inversion) is the responsibility of the slicer (Tomo / VAMToolbox) **before** transferring the MP4 file to the Pi.
- With the Optoma ML1080 mounted on its left side, Wayland's display transform (`wlr-randr --transform 90` or `270`) makes $(1080 \times 1920)$ the native physical canvas.

```
       +-------------------------------------------------------------+
       |             SLICER (Tomo / VAMToolbox on PC)                |
       |  - 3D Mesh Loading & Voxelization                           |
       |  - Radon Transform / OSMO / BCLP Optimization               |
       |  - Cylindrical Refraction Correction (rebinFanBeam)         |
       |  - Vertical Frame Inversion (np.flipud)                     |
       |  - Output: 1080 x 1920 Portrait MP4 (yuv420p)               |
       +-------------------------------------------------------------+
                                      |
                                      | Export via USB / Network
                                      v
       +-------------------------------------------------------------+
       |             OpenCAL CONTROLLER (Raspberry Pi 5)             |
       |  - Motor Rotation (e.g. 9.000 RPM CCW)                      |
       |  - LED Ring Illumination ((0, 240, 0, 0))                   |
       |  - Direct 1:1 Fullscreen Playback via cvlc                  |
       |  - Camera Timelapse / Process Video Logging                 |
       +-------------------------------------------------------------+
```

### Enhanced Alignment Calibration Image (With Directional Arrows)
To give the operator foolproof visual verification on the physical rig, [`opencal/utils/calibration/generate_alignment_image.py`](file:///C:/Users/runiza/GoogleProjects/OpenCAL-alternative/opencal/utils/calibration/generate_alignment_image.py) should be updated to render prominent orientation arrows:

```
+---------------------------------------------------------------+
|  [1]            ▲ TOP / CHUCK (Y = 0)                   [2]  |
|                                                               |
|  ◄ LEFT                                            RIGHT ►   |
|  (X = 0)                       |                  (X = 1080) |
|                                |                              |
|                       +--------+--------+                     |
|                       |  Cross Strut    |                     |
|                       |  Alignment Tool |                     |
|                       +--------+--------+                     |
|                                |                              |
|                                | Axis of Rotation (X = 540)   |
|                                                               |
|  [3]           ▼ BOTTOM / BASE (Y = 1920)               [4]  |
+---------------------------------------------------------------+
```

#### Python Implementation for `generate_alignment_image.py`:
```python
def draw_orientation_indicators(draw: ImageDraw.ImageDraw, W: int, H: int, font):
    """Draws prominent directional arrows and labels so orientation is immediately verified."""
    # 1. Axis of Rotation (AoR) vertical centerline (dashed/subtle)
    cx = W // 2
    draw.line([(cx, 100), (cx, H - 100)], fill=(80, 80, 80), width=1)

    # 2. TOP Arrow (pointing up towards the rotary chuck / vial top)
    draw.polygon([(cx, 30), (cx - 25, 75), (cx + 25, 75)], fill="white")
    draw.text((cx - 160, 85), "▲ TOP / CHUCK (Y=0)", fill="white", font=font)

    # 3. BOTTOM Arrow (pointing down towards vial base)
    draw.polygon([(cx, H - 30), (cx - 25, H - 75), (cx + 25, H - 75)], fill="white")
    draw.text((cx - 170, H - 125), "▼ BOTTOM / BASE", fill="white", font=font)

    # 4. LEFT / RIGHT lateral indicators
    draw.text((30, H // 2 - 20), "◄ LEFT", fill="white", font=font)
    draw.text((W - 140, H // 2 - 20), "RIGHT ►", fill="white", font=font)
```

---

## 9. The Physics & Mathematics of the "Green Usable Printable Cylinder" (Refraction Limit)

### Why Cylindrical Refraction Shrinks the Printable Volume
When light rays travel from air ($n_{\text{air}} = 1.0$) into a cylindrical glass/resin vial ($n_{\text{resin}} \approx 1.51$), Snell's law governs refraction at the curved boundary:
$$\sin(\theta_t) = \frac{n_{\text{air}}}{n_{\text{resin}}} \sin(\theta_i) = \frac{1}{n_{\text{resin}}} \sin(\theta_i)$$

At grazing incidence ($\theta_i \rightarrow 90^\circ, \sin(\theta_i) \rightarrow 1.0$), the maximum transmitted angle inside the resin is:
$$\sin(\theta_t)_{\text{max}} = \frac{1}{n_{\text{resin}}} \approx \frac{1}{1.51} \approx 0.6622$$

In Radon space, the physical ray coordinate $x_v$ inside the vial has a hard mathematical ceiling:
$$r_{\text{usable}} = R_v \cdot \frac{1}{n_{\text{resin}}}$$

Where $R_v$ is the inner radius of the resin container.

```
                   Physical Vial vs. Usable Printable Zone
       +-------------------------------------------------------+
       |                                                       |
       |             /~~~~~~~~~~~~~~~~~~~~~~~~~\               |
       |            /     /~~~~~~~~~~~~~~~\     \              |
       |           |     |                 |     |             |
       |  Air      |     |  GREEN CYLINDER |     |  Air        |
       | (n=1.0)   |     | (Usable Volume) |     | (n=1.0)     |
       |           |     |  r <= Rv / n    |     |             |
       |           |     |                 |     |             |
       |            \     \_______________/     /              |
       |             \_________________________/               |
       |               BLUE CYLINDER (Vial Wall, Rv)           |
       |                                                       |
       +-------------------------------------------------------+
```

### Usable Printable Diameter Table for Standard Vials ($n_{\text{resin}} = 1.51$)

| Vial Preset | Physical Inner Diameter ($\varnothing_{\text{vial}}$) | Physical Inner Radius ($R_v$) | Maximum Usable Radius ($r_{\text{usable}}$) | Usable Printable Diameter ($\varnothing_{\text{green}}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Small Vial** | $\mathbf{20.0\text{ mm}}$ | $10.0\text{ mm}$ | $\mathbf{6.62\text{ mm}}$ | $\mathbf{\approx 13.24\text{ mm}}$ |
| **Medium Vial** | $\mathbf{30.0\text{ mm}}$ | $15.0\text{ mm}$ | $\mathbf{9.93\text{ mm}}$ | $\mathbf{\approx 19.86\text{ mm}}$ |
| **Large Vial** | $\mathbf{48.8\text{ mm}}$ | $24.4\text{ mm}$ | $\mathbf{16.16\text{ mm}}$ | $\mathbf{\approx 32.32\text{ mm}}$ |

### Practical Implications for the Slicer
1. **The Blue Cylinder (3D View)**: Shows the physical glass inner wall.
2. **The Green Cylinder (3D View)**: Shows $r_{\text{usable}} = R_v / n_{\text{resin}}$.
3. **Pre-Flight Rule**: If any part of the 3D model extends outside the **green cylinder**, light rays cannot reach those outer features after refraction without total internal reflection / ray cut-off. In [`geometry.py:1175-1179`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/geometry.py#L1175-L1179), `rebinFanBeam` will flag:
   ```
   *** CUT-OFF: outer X px of sinogram are unreachable - edge features will be missing at widest projection angles ***
   ```
4. **VAMToolbox Implementation**: `VAMToolbox` must enforce $r_{\text{usable}}$ during pre-flight checks in [`pipeline.py`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/pipeline.py) and reject/warn if a model exceeds the green boundary.

---

## 10. The "Pixel Size" Equation Decoded — Where and Why It Is Used

In OpenCAL's documentation:
$$\text{Pixel Size (mm/pixel)} = \frac{\text{Vial Inner Diameter (mm)}}{\text{Measured Vial Width in Pixels (pixels)}}$$

### Exactly Where Pixel Size Is Used Across the Codebase

```
+---------------------------------------------------------------------------------------+
|                                    PIXEL SIZE                                         |
|                       (e.g., 20.0 mm / 200 px = 0.100 mm/px)                          |
+---------------------------------------------------------------------------------------+
        |                                   |                                   |
        v                                   v                                   v
[ 1. Slicer Model Scaling ]       [ 2. Refraction Rebinning ]         [ 3. Light Attenuation ]
proj_width = 1080 * pixel_size    vial_width_px = d_vial / pixel_size  depth_cm = r * pixel_size
true_scale = pitch / pixel_size   Rv_px = vial_width_px / 2           mask = exp(-mu * depth_cm)
```

1. **Physical Model Scaling in Tomo / Slicer ([`VAM_Ob.py:480-495`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L480-L495))**:
   - The projector has a fixed resolution of $1080 \text{ px}$ across the vial diameter axis.
   - The physical width that this $1080\text{ px}$ canvas covers at the vial focal plane is:
     $$\text{proj\_width (mm)} = 1080\text{ px} \times \text{Pixel Size (mm/px)}$$
   - When Tomo maps 3D voxels (e.g. $80\text{ }\mu\text{m} = 0.080\text{ mm}$) to projector pixels, it calculates:
     $$\text{true\_scale} = \frac{\text{voxel\_pitch (mm)} \times 1080\text{ px}}{\text{proj\_width (mm)}} = \frac{\text{voxel\_pitch (mm)}}{\text{Pixel Size (mm/px)}}$$
   - **If Pixel Size is incorrect**: A $10\text{ mm}$ printed cube will physically come out at $14.2\text{ mm}$ or $7.1\text{ mm}$ because the optical scaling factor is wrong.

2. **Cylindrical Refraction Ray Tracing in `rebinFanBeam` ([`geometry.py:1086-1100`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/geometry.py#L1086-L1100))**:
   - `rebinFanBeam` needs to know the exact curvature radius of the glass container in units of projector pixels:
     $$R_{v\text{ (pixels)}} = \frac{\text{Vial Inner Diameter (mm)}}{2 \times \text{Pixel Size (mm/px)}} = \frac{W_{\text{vial\_px}}}{2}$$
   - This pixel radius determines the angle of incidence $\theta_i = \arcsin(x_p / R_v)$ for every ray across the screen.

3. **Beer-Lambert Light Absorption Correction ([`geometry.py:201`](file:///C:/Users/runiza/GoogleProjects/VAMToolbox_alternative/vamtoolbox/geometry.py#L201))**:
   - The attenuation mask computes physical penetration depth:
     $$z_{\text{cm}} = (R_{\text{vial}} - r) \times \left(\frac{\text{Pixel Size (mm/px)}}{10.0}\right)$$
   - The exponential light decay profile $I(z) = I_0 e^{-\mu z}$ is applied in true physical centimeters based on this pixel size.

4. **Vertical Offset & Centering ([`VAM_Ob.py:510-520`](file:///c:/Users/runiza/GoogleProjects/tomo-alternative/UIMain/Python_Backend/VAM_Ob.py#L510-L520))**:
   - When the user shifts a model by $+5.0\text{ mm}$ in $Z$, the vertical translation in projector pixels is:
     $$\text{v\_offset\_px} = \frac{5.0\text{ mm}}{\text{Pixel Size (mm/px)}} = \frac{5.0}{0.100} = 50\text{ px}$$

---

## 11. Multi-Project Synchronization Checklist

### Project 1: `OpenCAL-alternative` (RPi Controller & Video Player)
- [x] **Keep Role Pure**: Operate purely as hardware controller and video player. No real-time video flipping or re-scaling.
- [ ] **Alignment Image Overlay**: Update `generate_alignment_image.py` to draw `▲ TOP / CHUCK (Y=0)`, `▼ BOTTOM / BASE (Y=1920)`, `◄ LEFT`, `RIGHT ►`, and the $AoR$ centerline ($X=540$).
- [ ] **Config Standardization**: Ensure `config.json` defaults `vial_width_px` to measured physical value (e.g. 200) and `alignment_y_offset_px` to 0.
- [ ] **Wayland Rotation**: Ensure `wlr-randr --output HDMI-A-1 --transform 90` (or `270`) is set on boot so Wayland natively provides a $1080 \times 1920$ canvas.

### Project 2: `tomo-alternative` (Desktop GUI & Slicer)
- [x] **Maintain Voxel Z-Fix**: Keep `arr[:, :, ::-1]` and transpose `(1, 0, 2)` in `VAM_Ob.py:152-156`.
- [x] **Maintain Video Flip**: Keep `np.flipud()` in `VAM_Ob.py:613`.
- [ ] **Direct Pixel Size Input**: Expose a single `Pixel Size (µm/px)` or `Measured Vial Width (px)` setting in the GUI so users don't have to manually calculate `proj_width`.
- [ ] **Green Cylinder Pre-Flight Warning**: Enforce $r \le R_v / n_{\text{resin}}$ in the GUI 3D viewport, warning the user if any part of the model extends into the unprintable refraction zone.

### Project 3: `VAMToolbox_alternative` (Core Math & Tuning Studio)
- [ ] **Adopt 1:1 Tomo Video Engine**: Replace raw `imagesequence.py:saveAsVideo` with `tomo_video_engine.py` (including `np.flipud`, 3-channel RGB `yuv420p`, sub-frame angular stepping, and stream loops).
- [ ] **Fix Pipeline Double-Scaling**: In `pipeline.py:save_video()`, set `size_scale = 1.0` if the sinogram has already been rebinned.
- [ ] **Remove Minimum Subtraction**: Remove `arr - np.min(arr)` baseline subtraction to prevent clipping low-intensity peripheral rays.
- [ ] **Sync Default FOV**: Update `PrintConfig.mm_per_pix` in `pipeline.py` to match Tomo's $108.0\text{ mm} / 1080\text{ px} = 0.100\text{ mm/px}$.

---

## 12. Repository Isolation & AI Agent Directives

> [!IMPORTANT]
> **To any AI Agent assigned to this ecosystem**:  
> You must operate **ONLY** within your designated target repository. Do **NOT** modify or attempt to refactor files belonging to the other repositories. Treat this document as an immutable interface contract.

### 🤖 Directive for OpenCAL Agent (`OpenCAL-alternative`)
* **Working Root**: `C:\Users\runiza\GoogleProjects\OpenCAL-alternative`
* **Your Strict Role**: Hardware Controller, Stepper Motor Driver, Optical Telemetry, and Video Player on Raspberry Pi 5.
* **Allowed Edits**:
  - `opencal/utils/calibration/generate_alignment_image.py`: Add the directional orientation arrows (`TOP`, `BOTTOM`, `LEFT`, `RIGHT`, `AoR`).
  - `opencal/utils/config.json`: Clean up default hardware settings and remove misleading comments.
  - `opencal/hardware/projector_controller.py`: Maintain `wlr-randr` 90/270 support; ensure fullscreen `cvlc` stays non-blocking.
* **STRICT FORBIDDEN ACTIONS**:
  - ❌ Do **NOT** add on-the-fly video flipping (`np.flipud`), rotation, or homography into `print_controller.py` or `projector_controller.py`. Incoming video files from Tomo/VAMToolbox are already pre-formatted.
  - ❌ Do **NOT** touch `tomo-alternative` or `VAMToolbox_alternative` code.

---

### 🤖 Directive for VAMToolbox Agent (`VAMToolbox_alternative`)
* **Working Root**: `C:\Users\runiza\GoogleProjects\VAMToolbox_alternative`
* **Your Strict Role**: Core Mathematical Optimizer (BCLP/OSMO/CAL), Ray-Tracing Slicer, Fan-Beam Rebinning, and Tuning Studio.
* **Allowed Edits**:
  - `vamtoolbox/imagesequence.py`: Update `saveAsVideo()` to adopt Tomo's 1:1 video generation (`np.flipud`, 3-channel RGB `yuv420p`, sub-frame angular stepping via `deg_per_frame`, and `-stream_loop`).
  - `vamtoolbox/pipeline.py`: Fix the double-scaling bug by clamping `size_scale = 1.0` if the sinogram has already been rebinned in `rebin()`.
  - `vam_tuning_studio/tomo_video_engine.py`: Remove `bg = np.min(arr)` baseline subtraction so low-intensity boundary rays are preserved.
  - `vamtoolbox/geometry.py`: Expose the green usable printable cylinder boundary ($r_{\text{usable}} = R_v / n_{\text{resin}}$) in pre-flight checks.
* **STRICT FORBIDDEN ACTIONS**:
  - ❌ Do **NOT** change OpenCAL hardware GPIO pinouts, motor drivers, or LCD menus.
  - ❌ Do **NOT** touch `tomo-alternative` GUI code.

---

### 🤖 Directive for Tomo Agent (`tomo-alternative`)
* **Working Root**: `c:\Users\runiza\GoogleProjects\tomo-alternative`
* **Your Strict Role**: Electron / Flask Desktop GUI Slicer and User Interface.
* **Allowed Edits**:
  - `UIMain/Front_End/`: Add UI controls for direct `Pixel Size (µm/px)` or `Measured Vial Width (px)` input so users don't have to manually calculate `proj_width`.
  - `UIMain/Python_Backend/VAM_Ob.py`: Ensure `arr[:, :, ::-1]` ($Z$-fix) and `np.flipud()` are preserved.
  - `UIMain/Front_End/`: Ensure the 3D viewport clearly highlights the green usable printable zone ($r \le R_v / n_{\text{resin}}$) and warns if the part clips outside it.
* **STRICT FORBIDDEN ACTIONS**:
  - ❌ Do **NOT** modify Raspberry Pi specific drivers in OpenCAL.
  - ❌ Do **NOT** touch core mathematical solver files in `VAMToolbox_alternative`.
