# Volumetric Additive Manufacturing (VAM / CAL): The Definitive Guide
## Physics, Geometry Limits, Optimizers (OSMO vs. BCLP), Tomo Breakdown, and OpenCAL + Optoma ML1080 Calibration

---

## 1. Executive Summary & The Core Mental Shift

In traditional 3D printing (FDM, SLA, DLP, SLS), objects are built **layer-by-layer**:
* The printer slices a 3D model into 2D sheets ($XY$).
* At layer $Z_k$, the printer projects or extrudes material **only at layer $Z_k$**.
* Layers $Z_0 \dots Z_{k-1}$ and $Z_{k+1} \dots Z_{max}$ are completely isolated and untouched.

In **Computed Axial Lithography (CAL) / Volumetric Additive Manufacturing (VAM)**:
* There are **NO layers**. The entire 3D part is formed simultaneously in a rotating vial of photopolymer resin in seconds.
* 2D light images are projected through the **entire volume** from 360 degrees.
* Every single ray of light enters one side of the vial, travels all the way through the resin, and exits the other side.
* The total accumulated energy (dose $D(x, y, z)$) at every point in space is the sum of all light rays that passed through that point across all rotational angles:
  $$D(x, y, z) = \int_0^{2\pi} I(\theta, u(x, y, \theta), z) \, d\theta$$

```
   TRADITIONAL SLA / DLP                   VOLUMETRIC (CAL / VAM)
  ┌───────────────────────┐              ┌───────────────────────┐
  │ Layer Z+1 (Untouched) │              │ Entire volume exposed │
  ├───────────────────────┤              │ simultaneously from   │
  │ Current Layer Z (Lit) │              │ 360 degrees of rays.  │
  ├───────────────────────┤              │                       │
  │ Layer Z-1 (Cured)     │              │ All rays pierce void  │
  └───────────────────────┘              │ and part alike!       │
                                         └───────────────────────┘
```

Because **light cannot be negative** ($I \ge 0$, you cannot shine "anti-photons" to erase dose), any light ray required to cure a feature inside the target **inevitably deposits background dose in the empty liquid resin (void)** before entering and after leaving the part.

---

## 2. Why a Solid Cube is the "Kryptonite" of VAM

### 2.1 The Geometry Problem: Square vs. Circle
When you rotate a square (the cross-section of a cube) inside a circular projection domain:
* From flat face to flat face, the optical thickness is $L$.
* From corner to diagonal corner, the optical thickness is $\sqrt{2} \cdot L \approx 1.414 \cdot L$ (41% thicker!).

```
       0° Flat View                        45° Diagonal View
      ┌─────────────┐                            ▲
      │             │                           / \
      │             │                          /   \
      │  Solid Core │                         /     \
      │             │                        /  Core \
      │             │                        \       /
      └─────────────┘                         \     /
      ◄────── L ────►                          \   /
                                                ▼
                                             ◄ 1.414 L ►
```

### 2.2 The "Center Hotspot vs. Corner Underdose" Trap
1. **The Sharp Corners**: Sharp corners have very few overlapping ray angles compared to the core. To deposit enough energy into the sharp external corner voxels to push them past the curing threshold ($D_{gel} \ge 1.0$), high-intensity beams must be projected through those angles.
2. **The 360° Ray Intersection at the Center**: Every single ray aimed at any corner or wall travels straight through the center of the cube.
3. **The Result**: 
   * The center of a solid cube receives massive cumulative dose (e.g., $D_{center} = 3.0 - 5.0$).
   * The diagonal corner beams project straight out into the liquid resin beyond the flat walls, creating intense **streak/halo artifacts** ("horns" or "wings").
   * The empty liquid resin outside the cube receives an out-of-part dose $D_{void} = 1.6 - 2.0$.
   * Because $D_{void} > D_{gel}$ ($1.6 > 1.0$), the liquid outside the cube solidifies into an amorphous blob of jelly.

### 2.3 The Process Window ($PW$) Explanation
The **Process Window ($PW$)** is defined as:
$$PW = D_{gel}^{min} - D_{void}^{max}$$
* If $PW > 0$: The minimum dose inside the part is strictly greater than the maximum dose anywhere in the liquid void. A single chemical threshold can separate solid part from liquid resin.
* If $PW < 0$: The highest dose in the void is greater than the lowest dose in the part. **It is physically impossible to cure the full part without curing chunks of the void.**

A solid 100% infill cube under linear parallel-beam projections has a severely **negative process window** ($PW \approx -0.8$).

### 2.4 How to Fix Cubes in VAM
1. **Shell / Hollow the Cube**: Give it a $1.5 - 2.0\text{ mm}$ wall thickness with an open internal cavity or drain holes. Removing the solid core eliminates the central ray congestion and cuts streak dose by over 80%.
2. **Internal Gyroid / Lattice Infill**: If solid structural behavior is needed, use a $15-20\%$ open-cell gyroid lattice inside the cube.

---

## 3. Why Benchy Fails Upright & Why Tilting Saves It

### 3.1 The Anatomy of 3DBenchy
A 3DBenchy has three extreme geometric features:
1. **The Cabin Roof and Chimney**: Thin solid slabs elevated high above the deck.
2. **The Open Cabin**: A large empty void directly beneath the roof, surrounded by 4 thin vertical pillars.
3. **The Bow & Keel Overhangs**: Deep horizontal concavities.

```
       UPRIGHT BENCHY (Severe Streaks)          TILTED BENCHY (Distributed Rays)
                 [Chimney]                                  / [Chimney]
                 ┌───┐                                     / ┌───┐
            ─────┴───┴───── (Roof)                        / /   /
            │   [Void]    │ (Open Cabin)                 / / [Void]
            └───┬─────┬───┘                             / /     /
        ────────┘     └────────                        / └─────┘
        \     Hull Void       /                       /  Hull  /
         \───────────────────/                       /────────/
   All horizontal rays crossing the roof      Rays slice diagonally through 
   blast through empty space in the same      features, spreading dose across 
   Z-slices, curing the open cabin.           multiple Z-planes!
```

### 3.2 What Happens When Benchy is Sliced Upright ($Z$-Up)
* In parallel-beam VAM, each horizontal $Z$-slice is optimized independently.
* In the slice containing the cabin roof:
  * The algorithm must project light to cure the flat roof.
  * In the exact same $Z$-slice, the rest of the space (the air around the boat) is empty void.
  * Rays from 360° passing through the thin roof overlap in the surrounding air.
* In the slice containing the open cabin:
  * The algorithm must cure 4 tiny $1\text{ mm}$ posts while keeping the large interior cabin 100% liquid.
  * Rays crossing diagonally between pillars dump dose into the center of the cabin.
* **The Result**: The cabin fills with solid cured resin, the chimney creates a ring of ghost jelly around the top, and the hull railings blur into a solid lump. The error reported by Tomo was $18.11\%$ voxel error with $PW = -0.82$.

### 3.3 Why Tilting (30°–45° Angle) Solves the Problem
1. **Breaks Coplanar Slice Alignment**: When Benchy is tilted at $35^\circ - 45^\circ$, no single $Z$-slice consists solely of a wide flat roof or solely of isolated tall posts.
2. **Distributes Optical Density Evenly Across $Z$**: The roof, cabin, and hull now span a continuum of slices. High-intensity rays for the roof enter at one $Z$-height and exit at another, spreading background dose across a large vertical volume instead of concentrating it into a single slice.
3. **Improves Radon Condition Number**: Slicing through angled surfaces prevents singular ray overlaps, dramatically widening the process window ($PW$) from negative to positive ($PW > 0$).

---

## 4. Demystifying the Optimizers: OSMO vs. BCLP

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             OPTIMIZER COMPARISON                                 │
├─────────────────────────┬────────────────────────────────────────────────────────┤
│          OSMO           │  • Heuristic target-space clipping & boosting.         │
│ (Object-Space Model     │  • Stalls in ~20 iterations on hard geometries.        │
│  Optimization)          │  • No mathematical guarantee; oscillations occur.      │
├─────────────────────────┼────────────────────────────────────────────────────────┤
│          BCLP           │  • Rigorous mathematical formulation (Lp-norm).        │
│ (Band-Constrained       │  • Enforces hard lower (gel) and upper (void) bounds.  │
│  Linear Programming)    │  • Integrates non-linear resin response & inhibition.  │
└─────────────────────────┴────────────────────────────────────────────────────────┘
```

### 4.1 What is OSMO (Object-Space Model Optimization)?

OSMO is an **iterative heuristic algorithm** that tries to guess an "ideal virtual target" $x_m$ such that when $x_m$ is projected and back-projected, the simulated dose $x$ matches the real target geometry.

#### How OSMO Steps Work:
1. **Initialize**: Start with the binary target geometry $x_m^{(0)}$ (1 inside part, 0 outside).
2. **Filter & Project**: Apply a high-pass ramp filter (Ram-Lak or Hamming) and forward-project to create an initial sinogram $b = A \cdot x_m$.
3. **Simulate Dose (Backprojection)**: Calculate the reconstructed dose in the vat: $x = A^T \cdot b$.
4. **Step Void (Reduce background dose)**:
   * Look at all voxels that should be empty (void).
   * If any void voxel received dose $x > d_l$ (upper void limit, e.g. $0.6$), calculate the excess error:
     $$\Delta_{void} = x - d_l$$
   * Subtract that error directly from the virtual model: $x_m[void] = x_m[void] - \Delta_{void}$.
5. **Step Gel (Boost part dose)**:
   * Look at all voxels that should be solid (gel).
   * If any gel voxel received dose $x < d_h$ (lower gel limit, e.g. $0.85$), calculate the deficit:
     $$\Delta_{gel} = d_h - x$$
   * Add that deficit to the virtual model: $x_m[gel] = x_m[gel] + \Delta_{gel}$.
6. **Project & Repeat**: Compute new sinogram $b = A \cdot x_m$, enforce non-negativity ($b \ge 0$), and repeat.

#### Why OSMO Stalls / Fails After ~20 Iterations:
* OSMO is **not a true mathematical optimizer** with a strictly decreasing objective function.
* When a geometry is physically constrained (e.g. solid cube or upright Benchy where non-negative rays *must* intersect the void), `stepVoid` removes dose from the virtual model, which starves the adjacent gel voxels. In the next half-iteration, `stepGel` pumps dose back in, which immediately spikes the void dose again.
* **The algorithm enters an infinite oscillatory plateau**. Iterations 25 through 150 produce zero improvement.

---

### 4.2 What is the Second Optimizer: BCLP (Band-Constrained Linear Programming)?

BCLP was formulated to solve the exact mathematical breakdown of OSMO.

#### The BCLP Mathematical Formulation:
Instead of heuristic guessing, BCLP treats VAM as a **constrained functional optimization problem**:
$$\min_{g \ge 0} \; \| \mathcal{W} \cdot \max(0, |\mathcal{R}(P \cdot g) - f_T| - \varepsilon) \|_{p}^{q}$$

Where:
* $g$: The projection sinogram (light intensity over angle and space), strictly non-negative ($g \ge 0$).
* $P$: The forward projection matrix (Radon forward transform / ray tracing).
* $P \cdot g$: The physical accumulated dose field in the resin vat.
* $\mathcal{R}(\cdot)$: The **Material Response Model** (converts optical dose to chemical gelation conversion, modeling oxygen inhibition threshold $D_0$).
* $f_T$: The target binary geometry ($1$ for gel, $0$ for void).
* $\varepsilon$: The **tolerance band** (acceptable dose margin where error penalty is zero).
* $\mathcal{W}$: Weighting mask (allows placing higher priority on keeping void un-cured).
* $\|\cdot\|_p$: The $L_p$ norm ($p=2$ for smooth least-squares, $p=\infty$ for minimax peak error suppression).

#### Why BCLP Outperforms OSMO:
1. **True Gradient Descent with Bounds**: Computes the exact analytical adjoint gradient $\nabla_g \mathcal{L}$ and projects onto the feasible space $g \ge 0$.
2. **Tolerance Band ($\varepsilon$)**: Recognizes that void voxels don't need zero dose; they just need to stay below the chemical induction threshold $D_{th} - \varepsilon$.
3. **Resin Chemistry Coupling**: Directly incorporates the non-linear induction curve of photopolymer resins (e.g. oxygen scavenging).

---

### 4.3 Summary Comparison of All VAM Optimizers

| Method | Type | Speed | Accuracy on Complex Shapes | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **FBP** (Filtered Backprojection) | Analytical Inverse | Instant ($<1\text{ s}$) | Very Low (Severe streaks, $PW < 0$) | Rough preview only |
| **OSMO** | Target-space Heuristic | Fast ($5-15\text{ s}$) | Good on smooth, convex, hollow shapes; Stalls on cubes/overhangs | Simple hollow geometries, thin shells |
| **BCLP** | Constrained $L_p$ Optimization | Moderate ($15-45\text{ s}$) | **Highest** (Strict threshold bounds, suppresses peak void dose) | Complex geometries, Benchy, lattices, industrial parts |
| **CAL** (Sigmoidal GD) | Gradient Descent + Sigmoid | Moderate ($20-40\text{ s}$) | High (Smooth transition modeling) | Organic shapes, biomaterials |
| **PM** (Penalized Maximum) | Penalty Function | Moderate ($20-30\text{ s}$) | High (Balances peak void vs mean gel) | Optical components, lenses |

---

## 5. What is Wrong with TOMO (The GUI)?

The repository [`computed-axial-lithography/tomo`](https://github.com/computed-axial-lithography/tomo) is an Electron/desktop GUI wrapper designed to provide a "click-and-print" interface for OpenCAL.

```
                  ┌─────────────────────────────────────┐
                  │              TOMO GUI               │
                  │  (High-level desktop UI wrapper)    │
                  └──────────────────┬──────────────────┘
                                     │ Calls via subprocess / bindings
                                     ▼
                  ┌─────────────────────────────────────┐
                  │             VAMToolbox              │
                  │   (Core Python mathematical engine) │
                  └─────────────────────────────────────┘
```

### The 5 Architectural & Practical Limitations of TOMO:

1. **Fixed / Naive Optimization Presets**:
   * Tomo defaults to running standard unconstrained OSMO with fixed iteration counts and default $d_h/d_l$ thresholds.
   * It does not expose advanced BCLP $L_p$-norm tuning, relaxation tolerances ($\varepsilon$), or customized spatial weight masks ($\mathcal{W}$).

2. **No Built-in Geometry Pre-flight Warnings**:
   * Tomo lets users import solid, un-shelled STLs (like a solid cube or upright Benchy) without warning them that solid blocks violate Radon non-negativity.
   * Users run 150 cycles expecting the error to drop, unaware that the physics of the geometry makes convergence impossible.

3. **Linear Metric Assumption vs. Chemical Reality**:
   * Tomo calculates error assuming a strict linear step-function without calibrating for resin photo-inhibition.
   * In your screenshot, Tomo flagged an $18.11\%$ voxel error because the predicted optical dose in the void reached $1.64$. In real printing, if resin has a high oxygen threshold, that background dose might not cure, but Tomo flags it as a massive failure because it lacks chemical calibration parameters.

4. **Aspect Ratio & Projector Canvas Constraints**:
   * Tomo's video export pipeline assumes specific standard projector aspect ratios and fixed frame-rate timings.
   * When using high-end hardware like the **Optoma ML1080** (native 1080p RGB laser, $1080 \times 1920$ portrait), Tomo may apply unwanted scaling or letterboxing mismatches.

5. **Why the OpenCAL Creators Use the Python Pipeline Directly**:
   * In the Discord transcript, Taylor W noted: *"The python package should actually work best"*.
   * The core engine `vamtoolbox` contains the full array of algorithms (BCLP, Slabbed Optimization, GPU Raytracing, Custom ImageSeq formatting), whereas Tomo only exposes a small fraction of these capabilities.

---

## 6. Hardware & Optics: OpenCAL + Optoma ML1080

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      OPENCAL + OPTOMA ML1080 BEST PRACTICES                     │
├─────────────────────────┬─────────────────────────────┬─────────────────────────┤
│ 1. Laser Flux (~1200lm) │ 2. Native Portrait Canvas   │ 3. Speed Sync (9.0 RPM) │
│ High intensity means    │ $1080 \times 1920$ native;  │ Motor speed must match  │
│ 1-3 rotations max.      │ 1:1 pixel projection;       │ video duration:         │
│ Overexposure is fast!   │ manual focus & keystone on. │ $6.667\text{ s}$ / rev. │
└─────────────────────────┴─────────────────────────────┴─────────────────────────┘
```

### 6.1 Understanding the Optoma ML1080 Laser Engine
* **Pure RGB Triple Laser**: Extremely narrow emission spectra (Blue $\sim 465\text{ nm}$, Green $\sim 525\text{ nm}$, Red $\sim 638\text{ nm}$).
* **Photoinitiator Selection**: Ensure your resin uses a blue-absorbing initiator:
  * **TPO / TPO-L**: Absorbs up to $420-430\text{ nm}$ (weak at $465\text{ nm}$, requires higher power).
  * **BAPO / Irgacure 819**: Good absorption extending to $450-460\text{ nm}$.
  * **Camphorquinone (CQ) + Amine**: Peak absorption right at $468\text{ nm}$ (perfect match for blue laser diode).
* **Extreme Contrast & Zero Black-Level Leakage**: Unlike cheap LCD or low-contrast LED projectors that leak gray background light into the resin, the ML1080 laser shuts off completely on black pixels ($0$ RGB = true zero photon flux).

### 6.2 Rotational Synchronization with OpenCAL
* OpenCAL runs at a calibrated speed of **$9.000\text{ RPM}$**.
* One full $360^\circ$ rotation takes exactly:
  $$T_{rev} = \frac{60}{9.0} = 6.6667\text{ seconds}$$
* At **$30\text{ FPS}$**, one rotation is exactly:
  $$N_{frames} = 6.6667 \times 30 = 200\text{ frames}$$
* At **$60\text{ FPS}$**, one rotation is exactly:
  $$N_{frames} = 6.6667 \times 60 = 400\text{ frames}$$
* **Exposure Dosage Rule of Thumb**:
  * With a 1200-lumen laser, most acrylate resins gel in **$1 - 3\text{ full rotations}$** ($7 - 20\text{ seconds}$ total print time).
  * Do not loop the video 10+ times as is common with 50-lumen LED projectors, or the entire vial will solidify.

---

## 7. The Step-by-Step Practical Recipe (How to Get Flawless Prints)

Follow this 4-step workflow to prepare, optimize, and print any model without errors:

```
  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
  │ 1. CAD Prep    │ ──► │ 2. Voxelize    │ ──► │ 3. BCLP Optim  │ ──► │ 4. OpenCAL MP4 │
  │ Shell/Hollow & │     │ High-res grid  │     │ Enforce hard   │     │ 1080x1920 @    │
  │ Tilt 35°-45°   │     │ (180-250 layers│     │ void/gel bounds│     │ 9 RPM (30 FPS) │
  └────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘
```

### Step 1: CAD / Slicer Preparation
1. **Hollow the part**: Set wall thickness to $1.5 - 2.5\text{ mm}$.
2. **Orient the part**: Rotate the STL $35^\circ - 45^\circ$ relative to the vertical $Z$-axis.
3. Export as binary `.stl`.

### Step 2: Run the Optimization Script
Save and run the following script using the Python environment in `VAMToolbox_alternative`:

```python
#!/usr/bin/env python3
"""
Full VAMToolbox Optimization Pipeline for OpenCAL + Optoma ML1080
"""
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import vamtoolbox as vam

# ── Configuration ─────────────────────────────────────────────────────────────
STL_FILE    = "models/benchy_tilted_hollow.stl"
OUTPUT_MP4  = "output/benchy_ml1080_opt.mp4"
PROJ_W      = 1080           # Optoma ML1080 portrait width
PROJ_H      = 1920           # Optoma ML1080 portrait height
RPM         = 9.0            # OpenCAL motor rotational speed
FPS         = 30             # Video framerate
LOOPS       = 2              # 2 rotations = 13.3 seconds total exposure
RESOLUTION  = 200            # Voxel grid resolution

# ── 1. Voxelization ───────────────────────────────────────────────────────────
print(f"[1/4] Voxelizing {STL_FILE} at {RESOLUTION} layers...")
target_geo = vam.geometry.TargetGeometry(stlfilename=STL_FILE, resolution=RESOLUTION)

# ── 2. Projection Geometry ────────────────────────────────────────────────────
N_ANGLES = 360
angles = np.linspace(0, 360 - 360 / N_ANGLES, N_ANGLES)
proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)

# ── 3. BCLP Optimization ──────────────────────────────────────────────────────
print(f"[2/4] Optimizing with BCLP (Band-Constrained Lp Optimization)...")
options = vam.optimize.Options(
    method="BCLP",
    n_iter=30,
    d_h=0.90,              # Minimum target gel dose
    d_l=0.45,              # Maximum void background dose
    eps=0.08,              # Tolerance band around target
    p=2,                   # L2-norm regularization
    filter="hamming",
    verbose="iter"
)

opt_sino, opt_recon, error = vam.optimize.optimize(target_geo, proj_geo, options)

# ── 4. Render 1080x1920 MP4 for Optoma ML1080 ────────────────────────────────
print(f"[3/4] Formatting {PROJ_W}x{PROJ_H} portrait frames for ML1080...")
sino = opt_sino.array
sino_norm = sino / np.max(sino)

frames_per_rev = int(round((60.0 / RPM) * FPS))   # 200 frames @ 9 RPM, 30 FPS
angle_indices = np.linspace(0, N_ANGLES - 1, frames_per_rev, dtype=int)

def make_canvas_frame(slice_2d):
    arr = (slice_2d.T * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    
    # Scale to 85% of projector width to maintain clean margins in the vial
    src_w, src_h = img.size
    scale = min(PROJ_W / src_w, PROJ_H / src_h) * 0.85
    nw, nh = int(round(src_w * scale)), int(round(src_h * scale))
    img_res = img.resize((nw, nh), Image.LANCZOS)
    
    canvas = Image.new("RGB", (PROJ_W, PROJ_H), (0, 0, 0))
    canvas.paste(img_res, ((PROJ_W - nw) // 2, (PROJ_H - nh) // 2))
    return np.array(canvas)

print(f"[4/4] Writing MP4 video ({LOOPS} loops, {LOOPS * (60.0/RPM):.1f}s exposure)...")
os.makedirs(os.path.dirname(OUTPUT_MP4) or ".", exist_ok=True)
with imageio.get_writer(OUTPUT_MP4, fps=FPS, macro_block_size=1, codec="libx264") as writer:
    for loop in range(LOOPS):
        for idx in angle_indices:
            writer.append_data(make_canvas_frame(sino_norm[:, idx, :]))
    
    # Add a 0.5-second pure black tail frame so projector doesn't linger on last image
    black = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
    for _ in range(int(FPS * 0.5)):
        writer.append_data(black)

print(f"DONE! Saved projection video: {OUTPUT_MP4}")
```

---

## 8. Summary Checklist Before You Print

1. [ ] **Model is hollowed** ($1.5 - 2.5\text{ mm}$ wall thickness) or has internal drain channels.
2. [ ] **Model is tilted** ($30^\circ - 45^\circ$) to break coplanar overhangs.
3. [ ] **Optimizer set to BCLP** with strict void limits ($d_l \le 0.45 - 0.50$).
4. [ ] **Optoma ML1080 is keystoned and manually focused** onto the center axis of the resin vial.
5. [ ] **Video canvas is native $1080 \times 1920$ portrait** with black margins.
6. [ ] **OpenCAL motor speed is locked at 9.000 RPM** (`CORRECTION_FACTOR` verified).
7. [ ] **Exposure loops set low (e.g., 8–12 loops / ~1–1.5 min)** to match resin gel point.

---

## 9. RPM Calibration & Exposure Control: Preventing the 5-Minute Over-Cure

### 9.1 Why You Saw the Cube at 1–1.5 min, But 5 min Ruined It

In CAL, chemical conversion is a race between the **In-Part Gel Dose** and the **Out-of-Part Background Dose**:

```
 Accumulated
 Dose (D)
    ▲
    │                                                   OVER-CURED BLOB
    │                                              (Both Part & Void Gel!)
    │                                          ───────► ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    │                                                 /        /
    │                                                /  Void  /
    │                            PERFECT WINDOW     /  Dose  /
    │                           ┌──────────────┐   /        /
D_th├───────────────────────────┼──────────────┼──/────────/───────────── (Gel Threshold)
    │                          /              /  /
    │                         /  Part Dose   /  /
    │                        /              /  /
    │                       /              /  /
    │                      /              /  /
    │                     /              /  /
  0 └────────────────────┴──────────────┴──┴────────────────────────► Time
                         ▲              ▲  ▲
                         │              │  │
                     1.0 min         1.5 min  5.0 min (Ruined!)
                  (Part gels!)    (Void starts to gel)
```

1. **The Gelation Threshold ($D_{th}$)**: Resin only solidifies when dose exceeds $D_{th}$ (e.g. $1.0$).
2. **Dose Accumulation Rate**:
   * The part accumulates dose at rate $R_{part} = 1.0\text{ units/min}$.
   * The void accumulates background streak dose at rate $R_{void} = 0.6\text{ units/min}$.
3. **What Happened Yesterday**:
   * At **$t = 1.0\text{ min}$**: Part dose reaches $1.0\text{ units}$ $\rightarrow$ **The cube solidifies and becomes visible!**
   * At **$t = 1.0 - 1.5\text{ min}$**: Part dose is $1.2 - 1.5$, Void dose is $0.6 - 0.9$ (still below $1.0$, so liquid remains liquid). **THIS WAS YOUR PRINT WINDOW.**
   * At **$t = 5.0\text{ min}$**: Part dose is $5.0$, but Void dose reaches $3.0$ (which is $3\times$ the gel threshold!). The entire vial cured into a solid block of plastic.

> **RULE OF THUMB**: In VAM, exposure is **NOT** like a curing oven. You **MUST STOP** the projection immediately after the part gels.

---

### 9.2 The Angular Drift Trap: Why $\pm 0.001\text{ RPM}$ Calibration is Mandatory

In VAM, the video frame timing and the motor speed must match with extreme precision:

$$\text{Time per 360° Video Rotation } (T_{vid}) = \text{Time per Motor Physical Revolution } (T_{motor}) = \frac{60}{\text{RPM}}$$

If the motor turns at **$9.05\text{ RPM}$** while the video is playing for **$9.00\text{ RPM}$**:
* Error per revolution: $\Delta t = 0.037\text{ seconds}$ ($\approx 2.0^\circ$ angular shift per rotation).
* By revolution 10 (at $1.1\text{ min}$): Angular shift is **$20.0^\circ$**!
* By revolution 30 (at $3.3\text{ min}$): Angular shift is **$60.0^\circ$**!
* **The light is now projecting into completely wrong areas of the rotating vial**, smearing the sharp corners of the cube into a cylinder and dumping dose into the void.

---

### 9.3 Step-by-Step: How to Calibrate Motor RPM on OpenCAL (The Exact Discord Procedure)

Follow this calibration procedure developed by the OpenCAL engineering team:

#### Phase 1: Coarse Measurement (20 Rotations)
1. In the OpenCAL GUI/CLI, set the target speed to **$9\text{ RPM}$**.
2. Place a small piece of tape or a pointer mark on the rotating vial holder.
3. Start your phone's stopwatch (or film in slow-motion).
4. Count exactly **$20\text{ full revolutions}$**.
5. Stop the stopwatch at the exact completion of the 20th rotation and record time $T_{20}$ (in seconds).
6. Calculate measured RPM:
   $$\text{MEASURED\_RPM} = \frac{20 \times 60}{T_{20}}$$
   *(Example: If 20 rotations took $131.5\text{ s}$, then $\text{MEASURED\_RPM} = 1200 / 131.5 = 9.125\text{ RPM}$)*.

#### Phase 2: Compute and Apply Correction Factor
7. Calculate the correction multiplier:
   $$\text{CORRECTION\_FACTOR} = \frac{\text{MEASURED\_RPM}}{\text{TARGET\_RPM}} = \frac{\text{MEASURED\_RPM}}{9.000}$$
8. Open your OpenCAL motor driver file:
   * For UART / TMC driver: `opencal/hardware/stepper/uart.py` inside `_rpm_to_vactual`
   * For Tic controller: Update the step rate multiplier / scaling in `tic.py`
9. Apply the `CORRECTION_FACTOR` to the target velocity calculation.

#### Phase 3: Fine Verification (50 Rotations)
10. Run the motor for **$50\text{ full rotations}$**.
11. Expected duration for 50 rotations at $9.000\text{ RPM}$:
    $$T_{target} = \frac{50 \times 60}{9.000} = 333.33\text{ seconds } (5\text{ min } 33.33\text{ s})$$
12. Your measured time should be **$333.3 \pm 0.3\text{ seconds}$** ($9.000 \pm 0.001\text{ RPM}$).

---

### 9.4 How to Control Video Duration (No More Infinite Loops)

Since default video players loop infinitely, you should bake the exact exposure duration directly into the generated MP4 file.

#### The Formula:
$$\text{Total Duration (seconds)} = N_{loops} \times \frac{60}{\text{RPM}}$$

| Number of Loops | Exposure Duration @ 9 RPM | Use Case (Optoma ML1080) |
| :--- | :--- | :--- |
| **6 loops** | $40.0\text{ seconds}$ | High-reactivity resin / thick parts |
| **9 loops** | **$60.0\text{ seconds}$ (1.0 min)** | **Optimal standard test (Your Cube Gel Point)** |
| **12 loops** | **$80.0\text{ seconds}$ (1.33 min)** | Upper limit for hollow/thin-wall parts |
| **15 loops** | $100.0\text{ seconds}$ (1.67 min)** | Background void will start to gel |
| **45 loops** | $300.0\text{ seconds}$ (5.0 min) | **Guaranteed Over-cured Solid Block** |

#### How to Export a Finite Video with Black Tail:
In your Python script, set `NUM_LOOPS = 9` (or `10`) and append a 5-second black tail:
```python
NUM_LOOPS = 9   # Exactly 1.0 minute exposure at 9 RPM
# Script appends 200 frames * 9 = 1800 frames, then writes black frames.
# When the video ends, the projector displays pure black, instantly halting curing!
```

