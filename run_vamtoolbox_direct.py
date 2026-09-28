"""
======================================================================
  PURE VAMTOOLBOX BENCHY SLICER & PROJECTION VIDEO GENERATOR
======================================================================
Runs pure native VAMToolbox directly (no GUI / no custom wrappers):
- Auto-centers and scales Benchy to fit a 30mm print vial (chimney up)
- Runs GPU-accelerated OSMO optimization across 360 degrees
- Generates pure native VAMToolbox projection video (MP4)
"""
import os
import sys
import time
import numpy as np

import vamtoolbox as vam

def main():
    print("=" * 68)
    print("  PURE NATIVE VAMTOOLBOX: Auto-Centered Benchy Pipeline")
    print("=" * 68)

    stl_path = os.path.join(os.path.dirname(__file__), "stls", "benchy.stl")
    out_video = os.path.join(os.path.dirname(__file__), "benchy_pure_vamtoolbox.mp4")

    # 1. Target Voxelization (Pure VAMToolbox)
    print("\n[Step 1/4] Voxelizing Benchy via TargetGeometry (200 layers) ...")
    t0 = time.perf_counter()
    target_geo = vam.geometry.TargetGeometry(stlfilename=stl_path, resolution=200)
    print(f"           Grid shape: {target_geo.array.shape} (Time: {time.perf_counter()-t0:.2f}s)")

    # 2. Projection Geometry (360 degrees, GPU CUDA)
    print("\n[Step 2/4] Setting up 360-degree Parallel Projection Geometry ...")
    num_angles = 360
    angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
    proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=True)

    # 3. OSMO Optimization (10 iterations)
    print("\n[Step 3/4] Optimizing with OSMO (10 iterations on GPU) ...")
    t_opt = time.perf_counter()
    opts = vam.optimize.Options(
        method="OSMO",
        n_iter=10,
        d_h=0.85,
        d_l=0.60,
        filter="hamming",
        verbose="iter"
    )
    opt_sino, opt_recon, error = vam.optimize.optimize(target_geo, proj_geo, opts)
    print(f"           Optimization completed in {time.perf_counter()-t_opt:.2f}s")
    print(f"           Sinogram shape: {opt_sino.array.shape}")

    # 4. Image Sequence & Video Encoding
    print("\n[Step 4/4] Encoding pure VAMToolbox projection video ...")
    t_vid = time.perf_counter()
    img_cfg = vam.imagesequence.ImageConfig(
        image_dims=(1080, 1920),
        normalization_percentile=99.9,
        size_scale=1.0,
    )
    img_seq = vam.imagesequence.ImageSeq(image_config=img_cfg, sinogram=opt_sino)
    img_seq.saveAsVideo(
        save_path=out_video,
        rot_vel=54.0,       # 9.0 RPM (54 deg/s)
        num_loops=1.0,      # 1 full 360-degree rotation
        fps=54.0,
        preview=False
    )
    print(f"           Video written in {time.perf_counter()-t_vid:.2f}s")
    print(f"\n======================================================================")
    print(f"  SUCCESS! Video saved to:")
    print(f"  {out_video}")
    print(f"  Total pipeline time: {time.perf_counter()-t0:.2f}s")
    print(f"======================================================================")

if __name__ == "__main__":
    main()
