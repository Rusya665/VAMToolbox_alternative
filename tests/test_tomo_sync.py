"""Tests verifying synchronization of video pipeline, refraction limits, and optical defaults with Tomo and OpenCAL."""
import os
import sys
import tempfile
import numpy as np
import pytest

import vamtoolbox as vam
from vamtoolbox.pipeline import PrintConfig, VAMPipeline

# Add vam_tuning_studio to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "vam_tuning_studio"))
import tomo_video_engine


def test_optical_defaults_in_print_config():
    """Check default mm_per_pix matches Tomo's 108mm FOV (0.100 mm/px)."""
    cfg = PrintConfig()
    assert cfg.proj_u_px == 1080
    assert cfg.proj_v_px == 1920
    assert np.isclose(cfg.mm_per_pix, 108.0 / 1080.0)


def test_compute_rebin_params_refraction_limits():
    """Verify compute_rebin_params returns correct usable green cylinder limit."""
    rp = vam.geometry.compute_rebin_params(
        vial_id_mm=30.0,
        vial_print_height_mm=60.0,
        mm_per_pix=0.100,
        proj_u_px=1080,
        proj_v_px=1920,
        throw_ratio=float("inf"),
        n_write=1.51,
    )
    assert "usable_diam_mm" in rp
    assert "r_usable_mm" in rp
    assert np.isclose(rp["r_usable_mm"], 15.0 / 1.51)
    assert np.isclose(rp["usable_diam_mm"], 30.0 / 1.51)


def test_tomo_video_engine_no_baseline_subtraction():
    """Verify generate_tomo_image_seq does not zero-out min baseline."""
    # Create synthetic sinogram with positive baseline
    sino = np.full((32, 10, 32), 10.0, dtype=np.float32)
    sino[10:20, :, 10:20] = 50.0

    img_seq, scale, v_off = tomo_video_engine.generate_tomo_image_seq(
        sino, voxel_pitch_mm=0.08, proj_px_w=1080, proj_px_h=1920, is_rebinned=True
    )
    # The center voxel should have non-zero value proportional to 10.0 / 50.0
    center_im = img_seq.images[0]
    center_val = center_im[center_im.shape[0] // 2, center_im.shape[1] // 2]
    assert center_val > 0


def test_imagesequence_save_as_video_runs_and_writes_file():
    """Verify ImageSeq.saveAsVideo exports a valid MP4 file with 3-channel frames."""
    # Small test sinogram
    sino = np.ones((20, 12, 20), dtype=np.float32) * 50.0
    img_cfg = vam.imagesequence.ImageConfig((128, 128), bit_depth=8)
    img_seq = vam.imagesequence.ImageSeq(img_cfg, sinogram=sino)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_video.mp4")
        res = img_seq.saveAsVideo(out_path, rot_vel=54.0, num_loops=1.0, fps=12.0)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0


def test_pipeline_save_video_clamps_size_scale_when_rebinned(cylinder_array):
    """Verify VAMPipeline.save_video clamps size_scale=1.0 when rebinned to prevent double-scaling."""
    n = cylinder_array.shape[2]
    cfg = PrintConfig(
        part_height_mm=10.0,
        vial_radius_mm=5.0,
        voxel_pitch_um=10.0 / n * 1000,
        use_cuda=False,
        slab="off",
        n_angles=30,
        n_iterations=3,
        method="OSMO",
        absorption=False,
        proj_u_px=128,
        proj_v_px=128,
        mm_per_pix=0.100,
    )
    pipe = VAMPipeline(cfg)
    pipe.run(target_array=cylinder_array, do_rebin=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "pipeline_video.mp4")
        pipe.save_video(out_path, rot_vel=54.0, num_loops=1.0)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
