"""
Tomo-Native 1:1 High-Fidelity Video Engine for VAMToolbox
=========================================================
Generates print-ready projection video sequences (MP4) matching Tomo bit-for-bit:
  - Exact optical scale mapping (mm/voxel to projector px)
  - Bottom-origin DLP projection vertical flip (np.flipud)
  - 54.0 FPS @ RPM angular increment synchronization (1.0 deg/frame at 9 RPM)
  - Instant stream-looped encoding via bundled imageio-ffmpeg (H.264 / H.265 yuv420p)
  - Preview frame extraction for in-GUI real-time 360-degree scrubbing
"""

import os
import subprocess
import tempfile
import numpy as np
import vamtoolbox as vam

DEFAULT_PROJ_PX_W = 1080
DEFAULT_PROJ_PX_H = 1920
DEFAULT_PROJ_WIDTH_MM = 108.0    # 108mm FOV -> 0.1 mm/px (standard OpenCAL projector)
DEFAULT_PITCH_MM = 0.0797        # 79.7 um (OpenCAL V2 standard pitch)
DEFAULT_FPS = 54.0
DEFAULT_RPM = 9.0
DEFAULT_DURATION_S = 60.0


def compute_video_scale(sino_arr: np.ndarray, voxel_pitch_mm: float = DEFAULT_PITCH_MM,
                        proj_px_w: int = DEFAULT_PROJ_PX_W, proj_px_h: int = DEFAULT_PROJ_PX_H,
                        proj_width_mm: float = DEFAULT_PROJ_WIDTH_MM, is_rebinned: bool = False) -> float:
    """
    Computes exact projector px-per-voxel true scale (1:1 with Tomo).
    Rebinned sinogram is already in projector pixels (base 1.0);
    Parallel sinogram maps voxels -> projector via (voxel_pitch_mm * proj_px_w) / proj_width_mm.
    Clamped so the sinogram always fits inside the projector canvas.
    """
    if sino_arr is None or sino_arr.size == 0:
        return 1.0

    if is_rebinned:
        true_scale = 1.0
    else:
        true_scale = (float(voxel_pitch_mm) * float(proj_px_w)) / max(float(proj_width_mm), 1e-6)

    n_r, n_z = sino_arr.shape[0], sino_arr.shape[2]
    fit = min(proj_px_h / max(n_z, 1), proj_px_w / max(n_r, 1)) * 0.98
    if true_scale > fit:
        return fit
    return true_scale


def compute_vertical_offset_px(sino_arr: np.ndarray, v_offset_mm: float = 0.0,
                               true_scale: float = 1.0, proj_px_w: int = DEFAULT_PROJ_PX_W,
                               proj_px_h: int = DEFAULT_PROJ_PX_H, proj_width_mm: float = DEFAULT_PROJ_WIDTH_MM) -> int:
    """
    Converts physical Z vertical shift in millimeters to projector canvas pixels.
    """
    if sino_arr is None or sino_arr.size == 0:
        return 0
    mm_per_px = proj_width_mm / max(proj_px_w, 1)
    off_px = -(float(v_offset_mm) / max(mm_per_px, 1e-9))
    s_v = sino_arr.shape[2] * true_scale
    max_off = max(0.0, (proj_px_h - s_v) / 2.0 - 1)
    return int(round(max(-max_off, min(max_off, off_px))))


def generate_tomo_image_seq(sino: vam.geometry.Sinogram | np.ndarray,
                            voxel_pitch_mm: float = DEFAULT_PITCH_MM,
                            proj_px_w: int = DEFAULT_PROJ_PX_W,
                            proj_px_h: int = DEFAULT_PROJ_PX_H,
                            proj_width_mm: float = DEFAULT_PROJ_WIDTH_MM,
                            v_offset_mm: float = 0.0,
                            video_intensity: float = 1.0,
                            is_rebinned: bool = False,
                            vial_radius_mm: float = 15.0,
                            resin_ri: float = 1.51,
                            video_rotation_deg: float = 0.0,
                            vial_correction: bool = False):
    """
    Constructs the formatted VAMToolbox ImageSeq object matching Tomo bit-for-bit.
    Applies cylindrical vial refraction correction (rebinFanBeam) only if vial_correction=True and not already rebinned.
    """
    raw_arr = sino.array if isinstance(sino, vam.geometry.Sinogram) else sino
    arr = np.copy(raw_arr)

    sino_to_use = arr
    true_scale = compute_video_scale(arr, voxel_pitch_mm, proj_px_w, proj_px_h, proj_width_mm, is_rebinned or vial_correction)
    size_scale = true_scale

    # Only apply cylindrical vial refraction rebinning if explicitly requested (matching Tomo's self.vial_correction = False default)
    if vial_correction and not is_rebinned and arr.ndim == 3:
        try:
            mm_per_pix = proj_width_mm / max(proj_px_w, 1)
            rp = vam.geometry.compute_rebin_params(
                vial_id_mm=vial_radius_mm * 2.0,
                vial_print_height_mm=arr.shape[2] * voxel_pitch_mm,
                mm_per_pix=mm_per_pix,
                proj_u_px=proj_px_w,
                proj_v_px=proj_px_h,
                throw_ratio=float("inf"),
                n_write=resin_ri,
            )
            pg = getattr(sino, "proj_geo", None)
            if pg is None:
                pg = vam.geometry.ProjectionGeometry(
                    angles=np.linspace(0, 359, arr.shape[1]), ray_type="parallel"
                )
            s_obj = vam.geometry.Sinogram(arr.astype(np.float32), pg)
            rebinned = vam.geometry.rebinFanBeam(
                sinogram=s_obj,
                vial_width=rp["vial_width_px"],
                N_screen=rp["N_screen"],
                n_write=resin_ri,
                throw_ratio=float("inf")
            )
            sino_to_use = rebinned
            size_scale = 1.0
        except Exception as e:
            print(f"[Rebin Fallback] {e}")
            sino_to_use = arr
            size_scale = true_scale

    if isinstance(sino_to_use, vam.geometry.Sinogram):
        sino_obj = sino_to_use
    else:
        pg = getattr(sino, "proj_geo", None)
        opt = getattr(sino, "options", None)
        sino_obj = vam.geometry.Sinogram(sino_to_use, pg, opt) if pg is not None else sino_to_use

    v_off_px = compute_vertical_offset_px(arr, v_offset_mm, size_scale, proj_px_w, proj_px_h, proj_width_mm)

    if int(video_rotation_deg) in [90, 270]:
        canvas_dims = (int(proj_px_h), int(proj_px_w))
    else:
        canvas_dims = (int(proj_px_w), int(proj_px_h))

    iconfig = vam.imagesequence.ImageConfig(
        image_dims=canvas_dims,
        rotate_angle=float(video_rotation_deg),
        size_scale=size_scale,
        v_offset=v_off_px,
        normalization_percentile=99.9,
        intensity_scale=float(video_intensity),
    )
    image_seq = vam.imagesequence.ImageSeq(
        image_config=iconfig, sinogram=sino_obj
    )
    return image_seq, size_scale, v_off_px


def extract_preview_frames(sino: vam.geometry.Sinogram | np.ndarray,
                           voxel_pitch_mm: float = DEFAULT_PITCH_MM,
                           proj_px_w: int = DEFAULT_PROJ_PX_W,
                           proj_px_h: int = DEFAULT_PROJ_PX_H,
                           proj_width_mm: float = DEFAULT_PROJ_WIDTH_MM,
                           v_offset_mm: float = 0.0,
                           video_intensity: float = 1.0,
                           is_rebinned: bool = False,
                           video_rotation_deg: float = 0.0,
                           vial_correction: bool = False) -> tuple[list[np.ndarray], float]:
    """
    Extracts one full rotation (360 degrees) of vertically-flipped frames as uint8 arrays.
    """
    image_seq, _, _ = generate_tomo_image_seq(
        sino, voxel_pitch_mm, proj_px_w, proj_px_h, proj_width_mm, v_offset_mm, video_intensity, is_rebinned,
        video_rotation_deg=video_rotation_deg, vial_correction=vial_correction
    )
    frames = [np.ascontiguousarray(np.flipud(im)) for im in image_seq.images]
    n = len(frames)
    fps = max(12.0, min(54.0, float(n)))
    return frames, fps


def render_exact_tomo_video(sino: vam.geometry.Sinogram | np.ndarray,
                            out_path: str,
                            voxel_pitch_mm: float = DEFAULT_PITCH_MM,
                            proj_px_w: int = DEFAULT_PROJ_PX_W,
                            proj_px_h: int = DEFAULT_PROJ_PX_H,
                            proj_width_mm: float = DEFAULT_PROJ_WIDTH_MM,
                            fps: float = DEFAULT_FPS,
                            rpm: float = DEFAULT_RPM,
                            duration_s: float = DEFAULT_DURATION_S,
                            v_offset_mm: float = 0.0,
                            video_intensity: float = 1.0,
                            codec: str = "h264",
                            is_rebinned: bool = False,
                            video_rotation_deg: float = 0.0,
                            vial_correction: bool = False) -> str:
    """
    Encodes the projection video directly via pure native VAMToolbox ImageSeq.saveAsVideo.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    image_seq, true_scale, v_off_px = generate_tomo_image_seq(
        sino, voxel_pitch_mm, proj_px_w, proj_px_h, proj_width_mm, v_offset_mm, video_intensity, is_rebinned,
        video_rotation_deg=video_rotation_deg, vial_correction=vial_correction
    )

    # Save keyframe screenshot PNGs directly into the run directory for immediate inspection
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(out_path)))
    if image_seq.images:
        try:
            import imageio
            for deg, name in [(0, "frame0.png"), (90, "frame90.png"), (180, "frame180.png"), (270, "frame270.png")]:
                idx = int(deg / 360.0 * len(image_seq.images)) % len(image_seq.images)
                snap = np.ascontiguousarray(np.flipud(image_seq.images[idx]))
                imageio.imwrite(os.path.join(run_dir, name), snap)
        except Exception as se:
            print(f"[Snapshot Export Warning] {se}")

    rot_vel = float(rpm) * 6.0  # RPM -> deg/s (e.g. 9 RPM = 54 deg/s)
    num_loops = max(1.0, float(duration_s) * abs(float(rpm)) / 60.0)

    # Pure native VAMToolbox video encoder
    image_seq.saveAsVideo(
        save_path=out_path,
        rot_vel=rot_vel,
        num_loops=num_loops,
        fps=fps,
        preview=False
    )
    return out_path
