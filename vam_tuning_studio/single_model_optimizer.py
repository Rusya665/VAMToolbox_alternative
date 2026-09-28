"""
Single-Model Sweet-Spot Optimizer Engine for VAMToolbox
======================================================
Focuses optimization on a single CAD model (STL) to discover the absolute
optimal printing parameters (Process Window, Voxel Error Rate, Dose Uniformity).

Streams trial updates on-the-fly to the GUI and persists all run artifacts into:
  <root>/runs/YYYY-MM-DD-HH-MM_<stl_name>/
"""

import os
import sys
import time
import json
import csv
import datetime
import trimesh
import numpy as np
import vamtoolbox as vam

from tomo_video_engine import (
    render_exact_tomo_video,
    DEFAULT_PROJ_PX_W,
    DEFAULT_PROJ_PX_H,
    DEFAULT_PROJ_WIDTH_MM,
    DEFAULT_PITCH_MM,
    DEFAULT_FPS,
    DEFAULT_RPM,
    DEFAULT_DURATION_S,
)

# Root directory resolution
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_RUNS_DIR = os.path.join(ROOT_DIR, "runs")
os.makedirs(ROOT_RUNS_DIR, exist_ok=True)

# Standard physical container boundaries
VIAL_DIAMETER_MM = 30.0
VIAL_HEIGHT_MM = 60.0
MAX_PRINTABLE_DIAMETER_MM = 25.0
MAX_PRINTABLE_HEIGHT_MM = 50.0


def sanitize_val(v):
    """Recursively converts any numpy scalar or array into native Python primitives."""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, np.ndarray):
        return [sanitize_val(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): sanitize_val(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [sanitize_val(x) for x in v]
    return v


def _embed_offset(arr: np.ndarray, off_mm: np.ndarray, pitch: float) -> np.ndarray:
    """
    Embeds an (nX, nY, nZ) voxel grid into a square, origin-centered grid
    preserving off-axis placement relative to the vial's rotational center (0,0).
    """
    a0, a1, nz = arr.shape
    ov0 = int(round(float(off_mm[0]) / max(pitch, 1e-6)))
    ov1 = int(round(float(off_mm[1]) / max(pitch, 1e-6)))
    m = max(a0 // 2 + abs(ov0), a1 // 2 + abs(ov1)) + 1
    side = 2 * m + 1
    out = np.zeros((side, side, nz), dtype=arr.dtype)
    s0 = m + ov0 - a0 // 2
    s1 = m + ov1 - a1 // 2
    out[s0:s0 + a0, s1:s1 + a1, :] = arr
    return out


def voxelize_model(stl_path: str, scale: float | list = 1.0,
                   rx: float = 0.0, ry: float = 0.0, rz: float = 0.0,
                   tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
                   pitch_mm: float = DEFAULT_PITCH_MM) -> tuple[vam.geometry.TargetGeometry, np.ndarray, float]:
    """
    Loads, transforms, and slices an STL model at true physical pitch (mm/voxel).
    Uses GPU OpenGL layer-slicer where available with robust trimesh fallback.
    """
    t0 = time.perf_counter()
    mesh = trimesh.load(stl_path, force="mesh")

    # 1. Center centroid initially for clean rotation
    mesh.apply_translation(-mesh.centroid)

    # 2. Apply Euler rotations (degrees)
    if rx != 0.0 or ry != 0.0 or rz != 0.0:
        rot_mat = trimesh.transformations.euler_matrix(np.radians(rx), np.radians(ry), np.radians(rz))
        mesh.apply_transform(rot_mat)

    # 3. Apply scale
    if isinstance(scale, (list, tuple, np.ndarray)):
        scale_mat = np.diag([float(scale[0]), float(scale[1]), float(scale[2]), 1.0])
        mesh.apply_transform(scale_mat)
    elif scale != 1.0:
        mesh.apply_scale(float(scale))

    # 4. Apply physical translations
    if tx != 0.0 or ty != 0.0 or tz != 0.0:
        mesh.apply_translation([float(tx), float(ty), float(tz)])

    extents = np.asarray(mesh.extents, dtype=float)
    b = mesh.bounds
    off_mm = (b[0][:2] + b[1][:2]) * 0.5

    # 5. Voxelize using GPU OpenGL layer-slicer or fast trimesh
    z_extent = float(extents[2]) if extents.size > 2 else max(pitch_mm, 1.0)
    n_layers = max(1, int(round(z_extent / max(pitch_mm, 1e-5))))

    target_body = None
    try:
        import tempfile
        fd, tmp_stl = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        mesh.export(tmp_stl)

        arr, _, _ = vam.voxelize.voxelizeTargetOpenGL(tmp_stl, n_layers)
        try:
            os.remove(tmp_stl)
        except Exception:
            pass

        arr = (np.asarray(arr) > 0).astype("uint8")
        arr = arr[:, :, ::-1]  # Undo OpenGL vertical inversion so chimney is UP

        # Pure VAMToolbox native centering on the rotation axis (0,0)
        nx, ny, nz = arr.shape
        max_xy = max(nx, ny)
        pad_x = (max_xy - nx) // 2
        pad_y = (max_xy - ny) // 2
        target_body = np.pad(arr, ((pad_x, max_xy - nx - pad_x), (pad_y, max_xy - ny - pad_y), (0, 0)))
    except Exception as e:
        try:
            vox = mesh.voxelized(pitch=pitch_mm).matrix.astype("uint8")
        except Exception:
            vox = trimesh.voxel.creation.voxelize_subdivide(mesh, pitch=pitch_mm).matrix.astype("uint8")
        nx, ny, nz = vox.shape
        max_xy = max(nx, ny)
        pad_x = (max_xy - nx) // 2
        pad_y = (max_xy - ny) // 2
        target_body = np.pad(vox, ((pad_x, max_xy - nx - pad_x), (pad_y, max_xy - ny - pad_y), (0, 0)))

    target_geo = vam.geometry.TargetGeometry(target=target_body)
    try:
        target_geo.zero_dose = None
    except Exception:
        pass

    t_vox = float(time.perf_counter() - t0)
    return target_geo, extents, t_vox


def compute_dose_metrics(recon: np.ndarray, target_arr: np.ndarray) -> dict:
    """
    Computes rigorous VAM dose metrics: in-part min, out-part max, robust Process Window, and balanced VER%.
    """
    r = np.asarray(recon, dtype=np.float32)
    tg = np.asarray(target_arr) > 0

    rmax = float(r.max()) if r.max() > 0 else 1.0
    r_norm = r / rmax

    in_voxels = r_norm[tg]
    out_voxels = r_norm[~tg]

    if len(in_voxels) == 0 or len(out_voxels) == 0:
        return {
            "in_min": 0.0, "in_mean": 0.0, "out_max": 0.0, "out_mean": 0.0,
            "window": -1.0, "ver_pct": 100.0, "threshold": 0.5,
            "in_hist": [], "out_hist": []
        }

    in_min = float(in_voxels.min())
    in_mean = float(in_voxels.mean())
    out_max = float(out_voxels.max())
    out_mean = float(out_voxels.mean())

    # Robust Process Window using 1st percentile of in-part and 99th percentile of out-part
    # (prevents single-pixel boundary noise or dust voxels from dominating the score)
    in_p01 = float(np.percentile(in_voxels, 1.0))
    out_p99 = float(np.percentile(out_voxels, 99.0))
    window = float(in_p01 - out_p99)

    # Find optimal threshold that separates solid part from liquid resin
    all_doses = np.sort(np.concatenate([in_voxels, out_voxels]))
    candidates = all_doses[::max(1, len(all_doses) // 100)]
    best_ver = float("inf")
    threshold = float((in_mean + out_mean) / 2.0)
    for c in candidates:
        under = float(np.sum(in_voxels < c) / len(in_voxels) * 100.0)
        over = float(np.sum(out_voxels >= c) / len(out_voxels) * 100.0)
        ver = (under + over) / 2.0
        if ver < best_ver:
            best_ver = ver
            threshold = float(c)

    in_under = float(np.sum(in_voxels < threshold) / max(1, len(in_voxels)) * 100.0)
    out_over = float(np.sum(out_voxels >= threshold) / max(1, len(out_voxels)) * 100.0)
    ver_pct = float((in_under + out_over) / 2.0)

    # Normalized percentage histograms (0% - 100% per bin) for clear visual comparison
    bins = np.linspace(0.0, 1.0, 65)
    in_h, _ = np.histogram(in_voxels, bins=bins)
    out_h, _ = np.histogram(out_voxels, bins=bins)

    in_hist_pct = [round(float(x / len(in_voxels) * 100.0), 2) for x in in_h]
    out_hist_pct = [round(float(x / len(out_voxels) * 100.0), 2) for x in out_h]

    return {
        "in_min": in_min,
        "in_mean": in_mean,
        "out_max": out_max,
        "out_mean": out_mean,
        "window": window,
        "ver_pct": ver_pct,
        "in_under_pct": in_under,
        "out_over_pct": out_over,
        "threshold": threshold,
        "in_hist": in_hist_pct,
        "out_hist": out_hist_pct,
    }


def generate_search_trials(mode: str = "quick", custom_params: dict = None) -> list[dict]:
    """
    Generates structured trial combinations for single-model hyperparameter search.
    Benchmarks OSMO, BCLP, and CAL with solid-volume parameterizations:
    - quick: 12 trials (6 OSMO + 6 BCLP)
    - deep: 32 trials (16 OSMO + 16 BCLP/CAL)
    """
    if mode == "custom" and custom_params:
        return [custom_params]

    trials = []
    if mode == "quick":
        # 6 OSMO trials (solid object-space optimization across iteration depth and dose margins)
        for n_iter in [5, 10]:
            for dh, dl in [(0.85, 0.50), (0.90, 0.40), (0.80, 0.60)]:
                trials.append({
                    "method": "OSMO",
                    "n_angles": 360,
                    "n_iter": int(n_iter),
                    "filter": "hamming",
                    "d_h": float(dh),
                    "d_l": float(dl),
                    "learning_rate": float(0.0),
                    "eps": float(0.10),
                    "inhibition": float(0.0),
                })
        # 6 BCLP trials (solid projection-space optimization with band constraints)
        for n_iter in [5, 10]:
            for dh, dl in [(0.85, 0.50), (0.90, 0.40), (0.80, 0.55)]:
                trials.append({
                    "method": "BCLP",
                    "n_angles": 360,
                    "n_iter": int(n_iter),
                    "filter": "hamming",
                    "d_h": float(dh),
                    "d_l": float(dl),
                    "learning_rate": float(0.005),
                    "eps": float(0.10),
                    "inhibition": float(0.0),
                })
        return trials[:12]

    elif mode == "deep":
        # 16 OSMO trials across iteration depths and threshold bands
        for n_iter in [5, 8, 12, 18]:
            for dh in [0.90, 0.85, 0.80, 0.75]:
                dl = round(max(0.20, dh - 0.35), 2)
                trials.append({
                    "method": "OSMO",
                    "n_angles": 360,
                    "n_iter": int(n_iter),
                    "filter": "hamming",
                    "d_h": float(dh),
                    "d_l": float(dl),
                    "learning_rate": float(0.0),
                    "eps": float(0.10),
                    "inhibition": float(0.0),
                })
        # 16 BCLP & CAL trials across iteration depths and threshold bands
        for m in ["BCLP", "CAL"]:
            for n_iter in [5, 10]:
                for dh in [0.90, 0.85, 0.80, 0.75]:
                    dl = round(max(0.20, dh - 0.35), 2)
                    trials.append({
                        "method": m,
                        "n_angles": 360,
                        "n_iter": int(n_iter),
                        "filter": "hamming",
                        "d_h": float(dh),
                        "d_l": float(dl),
                        "learning_rate": float(0.008 if m == "CAL" else 0.005),
                        "eps": float(0.08 if m == "BCLP" else 0.10),
                        "inhibition": float(0.0),
                    })
        return trials[:32]

    return trials


def run_single_model_optimization_stream(stl_path: str,
                                         stl_name: str = "model",
                                         scale: float | list = 1.0,
                                         rx: float = 0.0, ry: float = 0.0, rz: float = 0.0,
                                         tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
                                         pitch_mm: float = DEFAULT_PITCH_MM,
                                         search_mode: str = "quick",
                                         custom_params: dict = None,
                                         rpm: float = DEFAULT_RPM,
                                         duration_s: float = DEFAULT_DURATION_S,
                                         video_rotation_deg: float = 0.0,
                                         vial_correction: bool = False,
                                         stop_checker = None):
    """
    Generator that executes the single-model optimization, streams live trial events,
    creates the root run folder, and persists all publication charts and 1:1 Tomo video.
    Supports graceful early termination via stop_checker.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    clean_stl_name = os.path.splitext(os.path.basename(stl_name))[0].replace(" ", "_")
    run_folder_name = f"{now_str}_{clean_stl_name}"
    run_dir = os.path.join(ROOT_RUNS_DIR, run_folder_name)
    charts_dir = os.path.join(run_dir, "charts")
    videos_dir = os.path.join(run_dir, "videos")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    yield sanitize_val({
        "event": "started",
        "run_folder": run_dir,
        "run_name": run_folder_name,
        "stl_name": clean_stl_name,
        "timestamp": now_str,
    })

    # 2. Voxelize the model
    target_geo, extents, t_vox = voxelize_model(
        stl_path, scale=scale, rx=rx, ry=ry, rz=rz, tx=tx, ty=ty, tz=tz, pitch_mm=pitch_mm
    )

    yield sanitize_val({
        "event": "voxelized",
        "grid_shape": list(target_geo.array.shape),
        "voxel_count": int(np.sum(target_geo.array > 0)),
        "extents_mm": [float(extents[0]), float(extents[1]), float(extents[2])],
        "voxelize_time_s": round(float(t_vox), 2),
    })

    # 3. Generate Search Trials
    trials_list = generate_search_trials(search_mode, custom_params)
    total_trials = len(trials_list)

    best_score = -float("inf")
    best_trial_data = None
    best_sino = None
    best_recon = None
    all_trials_records = []
    stopped_early = False

    for i, t_cfg in enumerate(trials_list):
        # Check if user requested early stop
        if stop_checker:
            is_stopped = stop_checker() if callable(stop_checker) else getattr(stop_checker, "is_set", lambda: False)()
            if is_stopped:
                stopped_early = True
                break

        trial_num = i + 1
        t_start = time.perf_counter()

        n_angles = int(t_cfg.get("n_angles", 360))
        angles = np.linspace(0, 360 - 360 / n_angles, n_angles)
        proj_geo = vam.geometry.ProjectionGeometry(angles=angles, ray_type="parallel", CUDA=True)

        method = str(t_cfg.get("method", "OSMO"))
        n_iter = int(t_cfg.get("n_iter", 5))
        filt = t_cfg.get("filter", "hamming")
        if str(filt).lower() in ["none", "nan", ""]:
            filt = None

        opt_kwargs = {
            "method": method,
            "n_iter": n_iter,
            "d_h": float(t_cfg.get("d_h", 0.90)),
            "d_l": float(t_cfg.get("d_l", 0.30)),
            "filter": filt,
            "verbose": False,
        }
        if method == "OSMO":
            opt_kwargs["inhibition"] = float(t_cfg.get("inhibition", 0.0))
        elif method == "BCLP":
            opt_kwargs["eps"] = float(t_cfg.get("eps", 0.10))
            opt_kwargs["learning_rate"] = float(t_cfg.get("learning_rate", 0.005))
        elif method == "CAL":
            opt_kwargs["learning_rate"] = float(t_cfg.get("learning_rate", 0.01))

        opts = vam.optimize.Options(**opt_kwargs)
        sino, recon, _ = vam.optimize.optimize(target_geo, proj_geo, opts)
        opt_time = float(time.perf_counter() - t_start)

        recon_arr = recon.array if hasattr(recon, "array") else recon
        metrics = compute_dose_metrics(recon_arr, target_geo.array)

        solid_contrast = float(metrics["in_mean"] - metrics["out_mean"])
        score = float((metrics["window"] * 50.0) + (solid_contrast * 50.0) - (metrics["ver_pct"] * 0.5))
        is_new_best = bool(score > best_score)

        if is_new_best or best_trial_data is None:
            best_score = score
            best_sino = sino
            best_recon = recon_arr
            best_trial_data = sanitize_val({
                "trial_idx": int(trial_num),
                "score": round(score, 2),
                **t_cfg,
                **metrics,
                "opt_time_s": round(opt_time, 2),
            })

        trial_record = sanitize_val({
            "trial_idx": int(trial_num),
            "method": method,
            "n_angles": int(n_angles),
            "n_iter": int(n_iter),
            "filter": str(filt),
            "d_h": float(opt_kwargs["d_h"]),
            "d_l": float(opt_kwargs["d_l"]),
            "window": round(float(metrics["window"]), 3),
            "ver_pct": round(float(metrics["ver_pct"]), 2),
            "in_min": round(float(metrics["in_min"]), 3),
            "out_max": round(float(metrics["out_max"]), 3),
            "score": round(float(score), 2),
            "opt_time_s": round(float(opt_time), 2),
            "is_best": bool(is_new_best),
        })
        all_trials_records.append(trial_record)

        yield sanitize_val({
            "event": "trial_complete",
            "trial_idx": int(trial_num),
            "total_trials": int(total_trials),
            "trial_record": trial_record,
            "metrics": metrics,
            "is_new_best": bool(is_new_best),
            "best_score": round(float(best_score), 2),
            "best_trial_idx": int(best_trial_data["trial_idx"] if best_trial_data else trial_num),
        })

    if best_sino is not None:
        # 5. Persist Best Sinogram & Reconstruction Arrays
        np.save(os.path.join(run_dir, "best_sino.npy"), best_sino.array if hasattr(best_sino, "array") else best_sino)
        np.save(os.path.join(run_dir, "best_recon.npy"), best_recon)

        # 6. Save trials.csv and run_summary.json
        if all_trials_records:
            csv_path = os.path.join(run_dir, "trials.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_trials_records[0].keys()))
                writer.writeheader()
                writer.writerows(all_trials_records)

        summary_path = os.path.join(run_dir, "run_summary.json")
        with open(summary_path, "w") as f:
            json.dump(sanitize_val({
                "run_folder": run_dir,
                "stl_name": clean_stl_name,
                "timestamp": now_str,
                "pitch_mm": float(pitch_mm),
                "extents_mm": [float(x) for x in extents],
                "voxel_shape": list(target_geo.array.shape),
                "total_trials": len(all_trials_records),
                "planned_trials": total_trials,
                "stopped_early": stopped_early,
                "best_trial": best_trial_data,
                "trials": all_trials_records,
            }), f, indent=2)

        # 7. Render 1:1 Exact Tomo Video into videos/
        video_filename = f"projection_{clean_stl_name}_{rpm:.1f}rpm_{int(duration_s)}s.mp4"
        video_out_path = os.path.join(videos_dir, video_filename)

        render_exact_tomo_video(
            sino=best_sino,
            out_path=video_out_path,
            voxel_pitch_mm=pitch_mm,
            fps=DEFAULT_FPS,
            rpm=rpm,
            duration_s=duration_s,
            v_offset_mm=tz,
            video_intensity=1.0,
            video_rotation_deg=float(video_rotation_deg),
            vial_correction=bool(vial_correction),
        )

        yield sanitize_val({
            "event": "finished",
            "run_folder": run_dir,
            "run_name": run_folder_name,
            "summary_file": summary_path,
            "csv_file": csv_path if all_trials_records else None,
            "video_file": video_out_path,
            "video_filename": video_filename,
            "best_trial": best_trial_data,
            "stopped_early": stopped_early,
        })
