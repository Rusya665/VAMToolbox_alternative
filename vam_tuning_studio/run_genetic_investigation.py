#!/usr/bin/env python3
"""
OpenCAL / VAM Parameter Sensitivity & Genetic-Greedy Exploration Engine

Features:
  - 3D Arbitrary STL Voxelization with interactive transforms (Scaling, Rotation Euler XYZ, Translation).
  - Robust CPU Trimesh voxelizer (Zero OpenGL dependencies, zero wglChoosePixelFormatARB crashes).
  - Target-specific Sweet-Spot Auto-Optimizer with live callbacks.
  - Thread-safe callable directly from GUI or standalone CLI.
  - Graceful start / stop via threading.Event.
  - Atomic per-run isolation (runs/run_YYYYMMDD_HHMMSS/).
"""

import os
import sys
import time
import json
import random
import threading
import psutil
import numpy as np
import pandas as pd
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local package path is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import vamtoolbox as vam

try:
    from generate_charts import generate_all_charts
except ImportError:
    from vam_tuning_studio.generate_charts import generate_all_charts

STUDIO_DIR = os.path.abspath(os.path.dirname(__file__))
RUNS_DIR = os.path.join(STUDIO_DIR, "runs")
UPLOADS_DIR = os.path.join(STUDIO_DIR, "uploads")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

STLS = {
    "cube": "stls/cube.stl",
    "sphere": "stls/sphere.stl",
    "captiveRing": "stls/captiveRing.stl",
    "benchy": "stls/benchy.stl",
}

# Physical Vial Constraints
VIAL_DIAMETER_MM = 30.0
VIAL_HEIGHT_MM = 60.0
MAX_PRINTABLE_DIAMETER_MM = 25.0  # 2.5mm margin from glass wall
MAX_PRINTABLE_HEIGHT_MM = 50.0    # 5mm margin from top/bottom

DEFAULT_RESOLUTION = 75
CUDA_AVAILABLE = True
_GEOMETRY_CACHE = {}

def voxelize_transformed_mesh(stl_path: str, scale: float = 1.0, rx: float = 0.0, ry: float = 0.0, rz: float = 0.0, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0, resolution: int = DEFAULT_RESOLUTION):
    """
    Loads, centers, transforms (scale, rotate, translate) an arbitrary STL mesh,
    verifies boundary conditions, and voxelizes it onto a cylindrical-safe grid.
    """
    scale_str = f"({scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f})" if isinstance(scale, (list, tuple, np.ndarray)) else f"{float(scale):.3f}"
    print(f"[Voxelizer] Loading {stl_path} | scale={scale_str} | rot=({rx:.1f}, {ry:.1f}, {rz:.1f}) | pos=({tx:.1f}, {ty:.1f}, {tz:.1f}) ...")
    t0 = time.perf_counter()

    mesh = trimesh.load(stl_path)
    # 1. Center centroid first to allow clean rotation around center
    mesh.apply_translation(-mesh.centroid)

    # 2. Apply Euler rotations (XYZ)
    if rx != 0.0 or ry != 0.0 or rz != 0.0:
        rot_matrix = trimesh.transformations.euler_matrix(np.radians(rx), np.radians(ry), np.radians(rz))
        mesh.apply_transform(rot_matrix)

    # 3. Apply scale (supports uniform float or [sx, sy, sz] non-uniform)
    if isinstance(scale, (list, tuple, np.ndarray)):
        scale_matrix = np.diag([float(scale[0]), float(scale[1]), float(scale[2]), 1.0])
        mesh.apply_transform(scale_matrix)
    elif scale != 1.0:
        mesh.apply_scale(float(scale))

    # 4. Apply physical translations
    if tx != 0.0 or ty != 0.0 or tz != 0.0:
        mesh.apply_translation([tx, ty, tz])

    extents = mesh.extents
    print(f"[Voxelizer] Final physical extents: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} mm")

    # 5. Voxelize using fast subdivide (runs in ~1.5s even on 3M face models)
    pitch = MAX_PRINTABLE_DIAMETER_MM / resolution
    try:
        vox = trimesh.voxel.creation.voxelize_subdivide(mesh, pitch=pitch).matrix.astype(np.float32)
    except Exception:
        vox = mesh.voxelized(pitch=pitch).matrix.astype(np.float32)

    # 6. Square-pad in X-Y to maintain circular reconstruction symmetry for ProjectionGeometry
    nx, ny, nz = vox.shape
    max_xy = max(nx, ny, resolution)
    pad_x = (max_xy - nx) // 2
    pad_y = (max_xy - ny) // 2
    padded = np.pad(vox, ((pad_x, max_xy - nx - pad_x), (pad_y, max_xy - ny - pad_y), (0, 0)))

    target = vam.geometry.TargetGeometry(target=padded)
    t_vox = time.perf_counter() - t0
    print(f"[Voxelizer] Voxelized in {t_vox:.2f}s (Shape: {target.array.shape}, Gel voxels: {len(target.gel_inds[0])}, Fill: {target.array.mean()*100:.2f}%)")
    return target, extents

def get_cached_geometry(geom_name: str, resolution: int = DEFAULT_RESOLUTION):
    """
    Default auto-scaling for the standard 4 benchmarks.
    """
    cache_key = (geom_name, resolution)
    if cache_key not in _GEOMETRY_CACHE:
        stl_path = STLS[geom_name]
        mesh = trimesh.load(stl_path)
        extents = mesh.extents
        scale_xy = MAX_PRINTABLE_DIAMETER_MM / max(extents[0], extents[1])
        scale_z = MAX_PRINTABLE_HEIGHT_MM / extents[2]
        scale_factor = min(scale_xy, scale_z, 1.0) if geom_name != "cube" else scale_xy

        target, _ = voxelize_transformed_mesh(stl_path, scale=scale_factor, resolution=resolution)
        _GEOMETRY_CACHE[cache_key] = target

    return _GEOMETRY_CACHE[cache_key]

def compute_detailed_metrics(target_geo, recon_array):
    recon = np.copy(recon_array)
    recon_max = np.max(recon)
    if recon_max > 0:
        recon = recon / recon_max

    gel_inds = target_geo.gel_inds
    void_inds = target_geo.void_inds

    gel_vals = recon[gel_inds]
    void_vals = recon[void_inds]
    num_total = len(gel_vals) + len(void_vals)

    min_gel = float(np.min(gel_vals))
    max_gel = float(np.max(gel_vals))
    mean_gel = float(np.mean(gel_vals))
    std_gel = float(np.std(gel_vals))
    p5_gel = float(np.percentile(gel_vals, 5))

    min_void = float(np.min(void_vals))
    max_void = float(np.max(void_vals))
    mean_void = float(np.mean(void_vals))
    std_void = float(np.std(void_vals))
    p95_void = float(np.percentile(void_vals, 95))

    # 1. Volumetric Error Rate (VER)
    n_overlap = int(np.sum(void_vals >= min_gel))
    ver = float(n_overlap / num_total)

    # 2. Process Window (PW)
    pw = float(min_gel - max_void)

    # 3. Coefficient of Variance (CV)
    cv = float(std_gel / mean_gel) if mean_gel > 1e-6 else 1.0

    # 4. In-part Dose Range (IPDR)
    ipdr = float(max_gel - min_gel)

    # 5. Dose Contrast
    contrast = float(mean_gel / max(1e-6, mean_void))

    # Multi-objective fitness
    fitness = float(3.0 * (1.0 - ver) + 2.0 * pw - 1.0 * cv + 0.5 * min(4.0, contrast))

    return {
        "ver": ver,
        "pw": pw,
        "cv": cv,
        "ipdr": ipdr,
        "contrast": contrast,
        "fitness": fitness,
        "min_gel": min_gel,
        "max_gel": max_gel,
        "mean_gel": mean_gel,
        "std_gel": std_gel,
        "p5_gel": p5_gel,
        "min_void": min_void,
        "max_void": max_void,
        "mean_void": mean_void,
        "std_void": std_void,
        "p95_void": p95_void,
    }

def run_single_optimization_on_target(target_geo, geom_name: str, params: dict, run_dir: str, phase_label: str = "sweep"):
    n_angles = int(params.get("n_angles", 180))
    angles = np.linspace(0, 360 - 360 / n_angles, n_angles)
    proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=CUDA_AVAILABLE)

    method = params.get("method", "OSMO")
    n_iter = int(params.get("n_iter", 15))
    d_h = float(params.get("d_h", 0.85))
    d_l = float(params.get("d_l", 0.60))
    filt = params.get("filter", "ram-lak")
    eps = float(params.get("eps", 0.10))
    p_norm = float(params.get("p_norm", 2.0))
    inhibition = float(params.get("inhibition", 0.0))
    lr = float(params.get("learning_rate", 0.01))

    opt_kwargs = {
        "method": method,
        "n_iter": n_iter,
        "d_h": d_h,
        "d_l": d_l,
        "filter": filt if filt != "None" and filt != "nan" else None,
        "verbose": False,
    }

    if method == "OSMO":
        opt_kwargs["inhibition"] = inhibition
    elif method == "BCLP":
        opt_kwargs["eps"] = eps
        opt_kwargs["p"] = p_norm
        opt_kwargs["learning_rate"] = lr
    elif method == "CAL":
        opt_kwargs["learning_rate"] = lr

    opts = vam.optimize.Options(**opt_kwargs)

    t_start = time.perf_counter()
    sino, recon, err = vam.optimize.optimize(target_geo, proj_geo, opts)
    t_run = time.perf_counter() - t_start

    final_err = float(err[~np.isnan(err)][-1]) if err is not None and len(err) > 0 and np.any(~np.isnan(err)) else 0.0
    metrics = compute_detailed_metrics(target_geo, recon.array)
    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geometry": geom_name,
        "phase": phase_label,
        "method": method,
        "n_angles": n_angles,
        "n_iter": n_iter,
        "d_h": d_h,
        "d_l": d_l,
        "filter": str(filt),
        "eps": eps,
        "p_norm": p_norm,
        "inhibition": inhibition,
        "learning_rate": lr,
        "resolution": params.get("resolution", DEFAULT_RESOLUTION),
        "runtime_s": round(t_run, 3),
        "memory_mb": round(mem_mb, 1),
        "final_loss": round(final_err, 5),
        **metrics,
    }

    log_csv = os.path.join(run_dir, "live_experiment_log.csv")
    log_jsonl = os.path.join(run_dir, "live_experiment_log.jsonl")

    df_single = pd.DataFrame([record])
    if not os.path.exists(log_csv):
        df_single.to_csv(log_csv, index=False)
    else:
        df_single.to_csv(log_csv, mode="a", header=False, index=False)

    with open(log_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record, recon.array, sino

def save_cross_section_comparison(geom_name: str, best_recon, worst_recon, target_geo, output_dir: str):
    z_mid = target_geo.array.shape[2] // 2
    target_slice = target_geo.array[:, :, z_mid]
    best_slice = best_recon[:, :, z_mid] / max(1e-6, np.max(best_recon))
    worst_slice = worst_recon[:, :, z_mid] / max(1e-6, np.max(worst_recon))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(target_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"[{geom_name}] Vial Fitted Target (Z={z_mid})")
    axes[0].axis("off")

    im1 = axes[1].imshow(best_slice.T, cmap="hot", origin="lower", vmin=0, vmax=1)
    axes[1].set_title(f"Best Recon (PW: max)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(worst_slice.T, cmap="hot", origin="lower", vmin=0, vmax=1)
    axes[2].set_title(f"Baseline Recon")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"reconstruction_slices_{geom_name}.png"), bbox_inches="tight")
    plt.close()

def run_sweet_spot_search(target_geo, geom_name: str, run_dir: str, stop_event: threading.Event = None, progress_callback = None):
    """
    Executes the comprehensive 4-phase Sweet-Spot Auto-Optimizer on any target geometry.
    """
    slices_dir = os.path.join(run_dir, "slices")
    charts_dir = os.path.join(run_dir, "charts")
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    filters = ["ram-lak", "hamming", "shepp-logan", "cosine", "None"]
    population = []
    geom_records = []
    geom_recons = {}

    # Phase 1: Quasi-Random Jumps (15 trials)
    for i in range(15):
        if stop_event and stop_event.is_set(): break
        if progress_callback:
            progress_callback(geom_name, f"Phase 1: Jumps ({i+1}/15)", i)

        m = random.choice(["OSMO", "BCLP", "CAL", "FBP"])
        p = {
            "method": m,
            "n_angles": random.choice([90, 180, 240, 360]),
            "n_iter": random.choice([10, 12, 15, 20]),
            "d_h": round(random.uniform(0.75, 0.95), 2),
            "d_l": round(random.uniform(0.40, 0.70), 2),
            "filter": random.choice(filters),
            "eps": round(random.uniform(0.04, 0.20), 2),
            "p_norm": random.choice([1.5, 2.0, 3.0]),
            "inhibition": round(random.uniform(0.0, 0.20), 2),
            "learning_rate": random.choice([0.005, 0.01, 0.02]),
        }
        rec, recon_arr, _ = run_single_optimization_on_target(target_geo, geom_name, p, run_dir, phase_label="phase1_random_jumps")
        geom_records.append(rec)
        geom_recons[rec["timestamp"] + f"_{i}"] = (rec, recon_arr)
        population.append(p)
        print(f"  [{i+1:02d}/15] {m:4s} | Angles={p['n_angles']:3d} | Filter={p['filter']:10s} | PW={rec['pw']:+.3f} | VER={rec['ver']*100:5.2f}%")

    # Phase 2: Genetic Evolution (2 gens x 4 children = 8 trials)
    for gen in range(2):
        if stop_event and stop_event.is_set(): break
        indexed_pop = list(zip(geom_records[-len(population):], population))
        indexed_pop.sort(key=lambda x: x[0]["fitness"], reverse=True)
        elites = [p for _, p in indexed_pop[:4]]

        new_population = []
        for e_idx, parent1 in enumerate(elites):
            if stop_event and stop_event.is_set(): break
            if progress_callback:
                progress_callback(geom_name, f"Phase 2: Genetic Gen {gen+1} ({e_idx+1}/4)", 15 + gen * 4 + e_idx)

            parent2 = random.choice(elites)
            child = {
                "method": parent1["method"],
                "n_angles": random.choice([parent1["n_angles"], parent2["n_angles"]]),
                "n_iter": 15,
                "d_h": round(np.clip(0.5 * (parent1["d_h"] + parent2["d_h"]) + random.uniform(-0.03, 0.03), 0.70, 0.98), 2),
                "d_l": round(np.clip(0.5 * (parent1["d_l"] + parent2["d_l"]) + random.uniform(-0.03, 0.03), 0.35, 0.75), 2),
                "filter": random.choice([parent1["filter"], parent2["filter"]]),
                "eps": round(np.clip(0.5 * (parent1["eps"] + parent2["eps"]) + random.uniform(-0.02, 0.02), 0.02, 0.25), 2),
                "p_norm": random.choice([parent1["p_norm"], parent2["p_norm"]]),
                "inhibition": round(np.clip(0.5 * (parent1["inhibition"] + parent2["inhibition"]) + random.uniform(-0.02, 0.02), 0.0, 0.25), 2),
                "learning_rate": parent1["learning_rate"],
            }
            rec, recon_arr, _ = run_single_optimization_on_target(target_geo, geom_name, child, run_dir, phase_label=f"phase2_genetic_gen{gen+1}")
            geom_records.append(rec)
            geom_recons[rec["timestamp"] + f"_gen{gen}_{e_idx}"] = (rec, recon_arr)
            new_population.append(child)
        population = new_population

    # Phase 3: Greedy Descent (6 steps)
    geom_records.sort(key=lambda x: (x["pw"], -x["ver"]), reverse=True)
    champion_rec = geom_records[0]
    champ_params = {
        "method": champion_rec["method"],
        "n_angles": int(champion_rec["n_angles"]),
        "n_iter": 20,
        "d_h": float(champion_rec["d_h"]),
        "d_l": float(champion_rec["d_l"]),
        "filter": champion_rec["filter"],
        "eps": float(champion_rec["eps"]),
        "p_norm": float(champion_rec["p_norm"]),
        "inhibition": float(champion_rec["inhibition"]),
        "learning_rate": float(champion_rec["learning_rate"]),
    }

    perturbations = [
        {"d_h": round(min(0.98, champ_params["d_h"] + 0.04), 2)},
        {"d_h": round(max(0.70, champ_params["d_h"] - 0.04), 2)},
        {"d_l": round(min(0.75, champ_params["d_l"] + 0.04), 2)},
        {"d_l": round(max(0.35, champ_params["d_l"] - 0.04), 2)},
        {"n_angles": min(360, champ_params["n_angles"] + 60)},
        {"filter": "hamming" if champ_params["filter"] != "hamming" else "ram-lak"},
    ]

    for step_idx, delta in enumerate(perturbations):
        if stop_event and stop_event.is_set(): break
        if progress_callback:
            progress_callback(geom_name, f"Phase 3: Greedy Step ({step_idx+1}/6)", 23 + step_idx)

        test_p = dict(champ_params)
        test_p.update(delta)
        rec, recon_arr, _ = run_single_optimization_on_target(target_geo, geom_name, test_p, run_dir, phase_label="phase3_greedy_step")
        geom_records.append(rec)
        geom_recons[rec["timestamp"] + f"_greedy_{step_idx}"] = (rec, recon_arr)
        if rec["pw"] > champ_params.get("pw", -999):
            champ_params.update(delta)
            champ_params["pw"] = rec["pw"]

    # Phase 4: Dense 2D Interaction Sweeps
    # dh vs dl (5x5 = 25 points)
    sweep_count = 0
    for dh in [0.75, 0.80, 0.85, 0.90, 0.95]:
        for dl in [0.45, 0.50, 0.55, 0.60, 0.65]:
            if stop_event and stop_event.is_set(): break
            sweep_count += 1
            if progress_callback:
                progress_callback(geom_name, f"Phase 4: dh x dl ({sweep_count}/25)", 29 + sweep_count)
            p = {"method": "OSMO", "n_angles": 180, "n_iter": 12, "d_h": dh, "d_l": dl, "filter": "ram-lak"}
            rec, recon_arr, _ = run_single_optimization_on_target(target_geo, geom_name, p, run_dir, phase_label="phase4_sweep_dh_dl")
            geom_records.append(rec)
            geom_recons[rec["timestamp"] + f"_dh{dh}_dl{dl}"] = (rec, recon_arr)

    # Save Cross Sections & Charts
    if len(geom_records) > 0:
        geom_records.sort(key=lambda x: (x["pw"], -x["ver"]), reverse=True)
        best_rec = geom_records[0]
        worst_rec = geom_records[-1]

        b_matches = [v[1] for k, v in geom_recons.items() if v[0]["timestamp"] == best_rec["timestamp"]]
        w_matches = [v[1] for k, v in geom_recons.items() if v[0]["timestamp"] == worst_rec["timestamp"]]
        if b_matches and w_matches:
            save_cross_section_comparison(geom_name, b_matches[0], w_matches[0], target_geo, slices_dir)

    log_csv = os.path.join(run_dir, "live_experiment_log.csv")
    if os.path.exists(log_csv):
        try:
            generate_all_charts(log_csv, charts_dir)
        except Exception as ce:
            print(f"[RunManager] Chart generation notice: {ce}")

    return geom_records[0] if geom_records else None

def execute_custom_model_optimization(stl_path: str, transform_dict: dict, stop_event: threading.Event = None, progress_callback = None):
    """
    Voxelizes custom user-transformed mesh and executes the Sweet-Spot search.
    """
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    slices_dir = os.path.join(run_dir, "slices")
    charts_dir = os.path.join(run_dir, "charts")
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    geom_name = transform_dict.get("name", os.path.splitext(os.path.basename(stl_path))[0])
    if "scale_x" in transform_dict and "scale_y" in transform_dict and "scale_z" in transform_dict:
        scale = [float(transform_dict["scale_x"]), float(transform_dict["scale_y"]), float(transform_dict["scale_z"])]
    else:
        scale = float(transform_dict.get("scale", 1.0))
    rx = float(transform_dict.get("rx", 0.0))
    ry = float(transform_dict.get("ry", 0.0))
    rz = float(transform_dict.get("rz", 0.0))
    tx = float(transform_dict.get("tx", 0.0))
    ty = float(transform_dict.get("ty", 0.0))
    tz = float(transform_dict.get("tz", 0.0))
    res = int(transform_dict.get("resolution", DEFAULT_RESOLUTION))

    status_file = os.path.join(run_dir, "status.json")
    with open(status_file, "w") as f:
        json.dump({"run_id": run_id, "status": "RUNNING", "start_time": time.time(), "geometry": geom_name, "transforms": transform_dict}, f, indent=2)

    target_geo, extents = voxelize_transformed_mesh(stl_path, scale=scale, rx=rx, ry=ry, rz=rz, tx=tx, ty=ty, tz=tz, resolution=res)

    best_rec = run_sweet_spot_search(target_geo, geom_name, run_dir, stop_event, progress_callback)

    log_csv = os.path.join(run_dir, "live_experiment_log.csv")
    final_csv = os.path.join(run_dir, "final_experiment_log.csv")

    if stop_event and stop_event.is_set():
        with open(status_file, "w") as f:
            json.dump({"run_id": run_id, "status": "STOPPED", "stopped_at": time.time()}, f, indent=2)
        if os.path.exists(log_csv):
            import shutil
            shutil.copyfile(log_csv, final_csv)
            generate_all_charts(final_csv, charts_dir)
        return run_id, best_rec

    if os.path.exists(log_csv):
        import shutil
        shutil.copyfile(log_csv, final_csv)
        generate_all_charts(final_csv, charts_dir)

    complete_manifest = {
        "run_id": run_id,
        "status": "COMPLETED",
        "geometry": geom_name,
        "extents_mm": list(extents),
        "transforms": transform_dict,
        "completion_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_configuration": best_rec,
        "charts_directory": charts_dir,
        "slices_directory": slices_dir,
    }

    with open(os.path.join(run_dir, "RUN_COMPLETE.json"), "w") as f:
        json.dump(complete_manifest, f, indent=2)

    with open(status_file, "w") as f:
        json.dump({"run_id": run_id, "status": "COMPLETED", "completed_at": time.time()}, f, indent=2)

    return run_id, best_rec

def execute_investigation(stop_event: threading.Event = None, progress_callback = None):
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    slices_dir = os.path.join(run_dir, "slices")
    charts_dir = os.path.join(run_dir, "charts")
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    status_file = os.path.join(run_dir, "status.json")
    with open(status_file, "w") as f:
        json.dump({"run_id": run_id, "status": "RUNNING", "start_time": time.time(), "geometries": list(STLS.keys())}, f, indent=2)

    all_best_records = {}
    try:
        for g_idx, geom_name in enumerate(STLS.keys()):
            if stop_event and stop_event.is_set(): break
            target_geo = get_cached_geometry(geom_name)
            best_rec = run_sweet_spot_search(target_geo, geom_name, run_dir, stop_event, progress_callback)
            if best_rec:
                all_best_records[geom_name] = best_rec

        log_csv = os.path.join(run_dir, "live_experiment_log.csv")
        final_csv = os.path.join(run_dir, "final_experiment_log.csv")

        if stop_event and stop_event.is_set():
            with open(status_file, "w") as f:
                json.dump({"run_id": run_id, "status": "STOPPED", "stopped_at": time.time()}, f, indent=2)
            if os.path.exists(log_csv):
                import shutil
                shutil.copyfile(log_csv, final_csv)
                generate_all_charts(final_csv, charts_dir)
            return

        if os.path.exists(log_csv):
            import shutil
            shutil.copyfile(log_csv, final_csv)
            generate_all_charts(final_csv, charts_dir)

        complete_manifest = {
            "run_id": run_id,
            "status": "COMPLETED",
            "completion_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_configurations": all_best_records,
            "charts_directory": charts_dir,
            "slices_directory": slices_dir,
        }

        with open(os.path.join(run_dir, "RUN_COMPLETE.json"), "w") as f:
            json.dump(complete_manifest, f, indent=2)

        with open(status_file, "w") as f:
            json.dump({"run_id": run_id, "status": "COMPLETED", "completed_at": time.time()}, f, indent=2)

    except Exception as e:
        with open(status_file, "w") as f:
            json.dump({"run_id": run_id, "status": "CRASHED", "error": str(e), "timestamp": time.time()}, f, indent=2)
        raise

if __name__ == "__main__":
    execute_investigation()
