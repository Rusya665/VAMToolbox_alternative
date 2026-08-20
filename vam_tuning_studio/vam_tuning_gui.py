#!/usr/bin/env python3
"""
OpenCAL / VAM 3D Interactive Slicing & Sweet-Spot Optimization Studio
- Interactive 3D Three.js Glass Vial Viewport (30mm x 60mm physical vial, 25mm x 50mm safe zone).
- Fast Client-side STL parsing with FileReader & STLLoader (handles 500MB+ STLs instantly).
- Full Non-Uniform & Uniform Numerical Scaling (typed numbers in mm and multiplier scale).
- Direct Numerical 3-Axis Euler Rotation and Translation controls.
- On-demand Sweet-Spot Auto-Optimizer on arbitrary oriented models.
- Clean quiet terminal output and real-time GPU acceleration.
"""

import os
import sys
import json
import socket
import logging
import threading
import time
import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory, request, render_template_string
from werkzeug.utils import secure_filename

# Silence noisy Werkzeug HTTP access logs in user terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Ensure local package path is available
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import vamtoolbox as vam

try:
    from run_genetic_investigation import (
        execute_investigation,
        execute_custom_model_optimization,
        voxelize_transformed_mesh,
        get_cached_geometry,
        STLS,
        VIAL_DIAMETER_MM,
        VIAL_HEIGHT_MM,
        MAX_PRINTABLE_DIAMETER_MM,
        MAX_PRINTABLE_HEIGHT_MM,
    )
except ImportError:
    from vam_tuning_studio.run_genetic_investigation import (
        execute_investigation,
        execute_custom_model_optimization,
        voxelize_transformed_mesh,
        get_cached_geometry,
        STLS,
        VIAL_DIAMETER_MM,
        VIAL_HEIGHT_MM,
        MAX_PRINTABLE_DIAMETER_MM,
        MAX_PRINTABLE_HEIGHT_MM,
    )

STUDIO_DIR = os.path.abspath(os.path.dirname(__file__))
RUNS_DIR = os.path.join(STUDIO_DIR, "runs")
UPLOADS_DIR = os.path.join(STUDIO_DIR, "uploads")
STLS_DIR = os.path.join(STUDIO_DIR, "stls")
ROOT_STLS_DIR = os.path.join(ROOT_DIR, "stls")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STLS_DIR, exist_ok=True)

app = Flask(__name__)
# Support up to 1GB for high-poly STL uploads
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

LOCAL_IP = get_local_ip()
PORT = 5050

# ── Background Worker Controller ─────────────────────────────────────────────
class ExplorationController:
    def __init__(self):
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.active_geom = "Idle"
        self.active_phase = "None"
        self.progress_index = 0
        self.total_trials_est = 300
        self.active_run_id = ""
        self.workers_telemetry = {
            "worker_1": {
                "id": 1,
                "name": "GPU Agent #1 (CUDA ASTRA)",
                "status": "Idle",
                "geometry": "Idle",
                "phase": "Ready",
                "params": "None",
                "last_pw": "+0.00",
                "last_ver": "0.0%",
                "trial": 0,
            }
        }

    def progress_hook(self, geom, phase, idx):
        self.active_geom = geom
        self.active_phase = phase
        self.progress_index = idx
        self.workers_telemetry["worker_1"]["status"] = "Optimizing"
        self.workers_telemetry["worker_1"]["geometry"] = geom
        self.workers_telemetry["worker_1"]["phase"] = phase
        self.workers_telemetry["worker_1"]["trial"] = idx + 1

    def _worker_multi_target(self):
        self.is_running = True
        self.stop_event.clear()
        self.workers_telemetry["worker_1"]["status"] = "Active"
        try:
            execute_investigation(stop_event=self.stop_event, progress_callback=self.progress_hook)
        except Exception as e:
            print(f"[Controller] Worker exception: {e}")
            self.workers_telemetry["worker_1"]["status"] = f"Error: {e}"
        finally:
            self.is_running = False
            self.active_geom = "Completed"
            self.active_phase = "Finished"
            self.workers_telemetry["worker_1"]["status"] = "Idle / Finished"

    def _worker_custom_target(self, stl_path, transform_dict):
        self.is_running = True
        self.stop_event.clear()
        self.workers_telemetry["worker_1"]["status"] = "Active"
        self.active_geom = transform_dict.get("name", "Custom Model")
        try:
            run_id, best_rec = execute_custom_model_optimization(
                stl_path,
                transform_dict,
                stop_event=self.stop_event,
                progress_callback=self.progress_hook
            )
            self.active_run_id = run_id
        except Exception as e:
            print(f"[Controller] Custom model optimization exception: {e}")
            self.workers_telemetry["worker_1"]["status"] = f"Error: {e}"
        finally:
            self.is_running = False
            self.active_geom = "Completed"
            self.active_phase = "Finished"
            self.workers_telemetry["worker_1"]["status"] = "Idle / Finished"

    def start_all_run(self):
        if self.is_running:
            self.stop_run()
            time.sleep(0.6)
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_multi_target, daemon=True)
        self.worker_thread.start()
        time.sleep(0.5)
        latest = self.get_latest_run_id()
        self.active_run_id = latest
        return True, "Benchmark exploration started.", latest

    def start_custom_optimization(self, stl_path, transform_dict):
        if self.is_running:
            self.stop_run()
            time.sleep(0.6)
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_custom_target, args=(stl_path, transform_dict), daemon=True)
        self.worker_thread.start()
        time.sleep(0.5)
        latest = self.get_latest_run_id()
        self.active_run_id = latest
        return True, "Sweet-spot search started for model.", latest

    def stop_run(self):
        self.stop_event.set()
        self.is_running = False
        return True, "Stop signal sent to optimizer."

    def get_latest_run_id(self):
        if os.path.exists(RUNS_DIR):
            dirs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
            if dirs:
                return sorted(dirs, reverse=True)[0]
        return ""

CONTROLLER = ExplorationController()

# ── Video Generation Engine ──────────────────────────────────────────────────
def generate_projection_video(run_id: str, geom_name: str, rpm: float = 9.0, duration_sec: float = 60.0):
    run_dir = os.path.join(RUNS_DIR, run_id)
    videos_dir = os.path.join(run_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    log_csv = os.path.join(run_dir, "live_experiment_log.csv")
    status_file = os.path.join(run_dir, "status.json")

    if not os.path.exists(log_csv):
        raise FileNotFoundError(f"Run data for {run_id} not found.")

    df = pd.read_csv(log_csv)
    sub = df[df["geometry"] == geom_name]
    if len(sub) == 0:
        sub = df

    best_row = sub.sort_values(by=["pw", "ver"], ascending=[False, True]).iloc[0]

    # Reconstruct Target Geometry
    transforms = {}
    comp_file = os.path.join(run_dir, "RUN_COMPLETE.json")
    if os.path.exists(comp_file):
        try:
            with open(comp_file, "r") as f:
                st = json.load(f)
                transforms = st.get("transforms", {})
        except Exception:
            pass
    if not transforms and os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                st = json.load(f)
                transforms = st.get("transforms", {})
        except Exception:
            pass

    scale_val = transforms.get("scale", 1.0)
    if "scale_x" in transforms and "scale_y" in transforms and "scale_z" in transforms:
        scale_input = [float(transforms["scale_x"]), float(transforms["scale_y"]), float(transforms["scale_z"])]
    else:
        scale_input = float(scale_val)

    if transforms and "stl_path" in transforms and os.path.exists(transforms["stl_path"]):
        target_geo, _ = voxelize_transformed_mesh(
            transforms["stl_path"],
            scale=scale_input,
            rx=float(transforms.get("rx", 0.0)),
            ry=float(transforms.get("ry", 0.0)),
            rz=float(transforms.get("rz", 0.0)),
            tx=float(transforms.get("tx", 0.0)),
            ty=float(transforms.get("ty", 0.0)),
            tz=float(transforms.get("tz", 0.0)),
            resolution=int(best_row.get("resolution", 75))
        )
    elif geom_name in STLS:
        target_geo = get_cached_geometry(geom_name, resolution=int(best_row.get("resolution", 75)))
    else:
        target_geo = get_cached_geometry("cube", resolution=int(best_row.get("resolution", 75)))

    n_angles = int(best_row.get("n_angles", 180))
    angles = np.linspace(0, 360 - 360 / n_angles, n_angles)
    proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=True)

    method = str(best_row.get("method", "OSMO"))
    filt = str(best_row.get("filter", "ram-lak"))
    if filt == "nan" or filt == "None" or not filt:
        filt = None

    opt_kwargs = {
        "method": method,
        "n_iter": int(best_row.get("n_iter", 15)),
        "d_h": float(best_row.get("d_h", 0.85)),
        "d_l": float(best_row.get("d_l", 0.60)),
        "filter": filt,
        "verbose": False,
    }
    if method == "OSMO":
        opt_kwargs["inhibition"] = float(best_row.get("inhibition", 0.0))
    elif method == "BCLP":
        opt_kwargs["eps"] = float(best_row.get("eps", 0.10))
        opt_kwargs["p"] = float(best_row.get("p_norm", 2.0))
        opt_kwargs["learning_rate"] = float(best_row.get("learning_rate", 0.01))
    elif method == "CAL":
        opt_kwargs["learning_rate"] = float(best_row.get("learning_rate", 0.01))

    opts = vam.optimize.Options(**opt_kwargs)
    sino, recon, err = vam.optimize.optimize(target_geo, proj_geo, opts)

    # ── Tomo-Native High-Fidelity Video Maker Engine ──
    proj_px_w, proj_px_h = 1920, 1080
    proj_width_mm = 108.0   # Standard VAM projector FOV width in mm (matching Tomo)
    res_vox = int(best_row.get("resolution", 75))

    # Physical voxel pitch in mm per voxel
    voxel_pitch_mm = MAX_PRINTABLE_DIAMETER_MM / max(res_vox, 1)   # 25.0mm / 75 = 0.333 mm/vox

    # 1. Projector px-per-voxel true scale (matching Tomo's physical optics)
    arr = sino.array
    n_r, n_z = arr.shape[0], arr.shape[2]
    true_scale = (voxel_pitch_mm * proj_px_w) / max(proj_width_mm, 1e-3)
    fit_scale = min(proj_px_h / max(n_z, 1), proj_px_w / max(n_r, 1)) * 0.95
    true_scale = min(true_scale, fit_scale)

    # 2. Vertical offset in projector pixels
    v_offset_mm = float(transforms.get("tz", 0.0))
    mm_per_px = proj_width_mm / max(proj_px_w, 1)
    off_px = -(v_offset_mm / max(mm_per_px, 1e-9))
    s_v = n_z * true_scale
    max_off = max(0.0, (proj_px_h - s_v) / 2.0 - 1)
    v_offset_px = int(round(max(-max_off, min(max_off, off_px))))

    # 3. Process sinogram through ImageConfig & ImageSeq
    iconfig = vam.imagesequence.ImageConfig(
        image_dims=(int(proj_px_w), int(proj_px_h)),
        rotated_angle=0,
        size_scale=true_scale,
        v_offset=v_offset_px,
        normalization_percentile=99.9,
        intensity_scale=1.0,
    )
    image_seq = vam.imagesequence.ImageSeq(
        image_config=iconfig, sinogram=sino
    )

    n_images = len(image_seq.images)
    fps = 30.0
    total_frames = max(1, int(round(fps * duration_sec)))
    deg_per_frame = rpm * 6.0 / fps
    W = iconfig.N_u
    H = iconfig.N_v

    video_filename = f"projection_{geom_name}_{rpm:.1f}rpm_{int(duration_sec)}s.mp4"
    video_path = os.path.join(videos_dir, video_filename)

    def _get_frame_rgb(k):
        angle = (k * deg_per_frame) % 360.0
        idx = int(angle / 360.0 * n_images) % n_images
        # np.flipud to match DLP projector bottom-origin optics
        g = np.flipud(image_seq.images[idx])
        return np.repeat(g[:, :, None], 3, axis=2)

    # Fast stream-copy loop if full rotations match
    fpr = (60.0 * fps / abs(rpm)) if abs(rpm) > 1e-9 else 0.0
    n_unique = int(round(fpr))
    n_loops = int(round(total_frames / n_unique)) if n_unique > 0 else 0
    fast = (n_unique >= 2 and n_loops >= 2 and abs(fpr - n_unique) < 1e-3)

    try:
        import imageio_ffmpeg
        import subprocess

        ff_codec = "libx264"
        if fast:
            seg = video_path + ".seg.mp4"
            writer = imageio_ffmpeg.write_frames(
                seg, (W, H), fps=fps, codec=ff_codec,
                pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2, quality=6,
            )
            writer.send(None)
            for k in range(n_unique):
                writer.send(np.ascontiguousarray(_get_frame_rgb(k), dtype=np.uint8).tobytes())
            writer.close()

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [exe, "-y", "-stream_loop", str(n_loops - 1), "-i", seg,
                 "-c", "copy", "-fflags", "+genpts", video_path],
                check=True, capture_output=True)
            try:
                if os.path.exists(seg):
                    os.remove(seg)
            except Exception:
                pass
            return video_filename, os.path.getsize(video_path)

        writer = imageio_ffmpeg.write_frames(
            video_path, (W, H), fps=fps, codec=ff_codec,
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2, quality=6,
        )
        writer.send(None)
        for k in range(total_frames):
            writer.send(np.ascontiguousarray(_get_frame_rgb(k), dtype=np.uint8).tobytes())
        writer.close()
        return video_filename, os.path.getsize(video_path)

    except Exception as e:
        print(f"[VideoMaker] imageio-ffmpeg failed ({e}); falling back to OpenCV")
        vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (W, H))
        for k in range(total_frames):
            f_bgr = cv2.cvtColor(_get_frame_rgb(k)[:, :, 0], cv2.COLOR_GRAY2BGR)
            vw.write(f_bgr)
        vw.release()
        return video_filename, os.path.getsize(video_path)

# ── Complete HTML / 3D Three.js Slicer Interface ──────────────────────────────
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>OpenCAL VAM 3D Slicer & Parameter Tuning Studio</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <!-- Chart.js & Three.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>

    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: rgba(17, 24, 39, 0.85);
            --border-color: rgba(75, 85, 99, 0.35);
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-purple: #8b5cf6;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                              radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 24px;
            background: rgba(11, 15, 25, 0.95);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
            flex-wrap: wrap;
            gap: 10px;
        }

        .logo-badge {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            color: #fff;
            font-weight: 800;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 13px;
            box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            border: none;
            transition: all 0.2s ease;
        }

        .btn-start {
            background: linear-gradient(135deg, #10b981, #059669);
            color: #fff;
            box-shadow: 0 0 14px rgba(16, 185, 129, 0.4);
        }

        .btn-start:hover { opacity: 0.9; transform: scale(1.02); }
        .btn-start:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }

        .btn-stop {
            background: linear-gradient(135deg, #f43f5e, #e11d48);
            color: #fff;
            box-shadow: 0 0 14px rgba(244, 63, 94, 0.4);
        }

        .btn-stop:hover { opacity: 0.9; transform: scale(1.02); }
        .btn-stop:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }

        .btn-action {
            background: linear-gradient(135deg, var(--accent-cyan), #0284c7);
            color: #fff;
            box-shadow: 0 0 14px rgba(6, 182, 212, 0.4);
        }

        .btn-action:hover { opacity: 0.9; transform: scale(1.02); }

        .btn-secondary {
            background: rgba(31, 41, 55, 0.8);
            color: #fff;
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover { background: rgba(55, 65, 81, 0.8); }

        .nav-tabs {
            display: flex;
            gap: 6px;
            padding: 8px 24px;
            background: rgba(17, 24, 39, 0.7);
            border-bottom: 1px solid var(--border-color);
            overflow-x: auto;
        }

        .tab-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            flex-shrink: 0;
        }

        .tab-btn.active {
            color: #fff;
            background: rgba(6, 182, 212, 0.15);
            border-color: rgba(6, 182, 212, 0.4);
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.2);
        }

        main { padding: 18px 24px; flex: 1; display: flex; flex-direction: column; gap: 18px; }
        .tab-pane { display: none; flex-direction: column; gap: 18px; }
        .tab-pane.active { display: flex; }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .card-header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; }
        .card-title { font-size: 15px; font-weight: 700; }

        /* 3D Slicer Layout */
        .slicer-3d-grid {
            display: grid;
            grid-template-columns: 1.8fr 1.2fr;
            gap: 18px;
            min-height: 680px;
        }

        .viewport-box {
            position: relative;
            background: radial-gradient(circle at center, #111827 0%, #030712 100%);
            border-radius: 14px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            min-height: 640px;
        }

        #threeCanvas { width: 100%; height: 100%; display: block; }

        .dropzone-overlay {
            position: absolute;
            top: 14px;
            left: 14px;
            background: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(8px);
            border: 1px dashed var(--accent-cyan);
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 11px;
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            z-index: 10;
        }

        .vial-info-pill {
            position: absolute;
            bottom: 14px;
            left: 14px;
            background: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 11px;
            color: var(--text-muted);
            line-height: 1.4;
            z-index: 10;
            max-width: 90%;
        }

        .controls-sidebar {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 720px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .ctrl-group {
            background: rgba(31, 41, 55, 0.5);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .ctrl-title { font-size: 12px; font-weight: 700; color: var(--accent-cyan); text-transform: uppercase; display: flex; justify-content: space-between; align-items: center; }

        .input-row-3 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
        }

        .num-field {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }

        .num-field label {
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        input[type=number], input[type=text], select {
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            outline: none;
            width: 100%;
        }

        input[type=number]:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 8px rgba(6, 182, 212, 0.3);
        }

        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }
        .kpi-card {
            background: var(--bg-card);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 18px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .kpi-label { font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; }
        .kpi-value { font-size: 24px; font-weight: 800; letter-spacing: -0.5px; }
        .kpi-sub { font-size: 11px; color: var(--text-muted); }

        .image-gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
        .gallery-card { background: rgba(11, 15, 25, 0.6); border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; }
        .gallery-card img { width: 100%; height: auto; display: block; background: #000; }
        .gallery-caption { padding: 10px 14px; font-size: 12px; font-weight: 600; border-top: 1px solid var(--border-color); }

        .heatmap-table-wrap { overflow-x: auto; border-radius: 10px; border: 1px solid var(--border-color); }
        table.heatmap-table { width: 100%; border-collapse: collapse; text-align: center; font-size: 11px; }
        table.heatmap-table th, table.heatmap-table td { padding: 9px 12px; border: 1px solid rgba(75, 85, 99, 0.25); }
        table.heatmap-table th { background: rgba(31, 41, 55, 0.6); color: var(--text-muted); font-weight: 600; }
        .cell-value { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; }

        .log-table-wrap { max-height: 420px; overflow-y: auto; border-radius: 10px; border: 1px solid var(--border-color); }
        table.log-table { width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }
        table.log-table th, table.log-table td { padding: 9px 12px; border-bottom: 1px solid rgba(75, 85, 99, 0.2); }
        table.log-table th { background: rgba(17, 24, 39, 0.9); position: sticky; top: 0; z-index: 10; color: var(--text-muted); }

        .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
        .badge-bclp { background: rgba(139, 92, 246, 0.2); color: var(--accent-purple); border: 1px solid rgba(139, 92, 246, 0.4); }
        .badge-osmo { background: rgba(6, 182, 212, 0.2); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.4); }
        .badge-cal { background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-fbp { background: rgba(156, 163, 175, 0.2); color: var(--text-muted); border: 1px solid rgba(156, 163, 175, 0.4); }

        .video-container { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; background: #000; border-radius: 12px; padding: 16px; border: 1px solid var(--border-color); }
        video { max-width: 100%; max-height: 480px; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.8); }

        /* Modal Dialog & Tomo Settings Inspector */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(3, 7, 18, 0.88);
            backdrop-filter: blur(10px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 999;
            padding: 20px;
        }
        .modal-overlay.active { display: flex; }
        .modal-box {
            background: #0f172a;
            border: 1px solid var(--accent-cyan);
            border-radius: 16px;
            max-width: 920px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 18px;
            box-shadow: 0 0 50px rgba(6, 182, 212, 0.3);
            animation: modalFadeIn 0.2s ease-out;
        }
        @keyframes modalFadeIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1.0); }
        }
        .recipe-card-interactive {
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
        }
        .recipe-card-interactive:hover {
            transform: translateY(-2px);
            border-color: var(--accent-cyan) !important;
            box-shadow: 0 0 18px rgba(6, 182, 212, 0.3);
        }
        .settings-grid-3 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
        }
        .setting-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            background: rgba(17, 24, 39, 0.7);
            border-radius: 6px;
            font-size: 11px;
        }
        .setting-item .k { color: var(--text-muted); font-weight: 500; }
        .setting-item .v { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #fff; }

        @media (max-width: 960px) {
            .slicer-3d-grid { grid-template-columns: 1fr !important; }
            .grid-2col { grid-template-columns: 1fr !important; }
        }
    </style>
</head>
<body>

    <header>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div class="logo-badge">OPENCAL 3D</div>
            <div>
                <h1 style="font-size: 16px; font-weight: 700;">VAM 3D Slicer & Sweet-Spot Studio</h1>
                <p style="font-size: 11px; color: var(--text-muted);">Vial Placer (30x60mm) | Real-Time Optimizer & 9 RPM Film Master</p>
            </div>
        </div>

        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
            <button id="btnStartBenchmark" class="btn btn-start" onclick="triggerBenchmarkAll()">▶ Benchmark 4 Models</button>
            <button id="btnStop" class="btn btn-stop" onclick="triggerStopRun()">⏹ Stop</button>
            <div style="width: 1px; height: 24px; background: var(--border-color); margin: 0 4px;"></div>
            <select id="runSelector" onchange="onRunChanged()" style="font-weight: 600; max-width: 240px;"></select>
            <div id="runStatusBadge" class="badge" style="padding: 6px 10px; font-size: 11px;">IDLE</div>
        </div>
    </header>

    <div class="nav-tabs">
        <button class="tab-btn active" data-tab="slicer3d" onclick="switchTab('slicer3d')">🧊 3D Slicer & Vial Placer</button>
        <button class="tab-btn" data-tab="dashboard" onclick="switchTab('dashboard')">📊 Live Dashboard</button>
        <button class="tab-btn" data-tab="heatmaps" onclick="switchTab('heatmaps')">2D Heatmaps</button>
        <button class="tab-btn" data-tab="pareto" onclick="switchTab('pareto')">Pareto Charts</button>
        <button class="tab-btn" data-tab="slices" onclick="switchTab('slices')">Slices</button>
        <button class="tab-btn" data-tab="videos" onclick="switchTab('videos')">🎬 9 RPM Film & Slicer Export</button>
        <button class="tab-btn" data-tab="logs" onclick="switchTab('logs')">Full Logs</button>
    </div>

    <main>
        <!-- TAB 1: 3D SLICER & VIAL PLACER -->
        <div id="tab-slicer3d" class="tab-pane active">
            <div class="slicer-3d-grid">
                <!-- 3D Viewport -->
                <div class="viewport-box" id="viewportContainer">
                    <div class="dropzone-overlay" onclick="document.getElementById('stlFileInput').click()">
                        <span id="uploadOverlayText">📥 <b>Drop STL Here</b> or Click to Upload</span>
                        <input type="file" id="stlFileInput" accept=".stl" style="display: none;" onchange="handleFileUpload(event)">
                    </div>

                    <div class="vial-info-pill">
                        <div><b>Glass Resin Vial:</b> 30.0 mm dia &times; 60.0 mm height (Clear Cylinder)</div>
                        <div><b>Safe Printable Zone:</b> 25.0 mm dia &times; 50.0 mm height (Green Wireframe)</div>
                        <div id="meshBoundsText" style="color: var(--accent-cyan); margin-top: 2px;">Loading model ...</div>
                    </div>

                    <canvas id="threeCanvas"></canvas>
                </div>

                <!-- Transform & Slicer Drawer -->
                <div class="controls-sidebar">
                    <!-- Quick Load Benchmarks -->
                    <div class="ctrl-group">
                        <div class="ctrl-title">Load Sample STL</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                            <button class="btn btn-secondary" onclick="loadSampleStl('benchy')">⛵ Benchy</button>
                            <button class="btn btn-secondary" onclick="loadSampleStl('cube')">🧊 Cube</button>
                            <button class="btn btn-secondary" onclick="loadSampleStl('sphere')">🔮 Sphere</button>
                            <button class="btn btn-secondary" onclick="loadSampleStl('captiveRing')">💍 Captive Ring</button>
                        </div>
                    </div>

                    <!-- Dimension & Scaling Controls (Typed Numbers) -->
                    <div class="ctrl-group">
                        <div class="ctrl-title">
                            <span>1. Dimensions & Scaling</span>
                            <button class="btn btn-secondary" style="padding: 3px 8px; font-size: 10px;" onclick="autoFitToSafeZone()">⚡ Auto-Fit Green Zone (25mm)</button>
                        </div>

                        <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; margin-bottom: 2px;">
                            <input type="checkbox" id="uniformScaleCheck" checked onchange="toggleUniformScale()">
                            <label for="uniformScaleCheck" style="color: #fff; font-weight: 600;">Lock Uniform Scaling (XYZ)</label>
                        </div>

                        <!-- Physical Dimensions in mm -->
                        <div class="input-row-3">
                            <div class="num-field">
                                <label>Size X (mm)</label>
                                <input type="number" id="dimXInput" step="0.1" value="25.0" onchange="onDimensionChanged('x')">
                            </div>
                            <div class="num-field">
                                <label>Size Y (mm)</label>
                                <input type="number" id="dimYInput" step="0.1" value="12.9" onchange="onDimensionChanged('y')">
                            </div>
                            <div class="num-field">
                                <label>Size Z (mm)</label>
                                <input type="number" id="dimZInput" step="0.1" value="20.0" onchange="onDimensionChanged('z')">
                            </div>
                        </div>

                        <!-- Scale Multipliers -->
                        <div class="input-row-3">
                            <div class="num-field">
                                <label>Scale X</label>
                                <input type="number" id="scaleXInput" step="0.01" value="0.417" onchange="onScaleMultiplierChanged('x')">
                            </div>
                            <div class="num-field">
                                <label>Scale Y</label>
                                <input type="number" id="scaleYInput" step="0.01" value="0.417" onchange="onScaleMultiplierChanged('y')">
                            </div>
                            <div class="num-field">
                                <label>Scale Z</label>
                                <input type="number" id="scaleZInput" step="0.01" value="0.417" onchange="onScaleMultiplierChanged('z')">
                            </div>
                        </div>
                    </div>

                    <!-- Rotation Controls & Tilting Presets (Typed Numbers) -->
                    <div class="ctrl-group">
                        <div class="ctrl-title">
                            <span>2. Orientation & Tilt</span>
                            <span style="font-size: 10px; color: var(--text-muted);">Degrees (°)</span>
                        </div>
                        <div style="display: flex; gap: 4px; margin-bottom: 4px; flex-wrap: wrap;">
                            <button class="btn btn-secondary" style="flex: 1; padding: 4px 6px; font-size: 10px;" onclick="setTilt(0, 0, 0)">Standing 0°</button>
                            <button class="btn btn-secondary" style="flex: 1; padding: 4px 6px; font-size: 10px;" onclick="setTilt(35, 0, 0)">Tilt 35° (Benchy)</button>
                            <button class="btn btn-secondary" style="flex: 1; padding: 4px 6px; font-size: 10px;" onclick="setTilt(45, 0, 0)">Tilt 45°</button>
                            <button class="btn btn-secondary" style="flex: 1; padding: 4px 6px; font-size: 10px;" onclick="setTilt(0, 0, 0)">Reset</button>
                        </div>
                        <div class="input-row-3">
                            <div class="num-field">
                                <label>Pitch X (°)</label>
                                <input type="number" id="rotXInput" step="1" value="35" onchange="onTransformChanged()">
                            </div>
                            <div class="num-field">
                                <label>Roll Y (°)</label>
                                <input type="number" id="rotYInput" step="1" value="0" onchange="onTransformChanged()">
                            </div>
                            <div class="num-field">
                                <label>Yaw Z (°)</label>
                                <input type="number" id="rotZInput" step="1" value="0" onchange="onTransformChanged()">
                            </div>
                        </div>
                    </div>

                    <!-- Position Controls (Typed Numbers) -->
                    <div class="ctrl-group">
                        <div class="ctrl-title">
                            <span>3. Translation Offsets</span>
                            <button class="btn btn-secondary" style="padding: 3px 8px; font-size: 10px;" onclick="autoCenterAoR()">Center AoR (0,0)</button>
                        </div>
                        <div class="input-row-3">
                            <div class="num-field">
                                <label>X Offset (mm)</label>
                                <input type="number" id="posXInput" step="0.5" value="0.0" onchange="onTransformChanged()">
                            </div>
                            <div class="num-field">
                                <label>Y Offset (mm)</label>
                                <input type="number" id="posYInput" step="0.5" value="0.0" onchange="onTransformChanged()">
                            </div>
                            <div class="num-field">
                                <label>Z Offset (mm)</label>
                                <input type="number" id="posZInput" step="0.5" value="0.0" onchange="onTransformChanged()">
                            </div>
                        </div>
                    </div>

                    <!-- Primary Action Button -->
                    <button id="btnFindSweetSpot" class="btn btn-action" style="padding: 14px; font-size: 13px;" onclick="triggerOptimizeCustomModel()">
                        ⚡ Find Slicing Sweet Spot (Auto-Tune)
                    </button>
                    <div id="slicerStatusMsg" style="font-size: 11px; color: var(--accent-cyan); text-align: center; min-height: 16px;"></div>
                </div>
            </div>
        </div>

        <!-- TAB 2: LIVE DASHBOARD -->
        <div id="tab-dashboard" class="tab-pane">
            <div class="kpi-grid">
                <div class="kpi-card">
                    <span class="kpi-label">Trials Completed In Run</span>
                    <span class="kpi-value" id="kpiTotalRuns" style="color: var(--accent-cyan);">0</span>
                    <span class="kpi-sub" id="kpiRunId">No run</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Best Process Window</span>
                    <span class="kpi-value" id="kpiBestPW" style="color: var(--accent-emerald);">+0.00</span>
                    <span class="kpi-sub" id="kpiBestPWGeom">Across models</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Lowest Volumetric Error</span>
                    <span class="kpi-value" id="kpiLowestVER" style="color: var(--accent-purple);">0.0%</span>
                    <span class="kpi-sub" id="kpiLowestVERGeom">Over-cured void</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Active Target Geometry</span>
                    <span class="kpi-value" id="kpiActiveGeom" style="color: var(--accent-amber); font-size: 18px;">Idle</span>
                    <span class="kpi-sub" id="kpiActivePhase">Ready</span>
                </div>
            </div>

            <div class="grid-2col" style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px;">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Live Process Window & Error Trajectory</span>
                    </div>
                    <div style="height: 260px; position: relative;">
                        <canvas id="liveMetricChart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Optimal Recipe by Model</span>
                    </div>
                    <div id="bestConfigsContainer" style="display: flex; flex-direction: column; gap: 10px; font-size: 12px;"></div>
                </div>
            </div>
        </div>

        <!-- TAB 3: HEATMAPS -->
        <div id="tab-heatmaps" class="tab-pane">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Interactive 2D Sensitivity Heatmaps</span>
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <select id="heatmapGeomSelect" onchange="renderHeatmap()"></select>
                        <select id="heatmapAxisSelect" onchange="renderHeatmap()">
                            <option value="dh_vs_dl">d_h (Gel Floor) vs d_l (Void Ceiling)</option>
                            <option value="angles_vs_filter">Angular Sampling vs Filter Type</option>
                            <option value="eps_vs_p">BCLP: Relaxation (eps) vs Lp Norm (p)</option>
                        </select>
                    </div>
                </div>
                <div class="heatmap-table-wrap" id="heatmapTableContainer"></div>
            </div>
        </div>

        <!-- TAB 4: PARETO -->
        <div id="tab-pareto" class="tab-pane">
            <div class="image-gallery" id="chartsGallery"></div>
        </div>

        <!-- TAB 5: SLICES -->
        <div id="tab-slices" class="tab-pane">
            <div class="image-gallery" id="slicesGallery"></div>
        </div>

        <!-- TAB 6: VIDEOS (9 RPM) -->
        <div id="tab-videos" class="tab-pane">
            <div class="card">
                <div class="card-header">
                    <div>
                        <span class="card-title">🎬 9 RPM VAM Projection Film & Slicer Export</span>
                        <p style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Generates calibrated forward projection sinograms for direct printer deployment.</p>
                    </div>
                </div>

                <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap; background: rgba(31, 41, 55, 0.4); padding: 14px; border-radius: 10px; border: 1px solid var(--border-color);">
                    <div style="display: flex; flex-direction: column; gap: 4px;">
                        <label style="font-size: 11px; color: var(--text-muted); font-weight: 600;">GEOMETRY</label>
                        <select id="videoGeomSelect"></select>
                    </div>

                    <div style="display: flex; flex-direction: column; gap: 4px;">
                        <label style="font-size: 11px; color: var(--text-muted); font-weight: 600;">ROTATION SPEED</label>
                        <select id="videoRpmSelect">
                            <option value="9.0">9.0 RPM (Standard VAM)</option>
                            <option value="6.0">6.0 RPM (Slow Curing)</option>
                            <option value="12.0">12.0 RPM (High Speed)</option>
                        </select>
                    </div>

                    <div style="display: flex; flex-direction: column; gap: 4px;">
                        <label style="font-size: 11px; color: var(--text-muted); font-weight: 600;">DURATION</label>
                        <select id="videoDurationSelect">
                            <option value="60">1 Minute (9 Full Cycles, 60s)</option>
                            <option value="6.67">1 Full Revolution (1 Cycle, 6.7s)</option>
                        </select>
                    </div>

                    <div style="display: flex; flex-direction: column; gap: 4px; justify-content: flex-end; padding-top: 18px;">
                        <button id="btnGenVideo" class="btn btn-action" onclick="triggerGenerateVideo()">🎬 Generate Film</button>
                    </div>

                    <span id="videoStatusMsg" style="font-size: 12px; color: var(--accent-cyan); font-weight: 600; margin-left: 10px;"></span>
                </div>

                <div class="video-container" id="videoPlayerBox" style="display: none;">
                    <video id="videoPlayer" controls autoplay loop playsinline></video>
                    <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                        <a id="videoDownloadBtn" class="btn btn-start" href="#" download>⬇ Download MP4 Film</a>
                        <a id="manifestDownloadBtn" class="btn btn-secondary" href="#" download>⬇ Download Slicer Manifest (.json)</a>
                        <span id="videoMetaText" style="font-size: 12px; color: var(--text-muted);"></span>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 7: LOGS -->
        <div id="tab-logs" class="tab-pane">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Trial Evaluation Log</span>
                    <input type="text" id="logSearchInput" placeholder="Filter trials..." onkeyup="filterLogTable()">
                </div>
                <div class="log-table-wrap">
                    <table class="log-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Geometry</th>
                                <th>Phase</th>
                                <th>Method</th>
                                <th>Angles</th>
                                <th>Filter</th>
                                <th>d_h</th>
                                <th>d_l</th>
                                <th>PW</th>
                                <th>VER (%)</th>
                                <th>Contrast</th>
                                <th>Time (s)</th>
                            </tr>
                        </thead>
                        <tbody id="logTableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TOMO COMPLETE SETTINGS & VIDEO INSPECTOR MODAL -->
        <div id="recipeModalOverlay" class="modal-overlay" onclick="closeRecipeModal(event)">
            <div class="modal-box" onclick="event.stopPropagation()">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid var(--border-color); padding-bottom: 14px;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span class="logo-badge" id="modalGeomBadge">OSMO</span>
                            <h2 style="font-size: 18px; font-weight: 800; color: #fff;" id="modalTitle">Optimal Slicing Recipe</h2>
                        </div>
                        <p style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Complete Tomo / OpenCAL Machine Configuration & 1080p Film Generator</p>
                    </div>
                    <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 14px;" onclick="document.getElementById('recipeModalOverlay').classList.remove('active')">✕</button>
                </div>

                <!-- 3 Setting Sections Grid -->
                <div style="display: flex; flex-direction: column; gap: 14px;">
                    <!-- 1. Tomo Optimizer Core Parameters -->
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <span style="font-size: 11px; font-weight: 700; color: var(--accent-cyan); text-transform: uppercase;">1. Mathematical Optimizer Settings (Tomo Core)</span>
                        <div class="settings-grid-3" id="modalOptParams"></div>
                    </div>

                    <!-- 2. Physical Optical & Resin Parameters -->
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <span style="font-size: 11px; font-weight: 700; color: var(--accent-purple); text-transform: uppercase;">2. Optical & Physical Machine Setup (Tomo Slicer)</span>
                        <div class="settings-grid-3" id="modalPhysParams"></div>
                    </div>

                    <!-- 3. Slicing Quality & Convergence Telemetry -->
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <span style="font-size: 11px; font-weight: 700; color: var(--accent-emerald); text-transform: uppercase;">3. Quality Metrics & Convergence Performance</span>
                        <div class="settings-grid-3" id="modalQualityParams"></div>
                    </div>
                </div>

                <!-- 4. Direct 1080p Film Video Generation in Modal -->
                <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid var(--border-color); border-radius: 12px; padding: 14px; display: flex; flex-direction: column; gap: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <span style="font-size: 12px; font-weight: 700; color: #fff;">🎬 Generate 1080p Projection Video from this Recipe</span>
                            <p style="font-size: 10px; color: var(--text-muted);">Encodes calibrated 360° rotating projection sinogram for your projector.</p>
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                            <select id="modalVideoRpm" style="max-width: 140px; font-size: 11px;">
                                <option value="9.0">9.0 RPM (Standard)</option>
                                <option value="6.0">6.0 RPM (Slow)</option>
                                <option value="12.0">12.0 RPM (Fast)</option>
                            </select>
                            <select id="modalVideoDur" style="max-width: 150px; font-size: 11px;">
                                <option value="6.67">1 Revolution (6.7s Loop)</option>
                                <option value="60">1 Minute (60s, 9 Cycles)</option>
                            </select>
                            <button id="btnModalGenVideo" class="btn btn-start" style="padding: 6px 12px;" onclick="generateVideoFromModal()">🎬 Generate & Play Film</button>
                        </div>
                    </div>

                    <div id="modalVideoStatus" style="font-size: 11px; color: var(--accent-cyan); font-weight: 600; min-height: 14px;"></div>

                    <div id="modalVideoPlayerBox" style="display: none; flex-direction: column; align-items: center; gap: 10px; background: #000; padding: 12px; border-radius: 8px; border: 1px solid var(--border-color);">
                        <video id="modalVideoPlayer" controls autoplay loop playsinline style="max-height: 320px; width: 100%; border-radius: 6px;"></video>
                        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                            <a id="modalVideoDlBtn" class="btn btn-start" href="#" download>⬇ Download MP4 Video (1080p)</a>
                            <a id="modalManifestDlBtn" class="btn btn-secondary" href="#" download>⬇ Download Slicer JSON Profile</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Three.js 3D Viewport Script -->
    <script>
        let scene, camera, renderer, controls;
        let currentMesh = null;
        let originalGeometry = null;
        let activeStlPath = "stls/benchy.stl";
        let activeModelName = "benchy";

        function init3DViewport() {
            const container = document.getElementById('viewportContainer');
            const canvas = document.getElementById('threeCanvas');
            const width = container.clientWidth || 800;
            const height = container.clientHeight || 640;

            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            camera.position.set(45, 35, 55);

            renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
            renderer.setSize(width, height);
            renderer.setPixelRatio(window.devicePixelRatio);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(0, 0, 0);

            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);

            const dirLight1 = new THREE.DirectionalLight(0x06b6d4, 0.9);
            dirLight1.position.set(35, 55, 45);
            scene.add(dirLight1);

            const dirLight2 = new THREE.DirectionalLight(0x8b5cf6, 0.6);
            dirLight2.position.set(-35, -20, -35);
            scene.add(dirLight2);

            // Base Grid
            const gridHelper = new THREE.GridHelper(60, 20, 0x06b6d4, 0x374151);
            gridHelper.position.y = -30;
            scene.add(gridHelper);

            // 1. Outer Glass Vial Cylinder (30mm dia x 60mm height, radius 15mm)
            const vialGeo = new THREE.CylinderGeometry(15, 15, 60, 36, 1, true);
            const vialMat = new THREE.MeshPhysicalMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.16,
                roughness: 0.1,
                transmission: 0.9,
                thickness: 1.2,
                side: THREE.DoubleSide,
            });
            const vialMesh = new THREE.Mesh(vialGeo, vialMat);
            scene.add(vialMesh);

            const vialEdges = new THREE.EdgesGeometry(vialGeo);
            const vialLine = new THREE.LineSegments(vialEdges, new THREE.LineBasicMaterial({ color: 0x6b7280, transparent: true, opacity: 0.4 }));
            scene.add(vialLine);

            // 2. Inner Safe Printable Zone (25mm dia x 50mm height, radius 12.5mm, Green Cylinder)
            const safeGeo = new THREE.CylinderGeometry(12.5, 12.5, 50, 24, 1, true);
            const safeEdges = new THREE.EdgesGeometry(safeGeo);
            const safeLine = new THREE.LineSegments(safeEdges, new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.55 }));
            scene.add(safeLine);

            // 3. Center Axis of Rotation (AoR) line
            const aorGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -32, 0), new THREE.Vector3(0, 32, 0)]);
            const aorLine = new THREE.Line(aorGeo, new THREE.LineBasicMaterial({ color: 0x06b6d4, linewidth: 2 }));
            scene.add(aorLine);

            window.addEventListener('resize', onWindowResize);
            setupDragAndDrop();
            loadSampleStl('benchy');
            animate();
        }

        function onWindowResize() {
            const container = document.getElementById('viewportContainer');
            if (!container || !renderer || !camera) return;
            const width = container.clientWidth || 800;
            const height = container.clientHeight || 640;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }

        function animate() {
            requestAnimationFrame(animate);
            if (controls) controls.update();
            if (renderer && scene && camera) renderer.render(scene, camera);
        }

        function setupDragAndDrop() {
            const dropzone = document.getElementById('viewportContainer');
            dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = 'var(--accent-cyan)'; });
            dropzone.addEventListener('dragleave', (e) => { e.preventDefault(); dropzone.style.borderColor = 'var(--border-color)'; });
            dropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropzone.style.borderColor = 'var(--border-color)';
                if (e.dataTransfer.files.length > 0) {
                    const file = e.dataTransfer.files[0];
                    if (file.name.toLowerCase().endsWith('.stl')) {
                        handleLocalStlFile(file);
                    }
                }
            });
        }

        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) handleLocalStlFile(file);
        }

        // Instant Client-Side Parsing via FileReader + STLLoader.parse (Works on 500MB+ STLs instantly!)
        function handleLocalStlFile(file) {
            const overlayText = document.getElementById('uploadOverlayText');
            const sizeMb = (file.size / 1024 / 1024).toFixed(1);
            overlayText.innerHTML = `⏳ <b>Parsing ${file.name} (${sizeMb} MB) locally ...</b>`;

            activeModelName = file.name.replace(/\.[^/.]+$/, "");

            const reader = new FileReader();
            reader.onload = function (e) {
                try {
                    const loader = new THREE.STLLoader();
                    const geometry = loader.parse(e.target.result);
                    setupParsedGeometry(geometry, activeModelName);
                    overlayText.innerHTML = `✅ <b>${file.name} Loaded</b> (${sizeMb} MB)`;
                } catch (err) {
                    overlayText.innerHTML = `❌ Error parsing STL: ${err.message}`;
                }
            };
            reader.readAsArrayBuffer(file);

            // Background Upload for Optimizer
            uploadStlToServer(file);
        }

        async function uploadStlToServer(file) {
            const msg = document.getElementById('slicerStatusMsg');
            msg.innerText = `Uploading ${file.name} in background to optimizer ...`;
            const formData = new FormData();
            formData.append('stl', file);
            try {
                const res = await fetch('/api/upload_stl', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    activeStlPath = data.filepath;
                    msg.innerText = `Ready to slice ${file.name}!`;
                } else {
                    msg.innerText = `Upload notice: ${data.message}`;
                }
            } catch (err) {
                msg.innerText = `Upload notice: ${err.message}`;
            }
        }

        function loadSampleStl(name) {
            activeStlPath = `stls/${name}.stl`;
            activeModelName = name;
            const loader = new THREE.STLLoader();
            loader.load(`/api/stl_file?name=${name}.stl`, function (geometry) {
                setupParsedGeometry(geometry, name);
            });
        }

        function setupParsedGeometry(geometry, modelName) {
            if (currentMesh) scene.remove(currentMesh);

            geometry.computeVertexNormals();
            geometry.center();
            originalGeometry = geometry.clone();

            const material = new THREE.MeshPhysicalMaterial({
                color: 0x06b6d4,
                metalness: 0.1,
                roughness: 0.25,
                clearcoat: 0.5,
            });

            currentMesh = new THREE.Mesh(geometry, material);
            scene.add(currentMesh);

            // Preset optimal tilt for benchy or reset
            if (modelName.toLowerCase().includes('benchy')) {
                document.getElementById('rotXInput').value = 35;
                document.getElementById('rotYInput').value = 0;
                document.getElementById('rotZInput').value = 0;
            } else {
                document.getElementById('rotXInput').value = 0;
                document.getElementById('rotYInput').value = 0;
                document.getElementById('rotZInput').value = 0;
            }

            autoFitToSafeZone();
        }

        function computeTrueCylinderBounds(mesh) {
            mesh.updateMatrixWorld(true);
            const geometry = mesh.geometry;
            const pos = geometry.attributes.position;
            const matrix = mesh.matrixWorld;

            let maxRad = 0;
            let minY = Infinity, maxY = -Infinity;
            let minX = Infinity, maxX = -Infinity;
            let minZ = Infinity, maxZ = -Infinity;

            const v = new THREE.Vector3();
            for (let i = 0; i < pos.count; i++) {
                v.fromBufferAttribute(pos, i).applyMatrix4(matrix);
                const r = Math.sqrt(v.x * v.x + v.z * v.z);
                if (r > maxRad) maxRad = r;
                if (v.y < minY) minY = v.y;
                if (v.y > maxY) maxY = v.y;
                if (v.x < minX) minX = v.x;
                if (v.x > maxX) maxX = v.x;
                if (v.z < minZ) minZ = v.z;
                if (v.z > maxZ) maxZ = v.z;
            }

            return {
                maxRadius: maxRad,
                maxDiameter: maxRad * 2.0,
                height: maxY - minY,
                minY: minY,
                maxY: maxY,
                sizeX: maxX - minX,
                sizeZ: maxZ - minZ
            };
        }

        function autoFitToSafeZone() {
            if (!currentMesh) return;

            currentMesh.scale.set(1, 1, 1);
            currentMesh.position.set(0, 0, 0);

            const rx = parseFloat(document.getElementById('rotXInput').value) || 0;
            const ry = parseFloat(document.getElementById('rotYInput').value) || 0;
            const rz = parseFloat(document.getElementById('rotZInput').value) || 0;
            currentMesh.rotation.set(THREE.MathUtils.degToRad(rx), THREE.MathUtils.degToRad(ry), THREE.MathUtils.degToRad(rz));

            const rawBounds = computeTrueCylinderBounds(currentMesh);

            const scaleRad = rawBounds.maxRadius > 0 ? (12.5 / rawBounds.maxRadius) : 1.0;
            const scaleH = rawBounds.height > 0 ? (50.0 / rawBounds.height) : 1.0;
            const targetScale = Math.min(scaleRad, scaleH, 1.0);

            document.getElementById('uniformScaleCheck').checked = true;
            document.getElementById('scaleXInput').value = targetScale.toFixed(3);
            document.getElementById('scaleYInput').value = targetScale.toFixed(3);
            document.getElementById('scaleZInput').value = targetScale.toFixed(3);

            document.getElementById('posXInput').value = 0.0;
            document.getElementById('posYInput').value = 0.0;
            document.getElementById('posZInput').value = 0.0;

            onTransformChanged();
        }

        function toggleUniformScale() {
            const isUniform = document.getElementById('uniformScaleCheck').checked;
            if (isUniform) {
                const sx = parseFloat(document.getElementById('scaleXInput').value) || 1.0;
                document.getElementById('scaleYInput').value = sx.toFixed(3);
                document.getElementById('scaleZInput').value = sx.toFixed(3);
                onScaleMultiplierChanged('x');
            }
        }

        function onScaleMultiplierChanged(axis) {
            const isUniform = document.getElementById('uniformScaleCheck').checked;
            let sx = parseFloat(document.getElementById('scaleXInput').value) || 1.0;
            let sy = parseFloat(document.getElementById('scaleYInput').value) || 1.0;
            let sz = parseFloat(document.getElementById('scaleZInput').value) || 1.0;

            if (isUniform) {
                const targetVal = parseFloat(document.getElementById(`scale${axis.toUpperCase()}Input`).value) || 1.0;
                sx = targetVal; sy = targetVal; sz = targetVal;
                document.getElementById('scaleXInput').value = targetVal.toFixed(3);
                document.getElementById('scaleYInput').value = targetVal.toFixed(3);
                document.getElementById('scaleZInput').value = targetVal.toFixed(3);
            }

            onTransformChanged();
        }

        function onDimensionChanged(axis) {
            if (!currentMesh) return;
            const bounds = computeTrueCylinderBounds(currentMesh);
            const currentDim = axis === 'x' ? bounds.sizeX : (axis === 'y' ? bounds.sizeZ : bounds.height);
            const targetDim = parseFloat(document.getElementById(`dim${axis.toUpperCase()}Input`).value) || 1.0;
            const factor = targetDim / Math.max(0.001, currentDim);

            let sx = (parseFloat(document.getElementById('scaleXInput').value) || 1.0) * factor;
            let sy = (parseFloat(document.getElementById('scaleYInput').value) || 1.0) * factor;
            let sz = (parseFloat(document.getElementById('scaleZInput').value) || 1.0) * factor;

            const isUniform = document.getElementById('uniformScaleCheck').checked;
            if (isUniform) {
                document.getElementById('scaleXInput').value = sx.toFixed(3);
                document.getElementById('scaleYInput').value = sx.toFixed(3);
                document.getElementById('scaleZInput').value = sz.toFixed(3);
            } else {
                document.getElementById(`scale${axis.toUpperCase()}Input`).value = (axis === 'x' ? sx : (axis === 'y' ? sy : sz)).toFixed(3);
            }

            onTransformChanged();
        }

        function onTransformChanged() {
            if (!currentMesh) return;

            const sx = parseFloat(document.getElementById('scaleXInput').value) || 1.0;
            const sy = parseFloat(document.getElementById('scaleYInput').value) || 1.0;
            const sz = parseFloat(document.getElementById('scaleZInput').value) || 1.0;

            const rx = parseFloat(document.getElementById('rotXInput').value) || 0.0;
            const ry = parseFloat(document.getElementById('rotYInput').value) || 0.0;
            const rz = parseFloat(document.getElementById('rotZInput').value) || 0.0;

            const tx = parseFloat(document.getElementById('posXInput').value) || 0.0;
            const ty = parseFloat(document.getElementById('posYInput').value) || 0.0;
            const tz = parseFloat(document.getElementById('posZInput').value) || 0.0;

            currentMesh.scale.set(sx, sy, sz);
            currentMesh.rotation.set(THREE.MathUtils.degToRad(rx), THREE.MathUtils.degToRad(ry), THREE.MathUtils.degToRad(rz));
            currentMesh.position.set(tx, tz, ty);

            const bounds = computeTrueCylinderBounds(currentMesh);

            document.getElementById('dimXInput').value = bounds.sizeX.toFixed(1);
            document.getElementById('dimYInput').value = bounds.sizeZ.toFixed(1);
            document.getElementById('dimZInput').value = bounds.height.toFixed(1);

            const isGlassCollision = bounds.maxDiameter > 30.0 || bounds.height > 60.0;
            const isOutsideSafeZone = bounds.maxDiameter > 25.0 || bounds.height > 50.0;

            if (isGlassCollision) {
                currentMesh.material.color.setHex(0xf43f5e);
            } else if (isOutsideSafeZone) {
                currentMesh.material.color.setHex(0xf59e0b);
            } else {
                currentMesh.material.color.setHex(0x06b6d4);
            }

            let statusHtml = `<span style="color: #10b981; font-weight: 700;">🟢 SAFE IN GREEN ZONE (Strict ≤ 25mm)</span>`;
            if (isGlassCollision) {
                statusHtml = `<span style="color: #f43f5e; font-weight: 700;">🔴 COLLISION (Hits 30mm Glass Wall!)</span>`;
            } else if (isOutsideSafeZone) {
                statusHtml = `<span style="color: #f59e0b; font-weight: 700;">🟡 CAUTION (Exceeds 25mm Safe Printable Margin)</span>`;
            }

            document.getElementById('meshBoundsText').innerHTML = `
                <b>${activeModelName.toUpperCase()}</b> | Extents: ${bounds.sizeX.toFixed(1)} &times; ${bounds.sizeZ.toFixed(1)} &times; ${bounds.height.toFixed(1)} mm 
                | Max Cylinder Dia: <b>${bounds.maxDiameter.toFixed(1)} mm</b> | ${statusHtml}
            `;
        }

        function setTilt(rx, ry, rz) {
            document.getElementById('rotXInput').value = rx;
            document.getElementById('rotYInput').value = ry;
            document.getElementById('rotZInput').value = rz;
            autoFitToSafeZone();
        }

        function autoCenterAoR() {
            document.getElementById('posXInput').value = 0.0;
            document.getElementById('posYInput').value = 0.0;
            document.getElementById('posZInput').value = 0.0;
            onTransformChanged();
        }

        async function triggerOptimizeCustomModel() {
            const btn = document.getElementById('btnFindSweetSpot');
            const msg = document.getElementById('slicerStatusMsg');
            btn.disabled = true;
            btn.innerText = "⚡ Voxelizing & Starting GPU Optimization ...";
            msg.innerText = "Transforming mesh and dispatching Genetic-Greedy search on GPU ...";

            const transformPayload = {
                stl_path: activeStlPath,
                name: activeModelName,
                scale_x: parseFloat(document.getElementById('scaleXInput').value) || 1.0,
                scale_y: parseFloat(document.getElementById('scaleYInput').value) || 1.0,
                scale_z: parseFloat(document.getElementById('scaleZInput').value) || 1.0,
                rx: parseFloat(document.getElementById('rotXInput').value) || 0.0,
                ry: parseFloat(document.getElementById('rotYInput').value) || 0.0,
                rz: parseFloat(document.getElementById('rotZInput').value) || 0.0,
                tx: parseFloat(document.getElementById('posXInput').value) || 0.0,
                ty: parseFloat(document.getElementById('posYInput').value) || 0.0,
                tz: parseFloat(document.getElementById('posZInput').value) || 0.0,
                resolution: 75
            };

            try {
                const res = await fetch('/api/optimize_custom_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(transformPayload)
                });
                const data = await res.json();
                if (data.success) {
                    selectedRunId = data.run_id;
                    userManuallySelectedRun = false;
                    msg.innerText = `Optimization live! Active Run: ${data.run_id}`;
                    switchTab('dashboard');
                    await fetchRunsList();
                    fetchRunData();
                } else {
                    msg.innerText = "Error: " + data.message;
                    alert("Optimizer notice: " + data.message);
                }
            } catch (err) {
                msg.innerText = "Request failed: " + err.message;
                alert("Request error: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerText = "⚡ Find Slicing Sweet Spot (Auto-Tune)";
            }
        }
    </script>

    <!-- App Dashboard & Communication Script -->
    <script>
        let allLogRecords = [];
        let metricChart = null;
        let selectedRunId = "";
        let isEngineRunning = false;
        let userManuallySelectedRun = false;
        let lastRenderedRunId = "";

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => {
                if (btn.getAttribute('data-tab') === tabId) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('active');
            });
            const targetPane = document.getElementById('tab-' + tabId);
            if (targetPane) targetPane.classList.add('active');
            if (tabId === 'heatmaps') renderHeatmap();
            if (tabId === 'slicer3d') onWindowResize();
            if (tabId === 'pareto' || tabId === 'slices') {
                const geoms = [...new Set(allLogRecords.map(r => r.geometry))];
                renderGalleries(selectedRunId, geoms);
            }
        }

        function initChart() {
            const ctx = document.getElementById('liveMetricChart').getContext('2d');
            metricChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Process Window (PW)', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', fill: true, tension: 0.3, yAxisID: 'y' },
                        { label: 'VER (%)', data: [], borderColor: '#f43f5e', backgroundColor: 'rgba(244, 63, 94, 0.05)', fill: false, tension: 0.3, yAxisID: 'y1' }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: 'rgba(75, 85, 99, 0.15)' }, ticks: { color: '#9ca3af', font: { size: 10 } } },
                        y: { type: 'linear', position: 'left', title: { display: true, text: 'PW', color: '#10b981' }, grid: { color: 'rgba(75, 85, 99, 0.15)' }, ticks: { color: '#10b981', font: { size: 10 } } },
                        y1: { type: 'linear', position: 'right', title: { display: true, text: 'VER (%)', color: '#f43f5e' }, grid: { drawOnChartArea: false }, ticks: { color: '#f43f5e', font: { size: 10 } } }
                    },
                    plugins: { legend: { labels: { color: '#f3f4f6', font: { size: 11 } } } }
                }
            });
        }

        async function triggerBenchmarkAll() {
            userManuallySelectedRun = false;
            const res = await fetch('/api/start_run', { method: 'POST' });
            const data = await res.json();
            if (data.run_id) selectedRunId = data.run_id;
            switchTab('dashboard');
            fetchRunsList();
        }

        async function triggerStopRun() {
            const res = await fetch('/api/stop_run', { method: 'POST' });
            const data = await res.json();
        }

        async function triggerGenerateVideo() {
            const geom = document.getElementById('videoGeomSelect').value;
            const rpm = document.getElementById('videoRpmSelect').value;
            const dur = document.getElementById('videoDurationSelect').value;
            const statusMsg = document.getElementById('videoStatusMsg');
            const btn = document.getElementById('btnGenVideo');

            if (!selectedRunId) {
                statusMsg.innerText = "Please select a run first.";
                return;
            }

            btn.disabled = true;
            statusMsg.innerText = `Generating ${dur}s film at ${rpm} RPM ...`;

            try {
                const res = await fetch(`/api/generate_video?run_id=${selectedRunId}&geometry=${geom}&rpm=${rpm}&duration=${dur}`);
                const data = await res.json();
                if (data.success) {
                    statusMsg.innerText = `Film generated! (${(data.size_bytes/1024/1024).toFixed(2)} MB)`;
                    const playerBox = document.getElementById('videoPlayerBox');
                    const player = document.getElementById('videoPlayer');
                    const dlBtn = document.getElementById('videoDownloadBtn');
                    const mfBtn = document.getElementById('manifestDownloadBtn');
                    const meta = document.getElementById('videoMetaText');

                    player.src = data.video_url + "&t=" + new Date().getTime();
                    dlBtn.href = data.video_url;
                    dlBtn.download = data.filename;
                    mfBtn.href = `/api/export_manifest?run_id=${selectedRunId}`;
                    mfBtn.download = `manifest_${selectedRunId}.json`;
                    meta.innerText = `${data.filename} (${(data.size_bytes/1024/1024).toFixed(2)} MB)`;
                    playerBox.style.display = 'flex';
                    player.play();
                } else {
                    statusMsg.innerText = "Error: " + (data.error || "Generation failed.");
                }
            } catch (err) {
                statusMsg.innerText = "Request failed: " + err.message;
            } finally {
                btn.disabled = false;
            }
        }

        async function fetchEngineStatus() {
            try {
                const res = await fetch('/api/engine_status');
                const st = await res.json();
                isEngineRunning = st.is_running;

                document.getElementById('btnStartBenchmark').disabled = isEngineRunning;
                document.getElementById('btnStop').disabled = !isEngineRunning;

                if (isEngineRunning) {
                    document.getElementById('kpiActiveGeom').innerText = st.active_geom;
                    document.getElementById('kpiActivePhase').innerText = st.active_phase;
                    if (st.active_run_id && !userManuallySelectedRun && selectedRunId !== st.active_run_id) {
                        selectedRunId = st.active_run_id;
                        document.getElementById('runSelector').value = st.active_run_id;
                    }
                }
            } catch (e) {}
        }

        async function fetchRunsList() {
            try {
                const res = await fetch('/api/runs');
                const data = await res.json();
                const sel = document.getElementById('runSelector');
                const prev = sel.value;
                sel.innerHTML = "";

                let runningRunId = "";
                data.runs.forEach(r => {
                    const opt = document.createElement('option');
                    opt.value = r.run_id;
                    opt.innerText = `${r.run_id} [${r.status}] (${r.evaluations} trials)`;
                    if (r.status === "RUNNING") runningRunId = r.run_id;
                    sel.appendChild(opt);
                });

                if (!userManuallySelectedRun) {
                    if (runningRunId) {
                        sel.value = runningRunId;
                    } else if (data.runs.length > 0) {
                        sel.value = data.runs[0].run_id;
                    }
                    selectedRunId = sel.value;
                } else if (prev && [...sel.options].some(o => o.value === prev)) {
                    sel.value = prev;
                    selectedRunId = prev;
                }
            } catch (e) {}
        }

        async function fetchRunData() {
            if (!selectedRunId) return;
            try {
                const res = await fetch(`/api/run_data?run_id=${selectedRunId}`);
                const data = await res.json();

                const badge = document.getElementById('runStatusBadge');
                badge.innerText = data.status || "UNKNOWN";
                if (data.status === "COMPLETED") {
                    badge.style.background = "rgba(16, 185, 129, 0.2)";
                    badge.style.color = "var(--accent-emerald)";
                    badge.style.border = "1px solid rgba(16, 185, 129, 0.4)";
                } else if (data.status === "RUNNING") {
                    badge.style.background = "rgba(6, 182, 212, 0.2)";
                    badge.style.color = "var(--accent-cyan)";
                    badge.style.border = "1px solid rgba(6, 182, 212, 0.4)";
                } else {
                    badge.style.background = "rgba(244, 63, 94, 0.2)";
                    badge.style.color = "var(--accent-rose)";
                    badge.style.border = "1px solid rgba(244, 63, 94, 0.4)";
                }

                allLogRecords = data.records || [];
                document.getElementById('kpiTotalRuns').innerText = allLogRecords.length;
                document.getElementById('kpiRunId').innerText = selectedRunId;

                const geoms = [...new Set(allLogRecords.map(r => r.geometry))];
                updateGeomDropdowns(geoms);

                if (allLogRecords.length > 0) {
                    let bestPW = -999, bestPWGeom = "", lowestVER = 999, lowestVERGeom = "";
                    allLogRecords.forEach(r => {
                        if (r.pw > bestPW) { bestPW = r.pw; bestPWGeom = r.geometry; }
                        if (r.ver < lowestVER) { lowestVER = r.ver; lowestVERGeom = r.geometry; }
                    });

                    document.getElementById('kpiBestPW').innerText = (bestPW >= 0 ? "+" : "") + bestPW.toFixed(3);
                    document.getElementById('kpiBestPWGeom').innerText = `Best on ${bestPWGeom}`;
                    document.getElementById('kpiLowestVER').innerText = (lowestVER * 100).toFixed(1) + "%";
                    document.getElementById('kpiLowestVERGeom').innerText = `Best on ${lowestVERGeom}`;
                }

                if (metricChart && allLogRecords.length > 0) {
                    const recent = allLogRecords.slice(-50);
                    metricChart.data.labels = recent.map((_, idx) => "#" + (allLogRecords.length - recent.length + idx + 1));
                    metricChart.data.datasets[0].data = recent.map(r => r.pw);
                    metricChart.data.datasets[1].data = recent.map(r => r.ver * 100);
                    metricChart.update();
                }

                renderBestConfigs(data.best_by_geom || {});
                renderLogTable(allLogRecords);

                if (data.status === "COMPLETED" && lastRenderedRunId !== selectedRunId) {
                    renderGalleries(selectedRunId, geoms);
                    lastRenderedRunId = selectedRunId;
                }

                if (document.getElementById('tab-heatmaps').classList.contains('active')) {
                    renderHeatmap();
                }
            } catch (err) {
                console.error("fetchRunData error:", err);
            }
        }

        function updateGeomDropdowns(geoms) {
            const hSel = document.getElementById('heatmapGeomSelect');
            const vSel = document.getElementById('videoGeomSelect');
            if (geoms.length === 0) return;

            [hSel, vSel].forEach(sel => {
                const prev = sel.value;
                sel.innerHTML = "";
                geoms.forEach(g => {
                    const opt = document.createElement('option');
                    opt.value = g;
                    opt.innerText = g.toUpperCase();
                    sel.appendChild(opt);
                });
                if (prev && geoms.includes(prev)) sel.value = prev;
            });
        }

        function onRunChanged() {
            userManuallySelectedRun = true;
            selectedRunId = document.getElementById('runSelector').value;
            fetchRunData();
        }

        let activeBestConfigs = {};
        let activeModalGeom = "";
        let activeModalRunId = "";

        function renderBestConfigs(bestMap) {
            activeBestConfigs = bestMap;
            const container = document.getElementById('bestConfigsContainer');
            container.innerHTML = "";
            for (const [geom, rec] of Object.entries(bestMap)) {
                const div = document.createElement('div');
                div.className = "recipe-card-interactive";
                div.style.background = "rgba(31, 41, 55, 0.5)";
                div.style.padding = "12px 14px";
                div.style.borderRadius = "10px";
                div.style.border = "1px solid var(--border-color)";
                div.onclick = () => openRecipeModal(geom, selectedRunId);
                div.innerHTML = `
                    <div style="display: flex; justify-content: space-between; font-weight: 700; margin-bottom: 3px;">
                        <span style="color: var(--accent-cyan); text-transform: uppercase;">${geom}</span>
                        <span style="color: ${rec.pw >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; font-size: 13px;">PW: ${(rec.pw >= 0 ? '+' : '') + rec.pw.toFixed(3)}</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted); line-height: 1.4;">
                        Method: <b style="color: #fff;">${rec.method}</b> | Filter: <b style="color: #fff;">${rec.filter || 'None'}</b> | Angles: <b style="color: #fff;">${rec.n_angles}</b><br>
                        d_h: <b style="color: #fff;">${rec.d_h}</b> | d_l: <b style="color: #fff;">${rec.d_l}</b> | VER: <b style="color: #fff;">${(rec.ver * 100).toFixed(1)}%</b> | Contrast: <b style="color: #fff;">${rec.contrast ? rec.contrast.toFixed(1) : '-'}x</b>
                    </div>
                    <div style="margin-top: 6px; font-size: 10px; color: var(--accent-cyan); font-weight: 700; display: flex; align-items: center; gap: 4px;">
                        🔍 Click to view ALL Tomo settings & generate 1080p film ➔
                    </div>
                `;
                container.appendChild(div);
            }
        }

        function openRecipeModal(geom, runId) {
            activeModalGeom = geom;
            activeModalRunId = runId || selectedRunId;
            const rec = activeBestConfigs[geom] || {};
            const overlay = document.getElementById('recipeModalOverlay');
            document.getElementById('modalTitle').innerText = `${geom.toUpperCase()} — Complete Tomo Recipe`;
            document.getElementById('modalGeomBadge').innerText = rec.method || "OSMO";
            document.getElementById('modalVideoStatus').innerText = "";
            document.getElementById('modalVideoPlayerBox').style.display = "none";

            // 1. Optimizer Parameters (Tomo Core)
            const optDiv = document.getElementById('modalOptParams');
            optDiv.innerHTML = `
                <div class="setting-item"><span class="k">Algorithm (method):</span><span class="v" style="color: var(--accent-cyan);">${rec.method || 'OSMO'}</span></div>
                <div class="setting-item"><span class="k">Angular Sampling (n_angles):</span><span class="v">${rec.n_angles || 180} projections</span></div>
                <div class="setting-item"><span class="k">Filter Function (filter):</span><span class="v">${rec.filter || 'None'}</span></div>
                <div class="setting-item"><span class="k">In-Target Gel Floor (d_h):</span><span class="v" style="color: var(--accent-emerald);">${rec.d_h != null ? rec.d_h : '0.85'}</span></div>
                <div class="setting-item"><span class="k">Void Ceiling Limit (d_l):</span><span class="v" style="color: var(--accent-rose);">${rec.d_l != null ? rec.d_l : '0.60'}</span></div>
                <div class="setting-item"><span class="k">Iterations (n_iter):</span><span class="v">${rec.n_iter || 15}</span></div>
                <div class="setting-item"><span class="k">OSMO Inhibition (γ):</span><span class="v">${rec.inhibition != null ? rec.inhibition : '0.00'}</span></div>
                <div class="setting-item"><span class="k">BCLP Band Epsilon (eps):</span><span class="v">${rec.eps != null ? rec.eps : '0.10'}</span></div>
                <div class="setting-item"><span class="k">BCLP Lp Norm (p):</span><span class="v">${rec.p_norm != null ? rec.p_norm : '2.0'}</span></div>
                <div class="setting-item"><span class="k">Optimizer LR (learning_rate):</span><span class="v">${rec.learning_rate != null ? rec.learning_rate : '0.010'}</span></div>
            `;

            // 2. Physical Optical & Resin Parameters (Tomo Slicer)
            const physDiv = document.getElementById('modalPhysParams');
            physDiv.innerHTML = `
                <div class="setting-item"><span class="k">Projector Resolution:</span><span class="v">1920 × 1080 px (1080p)</span></div>
                <div class="setting-item"><span class="k">Optical Beam / Collimation:</span><span class="v">Telecentric (Parallel, TR=∞)</span></div>
                <div class="setting-item"><span class="k">Projector FOV Width:</span><span class="v">108.0 mm</span></div>
                <div class="setting-item"><span class="k">Physical Resin Vial:</span><span class="v">30.0 mm dia × 60.0 mm</span></div>
                <div class="setting-item"><span class="k">Safe Printable Margin:</span><span class="v" style="color: var(--accent-emerald);">25.0 mm dia × 50.0 mm</span></div>
                <div class="setting-item"><span class="k">Resin Refractive Index (n):</span><span class="v">1.48 (vial_correction = ON)</span></div>
                <div class="setting-item"><span class="k">Dynamic Normalization:</span><span class="v">99.9th Percentile</span></div>
                <div class="setting-item"><span class="k">DLP Optical Coordinate:</span><span class="v">np.flipud (Bottom-Origin)</span></div>
                <div class="setting-item"><span class="k">Optical Attenuation (μ):</span><span class="v">0.0 mm⁻¹ (Homogeneous)</span></div>
            `;

            // 3. Quality & Convergence Metrics
            const qDiv = document.getElementById('modalQualityParams');
            const pwVal = rec.pw != null ? rec.pw : 0;
            const verVal = rec.ver != null ? (rec.ver * 100).toFixed(1) + "%" : "0.0%";
            qDiv.innerHTML = `
                <div class="setting-item"><span class="k">Process Window (PW):</span><span class="v" style="color: ${pwVal >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; font-size: 13px;">${(pwVal >= 0 ? '+' : '') + pwVal.toFixed(3)}</span></div>
                <div class="setting-item"><span class="k">Volumetric Over-Cure (VER):</span><span class="v" style="color: ${rec.ver < 0.05 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${verVal}</span></div>
                <div class="setting-item"><span class="k">Peak Contrast Ratio:</span><span class="v">${rec.contrast ? rec.contrast.toFixed(2) + 'x' : '-'}</span></div>
                <div class="setting-item"><span class="k">GPU Trial Runtime:</span><span class="v">${rec.runtime_s ? rec.runtime_s.toFixed(2) + 's' : '-'}</span></div>
                <div class="setting-item"><span class="k">VRAM Memory Footprint:</span><span class="v">${rec.memory_mb ? rec.memory_mb.toFixed(1) + ' MB' : '-'}</span></div>
                <div class="setting-item"><span class="k">GPU Backend:</span><span class="v" style="color: var(--accent-cyan);">ASTRA CUDA 12.x</span></div>
            `;

            overlay.classList.add('active');
        }

        function closeRecipeModal(e) {
            if (e.target.id === 'recipeModalOverlay') {
                document.getElementById('recipeModalOverlay').classList.remove('active');
            }
        }

        async function generateVideoFromModal() {
            const rpm = document.getElementById('modalVideoRpm').value;
            const dur = document.getElementById('modalVideoDur').value;
            const statusMsg = document.getElementById('modalVideoStatus');
            const btn = document.getElementById('btnModalGenVideo');

            btn.disabled = true;
            statusMsg.innerText = `🎬 Voxelizing & encoding 1080p Tomo-grade projection film (${dur}s @ ${rpm} RPM) ...`;

            const targetRunId = activeModalRunId || selectedRunId;
            try {
                const res = await fetch(`/api/generate_video?run_id=${targetRunId}&geometry=${activeModalGeom}&rpm=${rpm}&duration=${dur}`);
                const data = await res.json();
                if (data.success) {
                    statusMsg.innerText = `✅ 1080p Film Generated Successfully! (${(data.size_bytes/1024/1024).toFixed(2)} MB)`;
                    const playerBox = document.getElementById('modalVideoPlayerBox');
                    const player = document.getElementById('modalVideoPlayer');
                    const dlBtn = document.getElementById('modalVideoDlBtn');
                    const mfBtn = document.getElementById('modalManifestDlBtn');

                    player.src = data.video_url + "&t=" + new Date().getTime();
                    dlBtn.href = data.video_url;
                    dlBtn.download = data.filename;
                    mfBtn.href = `/api/export_manifest?run_id=${targetRunId}`;
                    mfBtn.download = `manifest_${targetRunId}.json`;
                    playerBox.style.display = 'flex';
                    player.play();
                } else {
                    statusMsg.innerText = "❌ Error: " + (data.error || "Generation failed.");
                }
            } catch (err) {
                statusMsg.innerText = "❌ Request error: " + err.message;
            } finally {
                btn.disabled = false;
            }
        }

        function renderGalleries(runId, geoms) {
            const chartDiv = document.getElementById('chartsGallery');
            const ts = new Date().getTime();
            chartDiv.innerHTML = `
                <div class="gallery-card"><img src="/api/image?run_id=${runId}&type=charts&name=pareto_frontier_ver_vs_pw.png&t=${ts}" alt="Pareto" onerror="this.parentElement.style.display='none'"><div class="gallery-caption">Volumetric Error vs Process Window Pareto Frontier</div></div>
                <div class="gallery-card"><img src="/api/image?run_id=${runId}&type=charts&name=radar_geometry_profiles.png&t=${ts}" alt="Radar" onerror="this.parentElement.style.display='none'"><div class="gallery-caption">Optimal Quality Profile Radar by Geometry</div></div>
                <div class="gallery-card"><img src="/api/image?run_id=${runId}&type=charts&name=parameter_importance_ranking.png&t=${ts}" alt="Importance" onerror="this.parentElement.style.display='none'"><div class="gallery-caption">Quantitative Parameter Sensitivity & Correlation</div></div>
                <div class="gallery-card"><img src="/api/image?run_id=${runId}&type=charts&name=optimizer_method_comparison_boxplots.png&t=${ts}" alt="Boxplot" onerror="this.parentElement.style.display='none'"><div class="gallery-caption">Optimizer Method Performance Distribution</div></div>
            `;

            const sliceDiv = document.getElementById('slicesGallery');
            sliceDiv.innerHTML = "";
            (geoms || []).forEach(g => {
                const card = document.createElement('div');
                card.className = "gallery-card";
                card.innerHTML = `<img src="/api/image?run_id=${runId}&type=slices&name=reconstruction_slices_${g}.png&t=${ts}" alt="${g}" onerror="this.parentElement.style.display='none'"><div class="gallery-caption">${g.toUpperCase()} Vial-Fitted Recon (Z=mid)</div>`;
                sliceDiv.appendChild(card);
            });
        }

        function renderLogTable(records) {
            const tbody = document.getElementById('logTableBody');
            tbody.innerHTML = "";
            records.slice().reverse().forEach((r, idx) => {
                const tr = document.createElement('tr');
                const badgeClass = "badge-" + (r.method ? r.method.toLowerCase() : "fbp");
                tr.innerHTML = `
                    <td style="color: var(--text-muted);">${records.length - idx}</td>
                    <td><b>${r.geometry}</b></td>
                    <td style="font-size: 11px; color: var(--text-muted);">${r.phase}</td>
                    <td><span class="badge ${badgeClass}">${r.method}</span></td>
                    <td>${r.n_angles}</td>
                    <td>${r.filter || 'None'}</td>
                    <td>${r.d_h}</td>
                    <td>${r.d_l}</td>
                    <td style="font-weight: 700; color: ${r.pw >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${(r.pw >= 0 ? '+' : '') + r.pw.toFixed(3)}</td>
                    <td style="color: ${r.ver < 0.05 ? 'var(--accent-emerald)' : 'var(--text-main)'};">${(r.ver * 100).toFixed(1)}%</td>
                    <td>${r.contrast ? r.contrast.toFixed(2) : '-'}x</td>
                    <td style="color: var(--text-muted);">${r.runtime_s ? r.runtime_s.toFixed(2) : '-'}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function filterLogTable() {
            const q = document.getElementById('logSearchInput').value.toLowerCase();
            const filtered = allLogRecords.filter(r => 
                (r.geometry && r.geometry.toLowerCase().includes(q)) ||
                (r.method && r.method.toLowerCase().includes(q)) ||
                (r.phase && r.phase.toLowerCase().includes(q)) ||
                (r.filter && String(r.filter).toLowerCase().includes(q))
            );
            renderLogTable(filtered);
        }

        function renderHeatmap() {
            const geom = document.getElementById('heatmapGeomSelect').value;
            const axis = document.getElementById('heatmapAxisSelect').value;
            const container = document.getElementById('heatmapTableContainer');

            let sub = allLogRecords.filter(r => r.geometry === geom && r.phase.includes(axis.replace("_vs_", "_")));
            if (sub.length === 0) sub = allLogRecords.filter(r => r.geometry === geom);

            if (sub.length === 0) {
                container.innerHTML = `<div style="padding: 20px; color: var(--text-muted); font-size: 12px;">No records available yet for ${geom} in this run.</div>`;
                return;
            }

            let rowKey = "d_h", colKey = "d_l";
            if (axis === "angles_vs_filter") { rowKey = "filter"; colKey = "n_angles"; }
            else if (axis === "eps_vs_p") { rowKey = "eps"; colKey = "p_norm"; }

            const rows = [...new Set(sub.map(r => r[rowKey]))].sort();
            const cols = [...new Set(sub.map(r => r[colKey]))].sort();

            let html = `<table class="heatmap-table"><thead><tr><th>${rowKey} \\ ${colKey}</th>`;
            cols.forEach(c => { html += `<th>${c || 'None'}</th>`; });
            html += `</tr></thead><tbody>`;

            rows.forEach(rVal => {
                html += `<tr><th><b>${rVal || 'None'}</b></th>`;
                cols.forEach(cVal => {
                    const match = sub.find(x => x[rowKey] == rVal && x[colKey] == cVal);
                    if (match) {
                        const pw = match.pw;
                        const ver = match.ver * 100;
                        const bgCol = pw >= 0 ? `rgba(16, 185, 129, ${Math.min(1.0, 0.2 + pw * 0.8)})` : `rgba(244, 63, 94, ${Math.min(1.0, 0.2 + Math.abs(pw) * 0.8)})`;
                        html += `<td style="background: ${bgCol};">
                            <div class="cell-value">${(pw >= 0 ? '+' : '') + pw.toFixed(2)}</div>
                            <div style="font-size: 9px; opacity: 0.8;">VER: ${ver.toFixed(0)}%</div>
                        </td>`;
                    } else {
                        html += `<td>-</td>`;
                    }
                });
                html += `</tr>`;
            });
            html += `</tbody></table>`;
            container.innerHTML = html;
        }

        window.onload = () => {
            init3DViewport();
            initChart();
            fetchRunsList();
            setInterval(fetchEngineStatus, 1000);
            setInterval(fetchRunsList, 5000);
            setInterval(fetchRunData, 1500);
        };
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, local_ip=LOCAL_IP, port=PORT)

@app.route("/api/upload_stl", methods=["POST"])
def api_upload_stl():
    if "stl" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    file = request.files["stl"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOADS_DIR, filename)
    file.save(save_path)

    name = os.path.splitext(filename)[0]
    return jsonify({
        "success": True,
        "name": name,
        "filename": filename,
        "filepath": save_path,
        "stl_url": f"/api/stl_file?name={filename}&uploaded=1",
    })

@app.route("/api/stl_file")
def api_stl_file():
    name = request.args.get("name")
    is_uploaded = request.args.get("uploaded", "0") == "1"
    if not name:
        return "Not found", 404

    if is_uploaded and os.path.exists(os.path.join(UPLOADS_DIR, name)):
        return send_from_directory(UPLOADS_DIR, name)
    elif os.path.exists(os.path.join(STLS_DIR, name)):
        return send_from_directory(STLS_DIR, name)
    elif os.path.exists(os.path.join(ROOT_STLS_DIR, name)):
        return send_from_directory(ROOT_STLS_DIR, name)
    return "Not found", 404

@app.route("/api/optimize_custom_model", methods=["POST"])
def api_optimize_custom_model():
    payload = request.get_json()
    if not payload or "stl_path" not in payload:
        return jsonify({"success": False, "message": "Invalid transform payload"}), 400

    stl_path = payload["stl_path"]
    if not os.path.isabs(stl_path):
        candidate_paths = [
            os.path.abspath(os.path.join(STUDIO_DIR, stl_path)),
            os.path.abspath(os.path.join(ROOT_DIR, stl_path)),
            os.path.abspath(os.path.join(STLS_DIR, os.path.basename(stl_path))),
            os.path.abspath(os.path.join(ROOT_STLS_DIR, os.path.basename(stl_path))),
            os.path.abspath(os.path.join(UPLOADS_DIR, os.path.basename(stl_path))),
        ]
        for cp in candidate_paths:
            if os.path.exists(cp):
                stl_path = cp
                break

    if not os.path.exists(stl_path):
        return jsonify({"success": False, "message": f"STL file not found: {stl_path}"}), 404

    ok, msg, run_id = CONTROLLER.start_custom_optimization(stl_path, payload)
    return jsonify({"success": ok, "message": msg, "run_id": run_id})

@app.route("/api/start_run", methods=["POST"])
def api_start():
    ok, msg, run_id = CONTROLLER.start_all_run()
    return jsonify({"success": ok, "message": msg, "run_id": run_id})

@app.route("/api/stop_run", methods=["POST"])
def api_stop():
    ok, msg = CONTROLLER.stop_run()
    return jsonify({"success": ok, "message": msg})

@app.route("/api/generate_video")
def api_generate_video():
    run_id = request.args.get("run_id")
    geom = request.args.get("geometry", "cube")
    rpm = float(request.args.get("rpm", 9.0))
    duration = float(request.args.get("duration", 60.0))

    if not run_id:
        return jsonify({"success": False, "error": "No run_id specified."}), 400

    try:
        filename, size_bytes = generate_projection_video(run_id, geom, rpm, duration)
        video_url = f"/api/video?run_id={run_id}&name={filename}"
        return jsonify({
            "success": True,
            "filename": filename,
            "size_bytes": size_bytes,
            "video_url": video_url,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/video")
def api_video():
    run_id = request.args.get("run_id")
    filename = request.args.get("name")
    if not run_id or not filename:
        return "Not found", 404
    video_dir = os.path.join(RUNS_DIR, run_id, "videos")
    return send_from_directory(video_dir, filename, mimetype="video/mp4")

@app.route("/api/export_manifest")
def api_export_manifest():
    run_id = request.args.get("run_id")
    if not run_id:
        return "Not found", 404
    run_dir = os.path.join(RUNS_DIR, run_id)
    comp_file = os.path.join(run_dir, "RUN_COMPLETE.json")
    if os.path.exists(comp_file):
        return send_from_directory(run_dir, "RUN_COMPLETE.json", mimetype="application/json")
    status_file = os.path.join(run_dir, "status.json")
    return send_from_directory(run_dir, "status.json", mimetype="application/json")

@app.route("/api/engine_status")
def api_engine_status():
    return jsonify({
        "is_running": CONTROLLER.is_running,
        "active_geom": CONTROLLER.active_geom,
        "active_phase": CONTROLLER.active_phase,
        "progress_index": CONTROLLER.progress_index,
        "active_run_id": CONTROLLER.active_run_id or CONTROLLER.get_latest_run_id(),
        "workers": CONTROLLER.workers_telemetry,
    })

@app.route("/api/runs")
def api_runs():
    runs_list = []
    if os.path.exists(RUNS_DIR):
        for entry in sorted(os.listdir(RUNS_DIR), reverse=True):
            entry_path = os.path.join(RUNS_DIR, entry)
            if os.path.isdir(entry_path):
                status_file = os.path.join(entry_path, "status.json")
                log_csv = os.path.join(entry_path, "live_experiment_log.csv")
                comp_file = os.path.join(entry_path, "RUN_COMPLETE.json")

                status = "INCOMPLETE"
                if os.path.exists(comp_file):
                    status = "COMPLETED"
                elif os.path.exists(status_file):
                    try:
                        with open(status_file, "r") as f:
                            st = json.load(f)
                            status = st.get("status", "INCOMPLETE")
                    except:
                        pass

                eval_count = 0
                if os.path.exists(log_csv):
                    try:
                        df = pd.read_csv(log_csv)
                        eval_count = len(df)
                    except:
                        pass

                runs_list.append({"run_id": entry, "status": status, "evaluations": eval_count})

    return jsonify({"runs": runs_list})

@app.route("/api/run_data")
def api_run_data():
    run_id = request.args.get("run_id")
    if not run_id:
        return jsonify({"records": [], "best_by_geom": {}, "status": "UNKNOWN"})

    run_dir = os.path.join(RUNS_DIR, run_id)
    log_csv = os.path.join(run_dir, "live_experiment_log.csv")
    status_file = os.path.join(run_dir, "status.json")
    comp_file = os.path.join(run_dir, "RUN_COMPLETE.json")

    status = "COMPLETED" if os.path.exists(comp_file) else "INCOMPLETE"
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                st = json.load(f)
                status = st.get("status", status)
        except:
            pass

    if not os.path.exists(log_csv):
        return jsonify({"records": [], "best_by_geom": {}, "status": status})

    try:
        df = pd.read_csv(log_csv)
        records = json.loads(df.to_json(orient="records"))

        best_by_geom = {}
        for geom in df["geometry"].unique():
            sub = df[df["geometry"] == geom]
            best_row = sub.sort_values(by=["pw", "ver"], ascending=[False, True]).iloc[0:1]
            best_by_geom[geom] = json.loads(best_row.to_json(orient="records"))[0]

        return jsonify({"records": records, "best_by_geom": best_by_geom, "status": status})
    except Exception as e:
        return jsonify({"records": [], "best_by_geom": {}, "status": status, "error": str(e)})

@app.route("/api/image")
def api_image():
    run_id = request.args.get("run_id")
    folder_type = request.args.get("type", "charts")
    image_name = request.args.get("name")
    if not run_id or not image_name:
        return "Not found", 404

    target_dir = os.path.join(RUNS_DIR, run_id, folder_type)
    return send_from_directory(target_dir, image_name)

def start_flask_server():
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)

def launch_gui():
    server_thread = threading.Thread(target=start_flask_server, daemon=True)
    server_thread.start()
    time.sleep(0.8)

    url_local = f"http://127.0.0.1:{PORT}"
    print("=" * 70)
    print("  OPENCAL 3D VAM SLICER & SWEET-SPOT STUDIO IS LIVE")
    print(f"  • Local Desktop URL : {url_local}")
    print("=" * 70)

    try:
        import webview
        print("[GUI] Spawning native pywebview window ...")
        webview.create_window(
            "OpenCAL VAM 3D Slicer & Sweet-Spot Studio",
            url=url_local,
            width=1440,
            height=940,
            min_size=(1080, 720),
            background_color="#0a0e17",
        )
        webview.start()
    except Exception as e:
        print(f"[GUI] Native webview notice: {e}")
        print(f"[GUI] Web dashboard is live at: {url_local}")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    launch_gui()
