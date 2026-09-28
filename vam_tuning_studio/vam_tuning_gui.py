#!/usr/bin/env python3
"""
======================================================================
  OPENCAL 3D VAM SINGLE-MODEL STUDIO & TOMO SLICER
======================================================================
Interactive desktop GUI for single-model tomographic 3D printing optimization.
Streams trial parameter changes on-the-fly, locates the optimal Sweet Spot,
and renders bit-for-bit identical projection videos matching Tomo.
"""

import os
import sys
import time
import json
import queue
import threading
import subprocess
import datetime
import trimesh
import numpy as np
from flask import Flask, render_template_string, request, jsonify, Response, send_file
from flask_cors import CORS

# Add root directory to python path
STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(STUDIO_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, STUDIO_DIR)

import vamtoolbox as vam
from single_model_optimizer import (
    run_single_model_optimization_stream,
    voxelize_model,
    sanitize_val,
    ROOT_RUNS_DIR,
    DEFAULT_PITCH_MM,
    DEFAULT_FPS,
    DEFAULT_RPM,
    DEFAULT_DURATION_S,
)
from tomo_video_engine import render_exact_tomo_video, extract_preview_frames
from generate_charts import generate_single_model_charts

# Built-in STL models
PRESET_STLS = {
    "cube": os.path.join(ROOT_DIR, "stls", "cube.stl"),
    "benchy": os.path.join(ROOT_DIR, "stls", "benchy.stl"),
    "captiveRing": os.path.join(ROOT_DIR, "stls", "captiveRing.stl"),
    "sphere": os.path.join(ROOT_DIR, "stls", "sphere.stl"),
}

UPLOADS_DIR = os.path.join(STUDIO_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Global Optimization & Control State
_CURRENT_RUN_STATE = {
    "is_running": False,
    "current_run_dir": None,
    "current_stl": "cube",
    "total_trials": 0,
    "current_trial_idx": 0,
    "best_trial": None,
    "trials": [],
    "active_video_path": None,
}

_TELEMETRY_QUEUE = queue.Queue()
_STOP_EVENT = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# HTML / CSS / JS SINGLE-PAGE APPLICATION TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
GUI_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>OpenCAL VAM Single-Model Studio & Tomo Slicer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>

  <style>
    :root {
      --bg-base: #0c0d14;
      --bg-surface: #141622;
      --bg-card: #1b1e30;
      --bg-card-hover: #22263d;
      --border: #282c47;
      --border-focus: #3d446e;
      --text-main: #f0f2fc;
      --text-muted: #8b92b2;
      --text-dim: #5c6280;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --accent-purple: #8b5cf6;
      --gold-glow: 0 0 15px rgba(245, 158, 11, 0.35);
      --card-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-base);
      color: var(--text-main);
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* Top Navigation Header */
    header {
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 16px;
      color: #fff;
    }
    .brand h1 {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .brand p {
      font-size: 11px;
      color: var(--text-muted);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
    }
    .badge-primary { background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-success { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-gold { background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-danger { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.3); }

    /* Main Workspace Layout (3 Column Grid) */
    .app-body {
      display: grid;
      grid-template-columns: 360px 1fr 440px;
      flex: 1;
      overflow: hidden;
    }

    /* Left Sidebar: Controls & Transforms */
    .sidebar {
      background: var(--bg-surface);
      border-right: 1px solid var(--border);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      overflow-y: auto;
    }
    .section-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .control-group {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    label {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }
    select, input[type="text"], input[type="number"] {
      width: 100%;
      background: var(--bg-base);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 10px;
      color: var(--text-main);
      font-size: 12px;
      font-family: inherit;
      outline: none;
      transition: border 0.15s;
    }
    select:focus, input:focus {
      border-color: var(--accent-blue);
    }
    .input-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 6px;
    }
    .input-row input {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
    }
    .btn-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .btn {
      width: 100%;
      padding: 9px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border: none;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .btn-sm {
      padding: 5px 10px;
      font-size: 11px;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-blue), #2563eb);
      color: #fff;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .btn-primary:hover { opacity: 0.92; transform: translateY(-1px); }
    .btn-success {
      background: linear-gradient(135deg, var(--accent-green), #059669);
      color: #fff;
    }
    .btn-danger {
      background: linear-gradient(135deg, var(--accent-rose), #dc2626);
      color: #fff;
      box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
    }
    .btn-danger:hover { opacity: 0.92; }
    .btn-secondary {
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border);
    }
    .btn-secondary:hover { background: var(--bg-card-hover); }
    .btn-cyan {
      background: rgba(6, 182, 212, 0.15);
      color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.35);
    }
    .btn-cyan:hover { background: rgba(6, 182, 212, 0.25); }

    /* Center Pane: 3D Viewport & Telemetry Logs */
    .center-pane {
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: var(--bg-base);
    }
    .viewport-container {
      height: 420px;
      position: relative;
      background: #08090e;
      border-bottom: 1px solid var(--border);
      transition: height 0.25s ease;
    }
    .viewport-container.expanded {
      height: 600px;
    }
    #three-canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .viewport-overlay-top {
      position: absolute;
      top: 10px;
      left: 12px;
      display: flex;
      gap: 8px;
      align-items: center;
      z-index: 10;
    }
    .viewport-overlay-controls {
      position: absolute;
      top: 10px;
      right: 12px;
      display: flex;
      gap: 6px;
      z-index: 10;
    }
    .telemetry-pane {
      flex: 1;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      overflow: hidden;
    }
    .progress-bar-container {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .progress-track {
      height: 8px;
      background: var(--bg-base);
      border-radius: 4px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
      transition: width 0.25s ease;
    }
    .table-container {
      flex: 1;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow-y: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
    }
    th {
      position: sticky;
      top: 0;
      background: var(--bg-card);
      padding: 8px 10px;
      text-align: left;
      font-weight: 700;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
    }
    td {
      padding: 7px 10px;
      border-bottom: 1px solid rgba(40, 44, 71, 0.4);
      color: var(--text-main);
    }
    tr.best-row {
      background: rgba(245, 158, 11, 0.15) !important;
      font-weight: 700;
      border-left: 3px solid var(--accent-amber);
    }
    tr:hover { background: var(--bg-card); }

    /* Right Pane: Sweet Spot Hero Card, Plots & Tomo Video Studio */
    .right-pane {
      background: var(--bg-surface);
      border-left: 1px solid var(--border);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      overflow-y: auto;
    }
    .sweetspot-card {
      background: linear-gradient(145deg, #1f2238, #181a2c);
      border: 1px solid rgba(245, 158, 11, 0.4);
      box-shadow: var(--gold-glow);
      border-radius: 10px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .metric-box {
      background: var(--bg-base);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 10px;
    }
    .metric-box span {
      font-size: 10px;
      color: var(--text-muted);
      display: block;
    }
    .metric-box strong {
      font-size: 15px;
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent-cyan);
    }
    .chart-container {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      height: 160px;
    }
    .video-studio-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    video {
      width: 100%;
      height: 190px;
      background: #000;
      border-radius: 6px;
      object-fit: contain;
    }
  </style>
</head>
<body>

  <!-- Top Header -->
  <header>
    <div class="brand">
      <div class="brand-logo">V</div>
      <div>
        <h1>OPENCAL VAM STUDIO & TOMO SLICER</h1>
        <p>Single-Model Sweet-Spot Optimization & 1:1 Hardware Projection Video Generator</p>
      </div>
    </div>
    <div class="header-actions">
      <span id="hw-badge" class="badge badge-primary">CUDA GPU ACTIVE</span>
      <button class="btn btn-secondary" style="width: auto; padding: 6px 12px;" onclick="openCurrentRunFolder()">📁 Open Run Folder</button>
    </div>
  </header>

  <!-- Application Body -->
  <div class="app-body">

    <!-- Left Column: CAD Placement & Physical Settings -->
    <div class="sidebar">
      <div class="section-title">
        <span>1. Target CAD Placement</span>
        <span id="fit-badge" class="badge badge-success">Inside Safe Boundary ✓</span>
      </div>
      <div class="control-group">
        <label>Target Model</label>
        <select id="stl-select" onchange="onModelChange()">
          <option value="cube">Preset: Solid Cube (17.7mm)</option>
          <option value="benchy">Preset: 3DBenchy (Tilt 35° Recommended)</option>
          <option value="captiveRing">Preset: Captive Ring</option>
          <option value="sphere">Preset: UV Sphere</option>
          <option value="custom">-- Custom Uploaded STL --</option>
        </select>
        <input type="file" id="stl-upload" style="display: none;" accept=".stl" onchange="handleFileUpload(event)">
        <button class="btn btn-secondary" onclick="document.getElementById('stl-upload').click()">📂 Upload Custom STL</button>

        <div style="display: flex; justify-content: space-between; align-items: center;">
          <label style="margin: 0;">Scale Factor</label>
          <button class="btn btn-cyan btn-sm" style="width: auto;" onclick="autoFitToVial()">⚡ Auto-Fit to Vial</button>
        </div>
        <input type="number" id="scale-input" value="1.0" step="0.01" oninput="updateModelTransform()">

        <div id="dims-display" style="font-size: 11px; color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; padding: 2px 0;">
          Physical Size: -- x -- x -- mm
        </div>

        <label>Rotation (X, Y, Z degrees)</label>
        <div class="input-row">
          <input type="number" id="rx" value="0" step="5" placeholder="Rx" oninput="updateModelTransform()">
          <input type="number" id="ry" value="0" step="5" placeholder="Ry" oninput="updateModelTransform()">
          <input type="number" id="rz" value="0" step="5" placeholder="Rz" oninput="updateModelTransform()">
        </div>
        <div class="btn-row">
          <button class="btn btn-secondary btn-sm" onclick="setTilt(35, 0, 0)">Tilt 35° (Benchy)</button>
          <button class="btn btn-secondary btn-sm" onclick="resetTransforms()">Reset 0,0,0</button>
        </div>

        <label>Translation (X, Y, Z mm)</label>
        <div class="input-row">
          <input type="number" id="tx" value="0" step="0.5" placeholder="Tx" oninput="updateModelTransform()">
          <input type="number" id="ty" value="0" step="0.5" placeholder="Ty" oninput="updateModelTransform()">
          <input type="number" id="tz" value="0" step="0.5" placeholder="Tz" oninput="updateModelTransform()">
        </div>
      </div>

      <div class="section-title">2. Hardware Optics & Tuning</div>
      <div class="control-group">
        <label>Projector Optics Profile</label>
        <select id="projector-profile">
          <option value="opencal2">OpenCAL V2 (1080x1920 @ 79.7µm Pitch, 108mm FOV)</option>
        </select>

        <label>Vial Rotation Speed (RPM)</label>
        <input type="number" id="rpm-input" value="9.0" step="0.5">

        <label>Print Exposure Duration (s)</label>
        <input type="number" id="duration-input" value="60" step="10">

        <label>Video Output Orientation</label>
        <select id="video-rotation">
          <option value="0">Portrait 1080x1920 (OpenCAL Display Rotated 90°)</option>
          <option value="90">Landscape 1920x1080 (Pre-Rotated 90° CW / Sideways Mount)</option>
          <option value="270">Landscape 1920x1080 (Pre-Rotated 270° CCW)</option>
        </select>

        <label>Tuning Exploration Strategy</label>
        <select id="tuning-mode">
          <option value="quick">⚡ Quick Sweet-Spot Search (12 Trials)</option>
          <option value="deep">🔬 Deep Hyperparameter Grid (32 Trials)</option>
        </select>

        <div style="margin: 8px 0 12px 0; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 6px; padding: 8px 10px;">
          <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; font-size: 12px; color: #10b981; text-transform: none; letter-spacing: 0; font-weight: 600;">
            <input type="checkbox" id="vial-correction-toggle" onchange="onVialCorrectionToggle()" style="width: auto; margin: 0; cursor: pointer;">
            <span>Cylindrical Vial Refraction (Fan-Beam)</span>
          </label>
          <div id="vial-correction-desc" style="font-size: 10.5px; color: var(--text-muted); margin-top: 4px; line-height: 1.35;">
            Parallel beam mode (Ø27mm build area). Check to enable Snell refraction correction for bare cylindrical vials.
          </div>
        </div>

        <div style="display: flex; gap: 8px;">
          <button id="start-btn" class="btn btn-primary" style="flex: 2;" onclick="startOptimization()">🚀 Find Sweet Spot</button>
          <button id="stop-btn" class="btn btn-danger" style="flex: 1; display: none;" onclick="stopOptimization()">🛑 Stop</button>
        </div>
      </div>
    </div>

    <!-- Center Column: 3D Viewport & Live Stream Table -->
    <div class="center-pane">
      <div id="viewport-container" class="viewport-container">
        <div class="viewport-overlay-top">
          <span class="badge badge-primary">Glass Vial (Ø30mm)</span>
          <span class="badge badge-success">Green Safe Zone (Ø25x50mm)</span>
        </div>
        <div class="viewport-overlay-controls">
          <button class="btn btn-secondary btn-sm" style="width: auto;" onclick="setCameraView('iso')">⟲ 3D</button>
          <button class="btn btn-secondary btn-sm" style="width: auto;" onclick="setCameraView('top')">⬆ Top</button>
          <button class="btn btn-secondary btn-sm" style="width: auto;" onclick="setCameraView('front')">➡ Front</button>
          <button class="btn btn-secondary btn-sm" style="width: auto;" onclick="toggleViewportExpand()">⤢ Resize View</button>
        </div>
        <div id="three-canvas"></div>
      </div>

      <div class="telemetry-pane">
        <div class="progress-bar-container">
          <div style="display: flex; justify-content: space-between; font-size: 11px;">
            <span id="progress-status" style="font-weight: 600;">System Ready. Select model or click "Find Sweet Spot".</span>
            <span id="progress-pct" style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--accent-cyan);">0%</span>
          </div>
          <div class="progress-track">
            <div id="progress-fill" class="progress-fill"></div>
          </div>
        </div>

        <div class="section-title" style="margin-bottom: 0;">Live On-The-Fly Trial Telemetry</div>
        <div class="table-container">
          <table id="telemetry-table">
            <thead>
              <tr>
                <th>Trial</th>
                <th>Method</th>
                <th>Angles</th>
                <th>Iter</th>
                <th>Filter</th>
                <th>Void Ceiling (d_L)</th>
                <th>Gel Floor (d_H)</th>
                <th>PW (Window)</th>
                <th>VER (%)</th>
                <th>Time (s)</th>
                <th>Champion Badge</th>
              </tr>
            </thead>
            <tbody id="telemetry-tbody">
              <tr>
                <td colspan="11" style="text-align: center; color: var(--text-dim); padding: 24px;">No optimization run active. Click "Find Sweet Spot" to stream trials.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Right Column: Champion Sweet Spot & 1:1 Tomo Video Maker -->
    <div class="right-pane">
      <div class="section-title">Champion Sweet Spot Result</div>
      <div class="sweetspot-card" id="sweetspot-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <strong style="color: var(--accent-amber); font-size: 13px;">★ OPTIMAL PRINT RECIPE</strong>
          <span id="best-trial-tag" class="badge badge-gold">Trial #--</span>
        </div>
        <div class="metric-grid">
          <div class="metric-box">
            <span>Process Window (PW)</span>
            <strong id="best-pw" style="color: var(--accent-green);">--</strong>
          </div>
          <div class="metric-box">
            <span>Voxel Error (VER)</span>
            <strong id="best-ver">--%</strong>
          </div>
          <div class="metric-box">
            <span>Algorithm & Filter</span>
            <strong id="best-method" style="font-size: 12px; color: var(--text-main);">--</strong>
          </div>
          <div class="metric-box">
            <span>Dose Thresholds (d_L → d_H)</span>
            <strong id="best-margins" style="font-size: 12px; color: var(--text-main);">--</strong>
          </div>
        </div>
      </div>

      <div class="section-title">Dose Distribution Histogram</div>
      <div class="chart-container">
        <canvas id="dose-histogram-chart"></canvas>
      </div>

      <div class="section-title">1:1 Tomo Video Studio</div>
      <div class="video-studio-card">
        <video id="projection-video" controls autoplay loop muted></video>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted);">
          <span>Resolution: <strong>1080x1920</strong></span>
          <span>FPS: <strong>54.0</strong></span>
          <span>Format: <strong>MP4 (H.264)</strong></span>
        </div>
        <button id="download-video-btn" class="btn btn-success" onclick="downloadCurrentVideo()" disabled>💾 Download Projection MP4</button>
      </div>
    </div>

  </div>

  <!-- Javascript Logic -->
  <script>
    let scene, camera, renderer, controls, currentMesh, rawGeometry;
    let outerVialMesh, innerSafeVialMesh, aorLine;
    let doseChartInstance = null;
    let currentUploadedPath = null;
    let currentRunFolder = null;
    let activeVideoUrl = null;
    let activeChampionTrialIdx = -1;

    // Physical Limits
    let greenSafeGroup = null;
    function getActiveSafeLimits() {
      const isVialCorrection = document.getElementById("vial-correction-toggle") ? document.getElementById("vial-correction-toggle").checked : false;
      return {
        r: isVialCorrection ? 9.93 : 13.5, // 19.87mm refraction limit vs 27mm physical vial limit
        diam: isVialCorrection ? 19.87 : 27.0,
        h: 50.0
      };
    }

    function buildGreenSafeZone() {
      if (greenSafeGroup) scene.remove(greenSafeGroup);
      greenSafeGroup = new THREE.Group();

      const limits = getActiveSafeLimits();
      const R = limits.r;
      const H = limits.h;

      // 1. Wireframe green cylinder
      const innerGeo = new THREE.CylinderGeometry(R, R, H, 36, 6, true);
      const innerMat = new THREE.MeshBasicMaterial({
        color: 0x10b981, wireframe: true, transparent: true, opacity: 0.65
      });
      greenSafeGroup.add(new THREE.Mesh(innerGeo, innerMat));

      // 2. Inner translucent volume
      const innerFillMat = new THREE.MeshBasicMaterial({
        color: 0x10b981, transparent: true, opacity: 0.08, side: THREE.DoubleSide
      });
      greenSafeGroup.add(new THREE.Mesh(innerGeo, innerFillMat));

      // 3. Top and bottom safety rings
      const ringGeo = new THREE.RingGeometry(R - 0.35, R, 36);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x10b981, side: THREE.DoubleSide });
      const topRing = new THREE.Mesh(ringGeo, ringMat);
      topRing.rotation.x = Math.PI / 2;
      topRing.position.y = H / 2;
      greenSafeGroup.add(topRing);

      const botRing = new THREE.Mesh(ringGeo, ringMat);
      botRing.rotation.x = Math.PI / 2;
      botRing.position.y = -H / 2;
      greenSafeGroup.add(botRing);

      scene.add(greenSafeGroup);
    }

    function onVialCorrectionToggle() {
      buildGreenSafeZone();
      const isVialCorrection = document.getElementById("vial-correction-toggle").checked;
      const desc = document.getElementById("vial-correction-desc");
      if (desc) {
        if (isVialCorrection) {
          desc.innerHTML = "<b>Fan-beam refraction ON</b>: Usable region shrunk to green inner cylinder (radius \u2248 9.9mm / \u00d819.9mm). Keep part inside it.";
          desc.style.color = "#10b981";
        } else {
          desc.innerHTML = "<b>Parallel beam mode</b> (\u00d827mm safe build area). Standard 1:1 projection matching pure VAMToolbox & Tomo.";
          desc.style.color = "var(--text-muted)";
        }
      }
      autoFitToVial();
    }

    // Initialize 3D Viewport
    function initThree() {
      const container = document.getElementById("three-canvas");
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x08090e);

      camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
      camera.position.set(0, 45, 95);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;

      // Lighting
      scene.add(new THREE.AmbientLight(0xffffff, 0.75));
      const keyLight = new THREE.DirectionalLight(0x00e5ff, 0.85);
      keyLight.position.set(50, 80, 50);
      scene.add(keyLight);
      const fillLight = new THREE.DirectionalLight(0x3b82f6, 0.5);
      fillLight.position.set(-50, -30, -50);
      scene.add(fillLight);

      // 1. Outer Glass Resin Vial (Ø30mm x 60mm) - Blue Translucent
      const outerGeo = new THREE.CylinderGeometry(15, 15, 60, 36, 1, true);
      const outerMat = new THREE.MeshStandardMaterial({
        color: 0x3b82f6, transparent: true, opacity: 0.15, roughness: 0.1, metalness: 0.8, side: THREE.DoubleSide
      });
      outerVialMesh = new THREE.Mesh(outerGeo, outerMat);
      scene.add(outerVialMesh);

      const outerEdgeGeo = new THREE.EdgesGeometry(outerGeo);
      const outerEdgeMat = new THREE.LineBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.45 });
      scene.add(new THREE.LineSegments(outerEdgeGeo, outerEdgeMat));

      // 2. Dynamic Green Safe Zone (reactive to vial correction toggle)
      buildGreenSafeZone();

      // 3. Axis of Rotation (AoR) Centerline
      const lineMat = new THREE.LineDashedMaterial({ color: 0x06b6d4, dashSize: 2, gapSize: 1.5, transparent: true, opacity: 0.7 });
      const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -35, 0), new THREE.Vector3(0, 35, 0)]);
      aorLine = new THREE.Line(lineGeo, lineMat);
      aorLine.computeLineDistances();
      scene.add(aorLine);

      // 4. Floor Grid
      const grid = new THREE.GridHelper(60, 12, 0x1e293b, 0x0f172a);
      grid.position.y = -30;
      scene.add(grid);

      loadModelMesh("cube");

      window.addEventListener("resize", onWindowResize);

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }
      animate();
    }

    function onWindowResize() {
      const container = document.getElementById("three-canvas");
      if (!container || !renderer || !camera) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }

    function toggleViewportExpand() {
      const vp = document.getElementById("viewport-container");
      vp.classList.toggle("expanded");
      setTimeout(onWindowResize, 260);
    }

    function setCameraView(view) {
      if (view === "top") {
        camera.position.set(0, 110, 0);
        controls.target.set(0, 0, 0);
      } else if (view === "front") {
        camera.position.set(0, 0, 110);
        controls.target.set(0, 0, 0);
      } else {
        camera.position.set(0, 45, 95);
        controls.target.set(0, 0, 0);
      }
      controls.update();
    }

    function loadModelMesh(modelKey) {
      if (currentMesh) scene.remove(currentMesh);
      const loader = new THREE.STLLoader();
      const url = `/api/get_stl?name=${modelKey}` + (currentUploadedPath ? `&custom_path=${encodeURIComponent(currentUploadedPath)}` : "");
      
      loader.load(url, (geo) => {
        // Standard CAD STLs are Z-up. Rotate by -90 deg on X so they stand upright along Three.js Y-axis
        geo.rotateX(-Math.PI / 2);
        geo.center();
        geo.computeBoundingBox();
        rawGeometry = geo;

        const mat = new THREE.MeshStandardMaterial({
          color: 0x06b6d4, roughness: 0.35, metalness: 0.25, side: THREE.DoubleSide
        });
        currentMesh = new THREE.Mesh(geo, mat);
        scene.add(currentMesh);

        // If Benchy, automatically preset standard 35 degree printing tilt
        if (modelKey === "benchy") {
          document.getElementById("rx").value = "35";
          document.getElementById("ry").value = "0";
          document.getElementById("rz").value = "0";
        } else {
          document.getElementById("rx").value = "0";
          document.getElementById("ry").value = "0";
          document.getElementById("rz").value = "0";
        }

        // Auto fit on initial model load
        autoFitToVial();
      });
    }

    function autoFitToVial() {
      if (!rawGeometry) return;

      const rx = THREE.MathUtils.degToRad(parseFloat(document.getElementById("rx").value) || 0);
      const ry = THREE.MathUtils.degToRad(parseFloat(document.getElementById("ry").value) || 0);
      const rz = THREE.MathUtils.degToRad(parseFloat(document.getElementById("rz").value) || 0);

      // Clone geometry and apply current Euler rotations to get the true rotated bounding box
      const tempGeo = rawGeometry.clone();
      tempGeo.rotateX(rx);
      tempGeo.rotateY(ry);
      tempGeo.rotateZ(rz);
      tempGeo.computeBoundingBox();
      const bb = tempGeo.boundingBox;
      
      // Rotated extents: X is width, Y is height along vial axis, Z is depth
      const dx = Math.max(bb.max.x - bb.min.x, 1e-3);
      const dy = Math.max(bb.max.y - bb.min.y, 1e-3);
      const dz = Math.max(bb.max.z - bb.min.z, 1e-3);

      // True rotational cylinder diameter in horizontal plane (X-Z)
      const diagonal_diam = Math.sqrt(dx * dx + dz * dz);
      const height = dy;

      // Fit within dynamic green safe printable boundary (19.87mm refraction limit vs 27mm physical limit)
      const limits = getActiveSafeLimits();
      const targetScale = Math.min((limits.diam / diagonal_diam), (limits.h / height)) * 0.92;

      document.getElementById("scale-input").value = targetScale.toFixed(4);
      document.getElementById("tx").value = "0";
      document.getElementById("ty").value = "0";
      document.getElementById("tz").value = "0";

      updateModelTransform();
    }

    function resetTransforms() {
      document.getElementById("rx").value = "0";
      document.getElementById("ry").value = "0";
      document.getElementById("rz").value = "0";
      document.getElementById("tx").value = "0";
      document.getElementById("ty").value = "0";
      document.getElementById("tz").value = "0";
      autoFitToVial();
    }

    function setTilt(rx, ry, rz) {
      document.getElementById("rx").value = rx;
      document.getElementById("ry").value = ry;
      document.getElementById("rz").value = rz;
      autoFitToVial();
    }

    function updateModelTransform() {
      if (!currentMesh || !rawGeometry) return;
      const s = parseFloat(document.getElementById("scale-input").value) || 1.0;
      const rx = THREE.MathUtils.degToRad(parseFloat(document.getElementById("rx").value) || 0);
      const ry = THREE.MathUtils.degToRad(parseFloat(document.getElementById("ry").value) || 0);
      const rz = THREE.MathUtils.degToRad(parseFloat(document.getElementById("rz").value) || 0);
      const tx = parseFloat(document.getElementById("tx").value) || 0;
      const ty = parseFloat(document.getElementById("ty").value) || 0;
      const tz = parseFloat(document.getElementById("tz").value) || 0;

      currentMesh.scale.set(s, s, s);
      currentMesh.rotation.set(rx, ry, rz);
      currentMesh.position.set(tx, ty, tz);

      // Compute rotated physical extents
      const tempGeo = rawGeometry.clone();
      tempGeo.rotateX(rx);
      tempGeo.rotateY(ry);
      tempGeo.rotateZ(rz);
      tempGeo.computeBoundingBox();
      const bb = tempGeo.boundingBox;

      const sx = bb.max.x - bb.min.x;
      const sy = bb.max.y - bb.min.y;
      const sz = bb.max.z - bb.min.z;
      const px = (sx * s).toFixed(1);
      const py = (sy * s).toFixed(1);
      const pz = (sz * s).toFixed(1);

      document.getElementById("dims-display").innerText = `Physical Size: ${px} x ${py} x ${pz} mm`;

      // True rotational cylinder radius: half-diagonal + translational offset from center
      const half_diagonal = Math.sqrt(Math.pow((sx * s) / 2.0, 2) + Math.pow((sz * s) / 2.0, 2));
      const trans_offset_rad = Math.sqrt(tx * tx + tz * tz);
      const max_rad = half_diagonal + trans_offset_rad;
      const max_h = (sy * s) + Math.abs(ty);

      const is_inside = (max_rad <= MAX_PRINT_R) && (max_h <= MAX_PRINT_H);

      const badge = document.getElementById("fit-badge");
      if (is_inside) {
        badge.className = "badge badge-success";
        badge.innerText = "Inside Safe Boundary ✓";
        currentMesh.material.color.setHex(0x06b6d4);
      } else {
        badge.className = "badge badge-danger";
        badge.innerText = `Exceeds Boundary ⚠ (Ø${(max_rad * 2).toFixed(1)} > 25mm)`;
        currentMesh.material.color.setHex(0xf43f5e);
      }
    }

    function onModelChange() {
      const val = document.getElementById("stl-select").value;
      if (val === "custom") {
        document.getElementById("stl-upload").click();
      } else {
        currentUploadedPath = null;
        loadModelMesh(val);
      }
    }

    function handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append("stl", file);

      fetch("/api/upload_stl", { method: "POST", body: formData })
        .then(r => r.json())
        .then(d => {
          if (d.status === "ok") {
            currentUploadedPath = d.path;
            loadModelMesh("custom");
          }
        });
    }

    // Initialize Chart
    function initDoseChart() {
      const ctx = document.getElementById("dose-histogram-chart").getContext("2d");
      doseChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels: Array.from({ length: 64 }, (_, i) => (i / 64).toFixed(2)),
          datasets: [
            { label: "In-Part Dose (%)", data: [], backgroundColor: "rgba(59, 130, 246, 0.75)", barPercentage: 1.0, categoryPercentage: 1.0 },
            { label: "Out-Part Dose (%)", data: [], backgroundColor: "rgba(244, 63, 94, 0.75)", barPercentage: 1.0, categoryPercentage: 1.0 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { 
              display: true,
              ticks: { color: "#8b92b2", font: { size: 8 }, maxTicksLimit: 8, callback: function(val, idx) { return (idx / 64).toFixed(1); } },
              grid: { display: false }
            },
            y: { 
              ticks: { 
                color: "#8b92b2", 
                font: { size: 9 },
                callback: function(value) { return value + "%"; }
              }, 
              grid: { color: "#282c47" } 
            }
          },
          plugins: {
            legend: { labels: { color: "#f0f2fc", font: { size: 10 } } },
            tooltip: {
              callbacks: {
                label: function(context) { return `${context.dataset.label}: ${context.parsed.y}%`; }
              }
            }
          }
        }
      });
    }

    // Start Optimization & Stream Telemetry
    function startOptimization() {
      const btn = document.getElementById("start-btn");
      const stopBtn = document.getElementById("stop-btn");
      btn.disabled = true;
      btn.innerHTML = "⏳ Optimizing...";
      stopBtn.style.display = "flex";

      const payload = {
        stl_name: document.getElementById("stl-select").value,
        custom_path: currentUploadedPath,
        scale: parseFloat(document.getElementById("scale-input").value) || 1.0,
        rx: parseFloat(document.getElementById("rx").value) || 0,
        ry: parseFloat(document.getElementById("ry").value) || 0,
        rz: parseFloat(document.getElementById("rz").value) || 0,
        tx: parseFloat(document.getElementById("tx").value) || 0,
        ty: parseFloat(document.getElementById("ty").value) || 0,
        tz: parseFloat(document.getElementById("tz").value) || 0,
        rpm: parseFloat(document.getElementById("rpm-input").value) || 9.0,
        duration: parseFloat(document.getElementById("duration-input").value) || 60.0,
        tuning_mode: document.getElementById("tuning-mode").value,
        video_rotation_deg: parseFloat(document.getElementById("video-rotation").value) || 0.0,
        vial_correction: document.getElementById("vial-correction-toggle") ? document.getElementById("vial-correction-toggle").checked : false,
      };

      document.getElementById("telemetry-tbody").innerHTML = "";
      activeChampionTrialIdx = -1;

      fetch("/api/start_optimization", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(() => {
        listenToTelemetryStream();
      });
    }

    function stopOptimization() {
      const stopBtn = document.getElementById("stop-btn");
      stopBtn.disabled = true;
      stopBtn.innerHTML = "Stopping...";
      fetch("/api/stop_optimization", { method: "POST" });
    }

    function listenToTelemetryStream() {
      const evtSource = new EventSource("/api/optimization_stream");
      evtSource.onmessage = (e) => {
        const data = JSON.parse(e.data);

        if (data.event === "started") {
          currentRunFolder = data.run_folder;
          document.getElementById("progress-status").innerText = `Optimizing: ${data.stl_name} -> ${data.run_name}`;
        }
        else if (data.event === "voxelized") {
          document.getElementById("progress-status").innerText = `Voxelized (${data.grid_shape.join("x")}) in ${data.voxelize_time_s}s`;
        }
        else if (data.event === "trial_complete") {
          const pct = Math.round((data.trial_idx / data.total_trials) * 100);
          document.getElementById("progress-pct").innerText = `${pct}%`;
          document.getElementById("progress-fill").style.width = `${pct}%`;
          document.getElementById("progress-status").innerText = `Trial [${data.trial_idx}/${data.total_trials}] | Current Best Score: ${data.best_score}`;

          // Append to table with strictly ONE champion badge
          appendTelemetryRow(data.trial_record, data.is_new_best);

          // Update Dose Histogram
          if (data.metrics && doseChartInstance) {
            doseChartInstance.data.datasets[0].data = data.metrics.in_hist;
            doseChartInstance.data.datasets[1].data = data.metrics.out_hist;
            doseChartInstance.update();
          }

          // Update Sweet Spot Card if champion
          if (data.is_new_best) {
            updateSweetSpotCard(data.trial_record);
          }
        }
        else if (data.event === "finished") {
          evtSource.close();
          const btn = document.getElementById("start-btn");
          const stopBtn = document.getElementById("stop-btn");
          btn.disabled = false;
          btn.innerHTML = "🚀 Find Sweet Spot";
          stopBtn.style.display = "none";
          stopBtn.disabled = false;
          stopBtn.innerHTML = "🛑 Stop";

          const statusMsg = data.stopped_early ? "Optimization Stopped early! Sweet Spot Video Generated." : "Optimization Complete! Video Generated.";
          document.getElementById("progress-status").innerText = statusMsg;

          // Load video
          if (data.video_file) {
            activeVideoUrl = `/api/video?path=${encodeURIComponent(data.video_file)}`;
            const vid = document.getElementById("projection-video");
            vid.src = activeVideoUrl;
            vid.load();
            vid.play();
            document.getElementById("download-video-btn").disabled = false;
          }
        }
      };
    }

    function appendTelemetryRow(rec, isBest) {
      const tbody = document.getElementById("telemetry-tbody");

      // If this is a new best, remove champion highlight from previous champion row
      if (isBest) {
        document.querySelectorAll("#telemetry-tbody tr.best-row").forEach(r => {
          r.classList.remove("best-row");
          const badgeCell = r.querySelector(".badge-gold");
          if (badgeCell) {
            badgeCell.className = "badge badge-primary";
            badgeCell.innerText = "OK";
          }
        });
        activeChampionTrialIdx = rec.trial_idx;
      }

      const tr = document.createElement("tr");
      tr.id = `trial-row-${rec.trial_idx}`;
      if (isBest) tr.className = "best-row";

      const pwColor = rec.window >= 0 ? "var(--accent-green)" : "var(--accent-rose)";
      const badgeHtml = isBest ? `<span class="badge badge-gold">★ SWEET SPOT</span>` : `<span class="badge badge-primary">OK</span>`;

      tr.innerHTML = `
        <td>#${rec.trial_idx}</td>
        <td><strong>${rec.method}</strong></td>
        <td>${rec.n_angles}°</td>
        <td>${rec.n_iter}</td>
        <td>${rec.filter}</td>
        <td>${rec.d_l.toFixed(2)}</td>
        <td>${rec.d_h.toFixed(2)}</td>
        <td style="color: ${pwColor}; font-weight: 700;">${rec.window >= 0 ? "+" : ""}${rec.window.toFixed(3)}</td>
        <td>${rec.ver_pct.toFixed(1)}%</td>
        <td>${rec.opt_time_s}s</td>
        <td>${badgeHtml}</td>
      `;
      tbody.appendChild(tr);
      tr.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function updateSweetSpotCard(rec) {
      document.getElementById("best-trial-tag").innerText = `Trial #${rec.trial_idx}`;
      document.getElementById("best-pw").innerText = (rec.window >= 0 ? "+" : "") + rec.window.toFixed(3);
      document.getElementById("best-ver").innerText = `${rec.ver_pct.toFixed(1)}%`;
      document.getElementById("best-method").innerText = `${rec.method} (${rec.n_iter} iters, ${rec.filter})`;
      document.getElementById("best-margins").innerText = `Void: ${rec.d_l.toFixed(2)} → Gel: ${rec.d_h.toFixed(2)}`;
    }

    function openCurrentRunFolder() {
      fetch("/api/open_folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder: currentRunFolder })
      });
    }

    function downloadCurrentVideo() {
      if (activeVideoUrl) {
        window.open(activeVideoUrl, "_blank");
      }
    }

    window.onload = () => {
      initThree();
      initDoseChart();
    };
  </script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# FLASK BACKEND REST ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(GUI_HTML)


@app.route("/api/get_stl")
def get_stl():
    name = request.args.get("name", "cube")
    custom_path = request.args.get("custom_path", None)

    if custom_path and os.path.exists(custom_path):
        return send_file(custom_path, mimetype="application/octet-stream")
    if name in PRESET_STLS and os.path.exists(PRESET_STLS[name]):
        return send_file(PRESET_STLS[name], mimetype="application/octet-stream")
    return send_file(PRESET_STLS["cube"], mimetype="application/octet-stream")


@app.route("/api/upload_stl", methods=["POST"])
def upload_stl():
    if "stl" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    file = request.files["stl"]
    save_path = os.path.join(UPLOADS_DIR, file.filename)
    file.save(save_path)
    return jsonify({"status": "ok", "path": save_path, "filename": file.filename})


@app.route("/api/start_optimization", methods=["POST"])
def start_optimization():
    data = request.json or {}
    stl_name = data.get("stl_name", "cube")
    custom_path = data.get("custom_path", None)

    if custom_path and os.path.exists(custom_path):
        stl_path = custom_path
    elif stl_name in PRESET_STLS:
        stl_path = PRESET_STLS[stl_name]
    else:
        stl_path = PRESET_STLS["cube"]

    scale = data.get("scale", 1.0)
    rx = float(data.get("rx", 0.0))
    ry = float(data.get("ry", 0.0))
    rz = float(data.get("rz", 0.0))
    tx = float(data.get("tx", 0.0))
    ty = float(data.get("ty", 0.0))
    tz = float(data.get("tz", 0.0))
    rpm = float(data.get("rpm", DEFAULT_RPM))
    duration = float(data.get("duration", DEFAULT_DURATION_S))
    tuning_mode = str(data.get("tuning_mode", "quick"))
    video_rotation_deg = float(data.get("video_rotation_deg", 0.0))
    vial_correction = bool(data.get("vial_correction", False))

    # Reset stop flag and empty queue
    _STOP_EVENT.clear()
    while not _TELEMETRY_QUEUE.empty():
        try: _TELEMETRY_QUEUE.get_nowait()
        except Exception: break

    def _worker():
        _CURRENT_RUN_STATE["is_running"] = True
        try:
            for event in run_single_model_optimization_stream(
                stl_path=stl_path,
                stl_name=stl_name,
                scale=scale,
                rx=rx, ry=ry, rz=rz,
                tx=tx, ty=ty, tz=tz,
                search_mode=tuning_mode,
                rpm=rpm,
                duration_s=duration,
                video_rotation_deg=video_rotation_deg,
                vial_correction=vial_correction,
                stop_checker=_STOP_EVENT
            ):
                _TELEMETRY_QUEUE.put(sanitize_val(event))
                if event.get("event") == "finished":
                    _CURRENT_RUN_STATE["current_run_dir"] = event.get("run_folder")
                    _CURRENT_RUN_STATE["active_video_path"] = event.get("video_file")
                    try:
                        generate_single_model_charts(
                            trials_csv_path=event.get("csv_file"),
                            output_dir=os.path.join(event.get("run_folder"), "charts"),
                            stl_name=stl_name
                        )
                    except Exception as ce:
                        print(f"[ChartGen Error] {ce}")
        except Exception as e:
            print(f"[Optimizer Worker Error] {e}")
            _TELEMETRY_QUEUE.put(sanitize_val({
                "event": "finished",
                "stopped_early": True,
                "error": str(e),
                "run_folder": _CURRENT_RUN_STATE.get("current_run_dir"),
                "video_file": _CURRENT_RUN_STATE.get("active_video_path"),
            }))
        finally:
            _CURRENT_RUN_STATE["is_running"] = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/stop_optimization", methods=["POST"])
def stop_optimization():
    _STOP_EVENT.set()
    return jsonify({"status": "stopping"})


@app.route("/api/optimization_stream")
def optimization_stream():
    def _stream():
        while True:
            try:
                event = _TELEMETRY_QUEUE.get(timeout=1.0)
                clean_event = sanitize_val(event)
                yield f"data: {json.dumps(clean_event, default=str)}\n\n"
                if clean_event.get("event") == "finished":
                    break
            except queue.Empty:
                if not _CURRENT_RUN_STATE["is_running"]:
                    break
                yield ": keepalive\n\n"
    return Response(_stream(), mimetype="text/event-stream")


@app.route("/api/video")
def get_video():
    path = request.args.get("path", "")
    if path and os.path.exists(path):
        return send_file(path, mimetype="video/mp4")
    return jsonify({"status": "error", "message": "Video not found"}), 404


@app.route("/api/open_folder", methods=["POST"])
def open_folder():
    data = request.json or {}
    folder = data.get("folder") or _CURRENT_RUN_STATE.get("current_run_dir") or ROOT_RUNS_DIR
    if os.path.exists(folder):
        if sys.platform == "win32":
            subprocess.run(["explorer", os.path.abspath(folder)], check=False)
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Folder not found"}), 404


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT (Pywebview Native App + Local Flask)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    port = 5050
    print("=" * 70)
    print("  OPENCAL 3D VAM SINGLE-MODEL STUDIO & TOMO SLICER IS LIVE")
    print(f"  • Local Desktop URL : http://127.0.0.1:{port}")
    print(f"  • Root Runs Folder  : {ROOT_RUNS_DIR}")
    print("=" * 70)

    flask_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    time.sleep(1.0)

    try:
        import webview
        print("[GUI] Spawning native pywebview window ...")
        webview.create_window(
            title="OpenCAL 3D VAM Single-Model Studio & Tomo Slicer",
            url=f"http://127.0.0.1:{port}",
            width=1460,
            height=920,
            resizable=True
        )
        webview.start()
    except Exception as e:
        print(f"[GUI] pywebview not active ({e}); keeping local Flask server open.")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
