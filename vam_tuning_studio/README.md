# OpenCAL & VAM Parameter Tuning Studio

A self-contained engineering suite for exploring VAM optimization parameters, investigating metric sensitivity, running genetic-greedy searches across 4 benchmark geometries (`cube.stl`, `sphere.stl`, `captiveRing.stl`, `benchy.stl`), and visualizing results via an interactive **PyWebView GUI Dashboard**.

---

## 1. Directory Structure

```
vam_tuning_studio/
├── run_genetic_investigation.py    # Master exploration runner (atomic per-run isolation)
├── vam_tuning_gui.py               # Interactive desktop & web GUI studio (PyWebView)
├── generate_charts.py              # Publication-quality 2D heatmaps & Pareto charts
├── README.md                       # This documentation
└── runs/                           # Isolated run sessions
    └── run_YYYYMMDD_HHMMSS/        # Dedicated folder per session
        ├── status.json             # Live status (RUNNING / COMPLETED / CRASHED)
        ├── live_experiment_log.csv # Real-time per-trial log
        ├── live_experiment_log.jsonl
        ├── final_experiment_log.csv# Published only on clean completion
        ├── RUN_COMPLETE.json       # Manifest written ONLY when all geometries finish
        ├── charts/                 # Generated 2D heatmaps, radar & Pareto plots
        └── slices/                 # Target vs Best vs Baseline 2D slice images
```

---

## 2. Crash-Proof Atomic Per-Run Architecture

1. **Isolated Run Directory**: Every time you launch `run_genetic_investigation.py`, a new timestamped folder (`runs/run_YYYYMMDD_HHMMSS/`) is created.
2. **Crash & Abort Protection**: If a run is cancelled, interrupted, or crashes, it is tagged as `INCOMPLETE` / `CRASHED` in `status.json` and **no `RUN_COMPLETE.json` is created**. The GUI automatically ignores incomplete runs when displaying finalized conclusions.
3. **Clean Finalization**: When all 4 geometries complete all exploration phases, the suite:
   - Copies `live_experiment_log.csv` to `final_experiment_log.csv`.
   - Generates high-resolution heatmaps and charts into `charts/`.
   - Writes `RUN_COMPLETE.json` with global champions and summary metrics.
   - Marks status as `COMPLETED`.

---

## 3. How to Run

Make sure your terminal is using the project virtual environment (configured in `.venv` with Python 3.13):

### A. Run Parameter Exploration (Headless / Terminal)
```powershell
.\.venv\Scripts\python.exe vam_tuning_studio\run_genetic_investigation.py
```

### B. Launch Interactive GUI Studio (Desktop Webview & Browser)
```powershell
.\.venv\Scripts\python.exe vam_tuning_studio\vam_tuning_gui.py
```
* Spawns a native desktop window via `pywebview`.
* Also accessible in any web browser at: `http://127.0.0.1:5050`
* Features a **Run Session Selector** dropdown allowing you to switch between live active runs and historical completed runs.

---

## 4. Key Metrics Quick Reference

| Metric | Goal | Meaning |
| :--- | :--- | :--- |
| **Volumetric Error Rate ($VER$)** | $\to 0.0\%$ | Fraction of liquid bath receiving $\ge$ weakest part voxel dose. If $>0$, resin solidifies prematurely. |
| **Process Window ($PW$)** | $> 0.0$ | Dose gap between coldest part voxel and hottest stray void voxel. Positive $=$ clean print window. |
| **Coefficient of Variance ($CV$)** | $\to 0.0$ | Dose non-uniformity across the printed object. Lower $=$ less internal stress and warping. |
| **In-Part Dose Range ($IPDR$)** | Minimize | Peak-to-peak dose span inside the solid part. |
| **Dose Contrast Ratio** | Maximize | Mean optical signal-to-noise ratio between target object and surrounding bath. |
