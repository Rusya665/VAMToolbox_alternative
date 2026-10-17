import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PyQt6.QtCore import Qt, QTimer

class VAMVisualizer(QMainWindow):
    def __init__(self, num_steps=360, resolution=80):
        super().__init__()
        self.setWindowTitle("VAM Step-by-Step 3D Dose Visualizer (3DBenchy)")
        self.num_steps = num_steps
        self.resolution = resolution
        self.current_step = 0
        
        # Debounce timer for smooth 3D rendering (200ms delay of inactivity before redrawing 3D)
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_3d_plots)
        
        self.generate_mock_data()
        self.init_ui()
        self.update_plots_init()

    def generate_mock_data(self):
        import scipy.ndimage
        import vamtoolbox as vam

        res = self.resolution
        steps = self.num_steps
        stl_filename = "3dbenchy.stl"
        
        # Create a dedicated cache directory
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(stl_filename))[0]
        voxels_cache_path = os.path.join(cache_dir, f"{base_name}_{res}_voxels.npy")
        sino_opt_cache_path = os.path.join(cache_dir, f"{base_name}_{res}_{steps}_sino_opt.npy")
        
        # 1. Load or generate voxel volume
        if os.path.exists(voxels_cache_path):
            print(f"Loading voxelization from cache: {voxels_cache_path}...")
            self.target_volume = np.load(voxels_cache_path)
        else:
            # If no cache exists, voxelize using VAMToolbox and save to cache
            print(f"No cache found. Voxelizing {stl_filename} STL model...")
            target_geo = vam.geometry.TargetGeometry(
                stlfilename=stl_filename,
                resolution=res
            )
            self.target_volume = target_geo.array.astype(float)
            np.save(voxels_cache_path, self.target_volume)
            print(f"Saved voxelized array to cache: {voxels_cache_path}")
            
        ny, nx, nz = self.target_volume.shape
        print(f"Voxel shape: {self.target_volume.shape}")
        
        # Bounding cylindrical container (vial) centered in XY
        x, y, z = np.ogrid[:ny, :nx, :nz]
        center_y = ny // 2
        center_x = nx // 2
        
        # Explicitly broadcast vial mask to 3D shape (ny, nx, nz) to avoid shape mismatch issues
        vial_2d = (x - center_y)**2 + (y - center_x)**2 < (min(ny, nx) // 2.1)**2
        self.in_vial = np.zeros((ny, nx, nz), dtype=bool) | vial_2d
        
        # 2. Load or generate optimized CAL sinogram
        if os.path.exists(sino_opt_cache_path):
            print(f"Loading optimized sinogram from cache: {sino_opt_cache_path}...")
            self.projections = np.load(sino_opt_cache_path)
        else:
            print("No cached optimized sinogram found. Running OSMO optimization...")
            target_geo = vam.geometry.TargetGeometry(target=self.target_volume, resolution=nz)
            angles = np.linspace(0, 360 - 360 / steps, steps)
            proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)
            
            optimizer_params = vam.optimize.Options(
                method="OSMO",
                n_iter=15,
                d_h=0.85,
                d_l=0.60,
                filter="hamming",
                verbose=False
            )
            opt_sino, opt_recon, error = vam.optimize.optimize(target_geo, proj_geo, optimizer_params)
            self.projections = opt_sino.array
            np.save(sino_opt_cache_path, self.projections)
            print(f"Saved optimized sinogram to cache: {sino_opt_cache_path}")
            
        # 3. Precompute or load the 3D physical ray shadowing and relaxation dose history
        self.downsample = 4  # Unified downsampling factor for responsive 3D viewport
        target_down = self.target_volume[::self.downsample, ::self.downsample, ::self.downsample]
        self.in_vial_down = self.in_vial[::self.downsample, ::self.downsample, ::self.downsample]
        ny_d, nx_d, nz_d = target_down.shape
        
        dose_cache_path = os.path.join(cache_dir, f"{base_name}_{res}_{steps}_dose_hist.npy")
        light_cache_path = os.path.join(cache_dir, f"{base_name}_{res}_{steps}_light_hist.npy")
        shadow_cache_path = os.path.join(cache_dir, f"{base_name}_{res}_{steps}_shadow_hist.npy")
        
        if os.path.exists(dose_cache_path) and os.path.exists(light_cache_path) and os.path.exists(shadow_cache_path):
            print("Loading precomputed dose, light, and shadow history from cache...")
            self.dose_history = np.load(dose_cache_path)
            self.light_history = np.load(light_cache_path)
            self.shadow_history = np.load(shadow_cache_path)
        else:
            print("Precomputing physical light propagation and relaxation dose history using VAMToolbox...")
            self.dose_history = np.zeros((steps, ny_d, nx_d, nz_d))
            self.light_history = np.zeros((steps, ny_d, nx_d, nz_d))
            self.shadow_history = np.zeros((steps, ny_d, nx_d, nz_d), dtype=bool)
            
            # Construct geometries and projector for the downsampled simulation
            target_geo_down = vam.geometry.TargetGeometry(target=target_down, resolution=nz_d)
            angles = np.linspace(0, 360 - 360 / steps, steps)
            proj_geo_down = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)
            A_down = vam.projectorconstructor.projectorconstructor(target_geo_down, proj_geo_down)
            
            # Downsample the sinogram array (nX, steps, nZ) -> (nX_d, steps, nZ_d)
            sino_down = self.projections[::self.downsample, :, ::self.downsample]
            sino_single = np.zeros_like(sino_down)
            
            relaxation_rate = 0.012  # 1.2% chemical relaxation/decay per step
            current_dose = np.zeros((ny_d, nx_d, nz_d))
            
            # Warm up ASTRA projector context
            A_down.backward(sino_single)
            
            for t in range(steps):
                # 1. Backproject using VAMToolbox to compute the unshadowed 3D light intensity field
                sino_single[:, t, :] = sino_down[:, t, :]
                I_t = A_down.backward(sino_single)
                sino_single[:, t, :] = 0.0  # reset for next step
                
                # 2. Compute target shadow mask
                angle_deg = (t / steps) * 360
                target_rot = scipy.ndimage.rotate(target_down, -angle_deg, axes=(0, 1), reshape=False, order=0)
                cumsum_target = np.cumsum(target_rot, axis=0)
                shadow_rot = (cumsum_target - target_rot) > 0
                shadow_vial = scipy.ndimage.rotate(shadow_rot, angle_deg, axes=(0, 1), reshape=False, order=0) > 0.5
                shadow_vial = shadow_vial & self.in_vial_down
                
                # 3. Apply shadow to the backprojected light intensity
                I_t = np.where(shadow_vial, 0.015, I_t) * self.in_vial_down
                I_t = I_t * 0.035  # Scale for physical dose normalization
                
                # 4. Update dose: shadowed decays by 1.2%, illuminated increases by intensity
                is_relaxing = shadow_vial | (I_t < 0.005)
                current_dose = np.where(
                    is_relaxing,
                    current_dose * (1.0 - relaxation_rate),
                    current_dose + I_t
                )
                
                self.dose_history[t] = current_dose.copy()
                self.light_history[t] = I_t.copy()
                self.shadow_history[t] = shadow_vial.copy()
                
            np.save(dose_cache_path, self.dose_history)
            np.save(light_cache_path, self.light_history)
            np.save(shadow_cache_path, self.shadow_history)
            print(f"Saved precomputed histories to cache.")
            
        self.threshold = 1.0

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create figure structure (3x 3D subplots)
        self.fig = plt.figure(figsize=(18, 5))
        self.ax1 = self.fig.add_subplot(131, projection='3d')     # 3D Current Light Field
        self.ax2 = self.fig.add_subplot(132, projection='3d')     # 3D Accumulated Dose
        self.ax3 = self.fig.add_subplot(133, projection='3d')     # 3D Solidified Shape
        
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)
        
        controls_layout = QHBoxLayout()
        self.slider_label = QLabel("Time Step / Angle: 0°")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_steps - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.slider_changed)
        controls_layout.addWidget(self.slider_label)
        controls_layout.addWidget(self.slider)
        main_layout.addLayout(controls_layout)
        
        self.ax1.set_title("3D Current Light Field (Vial Frame)")
        self.ax2.set_title("3D Accumulated Dose (Inferno=Dose, Blue=Relaxing)")
        self.ax3.set_title("Solidified 3D Volume (>= Threshold)")

    def slider_changed(self, value):
        self.current_step = value
        angle = int((value / self.num_steps) * 360)
        self.slider_label.setText(f"Time Step / Angle: {angle}°")
        
        # Debounce the heavy 3D viewport redraws (waits 200ms after slider stops moving)
        self.timer.stop()
        self.timer.start(200)

    def update_plots_init(self):
        # Direct initial update of the subplots
        self.update_3d_plots()

    def update_3d_plots(self):
        t = self.current_step
        ny, nx, nz = self.target_volume.shape
        center_y = ny // 2
        center_x = nx // 2
        
        # Load precomputed dose and light for step t
        dose_t = self.dose_history[t]
        light_t = self.light_history[t]
        
        # Detect which voxels are actively relaxing (dose decreased from the previous step)
        if t > 0:
            dose_prev = self.dose_history[t-1]
        else:
            dose_prev = np.zeros_like(dose_t)
        relaxing = dose_t < dose_prev
        
        # Compute solidified mask
        solid_mask_3d = dose_t >= self.threshold
        
        ny_d, nx_d, nz_d = dose_t.shape
        y_d, x_d, z_d = np.ogrid[:ny_d, :nx_d, :nz_d]
        cy_d = ny_d // 2
        cx_d = nx_d // 2
        cz_d = nz_d // 2
        
        slice_mask = (y_d == cy_d) | (x_d == cx_d) | (z_d == cz_d)
        
        # --- Update 3D Current Light Field ---
        self.ax1.clear()
        self.ax1.set_title("3D Current Light Field (Vial Frame)")
        
        # 1. Draw vial container on Plot 1
        r_vial = min(ny, nx) // 2.1
        theta = np.linspace(0, 2 * np.pi, 100)
        xs_down = (center_x + r_vial * np.cos(theta)) / self.downsample
        ys_down = (center_y + r_vial * np.sin(theta)) / self.downsample
        nz_down = self.in_vial_down.shape[2]
        
        self.ax1.plot(xs_down, ys_down, 0, color='gray', linestyle='--', alpha=0.5)
        self.ax1.plot(xs_down, ys_down, nz_down - 1, color='gray', linestyle='--', alpha=0.5)
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            lx = (center_x + r_vial * np.cos(angle)) / self.downsample
            ly = (center_y + r_vial * np.sin(angle)) / self.downsample
            self.ax1.plot([lx, lx], [ly, ly], [0, nz_down - 1], color='gray', linestyle='--', alpha=0.3)
            
        # 2. Render light intensity slices
        light_mask = slice_mask & self.in_vial_down
        if np.any(light_mask):
            max_light = 0.035
            norm_light = np.clip(light_t / max_light, 0.0, 1.0)
            colors_light = plt.cm.afmhot(norm_light)
            colors_light[..., 3] = norm_light * 0.7
            self.ax1.voxels(light_mask, facecolors=colors_light, edgecolor='none')
            
        self.ax1.set_xlim(0, self.in_vial_down.shape[0])
        self.ax1.set_ylim(0, self.in_vial_down.shape[1])
        self.ax1.set_zlim(0, self.in_vial_down.shape[2])
        self.ax1.axis('off')
        
        # --- Update 3D Accumulated Dose Heatmap ---
        self.ax2.clear()
        self.ax2.set_title("3D Accumulated Dose (Inferno=Dose, Blue=Relaxing)")
        
        # 1. Draw vial container on Plot 2
        self.ax2.plot(xs_down, ys_down, 0, color='gray', linestyle='--', alpha=0.5)
        self.ax2.plot(xs_down, ys_down, nz_down - 1, color='gray', linestyle='--', alpha=0.5)
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            lx = (center_x + r_vial * np.cos(angle)) / self.downsample
            ly = (center_y + r_vial * np.sin(angle)) / self.downsample
            self.ax2.plot([lx, lx], [ly, ly], [0, nz_down - 1], color='gray', linestyle='--', alpha=0.3)
            
        # 2. Render dose slices
        dose_mask = slice_mask & self.in_vial_down
        if np.any(dose_mask):
            max_dose = 1.8
            norm_dose = np.clip(dose_t / max_dose, 0.0, 1.0)
            colors = plt.cm.inferno(norm_dose)
            colors[..., 3] = norm_dose * 0.75
            colors[relaxing] = [0.1, 0.6, 1.0, 0.5]  # Semi-transparent cyan for actively relaxing
            self.ax2.voxels(dose_mask, facecolors=colors, edgecolor='none')
            
        self.ax2.set_xlim(0, self.in_vial_down.shape[0])
        self.ax2.set_ylim(0, self.in_vial_down.shape[1])
        self.ax2.set_zlim(0, self.in_vial_down.shape[2])
        self.ax2.axis('off')
        
        # --- Update 3D Solidified Volume ---
        self.ax3.clear()
        self.ax3.set_title("Solidified 3D Volume (>= Threshold)")
        if np.any(solid_mask_3d):
            self.ax3.voxels(solid_mask_3d, facecolors='forestgreen', edgecolor='darkgreen', alpha=0.8)
            
        self.ax3.set_xlim(0, solid_mask_3d.shape[0])
        self.ax3.set_ylim(0, solid_mask_3d.shape[1])
        self.ax3.set_zlim(0, solid_mask_3d.shape[2])
        self.ax3.axis('off')
        
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    visualizer = VAMVisualizer()
    visualizer.show()
    sys.exit(app.exec())
