import os
import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from src.monte_carlo import MetropolisMonteCarloEngine
from src.latent_space import LatentConformationalMapper

def generate_spinning_animation():
    print("[*] Generating 3D Spinning Free-Energy Landscape Animation for BioFold-Kinetic...")
    mc = MetropolisMonteCarloEngine(num_residues=16, seed=42)
    mapper = LatentConformationalMapper(input_dim=30, latent_dim=2, seed=42)
    
    # Sample 1500 steps to map FEL
    dihedral_snapshots = []
    temperatures = np.linspace(300.0, 380.0, 1500)
    for t in range(1500):
        mc.step(temperatures[t])
        dihedral_snapshots.append(mc.dihedrals.flatten())
        
    dihedral_mat = np.array(dihedral_snapshots)
    latent_coords = mapper.transform(dihedral_mat)
    X_fel, Y_fel, F_surface = mapper.compute_fel(latent_coords, temperature_K=310.0, bins=26)
    
    frames = []
    total_frames = 36
    
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(7, 6), dpi=120)
    
    for i, azim in enumerate(np.linspace(0, 360, total_frames, endpoint=False)):
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#030712")
        fig.patch.set_facecolor("#030712")
        
        # 3D Surface
        surf = ax.plot_surface(
            X_fel, Y_fel, F_surface,
            cmap="viridis",
            edgecolor="#1E293B",
            linewidth=0.25,
            alpha=0.92
        )
        
        # Active Folding Trajectory Particle
        curr_step = int((i / total_frames) * len(latent_coords)) % len(latent_coords)
        pt_x = latent_coords[curr_step, 0]
        pt_y = latent_coords[curr_step, 1]
        ax.scatter([pt_x], [pt_y], [np.min(F_surface) + 0.5], color="#F43F5E", s=60, edgecolors="#FFFFFF", linewidths=1.2, label="Folding State")
        
        elev = 30.0 + 8.0 * np.sin(np.radians(azim * 2))
        ax.view_init(elev=elev, azim=azim)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.axis("off")
        ax.set_title("3D Conformational Free-Energy Folding Funnel", color="#F8FAFC", fontsize=11, fontweight="bold", pad=-5)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#030712", pad_inches=0.1)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        
    plt.close()
    
    output_path = "assets/energy_landscape_spin.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=65,
        loop=0
    )
    print(f"[✓] Saved 3D spinning energy landscape animation to {output_path}")

if __name__ == "__main__":
    generate_spinning_animation()
