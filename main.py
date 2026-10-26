import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.monte_carlo import MetropolisMonteCarloEngine
from src.latent_space import LatentConformationalMapper

def run_simulation():
    print("[*] Initializing BioFold-Kinetic Stochastic Monte Carlo Engine...")
    mc = MetropolisMonteCarloEngine(num_residues=16, seed=42)
    mapper = LatentConformationalMapper(input_dim=30, latent_dim=2, seed=42)
    
    total_steps = 3000
    temperatures = np.linspace(300.0, 395.0, total_steps)  # Thermal ramp (K)
    
    energy_history = []
    rmsd_history = []
    dihedral_snapshots = []
    all_phi = []
    all_psi = []
    
    print(f"[*] Simulating {total_steps} Monte Carlo conformational sampling steps across 300K -> 395K...")
    for t in range(total_steps):
        T_curr = temperatures[t]
        energy, rmsd, accepted = mc.step(T_curr)
        
        energy_history.append(energy)
        rmsd_history.append(rmsd)
        dihedral_snapshots.append(mc.dihedrals.flatten())
        
        all_phi.extend(mc.dihedrals[:, 0])
        all_psi.extend(mc.dihedrals[:, 1])
        
        if (t + 1) % 750 == 0:
            print(f"    Step {t+1:04d}/{total_steps} | Temp: {T_curr:.1f} K | Energy: {energy:.2f} kcal/mol | RMSD: {rmsd:.2f} Å")
            
    print("[*] Computing 2D Latent Free Energy Landscape (FEL)...")
    dihedral_mat = np.array(dihedral_snapshots)
    latent_coords = mapper.transform(dihedral_mat)
    X_fel, Y_fel, F_surface = mapper.compute_fel(latent_coords, temperature_K=310.0, bins=30)
    
    print("[*] Generating publication-grade biomolecular kinetics visualization...")
    os.makedirs("assets", exist_ok=True)
    
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], wspace=0.25, hspace=0.32)
    
    # Panel A: 3D Free-Energy Landscape Surface
    ax_fel = fig.add_subplot(gs[0, :], projection='3d')
    surf = ax_fel.plot_surface(
        X_fel, Y_fel, F_surface,
        cmap="viridis",
        edgecolor="#1E293B",
        linewidth=0.3,
        alpha=0.9
    )
    cbar_f = fig.colorbar(surf, ax=ax_fel, shrink=0.5, pad=0.08)
    cbar_f.set_label("Free Energy ΔG (kcal/mol)", color="#E2E8F0", fontsize=9)
    ax_fel.set_title("A. 3D Conformational Free-Energy Landscape: Folding Funnels & Metastable Basins", fontsize=11, fontweight="bold", color="#F8FAFC", pad=12)
    ax_fel.set_xlabel("Latent Coordinate CV1", fontsize=8)
    ax_fel.set_ylabel("Latent Coordinate CV2", fontsize=8)
    ax_fel.set_zlabel("Free Energy (kcal/mol)", fontsize=8)
    ax_fel.view_init(elev=32, azim=-60)
    
    # Panel B: Thermal Unfolding & RMSD Trajectory
    ax_rmsd = fig.add_subplot(gs[1, 0])
    steps_arr = np.arange(1, total_steps + 1)
    
    color_r = "#38BDF8"
    ax_rmsd.plot(steps_arr, rmsd_history, color=color_r, alpha=0.3, linewidth=0.8)
    # Moving average
    window = 50
    rmsd_smooth = np.convolve(rmsd_history, np.ones(window)/window, mode='valid')
    ax_rmsd.plot(np.arange(window, total_steps + 1), rmsd_smooth, color="#0284C7", linewidth=2.0, label="C-α RMSD (50-step MA)")
    ax_rmsd.set_xlabel("Monte Carlo Steps (Thermal Ramp)", fontsize=9)
    ax_rmsd.set_ylabel("Backbone RMSD (Å)", color=color_r, fontsize=9)
    ax_rmsd.tick_params(axis='y', labelcolor=color_r)
    
    ax_temp = ax_rmsd.twinx()
    color_t = "#F59E0B"
    ax_temp.plot(steps_arr, temperatures, color=color_t, linestyle="--", linewidth=1.5, label="Temperature (K)")
    ax_temp.set_ylabel("Simulation Temp (Kelvin)", color=color_t, fontsize=9)
    ax_temp.tick_params(axis='y', labelcolor=color_t)
    ax_rmsd.set_title("B. Thermal Denaturation & Structural Unfolding Trajectory", fontsize=11, fontweight="bold", color="#F8FAFC", pad=8)
    
    # Panel C: Ramachandran Backbone Dihedral Distribution (phi vs psi)
    ax_rama = fig.add_subplot(gs[1, 1])
    phi_deg = np.degrees(all_phi)
    psi_deg = np.degrees(all_psi)
    
    h_rama = ax_rama.hexbin(phi_deg, psi_deg, gridsize=35, cmap="magma", mincnt=1)
    cbar_r = fig.colorbar(h_rama, ax=ax_rama)
    cbar_r.set_label("Conformational Density", color="#E2E8F0", fontsize=8)
    ax_rama.set_xlim(-180, 180)
    ax_rama.set_ylim(-180, 180)
    ax_rama.axhline(0, color="#475569", linestyle=":", linewidth=1)
    ax_rama.axvline(0, color="#475569", linestyle=":", linewidth=1)
    ax_rama.set_title("C. Ramachandran Backbone Dihedral Density (φ vs. ψ)", fontsize=11, fontweight="bold", color="#F8FAFC", pad=8)
    ax_rama.set_xlabel("Phi φ (Degrees)", fontsize=9)
    ax_rama.set_ylabel("Psi ψ (Degrees)", fontsize=9)
    ax_rama.grid(color="#334155", linestyle="--", alpha=0.4)
    
    plt.savefig("assets/biomolecular_kinetics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] BioFold-Kinetic telemetry saved to assets/biomolecular_kinetics.png")

if __name__ == "__main__":
    run_simulation()
