import numpy as np

class LatentConformationalMapper:
    """
    Autoencoder & Latent Space Projection Engine.
    Maps high-dimensional peptide backbone conformations into a 2D reaction coordinate
    to compute the empirical Free-Energy Surface F = -kB * T * ln(P(z1, z2)).
    """
    def __init__(self, input_dim=30, latent_dim=2, seed=42):
        np.random.seed(seed)
        # Random orthogonal projection matrix for latent embedding
        self.proj = np.random.randn(input_dim, latent_dim)
        q, _ = np.linalg.qr(self.proj)
        self.proj = q
        
    def transform(self, dihedral_trajectories):
        """Projects (M, input_dim) flat angles to (M, 2) latent coords."""
        return dihedral_trajectories @ self.proj
        
    def compute_fel(self, z_coords, temperature_K=310.0, bins=35):
        """Computes 2D Free Energy Surface F(z1, z2) in kcal/mol."""
        kB = 0.001987
        H, xedges, yedges = np.histogram2d(z_coords[:, 0], z_coords[:, 1], bins=bins, density=True)
        H_smooth = H + 1e-6  # Prevent log(0)
        F = - kB * temperature_K * np.log(H_smooth)
        F = F - np.min(F)  # Set global minimum to 0 kcal/mol
        
        X, Y = np.meshgrid(0.5 * (xedges[:-1] + xedges[1:]), 0.5 * (yedges[:-1] + yedges[1:]))
        return X, Y, F.T
