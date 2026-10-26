import numpy as np

class PeptidePotentialEnergy:
    """
    Empirical Biophysical Molecular Mechanics Potential:
    E_total = E_LJ + E_torsion + E_hbond
    """
    def __init__(self, num_residues=16):
        self.N = num_residues
        self.kB = 0.001987  # kcal / (mol * K)
        
    def compute_energy(self, dihedrals, coords):
        """
        Computes conformational potential energy in kcal/mol.
        dihedrals: (N-1, 2) array of (phi, psi) in radians.
        coords: (N, 3) 3D alpha-carbon positions.
        """
        # 1. Torsional potential (Ramachandran backbone preference)
        phi = dihedrals[:, 0]
        psi = dihedrals[:, 1]
        
        # Alpha-helix favorable basin (-60 deg, -45 deg) and Beta-sheet (-120 deg, +130 deg)
        e_torsion = np.sum(
            1.8 * (1.0 + np.cos(3 * phi)) +
            1.4 * (1.0 + np.cos(3 * psi)) +
            2.2 * np.sin(phi) * np.sin(psi)
        )
        
        # 2. Lennard-Jones 12-6 non-bonded contact potential between alpha carbons
        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        
        # Mask diagonal and immediate neighbors
        mask = np.triu(np.ones((self.N, self.N), dtype=bool), k=2)
        r = dists[mask]
        
        sigma = 3.8  # Characteristic alpha-carbon contact distance (Angstroms)
        eps = 0.15   # Potential depth (kcal/mol)
        
        sr6 = (sigma / np.maximum(r, 1.2)) ** 6
        sr12 = sr6 ** 2
        e_lj = np.sum(4.0 * eps * (sr12 - sr6))
        
        # 3. Native Hydrogen-bond network stabilization
        # Rewards compact folded contacts when 4 <= |i - j| <= 5
        h_bond_contacts = 0.0
        for i in range(self.N - 4):
            if dists[i, i + 4] < 5.5:
                h_bond_contacts += 2.5
                
        e_total = e_torsion + e_lj - h_bond_contacts
        return float(e_total)
