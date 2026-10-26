import numpy as np
from src.biophysics import PeptidePotentialEnergy

class MetropolisMonteCarloEngine:
    """
    Stochastic Metropolis-Hastings conformational sampler
    exploring peptide conformational thermodynamics across thermal ramps.
    """
    def __init__(self, num_residues=16, seed=42):
        np.random.seed(seed)
        self.N = num_residues
        self.potential = PeptidePotentialEnergy(num_residues=num_residues)
        
        # Initial alpha-helical conformation: phi ~ -1.05 rad (-60 deg), psi ~ -0.78 rad (-45 deg)
        self.dihedrals = np.tile([-1.05, -0.78], (self.N - 1, 1))
        self.coords = self._build_coordinates(self.dihedrals)
        self.current_energy = self.potential.compute_energy(self.dihedrals, self.coords)
        
        self.native_coords = self.coords.copy()
        
    def _build_coordinates(self, dihedrals):
        """Reconstructs 3D C-alpha backbone chain via rotational matrices."""
        coords = np.zeros((self.N, 3))
        bond_len = 3.8  # C_alpha - C_alpha distance in Angstroms
        
        curr_pos = np.array([0.0, 0.0, 0.0])
        curr_dir = np.array([1.0, 0.0, 0.0])
        
        coords[0] = curr_pos
        for i in range(1, self.N):
            phi, psi = dihedrals[i - 1]
            # Rotation around local axes
            R_z = np.array([
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi),  np.cos(phi), 0],
                [0, 0, 1]
            ])
            R_y = np.array([
                [np.cos(psi), 0, np.sin(psi)],
                [0, 1, 0],
                [-np.sin(psi), 0, np.cos(psi)]
            ])
            curr_dir = R_y @ R_z @ curr_dir
            curr_dir = curr_dir / (np.linalg.norm(curr_dir) + 1e-6)
            curr_pos = curr_pos + curr_dir * bond_len
            coords[i] = curr_pos
            
        return coords
        
    def step(self, temperature_K):
        """Single Metropolis-Hastings move."""
        # Perturb random residue's dihedral
        res_idx = np.random.randint(0, self.N - 1)
        angle_idx = np.random.randint(0, 2)
        
        cand_dihedrals = self.dihedrals.copy()
        cand_dihedrals[res_idx, angle_idx] += np.random.normal(0, 0.15)
        # Wrap to [-pi, pi]
        cand_dihedrals[res_idx, angle_idx] = (cand_dihedrals[res_idx, angle_idx] + np.pi) % (2 * np.pi) - np.pi
        
        cand_coords = self._build_coordinates(cand_dihedrals)
        cand_energy = self.potential.compute_energy(cand_dihedrals, cand_coords)
        
        delta_E = cand_energy - self.current_energy
        
        # Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(- delta_E / (self.potential.kB * temperature_K)):
            self.dihedrals = cand_dihedrals
            self.coords = cand_coords
            self.current_energy = cand_energy
            accepted = True
        else:
            accepted = False
            
        # Compute RMSD from native fold
        diff = self.coords - self.native_coords
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=-1)))
        
        return self.current_energy, rmsd, accepted
