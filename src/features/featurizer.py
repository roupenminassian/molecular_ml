import numpy as np
from typing import List, Tuple
import torch
from torch_geometric.data import Data
import multiprocessing as mp
from scipy.spatial import cKDTree

class MolecularFeaturizer:
    """Class for converting molecular data to graph representations."""
    
    def __init__(self, n_jobs: int = -1, batch_size: int = 100):
        """
        Initialize the featurizer.
        
        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            batch_size: Number of molecules to process in each batch
        """
        self.atom_encoding = {'Co': 0, 'Au': 1}  # Add more atoms as needed
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, mp.cpu_count() - 1)
        self.batch_size = batch_size
        
    def compute_distances_kdtree(self, coords: np.ndarray, cutoff: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise distances using KD-tree for efficiency.
        
        Args:
            coords: Array of atomic coordinates
            cutoff: Distance cutoff for creating edges
            
        Returns:
            Tuple of edge indices and distances
        """
        tree = cKDTree(coords)
        edges = tree.query_pairs(r=cutoff, output_type='ndarray')
        
        if len(edges) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            
        distances = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
        edges_both = np.vstack((
            np.concatenate([edges[:, 0], edges[:, 1]]),
            np.concatenate([edges[:, 1], edges[:, 0]])
        ))
        distances_both = np.concatenate([distances, distances])
        
        return edges_both, distances_both
    
    def create_graph(self, coords: np.ndarray, atoms: np.ndarray, cutoff: float = 5.0) -> Data:
        """
        Convert atomic coordinates to graph structure.
        
        Args:
            coords: Array of atomic coordinates
            atoms: Array of atom types
            cutoff: Distance cutoff for creating edges
            
        Returns:
            PyTorch Geometric Data object
        """
        edge_index, distances = self.compute_distances_kdtree(coords, cutoff)
        
        edge_index = torch.from_numpy(edge_index).long()
        edge_attr = torch.from_numpy(distances).float().reshape(-1, 1)
        
        atom_indices = np.array([self.atom_encoding[atom] for atom in atoms], dtype=np.float32)
        x = torch.from_numpy(atom_indices).reshape(-1, 1)
        
        pos = torch.from_numpy(coords.astype(np.float32))
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)