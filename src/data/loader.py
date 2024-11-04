import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from tqdm.auto import tqdm
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
import psutil
from torch.utils.data import Subset
import logging

class DataLoader:
    def __init__(self, xyz_dir: str, features_file: str):
        """
        Initialize the data loader.
        
        Args:
            xyz_dir: Directory containing XYZ files
            features_file: Path to features CSV file
        """
        self.xyz_dir = Path(xyz_dir)
        self.features_file = Path(features_file)
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _print_memory_usage(self, stage: str):
        """Print current memory usage."""
        memory_mb = self._get_memory_usage()
        print(f"Memory usage at {stage}: {memory_mb:.2f} MB")

    def parse_xyz_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a single XYZ file.
        
        Args:
            file_path: Path to XYZ file
            
        Returns:
            Tuple of (coordinates array, atoms array)
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        n_atoms = int(lines[0])
        atoms = []
        coords = []
        
        for line in lines[2:2+n_atoms]:
            atom, x, y, z = line.split()
            atoms.append(atom)
            coords.append([float(x), float(y), float(z)])
            
        return np.array(coords), np.array(atoms)

    def load_data(self) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
        """
        Load all molecules and their features.
        
        Returns:
            Tuple of (list of molecule data, features DataFrame)
        """
        print("\nLoading feature data...")
        features_df = pd.read_csv(self.features_file)
        
        print("\nLoading molecular structures...")
        molecule_data = []
        xyz_files = sorted(self.xyz_dir.glob('*.xyz'))
        
        for xyz_file in tqdm(xyz_files, desc="Loading molecules"):
            coords, atoms = self.parse_xyz_file(xyz_file)
            molecule_data.append((coords, atoms))
        
        print(f"\nSuccessfully loaded {len(molecule_data)} molecules")
        return molecule_data, features_df

def create_data_loaders(
    dataset,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool
) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    """
    Create train, validation, and test data loaders efficiently.
    
    Args:
        dataset: The full dataset
        train_idx: Indices for training set
        val_idx: Indices for validation set
        test_idx: Indices for test set
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("\nCreating data loaders...")
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    try:
        # Create subset datasets
        print("\nCreating dataset subsets...")
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        
        print(f"Memory usage after creating subsets: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Create train loader
        print("\nCreating train loader...")
        train_loader = PyGDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        print(f"Train loader created with {len(train_loader)} batches")
        
        # Create validation loader
        print("\nCreating validation loader...")
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(2, num_workers),  # Use fewer workers for validation
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )
        print(f"Validation loader created with {len(val_loader)} batches")
        
        # Create test loader
        print("\nCreating test loader...")
        test_loader = PyGDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(2, num_workers),  # Use fewer workers for testing
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )
        print(f"Test loader created with {len(test_loader)} batches")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"\nFinal memory usage: {final_memory:.2f} MB")
        print(f"Memory increase: {final_memory - initial_memory:.2f} MB")
        
        print("\nData loaders created successfully!")
        print(f"Total number of samples:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Validation: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"\nError creating data loaders: {str(e)}")
        print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        raise RuntimeError(f"Failed to create data loaders: {str(e)}") from e