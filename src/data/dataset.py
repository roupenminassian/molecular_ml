from typing import List, Tuple
import torch
from torch_geometric.data import Dataset
from tqdm.auto import tqdm
import os
import pandas as pd
import numpy as np
from pathlib import Path
from ..features.featurizer import MolecularFeaturizer

class MolecularDataset(Dataset):
    """PyTorch Geometric Dataset for molecular data."""
    
    def __init__(self, 
                 root: str,
                 molecule_data: List[Tuple[np.ndarray, np.ndarray]],
                 features_df: pd.DataFrame,
                 target_column: str,
                 transform=None):
        self.molecule_data = molecule_data
        self.features_df = features_df
        self.target_column = target_column
        super().__init__(root, transform)

    @property
    def raw_file_names(self) -> List[str]:
        """Files in the raw directory."""
        return []  # We don't use raw files directly

    @property
    def processed_file_names(self) -> List[str]:
        """Files in the processed directory."""
        return [f'data_{i}.pt' for i in range(len(self.molecule_data))]

    def _process_exists(self) -> bool:
        """Check if all processed files exist."""
        print("\nChecking for existing processed files...")
        processed_dir = Path(self.processed_dir)
        if not processed_dir.exists():
            print("No processed directory found.")
            return False
            
        expected_files = set(self.processed_file_names)
        existing_files = set(os.listdir(processed_dir))
        missing_files = expected_files - existing_files
        
        if not missing_files:
            print(f"Found complete set of {len(expected_files)} processed files.")
            return True
        else:
            print(f"Missing {len(missing_files)} processed files.")
            return False

    def download(self):
        """Download is not needed as we have the data."""
        pass

    def process(self):
        """Process raw data and save as PyTorch files."""
        # First check if we already have processed files
        if self._process_exists():
            print("Using existing processed files.")
            return

        print("\nProcessing molecules...")
        featurizer = MolecularFeaturizer(
            n_jobs=-1,
            batch_size=100
        )

        # Create processed directory if it doesn't exist
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {len(self.molecule_data)} molecules...")
        for idx in tqdm(range(len(self.molecule_data)), desc="Processing molecules"):
            # Process single molecule
            coords, atoms = self.molecule_data[idx]
            graph = featurizer.create_graph(coords, atoms)
            
            # Add target
            target = self.features_df[self.target_column].iloc[idx]
            graph.y = torch.tensor([target], dtype=torch.float)
            
            # Save to disk
            torch.save(graph, os.path.join(self.processed_dir, f'data_{idx}.pt'))

        print(f"\nProcessing complete! Saved {len(self.molecule_data)} graphs.")

    def len(self):
        return len(self.molecule_data)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))