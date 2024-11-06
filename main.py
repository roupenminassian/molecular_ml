import os
import warnings
import psutil
import torch
import wandb
from tqdm.auto import tqdm
from pathlib import Path
from typing import Dict, Any, List
import argparse
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import traceback
from datetime import datetime
from torch.utils.data import Subset

# Suppress warnings
os.environ["WANDB_SILENT"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

from src.config import Config, ModelConfig, TrainingConfig
from src.models.model import MolecularGNN
from src.utils.logger import Logger
from src.data.loader import DataLoader
from src.data.dataset import MolecularDataset

def get_memory_usage():
    """Get current memory usage of the process in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def print_memory_usage(stage: str):
    """Print current memory usage with timestamp."""
    memory_mb = get_memory_usage()
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Memory usage at {stage}: {memory_mb:.2f} MB")

def create_gpu_optimized_loaders(dataset, train_idx: List[int], val_idx: List[int], 
                               test_idx: List[int], batch_size: int):
    """Create data loaders optimized for GTX 1650 (4GB VRAM)."""
    
    print("\nCreating GPU-optimized data loaders...")
    print(f"Batch size: {batch_size} (adjust if you see out-of-memory errors)")
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Create optimized loaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    print(f"Number of batches - Train: {len(train_loader)}, "
          f"Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def create_sweep_config(config: Config) -> dict:
    """Create wandb sweep configuration from config file."""
    return config.wandb.sweep

def train_model(config: Dict[str, Any], base_config: Config, logger: Logger):
    """Training function for a single wandb run."""
    try:
        print("\n" + "="*80)
        print(f"Starting training pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print_memory_usage("start of training")

        # Update base config with sweep parameters
        model_params = vars(base_config.model_params).copy()
        training_params = vars(base_config.training_params).copy()
        
        for key, value in config.items():
            if key in model_params:
                model_params[key] = value
            if key in training_params:
                training_params[key] = value
        
        # Data loading and preprocessing
        print("\nLoading data...")
        print_memory_usage("before data loading")
        data_loader = DataLoader(base_config.xyz_dir, base_config.features_file)
        molecule_data, features_df = data_loader.load_data()
        print(f"Loaded {len(molecule_data)} molecules and their features")
        print_memory_usage("after data loading")
        
        # Create dataset
        print("\nInitializing dataset...")
        dataset = MolecularDataset(
            root=str(Path(base_config.xyz_dir).parent),
            molecule_data=molecule_data,
            features_df=features_df,
            target_column=base_config.target_column
        )
        
        print(f"\nDataset size: {len(dataset)} molecules")
        print_memory_usage("after dataset initialization")
        
        # Split data using indices
        print("\nSplitting dataset into train/val/test sets...")
        indices = list(range(len(dataset)))
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=base_config.data_params.test_size,
            random_state=base_config.data_params.random_seed
        )
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5)
        
        print(f"Train set size: {len(train_idx)}")
        print(f"Validation set size: {len(val_idx)}")
        print(f"Test set size: {len(test_idx)}")
        
        # GPU setup and optimization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
            print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
            torch.cuda.empty_cache()  # Clear GPU memory
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            print("\nWARNING: Running on CPU. Training will be slow.")
        
        # Create optimized data loaders
        batch_size = 1 if device.type == 'cuda' else training_params['batch_size']
        train_loader, val_loader, test_loader = create_gpu_optimized_loaders(
            dataset, train_idx, val_idx, test_idx, batch_size
        )
        print_memory_usage("after data loader creation")
        
        # Initialize model
        print("\nInitializing model...")
        model = MolecularGNN(ModelConfig(**model_params)).to(device)
        if device.type == 'cuda':
            model = model.half()  # Use half precision for GPU
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialize optimizer and scheduler
        print("\nSetting up optimizer and scheduler...")
        optimizer = (
            torch.optim.AdamW(model.parameters(),
                             lr=training_params['learning_rate'],
                             weight_decay=training_params['weight_decay'])
            if training_params['optimizer'] == 'AdamW'
            else torch.optim.Adam(model.parameters(),
                                lr=training_params['learning_rate'],
                                weight_decay=training_params['weight_decay'])
        )
        
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_params['lr_factor'],
            patience=training_params['lr_patience']
        )
        
        # Initialize gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
        
        # Training loop
        print("\n" + "="*80)
        print("Starting training loop")
        print("="*80)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(training_params['epochs']):
            if device.type == 'cuda':
                print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated")
            
            # Training phase
            model.train()
            total_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params['epochs']}", 
                     leave=True, ncols=100) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    batch = batch.to(device)
                    
                    # Use automatic mixed precision for GPU
                    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                        optimizer.zero_grad(set_to_none=True)
                        out = model(batch)
                        loss = criterion(out, batch.y)
                    
                    if device.type == 'cuda':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    
                    total_loss += loss.item() * batch.num_graphs
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    # Clear GPU memory periodically
                    if device.type == 'cuda' and batch_idx % 100 == 0:
                        torch.cuda.empty_cache()
            
            train_loss = total_loss / len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                        out = model(batch)
                        loss = criterion(out, batch.y)
                    val_loss += loss.item() * batch.num_graphs
                    predictions.extend(out.cpu().numpy())
                    targets.extend(batch.y.cpu().numpy())
            
            val_loss = val_loss / len(val_loader.dataset)
            val_rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))
            
            # Logging
            logger.log_metrics({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Val RMSE = {val_rmse:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_save_path = Path(base_config.output_dir) / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'model_params': model_params,
                    'training_params': training_params
                }, model_save_path)
                print(f"Saved new best model with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= training_params['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Clear GPU memory after each epoch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Final evaluation
        print("\n" + "="*80)
        print("Performing final evaluation...")
        print("="*80)
        
        model.eval()
        test_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = batch.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    out = model(batch)
                    loss = criterion(out, batch.y)
                test_loss += loss.item() * batch.num_graphs
                predictions.extend(out.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        test_loss = test_loss / len(test_loader.dataset)
        test_rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))
        
        # Log final metrics
        final_metrics = {
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }
        
        logger.log_metrics(final_metrics)
        
        # Save final results
        results_path = Path(base_config.output_dir) / 'final_results.txt'
        with open(results_path, 'w') as f:
            f.write(f"Final Training Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n")
            for metric, value in final_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final test RMSE: {test_rmse:.4f}")
        print(f"Results saved to: {results_path}")
        print_memory_usage("end of training")
        
    except Exception as e:
        print("\nError occurred during training:")
        print("-"*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nTraceback:")
        print("-"*80)
        traceback.print_exc()
        print_memory_usage("at error")
        raise e

def main():
    try:
        parser = argparse.ArgumentParser(description='Molecular Property Prediction Training Pipeline')
        parser.add_argument('--config', type=str, required=True, help='Path to config file')
        parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
        parser.add_argument('--count', type=int, default=50, help='Number of sweep runs')
        args = parser.parse_args()
        
        print("\n" + "="*80)
        print(f"Starting molecular property prediction pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using config file: {args.config}")
        print("="*80)
        
        # Load configuration
        print("\nLoading configuration...")
        config = Config.from_yaml(args.config)
        
        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Initialize logger
        print("\nInitializing logger...")
        logger = Logger(config)
        
        if args.sweep:
            print("\nStarting hyperparameter sweep...")
            sweep_config = create_sweep_config(config)
            sweep_id = wandb.sweep(sweep_config, 
                                 project=config.wandb.project,
                                 entity=config.wandb.entity)
            
            wandb.agent(sweep_id, 
                       lambda: train_model(wandb.config, config, logger),
                       count=args.count)
        else:
            print("\nRunning single training session...")
            logger.init_wandb()
            train_model(vars(config.model_params), config, logger)
            logger.finish()
            
    except Exception as e:
        print("\nError occurred during execution:")
        print("-"*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nTraceback:")
        print("-"*80)
        traceback.print_exc()
        print_memory_usage("at error")
        sys.exit(1)

if __name__ == "__main__":
    main()