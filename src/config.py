from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path

@dataclass
class WandbConfig:
    project: str
    entity: str
    tags: List[str]
    sweep: Dict[str, Any]

@dataclass
class ModelConfig:
    num_features: int
    conv_type: str
    num_layers: int
    hidden_channels: int
    dropout_rate: float
    use_residual: bool

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    early_stopping_patience: int
    lr_factor: float
    lr_patience: int

@dataclass
class DataConfig:
    random_seed: int
    test_size: float
    num_workers: int
    prefetch_factor: int
    pin_memory: bool

@dataclass
class Config:
    xyz_dir: Path
    features_file: Path
    target_column: str
    output_dir: Path
    cache_dir: Path
    wandb: WandbConfig
    model_params: ModelConfig
    training_params: TrainingConfig
    data_params: DataConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Convert paths to Path objects
        for key in ['xyz_dir', 'features_file', 'output_dir', 'cache_dir']:
            config_dict[key] = Path(config_dict[key])
            
        # Create nested config objects
        config_dict['wandb'] = WandbConfig(**config_dict['wandb'])
        config_dict['model_params'] = ModelConfig(**config_dict['model_params'])
        config_dict['training_params'] = TrainingConfig(**config_dict['training_params'])
        config_dict['data_params'] = DataConfig(**config_dict['data_params'])
        
        return cls(**config_dict)