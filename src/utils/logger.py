import os
import logging
import wandb
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from src.config import Config

class Logger:
    def __init__(self, config: Config):
        self.config = config
        self._setup_file_logging()
        self.run = None
    
    def _setup_file_logging(self):
        """Setup file logging."""
        log_dir = Path(self.config.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def init_wandb(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize W&B run."""
        # Get wandb username from environment variable or config
        wandb_entity = os.getenv('WANDB_ENTITY') or self.config.wandb.entity
        
        self.run = wandb.init(
            project=self.config.wandb.project,
            entity=wandb_entity,
            tags=self.config.wandb.tags,
            config=config_override
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to both file and W&B."""
        # Log to file
        metric_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metric_str}")
        
        # Log to W&B
        if self.run is not None:
            if step is not None:
                metrics['step'] = step
            wandb.log(metrics)
    
    def log_model(self, model, name: str = 'model'):
        """Save model to W&B."""
        if self.run is not None:
            torch.save(model.state_dict(), f'{name}.pt')
            wandb.save(f'{name}.pt')
    
    def finish(self):
        """Finish logging."""
        if self.run is not None:
            wandb.finish()