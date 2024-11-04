import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GraphConv
import wandb
from typing import Dict, Optional
from tqdm.auto import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from src.config import ModelConfig, TrainingConfig

class MolecularGNN(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.node_embedding = nn.Linear(config.num_features, config.hidden_channels)
        
        # Multiple convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(config.num_layers):
            if config.conv_type == 'GCN':
                conv = GCNConv(config.hidden_channels, config.hidden_channels)
            elif config.conv_type == 'GAT':
                conv = GATConv(config.hidden_channels, config.hidden_channels)
            else:
                conv = GraphConv(config.hidden_channels, config.hidden_channels)
            
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(config.hidden_channels))
        
        # Output layers
        self.linear1 = nn.Linear(config.hidden_channels, config.hidden_channels // 2)
        self.linear2 = nn.Linear(config.hidden_channels // 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Message passing layers
        for i, conv in enumerate(self.convs):
            identity = x
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropout(x)
            if self.config.use_residual:
                x = x + identity
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # MLP head
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x