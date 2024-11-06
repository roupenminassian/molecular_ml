# Molecular Property Prediction Pipeline

A machine learning pipeline for predicting molecular properties using Graph Neural Networks (GNN) based on molecular structure data from XYZ files and additional features. Features automated hyperparameter optimization using Weights & Biases (wandb).

## Overview

This pipeline processes molecular structure data from XYZ files along with additional features from a CSV file to predict specified molecular properties. It uses a Graph Neural Network approach to handle the varying sizes and structures of molecules, with automated hyperparameter optimization and experiment tracking.

## Features

- Processes molecular XYZ files for structural information
- Integrates additional molecular features from CSV files
- Converts molecular structures to graph representations
- Automated hyperparameter optimization using wandb
- Multiple GNN architectures (GCN, GAT, GraphConv)
- Configurable target property prediction
- Support for both CPU and GPU training
- Parallel data processing and caching
- Comprehensive logging and metric tracking
- Progress visualization with wandb
- Early stopping and learning rate scheduling

## Project Structure
```
molecular_ml/
│
├── data/
│   ├── raw/
│   │   ├── xyz_files/    # Place your .xyz files here
│   │   └── features.csv  # Place your features file here
│   └── processed/        # Processed data and cache
│       └── cache/        # Zarr cache storage
│
├── src/
│   ├── data/            # Data loading and processing
│   │   ├── __init__.py
│   │   └── loader.py    # XYZ and CSV data loading
│   ├── features/        # Feature engineering
│   │   ├── __init__.py
│   │   └── featurizer.py # Graph creation and featurization
│   ├── models/          # ML models
│   │   ├── __init__.py
│   │   └── model.py     # GNN model implementations
│   └── utils/           # Utility functions
│       ├── __init__.py
│       └── logger.py    # Logging and W&B integration
│
├── configs/             # Configuration files
│   └── config.yaml     # Main configuration file
├── main.py             # Main execution script
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd molecular_ml
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Weights & Biases:
```bash
wandb login
```

## Data Preparation

1. Place your XYZ files in `data/raw/xyz_files/`
2. Place your features CSV file in `data/raw/features.csv`

The features CSV file should contain:
- An 'ID' column matching XYZ filenames (format: '000001.xyz')
- Feature columns
- The target property column

## Configuration

The `config.yaml` file contains all settings including model architecture, training parameters, and hyperparameter sweep configurations.

Example configuration:
```yaml
# Data paths
xyz_dir: "data/raw/xyz_files"
features_file: "data/raw/features.csv"
target_column: "Formation_E"
output_dir: "output"
cache_dir: "data/processed/cache"

# Wandb configuration
wandb:
  project: "molecular-property-prediction"
  entity: "your-username"
  tags: ["GNN", "molecular-properties"]
  sweep:
    method: "bayes"
    metric:
      name: "val_rmse"
      goal: "minimize"
    parameters:
      conv_type:
        values: ["GCN", "GAT", "GraphConv"]
      num_layers:
        values: [2, 3, 4, 5]
      hidden_channels:
        values: [32, 64, 128, 256]

# Model and training parameters...
```

## Usage

### Single Training Run
```bash
python main.py --config configs/config.yaml
```

### Hyperparameter Optimization
```bash
python main.py --config configs/config.yaml --sweep --count 50
```

### Monitoring Training
1. Navigate to your wandb project page: `https://wandb.ai/[username]/[project-name]`
2. View real-time training metrics, parameter importance, and results visualization

## Model Architecture

The pipeline supports multiple GNN architectures:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- General Graph Convolution (GraphConv)

Features:
- Configurable number of layers
- Residual connections
- Batch normalization
- Dropout regularization
- Global pooling
- Multi-layer prediction head

## Advanced Features

### Caching
- Processed molecular graphs are cached using Zarr
- Significantly speeds up subsequent runs
- Configurable cache location

### Parallel Processing
- Multi-threaded data loading
- Parallel graph construction
- GPU acceleration when available

### Hyperparameter Optimization
- Bayesian optimization using wandb
- Configurable parameter spaces
- Automated tracking of all experiments
- Parameter importance visualization

### Logging and Monitoring
- Comprehensive logging to files
- Real-time metric tracking in wandb
- Progress bars for all operations
- Detailed error reporting

## Requirements

- Python 3.11+
- PyTorch
- PyTorch Geometric
- Weights & Biases
- Zarr
- Additional requirements in `requirements.txt`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Please ensure:
- Code follows project structure
- New features include tests
- Documentation is updated
- Commit messages are descriptive

## License

[Your chosen license]

## Contact

[Your contact information]

## Acknowledgments

- PyTorch Geometric team for the GNN implementations
- Weights & Biases for experiment tracking
- [Additional acknowledgments]