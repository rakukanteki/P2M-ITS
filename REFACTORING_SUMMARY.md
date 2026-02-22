# Project Refactoring Summary

## Overview
The monolithic script has been refactored into a modular, OOP-based architecture with clear separation of concerns.

## Directory Structure

```
P2M-ITS/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py                 # Configuration dataclasses
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py          # Data loading, sensor/video processing, scaler management
│   │   └── dataset.py                # Dataset classes, windowed sample generation, leakage checking
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py               # SensorEncoder, VideoEncoder, BackboneFactory
│   │   └── multimodal.py             # MultimodalModel (fusion architecture)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loss.py                   # FocalLoss
│   │   ├── trainer.py                # CentralizedTrainer
│   │   └── evaluator.py              # ModelEvaluator with bootstrap CI
│   │
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # Base FL strategy class
│   │   │   ├── fedavg.py             # FedAvg implementation
│   │   │   ├── fedprox.py            # FedProx implementation
│   │   │   ├── fedadam.py            # FedAdam implementation
│   │   │   ├── fednova.py            # FedNova implementation
│   │   │   └── scaffold.py           # SCAFFOLD implementation
│   │   ├── client.py                 # FLClient class
│   │   └── server.py                 # FLServer coordinator
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # Logger utilities
│       └── metrics.py                # Visualizer (plots)
│
├── main.py                           # Main pipeline
├── ablation.py                       # Ablation study runner
├── example_config.py                 # Configuration example
├── requirements.txt                  # Dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore rules
```

## Key Classes and Modules

### Configuration (config/)
- `DataConfig` - Data paths, splits, window sizes, frames
- `ModelConfig` - Backbone selection, embedding dimensions
- `TrainingConfig` - Learning rate, batch size, epochs, focal loss gamma
- `FederatedConfig` - Client count, rounds, local epochs, strategy
- `Config` - Main configuration container

### Data Processing (data/)
- `SensorPreprocessor` - Window engineering (Eqs 5-9)
- `VideoProcessor` - Frame extraction (Eq. 10)
- `DataLoader` - Folder loading
- `ScalerManager` - Scaler persistence
- `WindowedSampleGenerator` - Sliding window samples
- `FolderDataset` - PyTorch dataset
- `DataLeakageChecker` - Hash-based leakage verification

### Models (models/)
- `SensorEncoder` - 2-layer MLP encoder
- `VideoEncoder` - Generic backbone wrapper
- `BackboneFactory` - Creates different backbones (7 models)
- `MultimodalModel` - Dual-stream fusion

### Training (training/)
- `FocalLoss` - Class-imbalance handling
- `CentralizedTrainer` - Training loop with early stopping
- `ModelEvaluator` - Metrics, bootstrap CI, ROC curves

### Federated Learning (federated/)
- `FLStrategy` (base) - Abstract strategy interface
- `FedAvg` - Weighted averaging
- `FedProx` - Proximal term
- `FedAdam` - Server-side momentum
- `FedNova` - Weighted by local updates
- `SCAFFOLD` - Control variates
- `FLClient` - Client with local dataset
- `FLServer` - Coordinator

### Utilities (utils/)
- `Logger` - Section/info logging
- `Visualizer` - Plot functions

## Supported Models

1. **MobileNetV2** - Lightweight, efficient
2. **VGG16** - Classic deep architecture
3. **ResNet50** - Skip connections
4. **EfficientNet** - Compound scaling
5. **DenseNet121** - Dense connections
6. **InceptionV3** - Inception modules
7. **ConvNeXt-Tiny** - Modern architecture

## Supported FL Strategies

1. **FedAvg** - Weighted parameter averaging
2. **FedProx** - Proximity constraint
3. **FedAdam** - Server momentum (beta_1, beta_2)
4. **FedNova** - Weighted by local updates
5. **SCAFFOLD** - Control variates for drift correction

## Key Features

✅ Modular architecture with clear separation of concerns
✅ OOP-based design with reusable components
✅ Support for 7 backbone architectures
✅ Support for 5 FL strategies
✅ Configuration-driven (no hardcoding)
✅ Realistic, production-ready code
✅ Ablation study framework
✅ Comprehensive evaluation with CI
✅ Data leakage verification
✅ Client performance variance tracking

## Usage

### Basic Run
```bash
python main.py
```

### Custom Configuration
```bash
python example_config.py
```

### Ablation Study
```bash
python ablation.py
```

## Notes

- All models use frozen ImageNet pre-trained weights
- Sensor encoder uses Xavier initialization
- Video frames uniformly sampled (Eq. 10)
- Sensor features engineered (Eqs 5-9)
- FocalLoss for class imbalance
- Per-folder non-IID FL split
- Bootstrap-based confidence intervals
- No excessive comments - clean, readable code
