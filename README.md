# Multimodal Federated Learning Pipeline

Modular implementation of multimodal federated learning with support for multiple backbone architectures and federated learning strategies.

## Project Structure

```
├── src/
│   ├── config/
│   │   └── config.py              # Configuration classes
│   ├── data/
│   │   ├── preprocessing.py       # Data loading and preprocessing
│   │   └── dataset.py             # Dataset and sample generation
│   ├── models/
│   │   ├── encoders.py            # Sensor/Video encoders and backbones
│   │   └── multimodal.py          # Multimodal fusion model
│   ├── training/
│   │   ├── loss.py                # FocalLoss implementation
│   │   ├── trainer.py             # Centralized training
│   │   └── evaluator.py           # Model evaluation
│   ├── federated/
│   │   ├── strategies/
│   │   │   ├── base.py            # Base FL strategy
│   │   │   ├── fedavg.py          # FedAvg strategy
│   │   │   ├── fedprox.py         # FedProx strategy
│   │   │   ├── fedadam.py         # FedAdam strategy
│   │   │   ├── fednova.py         # FedNova strategy
│   │   │   └── scaffold.py        # SCAFFOLD strategy
│   │   ├── client.py              # FL client
│   │   └── server.py              # FL server
│   ├── xai/
│   │   ├── shap_explainer.py      # SHAP-based sensor explanations
│   │   ├── gradcam.py             # GradCAM video explanations
│   │   └── manager.py             # XAI orchestration
│   └── utils/
│       ├── logger.py              # Logging utilities
│       └── metrics.py             # Visualization utilities
├── main.py                        # Main pipeline
├── ablation.py                    # Ablation study runner
└── requirements.txt               # Dependencies
```

## Supported Models

- MobileNetV2
- VGG16
- ResNet50
- EfficientNet
- DenseNet121
- InceptionV3
- ConvNeXt-Tiny

## Supported FL Strategies

- FedAvg
- FedProx
- FedAdam
- FedNova
- SCAFFOLD

## Explainable AI (XAI)

- **SHAP** - Explains sensor feature importance with kernel SHAP
- **GradCAM** - Highlights important video regions via gradient-based activation maps

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python main.py
```

### Run Ablation Study

```bash
python ablation.py
```