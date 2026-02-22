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

## Configuration

Edit `src/config/config.py` to customize:
- Data paths and splits
- Model architecture and hyperparameters
- Training settings
- Federated learning parameters

## Output

Results are saved to `OUTPUT_DIR` with:
- `best_model.pth` - Best centralized model
- `best_federated_model.pth` - Best federated model
- `sensor_scaler.pkl` - Fitted scaler
- `training_history.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `roc_curves.png` - ROC curves
- `federated_history.png` - FL convergence
- `client_accuracy.png` - Client performance
- `final_results.json` - Results summary

### XAI Outputs (in `xai/` subfolder)
- `shap_summary.png` - Top sensor features from SHAP
- `shap_dependence.png` - Feature dependence plots
- `gradcam_sample_*.png` - GradCAM visualizations for test samples
- `xai_report.json` - XAI explanations summary
