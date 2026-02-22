"""
Example: Running with specific configuration
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import Config, DataConfig, ModelConfig, TrainingConfig, FederatedConfig

# Initialize config
config = Config()

# Customize data
config.data.BASE_PATH = '/path/to/roadsense/data'
config.data.OUTPUT_DIR = './results/experiment_1'
config.data.TRAIN_RATIO = 0.70

# Customize model
config.model.BACKBONE = 'resnet50'  # Options: mobilenetv2, vgg16, resnet50, efficientnet, densenet121, inceptionv3, convnext
config.model.SENSOR_EMBEDDING_DIM = 128
config.model.VIDEO_EMBEDDING_DIM = 256
config.model.DROPOUT = 0.3

# Customize training
config.training.BATCH_SIZE = 32
config.training.EPOCHS = 100
config.training.LEARNING_RATE = 1e-4
config.training.FOCAL_GAMMA = 2.0

# Customize federated learning
config.federated.NUM_CLIENTS = 5
config.federated.NUM_ROUNDS = 25
config.federated.LOCAL_EPOCHS = 10
config.federated.STRATEGY = 'scaffold'  # Options: fedavg, fedprox, fedadam, fednova, scaffold

# Run pipeline with custom config
if __name__ == '__main__':
    from main import main
    model, fed_model, cfg, results = main()
