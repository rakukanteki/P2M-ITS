import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    BASE_PATH: str = '/kaggle/input/roadsense4m-multimodaldataset/RoadSenseMultimodalData'
    OUTPUT_DIR: str = '/kaggle/working/outputs'
    RANDOM_STATE: int = 42
    
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    WINDOW_SIZE: int = 50
    STRIDE: int = 25
    
    FRAMES_PER_VIDEO: int = 30
    IMG_SIZE: tuple = field(default_factory=lambda: (224, 224))
    
    UNCALIBRATED_COLS: List[str] = field(default_factory=lambda: [
        'magnetometerUncalibrated_z', 'magnetometerUncalibrated_y', 'magnetometerUncalibrated_x',
        'gyroscopeUncalibrated_z', 'gyroscopeUncalibrated_y', 'gyroscopeUncalibrated_x',
        'accelerometerUncalibrated_z', 'accelerometerUncalibrated_y', 'accelerometerUncalibrated_x'
    ])


@dataclass
class ModelConfig:
    SENSOR_EMBEDDING_DIM: int = 128
    VIDEO_EMBEDDING_DIM: int = 256
    NUM_CLASSES: int = 3
    BACKBONE: str = 'mobilenetv2'
    DROPOUT: float = 0.3


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    FOCAL_GAMMA: float = 2.0
    PATIENCE: int = 25
    MIN_DELTA: float = 0.001
    GRAD_CLIP: float = 1.0


@dataclass
class FederatedConfig:
    NUM_CLIENTS: int = 5
    NUM_ROUNDS: int = 25
    LOCAL_EPOCHS: int = 10
    CLIENT_FRACTION: float = 1.0
    STRATEGY: str = 'scaffold'


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    resume_from: str = None
    
    def __post_init__(self):
        os.makedirs(self.data.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.data.OUTPUT_DIR, 'xai'), exist_ok=True)
