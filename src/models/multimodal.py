import torch
import torch.nn as nn
from .encoders import SensorEncoder, BackboneFactory


class MultimodalModel(nn.Module):
    def __init__(self, sensor_input_dim: int, num_classes: int = 3,
                 backbone: str = 'mobilenetv2', sensor_embed_dim: int = 128,
                 video_embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.sensor_encoder = SensorEncoder(sensor_input_dim, sensor_embed_dim, dropout)
        self.video_encoder = BackboneFactory.create(backbone, video_embed_dim, dropout)
        
        fused_dim = sensor_embed_dim + video_embed_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, video: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        h_s = self.sensor_encoder(sensor)
        h_v = self.video_encoder(video)
        h_f = torch.cat([h_s, h_v], dim=1)
        o = self.classifier(h_f)
        o = torch.nan_to_num(o, nan=0.0, posinf=1.0, neginf=-1.0)
        return o
