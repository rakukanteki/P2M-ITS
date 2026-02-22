import torch
import torch.nn as nn
import torchvision.models as models


class SensorEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.net(x)


class BackboneFactory:
    @staticmethod
    def create_mobilenetv2(embed_dim: int = 256, dropout: float = 0.3) -> nn.Module:
        mobilenet = models.mobilenet_v2(pretrained=True)
        backbone = mobilenet.features
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        projection = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        return VideoEncoder(backbone, 1280, projection)
    
    @staticmethod
    def create_vgg16(embed_dim: int = 256, dropout: float = 0.3) -> nn.Module:
        vgg = models.vgg16(pretrained=True)
        backbone = vgg.features
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        return VideoEncoder(backbone, 512, projection)
    
    @staticmethod
    def create_resnet50(embed_dim: int = 256, dropout: float = 0.3) -> nn.Module:
        resnet = models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        projection = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        return VideoEncoder(backbone, 2048, projection, remove_pool=True)
    
    @staticmethod
    def create_efficientnet(embed_dim: int = 256, dropout: float = 0.3) -> nn.Module:
        efficientnet = models.efficientnet_b0(pretrained=True)
        backbone = efficientnet.features
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        projection = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        return VideoEncoder(backbone, 1280, projection)
    
    @staticmethod
    def create_densenet121(embed_dim: int = 256, dropout: float = 0.3) -> nn.Module:
        densenet = models.densenet121(pretrained=True)
        backbone = densenet.features
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        projection = nn.Sequential(
            nn.Linear(1024, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        return VideoEncoder(backbone, 1024, projection)
    
    @staticmethod
    def create_inceptionv3(embed_dim: int = 256, dropout: float = 0.3) -> nn.Module:
        inception = models.inception_v3(pretrained=True)
        backbone = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        projection = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        return VideoEncoder(backbone, 2048, projection)
    
    @staticmethod
    def create_convnext_tiny(embed_dim: int = 256, dropout: float = 0.3) -> nn.Module:
        try:
            convnext = models.convnext_tiny(pretrained=True)
        except:
            convnext = models.convnext_tiny(weights='DEFAULT')
        
        backbone = convnext.features
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        return VideoEncoder(backbone, 768, projection)
    
    @staticmethod
    def create(backbone_name: str, embed_dim: int = 256, dropout: float = 0.3) -> nn.Module:
        factory_map = {
            'mobilenetv2': BackboneFactory.create_mobilenetv2,
            'vgg16': BackboneFactory.create_vgg16,
            'resnet50': BackboneFactory.create_resnet50,
            'efficientnet': BackboneFactory.create_efficientnet,
            'densenet121': BackboneFactory.create_densenet121,
            'inceptionv3': BackboneFactory.create_inceptionv3,
            'convnext': BackboneFactory.create_convnext_tiny,
        }
        
        if backbone_name not in factory_map:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        return factory_map[backbone_name](embed_dim, dropout)


class VideoEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, projection: nn.Module, 
                 remove_pool: bool = False):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.remove_pool = remove_pool
        
        if not remove_pool:
            self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.projection = projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)
        x = x.view(B * T, C, H, W)
        
        feats = self.backbone(x)
        
        if not self.remove_pool:
            feats = self.pool(feats)
            feats = feats.view(B, T, -1)
        else:
            if len(feats.shape) == 4:
                feats = feats.mean(dim=(2, 3))
            feats = feats.view(B, T, -1)
        
        feats = feats.mean(dim=1)
        return self.projection(feats)
