import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, Tuple, List, Any
import cv2


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_video: torch.Tensor, input_sensor: torch.Tensor,
                     target_class: int) -> np.ndarray:
        """Generate Class Activation Map for video input."""
        self.model.eval()
        
        input_video = input_video.to(self.device)
        input_sensor = input_sensor.to(self.device)
        
        with torch.enable_grad():
            input_video.requires_grad = True
            
            outputs = self.model(input_video, input_sensor)
            target_score = outputs[0, target_class]
            
            self.model.zero_grad()
            target_score.backward()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Could not capture gradients or activations")
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def visualize_frame_cam(self, frame: np.ndarray, cam: np.ndarray, 
                           output_path: str, alpha: float = 0.5):
        """Overlay CAM on a frame."""
        frame_resized = cv2.resize(frame, (cam.shape[1], cam.shape[0]))
        
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        
        overlay = cv2.addWeighted(frame_resized, 1 - alpha, heatmap, alpha, 0)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(overlay)
        plt.axis('off')
        plt.title('GradCAM Visualization')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_temporal_cam(self, video: torch.Tensor, sensor: torch.Tensor,
                              target_class: int, output_path: str, 
                              num_frames: int = 6):
        """Visualize CAM for multiple frames."""
        B, T, C, H, W = video.shape
        
        fig, axes = plt.subplots(2, num_frames // 2, figsize=(16, 8))
        axes = axes.flatten()
        
        frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)
        
        for idx, frame_idx in enumerate(frame_indices):
            frame_video = video[:, frame_idx:frame_idx+1, :, :, :]
            if frame_video.shape[1] < 30:
                padding = torch.ones(B, 30 - frame_video.shape[1], C, H, W)
                frame_video = torch.cat([frame_video, padding], dim=1)
            
            cam = self.generate_cam(frame_video, sensor, target_class)
            
            frame_np = video[0, frame_idx].cpu().permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            
            frame_resized = cv2.resize(frame_np, (cam.shape[1], cam.shape[0]))
            heatmap = cv2.applyColorMap(
                (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(frame_resized, 0.5, heatmap, 0.5, 0)
            
            axes[idx].imshow(overlay)
            axes[idx].set_title(f'Frame {frame_idx}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class MultimodalGradCAM:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.video_cam = None
        self.sensor_importance = None
    
    def explain_multimodal(self, video: torch.Tensor, sensor: torch.Tensor,
                          target_class: int) -> Dict[str, Any]:
        """Generate explanations for both modalities."""
        self.model.eval()
        
        video = video.to(self.device)
        sensor = sensor.to(self.device)
        
        with torch.enable_grad():
            sensor_clone = sensor.clone().detach().requires_grad_(True)
            outputs = self.model(video, sensor_clone)
            target_score = outputs[0, target_class]
            
            self.model.zero_grad()
            target_score.backward()
            
            if sensor_clone.grad is not None:
                self.sensor_importance = sensor_clone.grad.abs().cpu().numpy()
        
        try:
            self.video_cam = GradCAM(
                self.model, self.model.video_encoder.backbone, self.device
            )
            cam = self.video_cam.generate_cam(video, sensor, target_class)
        except:
            cam = None
        
        return {
            'video_cam': cam,
            'sensor_importance': self.sensor_importance,
            'target_class': target_class
        }
    
    def plot_multimodal_explanation(self, explanation: Dict, video: torch.Tensor,
                                   sensor: torch.Tensor, output_path: str,
                                   class_names: List[str]):
        """Plot combined explanations for video and sensor."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if explanation['video_cam'] is not None:
            cam = explanation['video_cam']
            im0 = axes[0].imshow(cam, cmap='jet')
            axes[0].set_title('Video CAM')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0])
        
        if explanation['sensor_importance'] is not None:
            importance = explanation['sensor_importance'].squeeze()
            axes[1].barh(range(min(20, len(importance))), 
                        importance[:min(20, len(importance))])
            axes[1].set_title('Sensor Importance (Gradients)')
            axes[1].set_xlabel('Gradient Magnitude')
        
        plt.suptitle(f'Class: {class_names[explanation["target_class"]]}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
