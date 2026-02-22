import torch
import torch.nn as nn
import numpy as np
import shap
from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt


class SHAPExplainer:
    def __init__(self, model: nn.Module, device: torch.device, 
                 background_data: torch.Tensor, test_data: torch.Tensor,
                 num_samples: int = 100):
        self.model = model
        self.device = device
        self.background = background_data.to(device)
        self.test_data = test_data.to(device)
        self.num_samples = num_samples
    
    def _model_wrapper(self, x: np.ndarray, data_type: str = 'sensor') -> np.ndarray:
        """Wrapper for SHAP to work with the model."""
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            if data_type == 'sensor':
                dummy_video = torch.ones(
                    x_tensor.shape[0], 30, 3, 224, 224
                ).to(self.device)
                outputs = self.model(dummy_video, x_tensor)
            else:
                dummy_sensor = torch.ones(
                    x_tensor.shape[0], x_tensor.shape[1]
                ).to(self.device)
                outputs = self.model(x_tensor, dummy_sensor)
        
        return outputs.cpu().numpy()
    
    def explain_sensor(self, num_samples: int = None) -> Dict[str, Any]:
        """Generate SHAP explanations for sensor features."""
        num_samples = num_samples or self.num_samples
        background_np = self.background.cpu().numpy()[:num_samples]
        
        explainer = shap.KernelExplainer(
            lambda x: self._model_wrapper(x, 'sensor'),
            background_np
        )
        
        test_np = self.test_data.cpu().numpy()[:10]
        shap_values = explainer.shap_values(test_np)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'background': background_np,
            'test_data': test_np,
            'type': 'sensor'
        }
    
    def plot_sensor_summary(self, explanation: Dict, output_path: str, 
                           class_names: List[str]):
        """Plot SHAP summary for sensor features."""
        shap_values = explanation['shap_values']
        test_data = explanation['test_data']
        
        fig, axes = plt.subplots(1, len(class_names), figsize=(15, 5))
        if len(class_names) == 1:
            axes = [axes]
        
        for i, class_name in enumerate(class_names):
            ax = axes[i] if len(class_names) > 1 else axes[0]
            
            sv = shap_values[i] if isinstance(shap_values, list) else shap_values
            
            base_vals = sv.mean(axis=0)[:10]
            vals = sv[:10] if len(sv.shape) > 1 else sv
            
            ax.barh(range(len(base_vals)), np.abs(base_vals))
            ax.set_title(f'Class: {class_name}')
            ax.set_ylabel('Sensor Features')
            ax.set_xlabel('|SHAP value|')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_sensor_dependence(self, explanation: Dict, output_path: str,
                              feature_idx: int = 0):
        """Plot dependence between feature and model output."""
        shap_values = explanation['shap_values']
        test_data = explanation['test_data']
        
        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[0])
        else:
            shap_vals = shap_values
        
        plt.figure(figsize=(10, 6))
        plt.scatter(test_data[:, feature_idx], shap_vals[:, feature_idx], alpha=0.6)
        plt.xlabel(f'Feature {feature_idx} Value')
        plt.ylabel(f'SHAP Value')
        plt.title(f'Sensor Feature {feature_idx} Dependence')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def get_top_sensor_features(explanation: Dict, top_k: int = 5) -> Dict[int, float]:
        """Get top-k most important sensor features."""
        shap_values = explanation['shap_values']
        
        if isinstance(shap_values, list):
            sv = np.array(shap_values[0])
        else:
            sv = shap_values
        
        importance = np.abs(sv).mean(axis=0)
        top_indices = np.argsort(importance)[-top_k:][::-1]
        
        return {int(idx): float(importance[idx]) for idx in top_indices}
