import os
import torch
import json
from typing import Dict, List
from .shap_explainer import SHAPExplainer
from .gradcam import MultimodalGradCAM


class XAIManager:
    def __init__(self, model, device, output_dir: str, class_names: List[str]):
        self.model = model
        self.device = device
        self.output_dir = os.path.join(output_dir, 'xai')
        os.makedirs(self.output_dir, exist_ok=True)
        self.class_names = class_names
        self.explanations = {}
    
    def generate_shap_explanations(self, background_loader, test_loader, 
                                  num_background: int = 50, 
                                  num_test: int = 10) -> Dict:
        """Generate SHAP explanations for sensor features."""
        device = self.device
        
        background_sensors = []
        for batch in background_loader:
            background_sensors.append(batch['sensor'])
            if len(background_sensors) * len(batch['sensor']) >= num_background:
                break
        
        background_sensors = torch.cat(background_sensors, dim=0)[:num_background]
        
        test_sensors = []
        test_labels = []
        for batch in test_loader:
            test_sensors.append(batch['sensor'])
            test_labels.append(batch['label'])
            if len(test_sensors) * len(batch['sensor']) >= num_test:
                break
        
        test_sensors = torch.cat(test_sensors, dim=0)[:num_test]
        test_labels = torch.cat(test_labels, dim=0)[:num_test]
        
        shap_explainer = SHAPExplainer(
            self.model, device, background_sensors, test_sensors, num_samples=50
        )
        
        try:
            explanation = shap_explainer.explain_sensor(num_samples=30)
            
            shap_explainer.plot_sensor_summary(
                explanation,
                os.path.join(self.output_dir, 'shap_summary.png'),
                self.class_names
            )
            
            shap_explainer.plot_sensor_dependence(
                explanation,
                os.path.join(self.output_dir, 'shap_dependence.png'),
                feature_idx=0
            )
            
            top_features = shap_explainer.get_top_sensor_features(explanation, top_k=10)
            self.explanations['shap'] = {
                'top_features': top_features,
                'explanation_type': 'SHAP',
                'num_samples': len(test_sensors)
            }
            
            return top_features
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return {}
    
    def generate_gradcam_explanations(self, test_loader, 
                                     num_samples: int = 5) -> Dict:
        """Generate GradCAM explanations for video frames."""
        try:
            gradcam = MultimodalGradCAM(self.model, self.device)
            
            batch_idx = 0
            for batch in test_loader:
                if batch_idx >= num_samples:
                    break
                
                video = batch['video']
                sensor = batch['sensor']
                labels = batch['label']
                
                for i in range(min(len(video), self.model.classifier[0].in_features)):
                    target_class = int(labels[i].item())
                    
                    explanation = gradcam.explain_multimodal(
                        video[i:i+1], sensor[i:i+1], target_class
                    )
                    
                    output_file = os.path.join(
                        self.output_dir, 
                        f'gradcam_sample_{batch_idx}_{i}_{self.class_names[target_class]}.png'
                    )
                    
                    gradcam.plot_multimodal_explanation(
                        explanation, video[i:i+1], sensor[i:i+1],
                        output_file, self.class_names
                    )
                
                batch_idx += 1
            
            self.explanations['gradcam'] = {
                'num_samples': num_samples,
                'explanation_type': 'GradCAM',
                'samples_explained': batch_idx
            }
            
            return {'samples_generated': batch_idx}
        
        except Exception as e:
            print(f"GradCAM explanation failed: {e}")
            return {}
    
    def save_xai_report(self):
        """Save XAI explanations to JSON."""
        report_path = os.path.join(self.output_dir, 'xai_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.explanations, f, indent=2)
