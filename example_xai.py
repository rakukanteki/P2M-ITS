"""
Example: Using XAI modules for explainability
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from xai.manager import XAIManager
from xai.shap_explainer import SHAPExplainer
from xai.gradcam import MultimodalGradCAM, GradCAM


def example_shap_only():
    """Generate only SHAP explanations"""
    print("=" * 80)
    print("EXAMPLE: SHAP Sensor Feature Explanations")
    print("=" * 80)
    
    model = torch.nn.Linear(100, 3)
    device = torch.device('cpu')
    
    background_sensors = torch.randn(50, 100)
    test_sensors = torch.randn(10, 100)
    
    explainer = SHAPExplainer(
        model=model,
        device=device,
        background_data=background_sensors,
        test_data=test_sensors,
        num_samples=100
    )
    
    explanation = explainer.explain_sensor(num_samples=30)
    top_features = explainer.get_top_sensor_features(explanation, top_k=5)
    
    print("\nTop 5 Important Sensor Features:")
    for feature_idx, importance in top_features.items():
        print(f"  Feature {feature_idx}: {importance:.4f}")
    
    print("\n✓ SHAP explanation completed")


def example_gradcam_only():
    """Generate only GradCAM explanations"""
    print("\n" + "=" * 80)
    print("EXAMPLE: GradCAM Video Explanations")
    print("=" * 80)
    
    from models.multimodal import MultimodalModel
    
    model = MultimodalModel(sensor_input_dim=100, num_classes=3)
    device = torch.device('cpu')
    
    gradcam = MultimodalGradCAM(model, device)
    
    video = torch.randn(1, 30, 3, 224, 224)
    sensor = torch.randn(1, 100)
    target_class = 1
    
    explanation = gradcam.explain_multimodal(video, sensor, target_class)
    
    print(f"\nGenerated explanation for class {target_class}")
    print(f"  Video CAM shape: {explanation['video_cam'].shape if explanation['video_cam'] is not None else 'N/A'}")
    print(f"  Sensor importance shape: {explanation['sensor_importance'].shape if explanation['sensor_importance'] is not None else 'N/A'}")
    
    print("\n✓ GradCAM explanation completed")


def example_xai_manager():
    """Use XAI Manager to orchestrate both"""
    print("\n" + "=" * 80)
    print("EXAMPLE: XAI Manager (SHAP + GradCAM)")
    print("=" * 80)
    
    from models.multimodal import MultimodalModel
    from data.dataset import FolderDataset
    import torch.utils.data
    
    model = MultimodalModel(sensor_input_dim=100, num_classes=3)
    device = torch.device('cpu')
    class_names = ['Normal', 'Anomaly', 'Degraded']
    
    xai_mgr = XAIManager(model, device, './xai_outputs', class_names)
    
    samples = [
        {
            'sensor': torch.randn(100),
            'frames': torch.randn(30, 224, 224, 3),
            'label': 0,
            'folder_id': '1',
            'hash': 'dummy_hash'
        }
        for _ in range(20)
    ]
    
    dataset = FolderDataset(samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    
    print("\nGenerating SHAP explanations...")
    try:
        shap_features = xai_mgr.generate_shap_explanations(loader, loader)
        print(f"  Top features identified: {len(shap_features)}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")
    
    print("\nGenerating GradCAM explanations...")
    try:
        gradcam_stats = xai_mgr.generate_gradcam_explanations(loader, num_samples=2)
        print(f"  GradCAM samples generated: {gradcam_stats.get('samples_generated', 0)}")
    except Exception as e:
        print(f"  GradCAM skipped: {e}")
    
    print("\nSaving XAI report...")
    xai_mgr.save_xai_report()
    print(f"  Report saved to: ./xai_outputs/xai_report.json")
    
    print("\n✓ XAI Manager completed")


if __name__ == '__main__':
    print("\n🎯 XAI Module Examples\n")
    
    try:
        example_shap_only()
    except Exception as e:
        print(f"SHAP example failed: {e}")
    
    try:
        example_gradcam_only()
    except Exception as e:
        print(f"GradCAM example failed: {e}")
    
    try:
        example_xai_manager()
    except Exception as e:
        print(f"XAI Manager example failed: {e}")
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
