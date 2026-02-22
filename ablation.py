import os
import sys
import torch
import numpy as np
import json
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import Config, ModelConfig, FederatedConfig
from main import main as run_pipeline
from utils.logger import Logger


class AblationStudy:
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir
        self.results = {}
    
    def run_ablation(self, models: list, strategies: list):
        Logger.section("ABLATION STUDY - RUNNING EXPERIMENTS")
        
        combinations = list(product(models, strategies))
        Logger.progress(f"Total combinations: {len(combinations)}")
        
        for idx, (model, strategy) in enumerate(combinations, 1):
            exp_name = f"{model}_{strategy}"
            exp_dir = os.path.join(self.base_output_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            Logger.progress(f"\n[{idx}/{len(combinations)}] Running: {exp_name}")
            
            try:
                config = Config()
                config.data.OUTPUT_DIR = exp_dir
                config.model.BACKBONE = model
                config.federated.STRATEGY = strategy
                
                model_obj, fed_model, cfg, final_results = run_pipeline()
                self.results[exp_name] = final_results
                
                Logger.progress(f"✅ Completed: {exp_name}")
            
            except Exception as e:
                Logger.progress(f"❌ Failed: {exp_name} - {str(e)}")
                self.results[exp_name] = {'error': str(e)}
        
        self._save_ablation_results()
    
    def _save_ablation_results(self):
        results_path = os.path.join(self.base_output_dir, 'ablation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        Logger.progress(f"\nAblation results saved to: {results_path}")
    
    def print_summary(self):
        Logger.section("ABLATION STUDY SUMMARY")
        
        for exp_name, results in self.results.items():
            if 'error' in results:
                Logger.progress(f"{exp_name}: ❌ ERROR")
            else:
                cent_acc = results.get('centralized', {}).get('accuracy', 0)
                fed_acc = results.get('federated', {}).get('best_val_accuracy', 0)
                Logger.progress(f"{exp_name}: Centralized={cent_acc:.2f}% | Federated={fed_acc:.2f}%")


if __name__ == '__main__':
    models = [
        'mobilenetv2',
        'vgg16',
        'resnet50',
        'efficientnet',
        'densenet121',
        'inceptionv3',
        'convnext'
    ]
    
    strategies = [
        'fedavg',
        'fedprox',
        'fedadam',
        'fednova',
        'scaffold'
    ]
    
    output_dir = './ablation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    ablation = AblationStudy(output_dir)
    ablation.run_ablation(models, strategies)
    ablation.print_summary()
