import numpy as np
import torch
import torch.nn.functional as F
import time
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from scipy import stats
from typing import Dict, List, Tuple, Any


class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader, class_names: List[str]) -> Dict[str, Any]:
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        inf_times = []
        
        with torch.no_grad():
            for batch in test_loader:
                video = batch['video'].to(self.device)
                sensor = batch['sensor'].to(self.device)
                labels = batch['label'].to(self.device)
                
                t0 = time.time()
                outputs = self.model(video, sensor)
                inf_times.append((time.time() - t0) / len(labels))
                
                probs = F.softmax(outputs, dim=1)
                _, pred = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        return self._compute_metrics(all_labels, all_preds, all_probs, 
                                     inf_times, class_names)
    
    def _compute_metrics(self, labels, preds, probs, inf_times, 
                        class_names) -> Dict[str, Any]:
        accuracy = 100. * accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        prec_cls = precision_score(labels, preds, average=None, zero_division=0)
        rec_cls = recall_score(labels, preds, average=None, zero_division=0)
        f1_cls = f1_score(labels, preds, average=None, zero_division=0)
        
        cm = confusion_matrix(labels, preds)
        
        n_cls = len(class_names)
        test_bin = np.eye(n_cls)[labels]
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_cls):
            fpr[i], tpr[i], _ = roc_curve(test_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fpr['micro'], tpr['micro'], _ = roc_curve(test_bin.ravel(), probs.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        
        inf_mean = np.mean([t * 1000 for t in inf_times])
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'precision': prec_cls,
            'recall': rec_cls,
            'f1_per_class': f1_cls,
            'confusion_matrix': cm,
            'roc_auc': {class_names[i]: float(roc_auc[i]) for i in range(n_cls)},
            'inference_time_ms': inf_mean,
            'predictions': preds,
            'labels': labels,
            'probabilities': probs,
            'class_names': class_names,
        }
    
    @staticmethod
    def compute_ci(values, conf=0.95):
        n = len(values)
        mean = np.mean(values)
        se = stats.sem(values)
        interval = se * stats.t.ppf((1 + conf) / 2., n - 1)
        return mean, mean - interval, mean + interval
    
    @staticmethod
    def bootstrap_ci(labels, preds, probs, n_boot=1000):
        boot_accs = []
        boot_f1s = []
        
        for _ in range(n_boot):
            idx = np.random.choice(len(labels), len(labels), replace=True)
            boot_accs.append(accuracy_score(labels[idx], preds[idx]) * 100)
            boot_f1s.append(f1_score(labels[idx], preds[idx], average='macro', 
                                    zero_division=0))
        
        acc_mean, acc_lo, acc_hi = ModelEvaluator.compute_ci(boot_accs)
        f1_mean, f1_lo, f1_hi = ModelEvaluator.compute_ci(boot_f1s)
        
        return {
            'accuracy': (acc_mean, acc_lo, acc_hi),
            'macro_f1': (f1_mean, f1_lo, f1_hi)
        }
