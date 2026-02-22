import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List


class Visualizer:
    @staticmethod
    def plot_training_history(history: Dict, output_path: str):
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Centralized Training History', fontsize=14, fontweight='bold')
        
        axes[0].plot(epochs, history['train_loss'], label='Train')
        axes[0].plot(epochs, history['val_loss'], label='Val')
        axes[0].set(title='Loss', xlabel='Epoch')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(epochs, history['train_acc'], label='Train')
        axes[1].plot(epochs, history['val_acc'], label='Val')
        axes[1].set(title='Accuracy (%)', xlabel='Epoch')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        axes[2].plot(epochs, gap, color='purple')
        axes[2].axhline(0, color='k', ls='--', alpha=0.4)
        axes[2].axhline(10, color='red', ls='--', alpha=0.4, label='10 % threshold')
        axes[2].set(title='Overfitting Gap (Train−Val)', xlabel='Epoch')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                             output_path: str, title: str = 'Confusion Matrix'):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_roc_curves(fpr: Dict, tpr: Dict, roc_auc: Dict, 
                       class_names: List[str], output_path: str):
        plt.figure(figsize=(10, 7))
        n_cls = len(class_names)
        colors = ['blue', 'red', 'green']
        
        for i in range(n_cls):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'{class_names[i]} (AUC={roc_auc[i]:.4f})')
        
        plt.plot(fpr['micro'], tpr['micro'], 'navy', ls='--', lw=2,
                label=f'Micro-avg (AUC={roc_auc["micro"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_federated_convergence(history: Dict, output_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history['round'], history['val_loss'], 'b-o', lw=2)
        axes[0].set(title='Federated – Val Loss', xlabel='Round', ylabel='Loss')
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(history['round'], history['val_acc'], 'g-o', lw=2)
        axes[1].set(title='Federated – Val Accuracy (%)', xlabel='Round', ylabel='Accuracy')
        axes[1].grid(alpha=0.3)
        
        plt.suptitle('FL Convergence', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_client_accuracy(client_accs: List[float], output_path: str):
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(client_accs)), client_accs, color='steelblue', edgecolor='black')
        plt.axhline(np.mean(client_accs), color='red', ls='--', 
                   label=f'Mean={np.mean(client_accs):.1f}%')
        plt.xlabel('Client ID')
        plt.ylabel('Accuracy (%)')
        plt.title('Client-wise Accuracy')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
