import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from src.utils.file_utils import ensure_dir

class ConfusionMatrixVisualizer:
    def __init__(self, class_names=['Human', 'Dog'], output_dir='results/figures'):
        self.class_names = class_names
        self.output_dir = output_dir
        ensure_dir(output_dir)
        
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path is None:
            suffix = '_normalized' if normalize else ''
            save_path = f'{self.output_dir}/confusion_matrix{suffix}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_classification_report(self, y_true, y_pred, save_path=None):
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        metrics_data = []
        for class_name in self.class_names:
            metrics_data.append([
                class_name,
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score'],
                report[class_name]['support']
            ])
        
        metrics_data.append([
            'macro avg',
            report['macro avg']['precision'],
            report['macro avg']['recall'],
            report['macro avg']['f1-score'],
            report['macro avg']['support']
        ])
        
        metrics_data.append([
            'weighted avg',
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            report['weighted avg']['f1-score'],
            report['weighted avg']['support']
        ])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Precision', 'Recall', 'F1-Score']
        x = np.arange(len(self.class_names))
        width = 0.25
        
        precision_scores = [report[cls]['precision'] for cls in self.class_names]
        recall_scores = [report[cls]['recall'] for cls in self.class_names]
        f1_scores = [report[cls]['f1-score'] for cls in self.class_names]
        
        ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Classification Report by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
            ax.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{self.output_dir}/classification_report.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_detailed_analysis(self, y_true, y_pred, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix (Counts)')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix (Normalized)')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
        
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        precision_scores = [report[cls]['precision'] for cls in self.class_names]
        recall_scores = [report[cls]['recall'] for cls in self.class_names]
        f1_scores = [report[cls]['f1-score'] for cls in self.class_names]
        
        axes[1, 0].bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        axes[1, 0].bar(x, recall_scores, width, label='Recall', alpha=0.8)
        axes[1, 0].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Per-Class Metrics')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(self.class_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        overall_metrics = ['accuracy', 'macro avg', 'weighted avg']
        overall_scores = [report['accuracy'], report['macro avg']['f1-score'], report['weighted avg']['f1-score']]
        
        axes[1, 1].bar(overall_metrics, overall_scores, color=['green', 'orange', 'blue'], alpha=0.8)
        axes[1, 1].set_title('Overall Performance')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for i, score in enumerate(overall_scores):
            axes[1, 1].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{self.output_dir}/detailed_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path