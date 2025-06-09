import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.utils.file_utils import ensure_dir

class ResultsPlotter:
    def __init__(self, output_dir='results/figures'):
        self.output_dir = output_dir
        ensure_dir(output_dir)
        plt.style.use('default')
        
    def plot_training_history(self, history, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        axes[0].plot(epochs, history['train_losses'], 'b-', label='Training Loss')
        axes[0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
        axes[1].plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{self.output_dir}/training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics_dict[metric] for metric in metric_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metric_names, values, color=['blue', 'green', 'orange', 'red'])
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        ax.set_ylim(0, 1)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{self.output_dir}/metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_class_distribution(self, class_counts, save_path=None):
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(classes, counts, color=['skyblue', 'lightcoral'])
        
        ax.set_ylabel('Number of Images')
        ax.set_title('Class Distribution in Dataset')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{self.output_dir}/class_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_learning_curves(self, train_sizes, train_scores, val_scores, save_path=None):
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{self.output_dir}/learning_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_prediction_confidence(self, confidences, predictions, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        correct_conf = [conf for conf, pred in zip(confidences, predictions) if pred]
        incorrect_conf = [conf for conf, pred in zip(confidences, predictions) if not pred]
        
        ax.hist(correct_conf, bins=20, alpha=0.7, label='Correct Predictions', color='green')
        ax.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
        
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Confidence')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{self.output_dir}/prediction_confidence.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_summary_report(self, metrics, history, class_counts):
        fig = plt.figure(figsize=(16, 12))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
        ax1.set_title('Training History')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2 = fig.add_subplot(gs[0, 2])
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        ax2.bar(metric_names, values, color=['blue', 'green', 'orange', 'red'])
        ax2.set_title('Performance Metrics')
        ax2.set_ylabel('Score')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3 = fig.add_subplot(gs[1, :])
        epochs = range(1, len(history['train_accuracies']) + 1)
        ax3.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
        ax3.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy')
        ax3.set_title('Accuracy Progress')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True)
        
        ax4 = fig.add_subplot(gs[2, :])
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        ax4.bar(classes, counts, color=['skyblue', 'lightcoral'])
        ax4.set_title('Dataset Distribution')
        ax4.set_ylabel('Number of Images')
        
        plt.suptitle('Thermal Object Classification - Training Summary', fontsize=16)
        
        save_path = f'{self.output_dir}/training_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path