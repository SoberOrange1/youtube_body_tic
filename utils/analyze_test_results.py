#!/usr/bin/env python3
"""
Test Results Analysis Script
Analyzes aggregated predictions CSV and generates comprehensive metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)
import argparse
import os
from pathlib import Path
import json


class TestResultsAnalyzer:
    """Comprehensive analysis of test results from aggregated CSV"""
    
    def __init__(self, csv_path: str, output_dir: str = None):
        """
        Initialize analyzer
        
        Args:
            csv_path: Path to aggregated_predictions.csv
            output_dir: Output directory for results (default: same as CSV)
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir else self.csv_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self._validate_data()
        
        # Extract arrays
        self.y_true = self.df['label'].values
        self.y_pred = self.df['pred'].values
        self.prob_non_tic = self.df['prob_non_tic'].values
        self.prob_tic = self.df['prob_tic'].values
        
        # Class names
        self.class_names = ['Non-tic', 'Tic']
        
        print(f"Loaded {len(self.df)} predictions")
        print(f"Label distribution: {np.bincount(self.y_true)}")
        
    def _validate_data(self):
        """Validate CSV data format"""
        required_cols = ['label', 'pred', 'prob_non_tic', 'prob_tic']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check label/pred values are 0 or 1
        if not all(self.df['label'].isin([0, 1])):
            raise ValueError("Labels must be 0 or 1")
        if not all(self.df['pred'].isin([0, 1])):
            raise ValueError("Predictions must be 0 or 1")
            
        print("Data validation passed")
    
    def compute_overall_metrics(self):
        """Compute overall metrics"""
        metrics = {
            'overall': {
                'accuracy': accuracy_score(self.y_true, self.y_pred),
                'precision_macro': precision_score(self.y_true, self.y_pred, average='macro'),
                'precision_weighted': precision_score(self.y_true, self.y_pred, average='weighted'),
                'recall_macro': recall_score(self.y_true, self.y_pred, average='macro'),
                'recall_weighted': recall_score(self.y_true, self.y_pred, average='weighted'),
                'f1_macro': f1_score(self.y_true, self.y_pred, average='macro'),
                'f1_weighted': f1_score(self.y_true, self.y_pred, average='weighted'),
            }
        }
        
        # ROC AUC if probabilities are available
        if not (np.isnan(self.prob_tic).any() or np.isnan(self.prob_non_tic).any()):
            try:
                metrics['overall']['roc_auc'] = roc_auc_score(self.y_true, self.prob_tic)
            except Exception as e:
                print(f"Warning: Could not compute ROC AUC: {e}")
                metrics['overall']['roc_auc'] = None
        else:
            metrics['overall']['roc_auc'] = None
            
        return metrics
    
    def compute_per_class_metrics(self):
        """Compute per-class metrics"""
        # Get per-class precision, recall, f1
        precision_per_class = precision_score(self.y_true, self.y_pred, average=None)
        recall_per_class = recall_score(self.y_true, self.y_pred, average=None)
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None)
        
        # Support (number of true instances per class)
        support = np.bincount(self.y_true, minlength=2)
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1_score': f1_per_class[i],
                'support': int(support[i])
            }
        
        return per_class_metrics
    
    def generate_confusion_matrix(self):
        """Generate and save confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        cm_path = self.output_dir / 'detailed_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {cm_path}")
        return cm
    
    def generate_roc_curve(self):
        """Generate and save ROC curve"""
        if np.isnan(self.prob_tic).any():
            print("Warning: Missing probability values, skipping ROC curve")
            return None
            
        try:
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(self.y_true, self.prob_tic)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5,
                    label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            roc_path = self.output_dir / 'detailed_roc_curve.png'
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"ROC curve saved to: {roc_path}")
            return {'auc': roc_auc, 'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            
        except Exception as e:
            print(f"Error generating ROC curve: {e}")
            return None
    
    def generate_metrics_summary_plot(self, overall_metrics, per_class_metrics):
        """Generate a summary visualization of all metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall metrics bar plot
        overall_values = [
            overall_metrics['overall']['accuracy'],
            overall_metrics['overall']['precision_weighted'],
            overall_metrics['overall']['recall_weighted'],
            overall_metrics['overall']['f1_weighted']
        ]
        metric_names = ['Accuracy', 'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1-Score\n(Weighted)']
        
        bars = ax1.bar(metric_names, overall_values, color=['skyblue', 'lightgreen', 'coral', 'gold'])
        ax1.set_title('Overall Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, overall_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Per-class metrics
        non_tic_metrics = [per_class_metrics['Non-tic'][m] for m in ['precision', 'recall', 'f1_score']]
        tic_metrics = [per_class_metrics['Tic'][m] for m in ['precision', 'recall', 'f1_score']]
        
        x = np.arange(3)
        width = 0.35
        
        ax2.bar(x - width/2, non_tic_metrics, width, label='Non-tic', color='lightblue')
        ax2.bar(x + width/2, tic_metrics, width, label='Tic', color='lightcoral')
        
        ax2.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Metrics', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # 3. Class distribution
        class_counts = [per_class_metrics['Non-tic']['support'], per_class_metrics['Tic']['support']]
        wedges, texts, autotexts = ax3.pie(class_counts, labels=self.class_names, autopct='%1.1f%%',
                                          colors=['lightblue', 'lightcoral'], startangle=90)
        ax3.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # 4. Confusion matrix (normalized)
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax4.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax4.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        tick_marks = np.arange(len(self.class_names))
        ax4.set_xticks(tick_marks)
        ax4.set_yticks(tick_marks)
        ax4.set_xticklabels(self.class_names)
        ax4.set_yticklabels(self.class_names)
        ax4.set_xlabel('Predicted Label', fontsize=12)
        ax4.set_ylabel('True Label', fontsize=12)
        
        # Add text annotations
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax4.text(j, i, f'{cm_normalized[i, j]:.3f}',
                        ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        summary_path = self.output_dir / 'detailed_metrics_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics summary plot saved to: {summary_path}")
    
    def save_detailed_report(self, overall_metrics, per_class_metrics, cm, roc_info=None):
        """Save detailed text report"""
        report_path = self.output_dir / 'detailed_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED TEST RESULTS ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Non-tic samples: {per_class_metrics['Non-tic']['support']} ({per_class_metrics['Non-tic']['support']/len(self.df)*100:.1f}%)\n")
            f.write(f"Tic samples: {per_class_metrics['Tic']['support']} ({per_class_metrics['Tic']['support']/len(self.df)*100:.1f}%)\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {overall_metrics['overall']['accuracy']:.4f} ({overall_metrics['overall']['accuracy']*100:.2f}%)\n")
            f.write(f"Precision (Macro): {overall_metrics['overall']['precision_macro']:.4f}\n")
            f.write(f"Precision (Weighted): {overall_metrics['overall']['precision_weighted']:.4f}\n")
            f.write(f"Recall (Macro): {overall_metrics['overall']['recall_macro']:.4f}\n")
            f.write(f"Recall (Weighted): {overall_metrics['overall']['recall_weighted']:.4f}\n")
            f.write(f"F1-Score (Macro): {overall_metrics['overall']['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {overall_metrics['overall']['f1_weighted']:.4f}\n")
            if overall_metrics['overall']['roc_auc'] is not None:
                f.write(f"ROC AUC: {overall_metrics['overall']['roc_auc']:.4f}\n")
            f.write("\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 40 + "\n")
            for class_name in self.class_names:
                metrics = per_class_metrics[class_name]
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                f.write(f"  Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n")
                f.write(f"  Support: {metrics['support']}\n\n")
            
            # Confusion matrix
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")
            f.write("Predicted →   Non-tic    Tic\n")
            f.write("Actual ↓\n")
            for i, true_class in enumerate(self.class_names):
                f.write(f"{true_class:8s}      {cm[i, 0]:6d}   {cm[i, 1]:6d}\n")
            f.write("\n")
            
            # Additional analysis
            f.write("ADDITIONAL ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Classification errors
            false_positives = np.sum((self.y_true == 0) & (self.y_pred == 1))
            false_negatives = np.sum((self.y_true == 1) & (self.y_pred == 0))
            f.write(f"False Positives (Non-tic → Tic): {false_positives}\n")
            f.write(f"False Negatives (Tic → Non-tic): {false_negatives}\n")
            
            # Error rates
            fpr = false_positives / per_class_metrics['Non-tic']['support'] if per_class_metrics['Non-tic']['support'] > 0 else 0
            fnr = false_negatives / per_class_metrics['Tic']['support'] if per_class_metrics['Tic']['support'] > 0 else 0
            f.write(f"False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)\n")
            f.write(f"False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)\n\n")
            
            f.write("="*80 + "\n")
            f.write("Report generated successfully\n")
            f.write("="*80 + "\n")
        
        print(f"Detailed report saved to: {report_path}")
    
    def save_metrics_json(self, overall_metrics, per_class_metrics, roc_info=None):
        """Save metrics in JSON format"""
        json_path = self.output_dir / 'detailed_metrics.json'
        
        results = {
            'dataset_info': {
                'total_samples': len(self.df),
                'class_distribution': {
                    'non_tic': per_class_metrics['Non-tic']['support'],
                    'tic': per_class_metrics['Tic']['support']
                }
            },
            'overall_metrics': overall_metrics['overall'],
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix(self.y_true, self.y_pred).tolist()
        }
        
        if roc_info:
            results['roc_curve'] = roc_info
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Metrics JSON saved to: {json_path}")
    
    def run_full_analysis(self):
        """Run complete analysis and generate all outputs"""
        print("\n" + "="*60)
        print("STARTING DETAILED TEST RESULTS ANALYSIS")
        print("="*60)
        
        # Compute metrics
        print("\n1. Computing overall metrics...")
        overall_metrics = self.compute_overall_metrics()
        
        print("2. Computing per-class metrics...")
        per_class_metrics = self.compute_per_class_metrics()
        
        # Generate visualizations
        print("3. Generating confusion matrix...")
        cm = self.generate_confusion_matrix()
        
        print("4. Generating ROC curve...")
        roc_info = self.generate_roc_curve()
        
        print("5. Generating metrics summary...")
        self.generate_metrics_summary_plot(overall_metrics, per_class_metrics)
        
        # Save reports
        print("6. Saving detailed report...")
        self.save_detailed_report(overall_metrics, per_class_metrics, cm, roc_info)
        
        print("7. Saving metrics JSON...")
        self.save_metrics_json(overall_metrics, per_class_metrics, roc_info)
        
        # Print summary to console
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {overall_metrics['overall']['accuracy']:.4f} ({overall_metrics['overall']['accuracy']*100:.2f}%)")
        print(f"Overall F1-Score (Weighted): {overall_metrics['overall']['f1_weighted']:.4f}")
        if overall_metrics['overall']['roc_auc'] is not None:
            print(f"ROC AUC: {overall_metrics['overall']['roc_auc']:.4f}")
        
        print("\nPer-class metrics:")
        for class_name in self.class_names:
            metrics = per_class_metrics[class_name]
            print(f"  {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        print(f"\nAll results saved to: {self.output_dir}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze test results from aggregated CSV')
    parser.add_argument('csv_path', type=str, 
                       help='Path to aggregated_predictions.csv file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as CSV file)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return
    
    try:
        # Create analyzer and run analysis
        analyzer = TestResultsAnalyzer(args.csv_path, args.output_dir)
        analyzer.run_full_analysis()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()