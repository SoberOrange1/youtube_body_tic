#!/usr/bin/env python3
"""
Tester for Tic Motion ViT Processor
Tests multiple folds within an experiment and aggregates results
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from motion_patches_trainer import TicMotionPatchesDataset, MotionPatchesTicClassifierWithLoRA

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TicModelTester:
    """
    Tester for evaluating trained tic classification models
    Supports testing multiple folds within an experiment
    """
    
    def __init__(self, exp_dir: str, device: str = None, ckpt_type: str = 'best', checkpoint_epoch: int = None, checkpoint_map: dict = None):
        """
        Args:
            exp_dir: Path to experiment directory (e.g., outputs/exp_001)
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None
            ckpt_type: Checkpoint type to load ('best', 'last', or 'epoch')
            checkpoint_epoch: Epoch number to load when ckpt_type='epoch'
            checkpoint_map: Optional per-fold epoch mapping
        """
        self.exp_dir = Path(exp_dir)
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.ckpt_type = ckpt_type  # 'best', 'last', or 'epoch'
        self.checkpoint_epoch = checkpoint_epoch  # Uniform epoch for all folds if ckpt_type == 'epoch'
        self.checkpoint_map = checkpoint_map or {}  # Optional per-fold epoch mapping
        
        # Find all fold directories in the experiment
        self.fold_dirs = self._find_fold_directories()
        
        if not self.fold_dirs:
            raise ValueError(f"No fold directories found in {exp_dir}")
        
        log.info(f"Found {len(self.fold_dirs)} folds in experiment: {self.exp_dir.name}")
        log.info(f"Using device: {self.device}")
    
    def _find_fold_directories(self):
        """Find all fold directories within the experiment"""
        fold_dirs = []
        
        for item in self.exp_dir.iterdir():
            if item.is_dir() and ('fold' in item.name.lower() or 'train' in item.name.lower()):
                # Check if it has the required files
                if (item / 'best_model.pt').exists() and (item / 'video_split_info.json').exists():
                    fold_dirs.append(item)
        
        # Sort by directory name
        fold_dirs.sort(key=lambda x: x.name)
        
        return fold_dirs
    
    def load_fold_config(self, fold_dir: Path):
        """Load configuration for a specific fold"""
        # Load video split info
        with open(fold_dir / 'video_split_info.json', 'r') as f:
            split_info = json.load(f)
        
        # Load training config
        config_file = fold_dir / 'training_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                training_config = json.load(f)
        else:
            training_config = {}
        
        return split_info, training_config
    
    def _select_checkpoint_for_fold(self, fold_dir: Path):
        """Select checkpoint file name for a given fold.
        Priority: checkpoint_map (exact match) > checkpoint_map (substring match) > ckpt_type best/last/epoch.
        Returns (ckpt_name, source_str)."""
        # 1) If a mapping is provided, try exact key match
        if self.checkpoint_map:
            # Exact match by full folder name
            if fold_dir.name in self.checkpoint_map:
                entry = self.checkpoint_map[fold_dir.name]
                ckpt = self._interpret_checkpoint_entry(entry, fold_dir.name)
                if ckpt:
                    return ckpt, f"map:exact:{fold_dir.name}"
            # Substring match: allow keys without timestamp to match
            best_key = None
            for key in self.checkpoint_map.keys():
                if key and key in fold_dir.name:
                    if best_key is None or len(key) > len(best_key):
                        best_key = key
            if best_key is not None:
                entry = self.checkpoint_map[best_key]
                ckpt = self._interpret_checkpoint_entry(entry, best_key)
                if ckpt:
                    return ckpt, f"map:substr:{best_key}"
        
        # 2) Fall back to ckpt_type
        if self.ckpt_type == 'best':
            return 'best_model.pt', 'ckpt_type:best'
        if self.ckpt_type == 'last':
            return 'last_model.pt', 'ckpt_type:last'
        if self.ckpt_type == 'epoch':
            if self.checkpoint_epoch is None:
                raise ValueError("ckpt_type is 'epoch' but --checkpoint_epoch not provided and no mapping matched")
            return f"checkpoint_epoch_{int(self.checkpoint_epoch)}.pt", f"ckpt_type:epoch:{int(self.checkpoint_epoch)}"
        raise ValueError(f"Unknown ckpt_type: {self.ckpt_type}")
    
    def _interpret_checkpoint_entry(self, entry, key_for_error: str):
        """Interpret a checkpoint_map entry value and return a checkpoint file name or raise error."""
        if isinstance(entry, str):
            val = entry.strip().lower()
            if val == 'best':
                return 'best_model.pt'
            if val == 'last':
                return 'last_model.pt'
            if val.isdigit():
                return f"checkpoint_epoch_{int(val)}.pt"
            raise ValueError(f"Invalid checkpoint_map value for key '{key_for_error}': {entry} (use integer epoch, 'best', or 'last')")
        if isinstance(entry, (int, float)):
            return f"checkpoint_epoch_{int(entry)}.pt"
        raise ValueError(f"Invalid checkpoint_map type for key '{key_for_error}': {type(entry)}")
    
    def load_model(self, fold_dir: Path, training_config: dict):
        """Load trained model from checkpoint (best, last, or epoch-based)"""
        # Build model configuration
        model_config = training_config.get('model_config', {})
        lora_config = training_config.get('lora_config', {})
        prediction_mode = model_config.get('prediction_mode', 'segment')
        
        # Create model with same architecture
        model = MotionPatchesTicClassifierWithLoRA(
            encoder_type=model_config.get('encoder_type', 'basic'),
            model_name=model_config.get('model_name', 'vit_base_patch16_224_in21k'),
            pretrained=False,  # Don't load pretrained weights
            patch_size=model_config.get('patch_size', 16),
            max_frames=model_config.get('max_frames', 64),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.0),
            lora_r=lora_config.get('lora_r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            feature_dim=model_config.get('feature_dim', 7),
            prediction_mode=prediction_mode
        ).to(self.device)
        
        # Select checkpoint using mapping (with fallback to ckpt_type)
        ckpt_name, source = self._select_checkpoint_for_fold(fold_dir)
        checkpoint_path = fold_dir / ckpt_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for fold {fold_dir.name}: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning(f"Missing keys when loading {ckpt_name} (first 5): {missing[:5]}")
        if unexpected:
            log.warning(f"Unexpected keys when loading {ckpt_name} (first 5): {unexpected[:5]}")
        model.eval()
        
        log.info(f"Loaded model from: {checkpoint_path} (source: {source})")
        
        return model
    
    def create_test_loader(self, pose_data_dir: str, annotation_dir: str, 
                          test_videos: list, max_frames: int, encoder_type: str, 
                          batch_size: int = 32):
        """Create test data loader"""
        test_dataset = TicMotionPatchesDataset(
            pose_data_dir=pose_data_dir,
            annotation_dir=annotation_dir,
            video_ids=test_videos,
            max_frames=max_frames,
            feature_mode=encoder_type
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        log.info(f"Test dataset: {len(test_dataset)} segments from {len(test_videos)} videos")
        
        return test_loader, test_dataset
    
    @torch.no_grad()
    def evaluate_fold(self, model, test_loader):
        """Evaluate model on test data (supports segment and frame prediction modes)"""
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            out = model(batch_x)
            if isinstance(out, tuple) and len(out) == 2:
                _, logits = out
            else:
                logits = out
            
            if logits.dim() == 3:  # Frame-level: (B, T, C)
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)  # (B, T)
                all_preds.extend(preds.cpu().numpy().reshape(-1))
                all_labels.extend(batch_y.cpu().numpy().reshape(-1))
                all_probs.extend(probs.cpu().numpy().reshape(-1, probs.size(-1)))
            else:  # Segment-level: (B, C)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        return metrics
    
    def test_single_fold(self, fold_dir: Path, test_videos: list = None, 
                        pose_data_dir: str = None, annotation_dir: str = None):
        """Test a single fold"""
        log.info(f"\n{'='*70}")
        log.info(f"Testing fold: {fold_dir.name}")
        log.info(f"{'='*70}")
        
        # Load fold configuration
        split_info, training_config = self.load_fold_config(fold_dir)
        
        # Determine test videos
        if test_videos is None:
            test_videos = split_info.get('test_videos', []) or split_info.get('val_videos', [])
        
        # Log val/test IDs for this fold
        log.info(f"Fold {fold_dir.name} - Val videos: {split_info.get('val_videos', [])}")
        log.info(f"Fold {fold_dir.name} - Test videos: {test_videos}")
        
        if not test_videos:
            log.warning(f"No test videos specified for fold {fold_dir.name}, skipping")
            return None
        
        log.info(f"Test videos: {test_videos}")
        
        # Get data directories
        if pose_data_dir is None:
            paths = training_config.get('paths', {})
            pose_data_dir = paths.get('pose_data_dir', '')
        
        if annotation_dir is None:
            paths = training_config.get('paths', {})
            annotation_dir = paths.get('annotation_dir', '')
        
        # Get model configuration
        model_config = training_config.get('model_config', {})
        data_config = training_config.get('data_config', {})
        
        # Load model
        model = self.load_model(fold_dir, training_config)
        
        # Create test loader
        test_loader, test_dataset = self.create_test_loader(
            pose_data_dir=pose_data_dir,
            annotation_dir=annotation_dir,
            test_videos=test_videos,
            max_frames=data_config.get('max_frames', 64),
            encoder_type=model_config.get('encoder_type', 'basic'),
            batch_size=32
        )
        
        # Evaluate
        metrics = self.evaluate_fold(model, test_loader)
        
        # Print results
        log.info(f"\nTest Results for {fold_dir.name}:")
        log.info(f"  Accuracy:  {metrics['accuracy']:.2f}%")
        log.info(f"  Precision: {metrics['precision']:.2f}%")
        log.info(f"  Recall:    {metrics['recall']:.2f}%")
        log.info(f"  F1-Score:  {metrics['f1_score']:.2f}%")
        
        # Save fold results
        self._save_fold_results(fold_dir, metrics, test_videos, split_info)
        
        return {
            'fold_name': fold_dir.name,
            'test_videos': test_videos,
            'metrics': metrics,
            'split_info': split_info
        }
    
    def test_all_folds(self, test_videos_per_fold: dict = None,
                       pose_data_dir: str = None, annotation_dir: str = None):
        """
        Test all folds in the experiment
        
        Args:
            test_videos_per_fold: Dict mapping fold names to test video lists
                                 If None, uses videos from split_info
            pose_data_dir: Override pose data directory
            annotation_dir: Override annotation directory
        
        Returns:
            Aggregated results across all folds
        """
        log.info(f"\n{'='*70}")
        log.info(f"Testing Experiment: {self.exp_dir.name}")
        log.info(f"Total Folds: {len(self.fold_dirs)}")
        log.info(f"{'='*70}\n")
        
        all_fold_results = []
        
        for fold_dir in self.fold_dirs:
            # Get test videos for this fold
            if test_videos_per_fold and fold_dir.name in test_videos_per_fold:
                test_videos = test_videos_per_fold[fold_dir.name]
            else:
                test_videos = None  # Will be determined from split_info
            
            # Test this fold
            fold_result = self.test_single_fold(
                fold_dir=fold_dir,
                test_videos=test_videos,
                pose_data_dir=pose_data_dir,
                annotation_dir=annotation_dir
            )
            
            if fold_result:
                all_fold_results.append(fold_result)
        
        if not all_fold_results:
            log.error("No folds were successfully tested!")
            return None
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_fold_results)
        
        # Save aggregated results JSON and plot
        self._save_aggregated_results(aggregated_results)
        
        # Also save aggregated confusion matrix, classification report (per-class and overall),
        # predictions CSV, and ROC AUC curve into exp_dir/test
        self._save_aggregated_test_artifacts(all_fold_results)
        
        return aggregated_results
    
    def _aggregate_results(self, all_fold_results):
        """Aggregate results from all folds"""
        log.info(f"\n{'='*70}")
        log.info("AGGREGATED RESULTS ACROSS ALL FOLDS")
        log.info(f"{'='*70}")
        
        # Collect metrics from all folds
        accuracies = [r['metrics']['accuracy'] for r in all_fold_results]
        precisions = [r['metrics']['precision'] for r in all_fold_results]
        recalls = [r['metrics']['recall'] for r in all_fold_results]
        f1_scores = [r['metrics']['f1_score'] for r in all_fold_results]
        
        # Calculate statistics
        aggregated = {
            'experiment_name': self.exp_dir.name,
            'num_folds': len(all_fold_results),
            'timestamp': datetime.now().isoformat(),
            'per_fold_results': [
                {
                    'fold_name': r['fold_name'],
                    'test_videos': r['test_videos'],
                    'accuracy': r['metrics']['accuracy'],
                    'precision': r['metrics']['precision'],
                    'recall': r['metrics']['recall'],
                    'f1_score': r['metrics']['f1_score']
                }
                for r in all_fold_results
            ],
            'summary': {
                'accuracy': {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'min': float(np.min(accuracies)),
                    'max': float(np.max(accuracies))
                },
                'precision': {
                    'mean': float(np.mean(precisions)),
                    'std': float(np.std(precisions))
                },
                'recall': {
                    'mean': float(np.mean(recalls)),
                    'std': float(np.std(recalls))
                },
                'f1_score': {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores))
                }
            }
        }
        
        # Print summary
        log.info(f"\nAccuracy:  {aggregated['summary']['accuracy']['mean']:.2f}% ± {aggregated['summary']['accuracy']['std']:.2f}%")
        log.info(f"Precision: {aggregated['summary']['precision']['mean']:.2f}% ± {aggregated['summary']['precision']['std']:.2f}%")
        log.info(f"Recall:    {aggregated['summary']['recall']['mean']:.2f}% ± {aggregated['summary']['recall']['std']:.2f}%")
        log.info(f"F1-Score:  {aggregated['summary']['f1_score']['mean']:.2f}% ± {aggregated['summary']['f1_score']['std']:.2f}%")
        log.info(f"\nBest Fold: {all_fold_results[np.argmax(accuracies)]['fold_name']} (Acc: {np.max(accuracies):.2f}%)")
        log.info(f"Worst Fold: {all_fold_results[np.argmin(accuracies)]['fold_name']} (Acc: {np.min(accuracies):.2f}%)")
        log.info(f"{'='*70}\n")
        
        return aggregated
    
    def _save_fold_results(self, fold_dir: Path, metrics: dict, test_videos: list, split_info: dict):
        """Save test results for a single fold"""
        # Save detailed test results
        test_results = {
            'fold_name': fold_dir.name,
            'test_videos': test_videos,
            'split_info': split_info,
            'metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(fold_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Save confusion matrix
        class_names = ['Non-tic', 'Tic']
        cm = confusion_matrix(metrics['labels'], metrics['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Test Confusion Matrix - {fold_dir.name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(fold_dir / 'test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save classification report
        report = classification_report(metrics['labels'], metrics['predictions'], 
                                      target_names=class_names)
        
        with open(fold_dir / 'test_classification_report.txt', 'w') as f:
            f.write(f"Test Results for {fold_dir.name}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Test Videos: {test_videos}\n\n")
            f.write(report)
        
        log.info(f"Test results saved to: {fold_dir}")
    
    def _save_aggregated_results(self, aggregated_results: dict):
        """Save aggregated results for the entire experiment"""
        # Save JSON results
        results_file = self.exp_dir / 'aggregated_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        
        log.info(f"Aggregated results saved to: {results_file}")
        
        # Create summary visualization
        self._plot_fold_comparison(aggregated_results)
    
    def _save_aggregated_test_artifacts(self, all_fold_results: list):
        """Create exp_dir/test and save aggregated confusion matrix, classification report, predictions CSV, and ROC AUC curve."""
        # Create test output directory under the experiment directory
        test_dir = self.exp_dir / 'test'
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Concatenate predictions, labels, and probabilities across all folds
        all_labels = np.concatenate([r['metrics']['labels'] for r in all_fold_results])
        all_preds = np.concatenate([r['metrics']['predictions'] for r in all_fold_results])
        # Probabilities may be missing in edge cases; guard accordingly
        if all('probabilities' in r['metrics'] for r in all_fold_results):
            all_probs = np.concatenate([r['metrics']['probabilities'] for r in all_fold_results], axis=0)
        else:
            all_probs = None
        
        # Save overall confusion matrix image
        class_names = ['Non-tic', 'Tic']
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Aggregated Confusion Matrix (All Folds)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(test_dir / 'aggregated_confusion_matrix.png', dpi=300)
        plt.close()
        
        # Save classification report (includes per-class and overall metrics)
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        with open(test_dir / 'aggregated_classification_report.txt', 'w', encoding='utf-8') as f:
            f.write("Aggregated Test Results (All Folds)\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        # Save aggregated predictions to CSV (label, pred, prob_non_tic, prob_tic)
        csv_path = test_dir / 'aggregated_predictions.csv'
        with open(csv_path, 'w', encoding='utf-8') as fw:
            # Write header
            fw.write('label,pred,prob_non_tic,prob_tic\n')
            # Write rows; if probabilities are missing, write NaNs
            if all_probs is not None:
                for i in range(len(all_labels)):
                    p0 = float(all_probs[i, 0]) if all_probs.shape[1] > 0 else float('nan')
                    p1 = float(all_probs[i, 1]) if all_probs.shape[1] > 1 else float('nan')
                    fw.write(f"{int(all_labels[i])},{int(all_preds[i])},{p0:.6f},{p1:.6f}\n")
            else:
                for i in range(len(all_labels)):
                    fw.write(f"{int(all_labels[i])},{int(all_preds[i])},,\n")
        
        # Plot ROC AUC curve (binary) if probabilities are available
        if all_probs is not None and all_probs.shape[1] == 2:
            # Use probability of the positive class (Tic) as scores
            y_true = all_labels
            y_score = all_probs[:, 1]
            try:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Aggregated ROC Curve (All Folds)')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(test_dir / 'aggregated_roc_auc.png', dpi=300)
                plt.close()
                # Also save ROC AUC score in a small txt
                with open(test_dir / 'aggregated_roc_auc.txt', 'w', encoding='utf-8') as f:
                    f.write(f"ROC AUC: {roc_auc:.6f}\n")
            except Exception as e:
                log.warning(f"Failed to compute ROC AUC: {e}")
        else:
            log.info("Probabilities not available or not binary; skipping ROC AUC plot.")
    
    def _plot_fold_comparison(self, aggregated_results: dict):
        """Create visualization comparing all folds"""
        per_fold = aggregated_results['per_fold_results']
        
        fold_names = [r['fold_name'] for r in per_fold]
        accuracies = [r['accuracy'] for r in per_fold]
        precisions = [r['precision'] for r in per_fold]
        recalls = [r['recall'] for r in per_fold]
        f1_scores = [r['f1_score'] for r in per_fold]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(fold_names))
        width = 0.2
        
        ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score (%)')
        ax.set_title(f'Test Results Comparison - {aggregated_results["experiment_name"]}')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(len(fold_names))], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Add mean line
        mean_acc = aggregated_results['summary']['accuracy']['mean']
        ax.axhline(y=mean_acc, color='r', linestyle='--', alpha=0.5, 
                  label=f'Mean Acc: {mean_acc:.2f}%')
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'fold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Fold comparison plot saved to: {self.exp_dir / 'fold_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Test tic classification models by experiment')
    
    parser.add_argument('--exp_dir', type=str, default=None,
                       help='Path to experiment directory (e.g., outputs/exp_001)')
    
    parser.add_argument('--exp_num', type=str, default=None,
                       help='Experiment number (e.g., exp_001). Overrides exp_dir')
    
    parser.add_argument('--outputs_root', type=str, default=r'A:\youtube_body\data_folder\outputs',
                       help='Root outputs directory')
    
    parser.add_argument('--ckpt_type', type=str, choices=['best', 'last', 'epoch'], default='best',
                       help='Checkpoint type to test: best, last or specific epoch')
    
    parser.add_argument('--checkpoint_epoch', type=int, default=None,
                       help='Epoch number to load when ckpt_type=epoch (e.g., 15)')
    
    parser.add_argument('--checkpoint_map', type=str, default=None,
                       help='Path to JSON mapping fold_dir_name (full or prefix substring) -> epoch|"best"|"last" for per-fold checkpoint selection')
    
    parser.add_argument('--test_videos', type=str, nargs='+', default=None,
                       help='Test video IDs (e.g., 05 06). If not provided, uses split info')
    
    parser.add_argument('--pose_data_dir', type=str, default=None,
                       help='Override pose data directory')
    
    parser.add_argument('--annotation_dir', type=str, default=None,
                       help='Override annotation directory')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (auto-detect if not specified)')
    
    parser.add_argument('--single_fold', type=str, default=None,
                       help='Test only a specific fold (provide fold directory name)')
    
    parser.add_argument('--map_only', action='store_true',
                        help='If set, only test folds present (exact or substring) in checkpoint_map')
    
    args = parser.parse_args()
    
    if args.exp_num and not args.exp_dir:
        args.exp_dir = os.path.join(args.outputs_root, args.exp_num)
    
    if not args.exp_dir:
        raise ValueError('You must provide --exp_dir or --exp_num')
    
    # Load checkpoint map if provided
    ckpt_map_dict = None
    if args.checkpoint_map:
        with open(args.checkpoint_map, 'r', encoding='utf-8') as f:
            ckpt_map_dict = json.load(f)
    
    # Create tester
    tester = TicModelTester(exp_dir=args.exp_dir, device=args.device, ckpt_type=args.ckpt_type,
                            checkpoint_epoch=args.checkpoint_epoch, checkpoint_map=ckpt_map_dict)
    
    if args.single_fold:
        # Test single fold
        fold_dir = Path(args.exp_dir) / args.single_fold
        if not fold_dir.exists():
            log.error(f"Fold directory not found: {fold_dir}")
            return
        
        tester.test_single_fold(
            fold_dir=fold_dir,
            test_videos=args.test_videos,
            pose_data_dir=args.pose_data_dir,
            annotation_dir=args.annotation_dir
        )
    else:
        if args.map_only and ckpt_map_dict:
            keys = list(ckpt_map_dict.keys())
            filtered = []
            for fd in tester.fold_dirs:
                if fd.name in keys or any(k in fd.name for k in keys):
                    filtered.append(fd)
            tester.fold_dirs = filtered
            log.info(f"map_only active. Testing folds: {[fd.name for fd in tester.fold_dirs]}")
        tester.test_all_folds(
            pose_data_dir=args.pose_data_dir,
            annotation_dir=args.annotation_dir
        )
    
    log.info("\n" + "="*70)
    log.info("Testing completed!")
    log.info("="*70)


if __name__ == "__main__":
    main()
