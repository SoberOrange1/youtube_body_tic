#!/usr/bin/env python3
"""
Main script for Tic Motion ViT Processor with MotionPatches Training
Uses MotionPatches-inspired architecture with LoRA fine-tuning for memory efficiency
Supports video-based cross-validation with experiment management

All configuration parameters are defined here - no separate config file needed
"""

import os
import sys
import argparse
import random
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import trainer
from motion_patches_trainer import MotionPatchesLoRATrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Data processing configuration
DATA_CONFIG = {
    'max_sequence_length': 64,  # Maximum frames per sequence
    'sequence_stride': 16,  # Stride for sliding window (25% overlap, every step one patch length)
    'normalization_method': 'z_score',  # 'z_score', 'min_max', 'none'
    # Video split configuration for cross-validation
    'train_videos': ['01', '04', '05', '07', '08', '14', '17'],  # Training video IDs (use strings for leading zeros)
    'val_videos': ['10', '06'],  # Validation video IDs
    'test_videos': ['11', '16'],  # Test video IDs (optional)
    # All available videos for reference (use strings for leading zeros)
    'all_videos': ['01', '04', '05', '06', '07', '08', '10', '11', '14', '16', '17'],
}

# Model configuration
MODEL_CONFIG = {
    'model_name': 'vit_base_patch16_224.augreg_in21k',  # Updated model name
    'pretrained': True,
    'trainable': True,
    'num_classes': 2,  # tic vs non-tic
    # Encoder type: 'basic' (MotionPatchesEncoder) or 'enhanced' (EnhancedMotionPatchesEncoder)
    'encoder_type': 'basic',  # 'basic': single channel, 'enhanced': 7 channels
    # Prediction mode: 'segment' or 'frame'
    'prediction_mode': 'frame',  # NEW: 'segment' for 64-frame classification, 'frame' for per-frame prediction
    'patch_size': 16,  # For kinematic chain interpolation
    'num_patches': 5,  # Number of kinematic chains
    'feature_dim': 7,  # [x, y, z, visibility, ax, ay, az]
    'dropout': 0.3,  # Dropout rate (0.0 for most cases, 0.5 for HumanML3D)
}

# LoRA configuration for memory-efficient fine-tuning
LORA_CONFIG = {
    'lora_r': 8,  # LoRA rank (lower = less parameters)
    'lora_alpha': 16,  # LoRA alpha parameter
    'lora_dropout': 0.3,  # LoRA dropout rate
}

# Training configuration - adapted from MotionPatches
TRAIN_CONFIG = {
    'batch_size': 16,  
    'num_epochs': 100,  # Number of epochs (MotionPatches default)
    'weight_decay': 1e-4,
    'patience': 25,
    # Layered learning rates following MotionPatches design
    'motion_lr': 1e-5,  # Learning rate for motion encoder (LoRA)
    'head_lr': 1e-4,  # Learning rate for classification head
    # Scheduler
    'scheduler_type': 'cosine',  # 'cosine' or 'step'
    # Data loading
    'num_workers': 1, 
    'pin_memory': False,  
    'drop_last': True,  # MotionPatches uses drop_last=True
    # Add gradient clipping
    'max_grad_norm': 1.0,  # New: gradient clipping
    # Add learning rate warmup
    'warmup_epochs': 5,    # New: warmup epochs
    # Experiment tracking - now controlled via command line
    'exp_num': None,  # Will be set by --exp_num argument, no auto-generation
    # Save checkpoints every N epochs (set to 0 or None to disable)
    'save_interval': 3,
}

# Random seed for reproducibility
SEED = 42

# Data paths - Updated to match your actual structure
PATHS = {
    'pose_data_dir': r'A:\youtube_body\data_folder\body_detection_results',  # landmarks_root with XX subfolders
    'annotation_dir': r'A:\youtube_body\data_folder\annotations',  # annotations_XX.json files
    'output_dir': r'A:\youtube_body\data_folder\outputs',
    'pretrained_model': r'A:\youtube_body\motion_vit_processor\pre-train\best_model.pt',  # Pretrained model path
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility - borrowed from MotionPatches"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def get_next_experiment_number(output_dir):
    """
    Get the next available experiment number
    
    Args:
        output_dir: Base output directory
    
    Returns:
        Next experiment name (e.g., 'exp_002')
    """
    if not os.path.exists(output_dir):
        return 'exp_001'
    
    # Find all existing experiment directories
    existing_exps = []
    for item in os.listdir(output_dir):
        if item.startswith('exp_') and os.path.isdir(os.path.join(output_dir, item)):
            try:
                exp_num = int(item.split('_')[1])
                existing_exps.append(exp_num)
            except (IndexError, ValueError):
                continue
    
    # Get next number
    if existing_exps:
        next_num = max(existing_exps) + 1
    else:
        next_num = 1
    
    return f'exp_{next_num:03d}'


def create_fold_save_dir(exp_dir, train_videos, val_videos, test_videos=None, fold_name=None):
    """
    Create a unique directory for this fold within the experiment directory
    
    Args:
        exp_dir: Experiment base directory
        train_videos: List of training video IDs
        val_videos: List of validation video IDs
        test_videos: List of test video IDs (optional)
        fold_name: Optional custom fold name (e.g., 'fold_1')
    
    Returns:
        Path to the created directory
    
    Directory structure:
        outputs/
        └── exp_001/
            ├── fold_1_train_01_02_03_val_04_20250111_143022/
            │   ├── video_split_info.json
            │   ├── training_config.json
            │   ├── best_model.pt
            │   ├── last_model.pt
            │   ├── training_curves.png
            │   ├── confusion_matrix.png
            │   ├── classification_report.txt
            │   └── training_summary.json
            └── fold_2_train_02_03_04_val_01_20250111_153045/
                └── ...
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create fold identifier
    train_str = "_".join(train_videos)
    val_str = "_".join(val_videos)
    
    if fold_name:
        fold_id = f"{fold_name}_train_{train_str}_val_{val_str}"
    else:
        fold_id = f"train_{train_str}_val_{val_str}"
    
    # Create directory name
    dir_name = f"{fold_id}_{timestamp}"
    save_dir = os.path.join(exp_dir, dir_name)
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save video split information
    split_info = {
        'fold_name': fold_name if fold_name else 'auto',
        'timestamp': timestamp,
        'train_videos': train_videos,
        'val_videos': val_videos,
        'test_videos': test_videos if test_videos else [],
        'train_video_count': len(train_videos),
        'val_video_count': len(val_videos),
        'test_video_count': len(test_videos) if test_videos else 0,
    }
    
    with open(os.path.join(save_dir, 'video_split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    log.info(f"Created fold directory: {save_dir}")
    log.info(f"Video split info saved to: {os.path.join(save_dir, 'video_split_info.json')}")
    
    return save_dir


def validate_video_split(train_videos, val_videos, test_videos, pose_data_dir):
    """Validate video split configuration"""
    train_vids = set(train_videos)
    val_vids = set(val_videos)
    test_vids = set(test_videos)
    
    # Check for overlaps between train and val/test (keep important checks)
    overlap_train_val = train_vids & val_vids
    overlap_train_test = train_vids & test_vids
    
    if overlap_train_val:
        raise ValueError(f"Train and val videos overlap: {overlap_train_val}")
    if overlap_train_test:
        raise ValueError(f"Train and test videos overlap: {overlap_train_test}")
    
    # Remove val-test overlap check - allow videos to be in both val and test
    
    # Check if all videos exist
    missing_videos = []
    
    for video_id in train_vids | val_vids | test_vids:
        video_path = os.path.join(pose_data_dir, video_id)
        if not os.path.exists(video_path):
            missing_videos.append(video_id)
    
    if missing_videos:
        log.warning(f"⚠️  Warning: Videos not found: {missing_videos}")
    
    log.info(f"✓ Video split validation passed")
    return True


def print_training_summary(exp_name, encoder_type, train_videos, val_videos, test_videos, exp_dir):
    """Print training summary"""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Experiment: {exp_name}")
    print(f"Encoder: {encoder_type.upper()}")
    print(f"Model: {MODEL_CONFIG['model_name']}")
    print(f"Epochs: {TRAIN_CONFIG['num_epochs']}")
    print(f"Batch Size: {TRAIN_CONFIG['batch_size']}")
    print(f"Sequence Length: {DATA_CONFIG['max_sequence_length']} frames")
    print(f"\nVideo Split:")
    print(f"  Train: {train_videos} ({len(train_videos)} videos)")
    print(f"  Val: {val_videos} ({len(val_videos)} videos)")
    if test_videos:
        print(f"  Test: {test_videos} ({len(test_videos)} videos)")
    print(f"\nLearning Rates:")
    print(f"  Motion Encoder (LoRA): {TRAIN_CONFIG['motion_lr']}")
    print(f"  Classification Head: {TRAIN_CONFIG['head_lr']}")
    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {LORA_CONFIG['lora_r']}")
    print(f"  Alpha: {LORA_CONFIG['lora_alpha']}")
    print(f"  Dropout: {LORA_CONFIG['lora_dropout']}")
    print(f"\nOutput Directory: {exp_dir}")
    print("=" * 70)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train tic classification using MotionPatches with LoRA')
    
    # Experiment management
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (e.g., exp_001). If not provided, auto-increments')
    parser.add_argument('--exp_num', type=str, default=None,
                       help='Experiment number for output path (e.g., exp_001). Overrides auto-generation')
    
    # Data parameters
    parser.add_argument('--data_folder', type=str,
                       default=None,
                       help='Folder containing normalized motion features (overrides default)')
    
    # Video split parameters for manual cross-validation
    parser.add_argument('--train_videos', type=str, nargs='+',
                       default=None,
                       help='Training video IDs (e.g., 01 02 03)')
    parser.add_argument('--val_videos', type=str, nargs='+',
                       default=None,
                       help='Validation video IDs (e.g., 04)')
    parser.add_argument('--test_videos', type=str, nargs='+',
                       default=None,
                       help='Test video IDs (e.g., 05)')
    
    # Fold naming
    parser.add_argument('--fold_name', type=str, default=None,
                       help='Custom fold name (e.g., fold_1, fold_2)')
    
    # Model parameters
    parser.add_argument('--encoder_type', type=str, default=None,
                       choices=['basic', 'enhanced'],
                       help='Type of encoder (basic: single channel, enhanced: 7 channels)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=None,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=None,
                       help='LoRA alpha parameter')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set experiment name - use exp_num if provided, otherwise auto-increment
    if args.exp_num:
        exp_name = args.exp_num
    elif args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = get_next_experiment_number(PATHS['output_dir'])
    
    log.info(f"Experiment name: {exp_name}")
    
    # Handle video split (use command line args or defaults)
    train_videos = args.train_videos if args.train_videos else DATA_CONFIG['train_videos']
    val_videos = args.val_videos if args.val_videos else DATA_CONFIG['val_videos']
    test_videos = args.test_videos if args.test_videos else DATA_CONFIG['test_videos']
    
    # Print video split
    print(f"\n{'='*70}")
    print("VIDEO SPLIT CONFIGURATION")
    print(f"{'='*70}")
    print(f"Train videos ({len(train_videos)}): {train_videos}")
    print(f"Val videos ({len(val_videos)}): {val_videos}")
    if test_videos:
        print(f"Test videos ({len(test_videos)}): {test_videos}")
    print(f"{'='*70}\n")
    
    # Apply parameter overrides
    encoder_type = args.encoder_type if args.encoder_type else MODEL_CONFIG['encoder_type']
    epochs = args.epochs if args.epochs else TRAIN_CONFIG['num_epochs']
    batch_size = args.batch_size if args.batch_size else TRAIN_CONFIG['batch_size']
    seed = args.seed if args.seed else SEED
    data_folder = args.data_folder if args.data_folder else PATHS['pose_data_dir']
    
    # Update LoRA config if provided
    lora_r = args.lora_r if args.lora_r else LORA_CONFIG['lora_r']
    lora_alpha = args.lora_alpha if args.lora_alpha else LORA_CONFIG['lora_alpha']
    
    # Validate video split
    validate_video_split(train_videos, val_videos, test_videos, data_folder)
    
    # Set random seed
    set_seed(seed)
    
    # Create experiment directory
    exp_dir = os.path.join(PATHS['output_dir'], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log.info(f"Experiment directory: {exp_dir}")
    
    # Create fold-specific save directory
    save_dir = create_fold_save_dir(
        exp_dir=exp_dir,
        train_videos=train_videos,
        val_videos=val_videos,
        test_videos=test_videos,
        fold_name=args.fold_name
    )
    
    # Build complete configuration for trainer
    config = {
        # Model config
        'model_name': MODEL_CONFIG['model_name'],
        'pretrained': MODEL_CONFIG['pretrained'],
        'trainable': MODEL_CONFIG['trainable'],
        'encoder_type': encoder_type,
        'patch_size': MODEL_CONFIG['patch_size'],
        'num_patches': MODEL_CONFIG['num_patches'],
        'feature_dim': MODEL_CONFIG['feature_dim'],
        'num_classes': MODEL_CONFIG['num_classes'],
        'dropout': MODEL_CONFIG['dropout'],
        'prediction_mode': MODEL_CONFIG['prediction_mode'],  # Add prediction mode
        # LoRA config
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': LORA_CONFIG['lora_dropout'],
        # Training config
        'batch_size': batch_size,
        'num_epochs': epochs,
        'weight_decay': TRAIN_CONFIG['weight_decay'],
        'patience': TRAIN_CONFIG['patience'],
        'motion_lr': TRAIN_CONFIG['motion_lr'],
        'head_lr': TRAIN_CONFIG['head_lr'],
        'scheduler_type': TRAIN_CONFIG['scheduler_type'],
        # New training stability parameters
        'max_grad_norm': TRAIN_CONFIG.get('max_grad_norm', 1.0),  # Gradient clipping
        'warmup_epochs': TRAIN_CONFIG.get('warmup_epochs', 0),    # Warmup epochs
        'num_workers': TRAIN_CONFIG['num_workers'],
        'pin_memory': TRAIN_CONFIG['pin_memory'],
        'drop_last': TRAIN_CONFIG['drop_last'],
        # Experiment tracking
        'exp_num': TRAIN_CONFIG.get('exp_num') or exp_name,  # Use exp_name if exp_num not set
        # Checkpoint interval
        'save_interval': TRAIN_CONFIG.get('save_interval', 0),
        # Data config
        'max_sequence_length': DATA_CONFIG['max_sequence_length'],
        'max_frames': DATA_CONFIG['max_sequence_length'],
        'sequence_stride': DATA_CONFIG['sequence_stride'],
        'normalization_method': DATA_CONFIG['normalization_method'],
        'train_videos': train_videos,
        'val_videos': val_videos,
        'test_videos': test_videos,
        # Pretrained model path
        'pretrained_model_path': PATHS['pretrained_model'],  # Add pretrained model path
    }
    
    # Save training configuration
    config_save = {
        'experiment_name': exp_name,
        'model_config': {k: v for k, v in config.items() if k in MODEL_CONFIG or k in ['encoder_type']},
        'lora_config': {k: v for k, v in config.items() if k.startswith('lora_')},
        'train_config': {k: v for k, v in config.items() if k in TRAIN_CONFIG or k in ['batch_size', 'num_epochs']},
        'data_config': {k: v for k, v in config.items() if k in DATA_CONFIG or k in ['train_videos', 'val_videos', 'test_videos', 'max_frames']},
        'paths': {
            'pose_data_dir': data_folder,
            'annotation_dir': PATHS['annotation_dir'],
        },
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config_save, f, indent=2)
    
    log.info(f"Training configuration saved to: {os.path.join(save_dir, 'training_config.json')}")
    
    # Print training summary
    log.info("\n" + "=" * 70)
    log.info("MotionPatches-inspired Tic Classification Training with LoRA")
    log.info("=" * 70)
    print_training_summary(exp_name, encoder_type, train_videos, val_videos, test_videos, exp_dir)
    
    # Create trainer and train
    log.info("\nCreating trainer...")
    try:
        trainer = MotionPatchesLoRATrainer(config)
        log.info("Trainer created successfully!")
    except Exception as e:
        log.error(f"Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    log.info("\nStarting training...")
    try:
        best_acc = trainer.train(
            pose_data_dir=data_folder,
            annotation_dir=PATHS['annotation_dir'],
            epochs=epochs,
            batch_size=batch_size,
            save_dir=save_dir
        )
    except Exception as e:
        log.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save final summary
    final_summary = {
        'experiment_name': exp_name,
        'fold_name': args.fold_name if args.fold_name else 'auto',
        'best_val_accuracy': best_acc,
        'total_epochs': epochs,
        'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
        'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None,
        'final_train_acc': trainer.train_accs[-1] if trainer.train_accs else None,
        'final_val_acc': trainer.val_accs[-1] if trainer.val_accs else None,
        'timestamp_completed': datetime.now().isoformat(),
    }
    
    with open(os.path.join(save_dir, 'training_summary.json'), 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    log.info("=" * 70)
    log.info(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    log.info(f"Results saved to: {save_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()