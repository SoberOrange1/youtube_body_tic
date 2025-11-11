#!/usr/bin/env python3
"""
Motion Patches Trainer for Tic Classification with LoRA
Core trainer and dataset classes for MotionPatches-inspired architecture
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import logging
from peft import LoraConfig, get_peft_model, TaskType

# Fix import path for motion encoder
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

# Import motion encoder models
from motion_encoder import MotionPatchesEncoder, EnhancedMotionPatchesEncoder

log = logging.getLogger(__name__)

class TicMotionPatchesDataset(Dataset):
    """Dataset for tic classification using MotionPatches-inspired preprocessing"""
    
    def __init__(self, pose_data_dir: str, annotation_dir: str, video_ids: list, max_frames: int = 64, feature_mode: str = "basic", normalize_coords: bool = True):
        """
        Args:
            pose_data_dir: Root directory containing video_id subfolders with pose_results_XX.json
            annotation_dir: Directory containing annotations_XX.json files
            video_ids: List of video IDs to load (e.g., ['01', '02', '03'])
            max_frames: Maximum frames per sequence
            feature_mode: 'basic' or 'enhanced'
            normalize_coords: Whether to apply hip-centered coordinate normalization
        
        Expected file structure:
            pose_data_dir/
            ├── 01/
            │   └── pose_results_01.json
            ├── 02/
            │   └── pose_results_02.json
            └── ...
            
            annotation_dir/
            ├── annotations_01.json
            ├── annotations_02.json
            └── ...
        """
        self.pose_data_dir = pose_data_dir
        self.annotation_dir = annotation_dir
        self.video_ids = video_ids
        self.max_frames = max_frames
        self.feature_mode = feature_mode
        self.normalize_coords = normalize_coords  # Store normalization setting
        
        # Load all video data
        self.video_data = []
        self.labels = []
        self.video_sources = []  # Track which video each sample comes from
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load pose data and annotations from separate directories"""
        log.info(f"Loading pose and annotation data for videos: {self.video_ids}")
        
        for video_id in tqdm(self.video_ids, desc="Loading videos"):
            # Load pose data from subfolder: pose_data_dir/video_id/pose_results_video_id.json
            pose_file = os.path.join(self.pose_data_dir, video_id, f'pose_results_{video_id}.json')
            
            if not os.path.exists(pose_file):
                log.warning(f"Pose file not found: {pose_file}")
                continue
            
            # Load annotation data
            annotation_file = os.path.join(self.annotation_dir, f'annotations_{video_id}.json')
            
            if not os.path.exists(annotation_file):
                log.warning(f"Annotation file not found: {annotation_file}")
                continue
            
            try:
                # Load pose results
                with open(pose_file, 'r') as f:
                    pose_data = json.load(f)
                
                # Load annotations
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                
                # Extract and process features
                feature_vectors, frame_labels = self._process_video_data(
                    pose_data, annotation_data, video_id
                )
                
                # Process in segments
                if feature_vectors is not None and frame_labels is not None:
                    self._process_video_segments(feature_vectors, frame_labels, video_id)
                
            except Exception as e:
                log.error(f"Error loading video {video_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        log.info(f"Loaded {len(self.video_data)} motion segments from {len(self.video_ids)} videos")
        log.info(f"Tic segments: {sum(self.labels)}")
        log.info(f"Non-tic segments: {len(self.labels) - sum(self.labels)}")
    
    def _process_video_data(self, pose_data, annotation_data, video_id):
        """
        Process raw pose data and annotations into feature vectors
        
        Args:
            pose_data: Raw pose detection results
            annotation_data: Frame-level annotations
            video_id: Video identifier
        
        Returns:
            feature_vectors: (n_frames, 33, 7) numpy array
            frame_labels: List of frame-level labels ('tic' or 'non_tic')
        """
        # Extract pose landmarks per frame
        # Your format: {"video_id": "01", "frames": {"0": {...}, "1": {...}}}
        if 'frames' not in pose_data:
            log.warning(f"Invalid pose data format for video {video_id}")
            return None, None
        
        frames_dict = pose_data['frames']
        
        # Get annotation ranges for this video
        # Your format: {"01": {"frame_annotations": [...]}
        if video_id not in annotation_data:
            log.warning(f"Video {video_id} not found in annotations")
            return None, None
        
        frame_annotations = annotation_data[video_id]['frame_annotations']
        
        # Build a frame-to-label mapping
        frame_label_map = {}
        for ann in frame_annotations:
            start = ann['start_frame']
            end = ann['end_frame']
            label = ann['label']
            for frame_id in range(start, end + 1):
                frame_label_map[frame_id] = label
        
        # Build feature vectors for each frame
        feature_list = []
        label_list = []
        
        # Sort frame IDs to ensure sequential order
        frame_ids = sorted([int(fid) for fid in frames_dict.keys()])
        
        for frame_id in frame_ids:
            frame_str = str(frame_id)
            
            if frame_str not in frames_dict:
                continue
            
            frame_info = frames_dict[frame_str]
            landmarks = frame_info.get('landmarks')
            
            if landmarks is None or len(landmarks) == 0:
                continue
            
            # Extract 33 landmarks with 7 features: [x, y, z, visibility, ax, ay, az]
            frame_features = self._extract_frame_features(landmarks)
            
            if frame_features is not None:
                feature_list.append(frame_features)
                
                # Get label for this frame from annotation ranges
                label = frame_label_map.get(frame_id, 'non-tic')
                label_list.append(label)
        
        if len(feature_list) == 0:
            return None, None
        
        # Convert to numpy array: (n_frames, 33, 7)
        feature_vectors = np.array(feature_list)
        
        log.info(f"Video {video_id}: Processed {len(feature_list)} frames")
        
        return feature_vectors, label_list
    
    def _normalize_coordinates_by_hip(self, landmarks, enable_normalization=True):
        """
        Normalize coordinates using hip center as origin with scale normalization
        
        Args:
            landmarks: List of landmark dicts with format:
                      {"id": 0, "name": "NOSE", "x": 0.48, "y": 0.32, "z": -0.19, "visibility": 0.99}
            enable_normalization: Whether to apply hip-centered normalization
        
        Returns:
            normalized_landmarks: List of landmarks with normalized coordinates
        """
        if not enable_normalization:
            return landmarks
        
        try:
            # First extract all coordinates into array format
            coords = np.zeros((33, 3))
            landmark_map = {}
            
            for lm in landmarks:
                lm_id = lm.get('id')
                if lm_id is None or lm_id >= 33:
                    continue
                    
                coords[lm_id, 0] = lm.get('x', 0.0)
                coords[lm_id, 1] = lm.get('y', 0.0) 
                coords[lm_id, 2] = lm.get('z', 0.0)
                landmark_map[lm_id] = lm
            
            # Calculate hip center point (average of LEFT_HIP=23 and RIGHT_HIP=24)
            left_hip = coords[23]  # LEFT_HIP
            right_hip = coords[24]  # RIGHT_HIP
            hip_center = (left_hip + right_hip) / 2
            
            # Calculate body scale using hip width for normalization
            hip_width = np.linalg.norm(left_hip - right_hip)
            if hip_width < 1e-6:  # Avoid division by zero
                hip_width = 1.0
            
            # Normalize coordinates: subtract hip center and scale by hip width
            normalized_coords = (coords - hip_center) / hip_width
            
            # Create normalized landmark list
            normalized_landmarks = []
            for lm in landmarks:
                lm_id = lm.get('id')
                if lm_id is None or lm_id >= 33:
                    continue
                    
                # Create new landmark dict with normalized coordinates
                normalized_lm = lm.copy()
                normalized_lm['x'] = normalized_coords[lm_id, 0]
                normalized_lm['y'] = normalized_coords[lm_id, 1]
                normalized_lm['z'] = normalized_coords[lm_id, 2]
                normalized_landmarks.append(normalized_lm)
            
            return normalized_landmarks
            
        except Exception as e:
            log.warning(f"Error in coordinate normalization: {e}, using original coordinates")
            return landmarks
    
    def _extract_frame_features(self, landmarks):
        """
        Extract 33 landmarks × 7 features from raw landmark data with optional coordinate normalization
        
        Args:
            landmarks: List of landmark dicts with format:
                      {"id": 0, "name": "NOSE", "x": 0.48, "y": 0.32, "z": -0.19, "visibility": 0.99}
        
        Returns:
            features: (33, 7) numpy array with [x, y, z, visibility, ax, ay, az]
        """
        try:
            # Apply coordinate normalization based on config setting
            if self.normalize_coords:
                landmarks = self._normalize_coordinates_by_hip(landmarks, enable_normalization=True)
            
            features = np.zeros((33, 7), dtype=np.float32)
            
            for lm in landmarks:
                lm_id = lm.get('id')
                
                if lm_id is None or lm_id >= 33:
                    continue
                
                # Extract normalized or original coordinates
                features[lm_id, 0] = lm.get('x', 0.0)
                features[lm_id, 1] = lm.get('y', 0.0)
                features[lm_id, 2] = lm.get('z', 0.0)
                features[lm_id, 3] = lm.get('visibility', 1.0)
                
                # Acceleration (if available, otherwise 0)
                features[lm_id, 4] = lm.get('ax', 0.0)
                features[lm_id, 5] = lm.get('ay', 0.0)
                features[lm_id, 6] = lm.get('az', 0.0)
            
            return features
            
        except Exception as e:
            log.error(f"Error extracting frame features: {e}")
            return None
    
    def _process_video_segments(self, feature_vectors, frame_labels, video_id):
        """Process video into fixed-length segments for MotionPatches"""
        n_frames = len(feature_vectors)
        
        # Create segments with overlap - borrowed from MotionPatches design
        segment_step = self.max_frames // 2  # 50% overlap
        
        for start_idx in range(0, n_frames - self.max_frames + 1, segment_step):
            end_idx = start_idx + self.max_frames
            
            segment_features = feature_vectors[start_idx:end_idx]
            segment_labels = frame_labels[start_idx:end_idx]
            
            # Determine segment label (majority vote)
            tic_count = sum(1 for label in segment_labels if label == 'tic')
            segment_label = 1 if tic_count > len(segment_labels) / 2 else 0
            
            # Store the raw features for MotionPatchesEncoder processing
            self.video_data.append(segment_features)
            self.labels.append(segment_label)
            self.video_sources.append(video_id)
    
    def _get_frame_label(self, frame_id, annotation_data):
        """
        Get label for a specific frame from annotation data
        
        Args:
            frame_id: Frame number
            annotation_data: Annotation dictionary
        
        Returns:
            label: 'tic' or 'non_tic'
        
        Note: This method is now handled in _process_video_data for efficiency,
              but kept for backward compatibility
        """
        # This function is deprecated - labels are now processed in _process_video_data
        return 'non_tic'
    
    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        motion_features = self.video_data[idx]
        label = self.labels[idx]
        
        # Convert to tensor and ensure correct dimensions
        motion_tensor = torch.FloatTensor(motion_features)
        
        return motion_tensor, torch.LongTensor([label]).squeeze()


class MotionPatchesTicClassifierWithLoRA(nn.Module):
    """
    Tic classification model using MotionPatches-inspired architecture with LoRA
    Memory-efficient fine-tuning approach
    """
    
    def __init__(
        self,
        encoder_type: str = "basic",
        model_name: str = "vit_base_patch16_224_in21k",
        pretrained: bool = True,
        patch_size: int = 16,
        max_frames: int = 64,
        num_classes: int = 2,
        dropout: float = 0.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        log.info(f"Creating MotionPatchesTicClassifierWithLoRA...")
        log.info(f"  encoder_type: {encoder_type}")
        log.info(f"  model_name: {model_name}")
        log.info(f"  pretrained: {pretrained}")
        log.info(f"  LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        
        self.encoder_type = encoder_type
        
        # Create base encoder
        log.info(f"Creating base encoder ({encoder_type})...")
        if encoder_type == "basic":
            self.base_encoder = MotionPatchesEncoder(
                model_name=model_name,
                pretrained=pretrained,
                trainable=False,  # We'll use LoRA instead
                patch_size=patch_size,
                num_patches=5,
                max_frames=max_frames,
                input_channels=1,
                num_classes=num_classes
            )
        elif encoder_type == "enhanced":
            self.base_encoder = EnhancedMotionPatchesEncoder(
                model_name=model_name,
                pretrained=pretrained,
                trainable=False,  # We'll use LoRA instead
                patch_size=patch_size,
                num_patches=5,
                max_frames=max_frames,
                feature_dim=kwargs.get('feature_dim', 7),
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        log.info("Base encoder created successfully")
        
        # Apply LoRA to the motion encoder (ViT backbone)
        log.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv", "proj"],  # Target attention layers in ViT
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        log.info("Applying LoRA to motion encoder...")
        # Apply LoRA to the motion encoder
        self.base_encoder.motion_encoder = get_peft_model(
            self.base_encoder.motion_encoder, 
            lora_config
        )
        
        log.info(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}")
        log.info(f"Trainable parameters: {self.get_trainable_params()}")
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return f"{trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
    
    def forward(self, x):
        """Forward pass with LoRA-enabled encoder"""
        features, logits = self.base_encoder(x)
        return logits
    
    def extract_features(self, x):
        """Extract features without classification"""
        return self.base_encoder.extract_features(x)


class MotionPatchesLoRATrainer:
    """
    Trainer for tic classification using MotionPatches approach with LoRA
    Adapted from original MotionPatches training parameters
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        log.info(f"Initializing MotionPatchesLoRATrainer...")
        log.info(f"Device: {self.device}")
        log.info(f"Creating model with encoder_type: {config.get('encoder_type', 'basic')}")
        
        # Create model with LoRA
        self.model = MotionPatchesTicClassifierWithLoRA(
            encoder_type=config.get('encoder_type', 'basic'),
            model_name=config.get('model_name', 'vit_base_patch16_224_in21k'),
            pretrained=config.get('pretrained', True),
            patch_size=config.get('patch_size', 16),
            max_frames=config.get('max_frames', 64),
            num_classes=2,
            dropout=config.get('dropout', 0.0),
            lora_r=config.get('lora_r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.1),
            feature_dim=config.get('feature_dim', 7)
        ).to(self.device)
        
        log.info("Model created successfully")
        log.info(f"Model moved to device: {self.device}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        log.info("Loss function initialized")
        
        # Optimizer - following MotionPatches design with layered learning rates
        log.info("Setting up optimizer...")
        self._setup_optimizer()
        log.info("Optimizer setup complete")
        
        # Results tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        log.info("Trainer initialization complete")
    
    def _setup_optimizer(self):
        """Setup optimizer with layered learning rates following MotionPatches"""
        # Separate LoRA parameters and classifier parameters
        lora_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(param)
                elif 'classifier' in name.lower():
                    classifier_params.append(param)
                else:
                    classifier_params.append(param)  # fallback
        
        # Create parameter groups with different learning rates
        parameters = [
            {
                "params": lora_params,
                "lr": self.config.get('motion_lr', 1e-5),
            },
            {
                "params": classifier_params,
                "lr": self.config.get('head_lr', 1e-4),
            },
        ]
        
        # Use Adam optimizer like MotionPatches
        self.optimizer = Adam(parameters)
        
        log.info(f"LoRA parameters: {len(lora_params)}")
        log.info(f"Classifier parameters: {len(classifier_params)}")
        log.info(f"LoRA LR: {self.config.get('motion_lr', 1e-5)}")
        log.info(f"Head LR: {self.config.get('head_lr', 1e-4)}")
    
    def create_data_loaders(self, pose_data_dir, annotation_dir, batch_size=32):
        """Create train and validation data loaders based on video splits"""
        
        # Get video splits from config
        train_videos = self.config.get('train_videos', [])
        val_videos = self.config.get('val_videos', [])
        
        if not train_videos or not val_videos:
            raise ValueError("train_videos and val_videos must be specified in config")
        
        log.info(f"Creating data loaders with video-based split:")
        log.info(f"  Train videos: {train_videos}")
        log.info(f"  Val videos: {val_videos}")
        
        # Load train dataset
        train_dataset = TicMotionPatchesDataset(
            pose_data_dir=pose_data_dir,
            annotation_dir=annotation_dir,
            video_ids=train_videos,
            max_frames=self.config.get('max_frames', 64),
            feature_mode=self.config.get('encoder_type', 'basic')
        )
        
        # Load validation dataset
        val_dataset = TicMotionPatchesDataset(
            pose_data_dir=pose_data_dir,
            annotation_dir=annotation_dir,
            video_ids=val_videos,
            max_frames=self.config.get('max_frames', 64),
            feature_mode=self.config.get('encoder_type', 'basic')
        )
        
        # Create data loaders - following MotionPatches settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=self.config.get('drop_last', True),
            num_workers=self.config.get('num_workers', 8),
            pin_memory=self.config.get('pin_memory', True)
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.config.get('num_workers', 8),
            pin_memory=self.config.get('pin_memory', True)
        )
        
        log.info(f"Train dataset: {len(train_dataset)} segments")
        log.info(f"Val dataset: {len(val_dataset)} segments")
        
        return train_loader, val_loader
    
    def train(self, pose_data_dir, annotation_dir, epochs=None, batch_size=None, save_dir=None):
        """
        Complete training loop - adapted from MotionPatches training
        
        Args:
            pose_data_dir: Directory containing pose_results_XX.json files
            annotation_dir: Directory containing annotations_XX.json files
            epochs: Number of training epochs
            batch_size: Batch size
            save_dir: Directory to save results
        """
        # Use config defaults if not provided
        if epochs is None:
            epochs = self.config.get('num_epochs', 100)
        if batch_size is None:
            batch_size = self.config.get('batch_size', 32)
        if save_dir is None:
            save_dir = 'motion_patches_lora_checkpoints'
        
        # Get patience for early stopping
        patience = self.config.get('patience', 10)
        
        log.info(f"Training tic classification model using MotionPatches with LoRA")
        log.info(f"Encoder type: {self.config.get('encoder_type', 'basic')}")
        log.info(f"Device: {self.device}")
        log.info(f"Model trainable parameters: {self.model.get_trainable_params()}")
        log.info(f"Early stopping patience: {patience} epochs")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(pose_data_dir, annotation_dir, batch_size)
        
        # Create scheduler - following MotionPatches CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=len(train_loader) * epochs * 2
        )
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0
        best_val_loss = 1e5
        best_epoch = -1
        epochs_without_improvement = 0  # Early stopping counter
        
        for epoch in range(epochs):
            # Print progress - following MotionPatches format
            print(f"running epoch {epoch}, best test loss {best_val_loss:.4f} best val acc {best_val_acc:.2f}% after epoch {best_epoch}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Log results - following MotionPatches format
            log.info(f"epoch {epoch}, tr_loss {train_loss:.4f}, te_loss {val_loss:.4f}, val_acc {val_acc:.2f}%")
            
            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0  # Reset counter
                
                # Save model - following MotionPatches format
                torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                log.info(f"New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                epochs_without_improvement += 1
                log.info(f"No improvement for {epochs_without_improvement}/{patience} epochs")
            
            # Save last model - following MotionPatches
            torch.save(self.model.state_dict(), os.path.join(save_dir, "last_model.pt"))
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                log.info(f"\n{'='*70}")
                log.info(f"Early stopping triggered after {epoch + 1} epochs")
                log.info(f"No improvement in validation accuracy for {patience} consecutive epochs")
                log.info(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
                log.info(f"{'='*70}\n")
                break
        
        # Check if training completed without early stopping
        if epochs_without_improvement < patience:
            log.info(f"\nTraining completed all {epochs} epochs")
            log.info(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
        
        # Save final results
        self._save_training_results(save_dir, val_preds, val_labels)
        
        return best_val_acc

    def _save_training_results(self, save_dir, val_preds, val_labels):
        """Save training results and metrics"""
        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
        plt.close()
        
        # Classification report
        class_names = ['Non-tic', 'Tic']
        report = classification_report(val_labels, val_preds, target_names=class_names)
        
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write("Final Validation Results\n")
            f.write("========================\n\n")
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        log.info(f"Training results saved to: {save_dir}")