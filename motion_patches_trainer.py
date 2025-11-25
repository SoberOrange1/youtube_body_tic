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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from tqdm import tqdm
import logging
from peft import LoraConfig, get_peft_model, TaskType

# Fix import path for motion encoder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level to project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import motion encoder models - now from project root
from motion_encoder import MotionPatchesEncoder, EnhancedMotionPatchesEncoder

log = logging.getLogger(__name__)

class TicMotionPatchesDataset(Dataset):
    """Dataset for tic classification using MotionPatches-inspired preprocessing"""
    
    def __init__(self, pose_data_dir: str, annotation_dir: str, video_ids: list, max_frames: int = 64, feature_mode: str = "basic", normalize_coords: bool = True, sequence_stride: int = None):
        """
        Args:
            pose_data_dir: Root directory containing video_id subfolders with pose_results_XX.json
            annotation_dir: Directory containing annotations_XX.json files
            video_ids: List of video IDs to load (e.g., ['01', '02', '03'])
            max_frames: Maximum frames per sequence
            feature_mode: 'basic' or 'enhanced'
            normalize_coords: Whether to apply hip-centered coordinate normalization
            sequence_stride: Stride for sliding windows (default: max_frames // 2)
        
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
        
        # Store sequence stride for sliding windows
        self.sequence_stride = sequence_stride if sequence_stride is not None else max_frames // 2
        
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
        
        # Calculate statistics for frame-level labels
        total_tic_frames = 0
        total_non_tic_frames = 0
        
        for label_sequence in self.labels:
            if isinstance(label_sequence, list):
                # Frame-level labels: count tic and non-tic frames
                total_tic_frames += sum(label_sequence)
                total_non_tic_frames += len(label_sequence) - sum(label_sequence)
            else:
                # Segment-level labels (backward compatibility)
                if label_sequence == 1:
                    total_tic_frames += 1
                else:
                    total_non_tic_frames += 1
        
        log.info(f"Frame-level statistics:")
        log.info(f"  Total tic frames: {total_tic_frames}")
        log.info(f"  Total non-tic frames: {total_non_tic_frames}")
        log.info(f"  Tic segments: {len([seq for seq in self.labels if isinstance(seq, list) and sum(seq) > len(seq)//2])}")
        log.info(f"  Non-tic segments: {len([seq for seq in self.labels if isinstance(seq, list) and sum(seq) <= len(seq)//2])}")
    
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
        """Process video into fixed-length segments for MotionPatches with frame-level labels"""
        n_frames = len(feature_vectors)
        
        # Create segments with configurable stride for overlapping windows
        segment_step = self.sequence_stride  # Use instance variable
        
        for start_idx in range(0, n_frames - self.max_frames + 1, segment_step):
            end_idx = start_idx + self.max_frames
            
            segment_features = feature_vectors[start_idx:end_idx]  # (64, 33, 7)
            segment_labels = frame_labels[start_idx:end_idx]       # 64 labels
            
            # Convert frame labels to binary for each frame
            frame_level_labels = []
            for label in segment_labels:
                frame_level_labels.append(1 if label == 'tic' else 0)
            
            # Store segment features and frame-level labels
            self.video_data.append(segment_features)  # (64, 33, 7)
            self.labels.append(frame_level_labels)    # List of 64 labels [0,0,1,1,0,...]
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
        labels = self.labels[idx]  # Now this is a list of 64 frame labels
        
        # Convert to tensor
        motion_tensor = torch.FloatTensor(motion_features)  # (64, 33, 7)
        
        # For frame-level prediction: return 64 labels
        if isinstance(labels, list) and len(labels) == self.max_frames:
            label_tensor = torch.LongTensor(labels)  # (64,) - one label per frame
        else:
            # Fallback for segment-level (backward compatibility)
            label_tensor = torch.LongTensor([labels]).squeeze()
        
        return motion_tensor, label_tensor


class MotionPatchesTicClassifierWithLoRA(nn.Module):
    """
    Tic classification model using MotionPatches-inspired architecture with LoRA
    Now supports both segment-level and frame-level predictions
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
        prediction_mode: str = "segment",
        **kwargs
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.max_frames = max_frames
        self.num_classes = num_classes
        self.prediction_mode = prediction_mode
        
        log.info(f"Initializing MotionPatchesTicClassifierWithLoRA with prediction_mode: {prediction_mode}")
        
        # Create base encoder
        if encoder_type == "basic":
            self.base_encoder = MotionPatchesEncoder(
                model_name=model_name,
                pretrained=pretrained,
                trainable=kwargs.get('trainable', True),
                patch_size=patch_size,
                num_patches=kwargs.get('num_patches', 5),
                max_frames=max_frames,
                input_channels=1,
                num_classes=num_classes
            )
        elif encoder_type == "enhanced":
            self.base_encoder = EnhancedMotionPatchesEncoder(
                model_name=model_name,
                pretrained=pretrained,
                trainable=kwargs.get('trainable', True),
                patch_size=patch_size,
                num_patches=kwargs.get('num_patches', 5),
                max_frames=max_frames,
                feature_dim=kwargs.get('feature_dim', 7),
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Get feature dimension from the encoder
        # Note: This is the output dimension from the ViT encoder, not input dimension
        with torch.no_grad():
            # Create dummy input to determine feature dimension
            if encoder_type == "basic":
                dummy_input = torch.randn(1, max_frames, 33, 1)  # (batch, frames, joints, features)
            else:
                dummy_input = torch.randn(1, max_frames, 33, 7)  # (batch, frames, joints, features)
            
            dummy_features = self.base_encoder.extract_features(dummy_input)
            feature_dim = dummy_features.shape[-1]
            log.info(f"Detected feature dimension from encoder: {feature_dim}")
        
        self.feature_dim = feature_dim
        
        # Create frame classifier for frame-level prediction
        if prediction_mode == "frame":
            # Frame classifier: maps feature_dim -> num_classes for each frame
            self.frame_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)  # Output: (batch_size, num_classes) per frame
            )
            log.info(f"Created frame classifier: {feature_dim} -> {num_classes}")
        
        # Apply LoRA if rank > 0
        if lora_r > 0:
            log.info(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv", "proj"],  # Common ViT attention modules
                lora_dropout=lora_dropout,
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.base_encoder.motion_encoder = get_peft_model(
                self.base_encoder.motion_encoder, lora_config
            )
            log.info("LoRA applied successfully to base encoder")
        
        log.info(f"Model initialization complete. Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        """
        Forward pass with support for both segment and frame-level predictions
        
        Args:
            x: Input tensor (batch_size, frames, landmarks, features)
        
        Returns:
            features: Extracted features (batch_size, feature_dim)
            logits: Classification logits based on prediction_mode
        """
        batch_size = x.shape[0]
        
        if self.prediction_mode == "segment":
            # Extract features and get segment-level prediction from base encoder
            features, segment_logits = self.base_encoder(x)  # features: (batch_size, feature_dim)
            return features, segment_logits  # (batch_size, num_classes)
        
        elif self.prediction_mode == "frame":
            # Extract features without classification from base encoder
            features = self.base_encoder.extract_features(x)  # (batch_size, feature_dim)
            
            # Create frame-level predictions by repeating features for each frame
            # and applying frame classifier
            frame_logits_list = []
            for frame_idx in range(self.max_frames):
                # Apply frame classifier to the global features
                frame_logit = self.frame_classifier(features)  # (batch_size, num_classes)
                frame_logits_list.append(frame_logit.unsqueeze(1))  # (batch_size, 1, num_classes)
            
            # Concatenate all frame predictions
            frame_logits = torch.cat(frame_logits_list, dim=1)  # (batch_size, max_frames, num_classes)
            
            return features, frame_logits
        
        else:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}")

    def extract_features(self, x):
        """Extract features without classification"""
        return self.base_encoder.extract_features(x)

    def get_trainable_params(self):
        """Get number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return f"{trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"

    def get_total_params(self):
        """Get count of total parameters"""
        return sum(p.numel() for p in self.parameters())

    def print_model_info(self):
        """Print detailed model information"""
        trainable = self.get_trainable_params()
        total = self.get_total_params()
        
        log.info(f"Model Architecture: MotionPatchesTicClassifierWithLoRA")
        log.info(f"Encoder Type: {self.encoder_type}")
        log.info(f"Prediction Mode: {self.prediction_mode}")
        log.info(f"Max Frames: {self.max_frames}")
        log.info(f"Number of Classes: {self.num_classes}")
        log.info(f"Feature Dimension: {self.feature_dim}")
        log.info(f"Total Parameters: {total:,}")
        log.info(f"Trainable Parameters: {trainable:,}")
        log.info(f"Trainable Ratio: {trainable/total*100:.2f}%")
        
        # Print encoder info
        if hasattr(self.base_encoder, 'motion_encoder'):
            encoder_params = sum(p.numel() for p in self.base_encoder.motion_encoder.parameters() if p.requires_grad)
            log.info(f"Motion Encoder Trainable Parameters: {encoder_params:,}")
        
        # Print classifier info
        if hasattr(self, 'frame_classifier'):
            classifier_params = sum(p.numel() for p in self.frame_classifier.parameters() if p.requires_grad)
            log.info(f"Frame Classifier Trainable Parameters: {classifier_params:,}")


class MotionPatchesLoRATrainer:
    """
    Trainer for tic classification using MotionPatches approach with LoRA
    Adapted from original MotionPatches training parameters
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine prediction mode
        self.prediction_mode = config.get('prediction_mode', 'segment')
        
        log.info(f"Initializing MotionPatchesLoRATrainer...")
        log.info(f"Device: {self.device}")
        log.info(f"Prediction mode: {self.prediction_mode}")
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
            prediction_mode=self.prediction_mode,
            feature_dim=config.get('feature_dim', 7)
        ).to(self.device)
        
        log.info("Model created successfully")
        log.info(f"Model moved to device: {self.device}")
        
        # Load pretrained model if path is provided
        pretrained_model_path = config.get('pretrained_model_path')
        if (pretrained_model_path and os.path.exists(pretrained_model_path)):
            log.info(f"Loading pretrained model from: {pretrained_model_path}")
            try:
                checkpoint = torch.load(pretrained_model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Load state dict with strict=False to handle LoRA parameters
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    log.warning(f"Missing keys in pretrained model: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    log.warning(f"Unexpected keys in pretrained model: {unexpected_keys[:5]}...")  # Show first 5
                
                log.info("Pretrained model loaded successfully")
                
                # If checkpoint contains training info
                if isinstance(checkpoint, dict):
                    if 'epoch' in checkpoint:
                        log.info(f"Pretrained model was trained for {checkpoint['epoch']} epochs")
                    if 'best_val_acc' in checkpoint:
                        log.info(f"Pretrained model best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
                
            except Exception as e:
                log.error(f"Error loading pretrained model: {e}")
                log.warning("Continuing with randomly initialized model...")
        elif pretrained_model_path:
            log.warning(f"Pretrained model path specified but file not found: {pretrained_model_path}")
            log.warning("Continuing with randomly initialized model...")
        else:
            log.info("No pretrained model path specified, using randomly initialized model")
        
        # Loss function - adapt for frame-level or segment-level
        if self.prediction_mode == "frame":
            # Use FocalLoss for frame-level classification (alpha will be set after loaders are built)
            self.criterion = FocalLoss(gamma=2.0, alpha=None)
            log.info("Using frame-level FocalLoss (gamma=2.0)")
        else:
            # Use FocalLoss for segment-level classification
            self.criterion = FocalLoss(gamma=2.0, alpha=None)
            log.info("Using segment-level FocalLoss (gamma=2.0)")
        
        # Optimizer - following MotionPatches design with layered learning rates
        log.info("Setting up optimizer...")
        self._setup_optimizer()
        log.info("Optimizer setup complete")
        
        # Results tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        # Track validation precision, recall, F1 per epoch
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        
        # Checkpoint saving interval (epochs); 0 disables periodic saving
        self.save_interval = int(self.config.get('save_interval', 0))
        
        log.info("Trainer initialization complete")
    
    def _setup_optimizer(self):
        """Setup optimizer with layered learning rates + proper weight decay."""
        wd = float(self.config.get('weight_decay', 0.0))

        # Split params: LoRA vs others, and decay vs no_decay (bias/LayerNorm no decay)
        lora_decay, lora_no_decay = [], []
        cls_decay, cls_no_decay = [], []

        no_decay_keys = ('bias', 'norm', 'layernorm', 'ln', 'bn')  # common norm/bias keys

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_lora = ('lora' in name.lower())
            is_no_decay = any(k in name.lower() for k in no_decay_keys)
            if is_lora:
                (lora_no_decay if is_no_decay else lora_decay).append(p)
            else:
                (cls_no_decay if is_no_decay else cls_decay).append(p)

        parameters = []
        if lora_decay:
            parameters.append({"params": lora_decay, "lr": self.config.get('motion_lr', 1e-5), "weight_decay": wd})
        if lora_no_decay:
            parameters.append({"params": lora_no_decay, "lr": self.config.get('motion_lr', 1e-5), "weight_decay": 0.0})
        if cls_decay:
            parameters.append({"params": cls_decay, "lr": self.config.get('head_lr', 1e-4), "weight_decay": wd})
        if cls_no_decay:
            parameters.append({"params": cls_no_decay, "lr": self.config.get('head_lr', 1e-4), "weight_decay": 0.0})

        # Use AdamW (decoupled weight decay)
        self.optimizer = AdamW(parameters)

        # Warmup setup
        self.warmup_epochs = self.config.get('warmup_epochs', 0)
        self.base_motion_lr = self.config.get('motion_lr', 1e-5)
        self.base_head_lr = self.config.get('head_lr', 1e-4)

        log.info(f"LoRA groups (decay/no_decay): {len(lora_decay)}/{len(lora_no_decay)}")
        log.info(f"Classifier groups (decay/no_decay): {len(cls_decay)}/{len(cls_no_decay)}")
        log.info(f"Weight decay: {wd}")
        if self.warmup_epochs > 0:
            log.info(f"Warmup epochs: {self.warmup_epochs}")
    
    def _update_learning_rate_with_warmup(self, epoch):
        """Update learning rate with warmup schedule"""
        if epoch < self.warmup_epochs:
            # Linear warmup: gradually increase LR from 0 to base LR
            warmup_factor = (epoch + 1) / self.warmup_epochs
            motion_lr = self.base_motion_lr * warmup_factor
            head_lr = self.base_head_lr * warmup_factor
            
            # Update optimizer learning rates
            self.optimizer.param_groups[0]['lr'] = motion_lr  # LoRA params
            self.optimizer.param_groups[1]['lr'] = head_lr    # Classifier params
            
            log.info(f"Warmup epoch {epoch + 1}/{self.warmup_epochs}: "
                    f"Motion LR={motion_lr:.2e}, Head LR={head_lr:.2e}")
        else:
            # After warmup, use base learning rates (scheduler will handle further changes)
            self.optimizer.param_groups[0]['lr'] = self.base_motion_lr
            self.optimizer.param_groups[1]['lr'] = self.base_head_lr
    
    def _update_training_curves(self, save_dir: str):
        """Incrementally save training curves after each epoch to training_curves.png."""
        # Create a 1x3 subplot figure similar to the final summary
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))
        
        # Plot losses
        if len(self.train_losses) > 0:
            ax1.plot(self.train_losses, label='Train Loss')
        if len(self.val_losses) > 0:
            ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        if len(self.train_accs) > 0:
            ax2.plot(self.train_accs, label='Train Acc')
        if len(self.val_accs) > 0:
            ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot precision / recall / F1 (validation)
        if len(self.val_precisions) > 0:
            ax3.plot(self.val_precisions, label='Val Precision')
        if len(self.val_recalls) > 0:
            ax3.plot(self.val_recalls, label='Val Recall')
        if len(self.val_f1s) > 0:
            ax3.plot(self.val_f1s, label='Val F1')
        ax3.set_title('Validation Precision / Recall / F1')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score (%)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        out_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(tqdm(train_loader, desc="Training")):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            features, logits = self.model(batch_x)
            
            # Calculate loss based on prediction mode
            if self.config.get('prediction_mode', 'segment') == 'frame':
                # Frame-level prediction: logits shape [batch, seq_len, num_classes], labels shape [batch, seq_len]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                
                # Calculate accuracy for frame-level prediction
                predicted = torch.argmax(logits, dim=-1)  # [batch, seq_len]
                correct += (predicted == batch_y).sum().item()
                total += batch_y.numel()
            else:
                # Segment-level prediction: logits shape [batch, num_classes], labels shape [batch]
                loss = self.criterion(logits, batch_y)
                
                # Calculate accuracy for segment-level prediction
                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            
            loss.backward()
            
            # Apply gradient clipping if specified
            max_grad_norm = self.config.get('max_grad_norm', None)
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="Validating"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                features, logits = self.model(batch_x)
                
                # Calculate loss and predictions based on prediction mode
                if self.config.get('prediction_mode', 'segment') == 'frame':
                    # Frame-level prediction
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                    predicted = torch.argmax(logits, dim=-1)  # [batch, seq_len]
                    
                    # Flatten for metrics
                    all_preds.extend(predicted.cpu().numpy().flatten())
                    all_labels.extend(batch_y.cpu().numpy().flatten())
                    
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.numel()
                else:
                    # Segment-level prediction
                    loss = self.criterion(logits, batch_y)
                    predicted = torch.argmax(logits, dim=1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
                    
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_val_loss, accuracy, all_preds, all_labels
    
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
    
    def _compute_class_weights_from_dataset(self, dataset):
        """Compute per-class weights from the training dataset (frame-level aware)."""
        pos, neg = 0, 0
        for labels in dataset.labels:
            if isinstance(labels, list):
                # Frame-level labels: list of 0/1 per frame
                p = int(sum(labels))
                n = int(len(labels) - p)
            else:
                # Segment-level label
                p = int(labels == 1)
                n = int(labels == 0)
            pos += p
            neg += n

        # Avoid division by zero
        pos = max(pos, 1)
        neg = max(neg, 1)
        total = pos + neg

        # Class-balanced weights: w_c = total / (num_classes * count_c)
        w_non_tic = total / (2.0 * neg)
        w_tic = total / (2.0 * pos)

        weights = torch.tensor([w_non_tic, w_tic], dtype=torch.float, device=self.device)
        log.info(f"Class weights (non-tic, tic): {w_non_tic:.4f}, {w_tic:.4f}  (pos={pos}, neg={neg})")
        return weights

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

        # Rebuild criterion with class weights computed from training data (frame-level aware)
        try:
            class_weights = self._compute_class_weights_from_dataset(train_loader.dataset)
            self.criterion = FocalLoss(gamma=2.0, alpha=class_weights)
            log.info("Using class-weighted FocalLoss (gamma=2.0) to mitigate class imbalance")
        except Exception as e:
            log.warning(f"Failed to set class-weighted FocalLoss: {e}. Using unweighted FocalLoss.")
            self.criterion = FocalLoss(gamma=2.0, alpha=None)

        # Create scheduler per config
        sched_type = str(self.config.get('scheduler_type', 'cosine')).lower()
        if sched_type == 'cosine':
            # Step once per epoch -> set T_max in epochs (optionally minus warmup)
            tmax = max(1, int(epochs) - int(self.warmup_epochs or 0))
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=tmax)
            log.info(f"Scheduler: CosineAnnealingLR(T_max={tmax}) - stepped per epoch")
        elif sched_type == 'step':
            step_size = int(self.config.get('step_size', 30))
            gamma = float(self.config.get('step_gamma', 0.1))
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            log.info(f"Scheduler: StepLR(step_size={step_size}, gamma={gamma}) - stepped per epoch")
        else:
            self.scheduler = None
            log.info("Scheduler: disabled")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0
        best_val_loss = 1e5
        best_epoch = 0  # Change from -1 to 0 since epochs now start from 1
        epochs_without_improvement = 0  # Early stopping counter
        
        for epoch in range(1, epochs + 1):  # Start from epoch 1 instead of 0
            # Print progress - following MotionPatches format
            print(f"running epoch {epoch}, best test loss {best_val_loss:.4f} best val acc {best_val_acc:.2f}% after epoch {best_epoch}")
            
            # Update learning rate with warmup (epoch now starts from 1)
            self._update_learning_rate_with_warmup(epoch - 1)  # Pass 0-based epoch to warmup function
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            # Update scheduler (only after warmup)
            if self.scheduler is not None and (self.warmup_epochs == 0 or epoch > self.warmup_epochs):
                self.scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            # Compute precision, recall, F1 for validation set
            val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
            val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            self.val_precisions.append(val_precision * 100.0)
            self.val_recalls.append(val_recall * 100.0)
            self.val_f1s.append(val_f1 * 100.0)
            
            # Per-class metrics (Non-tic=0, Tic=1)
            try:
                cls_report = classification_report(
                    val_labels, val_preds,
                    labels=[0, 1], target_names=['Non-tic', 'Tic'],
                    output_dict=True, zero_division=0
                )
                nt = cls_report.get('Non-tic', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
                tc = cls_report.get('Tic', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
                log.info(
                    f"Per-class metrics | Non-tic: P {nt['precision']*100:.2f}% R {nt['recall']*100:.2f}% F1 {nt['f1-score']*100:.2f}% (support {int(nt['support'])}) | "
                    f"Tic: P {tc['precision']*100:.2f}% R {tc['recall']*100:.2f}% F1 {tc['f1-score']*100:.2f}% (support {int(tc['support'])})"
                )
            except Exception as e:
                log.warning(f"Failed to compute per-class metrics: {e}")
            
            # Log results - following MotionPatches format
            log.info(f"epoch {epoch}, tr_loss {train_loss:.4f}, te_loss {val_loss:.4f}, val_acc {val_acc:.2f}% val_P {val_precision*100:.2f}% val_R {val_recall*100:.2f}% val_F1 {val_f1*100:.2f}%")
            
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
            
            # Periodic checkpoint saving every N epochs (if enabled)
            if self.save_interval and self.save_interval > 0 and (epoch % self.save_interval == 0):
                ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                log.info(f"Checkpoint saved: {ckpt_path}")
            
            # Realtime update of training curves image after each epoch
            self._update_training_curves(save_dir)
            
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
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))
        
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
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # New subplot: precision / recall / F1
        ax3.plot(self.val_precisions, label='Val Precision')
        ax3.plot(self.val_recalls, label='Val Recall')
        ax3.plot(self.val_f1s, label='Val F1')
        ax3.set_title('Validation Precision / Recall / F1')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score (%)')
        ax3.legend()
        ax3.grid(True)
        
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


class FocalLoss(nn.Module):
    """Focal Loss for classification with optional class weights (alpha).
    This helps with class imbalance and hard example mining.
    Args:
        gamma (float): Focusing parameter. Default: 2.0
        alpha (Tensor or list or None): Class weights of shape [num_classes]. Default: None
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'
        ignore_index (int): Target value to ignore. Default: -100
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean', ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = alpha  # Tensor or None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute standard cross-entropy per-sample loss (no reduction)
        ce_loss = F.cross_entropy(
            logits, target,
            weight=self.alpha,
            reduction='none',
            ignore_index=self.ignore_index
        )
        # Convert CE to probability of correct class
        pt = torch.exp(-ce_loss)
        # Apply focal modulation
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        # Reduce as requested
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss