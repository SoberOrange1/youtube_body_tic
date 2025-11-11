"""
Tic Motion Encoder - Adapted from MotionPatches ViT
Processes tic body coordinate data using Vision Transformer with patch-based approach
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from typing import Optional, Tuple


class MotionPatchesEncoder(nn.Module):
    """
    Motion encoder inspired by MotionPatches architecture
    Converts kinematic chain data into 2D patch-based representation for ViT
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224_in21k",
        pretrained: bool = True,
        trainable: bool = True,
        patch_size: int = 16,
        num_patches: int = 5,  # Number of kinematic chains
        max_frames: int = 64,  # Sequence length
        input_channels: int = 1,  # Single channel for motion "image"
        num_classes: int = 2,  # tic vs non-tic
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.max_frames = max_frames
        self.input_channels = input_channels
        
        # Define kinematic chains for human pose (MediaPipe format)
        self.kinematic_chains = [
            [0, 2, 5, 8, 11],      # Face to torso
            [0, 1, 4, 7, 10],      # Face to torso (other side)
            [11, 12, 23, 24],      # Torso
            [11, 13, 15],          # Left arm
            [12, 14, 16],          # Right arm
        ]
        
        # MotionPatches style: treat motion as 2D image
        # Height: time frames, Width: patch_size * num_patches
        img_height = max_frames
        img_width = patch_size * num_patches
        
        # Create ViT backbone following MotionPatches design
        self.motion_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=(img_height, img_width),  # (64, 80)
            in_chans=input_channels
        )
        
        # Set trainable parameters
        for param in self.motion_encoder.parameters():
            param.requires_grad = trainable
        
        # Get feature dimension
        self.feature_dim = self.motion_encoder.num_features
        
        # Classification head for tic detection
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def apply_kinematic_chain_patches(self, motion_features: torch.Tensor) -> torch.Tensor:
        """
        Convert kinematic chain features to MotionPatches style 2D representation
        
        Args:
            motion_features: (batch_size, frames, landmarks, coords)
                           e.g., (batch, 64, 33, 3) for normalized coordinates
        
        Returns:
            patch_image: (batch_size, 1, frames, patch_size * num_patches)
                        e.g., (batch, 1, 64, 80)
        """
        batch_size, frames, landmarks, coords = motion_features.shape
        
        # Extract normalized coordinates (first 3 features: x, y, z)
        motion = motion_features[:, :, :, :3]  # (batch, frames, landmarks, 3)
        
        # Process each kinematic chain into patches
        patches = []
        
        for chain_idx, chain in enumerate(self.kinematic_chains):
            # Extract joints for this kinematic chain
            if all(j < landmarks for j in chain):
                chain_joints = motion[:, :, chain, :]  # (batch, frames, chain_len, 3)
                
                # Calculate motion magnitude for each frame and joint
                # Use Euclidean norm across coordinates as motion intensity
                chain_motion = torch.norm(chain_joints, dim=-1)  # (batch, frames, chain_len)
                
                # Interpolate to fixed patch size (16 points)
                chain_motion = chain_motion.permute(0, 2, 1)  # (batch, chain_len, frames)
                chain_motion = nn.functional.interpolate(
                    chain_motion.unsqueeze(1),  # (batch, 1, chain_len, frames)
                    size=(self.patch_size, frames),
                    mode='bilinear',
                    align_corners=False
                )  # (batch, 1, patch_size, frames)
                
                chain_motion = chain_motion.squeeze(1).permute(0, 2, 1)  # (batch, frames, patch_size)
                patches.append(chain_motion)
            else:
                # Fallback: create zero patch if chain indices are invalid
                zero_patch = torch.zeros(batch_size, frames, self.patch_size, 
                                       device=motion.device, dtype=motion.dtype)
                patches.append(zero_patch)
        
        # Concatenate all patches: (batch, frames, patch_size * num_patches)
        motion_patches = torch.cat(patches, dim=-1)  # (batch, frames, 80)
        
        # Add channel dimension for ViT: (batch, 1, frames, total_patch_width)
        motion_image = motion_patches.unsqueeze(1)  # (batch, 1, frames, 80)
        
        return motion_image
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with MotionPatches style processing
        
        Args:
            x: Input motion features (batch_size, frames, landmarks, features)
        
        Returns:
            features: Extracted features (batch_size, feature_dim)
            logits: Classification logits (batch_size, num_classes)
        """
        # Convert to motion patch representation
        motion_image = self.apply_kinematic_chain_patches(x)
        
        # Extract features using ViT
        features = self.motion_encoder(motion_image)
        
        # Classification
        logits = self.classifier(features)
        
        return features, logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        motion_image = self.apply_kinematic_chain_patches(x)
        return self.motion_encoder(motion_image)


class EnhancedMotionPatchesEncoder(nn.Module):
    """
    Enhanced version with multi-dimensional feature support
    Supports processing of all 7 features: [x, y, z, visibility, ax, ay, az]
    Uses concat fusion to treat each feature as a separate channel
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224_in21k",
        pretrained: bool = True,
        trainable: bool = True,
        patch_size: int = 16,
        num_patches: int = 5,
        max_frames: int = 64,
        feature_dim: int = 7,  # [x, y, z, visibility, ax, ay, az]
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.max_frames = max_frames
        self.feature_dim = feature_dim
        
        self.kinematic_chains = [
            [0, 2, 5, 8, 11],      # Face to torso
            [0, 1, 4, 7, 10],      # Face to torso (other side)
            [11, 12, 23, 24],      # Torso
            [11, 13, 15],          # Left arm
            [12, 14, 16],          # Right arm
        ]
        
        # Use feature_dim as input channels (7 channels)
        input_channels = feature_dim
        img_height = max_frames
        img_width = patch_size * num_patches
        
        # Create ViT backbone with 7 input channels
        self.motion_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=(img_height, img_width),  # (64, 80)
            in_chans=input_channels  # 7 channels
        )
        
        for param in self.motion_encoder.parameters():
            param.requires_grad = trainable
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.motion_encoder.num_features),
            nn.Dropout(0.1),
            nn.Linear(self.motion_encoder.num_features, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def apply_enhanced_patches(self, motion_features: torch.Tensor) -> torch.Tensor:
        """
        Enhanced patch processing with all 7 features as separate channels
        
        Args:
            motion_features: (batch_size, frames, landmarks, 7)
                           Features: [x, y, z, visibility, ax, ay, az]
        
        Returns:
            motion_image: (batch_size, 7, frames, 80) for ViT input
        """
        batch_size, frames, landmarks, feat_dim = motion_features.shape
        
        # Initialize patches for each feature dimension
        all_feature_patches = []
        
        # Process each feature dimension separately
        for f_idx in range(feat_dim):
            feature_data = motion_features[:, :, :, f_idx]  # (batch, frames, landmarks)
            
            # Process each kinematic chain for this feature
            feature_patches = []
            
            for chain_idx, chain in enumerate(self.kinematic_chains):
                if all(j < landmarks for j in chain):
                    # Extract this chain's data for this feature
                    chain_data = feature_data[:, :, chain]  # (batch, frames, chain_len)
                    
                    # Interpolate to patch size
                    chain_data = chain_data.permute(0, 2, 1)  # (batch, chain_len, frames)
                    chain_data = nn.functional.interpolate(
                        chain_data.unsqueeze(1),  # (batch, 1, chain_len, frames)
                        size=(self.patch_size, frames),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)  # (batch, patch_size, frames)
                    
                    chain_data = chain_data.permute(0, 2, 1)  # (batch, frames, patch_size)
                    feature_patches.append(chain_data)
                else:
                    # Zero patch for invalid chains
                    zero_patch = torch.zeros(batch_size, frames, self.patch_size,
                                           device=motion_features.device, 
                                           dtype=motion_features.dtype)
                    feature_patches.append(zero_patch)
            
            # Concatenate all patches for this feature: (batch, frames, 80)
            feature_motion_patch = torch.cat(feature_patches, dim=-1)
            all_feature_patches.append(feature_motion_patch)
        
        # Stack all features as channels: (batch, 7, frames, 80)
        motion_image = torch.stack(all_feature_patches, dim=1)
        
        return motion_image
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with 7-channel input
        
        Args:
            x: Input motion features (batch_size, frames, landmarks, 7)
        
        Returns:
            features: Extracted features (batch_size, feature_dim)
            logits: Classification logits (batch_size, num_classes)
        """
        # Convert to 7-channel motion image
        motion_image = self.apply_enhanced_patches(x)
        
        # Extract features using ViT
        features = self.motion_encoder(motion_image)
        
        # Classification
        logits = self.classifier(features)
        
        return features, logits


# 保持原有的其他编码器类不变
class TicMotionEncoder(nn.Module):
    """
    Motion encoder specifically designed for tic detection
    Adapts ViT architecture to process body coordinate sequences
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224_in21k",
        pretrained: bool = True,
        trainable: bool = True,
        patch_size: int = 16,
        input_channels: int = 3,  # Can be adjusted for [x,y,z] or more features
        num_joints: int = 33,
        max_sequence_length: int = 64,
        num_classes: int = 2,  # tic vs non-tic
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.num_joints = num_joints
        self.max_sequence_length = max_sequence_length
        
        # Calculate input image dimensions for ViT
        # We'll treat motion as an "image" where:
        # - Height: sequence_length (time dimension)
        # - Width: num_joints (spatial dimension)
        # - Channels: feature dimensions (x,y,z or x,y,z,ax,ay,az)
        img_height = max_sequence_length
        img_width = num_joints
        
        # Create ViT model
        self.vit_backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg",
            img_size=(img_height, img_width),
            in_chans=input_channels  # 关键：支持多通道输入
        )
        
        # Set trainable parameters
        for param in self.vit_backbone.parameters():
            param.requires_grad = trainable
        
        # Get feature dimension from ViT
        self.feature_dim = self.vit_backbone.num_features
        
        # Classification head for tic detection
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length, num_joints)
               For tic data: (batch_size, 3, frames, 33) for [x,y,z] coordinates
                          or (batch_size, 7, frames, 33) for all features
        
        Returns:
            features: Extracted features (batch_size, feature_dim)
            logits: Classification logits (batch_size, num_classes)
        """
        # Extract features using ViT backbone
        features = self.vit_backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return features, logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        return self.vit_backbone(x)