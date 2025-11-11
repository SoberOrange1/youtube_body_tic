# Tic Classification using MotionPatches with LoRA

A deep learning framework for automatic tic detection in video sequences using Vision Transformer (ViT) with MotionPatches-inspired architecture and LoRA fine-tuning.

## Overview

This project implements a tic classification system that processes human pose landmarks from video sequences to automatically detect tic behaviors. The system uses a MotionPatches-inspired approach to convert kinematic chain data into patch-based representations suitable for Vision Transformer processing.

## Features

- **MotionPatches Architecture**: Converts pose sequences into 2D patch representations for ViT
- **LoRA Fine-tuning**: Memory-efficient fine-tuning using Low-Rank Adaptation
- **Hip-centered Normalization**: Coordinate normalization using hip center as origin
- **Kinematic Chain Processing**: Processes 5 major kinematic chains (face-to-torso, arms, etc.)
- **Cross-validation Support**: Video-based splits to prevent data leakage
- **Comprehensive Evaluation**: Detailed metrics and visualization tools

## Architecture

### Encoders
- **Basic Encoder**: Single-channel motion intensity representation
- **Enhanced Encoder**: Multi-channel processing of all 7 features [x, y, z, visibility, ax, ay, az]

### Key Components
- MediaPipe pose landmark detection (33 keypoints)
- Kinematic chain-based feature extraction
- Vision Transformer backbone with LoRA adaptation
- Binary classification (tic vs non-tic)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd motion_vit_processor

# Install dependencies
pip install torch torchvision torchaudio
pip install timm
pip install peft
pip install scikit-learn
pip install matplotlib seaborn
pip install numpy pandas
pip install tqdm
```

## Data Structure

```
data_folder/
├── body_detection_results/
│   ├── 01/
│   │   └── pose_results_01.json
│   ├── 02/
│   │   └── pose_results_02.json
│   └── ...
└── annotations/
    ├── annotations_01.json
    ├── annotations_02.json
    └── ...
```

### Pose Data Format
```json
{
  "video_id": "01",
  "frames": {
    "0": {
      "landmarks": [
        {
          "id": 0,
          "name": "NOSE",
          "x": 0.5,
          "y": 0.3,
          "z": -0.1,
          "visibility": 0.99
        }
      ]
    }
  }
}
```

### Annotation Format
```json
{
  "01": {
    "frame_annotations": [
      {
        "start_frame": 10,
        "end_frame": 50,
        "label": "tic"
      }
    ]
  }
}
```

## Usage

### Training

```bash
# Basic training with default settings
python main.py

# Custom video split
python main.py --train_videos 01 02 03 --val_videos 04 --test_videos 05

# Enhanced encoder with coordinate normalization
python main.py --encoder_type enhanced --fold_name fold_1

# Custom LoRA parameters
python main.py --lora_r 32 --lora_alpha 64 --epochs 50
```

### Testing

```bash
# Test all folds in an experiment
python tester.py --exp_dir outputs/exp_001

# Test specific fold
python tester.py --exp_dir outputs/exp_001 --single_fold fold_1_train_01_02_03_val_04_20250111_143022

# Test with custom videos
python tester.py --exp_dir outputs/exp_001 --test_videos 05 06
```

## Configuration

Key parameters can be configured in `main.py`:

```python
# Model Configuration
MODEL_CONFIG = {
    'encoder_type': 'basic',  # or 'enhanced'
    'model_name': 'vit_base_patch16_224_in21k',
    'feature_dim': 7,  # [x, y, z, visibility, ax, ay, az]
}

# LoRA Configuration
LORA_CONFIG = {
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

# Data Configuration
DATA_CONFIG = {
    'max_sequence_length': 64,
    'normalize_coords': True,  # Hip-centered normalization
}
```

## Results

The system outputs comprehensive evaluation metrics including:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Training curves
- Cross-validation statistics

Results are saved in structured experiment directories:
```
outputs/
└── exp_001/
    ├── fold_1_train_01_02_03_val_04_20250111_143022/
    │   ├── best_model.pt
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   └── test_results.json
    └── aggregated_test_results.json
```

## Key Features

### Hip-centered Coordinate Normalization
- Normalizes all coordinates relative to hip center
- Scales by hip width for size invariance
- Configurable via `normalize_coords` parameter

### Kinematic Chains
1. Face to torso (left): [0, 2, 5, 8, 11]
2. Face to torso (right): [0, 1, 4, 7, 10]
3. Torso: [11, 12, 23, 24]
4. Left arm: [11, 13, 15]
5. Right arm: [12, 14, 16]

### LoRA Fine-tuning
- Reduces trainable parameters by ~90%
- Enables efficient fine-tuning of large ViT models
- Configurable rank and alpha parameters

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tic_classification_motionpatches,
  title={Tic Classification using MotionPatches with LoRA},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-username]/[repository-name]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MotionPatches architecture inspiration
- MediaPipe for pose detection
- Hugging Face PEFT for LoRA implementation
- timm for Vision Transformer models