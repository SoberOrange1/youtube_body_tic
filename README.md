# Tic Classification using MotionPatches with LoRA

A deep learning framework for automatic tic detection in video sequences using Vision Transformer (ViT) with MotionPatches-inspired architecture and LoRA fine-tuning.

## Overview

This project implements a tic classification system that processes human pose landmarks from video sequences to automatically detect tic behaviors. The system uses a MotionPatches-inspired approach to convert kinematic chain data into patch-based representations suitable for Vision Transformer processing.

## Key Features

- **MotionPatches Architecture**: Converts pose sequences into 2D patch representations for ViT
- **LoRA Fine-tuning**: Memory-efficient fine-tuning using Low-Rank Adaptation
- **Hip-centered Normalization**: Coordinate normalization using hip center as origin
- **Kinematic Chain Processing**: Processes 5 major kinematic chains (face-to-torso, arms, etc.)
- **Cross-validation Support**: 5-fold cross-validation with video-based splits

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd motion_vit_processor

# Install dependencies
pip install torch torchvision torchaudio timm peft scikit-learn matplotlib seaborn numpy pandas tqdm
```

## Data Structure

```
data_folder/
├── body_detection_results/
│   ├── 01/
│   │   └── pose_results_01.json
│   └── ...
└── annotations/
    ├── annotations_01.json
    └── ...
```

## Quick Start

### Training

Train all 5 folds using our predefined data splits:

```bash
# Fold 1
python main.py --train_videos 04 07 10 11 14 16 17 --val_videos 01 08 --test_videos 05 06 --fold_name fold_1

# Fold 2  
python main.py --train_videos 01 05 06 07 11 14 17 --val_videos 16 04 --test_videos 10 08 --fold_name fold_2

# Fold 3
python main.py --train_videos 05 07 08 10 11 14 16 --val_videos 17 06 --test_videos 01 04 --fold_name fold_3

# Fold 4
python main.py --train_videos 01 04 06 08 10 16 17 --val_videos 05 11 --test_videos 07 14 --fold_name fold_4

# Fold 5
python main.py --train_videos 01 04 05 07 08 14 17 --val_videos 10 06 --test_videos 11 16 --fold_name fold_5
```

**Training Options:**
```bash
# Enhanced encoder with all 7 features
python main.py --encoder_type enhanced --train_videos 04 07 10 11 14 16 17 --val_videos 01 08 --fold_name fold_1

# Custom LoRA parameters
python main.py --lora_r 32 --lora_alpha 64 --epochs 50 --fold_name fold_1
```

### Testing

Test all folds in an experiment:

```bash
# Test all folds with best checkpoints
python tester.py --exp_dir outputs/exp_001

# Test specific fold
python tester.py --exp_dir outputs/exp_001 --single_fold fold_1_train_04_07_10_11_14_16_17_val_01_08_20250111_143022

# Test with custom test videos
python tester.py --exp_dir outputs/exp_001 --test_videos 05 06
```

## Cross-Validation Setup

We use 5-fold cross-validation with the following video splits:

| Fold | Train Videos | Val Videos | Test Videos |
|------|-------------|------------|-------------|
| 1 | 04,07,10,11,14,16,17 | 01,08 | 05,06 |
| 2 | 01,05,06,07,11,14,17 | 16,04 | 10,08 |
| 3 | 05,07,08,10,11,14,16 | 17,06 | 01,04 |
| 4 | 01,04,06,08,10,16,17 | 05,11 | 07,14 |
| 5 | 01,04,05,07,08,14,17 | 10,06 | 11,16 |

## Configuration

Key parameters in `main.py`:

```python
# Model Configuration
MODEL_CONFIG = {
    'encoder_type': 'basic',  # 'basic' or 'enhanced'
    'model_name': 'vit_base_patch16_224.augreg_in21k',
}

# LoRA Configuration
LORA_CONFIG = {
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.3,
}
```

## Results

Results are saved in structured experiment directories:

```
outputs/
└── exp_001/
    ├── fold_1_train_04_07_10_11_14_16_17_val_01_08_20250111_143022/
    │   ├── best_model.pt
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   └── test_results.json
    ├── fold_2_train_01_05_06_07_11_14_17_val_16_04_20250111_153045/
    ├── ...
    └── aggregated_test_results.json
```

The system outputs:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices and ROC curves
- Cross-validation statistics
- Training curves

## Architecture Details

### Encoders
- **Basic**: Single-channel motion intensity representation
- **Enhanced**: Multi-channel processing of all 7 features [x,y,z,visibility,ax,ay,az]

### Kinematic Chains
1. Face to torso (left): [0,2,5,8,11]
2. Face to torso (right): [0,1,4,7,10]
3. Torso: [11,12,23,24]
4. Left arm: [11,13,15]
5. Right arm: [12,14,16]

### Hip-centered Normalization
- Normalizes coordinates relative to hip center (landmarks 23,24)
- Scales by hip width for size invariance
- Enable/disable via `normalize_coords` parameter


**MotionPatches Reference:**
This work is inspired by the MotionPatches architecture. Please also cite the original MotionPatches paper:

```bibtex

@inproceedings{yu_exploring_2024,
	address = {Seattle, WA, USA},
	title = {Exploring {Vision} {Transformers} for {3D} {Human} {Motion}-{Language} {Models} with {Motion} {Patches}},
	copyright = {https://doi.org/10.15223/policy-029},
	isbn = {9798350353006},
	url = {https://ieeexplore.ieee.org/document/10656394/},
	doi = {10.1109/CVPR52733.2024.00095},
	language = {en},
	urldate = {2025-10-28},
	booktitle = {2024 {IEEE}/{CVF} {Conference} on {Computer} {Vision} and {Pattern} {Recognition} ({CVPR})},
	publisher = {IEEE},
	author = {Yu, Qing and Tanaka, Mikihiro and Fujiwara, Kent},
	month = jun,
	year = {2024},
	pages = {937--946}
}

```

## Acknowledgments

- **MotionPatches**: This work adapts the MotionPatches architecture for tic detection
- **MediaPipe**: For robust pose detection and landmark extraction
- **Hugging Face PEFT**: For efficient LoRA implementation
- **timm**: For pretrained Vision Transformer models

## License

This project is licensed under the MIT License.
