# MTC-HSDNet

A Multi-Task Classification and High-resolution Segmentation Deep Network for medical image analysis.

## Overview

MTC-HSDNet is a deep learning framework that combines multi-task learning for both classification and segmentation tasks on medical images. The network architecture incorporates:

- **Swin Transformer backbone** for feature extraction
- **Feature Pyramid Network (FPN)** for multi-scale feature fusion
- **Mixture of Experts (MoE)** for adaptive feature processing
- **Multi-task learning** with uncertainty weighting

## Features

- **Multi-task Learning**: Simultaneous classification and segmentation
- **3D Medical Image Support**: Designed for 3D medical imaging data (NRRD format)
- **Advanced Augmentation**: Comprehensive data augmentation pipeline
- **Uncertainty Weighting**: Adaptive loss balancing between tasks
- **Knowledge Distillation**: Progressive supervised distillation for improved performance

## Requirements

See `requirements.txt` for detailed dependencies. Key requirements include:

- Python 3.8+
- PyTorch 2.4.0+
- MONAI 1.4.0+
- NumPy, SciPy, scikit-learn
- SimpleITK for medical image processing

## Installation

1. Clone the repository:
```bash
git clone git@github.com:wenjing-gg/MTC-HSDNet.git
cd MTC-HSDNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data_path /path/to/your/data --epochs 100 --batch_size 2
```

### Key Parameters

- `--data_path`: Path to the dataset directory
- `--weights`: Path to pre-trained weights (optional)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--freeze_layers`: Whether to freeze backbone layers

## Dataset Structure

The expected dataset structure:
```
data/
├── train/
│   ├── 0/  # No metastasis
│   │   ├── image1.nrrd
│   │   ├── image1_label.nrrd
│   │   └── ...
│   └── 1/  # Metastasis
│       ├── image2.nrrd
│       ├── image2_label.nrrd
│       └── ...
└── test/
    ├── 0/
    └── 1/
```

## Model Architecture

The network consists of:

1. **Encoder**: Swin Transformer with progressive feature extraction
2. **Decoder**: U-Net style decoder for segmentation
3. **FPN**: Feature Pyramid Network for multi-scale fusion
4. **MoE**: Mixture of Experts for adaptive processing
5. **Multi-head**: Separate heads for classification and segmentation

## Loss Functions

- **Classification**: Adaptive Uncertainty Focal Loss
- **Segmentation**: Combined Dice + Cross-Entropy Loss
- **Distillation**: Progressive supervised distillation
- **Multi-task**: Uncertainty-weighted combination

## Metrics

- **Classification**: Accuracy, AUC, Sensitivity, Specificity
- **Segmentation**: Dice Score, IoU, ASSD, HD95

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mtc-hsdnet,
  title={MTC-HSDNet: Multi-Task Classification and High-resolution Segmentation Deep Network},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact: [your-email@example.com]
