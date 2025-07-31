# MTC-HSDNet: Multi-Task Collaboration and Hierarchical Self-Distillation in a Multi-Level Transformer Fusion Network for Wilms' Tumor Segmentation and Metastasis Prediction

[![GitHub stars](https://img.shields.io/github/stars/wenjing-gg/MTC-HSDNet?style=social)](https://github.com/wenjing-gg/MTC-HSDNet/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/wenjing-gg/MTC-HSDNet?style=social)](https://github.com/wenjing-gg/MTC-HSDNet/network/members)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://github.com/wenjing-gg/MTC-HSDNet)
[![arXiv](https://img.shields.io/badge/arXiv-2024.00000-b31b1b.svg)](https://arxiv.org/abs/2024.00000)

**This repository is the official implementation of MTC-HSDNet.**

A deep learning framework for simultaneous Wilms' tumor segmentation and metastasis prediction using multi-task collaboration and hierarchical self-distillation in a multi-level transformer fusion network.

## Overview

MTC-HSDNet is a specialized deep learning framework designed for Wilms' tumor analysis, combining multi-task collaboration and hierarchical self-distillation for simultaneous tumor segmentation and metastasis prediction. The network architecture incorporates:

- **Multi-Level Transformer Fusion** for comprehensive feature integration
- **Hierarchical Self-Distillation** for progressive knowledge transfer
- **Multi-Task Collaboration** between segmentation and classification tasks
- **Swin Transformer backbone** for robust feature extraction
- **Feature Pyramid Network (FPN)** for multi-scale feature fusion
- **Mixture of Experts (MoE)** for adaptive feature processing

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

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mtc-hsdnet,
  title={MTC-HSDNet: Multi-Task Collaboration and Hierarchical Self-Distillation in a Multi-Level Transformer Fusion Network for Wilms' Tumor Segmentation and Metastasis Prediction},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact: [your-email@example.com]
