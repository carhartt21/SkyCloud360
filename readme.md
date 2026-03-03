# SkyCloudNet360

This repository contains the implementation of **SkyCloud360: Sky and Cloud Segmentation in Equirectangular Images**. It extends the original SkyCloudNet architecture with four geometric adaptation strategies for processing 360° equirectangular images.

![cloudseg_example_2](https://user-images.githubusercontent.com/24622304/187810011-82a2c390-9074-4d8f-92e3-6b350c29d566.png)

## 360° Adaptation Strategies

| Mode | Description |
|------|-------------|
| **SkyCloudNet-CM** | Standard cubemap: decomposes the equirectangular image into 6 faces (90° FoV), processes each face independently, and reassembles. |
| **SkyCloudNet-ECM** | Extended cubemap: uses overlapping faces (>90° FoV) with distance-based blending to reduce boundary artifacts. |
| **SkyCloudNet-TPP** | Tangent plane projection: samples multiple gnomonic projections on the sphere, processes each local perspective patch, and blends with cosine weighting. Achieves the highest accuracy (93.63% sky, 85.46% cloud). |
| **SkyCloudNet-EQC** | Equirectangular convolutions: replaces standard Conv2d with latitude-adaptive kernels that compensate for horizontal stretching near the poles. Lightweight, in-place modification. |

## Requirements
- Python >= 3.8
- CUDA 10.2+

Install dependencies:
```
pip install -r requirements.txt
```

## Dataset and Weights
The **SkyCloud** dataset and pretrained weights are available at: https://osf.io/y69ah/?view_only=889215916ccb4c52a5971fffc6af0dda

The **SkyCloud360** dataset (600 equirectangular images) is also available at the same link.

## Quick Start

1. Download the dataset and update `root_dataset` in `config/config.yaml`.
2. Download pretrained weights to the `weights/` folder.

### Evaluation

#### Baseline (standard SkyCloudNet)
```bash
python3 eval.py --cfg config/config.yaml MODEL.equirect_mode none
```

#### SkyCloudNet-TPP (Tangent Plane Projection)
```bash
python3 eval.py --cfg config/config.yaml MODEL.equirect_mode tpp
```

#### SkyCloudNet-ECM (Extended Cubemap)
```bash
python3 eval.py --cfg config/config.yaml MODEL.equirect_mode ecm
```

#### SkyCloudNet-EQC (Equirectangular Convolutions)
```bash
python3 eval.py --cfg config/config.yaml MODEL.equirect_mode eqc
```

#### SkyCloudNet-CM (Standard Cubemap)
```bash
python3 eval.py --cfg config/config.yaml MODEL.equirect_mode cm
```

### Training

Training uses a polynomial learning rate schedule (initial LR 0.02, power 0.9), SGD optimizer with momentum 0.9 and weight decay 1e-4. The attribute estimation head is frozen during training (weights from pretrained SkyCloudNet). Data augmentation includes random cropping, horizontal flipping, and resizing.

#### Baseline training
```bash
python3 train.py --cfg config/config.yaml MODEL.equirect_mode none
```

#### Train with Tangent Plane Projection (TPP)
```bash
python3 train.py --cfg config/config.yaml MODEL.equirect_mode tpp
```

#### Train with Equirectangular Convolutions (EQC)
```bash
python3 train.py --cfg config/config.yaml MODEL.equirect_mode eqc
```

#### Train with Extended Cubemap (ECM)
```bash
python3 train.py --cfg config/config.yaml MODEL.equirect_mode ecm
```

#### Train with Standard Cubemap (CM)
```bash
python3 train.py --cfg config/config.yaml MODEL.equirect_mode cm
```

#### Resume training from a checkpoint
```bash
python3 train.py --cfg config/config.yaml \
    TRAIN.start_epoch 10 \
    MODEL.equirect_mode tpp \
    TRAIN.optim_data weights/optimizer_state.pth
```

#### Training with validation
```bash
python3 train.py --cfg config/config.yaml \
    MODEL.equirect_mode tpp \
    TRAIN.eval True \
    TRAIN.eval_step 5
```

## Configuration

All 360° parameters can be set in `config/config.yaml` or via command-line overrides:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL.equirect_mode` | `none` | Adaptation mode: `none`, `cm`, `ecm`, `tpp`, `eqc` |
| `MODEL.cubemap_face_size` | `256` | Resolution of cubemap faces (CM/ECM) |
| `MODEL.ecm_overlap` | `0.1` | Overlap fraction for extended cubemap |
| `MODEL.tpp_patch_size` | `256` | Tangent plane patch resolution |
| `MODEL.tpp_fov_deg` | `60.0` | Field of view per tangent patch (degrees) |
| `MODEL.tpp_num_lat` | `4` | Number of latitude bands for TPP sampling |
| `MODEL.tpp_num_lon` | `8` | Longitude samples per latitude band |
| `MODEL.eqc_replace_encoder` | `True` | Replace encoder convolutions (EQC) |
| `MODEL.eqc_replace_decoder` | `True` | Replace decoder convolutions (EQC) |

## Repository Structure

```
├── model.py              # Original SkyCloudNet architecture
├── model_360.py          # 360° adaptation wrappers (CM, ECM, TPP, EQC)
├── equirect_utils.py     # Equirectangular coordinate transforms & projections
├── equirect_conv.py      # Latitude-adaptive equirectangular convolution
├── train.py              # Training script with 360° support
├── eval.py               # Evaluation script with 360° support
├── data.py               # Dataset loaders
├── utils.py              # Utility functions (metrics, visualization, etc.)
├── custom_transforms.py  # Data augmentation transforms
├── config/
│   ├── config.yaml       # Main configuration file
│   └── defaults.py       # Default configuration values
├── models/               # Backbone architectures (ResNet, MobileNet)
└── weights/              # Pretrained model weights
```

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@INPROCEEDINGS{11200399,
  author={Gerhardt, Christoph and Broll, Wolfgang},
  booktitle={2025 10th International Conference on Image, Vision and Computing (ICIVC)}, 
  title={SkyCloud360: Sky and Cloud Segmentation in Equirectangular Images}, 
  year={2025},
  volume={},
  number={},
  pages={48-58},
  keywords={Image analysis;Clouds;Semantic segmentation;Image edge detection;Neural networks;Weather forecasting;Solar energy;Semisupervised learning;Distortion;Monitoring;Semantic segmentation;scene understanding;image analysis;neural networks;datasets},
  doi={10.1109/ICIVC66358.2025.11200399}}
```
