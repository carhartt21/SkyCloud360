# SkyCloud360: Sky and Cloud Segmentation in Equirectangular Images

## Overview
This repository accompanies the paper *SkyCloud360: Neural Network-Based Sky and Cloud Segmentation in Equirectangular Images*. The paper introduces the **SkyCloud360 dataset**, the first benchmark dataset for sky and cloud segmentation in equirectangular images, alongside adaptations to the **SkyCloudNet architecture** for handling geometric distortions inherent in 360° imagery.

### Key Contributions:
- **SkyCloud360 Dataset**: A collection of 600 high-resolution equirectangular images with dense annotations for terrain, sky, thin clouds, thick clouds, and estimated sun positions. The dataset bridges a critical gap in panoramic atmospheric analysis.
- **Geometric Adaptations**: Optimized versions of SkyCloudNet leveraging cubemap projections, tangent plane projections, extended cubemaps, and equirectangular convolutions to address spherical distortions.
- **Comprehensive Evaluation**: Benchmarks of state-of-the-art methods for semantic segmentation and domain adaptation on equirectangular imagery.

The dataset is publicly available at [OSF](https://osf.io/a5ew), enabling researchers to explore new directions in panoramic image analysis.

---

## Repository Content
This repository will include:
1. **SkyCloudNet Adaptations**: Implementations of geometric processing strategies (cubemaps, tangent planes, equirectangular convolutions).
2. **Training Scripts**: Code for training SkyCloudNet on the SkyCloud dataset and evaluating it on SkyCloud360.
3. **Evaluation Metrics**: Tools for calculating pixel accuracy, mIoU, and class-specific performance metrics.
4. **Visualization Tools**: Scripts for generating qualitative segmentation results.

---

## Current Status
The code is currently being finalized and will be updated in the next few weeks. Stay tuned for:
- Pretrained models for each geometric adaptation.
- Detailed documentation on how to use the dataset and train/evaluate models.

---

## Dataset Access
The **SkyCloud360 dataset** can be downloaded from [OSF](https://osf.io/a5ew). It includes:
- 600 annotated equirectangular images.
- Labels for terrain, sky, thin clouds, thick clouds, and sun position estimates.


