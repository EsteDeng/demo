# Recurrent Cross-Modality Fusion for ToF Depth Denoising (PyTorch Version)

This repository is a PyTorch reimplementation of the original code from [Recurrent Cross-Modality Fusion for ToF Depth Denoising](https://github.com/gtdong-ustc/recurrent_tof_denoising).

## Description

- **Original Author:** GT Dong et al.
- **Original Repository:** https://github.com/gtdong-ustc/recurrent_tof_denoising

## My Modifications

- Complete migration of the original TensorFlow codebase to PyTorch
- Adapted network architecture and training pipeline for PyTorch
- Preserved the core algorithm and experimental settings from the original work

## Usage

This project is under active development. 

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

```bash
python recurrent_tof_denoising_pytorch/pipe/start.py \
  -b 2 \
  -s 200000 \
  -m sample_pyramid_add_kpn \
  -p size384 \
  -k depth_kinect_with_gt_msk \
  -l 0.0004 \
  -t tof_FT3 \
  -i 480 640 \
  -o mean_l1 \
  --addGradient sobel_gradient \
  -g 4 \
  -e 1200