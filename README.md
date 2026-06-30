# Sketch-to-Image Translation for Fashion Design Using Generative Models

**Graduation Thesis - Hanoi University of Science and Technology (HUST)**

**Author:** Vu Nhu Duc (Program: Data Science and Artificial Intelligence)  
**Supervisor:** Dr. Dang Tuan Linh

---

## 📖 Overview

This repository contains the source code, dataset generation scripts, and LaTeX documentation for the graduation thesis **"Sketch-to-Image Translation for Fashion Design Using Generative Models."**

Fashion sketch-to-image translation bridges the gap between conceptual hand-drawn sketches and realistic garment visualizations. While traditional models (such as Pix2Pix, CycleGAN, or MUNIT) struggle to disentangle structural layouts from style variations, state-of-the-art diffusion architectures incur immense computational overhead, rendering them impractical for real-time interactive design.

To address these limitations, this thesis proposes **Mobile Sketch-to-Image (MS2I)**, a highly efficient GAN-based framework. MS2I is specifically designed to deliver photorealistic, color-controlled image synthesis that strictly adheres to input topologies while drastically minimizing computational complexity and inference latency.

## ✨ Key Contributions

1. **Large-Scale Fashion Dataset:** An automated pipeline that constructs a comprehensive dataset of approximately 66,000 paired sketch-image samples, augmented with fundamental color annotations and multiple synthetic sketch modalities.
2. **MS2I Architecture (RepTransformer):** Features a U-Net topology integrated with RepTransformer blocks. These blocks leverage structural reparameterization and Singular Value Decomposition (SVD) to maximize representational capacity during training while achieving aggressive parameter compression during inference.
3. **Graduated Style Modulation:** A mechanism that injects color constraints across multiple decoder stages, ensuring precise style disentanglement without disrupting structural geometry.
4. **Lightweight & Near Real-Time:** 
   - Achieves the best FID, LPIPS, and CLIP-S scores compared to established baselines.
   - **Parameters:** 5.42 million
   - **Computational Cost:** 1.46 GFLOPs
   - **Inference Speed:** ~329 FPS on GPU and ~15 FPS on a standard CPU.

## 📁 Repository Structure

- `thesis_latex/`: The complete LaTeX source code for the graduation thesis document (including Abstract, Methodology, Numerical Results, etc.).
- `models_pSp/` / `models_pix2pix_cyclegan/`: Baseline generative models used for comparative evaluation.
- `Data/` / `Model/`: Directory structures for dataset storage and trained model checkpoints.
- `demo/` / `demo_baseline/`: Source code for the near real-time interactive web demonstration.
- `notebook/`: Jupyter Notebooks used for training, evaluation, and data exploration.
- Various Python scripts (`extract.py`, `patch_features.py`, etc.) for preprocessing and converting model topologies.

## 🚀 Future Work & Usage

*(Detailed instructions for environment setup, data preparation, training, and running the real-time inference web demo will be provided here.)*

## 📄 License

This project is submitted in partial fulfillment of the requirements for the Graduation Thesis at Hanoi University of Science and Technology. All rights reserved by the author.