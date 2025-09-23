# Gradient Descent Optimization of Spin System Parameters for NMR Metabolomics

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)

This repository contains the implementation of a gradient descent-based optimization algorithm for automatically determining spin system parameters (chemical shifts, J-coupling constants, relaxation rates, and intensity corrections) from experimental NMR spectra. The method is designed to improve the accuracy of metabolite identification in complex biological samples without manual intervention.

## 📖 Overview

Nuclear Magnetic Resonance (NMR) spectroscopy is a powerful tool for metabolomics, but manual processing of spectra remains a bottleneck. This project introduces a model built in PyTorch that optimizes parameters via gradient descent to match experimental data

The method has been validated on common amino acids, including L-proline.

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AleksandrIV/NMRSpectrumOptimization
cd NMRSpectrumOptimization
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook

# 3. Then in the browser:
#    - Click optimization.ipynb
#    - Change metabolite name in cell 2
#    - Run all cells (Cell → Run All)
