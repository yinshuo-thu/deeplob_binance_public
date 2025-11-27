# DeepLOB Binance - Cryptocurrency LOB Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Author**: [Shuo Yin](https://yinshuo-thu.github.io/) (yins25@mails.tsinghua.edu.cn)

This repository contains a **Millennium ML Internship Take-Home Project** focused on building high-frequency limit order book (LOB) prediction models for cryptocurrency markets using deep learning.

---

## 📋 Project Overview

This project implements and improves upon the DeepLOB architecture for predicting short-term returns from cryptocurrency limit order book data on Binance. Starting from the baseline CNN-LSTM architecture, we explore multiple enhancements including EMA smoothing, Temporal Convolutional Networks (TCN), and hierarchical multi-task learning.

### Key Achievements:
- ✅ Reconstructed historical Level 2 LOB data for 50 most liquid crypto assets on Binance
- ✅ Implemented DeepLOB baseline and 3 architectural improvements
- ✅ Achieved **0.31 correlation** and **2.55 BPS MAE** with TCN-EMA architecture
- ✅ 40% faster training with TCN vs LSTM
- ✅ Open-sourced 25M+ samples dataset and model weights

---

## 📁 Repository Structure

```
deeplob-binance-github/
├── README.md                          # This file - project overview
├── DeepLOB-Binance-EN.pdf             # English project report
├── DeepLOB-Binance.pdf                # Chinese project report (original)
└── deeplob_crypto_code/               # Main codebase
    ├── README.md                      # Detailed technical documentation
    ├── 1_crawl_top50.py               # Data: Crawl top 50 liquid symbols
    ├── 2_collect_lob_fast.py          # Data: Collect LOB from Tardis API
    ├── 2_config.json                  # Configuration for data collection
    ├── 3_deeplob_baseline.py          # Model: DeepLOB baseline (LSTM)
    ├── 4_deeplob_tcn.py               # Model: DeepLOB with TCN
    ├── 6_baseline_ema.py              # Model: DeepLOB with EMA smoothing
    ├── 7_baseline_tcu_ema.py          # Model: TCN + EMA (best performance)
    ├── 8_baseline_hierarchical_ema.py # Model: Hierarchical multi-task learning
    ├── candidates.txt                 # Top 50 liquid crypto symbols
    └── images/                        # Performance charts and visualizations
```

### 📄 Documentation Files

- **`DeepLOB-Binance.pdf`**: Original Chinese project report with detailed methodology and results
- **`DeepLOB-Binance-EN.pdf`**: English translation of the project report
- **`deeplob_crypto_code/README.md`**: Comprehensive technical documentation with usage instructions

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch numpy pandas scikit-learn matplotlib seaborn pyyaml tqdm tardis-dev huggingface_hub
```

### Run Training

```bash
cd deeplob_crypto_code

# 1. Crawl top 50 liquid symbols
python3 1_crawl_top50.py

# 2. Collect LOB data (requires Tardis API key)
python3 2_collect_lob_fast.py

# 3. Train baseline model
python3 3_deeplob_baseline.py

# 4. Train best model (TCN + EMA)
python3 7_baseline_tcu_ema.py
```

For detailed instructions, see [`deeplob_crypto_code/README.md`](deeplob_crypto_code/README.md).

---

## 📊 Results Summary

| Model | Architecture | Correlation ↑ | MAE (BPS) ↓ | RMSE (BPS) ↓ | Training Speed |
|-------|-------------|---------------|-------------|--------------|----------------|
| Baseline | CNN-LSTM | 0.084 | 3.32 | 4.86 | 1.0x |
| EMA-LSTM | LSTM + EMA | 0.274 | 2.55 | 3.68 | 1.0x |
| **TCN-EMA** ⭐ | **TCN + EMA** | **0.312** | **2.55** | **3.66** | **1.4x** |
| Hierarchical | Multi-task TCN | 0.213 | 2.84 | 3.96 | 1.4x |

**Winner**: TCN + EMA achieves the best predictive performance with 40% faster training.

---

## 🎯 Key Contributions

### 1. Data Infrastructure
- **Challenge**: Binance does not provide historical Level 2 order book data
- **Solution**: Reconstructed LOB data via Tardis.dev API
- **Output**: 25M+ samples across 30 symbols, 10 days (Aug 1-10, 2025)
- **Dataset**: Available on [HuggingFace](https://huggingface.co/datasets/yinelon/crypto_lob_10m)

### 2. Model Architecture Improvements

#### EMA Smoothing (+227% correlation improvement)
```
EMA[t] = alpha × value[t] + (1 - alpha) × EMA[t-1]
```
- Reduces target noise by 65% while preserving trends
- Alpha = 0.2 empirically optimal

#### Temporal Convolutional Networks (TCN)
- **Problem**: LSTM is sequential (slow) and suffers from gradient vanishing
- **Solution**: TCN with dilated causal convolutions
- **Benefits**: 
  - Fully parallelizable (40% faster training)
  - Better gradient flow via residual connections
  - Explicit receptive field control (61 time steps)

#### Hierarchical Multi-Task Learning
- Simultaneously predicts 10s and 60s returns
- Cross-attention mechanism for information flow
- Validates 10s return as optimal prediction horizon

### 3. Engineering Optimizations
- Multi-GPU parallel training (4 GPUs, 8 processes)
- HuberLoss for outlier robustness
- Checkpoint recovery for interrupted training
- Comprehensive evaluation and visualization

---

## 📈 Datasets & Model Weights

All datasets and pre-trained models are publicly available:

### Datasets
- [crypto_lob_10m](https://huggingface.co/datasets/yinelon/crypto_lob_10m) - 10 days, 30 symbols (used in this project)
- [crypto_lob_3m](https://huggingface.co/datasets/yinelon/crypto_lob_3m) - 3 months, 50 symbols (full dataset)

### Pre-trained Models
- [deeplob_crypto_baseline](https://huggingface.co/yinelon/deeplob_crypto_baseline) - CNN-LSTM baseline
- [deeplob_crypto_ema](https://huggingface.co/yinelon/deeplob_crypto_ema) - LSTM with EMA smoothing
- [deeplob_crypto_tcn](https://huggingface.co/yinelon/deeplob_crypto_tcn) - TCN with EMA (best model)
- [hierarchical_modeling](https://huggingface.co/yinelon/hierarchical_modeling) - Multi-task model

---

## 🔬 Technical Highlights

### Data Processing
- 10-depth LOB features (40 dimensions)
- 1-second downsampling from millisecond data
- Log returns + z-score normalization
- Train/Val/Test: 60%/20%/20% (chronological split)

### Model Architecture
```
Input (100 timesteps, 40 features)
    ↓
Conv2D (Temporal patterns)
    ↓
Conv2D (Price level relationships)
    ↓
Inception Module (Multi-scale features)
    ↓
TCN / LSTM (Temporal dependencies)
    ↓
FC Layers (Regression)
    ↓
Output (10s return prediction)
```

### Training Configuration
- Batch size: 2048 (optimized for RTX 5090)
- Sequence length: 100 time steps
- Optimizer: Adam (lr=0.001)
- Loss: HuberLoss (delta=1.0)
- Regularization: Dropout (0.3), Gradient clipping (1.0)
- Early stopping: Patience=5

---

## 📖 References

- **DeepLOB Paper**: [Zhang et al., 2019](https://arxiv.org/abs/1808.03668)
- **Tardis.dev**: Market data provider for historical LOB reconstruction
- **Original DeepLOB Implementation**: [GitHub](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)

---

## 🏢 About This Project

This project was completed as a **take-home assignment for Millennium ML Internship application**. The goal was to demonstrate:

1. **Data Engineering**: Handling real-world data challenges (LOB reconstruction)
2. **Deep Learning Expertise**: Implementing and improving state-of-the-art models
3. **Research Skills**: Systematic experimentation and ablation studies
4. **Engineering Best Practices**: Clean code, documentation, reproducibility

The project showcases end-to-end ML pipeline development, from data collection to model deployment, with a focus on practical financial applications.

---

## 👤 Author

**Shuo Yin**  
- 🌐 Website: [https://yinshuo-thu.github.io/](https://yinshuo-thu.github.io/)
- 📧 Email: yins25@mails.tsinghua.edu.cn
- 🐙 GitHub: [@yinshuo-thu](https://github.com/yinshuo-thu)

---

## 📄 License

This project is open source and available under the MIT License for research and educational purposes.

---

## 🙏 Acknowledgments

- **Millennium Management**: For the opportunity to work on this interesting problem
- **DeepLOB Authors**: For the original architecture and insights
- **Tardis.dev**: For providing historical market data API
- **HuggingFace**: For hosting datasets and model weights

---

**Star ⭐ this repo if you find it useful!**

