# DeepLOB Crypto LOB Training Project

Deep learning models for cryptocurrency Limit Order Book (LOB) data prediction using DeepLOB architecture. This project builds high-frequency LOB prediction models for the top 50 most liquid crypto assets on Binance, implementing and improving upon the DeepLOB baseline with various architectural enhancements.

## 🎯 Project Overview

This project explores multiple deep learning architectures for predicting short-term returns (10s and 60s) from limit order book data:

- **Baseline DeepLOB (LSTM)**: Standard CNN-Inception-LSTM architecture
- **DeepLOB with EMA Smoothing**: Label smoothing via Exponential Moving Average (alpha=0.2)
- **DeepLOB-TCN**: LSTM replaced with Temporal Convolutional Networks for better parallelization
- **Hierarchical Modeling**: Multi-task learning with cross-attention mechanism for multi-scale predictions

**Key Results**: DeepLOB-TCN with EMA smoothing achieved the best performance, with mean correlation of 0.31 and R² score of 0.05 across 30 crypto symbols.

## 📁 Project Structure

```
deeplob_crypto_code/
├── 1_crawl_top50.py              # Top 50 crypto symbols crawler
├── 2_collect_lob_fast.py         # Fast LOB data collection script
├── 2_config.json                 # Configuration for data collection
├── 3_deeplob_baseline.py         # DeepLOB baseline training (LSTM)
├── 4_deeplob_tcn.py              # DeepLOB-TCN training (TCN replaces LSTM)
├── 6_baseline_ema.py             # DeepLOB with EMA smoothing (LSTM)
├── 7_baseline_tcu_ema.py         # DeepLOB-TCN with EMA smoothing
├── 8_baseline_hierarchical_ema.py # DeepLOB with hierarchical EMA
├── candidates.txt                # Top liquidity crypto symbols list
├── images/                       # Performance visualizations and charts
│   ├── baseline_performance.png
│   ├── ema_performance.png
│   ├── tcu_ema_performance.png
│   ├── hierarchical_performance.png
│   └── ...
├── data_250801_250810/           # 10-day training dataset (Aug 1-10, 2025)
│   ├── metadata.json
│   ├── README.md
│   ├── BTCUSDT_20250801_20250810.npy
│   ├── ETHUSDT_20250801_20250810.npy
│   └── ... (30 symbols total)
├── 5_models/                     # TCN model outputs
├── 6_models/                     # EMA model outputs
├── 7_models/                     # TCN-EMA model outputs
├── 8_models/                     # Hierarchical EMA model outputs
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Crawl Top 50 Crypto Symbols

```bash
python3 1_crawl_top50.py
```

This script:
- Fetches all USDT perpetual contracts from Binance Futures
- Calculates average daily volume over the past 92 days to avoid volatility from new coins
- Selects top 50 most liquid symbols based on trading volume
- Saves results to `candidates.txt`

**Top 5 Most Liquid Symbols:**

| Rank | Ticker   | 30 Day Avg Daily Vol (USDT) |
|------|----------|----------------------------|
| 1    | ETHUSDT  | 18,662,928,909             |
| 2    | BTCUSDT  | 15,460,089,107             |
| 3    | SOLUSDT  | 5,465,214,265              |
| 4    | DOGEUSDT | 1,753,463,229              |
| 5    | XRPUSDT  | 1,741,977,252              |

### 2. Collect LOB Data

```bash
python3 2_collect_lob_fast.py
```

**Data Challenge**: Binance's official API does not provide **historical** Level 2 order book data. The public [Binance Market Data](https://data.binance.vision/) only offers real-time data and aggregated book depth (unsuitable for LOB analysis). Therefore, we use **Tardis.dev API** to reconstruct historical L2 data.

This script:
- Downloads LOB data from Tardis.dev API for symbols in `candidates.txt`
- Uses `book_snapshot_25` data type and downsamples to 1-second frequency
- Extracts 10 levels of bid/ask prices and volumes (matching DeepLOB original configuration)
- Processes and converts to NumPy format with vectorization for speed
- Calculates 10-second and 60-second forward returns based on mid-price
- Outputs `.npy` files with shape `(N, 43)`:
  - Column 0: Timestamp
  - Columns 1-40: LOB features (10 levels ask/bid prices & volumes)
  - Column 41: 10-second forward return (log return in BPS)
  - Column 42: 60-second forward return (log return in BPS)

**Memory Optimization**: 
- Original millisecond-level data is downsampled to 1s (10x compression)
- BTCUSDT 1-month L2 10-depth data: ~420MB in `.npy` format
- `.npy` format chosen over Parquet to enable NumPy Memory Mapping and avoid OOM issues

**Performance**: 
- Vectorization + ProcessPoolExecutor for parallel processing
- 50 tickers × 3 months of L2 LOB data collected in <20 hours

Configuration is in `2_config.json`.

### 3. Baseline Training (LSTM)

```bash
python3 3_deeplob_baseline.py
```

This script implements the standard DeepLOB architecture to predict 10-second returns from LOB data:

**Architecture**: CNN → Inception → LSTM → FC
- **1st Conv Layer**: Captures temporal patterns across time steps
- **2nd Conv Layer**: Captures relationships between price levels
- **Inception Module**: Multi-scale feature extraction
- **2-Layer Bidirectional LSTM**: Captures temporal dependencies (64 hidden units)
- **Dropout (0.2)**: Prevents overfitting
- **Gradient Clipping (max_norm=1.0)**: Stabilizes training

**Training Configuration**:
- Independent modeling: Each symbol trained separately
- 30 symbols from top 50 (most complete data)
- Time range: August 1-10, 2025 (10 days, ~25M samples)
- Train/Val/Test split: 60%/20%/20% (chronological)
- Multi-GPU parallel training (4 GPUs, 8 processes)

**Key Improvements over Standard DeepLOB**:
- LeakyReLU activation (avoids dead ReLU)
- Log returns + standardization for target variables
- HuberLoss instead of MSELoss (robust to outliers)
- Checkpoint recovery for interrupted training

**Baseline Results**:

| Metric   | Correlation | MAE (BPS) | RMSE (BPS) | R² Score |
|----------|-------------|-----------|------------|----------|
| Mean     | 0.0839      | 3.3207    | 4.8570     | 0.0063   |
| Median   | 0.0916      | 2.8191    | 4.0812     | 0.0031   |
| Std Dev  | 0.0706      | 1.8298    | 2.7183     | 0.0159   |

**Top 5 Performers**:

| Rank | Symbol    | Correlation | MAE  | RMSE | R²     |
|------|-----------|-------------|------|------|--------|
| 1    | BTCUSDT   | 0.2686      | 0.69 | 1.19 | 0.0701 |
| 2    | DOTUSDT   | 0.1676      | 2.45 | 3.55 | 0.0188 |
| 3    | FILUSDT   | 0.1643      | 2.54 | 3.73 | 0.0073 |
| 4    | BNBUSDT   | 0.1602      | 1.38 | 2.03 | 0.0226 |
| 5    | TRUMPUSDT | 0.1482      | 2.16 | 3.07 | 0.0139 |

### 4. TCN Architecture Training

```bash
python3 4_deeplob_tcn.py
```

This script:
- Replaces LSTM with TCN (Temporal Convolutional Network)
- Uses causal convolutions with dilated convolutions (dilation rates: 1, 2, 4, 8)
- Residual connections for gradient flow
- Fully parallelizable (faster than LSTM)
- Architecture: CNN → Inception → TCN → FC
- Outputs saved to `5_models/` directory
- Supports multi-GPU parallel training (4 GPUs, 2 processes per GPU)

**Key Features:**
- TCN uses exponential dilation rates for larger receptive field
- Batch Normalization and LeakyReLU activation
- Same training configuration as baseline for fair comparison

### 5. EMA Smoothing Training (LSTM)

```bash
python3 6_baseline_ema.py
```

**Motivation**: Raw returns are noisy, making training unstable. EMA smoothing reduces noise while preserving trends.

This script:
- Applies EMA (Exponential Moving Average) smoothing to target variables
- Formula: `EMA[t] = alpha × value[t] + (1 - alpha) × EMA[t-1]`
- EMA alpha parameter: 0.2 (20% new value, 80% historical value)
- Reduces target variable standard deviation by ~65%
- Uses same LSTM architecture as baseline
- Outputs saved to `6_models/` directory

**EMA Results** (Significant Improvement):

| Metric  | Correlation | MAE (BPS) | RMSE (BPS) | R² Score |
|---------|-------------|-----------|------------|----------|
| Mean    | **0.2744**  | **2.5531**| **3.683**  | **0.0514**|
| Median  | **0.2904**  | 2.2023    | 3.0943     | 0.0538   |
| Std Dev | 0.0974      | 1.3151    | 1.9779     | 0.0446   |

**Improvement vs Baseline**:
- Correlation: +227% (0.0839 → 0.2744)
- MAE: -23% (3.32 → 2.55 BPS)
- R²: +716% (0.0063 → 0.0514)

### 6. TCN with EMA Smoothing Training ⭐ **Best Performance**

```bash
python3 7_baseline_tcu_ema.py
```

**Motivation**: LSTM suffers from sequential computation (slow training) and gradient vanishing (long-term forgetting). TCN offers fully parallelizable architecture with better gradient flow.

This script combines the best of both worlds: TCN architecture + EMA smoothing

**Architecture**: CNN → Inception → TCN → FC

**TCN Advantages**:
- ✅ Fully parallelizable (30-50% faster training than LSTM)
- ✅ Residual connections ensure stable gradient flow
- ✅ Dilated convolutions expand receptive field exponentially
- ✅ Causal convolutions preserve temporal causality
- ✅ Lower memory usage (no hidden states)

**TCN Configuration**:
- 4 TCN layers, 64 channels each
- Dilation rates: 1, 2, 4, 8 (exponential growth)
- Kernel size: 3
- Receptive field: 61 time steps
- Batch Normalization + LeakyReLU

**TCN-EMA Results** (Best Performance):

| Metric  | Correlation | MAE (BPS) | RMSE (BPS) | R² Score |
|---------|-------------|-----------|------------|----------|
| Mean    | **0.3119**  | 2.554     | 3.6627     | **0.0516**|
| Median  | **0.3233**  | **2.1781**| **3.0274** | **0.083** |
| Std Dev | 0.0953      | 1.2907    | 1.8582     | 0.1322   |

**Improvement vs Baseline**:
- Correlation: +272% (0.0839 → 0.3119)
- MAE: -23% (3.32 → 2.55 BPS)
- RMSE: -25% (4.86 → 3.66 BPS)
- R²: +719% (0.0063 → 0.0516)

**Improvement vs LSTM-EMA**:
- Correlation: +14% (0.2744 → 0.3119)
- Training speed: 30-50% faster

### 7. Hierarchical Multi-Task Learning

```bash
python3 8_baseline_hierarchical_ema.py
```

**Motivation**: Multi-scale prediction (10s and 60s returns) can capture both short-term fluctuations and longer-term trends. The hypothesis is that 60s return can be viewed as cumulative 10s returns, allowing information sharing between tasks.

This script implements a hierarchical architecture with cross-attention mechanism:

**Architecture Components**:
1. **Short-term TCN Branch** (4 layers, receptive field: 15 steps)
   - Specialized for 10-second return prediction
   - Captures short-term price fluctuations

2. **Long-term TCN Branch** (5 layers, receptive field: 31 steps)
   - Specialized for 60-second return prediction
   - Captures longer-term trends

3. **Cross-Attention Mechanism**
   - Long-term features (Query) attend to short-term features (Key, Value)
   - Model learns which short-term patterns are important for long-term prediction
   - Selective attention improves feature utilization

4. **Gated Fusion**
   - Sigmoid gate controls how short-term features flow into long-term prediction
   - Adaptive fusion prevents information overload
   - Dynamic weighting based on input context

5. **Residual Connection**
   - Short-term predictions directly contribute to long-term predictions
   - Models cumulative relationship: `return_60s ≈ Σ(return_10s)`
   - Alleviates gradient vanishing

**Training Strategy**:
- Multi-task loss: `Loss = 0.5 × Loss_10s + 0.5 × Loss_60s`
- Shared feature extraction for both tasks
- Regularization effect from multi-task learning

**Hierarchical Results**:

| Metric  | Correlation | MAE (BPS) | RMSE (BPS) | R² Score |
|---------|-------------|-----------|------------|----------|
| Mean    | 0.2134      | 2.8447    | 3.9629     | -0.114   |
| Median  | 0.2508      | 2.3727    | 3.3146     | -0.0005  |
| Std Dev | 0.1172      | 1.5704    | 2.1692     | 0.3845   |

**Key Insights**:
- 10s return prediction performance decreases vs TCN-EMA (resource allocation between tasks)
- Provides **multi-scale predictions** for strategy flexibility
- Cross-validation between 10s and 60s predictions enhances robustness
- Validates that **10s return is optimal** for high-frequency LOB prediction
- Demonstrates interesting architectural exploration with practical strategy potential

## 📊 Model Comparison Summary

| Model | Architecture | Correlation | MAE (BPS) | RMSE (BPS) | R² Score | Training Speed |
|-------|-------------|-------------|-----------|------------|----------|----------------|
| Baseline | LSTM | 0.0839 | 3.32 | 4.86 | 0.0063 | Baseline |
| EMA-LSTM | LSTM + EMA | 0.2744 | 2.55 | 3.68 | 0.0514 | Baseline |
| **TCN-EMA** ⭐ | **TCN + EMA** | **0.3119** | **2.55** | **3.66** | **0.0516** | **+40%** |
| Hierarchical | TCN + Multi-task | 0.2134 | 2.84 | 3.96 | -0.114 | +40% |

**Conclusion**: DeepLOB-TCN with EMA achieves the best performance with 0.31 correlation and provides 40% faster training than LSTM-based models.

## 📊 Dataset

- **Source**: Binance Futures perpetual contracts
- **Time Range**: August 1-10, 2025 (10 days)
- **Symbols**: 30 cryptocurrencies
- **Total Samples**: 25.3 million records
- **Format**: NumPy `.npy` files, shape `(N, 43)`
  - Column 0: Timestamp
  - Columns 1-40: LOB features (10 levels ask/bid prices & volumes)
  - Column 41: 10-second forward return
  - Column 42: 60-second forward return

**Dataset Download**: [HuggingFace - crypto_lob_3m](https://huggingface.co/datasets/yinelon/crypto_lob_3m)

## 🏗️ Model Architectures

### DeepLOB Baseline (LSTM)
- **Input**: `(batch_size, 1, sequence_length, 40)`
- **Architecture**: CNN → Inception → LSTM → FC
- **Output**: Regression (log returns in BPS)
- **LSTM**: 1 layer, 64 hidden units

### DeepLOB-TCN
- **Architecture**: CNN → Inception → TCN → FC
- **TCN Configuration**:
  - 4 TCN layers, 64 channels each
  - Dilation rates: 1, 2, 4, 8 (exponential growth)
  - Kernel size: 3
  - Receptive field: 61 time steps
- **TCN Components**:
  - Causal Convolution: Only uses past information
  - Dilated Convolution: Expands receptive field exponentially
  - Residual Connection: Alleviates gradient vanishing
  - Batch Normalization: Accelerates training
  - LeakyReLU: Prevents dead ReLU
- **Advantages over LSTM**:
  - Fully parallelizable (faster training)
  - Better gradient flow
  - Larger effective receptive field

### DeepLOB with EMA Smoothing (LSTM)
- Same architecture as baseline
- EMA smoothing applied to target variables (alpha=0.2)
- Reduces noise for more stable training

### DeepLOB-TCN with EMA Smoothing
- Combines TCN architecture with EMA smoothing
- Architecture: CNN → Inception → TCN → FC
- EMA smoothing on target variables

### DeepLOB with Hierarchical EMA
- Multiple EMA layers for different time scales
- More sophisticated smoothing approach

## 📈 Training Configuration

- **Batch Size**: 2048
- **Sequence Length**: 100
- **Epochs**: 20 (with early stopping, patience=5)
- **Learning Rate**: 0.001 (with ReduceLROnPlateau scheduler)
- **Optimizer**: Adam
- **Loss**: HuberLoss (delta=1.0)
- **Regularization**: Dropout (0.3), Gradient Clipping (max_norm=1.0)

## 🔧 Key Features

- **Log Return Conversion**: Simple returns → log returns for better statistical properties
- **Target Standardization**: Z-score normalization for stable training
- **EMA Smoothing**: Optional exponential moving average on target variables
- **Multi-GPU Support**: Parallel training across multiple GPUs
  - Supports 1-4 GPUs with intelligent allocation
  - 4 GPUs: 8 parallel processes (2 per GPU)
  - Automatic GPU assignment and load balancing
- **Early Stopping**: Automatic training termination based on validation loss
- **Checkpoint Recovery**: Resume training from completed symbols
- **TCN Optimization**: 
  - Fully parallelizable architecture
  - 30-50% faster training than LSTM
  - Better gradient flow with residual connections
  - Exponential dilation rates for larger receptive field

## 📝 Output Files

Training generates:
- Model weights: `*_best_model.pth`
- Predictions: `*_predictions.npz`
- Training history: `*_history.pkl`
- Performance metrics: `training_summary_*.csv`
- Visualization charts: `fig1_core_performance.png`, `fig2_summary_statistics.png`, etc.

## 📚 Model Weights

Pre-trained model weights are available on HuggingFace:

- **Baseline (LSTM)**: [deeplob_crypto_baseline](https://huggingface.co/yinelon/deeplob_crypto_baseline)
- **EMA Smoothing (LSTM)**: [deeplob_crypto_ema](https://huggingface.co/yinelon/deeplob_crypto_ema)
- **TCN with EMA**: [deeplob_crypto_tcn](https://huggingface.co/yinelon/deeplob_crypto_tcn)

## ⚡ Performance Optimization

### Multi-GPU Training

All training scripts support intelligent multi-GPU allocation:

```python
# Automatic GPU allocation
if n_gpus >= 4:
    max_workers = 8  # 4 GPUs × 2 processes/GPU
elif n_gpus >= 2:
    max_workers = n_gpus * 2
else:
    max_workers = 2  # Single GPU with 2 processes
```

**Performance Gains:**
- 4 GPUs parallel: ~3.5x speedup vs single GPU
- TCN vs LSTM: 30-50% faster training
- Overall: 4-5x total speedup

### TCN vs LSTM Comparison

| Feature | LSTM | TCN |
|---------|------|-----|
| Parallelization | ❌ Sequential | ✅ Fully parallel |
| Training Speed | Slower | 30-50% faster |
| Gradient Flow | Can vanish | Residual connections |
| Receptive Field | Global (theoretical) | Exponential growth |
| Memory Usage | Higher (hidden states) | Lower (activations only) |

### Training Configuration

- **Batch Size**: 2048 (optimized for RTX 5090 32GB)
- **Sequence Length**: 100 time steps
- **Epochs**: 20 (with early stopping, patience=5)
- **Learning Rate**: 0.001 (with ReduceLROnPlateau scheduler)
- **Optimizer**: Adam
- **Loss**: HuberLoss (delta=1.0) - robust to outliers
- **Regularization**: 
  - Dropout: 0.3
  - Gradient Clipping: max_norm=1.0
  - Batch Normalization: All layers

## 🔬 Experimental Findings

### What Worked Well:
1. **EMA Smoothing**: Single most impactful improvement (+227% correlation)
   - Reduces noise by 65% without losing trend information
   - Stabilizes training and improves convergence

2. **TCN Architecture**: Better than LSTM for LOB prediction
   - 30-50% faster training due to parallelization
   - Better gradient flow via residual connections
   - Explicit receptive field control via dilation rates

3. **Log Returns + Standardization**: Essential for stable training
   - Log returns have better statistical properties
   - Z-score normalization prevents gradient explosion

4. **HuberLoss**: More robust than MSELoss for financial data
   - Handles outliers gracefully (delta=1.0)
   - Combines benefits of L1 and L2 loss

5. **Independent Modeling**: Better than pooled modeling
   - Each symbol has unique market dynamics
   - Per-symbol training captures individual characteristics

### What Didn't Work:
1. **Pooled Modeling**: Poor performance when concatenating all symbols
   - Different symbols have different scales and dynamics
   - Insufficient data preprocessing for cross-symbol learning

2. **Hierarchical Multi-Task**: Interesting but sacrifices 10s performance
   - Resource allocation trade-off between tasks
   - Useful for strategy design but not optimal for single-objective optimization

### Key Insights:
- **10s return is optimal** for high-frequency LOB prediction (vs 60s)
- **Independent modeling** per symbol is critical for crypto markets
- **TCN + EMA** is the winning combination for LOB prediction

## 📚 References

- [DeepLOB Paper](https://arxiv.org/abs/1808.03668)
- Dataset: 
  - [yinelon/crypto_lob_3m](https://huggingface.co/datasets/yinelon/crypto_lob_3m) (3 months, 50 symbols)
  - [yinelon/crypto_lob_10m](https://huggingface.co/datasets/yinelon/crypto_lob_10m) (10 days, 30 symbols - used in this project)
- Model Weights: 
  - [deeplob_crypto_baseline](https://huggingface.co/yinelon/deeplob_crypto_baseline) (LSTM)
  - [deeplob_crypto_ema](https://huggingface.co/yinelon/deeplob_crypto_ema) (LSTM + EMA)
  - [deeplob_crypto_tcn](https://huggingface.co/yinelon/deeplob_crypto_tcn) (TCN + EMA)
  - [hierarchical_modeling](https://huggingface.co/yinelon/hierarchical_modeling) (Multi-task)

## 🔄 Requirements

### Core Dependencies

```bash
# Core Data Processing
numpy>=1.24.0
pandas>=2.0.0

# Deep Learning Framework
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Machine Learning
scikit-learn>=1.3.0

# Data Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration
pyyaml>=6.0

# HTTP Requests
requests>=2.31.0
urllib3>=2.0.0

# Progress Bars
tqdm>=4.65.0

# Tardis.dev API (for downloading market data)
tardis-dev>=13.35.0

# Hugging Face Hub (for model upload/download)
huggingface_hub>=0.20.0
```

### Installation

```bash
pip install numpy pandas torch torchvision torchaudio scikit-learn matplotlib seaborn pyyaml requests urllib3 tqdm tardis-dev huggingface_hub
```

### Optional Dependencies

For better performance (optional):
- `numba>=0.57.0` - JIT compilation for numerical code
- `cupy-cuda11x>=12.0.0` - GPU-accelerated NumPy (if using CUDA 11.x)

---

## 👤 Author

**Shuo Yin**  
- Website: [https://yinshuo-thu.github.io/](https://yinshuo-thu.github.io/)
- Email: yins25@mails.tsinghua.edu.cn
- GitHub: [@yinshuo-thu](https://github.com/yinshuo-thu)

---

## 📄 License

This project is open source and available for research and educational purposes.
