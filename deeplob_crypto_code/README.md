# DeepLOB Crypto LOB Training Project

Deep learning models for cryptocurrency Limit Order Book (LOB) data prediction using DeepLOB architecture.

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
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Crawl Top 50 Crypto Symbols

```bash
python3 1_crawl_top50.py
```

This script:
- Fetches all USDT perpetual contracts from Binance Futures
- Calculates average daily volume over the past 92 days
- Selects top 50 most liquid symbols based on trading volume
- Saves results to `candidates.txt`

### 2. Collect LOB Data

```bash
python3 2_collect_lob_fast.py
```

**Data Challenge**: Binance's official API does not provide **historical** Level 2 order book data. Therefore, we use **Tardis.dev API** to reconstruct historical L2 data.

This script:
- Downloads LOB data from Tardis.dev API for symbols in `candidates.txt`
- Uses `book_snapshot_25` data type and downsamples to 1-second frequency
- Extracts 10 levels of bid/ask prices and volumes
- Processes and converts to NumPy format with vectorization for speed
- Calculates 10-second and 60-second forward returns based on mid-price
- Outputs `.npy` files with shape `(N, 43)`:
  - Column 0: Timestamp
  - Columns 1-40: LOB features (10 levels ask/bid prices & volumes)
  - Column 41: 10-second forward return (log return in BPS)
  - Column 42: 60-second forward return (log return in BPS)

Configuration is in `2_config.json`.

### 3. Baseline Training (LSTM)

```bash
python3 3_deeplob_baseline.py
```

Trains DeepLOB model with LSTM architecture:
- **Architecture**: CNN → Inception → LSTM → FC
- Independent modeling: Each symbol trained separately
- Multi-GPU parallel training support

### 4. TCN Architecture Training

```bash
python3 4_deeplob_tcn.py
```

Trains DeepLOB with Temporal Convolutional Network:
- Replaces LSTM with TCN
- Uses causal and dilated convolutions
- Fully parallelizable architecture
- Outputs saved to `5_models/` directory

### 5. EMA Smoothing Training (LSTM)

```bash
python3 6_baseline_ema.py
```

Applies EMA (Exponential Moving Average) smoothing to target variables:
- Formula: `EMA[t] = alpha × value[t] + (1 - alpha) × EMA[t-1]`
- EMA alpha parameter: 0.2
- Uses same LSTM architecture as baseline
- Outputs saved to `6_models/` directory

### 6. TCN with EMA Smoothing Training

```bash
python3 7_baseline_tcu_ema.py
```

Combines TCN architecture with EMA smoothing:
- **Architecture**: CNN → Inception → TCN → FC
- EMA smoothing on target variables
- Outputs saved to `7_models/` directory

### 7. Hierarchical Multi-Task Learning

```bash
python3 8_baseline_hierarchical_ema.py
```

Implements hierarchical architecture with cross-attention mechanism:
- Short-term TCN branch for 10-second return prediction
- Long-term TCN branch for 60-second return prediction
- Cross-attention mechanism between branches
- Multi-task loss for joint training
- Outputs saved to `8_models/` directory

## 📊 Dataset

### Dataset Information
- **Source**: Binance Futures perpetual contracts
- **Time Range**: August 1-10, 2025 (10 days)
- **Symbols**: 30 cryptocurrencies
- **Total Samples**: 25.3 million records
- **Format**: NumPy `.npy` files, shape `(N, 43)`
  - Column 0: Timestamp
  - Columns 1-40: LOB features (10 levels ask/bid prices & volumes)
  - Column 41: 10-second forward return
  - Column 42: 60-second forward return

### Dataset Downloads (HuggingFace)
- **10-day Dataset** (used in this project): [yinelon/crypto_lob_10m](https://huggingface.co/datasets/yinelon/crypto_lob_10m)
  - 30 symbols, Aug 1-10, 2025, ~25M samples
- **3-month Dataset** (full): [yinelon/crypto_lob_3m](https://huggingface.co/datasets/yinelon/crypto_lob_3m)
  - 50 symbols, 3 months

## 🏋️ Pre-trained Model Weights (HuggingFace)

- **Baseline (LSTM)**: [yinelon/deeplob_crypto_baseline](https://huggingface.co/yinelon/deeplob_crypto_baseline)
- **EMA Smoothing (LSTM)**: [yinelon/deeplob_crypto_ema](https://huggingface.co/yinelon/deeplob_crypto_ema)
- **TCN + EMA**: [yinelon/deeplob_crypto_tcn](https://huggingface.co/yinelon/deeplob_crypto_tcn)
- **Hierarchical Model**: [yinelon/hierarchical_modeling](https://huggingface.co/yinelon/hierarchical_modeling)

## 🏗️ Model Architectures

### DeepLOB Baseline (LSTM)
- **Architecture**: CNN → Inception → LSTM → FC
- **Input**: `(batch_size, 1, sequence_length, 40)`
- **Output**: Regression (log returns in BPS)

### DeepLOB-TCN
- **Architecture**: CNN → Inception → TCN → FC
- **TCN Configuration**:
  - 4 TCN layers, 64 channels each
  - Dilation rates: 1, 2, 4, 8 (exponential growth)
  - Kernel size: 3
  - Receptive field: 61 time steps

### DeepLOB with EMA Smoothing
- Same architecture as baseline
- EMA smoothing applied to target variables (alpha=0.2)

### DeepLOB-TCN with EMA Smoothing
- Combines TCN architecture with EMA smoothing
- **Architecture**: CNN → Inception → TCN → FC

### Hierarchical Multi-Task Model
- Dual TCN branches for short-term and long-term predictions
- Cross-attention mechanism for information flow
- Gated fusion and residual connections

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
  - Automatic GPU assignment and load balancing
- **Early Stopping**: Automatic training termination based on validation loss
- **Checkpoint Recovery**: Resume training from completed symbols

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

## 📚 References

- **DeepLOB Paper**: [Zhang et al., 2019](https://arxiv.org/abs/1808.03668)
- **Original Implementation**: [GitHub](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)

---

## 👤 Author

**Shuo Yin**  
- Website: [https://yinshuo-thu.github.io/](https://yinshuo-thu.github.io/)
- Email: yins25@mails.tsinghua.edu.cn
- GitHub: [@yinshuo-thu](https://github.com/yinshuo-thu)

---

## 📄 License

This project is open source and available for research and educational purposes.
