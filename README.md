# DeepLOB Binance - Cryptocurrency LOB Prediction

 [Shuo Yin](https://yinshuo-thu.github.io/)

yins25@mails.tsinghua.edu.cn

This repository contains a **Millennium ML Internship Take-Home Project** focused on building high-frequency limit order book (LOB) prediction models for cryptocurrency markets using deep learning.

---

## 📁 Repository Structure

```
deeplob-binance-github/
├── README.md                      # This file
├── DeepLOB-Binance.pdf            # Project report (Chinese)
├── DeepLOB-Binance-EN.pdf         # Project report (English)
└── deeplob_crypto_code/           # Source code and detailed documentation
    ├── README.md                  # Technical documentation
    ├── 1_crawl_top50.py           # Crawl top 50 liquid symbols
    ├── 2_collect_lob_fast.py      # Collect LOB data
    ├── 3_deeplob_baseline.py      # DeepLOB baseline (LSTM)
    ├── 4_deeplob_tcn.py           # DeepLOB with TCN
    ├── 6_baseline_ema.py          # DeepLOB with EMA smoothing
    ├── 7_baseline_tcu_ema.py      # TCN + EMA (best model)
    ├── 8_baseline_hierarchical_ema.py  # Hierarchical multi-task
    └── images/                    # Performance visualizations
```

---

## 📊 Project Summary

This project implements and improves upon the **DeepLOB architecture** for predicting 10-second returns from cryptocurrency limit order book data on Binance. 

**Key Models Explored:**
- DeepLOB Baseline 
- DeepLOB with EMA Smoothing
- DeepLOB-TCN 
- Hierarchical Multi-Task Learning

For detailed methodology, results, and analysis, please refer to:
- 📄 **Chinese Report**: [`DeepLOB-Binance.pdf`](DeepLOB-Binance.pdf)
- 📄 **English Report**: [`DeepLOB-Binance-EN.pdf`](DeepLOB-Binance-EN.pdf)
- 📖 **Technical Documentation**: [`deeplob_crypto_code/README.md`](deeplob_crypto_code/README.md)

---

## 📈 Datasets & Model Weights

### Datasets (HuggingFace)
- **10-day Dataset** (used in this project): [yinelon/crypto_lob_10m](https://huggingface.co/datasets/yinelon/crypto_lob_10m)
  - 30 symbols, Aug 1-10, 2025, ~25M samples
- **3-month Dataset** (full): [yinelon/crypto_lob_3m](https://huggingface.co/datasets/yinelon/crypto_lob_3m)
  - 50 symbols, 3 months

### Pre-trained Model Weights (HuggingFace)
- **Baseline**: [yinelon/deeplob_crypto_baseline](https://huggingface.co/yinelon/deeplob_crypto_baseline)
- **EMA Smoothing**: [yinelon/deeplob_crypto_ema](https://huggingface.co/yinelon/deeplob_crypto_ema)
- **TCN + EMA**: [yinelon/deeplob_crypto_tcn](https://huggingface.co/yinelon/deeplob_crypto_tcn)
- **Hierarchical Model**: [yinelon/hierarchical_modeling](https://huggingface.co/yinelon/hierarchical_modeling)

---

## 📚 References

- **DeepLOB Paper**: [Zhang et al., 2019](https://arxiv.org/abs/1808.03668)
- **Original Implementation**: [GitHub](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)

---

## 👤 Contact

**Shuo Yin**  
- 🌐 Website: [https://yinshuo-thu.github.io/](https://yinshuo-thu.github.io/)
- 📧 Email: yins25@mails.tsinghua.edu.cn
- 🐙 GitHub: [@yinshuo-thu](https://github.com/yinshuo-thu)

---

## 📄 License

This project is open source and available under the MIT License for research and educational purposes.
