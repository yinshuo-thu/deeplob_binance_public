#!/usr/bin/env python3
"""
DeepLOB-TCN Hierarchical Modeling - 多时间尺度预测

主要改进:
- Hierarchical Modeling: 同时预测return_10s和return_60s
- 层次化架构: 短期预测 -> 长期预测
- 共享特征提取: CNN特征提取层共享
- 多任务学习: 联合优化两个目标
- EMA平滑: 对两个目标变量都进行EMA平滑

架构设计:
1. 共享CNN特征提取层 (DeepLOB CNN部分)
2. 短期TCN分支: 预测return_10s (10秒收益)
3. 长期TCN分支: 预测return_60s (60秒收益)
   - 可以基于短期预测和共享特征
   - 使用更大的感受野处理长期模式
4. 多任务损失: 加权组合两个目标的损失

Hierarchical思想:
- return_60s可以看作是多个return_10s的累积
- 短期预测有助于长期预测
- 共享底层特征，分层预测不同时间尺度
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
import time
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
warnings.filterwarnings('ignore')

# 设置multiprocessing启动方法为'spawn'（PyTorch CUDA要求）
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# 1. Dataset with EMA Smoothing (Multi-Target)
# ============================================================================

def apply_ema_smoothing(values, alpha=0.2):
    """
    对目标变量应用指数移动平均(EMA)平滑
    
    Args:
        values: 原始目标变量数组
        alpha: 平滑因子 (0 < alpha <= 1)
              - alpha越小，平滑程度越高（更平滑）
              - alpha越大，平滑程度越低（更接近原始值）
              - 默认0.2表示20%新值，80%历史值
    
    Returns:
        smoothed: 平滑后的数组
    """
    if len(values) == 0:
        return values
    
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]  # 第一个值保持不变
    
    for i in range(1, len(values)):
        # EMA公式: EMA[t] = alpha * value[t] + (1 - alpha) * EMA[t-1]
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


class HierarchicalLOBDataset(Dataset):
    """Hierarchical LOB数据集 - 同时支持return_10s和return_60s"""
    
    def __init__(self, file_path, start_ratio=0.0, end_ratio=1.0, 
                 sequence_length=100, scaler=None, 
                 feature_dim=40, target_col_10s=40, target_col_60s=41,
                 fit_scaler=False, ema_alpha=0.2):
        """
        Args:
            target_col_10s: return_10s的列索引 (默认41)
            target_col_60s: return_60s的列索引 (默认42)
            ema_alpha: EMA平滑因子，默认0.2
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.target_col_10s = target_col_10s
        self.target_col_60s = target_col_60s
        self.ema_alpha = ema_alpha
        
        data = np.load(file_path, mmap_mode='r')
        n = len(data)
        start_idx = int(n * start_ratio)
        end_idx = int(n * end_ratio)
        segment = data[start_idx:end_idx]
        
        # ✅ 对两个目标变量都进行EMA平滑
        # 处理return_10s
        raw_targets_10s = segment[:, target_col_10s].copy()
        valid_mask_10s = np.isfinite(raw_targets_10s)
        if valid_mask_10s.sum() > 0:
            smoothed_targets_10s = np.zeros_like(raw_targets_10s)
            valid_indices_10s = np.where(valid_mask_10s)[0]
            valid_values_10s = raw_targets_10s[valid_indices_10s]
            smoothed_valid_10s = apply_ema_smoothing(valid_values_10s, alpha=ema_alpha)
            smoothed_targets_10s[valid_indices_10s] = smoothed_valid_10s
            smoothed_targets_10s[~valid_mask_10s] = raw_targets_10s[~valid_mask_10s]
        else:
            smoothed_targets_10s = raw_targets_10s
        
        # 处理return_60s
        raw_targets_60s = segment[:, target_col_60s].copy()
        valid_mask_60s = np.isfinite(raw_targets_60s)
        if valid_mask_60s.sum() > 0:
            smoothed_targets_60s = np.zeros_like(raw_targets_60s)
            valid_indices_60s = np.where(valid_mask_60s)[0]
            valid_values_60s = raw_targets_60s[valid_indices_60s]
            smoothed_valid_60s = apply_ema_smoothing(valid_values_60s, alpha=ema_alpha)
            smoothed_targets_60s[valid_indices_60s] = smoothed_valid_60s
            smoothed_targets_60s[~valid_mask_60s] = raw_targets_60s[~valid_mask_60s]
        else:
            smoothed_targets_60s = raw_targets_60s
        
        # 将平滑后的目标变量替换原始值
        segment_smoothed = segment.copy()
        segment_smoothed[:, target_col_10s] = smoothed_targets_10s
        segment_smoothed[:, target_col_60s] = smoothed_targets_60s
        
        if fit_scaler:
            features = segment_smoothed[:, :feature_dim]
            targets_10s = segment_smoothed[:, target_col_10s]
            targets_60s = segment_smoothed[:, target_col_60s]
            
            valid_mask = (np.isfinite(features).all(axis=1) & 
                         np.isfinite(targets_10s) & 
                         np.isfinite(targets_60s))
            features_clean = features[valid_mask]
            
            if scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(features_clean)
            else:
                self.scaler = scaler
        else:
            self.scaler = scaler
        
        self.data = segment_smoothed
        self.n_samples = len(self.data) - self.sequence_length
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        window = self.data[idx:idx + self.sequence_length, :self.feature_dim].copy()
        target_10s = self.data[idx + self.sequence_length - 1, self.target_col_10s].copy()
        target_60s = self.data[idx + self.sequence_length - 1, self.target_col_60s].copy()
        
        if not np.isfinite(window).all() or not np.isfinite(target_10s) or not np.isfinite(target_60s):
            return (torch.zeros(1, self.sequence_length, self.feature_dim), 
                   torch.zeros(1), torch.zeros(1))
        
        if self.scaler is not None:
            window = self.scaler.transform(window)
        
        # 转换为BPS (Basis Points)
        target_10s = np.log1p(target_10s) * 10000
        target_60s = np.log1p(target_60s) * 10000
        
        if not np.isfinite(target_10s) or not np.isfinite(target_60s):
            return (torch.zeros(1, self.sequence_length, self.feature_dim), 
                   torch.zeros(1), torch.zeros(1))
        
        x = torch.FloatTensor(window).unsqueeze(0)
        y_10s = torch.FloatTensor([target_10s])
        y_60s = torch.FloatTensor([target_60s])
        
        return x, y_10s, y_60s


# ============================================================================
# 2. TCN Model (与原始版本相同)
# ============================================================================

class TCNBlock(nn.Module):
    """
    TCN 残差块 (TCN Residual Block)
    
    核心组件:
    - 因果卷积 (Causal Convolution): 只使用过去的信息
    - 膨胀卷积 (Dilated Convolution): 扩大感受野
    - 残差连接 (Residual Connection): 缓解梯度消失
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        # 因果填充: 确保只使用过去的信息，避免未来信息泄露
        self.padding = (kernel_size - 1) * dilation
        
        # 第一个膨胀卷积层
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,  # 我们手动进行因果填充
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 第二个膨胀卷积层
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=0,  # 我们手动进行因果填充
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        
        # 残差连接: 如果输入输出维度不同，需要1x1卷积调整
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        
        # 权重初始化
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.residual is not None:
            nn.init.kaiming_normal_(self.residual.weight)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            out: (batch, channels, seq_len)
        """
        residual = x
        
        # 第一个卷积 + 因果填充
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第二个卷积 + 因果填充
        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.residual is not None:
            residual = self.residual(residual)
        
        out += residual
        out = self.relu(out)
        
        return out


class TCN(nn.Module):
    """
    时序卷积网络 (Temporal Convolutional Network)
    """
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            input_size: 输入特征维度
            num_channels: 每层的通道数列表，如 [64, 64, 64]
            kernel_size: 卷积核大小
            dropout: Dropout 比率
        """
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # 指数增长的膨胀率: 1, 2, 4, 8, ...
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, 
                        dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            out: (batch, channels, seq_len)
        """
        return self.network(x)


# ============================================================================
# 3. Hierarchical Model Architecture (Optimized)
# ============================================================================

class CrossAttention(nn.Module):
    """
    交叉注意力机制：让长期TCN关注短期特征中的重要部分
    """
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim=64):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.out_proj = nn.Linear(hidden_dim, value_dim)
        
    def forward(self, query, key, value):
        """
        Args:
            query: (batch, seq_len, query_dim) - 长期特征
            key: (batch, seq_len, key_dim) - 短期特征
            value: (batch, seq_len, value_dim) - 短期特征
        Returns:
            attended: (batch, seq_len, value_dim) - 注意力加权的短期特征
        """
        Q = self.query_proj(query)  # (batch, seq_len, hidden_dim)
        K = self.key_proj(key)  # (batch, seq_len, hidden_dim)
        V = self.value_proj(value)  # (batch, seq_len, hidden_dim)
        
        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (batch, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        
        # 应用注意力权重
        attended = torch.bmm(attn_weights, V)  # (batch, seq_len, hidden_dim)
        attended = self.out_proj(attended)  # (batch, seq_len, value_dim)
        
        return attended


class GatedFusion(nn.Module):
    """
    门控融合机制：控制短期特征如何融入长期预测
    """
    def __init__(self, shared_dim, short_dim, output_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(shared_dim + short_dim, output_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(shared_dim + short_dim, output_dim)
        
    def forward(self, shared_feat, short_feat):
        """
        Args:
            shared_feat: (batch, seq_len, shared_dim)
            short_feat: (batch, seq_len, short_dim)
        Returns:
            fused: (batch, seq_len, output_dim)
        """
        combined = torch.cat([shared_feat, short_feat], dim=-1)  # (batch, seq_len, shared_dim + short_dim)
        gate = self.gate(combined)  # (batch, seq_len, output_dim)
        transformed = self.transform(combined)  # (batch, seq_len, output_dim)
        fused = gate * transformed  # 门控融合
        return fused


class DeepLOB_Hierarchical_TCN(nn.Module):
    """
    DeepLOB-TCN Hierarchical Model (Optimized)
    
    架构流程:
    1. 共享CNN特征提取 (DeepLOB CNN部分)
    2. 短期TCN分支: 预测return_10s
    3. 长期TCN分支: 预测return_60s (基于短期预测和共享特征)
    4. 优化组件:
       - 交叉注意力机制: 长期TCN关注短期特征中的重要部分
       - 门控融合机制: 控制短期特征如何融入长期预测
       - 残差连接: 短期预测信息直接传递给长期预测
    """
    def __init__(self, input_channels=1, dropout=0.3, 
                 short_term_channels=[64, 64, 64, 64],
                 long_term_channels=[64, 64, 64, 64, 64],
                 use_attention=True, use_gated_fusion=True, use_residual=True):
        """
        Args:
            short_term_channels: 短期TCN的通道数列表
            long_term_channels: 长期TCN的通道数列表（可以更深）
            use_attention: 是否使用交叉注意力机制
            use_gated_fusion: 是否使用门控融合机制
            use_residual: 是否使用残差连接
        """
        super(DeepLOB_Hierarchical_TCN, self).__init__()
        
        self.use_attention = use_attention
        self.use_gated_fusion = use_gated_fusion
        self.use_residual = use_residual
        
        # ==================== 共享CNN特征提取部分 ====================
        # First convolutional block
        self.conv1a = nn.Conv2d(input_channels, 32, kernel_size=(1, 2), stride=(1, 2))
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(0, 0))
        self.bn1b = nn.BatchNorm2d(32)
        self.conv1c = nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(0, 0))
        self.bn1c = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2a = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.bn2a = nn.BatchNorm2d(32)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(0, 0))
        self.bn2b = nn.BatchNorm2d(32)
        self.conv2c = nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(0, 0))
        self.bn2c = nn.BatchNorm2d(32)
        
        # Third convolutional block
        self.conv3a = nn.Conv2d(32, 32, kernel_size=(1, 10))
        self.bn3a = nn.BatchNorm2d(32)
        
        # Inception module
        self.inception1 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=1)
        self.bn_inc1 = nn.BatchNorm2d(64)
        
        self.inception2a = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=1)
        self.inception2b = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.bn_inc2 = nn.BatchNorm2d(64)
        
        self.inception3a = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=1)
        self.inception3b = nn.Conv2d(64, 64, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.bn_inc3 = nn.BatchNorm2d(64)
        
        self.inception4 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.inception4_conv = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=1)
        self.bn_inc4 = nn.BatchNorm2d(64)
        
        # ==================== 短期TCN分支 (return_10s) ====================
        self.tcn_short = TCN(
            input_size=256,
            num_channels=short_term_channels,
            kernel_size=3,
            dropout=dropout
        )
        
        # 短期预测头
        self.fc_short_1 = nn.Linear(short_term_channels[-1], 64)
        self.bn_short_1 = nn.BatchNorm1d(64)
        self.fc_short_2 = nn.Linear(64, 1)
        self.dropout_short = nn.Dropout(dropout)
        
        # ==================== 长期TCN分支 (return_60s) ====================
        # 交叉注意力机制（可选）
        if use_attention:
            self.cross_attention = CrossAttention(
                query_dim=256,  # 共享特征维度
                key_dim=short_term_channels[-1],  # 短期特征维度
                value_dim=short_term_channels[-1],
                hidden_dim=64
            )
        
        # 门控融合机制（可选）
        if use_gated_fusion:
            self.gated_fusion = GatedFusion(
                shared_dim=256,
                short_dim=short_term_channels[-1],
                output_dim=256
            )
        
        # 长期TCN输入维度
        if use_attention or use_gated_fusion:
            # 如果使用注意力或门控融合，输入维度保持为256
            long_tcn_input_size = 256
        else:
            # 否则，简单拼接
            long_tcn_input_size = 256 + short_term_channels[-1]
        
        self.tcn_long = TCN(
            input_size=long_tcn_input_size,
            num_channels=long_term_channels,
            kernel_size=3,
            dropout=dropout
        )
        
        # 长期预测头
        self.fc_long_1 = nn.Linear(long_term_channels[-1], 64)
        self.bn_long_1 = nn.BatchNorm1d(64)
        self.fc_long_2 = nn.Linear(64, 1)
        self.dropout_long = nn.Dropout(dropout)
        
        # 残差连接：短期预测到长期预测（可选）
        if use_residual:
            self.residual_proj = nn.Linear(1, 1)  # 将短期预测投影到长期预测空间
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, seq_len, feature_dim)
        Returns:
            pred_10s: (batch, 1) - return_10s预测
            pred_60s: (batch, 1) - return_60s预测
        """
        # ==================== 共享CNN特征提取 ====================
        x = F.leaky_relu(self.bn1a(self.conv1a(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn1b(self.conv1b(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn1c(self.conv1c(x)), negative_slope=0.01)
        
        x = F.leaky_relu(self.bn2a(self.conv2a(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2b(self.conv2b(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2c(self.conv2c(x)), negative_slope=0.01)
        
        x = F.leaky_relu(self.bn3a(self.conv3a(x)), negative_slope=0.01)
        
        # Inception module
        branch1 = F.leaky_relu(self.bn_inc1(self.inception1(x)), negative_slope=0.01)
        branch2 = F.leaky_relu(self.inception2a(x), negative_slope=0.01)
        branch2 = F.leaky_relu(self.bn_inc2(self.inception2b(branch2)), negative_slope=0.01)
        branch3 = F.leaky_relu(self.inception3a(x), negative_slope=0.01)
        branch3 = F.leaky_relu(self.bn_inc3(self.inception3b(branch3)), negative_slope=0.01)
        branch4 = self.inception4(x)
        branch4 = F.leaky_relu(self.bn_inc4(self.inception4_conv(branch4)), negative_slope=0.01)
        
        shared_features = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        # Reshape for TCN: (batch, 256, seq_len)
        shared_features = shared_features.squeeze(-1)  # (batch, 256, seq_len)
        
        # ==================== 短期TCN分支 (return_10s) ====================
        short_tcn_out = self.tcn_short(shared_features)  # (batch, 64, seq_len)
        short_features = short_tcn_out[:, :, -1]  # (batch, 64) - 取最后时间步
        
        # 短期预测
        short_pred = F.leaky_relu(self.bn_short_1(self.fc_short_1(short_features)), negative_slope=0.01)
        short_pred = self.dropout_short(short_pred)
        pred_10s = self.fc_short_2(short_pred)
        
        # ==================== 长期TCN分支 (return_60s) ====================
        # 准备长期TCN输入
        seq_len = shared_features.size(2)
        
        # 将短期特征扩展到每个时间步: (batch, 64) -> (batch, seq_len, 64)
        short_features_seq = short_tcn_out.permute(0, 2, 1)  # (batch, seq_len, 64)
        shared_features_seq = shared_features.permute(0, 2, 1)  # (batch, seq_len, 256)
        
        # 优化1: 交叉注意力机制
        if self.use_attention:
            # 长期TCN关注短期特征中的重要部分
            attended_short = self.cross_attention(
                query=shared_features_seq,  # 长期特征作为query
                key=short_features_seq,      # 短期特征作为key
                value=short_features_seq    # 短期特征作为value
            )  # (batch, seq_len, 64)
            
            # 融合共享特征和注意力加权的短期特征
            if self.use_gated_fusion:
                # 优化2: 门控融合机制
                long_tcn_input = self.gated_fusion(
                    shared_feat=shared_features_seq,
                    short_feat=attended_short
                )  # (batch, seq_len, 256)
            else:
                # 简单拼接
                long_tcn_input = torch.cat([shared_features_seq, attended_short], dim=-1)  # (batch, seq_len, 320)
        elif self.use_gated_fusion:
            # 只使用门控融合，不使用注意力
            long_tcn_input = self.gated_fusion(
                shared_feat=shared_features_seq,
                short_feat=short_features_seq
            )  # (batch, seq_len, 256)
        else:
            # 简单拼接（原始方法）
            short_features_expanded = short_features.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, 64)
            long_tcn_input = torch.cat([shared_features_seq, short_features_expanded], dim=-1)  # (batch, seq_len, 320)
        
        # 转换回TCN输入格式: (batch, channels, seq_len)
        long_tcn_input = long_tcn_input.permute(0, 2, 1)  # (batch, channels, seq_len)
        
        long_tcn_out = self.tcn_long(long_tcn_input)  # (batch, 64, seq_len)
        long_features = long_tcn_out[:, :, -1]  # (batch, 64) - 取最后时间步
        
        # 长期预测
        long_pred = F.leaky_relu(self.bn_long_1(self.fc_long_1(long_features)), negative_slope=0.01)
        long_pred = self.dropout_long(long_pred)
        pred_60s_base = self.fc_long_2(long_pred)
        
        # 优化3: 残差连接（短期预测到长期预测）
        if self.use_residual:
            # 将短期预测投影并添加到长期预测
            pred_60s_residual = self.residual_proj(pred_10s)
            pred_60s = pred_60s_base + pred_60s_residual
        else:
            pred_60s = pred_60s_base
        
        return pred_10s, pred_60s


# ============================================================================
# 4. Training Function (单进程版本，用于并行调用)
# ============================================================================

def train_single_symbol_worker(args):
    """单标的训练函数（用于并行调用）"""
    symbol, data_dir, output_dir, log_dir, config, gpu_id = args
    
    # 设置当前进程使用的GPU
    if gpu_id >= torch.cuda.device_count():
        gpu_id = 0
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # 设置日志文件
    log_file = log_dir / f"{symbol}_training.log"
    log_f = open(log_file, 'w')
    
    def log_print(*args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()
    
    try:
        log_print(f"\n{'='*80}")
        log_print(f"🚀 开始训练: {symbol} (GPU {gpu_id}) - DeepLOB-TCN Hierarchical Model (Optimized)")
        log_print(f"   EMA Alpha: {config.get('ema_alpha', 0.2)}")
        log_print(f"   Loss Weight (10s/60s): {config.get('loss_weight_10s', 0.5)}/{config.get('loss_weight_60s', 0.5)}")
        log_print(f"   Use Attention: {config.get('use_attention', True)}")
        log_print(f"   Use Gated Fusion: {config.get('use_gated_fusion', True)}")
        log_print(f"   Use Residual: {config.get('use_residual', True)}")
        log_print(f"   Adaptive Loss Weight: {config.get('adaptive_loss_weight', False)}")
        log_print(f"{'='*80}")
        
        start_time = time.time()
        
        # 文件路径
        data_file = Path(data_dir) / f"{symbol}_20250801_20250810.npy"
        if not data_file.exists():
            log_print(f"   ❌ 文件不存在: {data_file}")
            return None
        
        # 创建数据集 - Hierarchical多目标
        log_print(f"   📂 加载数据 (Hierarchical, EMA平滑: alpha={config.get('ema_alpha', 0.2)})...")
        train_dataset = HierarchicalLOBDataset(
            data_file, start_ratio=0.0, end_ratio=0.6,
            sequence_length=config['sequence_length'], 
            fit_scaler=True,
            ema_alpha=config.get('ema_alpha', 0.2)
        )
        
        val_dataset = HierarchicalLOBDataset(
            data_file, start_ratio=0.6, end_ratio=0.8,
            sequence_length=config['sequence_length'],
            scaler=train_dataset.scaler,
            fit_scaler=False,
            ema_alpha=config.get('ema_alpha', 0.2)
        )
        
        test_dataset = HierarchicalLOBDataset(
            data_file, start_ratio=0.8, end_ratio=1.0,
            sequence_length=config['sequence_length'],
            scaler=train_dataset.scaler,
            fit_scaler=False,
            ema_alpha=config.get('ema_alpha', 0.2)
        )
        
        log_print(f"      Train: {len(train_dataset):,} samples")
        log_print(f"      Val:   {len(val_dataset):,} samples")
        log_print(f"      Test:  {len(test_dataset):,} samples")
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'],
            shuffle=True, num_workers=config['num_workers'], pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'],
            shuffle=False, num_workers=config['num_workers'], pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=config['batch_size'],
            shuffle=False, num_workers=config['num_workers'], pin_memory=True
        )
        
        # 创建模型
        log_print(f"   🏗️  创建模型... (DeepLOB-TCN Hierarchical Optimized)")
        model = DeepLOB_Hierarchical_TCN(
            input_channels=1, 
            dropout=config['dropout'],
            short_term_channels=config.get('short_term_channels', [64, 64, 64, 64]),
            long_term_channels=config.get('long_term_channels', [64, 64, 64, 64, 64]),
            use_attention=config.get('use_attention', True),
            use_gated_fusion=config.get('use_gated_fusion', True),
            use_residual=config.get('use_residual', True)
        ).to(device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.HuberLoss(delta=1.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # 损失权重（支持自适应调整）
        adaptive_loss_weight = config.get('adaptive_loss_weight', False)
        loss_weight_10s = config.get('loss_weight_10s', 0.5)
        loss_weight_60s = config.get('loss_weight_60s', 0.5)
        
        # 自适应损失权重：初始权重和调整参数
        if adaptive_loss_weight:
            initial_weight_10s = loss_weight_10s
            initial_weight_60s = loss_weight_60s
            log_print(f"   📊 自适应损失权重已启用")
        
        # 训练循环
        log_print(f"   🏋️  开始训练...")
        best_val_loss = float('inf')
        patience_counter = 0
        epoch_times = []
        history = {
            'train_loss': [],
            'train_loss_10s': [],
            'train_loss_60s': [],
            'val_loss': [],
            'val_loss_10s': [],
            'val_loss_60s': [],
            'lr': [],
            'epoch_times': []
        }
        
        for epoch in range(config['num_epochs']):
            epoch_start = time.time()
            
            # 自适应损失权重调整（如果启用）
            if adaptive_loss_weight and epoch > 0:
                # 根据上一轮的损失比例动态调整权重
                # 如果loss_10s相对较大，增加其权重；反之亦然
                if history['train_loss_10s'][-1] > 0 and history['train_loss_60s'][-1] > 0:
                    ratio_10s = history['train_loss_10s'][-1] / (history['train_loss_10s'][-1] + history['train_loss_60s'][-1])
                    ratio_60s = history['train_loss_60s'][-1] / (history['train_loss_10s'][-1] + history['train_loss_60s'][-1])
                    # 平滑调整：使用指数移动平均
                    alpha = 0.1  # 调整速度
                    loss_weight_10s = (1 - alpha) * loss_weight_10s + alpha * ratio_10s
                    loss_weight_60s = (1 - alpha) * loss_weight_60s + alpha * ratio_60s
                    # 归一化
                    total_weight = loss_weight_10s + loss_weight_60s
                    loss_weight_10s = loss_weight_10s / total_weight
                    loss_weight_60s = loss_weight_60s / total_weight
            
            # 训练
            model.train()
            train_loss = 0.0
            train_loss_10s = 0.0
            train_loss_60s = 0.0
            for batch_x, batch_y_10s, batch_y_60s in train_loader:
                batch_x = batch_x.to(device)
                batch_y_10s = batch_y_10s.to(device)
                batch_y_60s = batch_y_60s.to(device)
                
                optimizer.zero_grad()
                pred_10s, pred_60s = model(batch_x)
                
                loss_10s = criterion(pred_10s.squeeze(), batch_y_10s.squeeze())
                loss_60s = criterion(pred_60s.squeeze(), batch_y_60s.squeeze())
                loss = loss_weight_10s * loss_10s + loss_weight_60s * loss_60s
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_loss_10s += loss_10s.item()
                train_loss_60s += loss_60s.item()
            
            train_loss /= len(train_loader)
            train_loss_10s /= len(train_loader)
            train_loss_60s /= len(train_loader)
            
            # 验证
            model.eval()
            val_loss = 0.0
            val_loss_10s = 0.0
            val_loss_60s = 0.0
            with torch.no_grad():
                for batch_x, batch_y_10s, batch_y_60s in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y_10s = batch_y_10s.to(device)
                    batch_y_60s = batch_y_60s.to(device)
                    
                    pred_10s, pred_60s = model(batch_x)
                    
                    loss_10s = criterion(pred_10s.squeeze(), batch_y_10s.squeeze())
                    loss_60s = criterion(pred_60s.squeeze(), batch_y_60s.squeeze())
                    loss = loss_weight_10s * loss_10s + loss_weight_60s * loss_60s
                    
                    val_loss += loss.item()
                    val_loss_10s += loss_10s.item()
                    val_loss_60s += loss_60s.item()
            
            val_loss /= len(val_loader)
            val_loss_10s /= len(val_loader)
            val_loss_60s /= len(val_loader)
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            history['train_loss'].append(train_loss)
            history['train_loss_10s'].append(train_loss_10s)
            history['train_loss_60s'].append(train_loss_60s)
            history['val_loss'].append(val_loss)
            history['val_loss_10s'].append(val_loss_10s)
            history['val_loss_60s'].append(val_loss_60s)
            history['lr'].append(current_lr)
            history['epoch_times'].append(epoch_time)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': train_dataset.scaler,
                    'config': config
                }, output_dir / f"{symbol}_best_model.pth")
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                weight_info = ""
                if adaptive_loss_weight:
                    weight_info = f" | Weights: {loss_weight_10s:.3f}/{loss_weight_60s:.3f}"
                log_print(f"      Epoch {epoch+1:2d}/{config['num_epochs']} | "
                          f"Train: {train_loss:.6f} (10s: {train_loss_10s:.6f}, 60s: {train_loss_60s:.6f}) | "
                          f"Val: {val_loss:.6f} (10s: {val_loss_10s:.6f}, 60s: {val_loss_60s:.6f}) | "
                          f"LR: {current_lr:.6f}{weight_info} | Time: {epoch_time:.2f}s")
            
            if patience_counter >= config['early_stopping_patience']:
                log_print(f"      ⏹️  Early stopping at epoch {epoch+1}")
                break
        
        if epoch_times:
            avg_epoch_time = np.mean(epoch_times)
            log_print(f"      ⏱️  平均每个Epoch: {avg_epoch_time:.2f}秒")
        
        # 测试
        log_print(f"   📊 测试最佳模型...")
        checkpoint = torch.load(output_dir / f"{symbol}_best_model.pth", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        test_preds_10s = []
        test_targets_10s = []
        test_preds_60s = []
        test_targets_60s = []
        test_loss = 0.0
        test_loss_10s = 0.0
        test_loss_60s = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y_10s, batch_y_60s in test_loader:
                batch_x = batch_x.to(device)
                batch_y_10s = batch_y_10s.to(device)
                batch_y_60s = batch_y_60s.to(device)
                
                pred_10s, pred_60s = model(batch_x)
                
                loss_10s = criterion(pred_10s.squeeze(), batch_y_10s.squeeze())
                loss_60s = criterion(pred_60s.squeeze(), batch_y_60s.squeeze())
                loss = loss_weight_10s * loss_10s + loss_weight_60s * loss_60s
                
                test_loss += loss.item()
                test_loss_10s += loss_10s.item()
                test_loss_60s += loss_60s.item()
                
                test_preds_10s.append(pred_10s.cpu().numpy())
                test_targets_10s.append(batch_y_10s.cpu().numpy())
                test_preds_60s.append(pred_60s.cpu().numpy())
                test_targets_60s.append(batch_y_60s.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_loss_10s /= len(test_loader)
        test_loss_60s /= len(test_loader)
        
        test_preds_10s = np.concatenate(test_preds_10s).flatten()
        test_targets_10s = np.concatenate(test_targets_10s).flatten()
        test_preds_60s = np.concatenate(test_preds_60s).flatten()
        test_targets_60s = np.concatenate(test_targets_60s).flatten()
        
        # 计算指标 - return_10s
        mae_10s = mean_absolute_error(test_targets_10s, test_preds_10s)
        rmse_10s = np.sqrt(mean_squared_error(test_targets_10s, test_preds_10s))
        r2_10s = r2_score(test_targets_10s, test_preds_10s)
        corr_10s = np.corrcoef(test_targets_10s, test_preds_10s)[0, 1] if len(test_targets_10s) > 1 else 0.0
        
        # 计算指标 - return_60s
        mae_60s = mean_absolute_error(test_targets_60s, test_preds_60s)
        rmse_60s = np.sqrt(mean_squared_error(test_targets_60s, test_preds_60s))
        r2_60s = r2_score(test_targets_60s, test_preds_60s)
        corr_60s = np.corrcoef(test_targets_60s, test_preds_60s)[0, 1] if len(test_targets_60s) > 1 else 0.0
        
        training_time = time.time() - start_time
        
        log_print(f"\n   ✅ 训练完成!")
        log_print(f"      Test Loss: {test_loss:.6f} (10s: {test_loss_10s:.6f}, 60s: {test_loss_60s:.6f})")
        log_print(f"      Return 10s - MAE: {mae_10s:.6f}, RMSE: {rmse_10s:.6f}, R²: {r2_10s:.6f}, Corr: {corr_10s:.6f}")
        log_print(f"      Return 60s - MAE: {mae_60s:.6f}, RMSE: {rmse_60s:.6f}, R²: {r2_60s:.6f}, Corr: {corr_60s:.6f}")
        log_print(f"      Time: {training_time/60:.2f} min")
        
        # 保存结果
        result = {
            'symbol': symbol,
            'test_loss': float(test_loss),
            'test_loss_10s': float(test_loss_10s),
            'test_loss_60s': float(test_loss_60s),
            'mae_10s': float(mae_10s),
            'rmse_10s': float(rmse_10s),
            'r2_10s': float(r2_10s),
            'correlation_10s': float(corr_10s),
            'mae_60s': float(mae_60s),
            'rmse_60s': float(rmse_60s),
            'r2_60s': float(r2_60s),
            'correlation_60s': float(corr_60s),
            'best_val_loss': float(best_val_loss),
            'training_time_minutes': float(training_time / 60),
            'epochs_trained': len(history['train_loss']),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'ema_alpha': config.get('ema_alpha', 0.2)
        }
        
        # 保存历史
        with open(output_dir / f"{symbol}_history.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        # 保存预测
        np.savez(
            output_dir / f"{symbol}_predictions.npz",
            predictions_10s=test_preds_10s,
            targets_10s=test_targets_10s,
            predictions_60s=test_preds_60s,
            targets_60s=test_targets_60s
        )
        
        log_f.close()
        return result
        
    except Exception as e:
        log_print(f"   ❌ 训练失败: {e}")
        import traceback
        log_print(traceback.format_exc())
        log_f.close()
        return None


# ============================================================================
# 5. Report Generation Functions
# ============================================================================

def generate_single_timescale_report(df, timescale, output_dir, image_dir):
    """为单个时间尺度（10s或60s）生成类似7系列的图表和表格"""
    corr_col = f'correlation_{timescale}'
    mae_col = f'mae_{timescale}'
    rmse_col = f'rmse_{timescale}'
    r2_col = f'r2_{timescale}'
    
    # 准备数据（类似7系列的格式）
    df_single = pd.DataFrame({
        'Symbol': df['symbol'],
        'Correlation': df[corr_col],
        'MAE': df[mae_col],
        'RMSE': df[rmse_col],
        'R²': df[r2_col]
    })
    df_single = df_single.sort_values('Correlation', ascending=False).reset_index(drop=True)
    
    # ============================================================================
    # Figure: Core Performance (2x2 layout) - 类似7系列
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1a) Correlation by Symbol
    ax = axes[0, 0]
    colors = ['#27ae60' if x > 0.15 else '#3498db' if x > 0.05 else '#e74c3c' for x in df_single['Correlation']]
    bars = ax.barh(range(len(df_single)), df_single['Correlation'], color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(df_single)))
    ax.set_yticklabels(df_single['Symbol'], fontsize=9)
    ax.set_xlabel('Correlation Coefficient', fontweight='bold', fontsize=12)
    ax.set_title('(A) Prediction Correlation by Symbol', fontsize=14, fontweight='bold', pad=12)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.2)
    ax.axvline(df_single['Correlation'].mean(), color='red', linestyle='--', linewidth=2.5, 
                label=f'Mean: {df_single["Correlation"].mean():.3f}', alpha=0.8)
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.8)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    if len(df_single) > 0:
        ax.set_xlim(-0.05, max(0.30, df_single['Correlation'].max() * 1.2))
    
    # 1b) MAE vs Correlation (Scatter with size by RMSE)
    ax = axes[0, 1]
    scatter = ax.scatter(df_single['Correlation'], df_single['MAE'], s=df_single['RMSE']*30, 
                        c=df_single['Correlation'], cmap='RdYlGn', alpha=0.7, 
                        edgecolor='black', linewidth=1.2)
    # Annotate top performers
    for idx, row in df_single.head(min(3, len(df_single))).iterrows():
        ax.annotate(row['Symbol'], (row['Correlation'], row['MAE']), 
                    fontsize=9, fontweight='bold', ha='right', va='bottom',
                    xytext=(-5, 5), textcoords='offset points')
    ax.set_xlabel('Correlation Coefficient', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (BPS)', fontweight='bold', fontsize=12)
    ax.set_title('(B) Prediction Accuracy vs Correlation', fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Correlation', fontweight='bold', fontsize=11)
    
    # 1c) MAE by Symbol (Sorted)
    ax = axes[1, 0]
    df_sorted_mae = df_single.sort_values('MAE')
    colors_mae = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted_mae)))
    ax.barh(range(len(df_sorted_mae)), df_sorted_mae['MAE'], 
           color=colors_mae, edgecolor='black', linewidth=0.6, alpha=0.85)
    ax.set_yticks(range(len(df_sorted_mae)))
    ax.set_yticklabels(df_sorted_mae['Symbol'], fontsize=9)
    ax.set_xlabel('Mean Absolute Error (BPS)', fontweight='bold', fontsize=12)
    ax.set_title('(C) MAE by Symbol (Sorted)', fontsize=14, fontweight='bold', pad=12)
    ax.axvline(df_single['MAE'].mean(), color='red', linestyle='--', linewidth=2.5, 
              label=f'Mean: {df_single["MAE"].mean():.2f}', alpha=0.8)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.8)
    
    # 1d) Correlation vs MAE Relationship
    ax = axes[1, 1]
    ax.scatter(df_single['Correlation'], df_single['MAE'], s=150, alpha=0.7, 
              c=df_single['Correlation'], cmap='RdYlGn', edgecolor='black', linewidth=1.5)
    
    # Add regression line
    if len(df_single) > 1:
        z = np.polyfit(df_single['Correlation'], df_single['MAE'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_single['Correlation'].min(), df_single['Correlation'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=3, alpha=0.8, 
               label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Annotate top performers
    for idx, row in df_single.head(min(3, len(df_single))).iterrows():
        ax.annotate(row['Symbol'], (row['Correlation'], row['MAE']), 
                   fontsize=10, fontweight='bold', ha='right', va='bottom',
                   xytext=(-8, 8), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Correlation Coefficient', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (BPS)', fontweight='bold', fontsize=12)
    ax.set_title('(D) Correlation vs MAE Relationship', fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    
    plt.suptitle(f'DeepLOB-TCN Hierarchical Model - Return {timescale} Performance Metrics', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(image_dir / f'fig1_core_performance_{timescale}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"     ✓ Saved: fig1_core_performance_{timescale}.png")
    
    # ============================================================================
    # Summary Statistics Table (类似7系列格式)
    # ============================================================================
    summary_data = {
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Correlation': [
            df_single['Correlation'].mean(),
            df_single['Correlation'].median(),
            df_single['Correlation'].std(),
            df_single['Correlation'].min(),
            df_single['Correlation'].max()
        ],
        'MAE (BPS)': [
            df_single['MAE'].mean(),
            df_single['MAE'].median(),
            df_single['MAE'].std(),
            df_single['MAE'].min(),
            df_single['MAE'].max()
        ],
        'RMSE (BPS)': [
            df_single['RMSE'].mean(),
            df_single['RMSE'].median(),
            df_single['RMSE'].std(),
            df_single['RMSE'].min(),
            df_single['RMSE'].max()
        ],
        'R² Score': [
            df_single['R²'].mean(),
            df_single['R²'].median(),
            df_single['R²'].std(),
            df_single['R²'].min(),
            df_single['R²'].max()
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / f'summary_statistics_{timescale}.csv', index=False, float_format='%.4f')
    print(f"     ✓ Saved: summary_statistics_{timescale}.csv")

def generate_final_report(df_results, output_dir, image_dir):
    """生成最终报告：包含核心图表和表格"""
    print("\n生成报告图表和表格...")
    
    # 从所有预测文件中读取数据，确保包含所有已完成的标的
    print("  从预测文件中读取所有已完成的标的...")
    results = []
    for pred_file in sorted(output_dir.glob('*_predictions.npz')):
        symbol = pred_file.stem.replace('_predictions', '')
        try:
            data = np.load(pred_file)
            pred_10s = data['predictions_10s']
            target_10s = data['targets_10s']
            pred_60s = data['predictions_60s']
            target_60s = data['targets_60s']
            
            # 计算指标
            mae_10s = mean_absolute_error(target_10s, pred_10s)
            rmse_10s = np.sqrt(mean_squared_error(target_10s, pred_10s))
            r2_10s = r2_score(target_10s, pred_10s)
            corr_10s = np.corrcoef(target_10s, pred_10s)[0, 1] if len(target_10s) > 1 else 0.0
            
            mae_60s = mean_absolute_error(target_60s, pred_60s)
            rmse_60s = np.sqrt(mean_squared_error(target_60s, pred_60s))
            r2_60s = r2_score(target_60s, pred_60s)
            corr_60s = np.corrcoef(target_60s, pred_60s)[0, 1] if len(target_60s) > 1 else 0.0
            
            results.append({
                'symbol': symbol,
                'mae_10s': mae_10s,
                'rmse_10s': rmse_10s,
                'r2_10s': r2_10s,
                'correlation_10s': corr_10s,
                'mae_60s': mae_60s,
                'rmse_60s': rmse_60s,
                'r2_60s': r2_60s,
                'correlation_60s': corr_60s
            })
        except Exception as e:
            print(f"    ⚠️  {symbol}: {e}")
    
    # 使用从预测文件读取的数据
    df = pd.DataFrame(results)
    print(f"   ✅ 读取到 {len(df)} 个已完成的标的")
    
    # 创建综合性能指标（平均两个时间尺度）
    df['MAE_avg'] = (df['mae_10s'] + df['mae_60s']) / 2
    df['RMSE_avg'] = (df['rmse_10s'] + df['rmse_60s']) / 2
    df['R²_avg'] = (df['r2_10s'] + df['r2_60s']) / 2
    df['Correlation_avg'] = (df['correlation_10s'] + df['correlation_60s']) / 2
    
    df = df.sort_values('Correlation_avg', ascending=False).reset_index(drop=True)
    
    # ============================================================================
    # Figure 1: Core Performance Comparison (10s vs 60s)
    # ============================================================================
    print("  1. 创建核心性能对比图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1a) Correlation Comparison (10s vs 60s)
    ax = axes[0, 0]
    x_pos = np.arange(len(df))
    width = 0.35
    bars1 = ax.barh(x_pos - width/2, df['correlation_10s'], width, 
                   label='Return 10s', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax.barh(x_pos + width/2, df['correlation_60s'], width, 
                   label='Return 60s', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(df['symbol'], fontsize=9)
    ax.set_xlabel('Correlation Coefficient', fontweight='bold', fontsize=12)
    ax.set_title('(A) Correlation: 10s vs 60s', fontsize=14, fontweight='bold', pad=12)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.2)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.8)
    
    # 1b) MAE Comparison (10s vs 60s)
    ax = axes[0, 1]
    bars1 = ax.barh(x_pos - width/2, df['mae_10s'], width, 
                   label='Return 10s', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax.barh(x_pos + width/2, df['mae_60s'], width, 
                   label='Return 60s', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(df['symbol'], fontsize=9)
    ax.set_xlabel('Mean Absolute Error (BPS)', fontweight='bold', fontsize=12)
    ax.set_title('(B) MAE: 10s vs 60s', fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.8)
    
    # 1c) Correlation Scatter (10s vs 60s)
    ax = axes[1, 0]
    ax.scatter(df['correlation_10s'], df['correlation_60s'], 
              s=150, alpha=0.7, c=df['Correlation_avg'], cmap='RdYlGn', 
              edgecolor='black', linewidth=1.5)
    
    # 添加对角线
    min_corr = min(df['correlation_10s'].min(), df['correlation_60s'].min())
    max_corr = max(df['correlation_10s'].max(), df['correlation_60s'].max())
    ax.plot([min_corr, max_corr], [min_corr, max_corr], 'r--', linewidth=2, alpha=0.5, label='y=x')
    
    # 标注top performers
    for idx, row in df.head(min(3, len(df))).iterrows():
        ax.annotate(row['symbol'], (row['correlation_10s'], row['correlation_60s']), 
                   fontsize=9, fontweight='bold', ha='right', va='bottom',
                   xytext=(-5, 5), textcoords='offset points')
    
    ax.set_xlabel('Correlation (10s)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Correlation (60s)', fontweight='bold', fontsize=12)
    ax.set_title('(C) Correlation: 10s vs 60s Scatter', fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    
    # 1d) Average Performance by Symbol
    ax = axes[1, 1]
    colors = ['#27ae60' if x > 0.15 else '#3498db' if x > 0.05 else '#e74c3c' 
              for x in df['Correlation_avg']]
    bars = ax.barh(range(len(df)), df['Correlation_avg'], color=colors, 
                   alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['symbol'], fontsize=9)
    ax.set_xlabel('Average Correlation Coefficient', fontweight='bold', fontsize=12)
    ax.set_title('(D) Average Correlation (10s & 60s)', fontsize=14, fontweight='bold', pad=12)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.2)
    ax.axvline(df['Correlation_avg'].mean(), color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean: {df["Correlation_avg"].mean():.3f}', alpha=0.8)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.8)
    
    plt.suptitle('DeepLOB-TCN Hierarchical Model - Performance Comparison (10s vs 60s)', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(image_dir / 'fig1_core_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("     ✓ Saved: fig1_core_performance.png")
    
    # ============================================================================
    # Figure 2: Summary Statistics (Table Visualization)
    # ============================================================================
    print("  2. 创建汇总统计表格...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')
    
    # Calculate statistics for both time scales
    summary_data = {
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Corr 10s': [
            df['correlation_10s'].mean(),
            df['correlation_10s'].median(),
            df['correlation_10s'].std(),
            df['correlation_10s'].min(),
            df['correlation_10s'].max()
        ],
        'Corr 60s': [
            df['correlation_60s'].mean(),
            df['correlation_60s'].median(),
            df['correlation_60s'].std(),
            df['correlation_60s'].min(),
            df['correlation_60s'].max()
        ],
        'MAE 10s': [
            df['mae_10s'].mean(),
            df['mae_10s'].median(),
            df['mae_10s'].std(),
            df['mae_10s'].min(),
            df['mae_10s'].max()
        ],
        'MAE 60s': [
            df['mae_60s'].mean(),
            df['mae_60s'].median(),
            df['mae_60s'].std(),
            df['mae_60s'].min(),
            df['mae_60s'].max()
        ],
        'R² 10s': [
            df['r2_10s'].mean(),
            df['r2_10s'].median(),
            df['r2_10s'].std(),
            df['r2_10s'].min(),
            df['r2_10s'].max()
        ],
        'R² 60s': [
            df['r2_60s'].mean(),
            df['r2_60s'].median(),
            df['r2_60s'].std(),
            df['r2_60s'].min(),
            df['r2_60s'].max()
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_text_props(weight='normal')
    
    # Format numbers
    for i in range(1, len(summary_df) + 1):
        for j in range(1, len(summary_df.columns)):
            val = summary_df.iloc[i-1, j]
            if j <= 2:  # Correlation columns
                table[(i, j)].get_text().set_text(f'{val:.4f}')
            elif j <= 4:  # MAE columns
                table[(i, j)].get_text().set_text(f'{val:.2f}')
            else:  # R² columns
                table[(i, j)].get_text().set_text(f'{val:.4f}')
    
    ax.set_title('Summary Statistics (Hierarchical Model)', fontsize=18, fontweight='bold', pad=20)
    plt.savefig(image_dir / 'fig2_summary_statistics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("     ✓ Saved: fig2_summary_statistics.png")
    
    # ============================================================================
    # Figure 3: Top Performers (Table Visualization)
    # ============================================================================
    print("  3. 创建Top Performers表格...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.axis('off')
    
    top_n = df.copy()
    top_n.insert(0, 'Rank', range(1, len(top_n) + 1))
    top_n_display = top_n[['Rank', 'symbol', 'correlation_10s', 'correlation_60s', 
                           'mae_10s', 'mae_60s', 'Correlation_avg']].copy()
    
    # Format numbers
    top_n_display['correlation_10s'] = top_n_display['correlation_10s'].apply(lambda x: f'{x:.4f}')
    top_n_display['correlation_60s'] = top_n_display['correlation_60s'].apply(lambda x: f'{x:.4f}')
    top_n_display['mae_10s'] = top_n_display['mae_10s'].apply(lambda x: f'{x:.2f}')
    top_n_display['mae_60s'] = top_n_display['mae_60s'].apply(lambda x: f'{x:.2f}')
    top_n_display['Correlation_avg'] = top_n_display['Correlation_avg'].apply(lambda x: f'{x:.4f}')
    
    # Create table
    table = ax.table(cellText=top_n_display.values,
                    colLabels=top_n_display.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    
    # Style the table
    for i in range(len(top_n_display.columns)):
        table[(0, i)].set_facecolor('#27ae60')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(top_n_display) + 1):
        for j in range(len(top_n_display.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_text_props(weight='normal')
        
        # Highlight top 3
        if i <= min(3, len(top_n_display)):
            for j in range(len(top_n_display.columns)):
                table[(i, j)].set_facecolor('#d5f4e6')
    
    ax.set_title(f'Top {len(top_n)} Performers by Average Correlation', fontsize=18, fontweight='bold', pad=20)
    plt.savefig(image_dir / 'fig3_top_performers.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("     ✓ Saved: fig3_top_performers.png")
    
    # Save CSV tables
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False, float_format='%.4f')
    top_n[['Rank', 'symbol', 'correlation_10s', 'correlation_60s', 'mae_10s', 'mae_60s', 
           'Correlation_avg']].to_csv(
        output_dir / 'top_performers.csv', index=False, float_format='%.4f')
    print("     ✓ Saved: summary_statistics.csv")
    print("     ✓ Saved: top_performers.csv")
    
    # ============================================================================
    # 为10s和60s分别生成类似7系列的图表和表格
    # ============================================================================
    
    # 生成Return 10s的图表和表格
    print("\n  4. 生成Return 10s的图表和表格...")
    generate_single_timescale_report(df, '10s', output_dir, image_dir)
    
    # 生成Return 60s的图表和表格
    print("\n  5. 生成Return 60s的图表和表格...")
    generate_single_timescale_report(df, '60s', output_dir, image_dir)
    
    print("\n✅ 报告图表和表格生成完成！")


# ============================================================================
# 6. Main Function
# ============================================================================

def main():
    print("="*80)
    print("🚀 DeepLOB-TCN Hierarchical Modeling - Multi-Timescale Prediction (Optimized)")
    print("="*80)
    print("\n策略: 每个标的独立训练，支持多GPU并行")
    print("数据: 10天数据 (2025-08-01 to 2025-08-10)")
    print("模型: DeepLOB-TCN Hierarchical (同时预测return_10s和return_60s)")
    print("改进: Hierarchical架构 + EMA平滑 + 多任务学习")
    print("优化: 交叉注意力 + 门控融合 + 残差连接 + 自适应损失权重")
    
    # 配置
    config = {
        'sequence_length': 100,
        'batch_size': 2048,
        'num_workers': 2,
        'learning_rate': 0.001,
        'num_epochs': 20,
        'dropout': 0.3,
        'early_stopping_patience': 5,
        'ema_alpha': 0.2,  # EMA平滑因子
        'loss_weight_10s': 0.5,  # return_10s损失权重（初始值）
        'loss_weight_60s': 0.5,  # return_60s损失权重（初始值）
        'short_term_channels': [64, 64, 64, 64],  # 短期TCN通道数
        'long_term_channels': [64, 64, 64, 64, 64],  # 长期TCN通道数（更深）
        # 优化选项
        'use_attention': True,  # 使用交叉注意力机制
        'use_gated_fusion': True,  # 使用门控融合机制
        'use_residual': True,  # 使用残差连接
        'adaptive_loss_weight': False  # 自适应损失权重（可选，默认关闭）
    }
    
    # 路径 - 所有输出保存到8_models文件夹
    data_dir = Path('data_250801_250810')
    output_dir = Path('8_models')
    log_dir = output_dir / 'log'
    image_dir = output_dir / 'image'
    
    output_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    
    # 读取所有标的
    metadata_file = data_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        all_symbols = metadata.get('symbols', [])
    else:
        npy_files = sorted(data_dir.glob('*_20250801_20250810.npy'))
        all_symbols = [f.stem.replace('_20250801_20250810', '') for f in npy_files]
    
    # ✅ 过滤已完成的标的（检查predictions.npz文件是否存在）
    target_symbols = []
    skipped_symbols = []
    for symbol in all_symbols:
        predictions_file = output_dir / f"{symbol}_predictions.npz"
        if predictions_file.exists():
            skipped_symbols.append(symbol)
        else:
            target_symbols.append(symbol)
    
    print(f"\n找到 {len(all_symbols)} 个标的")
    print(f"  已跳过 {len(skipped_symbols)} 个已完成的标的: {', '.join(skipped_symbols) if skipped_symbols else '无'}")
    print(f"  待训练 {len(target_symbols)} 个标的")
    if target_symbols:
        print(f"  待训练标的: {', '.join(target_symbols)}")
    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"日志目录: {log_dir}")
    print(f"输出目录: {output_dir}")
    print(f"✅ EMA平滑因子: {config['ema_alpha']}")
    print(f"✅ 损失权重: 10s={config['loss_weight_10s']}, 60s={config['loss_weight_60s']}")
    print(f"✅ 优化选项:")
    print(f"   - 交叉注意力: {config.get('use_attention', True)}")
    print(f"   - 门控融合: {config.get('use_gated_fusion', True)}")
    print(f"   - 残差连接: {config.get('use_residual', True)}")
    print(f"   - 自适应损失权重: {config.get('adaptive_loss_weight', False)}")
    
    if not target_symbols:
        print(f"\n✅ 所有标的已完成训练，无需训练")
        return
    
    print(f"\n🔄 开始训练 {len(target_symbols)} 个标的...")
    
    # 准备并行训练参数
    n_gpus = torch.cuda.device_count()
    print(f"\n可用GPU数量: {n_gpus}")
    
    if n_gpus >= 4:
        max_workers = 4  # 使用GPU 0、1、2、3，每张GPU 1个进程
        print(f"✅ 将使用 {max_workers} 个进程并行训练 (GPU 0、1、2、3，每张GPU 1个进程)")
    elif n_gpus >= 2:
        max_workers = min(4, n_gpus)
        print(f"✅ 将使用 {max_workers} 个进程并行训练 (GPU 0、1，每张GPU {max_workers//2}个进程)")
    else:
        max_workers = min(4, n_gpus)
        print(f"✅ 将使用 {max_workers} 个进程共享1个GPU并行训练")
    
    print(f"   Batch Size: {config['batch_size']}")
    
    # 准备任务列表
    tasks = []
    for i, symbol in enumerate(target_symbols):
        if n_gpus >= 4:
            gpu_id = i % 4
        elif n_gpus >= 2:
            gpu_id = i % 2
        else:
            gpu_id = 0
        tasks.append((symbol, data_dir, output_dir, log_dir, config, gpu_id))
        print(f"   {symbol} -> GPU {gpu_id}")
    
    results = []
    failed_symbols = []
    
    if tasks:
        # 并行训练
        print(f"\n{'='*80}")
        print(f"🏋️  开始并行训练 ({max_workers} 个进程)")
        print(f"{'='*80}\n")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(train_single_symbol_worker, task): task[0] 
                               for task in tasks}
            
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"[{completed}/{len(tasks)}] ✅ {symbol} 完成")
                    else:
                        failed_symbols.append(symbol)
                        print(f"[{completed}/{len(tasks)}] ❌ {symbol} 失败")
                except Exception as e:
                    failed_symbols.append(symbol)
                    print(f"[{completed}/{len(tasks)}] ❌ {symbol} 异常: {e}")
    
    # 保存汇总报告
    print(f"\n{'='*80}")
    print(f"📊 训练汇总")
    print(f"{'='*80}")
    
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('correlation_10s', ascending=False)
    
        # 保存CSV
        csv_file = output_dir / 'training_summary_hierarchical.csv'
        df_results.to_csv(csv_file, index=False)
        print(f"\n✅ 汇总表格已保存: {csv_file}")
        
        # 打印统计
        print(f"\n📈 性能统计:")
        print(f"   成功训练: {len(results)}/{len(target_symbols)}")
        if len(results) > 0:
            print(f"   Return 10s - 平均 MAE:  {df_results['mae_10s'].mean():.6f}, 平均 Corr: {df_results['correlation_10s'].mean():.6f}")
            print(f"   Return 60s - 平均 MAE:  {df_results['mae_60s'].mean():.6f}, 平均 Corr: {df_results['correlation_60s'].mean():.6f}")
    
        print(f"\n🏆 Top 10 标的 (按Correlation_10s排序):")
        print(df_results[['symbol', 'correlation_10s', 'correlation_60s', 'mae_10s', 'mae_60s']].head(10).to_string(index=False))
        
        # ============================================================================
        # 生成报告图表和表格
        # ============================================================================
        print(f"\n{'='*80}")
        print(f"📊 生成报告图表和表格")
        print(f"{'='*80}")
        
        generate_final_report(df_results, output_dir, image_dir)
    
    if failed_symbols:
        print(f"\n❌ 失败的标的: {failed_symbols}")
    
    print(f"\n{'='*80}")
    print(f"✅ 所有训练完成!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

