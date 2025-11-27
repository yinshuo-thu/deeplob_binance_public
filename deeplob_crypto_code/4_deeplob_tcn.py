#!/usr/bin/env python3
"""
DeepLOB-TCN Per-Symbol Training - 并行版本

TCN Architecture - 用时序卷积网络(TCN)替换LSTM进行时序建模

主要改进:
- 用 TCN 替换 LSTM 部分
- 保留 DeepLOB 的 CNN 特征提取
- TCN 核心: 因果卷积 + 膨胀卷积 + 残差连接
- 训练速度更快，并行性更好

改进：
1. 支持2个标的并行训练
2. 日志存储到log文件夹
3. 训练完成后生成30个标的汇总时序图
4. 生成性能指标汇总表格
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
# 1. Dataset (与baseline相同)
# ============================================================================

class SingleSymbolLOBDataset(Dataset):
    """单个标的的LOB数据集"""
    
    def __init__(self, file_path, start_ratio=0.0, end_ratio=1.0, 
                 sequence_length=100, scaler=None, target_scaler=None,
                 feature_dim=40, target_col=40, fit_scaler=False):
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.target_col = target_col
        
        data = np.load(file_path, mmap_mode='r')
        n = len(data)
        start_idx = int(n * start_ratio)
        end_idx = int(n * end_ratio)
        segment = data[start_idx:end_idx]
        
        if fit_scaler:
            features = segment[:, :feature_dim]
            targets = segment[:, target_col]
            
            valid_mask = np.isfinite(features).all(axis=1) & np.isfinite(targets)
            features_clean = features[valid_mask]
            targets_clean = targets[valid_mask]
            
            if scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(features_clean)
            else:
                self.scaler = scaler
            
            targets_log = np.log1p(targets_clean)
            targets_log = targets_log[np.isfinite(targets_log)]
            
            if target_scaler is None:
                self.target_scaler = StandardScaler()
                self.target_scaler.fit(targets_log.reshape(-1, 1))
            else:
                self.target_scaler = target_scaler
        else:
            self.scaler = scaler
            self.target_scaler = target_scaler
        
        self.data = segment
        self.n_samples = len(self.data) - self.sequence_length
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        window = self.data[idx:idx + self.sequence_length, :self.feature_dim].copy()
        target = self.data[idx + self.sequence_length - 1, self.target_col].copy()
        
        if not np.isfinite(window).all() or not np.isfinite(target):
            return torch.zeros(1, self.sequence_length, self.feature_dim), torch.zeros(1)
        
        if self.scaler is not None:
            window = self.scaler.transform(window)
        
        target = np.log1p(target) * 10000  # BPS
        
        if not np.isfinite(target):
            return torch.zeros(1, self.sequence_length, self.feature_dim), torch.zeros(1)
        
        x = torch.FloatTensor(window).unsqueeze(0)
        y = torch.FloatTensor([target])
        
        return x, y


# ============================================================================
# 2. TCN Model (TCN architecture)
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
        # padding = (kernel_size - 1) * dilation，全部填充在左侧
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
        self.relu = nn.LeakyReLU(negative_slope=0.01)  # 使用LeakyReLU防止Dead ReLU
        
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
        # 左侧填充，右侧不填充，确保因果性
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
    
    特点:
    - 多层 TCN 块堆叠
    - 指数增长的膨胀率 (1, 2, 4, 8, ...)
    - 感受野随层数指数增长
    - 完全并行处理，训练速度快
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


class DeepLOB_TCN(nn.Module):
    """
    DeepLOB-TCN 模型架构 (with Batch Normalization)
    
    架构流程:
    1. CNN 特征提取 (保留 DeepLOB 的 CNN 部分 + BN)
       - 3个卷积块 + BN
       - Inception 模块 + BN
    2. TCN 时序建模 (替换 LSTM)
       - 多层 TCN 块
       - 因果卷积 + 膨胀卷积 + 残差连接
    3. 全连接层预测 + BN
    
    优势:
    - 保留了 DeepLOB 的 CNN 特征提取能力
    - 用 TCN 替换 LSTM，训练速度更快
    - 更好的并行性和长距离依赖建模
    - BN层防止梯度消失/爆炸，加速收敛
    """
    def __init__(self, input_channels=1, num_classes=1, dropout=0.3):
        super(DeepLOB_TCN, self).__init__()
        
        # ==================== CNN 特征提取部分 + BN ====================
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
        
        # ==================== TCN 时序建模部分 (替换 LSTM) ====================
        # 输入维度: 256 (Inception 输出: 64*4=256)
        # TCN 配置: 4层，每层64通道
        # 膨胀率: 1, 2, 4, 8
        self.tcn = TCN(
            input_size=256,
            num_channels=[64, 64, 64, 64],  # 4层TCN，每层64通道
            kernel_size=3,
            dropout=dropout
        )
        
        # ==================== 全连接层 + BN ====================
        # TCN 输出: 64 通道
        self.fc1 = nn.Linear(64, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # ==================== CNN 特征提取 + BN + LeakyReLU ====================
        # Input: (batch, 1, seq_len, features)
        x = F.leaky_relu(self.bn1a(self.conv1a(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn1b(self.conv1b(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn1c(self.conv1c(x)), negative_slope=0.01)
        
        x = F.leaky_relu(self.bn2a(self.conv2a(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2b(self.conv2b(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2c(self.conv2c(x)), negative_slope=0.01)
        
        x = F.leaky_relu(self.bn3a(self.conv3a(x)), negative_slope=0.01)
        
        # Inception module with BN + LeakyReLU
        branch1 = F.leaky_relu(self.bn_inc1(self.inception1(x)), negative_slope=0.01)
        
        branch2 = F.leaky_relu(self.inception2a(x), negative_slope=0.01)
        branch2 = F.leaky_relu(self.bn_inc2(self.inception2b(branch2)), negative_slope=0.01)
        
        branch3 = F.leaky_relu(self.inception3a(x), negative_slope=0.01)
        branch3 = F.leaky_relu(self.bn_inc3(self.inception3b(branch3)), negative_slope=0.01)
        
        branch4 = self.inception4(x)
        branch4 = F.leaky_relu(self.bn_inc4(self.inception4_conv(branch4)), negative_slope=0.01)
        
        x = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        # Output: (batch, 256, seq_len, 1)
        
        # Reshape for TCN: (batch, 256, seq_len)
        x = x.squeeze(-1).permute(0, 2, 1).permute(0, 2, 1)
        
        # ==================== TCN 时序建模 (替换 LSTM) ====================
        x = self.tcn(x)  # (batch, 64, seq_len)
        
        # 取最后时间步的输出
        x = x[:, :, -1]  # (batch, 64)
        
        # ==================== 全连接层 + BN + LeakyReLU ====================
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# 3. Training Function (单进程版本，用于并行调用)
# ============================================================================

def train_single_symbol_worker(args):
    """单标的训练函数（用于并行调用）"""
    symbol, data_dir, output_dir, log_dir, config, gpu_id = args
    
    # 设置当前进程使用的GPU
    import os
    # 对于单GPU并行，两个进程都使用GPU 0
    # 对于多GPU，每个进程使用不同的GPU
    if gpu_id >= torch.cuda.device_count():
        gpu_id = 0  # 如果GPU ID超出范围，使用GPU 0
    
    # 设置设备
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
        log_print(f"🚀 开始训练: {symbol} (GPU {gpu_id}) - DeepLOB-TCN")
        log_print(f"{'='*80}")
        
        start_time = time.time()
        
        # 文件路径
        data_file = Path(data_dir) / f"{symbol}_20250801_20250810.npy"
        if not data_file.exists():
            log_print(f"   ❌ 文件不存在: {data_file}")
            return None
        
        # 创建数据集
        log_print(f"   📂 加载数据...")
        train_dataset = SingleSymbolLOBDataset(
            data_file, start_ratio=0.0, end_ratio=0.6,
            sequence_length=config['sequence_length'], fit_scaler=True
        )
        
        val_dataset = SingleSymbolLOBDataset(
            data_file, start_ratio=0.6, end_ratio=0.8,
            sequence_length=config['sequence_length'],
            scaler=train_dataset.scaler,
            target_scaler=train_dataset.target_scaler,
            fit_scaler=False
        )
        
        test_dataset = SingleSymbolLOBDataset(
            data_file, start_ratio=0.8, end_ratio=1.0,
            sequence_length=config['sequence_length'],
            scaler=train_dataset.scaler,
            target_scaler=train_dataset.target_scaler,
            fit_scaler=False
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
        
        # 创建模型 - 使用TCN架构
        log_print(f"   🏗️  创建模型... (DeepLOB-TCN)")
        model = DeepLOB_TCN(
            input_channels=1, num_classes=1, dropout=config['dropout']
        ).to(device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.HuberLoss(delta=1.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # 训练循环
        log_print(f"   🏋️  开始训练...")
        best_val_loss = float('inf')
        patience_counter = 0
        epoch_times = []
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epoch_times': []
        }
        
        for epoch in range(config['num_epochs']):
            epoch_start = time.time()
            
            # 训练
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            history['epoch_times'].append(epoch_time)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': train_dataset.scaler,
                    'target_scaler': train_dataset.target_scaler,
                    'config': config
                }, output_dir / f"{symbol}_best_model.pth")
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log_print(f"      Epoch {epoch+1:2d}/{config['num_epochs']} | "
                          f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                          f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
            
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
        
        test_preds = []
        test_targets = []
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                test_loss += loss.item()
                
                test_preds.append(outputs.cpu().numpy())
                test_targets.append(batch_y.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_preds = np.concatenate(test_preds).flatten()
        test_targets = np.concatenate(test_targets).flatten()
        
        # 计算指标
        mae = mean_absolute_error(test_targets, test_preds)
        rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        r2 = r2_score(test_targets, test_preds)
        corr = np.corrcoef(test_targets, test_preds)[0, 1] if len(test_targets) > 1 else 0.0
        
        training_time = time.time() - start_time
        
        log_print(f"\n   ✅ 训练完成!")
        log_print(f"      Test Loss: {test_loss:.6f}")
        log_print(f"      MAE:       {mae:.6f}")
        log_print(f"      RMSE:      {rmse:.6f}")
        log_print(f"      R²:        {r2:.6f}")
        log_print(f"      Corr:      {corr:.6f}")
        log_print(f"      Time:      {training_time/60:.2f} min")
        
        # 保存结果
        result = {
            'symbol': symbol,
            'test_loss': float(test_loss),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'correlation': float(corr),
            'best_val_loss': float(best_val_loss),
            'training_time_minutes': float(training_time / 60),
            'epochs_trained': len(history['train_loss']),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        }
        
        # 保存历史
        with open(output_dir / f"{symbol}_history.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        # 保存预测
        np.savez(
            output_dir / f"{symbol}_predictions.npz",
            predictions=test_preds,
            targets=test_targets
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
# 4. Plotting Functions
# ============================================================================

def plot_timeseries_comparison(y_true, y_pred, symbol, output_dir):
    """绘制单个标的的时序对比图"""
    window_size = min(1000, len(y_true))
    min_zero_ratio = 1.0
    best_start = 0
    
    for i in range(len(y_true) - window_size + 1):
        window = y_true[i:i+window_size]
        zero_ratio = np.sum(np.abs(window) < 0.01) / window_size
        if zero_ratio < min_zero_ratio:
            min_zero_ratio = zero_ratio
            best_start = i
    
    if min_zero_ratio > 0.5:
        non_zero_mask = np.abs(y_true) > 0.01
        non_zero_indices = np.where(non_zero_mask)[0]
        
        if len(non_zero_indices) >= window_size:
            np.random.seed(42)
            selected_indices = np.random.choice(non_zero_indices, window_size, replace=False)
            selected_indices = np.sort(selected_indices)
            y_true_plot = y_true[selected_indices]
            y_pred_plot = y_pred[selected_indices]
            indices_plot = np.arange(len(y_true_plot))
        else:
            y_true_plot = y_true[non_zero_indices]
            y_pred_plot = y_pred[non_zero_indices]
            indices_plot = np.arange(len(y_true_plot))
    else:
        y_true_plot = y_true[best_start:best_start+window_size]
        y_pred_plot = y_pred[best_start:best_start+window_size]
        indices_plot = np.arange(len(y_true_plot))
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(indices_plot, y_true_plot, color='#2E86AB', linewidth=1.5, 
           label='True', alpha=0.8, marker='o', markersize=2, markevery=max(1, len(indices_plot)//20))
    ax.plot(indices_plot, y_pred_plot, color='#E63946', linewidth=1.5, 
           label='Pred', alpha=0.8, marker='s', markersize=2, markevery=max(1, len(indices_plot)//20))
    ax.axhline(0, color='black', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Index', fontsize=10)
    ax.set_ylabel('Return (BPS)', fontsize=10)
    ax.set_title(f'{symbol}', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(0.02, 0.98, f'Corr: {corr:.3f}', transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{symbol}_timeseries.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_all_symbols_timeseries(output_dir, symbols, image_dir):
    """绘制所有30个标的的汇总时序图"""
    print(f"\n{'='*80}")
    print(f"📊 生成30个标的汇总时序图 (TCN)")
    print(f"{'='*80}")
    
    # 计算布局：6行5列
    n_symbols = len(symbols)
    n_rows = 6
    n_cols = 5
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24))
    axes = axes.flatten()
    
    plot_count = 0
    
    for i, symbol in enumerate(symbols):
        pred_file = output_dir / f"{symbol}_predictions.npz"
        
        if not pred_file.exists():
            axes[i].axis('off')
            axes[i].text(0.5, 0.5, f'{symbol}\nNo Data', 
                        ha='center', va='center', fontsize=10)
            continue
        
        try:
            data = np.load(pred_file)
            y_true = data['targets']
            y_pred = data['predictions']
            
            # 选择绘图区域（零值较少的区域）
            window_size = min(500, len(y_true))  # 每个标的只显示500个样本
            min_zero_ratio = 1.0
            best_start = 0
            
            for j in range(len(y_true) - window_size + 1):
                window = y_true[j:j+window_size]
                zero_ratio = np.sum(np.abs(window) < 0.01) / window_size
                if zero_ratio < min_zero_ratio:
                    min_zero_ratio = zero_ratio
                    best_start = j
            
            if min_zero_ratio > 0.5:
                non_zero_mask = np.abs(y_true) > 0.01
                non_zero_indices = np.where(non_zero_mask)[0]
                if len(non_zero_indices) >= window_size:
                    np.random.seed(42)
                    selected_indices = np.random.choice(non_zero_indices, window_size, replace=False)
                    selected_indices = np.sort(selected_indices)
                    y_true_plot = y_true[selected_indices]
                    y_pred_plot = y_pred[selected_indices]
                    indices_plot = np.arange(len(y_true_plot))
                else:
                    y_true_plot = y_true[non_zero_indices]
                    y_pred_plot = y_pred[non_zero_indices]
                    indices_plot = np.arange(len(y_true_plot))
            else:
                y_true_plot = y_true[best_start:best_start+window_size]
                y_pred_plot = y_pred[best_start:best_start+window_size]
                indices_plot = np.arange(len(y_true_plot))
            
            # 绘制
            ax = axes[i]
            ax.plot(indices_plot, y_true_plot, color='#2E86AB', linewidth=1.0, 
                   label='True', alpha=0.7, markersize=1)
            ax.plot(indices_plot, y_pred_plot, color='#E63946', linewidth=1.0, 
                   label='Pred', alpha=0.7, markersize=1)
            ax.axhline(0, color='black', linestyle=':', linewidth=0.5, alpha=0.3)
            
            # 计算相关性
            corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
            
            ax.set_title(f'{symbol} (Corr: {corr:.3f})', fontsize=9, fontweight='bold', pad=3)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
            
            plot_count += 1
            
        except Exception as e:
            axes[i].axis('off')
            axes[i].text(0.5, 0.5, f'{symbol}\nError: {str(e)[:20]}', 
                        ha='center', va='center', fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_symbols, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Time Series Comparison - All 30 Symbols (TCN)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = image_dir / 'all_symbols_timeseries_summary_tcn.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ 汇总时序图已保存: {save_path}")
    print(f"      成功绘制: {plot_count}/{n_symbols} 个标的")


# ============================================================================
# 5. Main Function
# ============================================================================

def main():
    print("="*80)
    print("🚀 DeepLOB-TCN Per-Symbol Training (Parallel)")
    print("="*80)
    print("\n策略: 每个标的独立训练，支持多GPU并行")
    print("数据: 10天数据 (2025-08-01 to 2025-08-10)")
    print("模型: DeepLOB-TCN with BatchNorm + LeakyReLU + HuberLoss")
    
    # 配置
    config = {
        'sequence_length': 100,
        'batch_size': 2048,
        'num_workers': 2,
        'learning_rate': 0.001,
        'num_epochs': 20,
        'dropout': 0.3,
        'early_stopping_patience': 5
    }
    
    # 路径 - 所有输出保存到5_models文件夹
    data_dir = Path('data_250801_250810')
    output_dir = Path('5_models')
    log_dir = output_dir / 'log'
    image_dir = output_dir / 'image'
    
    output_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    
    # 读取所有标的
    metadata_file = data_dir / 'metadata.csv'
    if metadata_file.exists():
        metadata = pd.read_csv(metadata_file)
        symbols = metadata['symbol'].tolist()
    else:
        npy_files = sorted(data_dir.glob('*_20250801_20250810.npy'))
        symbols = [f.stem.replace('_20250801_20250810', '') for f in npy_files]
    
    print(f"\n找到 {len(symbols)} 个标的")
    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"日志目录: {log_dir}")
    print(f"输出目录: {output_dir}")
    
    # ✅ 断点恢复：检查已完成的标的
    completed_symbols = []
    pending_symbols = []
    for symbol in symbols:
        model_file = output_dir / f"{symbol}_best_model.pth"
        if model_file.exists():
            completed_symbols.append(symbol)
        else:
            pending_symbols.append(symbol)
    
    print(f"\n断点恢复:")
    print(f"   ✅ 已完成: {len(completed_symbols)} 个")
    print(f"   ⏳ 待训练: {len(pending_symbols)} 个")
    
    if completed_symbols:
        print(f"\n已完成的标的: {', '.join(completed_symbols[:10])}" +
              (f"... (共{len(completed_symbols)}个)" if len(completed_symbols) > 10 else ""))
    
    if pending_symbols:
        print(f"\n待训练的标的: {', '.join(pending_symbols)}")
        symbols = pending_symbols  # 只训练未完成的标的
    else:
        print("\n✅ 所有标的已训练完成！")
        symbols = []  # 不需要训练，直接生成汇总
    
    # 准备并行训练参数
    n_gpus = torch.cuda.device_count()
    print(f"\n可用GPU数量: {n_gpus}")
    
    # 🚀 支持每张GPU并行2个模型
    if n_gpus >= 4:
        max_workers = 8  # 4张GPU × 2个进程/GPU = 8个并行
        print(f"✅ 将使用 {max_workers} 个进程并行训练 (每张GPU 2个进程)")
    elif n_gpus >= 2:
        max_workers = n_gpus * 2  # 每张GPU 2个进程
        print(f"✅ 将使用 {max_workers} 个进程并行训练 (每张GPU 2个进程)")
    else:
        max_workers = 2  # 单GPU时使用2个进程共享
        print(f"✅ 将使用 {max_workers} 个进程共享1个GPU并行训练")
    
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   预计每张GPU显存使用: ~14GB × 2 = ~28GB (安全范围内)")
    
    # 准备任务列表 - 每张GPU分配2个任务
    tasks = []
    for i, symbol in enumerate(symbols):
        gpu_id = (i // 2) % n_gpus if n_gpus > 0 else 0  # 每2个任务分配到一个GPU
        tasks.append((symbol, data_dir, output_dir, log_dir, config, gpu_id))
    
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
    print(f"📊 训练汇总 - 加载所有已完成的标的")
    print(f"{'='*80}")
    
    # 读取所有标的（包括之前完成的和刚完成的）
    metadata = pd.read_csv(metadata_file)
    all_symbols_list = metadata['symbol'].tolist()
    
    all_results = []
    for symbol in all_symbols_list:
        pred_file = output_dir / f"{symbol}_predictions.npz"
        history_file = output_dir / f"{symbol}_history.pkl"
        
        if pred_file.exists():
            try:
                data = np.load(pred_file)
                y_true = data['targets']
                y_pred = data['predictions']
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                corr = np.corrcoef(y_true, y_pred)[0, 1]
                
                # 尝试从历史文件读取训练信息
                epochs_trained = 0
                training_time_minutes = 0.0
                best_val_loss = 0.0
                
                if history_file.exists():
                    try:
                        with open(history_file, 'rb') as f:
                            history = pickle.load(f)
                            epochs_trained = len(history.get('train_loss', []))
                            if 'epoch_times' in history and history['epoch_times']:
                                training_time_minutes = sum(history['epoch_times']) / 60.0
                            if 'val_loss' in history and history['val_loss']:
                                best_val_loss = min(history['val_loss'])
                    except:
                        pass
                
                all_results.append({
                    'symbol': symbol,
                    'test_loss': 0.0,  # 占位符
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'correlation': float(corr),
                    'best_val_loss': float(best_val_loss),
                    'training_time_minutes': float(training_time_minutes),
                    'epochs_trained': int(epochs_trained),
                    'train_samples': len(y_true),
                    'val_samples': 0,
                    'test_samples': len(y_true)
                })
            except Exception as e:
                print(f"⚠️  加载 {symbol} 时出错: {e}")
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values('correlation', ascending=False)
        
        # 保存CSV
        csv_file = output_dir / 'training_summary_tcn.csv'
        df_results.to_csv(csv_file, index=False)
        print(f"\n✅ 汇总表格已保存: {csv_file}")
        
        # 生成Markdown表格
        md_file = output_dir / 'training_summary_tcn.md'
        with open(md_file, 'w') as f:
            f.write("# DeepLOB-TCN Per-Symbol Training Summary\n\n")
            f.write(f"**训练日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**数据**: 10天数据 (2025-08-01 to 2025-08-10)\n\n")
            f.write(f"**成功训练**: {len(all_results)}/{len(all_symbols_list)} 个标的\n\n")
            
            f.write("## 性能指标汇总\n\n")
            f.write("| 排名 | 标的 | MAE | RMSE | R² | Correlation | 训练时间(分钟) | Epochs |\n")
            f.write("|------|------|-----|------|----|-------------|----------------|--------|\n")
            
            for idx, row in df_results.iterrows():
                f.write(f"| {idx+1} | {row['symbol']} | {row['mae']:.6f} | {row['rmse']:.6f} | "
                       f"{row['r2']:.6f} | **{row['correlation']:.6f}** | "
                       f"{row['training_time_minutes']:.2f} | {row['epochs_trained']} |\n")
            
            f.write("\n## 统计摘要\n\n")
            f.write(f"- **平均 MAE**: {df_results['mae'].mean():.6f}\n")
            f.write(f"- **平均 RMSE**: {df_results['rmse'].mean():.6f}\n")
            f.write(f"- **平均 R²**: {df_results['r2'].mean():.6f}\n")
            f.write(f"- **平均 Correlation**: {df_results['correlation'].mean():.6f}\n")
            f.write(f"- **总训练时间**: {df_results['training_time_minutes'].sum():.2f} 分钟\n")
            f.write(f"- **平均训练时间**: {df_results['training_time_minutes'].mean():.2f} 分钟\n")
        
        print(f"✅ Markdown表格已保存: {md_file}")
        
        # 打印统计
        print(f"\n📈 性能统计:")
        print(f"   成功训练: {len(all_results)}/{len(all_symbols_list)}")
        print(f"   平均 MAE:  {df_results['mae'].mean():.6f}")
        print(f"   平均 Corr: {df_results['correlation'].mean():.6f}")
        print(f"   平均 R²:   {df_results['r2'].mean():.6f}")
    
        print(f"\n🏆 Top 10 标的 (按Correlation排序):")
        print(df_results[['symbol', 'correlation', 'mae', 'r2']].head(10).to_string(index=False))
        
        # 打印完整结果表格
        print(f"\n{'='*80}")
        print(f"📊 完整训练结果表格")
        print(f"{'='*80}\n")
        
        # 格式化输出所有结果
        print(f"{'排名':<6}{'标的':<15}{'MAE':<12}{'RMSE':<12}{'R²':<12}{'Correlation':<15}{'训练时间':<12}")
        print(f"{'-'*85}")
        for idx, row in df_results.iterrows():
            print(f"{idx+1:<6}{row['symbol']:<15}{row['mae']:<12.6f}{row['rmse']:<12.6f}"
                  f"{row['r2']:<12.6f}{row['correlation']:<15.6f}{row['training_time_minutes']:<12.2f}")
        
        print(f"\n{'='*80}")
        print(f"📈 统计摘要")
        print(f"{'='*80}")
        print(f"  平均 MAE:           {df_results['mae'].mean():.6f}")
        print(f"  平均 RMSE:          {df_results['rmse'].mean():.6f}")
        print(f"  平均 R²:            {df_results['r2'].mean():.6f}")
        print(f"  平均 Correlation:   {df_results['correlation'].mean():.6f}")
        print(f"  最佳 Correlation:   {df_results['correlation'].max():.6f} ({df_results.loc[df_results['correlation'].idxmax(), 'symbol']})")
        print(f"  最差 Correlation:   {df_results['correlation'].min():.6f} ({df_results.loc[df_results['correlation'].idxmin(), 'symbol']})")
        print(f"  总训练时间:         {df_results['training_time_minutes'].sum():.2f} 分钟")
        print(f"  平均训练时间/标的:  {df_results['training_time_minutes'].mean():.2f} 分钟")
        
        # 生成汇总时序图
        print(f"\n{'='*80}")
        print(f"🎨 生成可视化图表")
        print(f"{'='*80}")
        plot_all_symbols_timeseries(output_dir, all_symbols_list, image_dir)
        
        # 输出文件位置
        print(f"\n{'='*80}")
        print(f"📁 输出文件位置")
        print(f"{'='*80}")
        print(f"  模型文件:     {output_dir.absolute()}/")
        print(f"  CSV汇总:      {csv_file.absolute()}")
        print(f"  MD汇总:       {md_file.absolute()}")
        print(f"  日志文件:     {log_dir.absolute()}/")
        print(f"  图表文件:     {image_dir.absolute()}/")
        
    if failed_symbols:
        print(f"\n❌ 失败的标的: {failed_symbols}")
    
    print(f"\n{'='*80}")
    print(f"✅ 所有训练完成!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
