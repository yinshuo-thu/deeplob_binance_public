#!/usr/bin/env python3
"""
DeepLOB Per-Symbol Training - EMA平滑目标变量版本 (LSTM架构)

主要改进:
- 对目标变量y进行EMA平滑处理，减少噪声
- 使用原始DeepLOB的LSTM架构（而非TCN）
- 保留 DeepLOB 的 CNN 特征提取
- LSTM 时序建模

EMA平滑:
- 对return_10s进行指数移动平均平滑
- alpha参数控制平滑程度（默认0.2，即20%新值，80%历史值）
- 平滑后的目标变量更稳定，有助于模型学习趋势
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
# 1. Dataset with EMA Smoothing
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


class SingleSymbolLOBDataset(Dataset):
    """单个标的的LOB数据集 - 带EMA平滑目标变量"""
    
    def __init__(self, file_path, start_ratio=0.0, end_ratio=1.0, 
                 sequence_length=100, scaler=None, target_scaler=None,
                 feature_dim=40, target_col=40, fit_scaler=False,
                 ema_alpha=0.2):
        """
        Args:
            ema_alpha: EMA平滑因子，默认0.2
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.target_col = target_col
        self.ema_alpha = ema_alpha
        
        data = np.load(file_path, mmap_mode='r')
        n = len(data)
        start_idx = int(n * start_ratio)
        end_idx = int(n * end_ratio)
        segment = data[start_idx:end_idx]
        
        # ✅ 关键改进：对目标变量进行EMA平滑
        raw_targets = segment[:, target_col].copy()
        # 处理无效值
        valid_mask = np.isfinite(raw_targets)
        if valid_mask.sum() > 0:
            # 对有效值进行EMA平滑
            smoothed_targets = np.zeros_like(raw_targets)
            valid_indices = np.where(valid_mask)[0]
            valid_values = raw_targets[valid_indices]
            
            # 应用EMA平滑
            smoothed_valid = apply_ema_smoothing(valid_values, alpha=ema_alpha)
            smoothed_targets[valid_indices] = smoothed_valid
            
            # 无效值保持为0或NaN
            smoothed_targets[~valid_mask] = raw_targets[~valid_mask]
            
            # 将平滑后的目标变量替换原始值
            segment_smoothed = segment.copy()
            segment_smoothed[:, target_col] = smoothed_targets
        else:
            segment_smoothed = segment
        
        if fit_scaler:
            features = segment_smoothed[:, :feature_dim]
            targets = segment_smoothed[:, target_col]
            
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
        
        self.data = segment_smoothed
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
# 2. DeepLOB Model with LSTM (原始架构)
# ============================================================================

class DeepLOB(nn.Module):
    """
    DeepLOB 模型架构 (with Batch Normalization and LSTM)
    
    架构流程:
    1. CNN 特征提取 (保留 DeepLOB 的 CNN 部分 + BN)
    2. LSTM 时序建模 (原始DeepLOB架构)
    3. 全连接层预测 + BN
    """
    def __init__(self, input_channels=1, num_classes=1, dropout=0.3):
        super(DeepLOB, self).__init__()
        
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
        
        # ==================== LSTM 时序建模部分 (原始DeepLOB架构) ====================
        self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True)
        
        # ==================== 全连接层 + BN ====================
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
        
        # Reshape for LSTM: (batch, seq_len, 256)
        x = x.squeeze(-1).permute(0, 2, 1)
        
        # ==================== LSTM 时序建模 (原始DeepLOB架构) ====================
        x, _ = self.lstm(x)  # (batch, seq_len, 64)
        
        # 取最后时间步的输出
        x = x[:, -1, :]  # (batch, 64)
        
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
        log_print(f"🚀 开始训练: {symbol} (GPU {gpu_id}) - DeepLOB with EMA Smoothing")
        log_print(f"   EMA Alpha: {config.get('ema_alpha', 0.2)}")
        log_print(f"{'='*80}")
        
        start_time = time.time()
        
        # 文件路径
        data_file = Path(data_dir) / f"{symbol}_20250801_20250810.npy"
        if not data_file.exists():
            log_print(f"   ❌ 文件不存在: {data_file}")
            return None
        
        # 创建数据集 - 使用EMA平滑
        log_print(f"   📂 加载数据 (EMA平滑: alpha={config.get('ema_alpha', 0.2)})...")
        train_dataset = SingleSymbolLOBDataset(
            data_file, start_ratio=0.0, end_ratio=0.6,
            sequence_length=config['sequence_length'], 
            fit_scaler=True,
            ema_alpha=config.get('ema_alpha', 0.2)
        )
        
        val_dataset = SingleSymbolLOBDataset(
            data_file, start_ratio=0.6, end_ratio=0.8,
            sequence_length=config['sequence_length'],
            scaler=train_dataset.scaler,
            target_scaler=train_dataset.target_scaler,
            fit_scaler=False,
            ema_alpha=config.get('ema_alpha', 0.2)
        )
        
        test_dataset = SingleSymbolLOBDataset(
            data_file, start_ratio=0.8, end_ratio=1.0,
            sequence_length=config['sequence_length'],
            scaler=train_dataset.scaler,
            target_scaler=train_dataset.target_scaler,
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
        
        # 创建模型 - 使用LSTM架构
        log_print(f"   🏗️  创建模型... (DeepLOB with LSTM)")
        model = DeepLOB(
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
            'test_samples': len(test_dataset),
            'ema_alpha': config.get('ema_alpha', 0.2)
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
# 4. Report Generation Functions
# ============================================================================

def generate_final_report(df_results, output_dir, image_dir):
    """生成最终报告：包含核心图表和表格"""
    print("\n生成报告图表和表格...")
    
    # 重命名列以匹配报告生成函数
    df = df_results.copy()
    df = df.rename(columns={
        'symbol': 'Symbol',
        'mae': 'MAE',
        'rmse': 'RMSE',
        'r2': 'R²',
        'correlation': 'Correlation'
    })
    df = df.sort_values('Correlation', ascending=False).reset_index(drop=True)
    
    # ============================================================================
    # Figure 1: Core Performance (2x2 layout)
    # ============================================================================
    print("  1. 创建核心性能图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1a) Correlation by Symbol
    ax = axes[0, 0]
    colors = ['#27ae60' if x > 0.15 else '#3498db' if x > 0.05 else '#e74c3c' for x in df['Correlation']]
    bars = ax.barh(range(len(df)), df['Correlation'], color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Symbol'], fontsize=9)
    ax.set_xlabel('Correlation Coefficient', fontweight='bold', fontsize=12)
    ax.set_title('(A) Prediction Correlation by Symbol', fontsize=14, fontweight='bold', pad=12)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.2)
    ax.axvline(df['Correlation'].mean(), color='red', linestyle='--', linewidth=2.5, 
                label=f'Mean: {df["Correlation"].mean():.3f}', alpha=0.8)
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.8)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    if len(df) > 0:
        ax.set_xlim(-0.05, max(0.30, df['Correlation'].max() * 1.2))
    
    # 1b) MAE vs Correlation (Scatter with size by RMSE)
    ax = axes[0, 1]
    scatter = ax.scatter(df['Correlation'], df['MAE'], s=df['RMSE']*30, 
                        c=df['Correlation'], cmap='RdYlGn', alpha=0.7, 
                        edgecolor='black', linewidth=1.2)
    # Annotate top performers
    for idx, row in df.head(min(3, len(df))).iterrows():
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
    df_sorted_mae = df.sort_values('MAE')
    colors_mae = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted_mae)))
    ax.barh(range(len(df_sorted_mae)), df_sorted_mae['MAE'], 
           color=colors_mae, edgecolor='black', linewidth=0.6, alpha=0.85)
    ax.set_yticks(range(len(df_sorted_mae)))
    ax.set_yticklabels(df_sorted_mae['Symbol'], fontsize=9)
    ax.set_xlabel('Mean Absolute Error (BPS)', fontweight='bold', fontsize=12)
    ax.set_title('(C) MAE by Symbol (Sorted)', fontsize=14, fontweight='bold', pad=12)
    ax.axvline(df['MAE'].mean(), color='red', linestyle='--', linewidth=2.5, 
              label=f'Mean: {df["MAE"].mean():.2f}', alpha=0.8)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.8)
    
    # 1d) Correlation vs MAE Relationship
    ax = axes[1, 1]
    ax.scatter(df['Correlation'], df['MAE'], s=150, alpha=0.7, 
              c=df['Correlation'], cmap='RdYlGn', edgecolor='black', linewidth=1.5)
    
    # Add regression line
    if len(df) > 1:
        z = np.polyfit(df['Correlation'], df['MAE'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['Correlation'].min(), df['Correlation'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=3, alpha=0.8, 
               label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Annotate top performers
    for idx, row in df.head(min(3, len(df))).iterrows():
        ax.annotate(row['Symbol'], (row['Correlation'], row['MAE']), 
                   fontsize=10, fontweight='bold', ha='right', va='bottom',
                   xytext=(-8, 8), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Correlation Coefficient', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (BPS)', fontweight='bold', fontsize=12)
    ax.set_title('(D) Correlation vs MAE Relationship', fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    
    plt.suptitle('DeepLOB Model with EMA Smoothing - Core Performance Metrics', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(image_dir / 'fig1_core_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("     ✓ Saved: fig1_core_performance.png")
    
    # ============================================================================
    # Figure 2: Summary Statistics (Table Visualization)
    # ============================================================================
    print("  2. 创建汇总统计表格...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    
    # Calculate statistics
    summary_data = {
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Correlation': [
            df['Correlation'].mean(),
            df['Correlation'].median(),
            df['Correlation'].std(),
            df['Correlation'].min(),
            df['Correlation'].max()
        ],
        'MAE (BPS)': [
            df['MAE'].mean(),
            df['MAE'].median(),
            df['MAE'].std(),
            df['MAE'].min(),
            df['MAE'].max()
        ],
        'RMSE (BPS)': [
            df['RMSE'].mean(),
            df['RMSE'].median(),
            df['RMSE'].std(),
            df['RMSE'].min(),
            df['RMSE'].max()
        ],
        'R² Score': [
            df['R²'].mean(),
            df['R²'].median(),
            df['R²'].std(),
            df['R²'].min(),
            df['R²'].max()
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
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
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
            if j == 1:  # Correlation
                table[(i, j)].get_text().set_text(f'{val:.4f}')
            elif j == 2 or j == 3:  # MAE, RMSE
                table[(i, j)].get_text().set_text(f'{val:.2f}')
            else:  # R²
                table[(i, j)].get_text().set_text(f'{val:.4f}')
    
    ax.set_title('Summary Statistics', fontsize=18, fontweight='bold', pad=20)
    plt.savefig(image_dir / 'fig2_summary_statistics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("     ✓ Saved: fig2_summary_statistics.png")
    
    # ============================================================================
    # Figure 3: Top Performers (Table Visualization)
    # ============================================================================
    print("  3. 创建Top Performers表格...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axis('off')
    
    top_n = df.copy()
    top_n.insert(0, 'Rank', range(1, len(top_n) + 1))
    top_n_display = top_n[['Rank', 'Symbol', 'Correlation', 'MAE', 'RMSE', 'R²']].copy()
    
    # Format numbers
    top_n_display['Correlation'] = top_n_display['Correlation'].apply(lambda x: f'{x:.4f}')
    top_n_display['MAE'] = top_n_display['MAE'].apply(lambda x: f'{x:.2f}')
    top_n_display['RMSE'] = top_n_display['RMSE'].apply(lambda x: f'{x:.2f}')
    top_n_display['R²'] = top_n_display['R²'].apply(lambda x: f'{x:.4f}')
    
    # Create table
    table = ax.table(cellText=top_n_display.values,
                    colLabels=top_n_display.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
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
    
    ax.set_title(f'Top {len(top_n)} Performers by Correlation', fontsize=18, fontweight='bold', pad=20)
    plt.savefig(image_dir / 'fig3_top_performers.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("     ✓ Saved: fig3_top_performers.png")
    
    # Save CSV tables
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False, float_format='%.4f')
    top_n[['Rank', 'Symbol', 'Correlation', 'MAE', 'RMSE', 'R²']].to_csv(
        output_dir / 'top_performers.csv', index=False, float_format='%.4f')
    print("     ✓ Saved: summary_statistics.csv")
    print("     ✓ Saved: top_performers.csv")
    
    print("\n✅ 报告图表和表格生成完成！")


# ============================================================================
# 4. Main Function
# ============================================================================

def main():
    print("="*80)
    print("🚀 DeepLOB Per-Symbol Training with EMA Smoothing (LSTM Architecture)")
    print("="*80)
    print("\n策略: 每个标的独立训练，支持多GPU并行")
    print("数据: 10天数据 (2025-08-01 to 2025-08-10)")
    print("模型: DeepLOB with LSTM + BatchNorm + LeakyReLU + HuberLoss")
    print("改进: EMA平滑目标变量 (减少噪声，提高训练稳定性)")
    
    # 配置
    config = {
        'sequence_length': 100,
        'batch_size': 2048,
        'num_workers': 2,
        'learning_rate': 0.001,
        'num_epochs': 20,
        'dropout': 0.3,
        'early_stopping_patience': 5,
        'ema_alpha': 0.2  # ✅ EMA平滑因子：0.2表示20%新值，80%历史值
    }
    
    # 路径 - 所有输出保存到6_models文件夹
    data_dir = Path('data_250801_250810')
    output_dir = Path('6_models')
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
    print(f"✅ EMA平滑因子: {config['ema_alpha']} (alpha越小越平滑)")
    
    if not target_symbols:
        print(f"\n✅ 所有标的已完成训练，无需训练")
        return
    
    print(f"\n🔄 开始训练 {len(target_symbols)} 个标的...")
    
    # 准备并行训练参数 - 2张卡，每张卡2个进程
    n_gpus = torch.cuda.device_count()
    print(f"\n可用GPU数量: {n_gpus}")
    
    if n_gpus >= 4:
        max_workers = 4  # 使用GPU 0、1、2、3，每张GPU 1个进程
        print(f"✅ 将使用 {max_workers} 个进程并行训练 (GPU 0、1、2、3，每张GPU 1个进程)")
    elif n_gpus >= 2:
        max_workers = min(4, n_gpus)  # 如果只有2张GPU，使用GPU 0、1，每张GPU 2个进程
        print(f"✅ 将使用 {max_workers} 个进程并行训练 (GPU 0、1，每张GPU {max_workers//2}个进程)")
    else:
        max_workers = min(4, n_gpus)
        print(f"✅ 将使用 {max_workers} 个进程共享1个GPU并行训练")
    
    print(f"   Batch Size: {config['batch_size']}")
    
    # 准备任务列表 - 使用GPU 0、1、2、3，每张卡1个进程，循环分配
    tasks = []
    for i, symbol in enumerate(target_symbols):
        # 使用GPU 0、1、2、3，每张卡1个进程，循环分配
        if n_gpus >= 4:
            # 循环分配到GPU 0、1、2、3，每张GPU一个进程
            gpu_id = i % 4
        elif n_gpus >= 2:
            # 如果只有2张GPU，循环分配到GPU 0、1
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
        df_results = df_results.sort_values('correlation', ascending=False)
    
        # 保存CSV
        csv_file = output_dir / 'training_summary_ema_lstm.csv'
        df_results.to_csv(csv_file, index=False)
        print(f"\n✅ 汇总表格已保存: {csv_file}")
        
        # 打印统计
        print(f"\n📈 性能统计:")
        print(f"   成功训练: {len(results)}/{len(target_symbols)}")
        if len(results) > 0:
            print(f"   平均 MAE:  {df_results['mae'].mean():.6f}")
            print(f"   平均 Corr: {df_results['correlation'].mean():.6f}")
            print(f"   平均 R²:   {df_results['r2'].mean():.6f}")
    
    if results:
        print(f"\n🏆 训练结果 (按Correlation排序):")
        print(df_results[['symbol', 'correlation', 'mae', 'r2']].to_string(index=False))
        
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

