#!/usr/bin/env python3
"""
DeepLOB 数据采集系统 - 极速版
优化策略：
1. 分离下载和处理阶段
2. 直接保存为npy格式（跳过Parquet）
3. 最大化并行处理（使用全部CPU核心）
4. 批量处理减少I/O
5. 内存优化的向量化处理
"""
import pandas as pd
import numpy as np
import gzip
import json
import os
from pathlib import Path
from datetime import datetime
from tardis_dev import datasets
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# 清除代理
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

print(f"🌐 直连模式: 不使用代理")

# ================= 配置加载 =================

def load_config(config_path='2_config.json'):
    """加载JSON配置文件"""
    # 如果使用相对路径，基于脚本所在目录
    script_dir = Path(__file__).parent
    config_file = script_dir / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

CONFIG = load_config()

# 获取脚本所在目录，用于处理相对路径
SCRIPT_DIR = Path(__file__).parent

API_KEY = CONFIG['api']['key']
EXCHANGE = CONFIG['api']['exchange']
DATA_TYPE = CONFIG['api']['data_type']
FROM_DATE = CONFIG['data']['from_date']
TO_DATE = CONFIG['data']['to_date']
NUM_SYMBOLS = CONFIG['data']['num_symbols']
# 处理相对路径：如果是相对路径，基于脚本目录
candidates_path = CONFIG['data']['candidates_file']
CANDIDATES_FILE = SCRIPT_DIR / candidates_path if not Path(candidates_path).is_absolute() else Path(candidates_path)
LOB_LEVELS = CONFIG['processing']['lob_levels']

# 最大化CPU利用
CPU_COUNT = mp.cpu_count()
DOWNLOAD_WORKERS = 1  # Tardis API限制
PROCESS_WORKERS = min(CPU_COUNT - 4, 96)  # 保留4核给系统

OUTPUT_DIR = Path(CONFIG['paths']['output_dir'])
DOWNLOAD_DIR = Path(CONFIG['paths']['download_dir'])
OUTPUT_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR.mkdir(exist_ok=True)

# ================= 核心处理函数 =================

def process_csv_ultra_fast(csv_path, lob_levels=10):
    """
    超快速处理单个CSV文件 - 直接输出npy格式
    返回: (N, 43)数组，包含40个特征 + timestamp + return_10s + return_60s
    """
    try:
        # 读取CSV
        with gzip.open(csv_path, 'rt') as f:
            df = pd.read_csv(f)
        
        if len(df) == 0:
            return None
        
        # 下采样到1秒
        df['timestamp_sec'] = pd.to_datetime(df['local_timestamp'], unit='us').dt.floor('1s')
        df_1s = df.groupby('timestamp_sec', as_index=False).last()
        
        n = len(df_1s)
        if n < 100:  # 需要至少100个样本
            return None
        
        # 预分配数组 (timestamp + 40 features + 2 returns)
        data = np.zeros((n, 43), dtype=np.float32)
        
        # 时间戳（转为Unix时间戳）
        data[:, 0] = df_1s['timestamp_sec'].astype(np.int64) // 10**9
        
        # 提取特征 - 向量化操作
        col_idx = 1
        for i in range(lob_levels):
            data[:, col_idx] = df_1s[f'asks[{i}].price'].values
            col_idx += 1
        for i in range(lob_levels):
            data[:, col_idx] = df_1s[f'asks[{i}].amount'].values
            col_idx += 1
        for i in range(lob_levels):
            data[:, col_idx] = df_1s[f'bids[{i}].price'].values
            col_idx += 1
        for i in range(lob_levels):
            data[:, col_idx] = df_1s[f'bids[{i}].amount'].values
            col_idx += 1
        
        # 计算中间价
        mid_price = (df_1s['asks[0].price'].values + df_1s['bids[0].price'].values) / 2.0
        
        # 计算return - 向量化
        # return_10s
        k_10s = 10
        returns_10s = np.zeros(n, dtype=np.float32)
        valid_10s = np.arange(n - k_10s)
        returns_10s[valid_10s] = (mid_price[valid_10s + k_10s] - mid_price[valid_10s]) / mid_price[valid_10s]
        data[:, 41] = returns_10s
        
        # return_60s
        k_60s = 60
        returns_60s = np.zeros(n, dtype=np.float32)
        valid_60s = np.arange(n - k_60s)
        returns_60s[valid_60s] = (mid_price[valid_60s + k_60s] - mid_price[valid_60s]) / mid_price[valid_60s]
        data[:, 42] = returns_60s
        
        # 移除最后60个样本
        data = data[:-k_60s]
        
        return data
        
    except Exception as e:
        return None


def process_symbol_batch(args):
    """批量处理单个标的的所有文件"""
    symbol, download_dir, output_dir, lob_levels = args
    start_time = time.time()
    
    try:
        # 查找该标的的所有CSV文件
        pattern = f"*_{symbol}.csv.gz"
        csv_files = sorted(download_dir.glob(pattern))
        
        if not csv_files:
            return {'symbol': symbol, 'status': 'no_files', 'time': 0}
        
        # 并行处理所有文件
        all_data = []
        for csv_file in csv_files:
            data = process_csv_ultra_fast(csv_file, lob_levels)
            if data is not None:
                all_data.append(data)
        
        if not all_data:
            return {'symbol': symbol, 'status': 'failed', 'time': time.time() - start_time}
        
        # 合并所有数据
        final_data = np.vstack(all_data)
        
        # 保存为npy
        date_range = f"{FROM_DATE.replace('-', '')}_{TO_DATE.replace('-', '')}"
        output_file = output_dir / f"{symbol}_{date_range}.npy"
        np.save(str(output_file), final_data)
        
        # 清理原始文件
        for csv_file in csv_files:
            csv_file.unlink()
        
        return {
            'symbol': symbol,
            'status': 'success',
            'samples': len(final_data),
            'files': len(csv_files),
            'size_mb': output_file.stat().st_size / (1024**2),
            'time': time.time() - start_time
        }
        
    except Exception as e:
        return {'symbol': symbol, 'status': 'error', 'error': str(e), 'time': time.time() - start_time}


# ================= 主程序 =================

def load_symbols():
    """读取标的列表"""
    symbols = []
    try:
        with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    symbols.append(line)
                    if len(symbols) >= NUM_SYMBOLS:
                        break
        return symbols
    except Exception as e:
        print(f"⚠️  读取标的文件失败: {e}")
        return []


def main():
    """主程序"""
    print("=" * 70)
    print("DeepLOB 数据采集系统 - 极速版")
    print("=" * 70)
    print(f"\n⏰ 程序开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  CPU核心数: {CPU_COUNT}")
    print(f"⚙️  下载并发: {DOWNLOAD_WORKERS}")
    print(f"⚙️  处理并发: {PROCESS_WORKERS}")
    print(f"📅 数据范围: {FROM_DATE} 至 {TO_DATE}")
    
    # 读取标的
    symbols = load_symbols()
    if not symbols:
        print("\n❌ 错误: 未找到可用的标的")
        return
    
    print(f"\n📊 处理标的: {len(symbols)} 个")
    for i, symbol in enumerate(symbols[:10], 1):
        print(f"    {i:2d}. {symbol}")
    if len(symbols) > 10:
        print(f"    ... 共 {len(symbols)} 个")
    
    # ================= 阶段1: 极速下载 =================
    
    print(f"\n{'=' * 70}")
    print("[阶段 1/2] 极速下载（单线程，Tardis API限制）")
    print("-" * 70)
    
    download_start = time.time()
    print(f"⏰ 下载开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📥 正在从 Tardis.dev 下载...")
    
    try:
        datasets.download(
            exchange=EXCHANGE,
            data_types=[DATA_TYPE],
            from_date=FROM_DATE,
            to_date=TO_DATE,
            symbols=symbols,
            api_key=API_KEY,
            download_dir=str(DOWNLOAD_DIR)
        )
    except Exception as e:
        print(f"   ⚠️  部分数据下载失败: {e}")
        print(f"   ℹ️  继续处理已下载的数据...")
    
    download_time = time.time() - download_start
    all_files = list(DOWNLOAD_DIR.glob("*.csv.gz"))
    total_size = sum(f.stat().st_size for f in all_files) / (1024**2) if all_files else 0
    
    print(f"\n✅ 下载完成")
    print(f"⏰ 下载结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  下载用时: {download_time/60:.1f} 分钟")
    print(f"📁 文件数: {len(all_files):,}")
    print(f"💾 总大小: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    if len(all_files) == 0:
        print("\n❌ 错误: 没有下载到任何文件")
        return
    
    # ================= 阶段2: 超高速并行处理 =================
    
    print(f"\n{'=' * 70}")
    print(f"[阶段 2/2] 超高速并行处理 ({PROCESS_WORKERS} 进程)")
    print("-" * 70)
    
    processing_start = time.time()
    print(f"⏰ 处理开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 输出格式: NPY (NumPy原生格式)")
    print()
    
    process_args = [
        (symbol, DOWNLOAD_DIR, OUTPUT_DIR, LOB_LEVELS)
        for symbol in symbols
    ]
    
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=PROCESS_WORKERS) as executor:
        futures = {executor.submit(process_symbol_batch, args): args[0] for args in process_args}
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if result['status'] == 'success':
                    print(f"  ✓ [{completed:2d}/{len(symbols)}] {symbol:15s} | "
                          f"{result['samples']:8,} 样本 | "
                          f"{result['size_mb']:7.2f} MB | "
                          f"{result['time']:5.1f}s")
                elif result['status'] == 'no_files':
                    print(f"  ⊘ [{completed:2d}/{len(symbols)}] {symbol:15s} | 跳过（无数据）")
                else:
                    print(f"  ⊘ [{completed:2d}/{len(symbols)}] {symbol:15s} | 跳过（失败）")
                        
            except Exception as e:
                print(f"  ✗ [{completed:2d}/{len(symbols)}] {symbol:15s} | 异常: {e}")
                completed += 1
    
    processing_time = time.time() - processing_start
    total_time = time.time() - download_start + download_time
    
    print(f"\n⏰ 处理结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  处理用时: {processing_time/60:.1f} 分钟")
    
    # ================= 统计报告 =================
    
    successful = [r for r in results if r['status'] == 'success']
    
    print(f"\n{'=' * 70}")
    print("处理完成 - 统计报告")
    print("=" * 70)
    
    print(f"\n📊 总体情况:")
    print(f"   处理标的数: {len(results)}")
    print(f"   成功: {len(successful)}")
    print(f"   跳过: {len(results) - len(successful)}")
    
    if successful:
        total_samples = sum(r['samples'] for r in successful)
        total_size_mb = sum(r['size_mb'] for r in successful)
        avg_time = np.mean([r['time'] for r in successful])
        
        print(f"\n📈 数据统计:")
        print(f"   总样本数: {total_samples:,}")
        print(f"   总文件大小: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
        print(f"   平均处理时间: {avg_time:.2f} 秒/标的")
        
        print(f"\n⏱️  时间统计:")
        print(f"   下载时间: {download_time/60:.1f} 分钟")
        print(f"   处理时间: {processing_time/60:.1f} 分钟")
        print(f"   总用时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
        
        print(f"\n⚡ 性能指标:")
        throughput = total_samples / processing_time
        print(f"   吞吐量: {throughput:,.0f} 样本/秒")
        print(f"   并行效率: {len(successful) / processing_time:.2f} 标的/秒")
        print(f"   CPU利用率: {PROCESS_WORKERS}/{CPU_COUNT} ({PROCESS_WORKERS/CPU_COUNT*100:.1f}%)")
        
        # 保存元数据
        metadata = {
            'date_range': f"{FROM_DATE.replace('-', '')}_{TO_DATE.replace('-', '')}",
            'total_symbols': len(successful),
            'total_samples': int(total_samples),
            'feature_dim': 40,
            'lob_levels': LOB_LEVELS,
            'target_col': 'return_10s',
            'additional_targets': ['return_60s'],
            'symbols': [r['symbol'] for r in successful],
            'download_date': datetime.now().isoformat(),
            'processing_time_minutes': round(total_time / 60, 2),
            'format': 'npy',
            'data_shape': '(N, 43) - [timestamp, 40 features, return_10s, return_60s]'
        }
        
        metadata_file = OUTPUT_DIR / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📄 元数据已保存: {metadata_file}")
    
    # 清理剩余原始文件
    remaining = list(DOWNLOAD_DIR.glob("*.csv.gz"))
    if remaining:
        print(f"\n🗑️  清理剩余 {len(remaining)} 个原始文件...")
        for f in remaining:
            f.unlink()
        print(f"   ✅ 清理完成")
    
    print(f"\n{'=' * 70}")
    print("⏰ 程序结束时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    print(f"\n✅ 全部完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"💾 数据格式: NPY (shape: N × 43)")
    print(f"   列0: timestamp")
    print(f"   列1-40: LOB features")
    print(f"   列41: return_10s")
    print(f"   列42: return_60s")


if __name__ == "__main__":
    main()

