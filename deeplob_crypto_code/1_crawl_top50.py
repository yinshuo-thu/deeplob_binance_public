import requests
import pandas as pd
import time
# 【修改点】: 去掉 .notebook，改用标准版 tqdm，兼容性更好，不会报错
from tqdm import tqdm 
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= 配置区域 =================
OUTPUT_POOL_SIZE = 50  # 修改为 50
OUTPUT_FILE = "candidates.txt"

# 流动性考察时间窗口 (2025/8/1-11/1 = 92天)
LOOKBACK_DAYS = 92 

# 代理设置 - 已启用
PROXIES = {
   "http": "http://127.0.0.1:7890",
   "https": "http://127.0.0.1:7890",
}

# API 配置
EXCHANGE_INFO_API = "https://fapi.binance.com/fapi/v1/exchangeInfo"
KLINE_API  = "https://fapi.binance.com/fapi/v1/klines"

# ================= 函数定义 =================

def get_session():
    """配置带有重试机制的 Session"""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def get_all_usdt_symbols(session):
    """
    获取市场上所有状态正常的 USDT 本位合约代码
    """
    print(f"[1/2] 正在获取全市场交易对列表...")
    try:
        # 使用 exchangeInfo 接口获取更准确的交易对状态
        # 注意：这里加入了 proxies 参数
        resp = session.get(EXCHANGE_INFO_API, timeout=15, proxies=PROXIES)
        resp.raise_for_status()
        data = resp.json()
        
        symbols = []
        for item in data['symbols']:
            # 过滤条件：
            # 1. 交易对必须是 TRADING 状态 (排除下架或暂停的)
            # 2. 结算资产是 USDT
            # 3. 排除合约类型不是 PERPETUAL (永续) 的
            if (item['status'] == 'TRADING' and 
                item['quoteAsset'] == 'USDT' and 
                item['contractType'] == 'PERPETUAL'):
                
                symbols.append(item['symbol'])
        
        print(f"      -> 成功获取 {len(symbols)} 个正在交易的 USDT 永续合约。")
        return symbols
        
    except Exception as e:
        print(f"获取交易对列表失败: {e}")
        return []

def get_avg_volume(session, symbol, days):
    """
    获取单个标的过去 N 天的平均成交额 (Quote Asset Volume)
    """
    params = {
        'symbol': symbol,
        'interval': '1d',
        'limit': days 
    }

    try:
        # K线数据格式: [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, ...]
        # 索引 7 是 Quote asset volume (USDT 成交额)
        # 注意：这里加入了 proxies 参数
        resp = session.get(KLINE_API, params=params, timeout=10, proxies=PROXIES)
        data = resp.json()
        
        if not data:
            return 0.0
            
        total_quote_vol = 0.0
        count = 0
        for k in data:
            total_quote_vol += float(k[7])
            count += 1
            
        if count == 0: 
            return 0.0
            
        return total_quote_vol / count
    except Exception as e:
        # print(f"获取 {symbol} 数据失败: {e}")
        return 0.0

# ================= 主执行逻辑 (Jupyter 直接运行) =================

# 1. 初始化 Session
session = get_session()

# 2. 获取全量列表
all_tickers = get_all_usdt_symbols(session)

if all_tickers:
    print(f"\n[2/2] 计算 {len(all_tickers)} 个标的过去 {LOOKBACK_DAYS} 天（2025/8/1-11/1）的平均流动性...")
    
    longterm_stats = []

    for symbol in tqdm(all_tickers, desc="Calculating Liquidity"):
        avg_vol = get_avg_volume(session, symbol, LOOKBACK_DAYS)
        longterm_stats.append({
            'symbol': symbol,
            'avg_vol_90d': avg_vol
        })
        # 稍微控制请求频率
        time.sleep(0.02)
        
    # 3. 数据处理与排序
    df = pd.DataFrame(longterm_stats)
    # 按平均成交额降序排列
    df = df.sort_values(by='avg_vol_90d', ascending=False)

    # 4. 提取 Top N
    final_list = df.head(OUTPUT_POOL_SIZE)['symbol'].tolist()
    
    # 5. 打印前5名预览
    print(f"\n=== Top 5 Liquidity (2025/8/1-11/1, 92 Days) ===")
    print("-" * 50)
    print(f"{'Symbol':<15} | {'Avg Daily Vol (USDT)':<30}")
    print("-" * 50)
    for index, row in df.head(5).iterrows():
        vol_str = f"{row['avg_vol_90d']:,.0f}"  # avg_vol_90d变量名保持不变，实际是92天
        print(f"{row['symbol']:<15} | {vol_str:<30}")
    print("-" * 50)
    
    # 6. 保存文件
    with open(OUTPUT_FILE, 'w') as f:
        for t in final_list:
            f.write(t + "\n")
            
    print(f"\n✅ Top {OUTPUT_POOL_SIZE} 列表已保存至当前目录下的: {OUTPUT_FILE}")

