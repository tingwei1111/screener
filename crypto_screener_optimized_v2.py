#!/usr/bin/env python3
"""
優化版加密貨幣篩選器 v2.0
整合所有性能優化技術：
- 向量化計算
- 智能緩存
- 並行處理
- 記憶體優化
- 性能監控
"""

import argparse
import time
import os
import numpy as np
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import warnings

# 導入我們的優化模組
from src.downloader import CryptoDownloader
from performance_monitor import performance_monitor, start_monitoring, print_performance_report
from cache_manager import cached, get_default_cache_manager

# 抑制警告
warnings.filterwarnings('ignore')

# 預編譯的常量
TIMEFRAME_BARS = {
    "5m": lambda days: 12 * 24 * days,
    "15m": lambda days: 4 * 24 * days,
    "30m": lambda days: 2 * 24 * days,
    "1h": lambda days: 24 * days,
    "2h": lambda days: 12 * days,
    "4h": lambda days: 6 * days,
    "8h": lambda days: 3 * days,
    "1d": lambda days: days,
}

@lru_cache(maxsize=32)
def calc_total_bars(time_interval: str, days: int) -> Optional[int]:
    """計算總K線數量（帶緩存）"""
    calc_func = TIMEFRAME_BARS.get(time_interval)
    return calc_func(days) if calc_func else None

@performance_monitor
def calculate_rs_score_vectorized(crypto_data: np.ndarray, required_bars: int) -> Tuple[bool, float, str]:
    """
    向量化RS分數計算 - 完全使用numpy優化
    
    Args:
        crypto_data: 預處理的numpy數組 [close, sma_30, sma_45, sma_60, atr]
        required_bars: 所需的K線數量
        
    Returns:
        (成功標誌, RS分數, 錯誤信息)
    """
    if len(crypto_data) < required_bars:
        return False, 0, f"數據不足: {len(crypto_data)} < {required_bars}"
    
    # 取最新數據
    data = crypto_data[-required_bars:]
    
    # 向量化提取列
    close_prices, sma_30, sma_45, sma_60, atr_values = data.T
    
    # 向量化分子計算
    numerator = (
        (close_prices - sma_30) +
        (close_prices - sma_45) +
        (close_prices - sma_60) +
        (sma_30 - sma_45) +
        (sma_30 - sma_60) +
        (sma_45 - sma_60)
    )
    
    # 向量化分母計算（避免除零）
    denominator = np.maximum(atr_values, 1e-18)
    
    # 向量化相對強度計算
    relative_strength = numerator / denominator
    
    # 向量化權重計算（指數衰減，越新權重越大）
    k = 2 * np.log(2) / required_bars
    indices = np.arange(required_bars)
    weights = np.exp(k * indices)
    
    # 加權平均
    rs_score = np.average(relative_strength, weights=weights)
    
    return True, float(rs_score), ""

@cached(ttl=1800)  # 緩存30分鐘
@performance_monitor
def get_interval_seconds(timeframe: str) -> int:
    """獲取時間間隔秒數（帶緩存）"""
    if "m" in timeframe:
        return int(timeframe.replace("m", "")) * 60
    elif "h" in timeframe:
        return int(timeframe.replace("h", "")) * 3600
    elif "d" in timeframe:
        return int(timeframe.replace("d", "")) * 86400
    else:
        return 3600  # 默認1小時

class OptimizedCryptoProcessor:
    """優化的加密貨幣處理器"""
    
    def __init__(self):
        self.downloader = CryptoDownloader()
        self._symbol_cache = None
        self._cache_time = 0
        
    @cached(ttl=3600)  # 緩存1小時
    def get_all_symbols(self) -> List[str]:
        """獲取所有交易對（帶緩存）"""
        return self.downloader.get_all_symbols()
    
    @performance_monitor
    def process_single_crypto(self, symbol: str, timeframe: str, days: int, 
                            max_retries: int = 3) -> Dict[str, any]:
        """處理單個加密貨幣"""
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                # 計算所需數據量
                required_bars = calc_total_bars(timeframe, days)
                if not required_bars:
                    return {"crypto": symbol, "status": "failed", "reason": "無效時間框架"}
                
                # 計算時間範圍
                now = int(time.time())
                interval_seconds = get_interval_seconds(timeframe)
                buffer_factor = 1.3  # 緩衝因子
                start_ts = now - int(required_bars * interval_seconds * buffer_factor)
                
                # 獲取數據
                success, data = self.downloader.get_data(
                    symbol, start_ts=start_ts, end_ts=now, 
                    timeframe=timeframe, atr=True
                )
                
                if not success or data.empty:
                    return {"crypto": symbol, "status": "failed", "reason": "數據獲取失敗"}
                
                # 檢查必需列
                required_columns = ['close', 'sma_30', 'sma_45', 'sma_60', 'atr']
                if not all(col in data.columns for col in required_columns):
                    return {"crypto": symbol, "status": "failed", "reason": f"缺少必需列: {required_columns}"}
                
                # 轉換為numpy數組進行快速計算
                crypto_array = data[required_columns].to_numpy(dtype=np.float64)
                
                # 計算RS分數
                success, rs_score, error = calculate_rs_score_vectorized(crypto_array, required_bars)
                
                if not success:
                    return {"crypto": symbol, "status": "failed", "reason": error}
                
                return {
                    "crypto": symbol,
                    "status": "success",
                    "rs_score": rs_score,
                    "data_points": len(data)
                }
                
            except Exception as e:
                error_msg = str(e)
                
                # 處理API限制
                if any(keyword in error_msg.lower() for keyword in ["rate limit", "too many", "1003"]):
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"{symbol} -> API限制，{wait_time:.1f}秒後重試 ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {"crypto": symbol, "status": "failed", "reason": "API限制超過重試次數"}
                else:
                    return {"crypto": symbol, "status": "failed", "reason": error_msg}
        
        return {"crypto": symbol, "status": "failed", "reason": "未知錯誤"}

@performance_monitor
def process_crypto_batch(symbols: List[str], timeframe: str, days: int, 
                        batch_size: int = 20, max_workers: int = 8) -> List[Dict]:
    """批量處理加密貨幣"""
    processor = OptimizedCryptoProcessor()
    all_results = []
    
    # 分批處理以控制記憶體使用
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        print(f"處理批次 {batch_num}/{total_batches} ({len(batch)} 個交易對)")
        
        # 使用線程池處理I/O密集型任務
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(processor.process_single_crypto, crypto, timeframe, days): crypto 
                for crypto in batch
            }
            
            batch_results = []
            for future in as_completed(futures):
                crypto = futures[future]
                try:
                    result = future.result(timeout=30)  # 30秒超時
                    batch_results.append(result)
                    
                    # 實時反饋
                    if result["status"] == "success":
                        print(f"✓ {crypto}: RS={result['rs_score']:.6f}")
                    else:
                        print(f"✗ {crypto}: {result['reason']}")
                        
                except Exception as e:
                    print(f"✗ {crypto}: 處理異常 - {e}")
                    batch_results.append({
                        "crypto": crypto, 
                        "status": "failed", 
                        "reason": f"處理異常: {e}"
                    })
        
        all_results.extend(batch_results)
        
        # 批次間短暫休息
        if i + batch_size < len(symbols):
            time.sleep(1)
    
    return all_results

@performance_monitor
def analyze_and_save_results(results: List[Dict], timeframe: str) -> None:
    """分析並保存結果"""
    # 分離成功和失敗的結果
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "failed"]
    
    # 統計信息
    total_processed = len(results)
    success_count = len(successful_results)
    failure_count = len(failed_results)
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"分析結果統計:")
    print(f"總處理數量: {total_processed}")
    print(f"成功處理: {success_count} ({success_rate:.1f}%)")
    print(f"處理失敗: {failure_count}")
    print(f"{'='*60}")
    
    if not successful_results:
        print("⚠️  沒有成功處理的交易對")
        return
    
    # 按RS分數排序
    successful_results.sort(key=lambda x: x["rs_score"], reverse=True)
    
    # 顯示前20名
    print(f"\n🏆 前20名交易對 (按RS分數排序):")
    print(f"{'排名':<4} {'交易對':<12} {'RS分數':<12} {'數據點':<8}")
    print("-" * 40)
    
    for idx, result in enumerate(successful_results[:20], 1):
        print(f"{idx:<4} {result['crypto']:<12} {result['rs_score']:<12.6f} {result.get('data_points', 'N/A'):<8}")
    
    # 保存結果文件
    save_results_to_file(successful_results, timeframe)
    
    # 錯誤統計
    if failed_results:
        error_stats = {}
        for result in failed_results:
            reason = result.get("reason", "未知錯誤")
            error_stats[reason] = error_stats.get(reason, 0) + 1
        
        print(f"\n❌ 錯誤統計:")
        for error, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error}: {count}個")

def save_results_to_file(results: List[Dict], timeframe: str) -> None:
    """保存結果到文件"""
    if not results:
        return
    
    # 創建輸出目錄
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d_%H-%M")
    
    from pathlib import Path
    output_dir = Path("output") / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成TradingView格式的文件內容
    content_lines = [
        "###BTCETH",
        "BINANCE:BTCUSDT.P,BINANCE:ETHUSDT",
        "###Targets (Sort by RS Score)"
    ]
    
    # 添加所有交易對
    symbols = [f"BINANCE:{result['crypto']}.P" for result in results]
    content_lines.append(",".join(symbols))
    
    # 寫入文件
    output_file = output_dir / f"{time_str}_crypto_{timeframe}_optimized_targets.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(content_lines))
    
    print(f"\n💾 結果已保存至: {output_file}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="優化版加密貨幣篩選器 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python crypto_screener_optimized_v2.py -t 15m -d 3
  python crypto_screener_optimized_v2.py --timeframe 1h --days 7 --batch-size 30
        """
    )
    
    parser.add_argument('-t', '--timeframe', type=str, default="15m",
                       help='時間框架 (5m, 15m, 30m, 1h, 2h, 4h, 8h, 1d)')
    parser.add_argument('-d', '--days', type=int, default=3,
                       help='計算天數 (默認: 3)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='批次大小 (默認: 20)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='最大工作線程數 (默認: 8)')
    parser.add_argument('--enable-monitoring', action='store_true',
                       help='啟用性能監控')
    parser.add_argument('--cache-stats', action='store_true',
                       help='顯示緩存統計')
    
    args = parser.parse_args()
    
    print("🚀 啟動優化版加密貨幣篩選器 v2.0")
    print(f"參數: 時間框架={args.timeframe}, 天數={args.days}, 批次大小={args.batch_size}")
    
    # 啟用性能監控
    if args.enable_monitoring:
        start_monitoring(interval=2.0)
        print("📊 性能監控已啟用")
    
    start_time = time.time()
    
    try:
        # 獲取所有交易對
        processor = OptimizedCryptoProcessor()
        all_symbols = processor.get_all_symbols()
        print(f"📈 找到 {len(all_symbols)} 個交易對")
        
        # 批量處理
        results = process_crypto_batch(
            all_symbols, 
            args.timeframe, 
            args.days,
            args.batch_size,
            args.max_workers
        )
        
        # 分析和保存結果
        analyze_and_save_results(results, args.timeframe)
        
    except KeyboardInterrupt:
        print("\n⏹️  用戶中斷執行")
    except Exception as e:
        print(f"\n💥 執行錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 執行時間統計
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  總執行時間: {elapsed_time:.2f} 秒")
        
        # 顯示性能報告
        if args.enable_monitoring:
            print("\n" + "="*60)
            print_performance_report()
        
        # 顯示緩存統計
        if args.cache_stats:
            from cache_manager import print_cache_stats
            print("\n" + "="*60)
            print_cache_stats()

if __name__ == '__main__':
    # 導入路徑修正
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    
    main()