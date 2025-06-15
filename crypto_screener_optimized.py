#!/usr/bin/env python3
"""
Optimized Crypto Screener
=========================

Enhanced version of crypto_screener.py with performance optimizations:
- Intelligent memory management
- Adaptive API rate limiting
- Optimized parallel processing
- Smart caching
- Performance monitoring
"""

import argparse
import time
import os
import numpy as np
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
import logging

from src.downloader import CryptoDownloader
from src.performance_optimizer import (
    MemoryManager, 
    AdaptiveRateLimiter, 
    OptimizedParallelProcessor,
    IntelligentCache,
    performance_monitor,
    DataPipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calc_total_bars(time_interval: str, days: int) -> int:
    """Calculate total bars needed for given timeframe and days"""
    bars_dict = {
        "5m": 12 * 24 * days,
        "15m": 4 * 24 * days,
        "30m": 2 * 24 * days,
        "1h": 24 * days,
        "2h": 12 * days,
        "4h": 6 * days,
        "8h": 3 * days,
        "1d": days,
    }
    return bars_dict.get(time_interval, 24 * days)

def calculate_rs_score(data, required_bars: int) -> tuple[bool, float, str]:
    """
    Calculate Relative Strength score with optimized memory usage
    """
    try:
        if len(data) < required_bars:
            return False, 0, f"Insufficient data: {len(data)} < {required_bars}"
        
        # Use the most recent data points
        recent_data = data.tail(required_bars).copy()
        
        # Optimize data types for memory efficiency
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in recent_data.columns:
                recent_data[col] = recent_data[col].astype(np.float32)
        
        # Calculate RS Score with vectorized operations
        rs_score = 0.0
        total_weight = 0.0
        
        # Pre-calculate moving averages if not present
        if 'sma_30' not in recent_data.columns:
            recent_data['sma_30'] = recent_data['close'].rolling(window=30, min_periods=1).mean()
            recent_data['sma_45'] = recent_data['close'].rolling(window=45, min_periods=1).mean()
            recent_data['sma_60'] = recent_data['close'].rolling(window=60, min_periods=1).mean()
        
        # Vectorized weight calculation
        bars = len(recent_data)
        indices = np.arange(bars)
        weights = np.exp(2 * np.log(2) * indices / bars)
        
        # Vectorized RS calculation
        close_prices = recent_data['close'].values
        sma_30 = recent_data['sma_30'].values
        sma_45 = recent_data['sma_45'].values
        sma_60 = recent_data['sma_60'].values
        atr_values = recent_data['atr'].values
        
        # Calculate normalized differences
        n_values = (
            (close_prices - sma_30) + 
            (close_prices - sma_45) + 
            (close_prices - sma_60) + 
            (sma_30 - sma_45) + 
            (sma_30 - sma_60) + 
            (sma_45 - sma_60)
        ) / atr_values
        
        # Apply weights and calculate final score
        weighted_sum = np.sum(n_values * weights)
        total_weight = np.sum(weights)
        
        rs_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        return True, float(rs_score), ""
        
    except Exception as e:
        return False, 0, f"Calculation error: {str(e)}"

class OptimizedCryptoProcessor:
    """Optimized crypto processing with advanced features"""
    
    def __init__(self):
        self.memory_manager = MemoryManager(max_memory_percent=75.0)
        self.rate_limiter = AdaptiveRateLimiter(initial_delay=0.3, max_delay=30.0)
        self.cache = IntelligentCache(max_size=500, ttl_seconds=1800)  # 30 min TTL
        self.downloader = CryptoDownloader()
        
    def process_crypto_optimized(self, symbol: str, timeframe: str, days: int) -> Dict[str, Any]:
        """Process a single cryptocurrency with optimizations"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Check cache first
                cache_key = f"{symbol}_{timeframe}_{days}_{int(time.time() // 1800)}"  # 30min buckets
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"{symbol} -> Using cached result")
                    return cached_result
                
                # Apply adaptive rate limiting
                self.rate_limiter.wait()
                
                # Calculate required parameters
                required_bars = calc_total_bars(timeframe, days)
                buffer_factor = 1.15  # Reduced buffer for efficiency
                now = int(time.time())
                
                # Optimized interval calculation
                interval_seconds = self._get_interval_seconds(timeframe)
                start_ts = now - int(required_bars * interval_seconds * buffer_factor)
                
                # Get crypto data
                success, data = self.downloader.get_data(
                    symbol, 
                    start_ts=start_ts, 
                    end_ts=now, 
                    timeframe=timeframe, 
                    atr=True
                )
                
                if not success or data.empty:
                    error_msg = "Failed to get data or empty dataset"
                    result = {"crypto": symbol, "status": "failed", "reason": error_msg}
                    self.rate_limiter.on_failure("data_error")
                    return result
                
                # Optimize DataFrame memory usage
                data = self.memory_manager.memory_efficient_dataframe_processing(data)
                
                # Calculate RS score
                success, rs_score, error = calculate_rs_score(data, required_bars)
                if not success:
                    result = {"crypto": symbol, "status": "failed", "reason": error}
                    self.rate_limiter.on_failure("calculation_error")
                    return result
                
                # Success
                result = {
                    "crypto": symbol,
                    "status": "success",
                    "rs_score": rs_score,
                    "data_points": len(data)
                }
                
                # Cache successful result
                self.cache.put(cache_key, result)
                self.rate_limiter.on_success()
                
                logger.info(f"{symbol} -> RS Score: {rs_score:.4f} ({len(data)} points)")
                return result
                
            except Exception as e:
                error_msg = str(e)
                
                # Handle specific error types
                if any(keyword in error_msg.lower() for keyword in ["rate limit", "too many requests", "1003"]):
                    if attempt < max_retries - 1:
                        wait_time = self.rate_limiter.current_delay * (2 ** attempt)
                        logger.warning(f"{symbol} -> Rate limit hit, retrying in {wait_time:.1f}s")
                        time.sleep(wait_time)
                        self.rate_limiter.on_failure("rate_limit")
                        continue
                    else:
                        logger.error(f"{symbol} -> Rate limit exceeded, max retries reached")
                        self.rate_limiter.on_failure("rate_limit_exceeded")
                else:
                    logger.error(f"{symbol} -> Error: {error_msg}")
                    self.rate_limiter.on_failure("general_error")
                
                return {"crypto": symbol, "status": "failed", "reason": error_msg}
        
        return {"crypto": symbol, "status": "failed", "reason": "Max retries exceeded"}
    
    def _get_interval_seconds(self, timeframe: str) -> int:
        """Get interval seconds for timeframe"""
        if "m" in timeframe:
            return int(timeframe.replace("m", "")) * 60
        elif "h" in timeframe:
            return int(timeframe.replace("h", "")) * 3600
        elif "d" in timeframe:
            return int(timeframe.replace("d", "")) * 86400
        else:
            return 3600  # Default to 1 hour
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'memory': self.memory_manager.get_memory_usage(),
            'cache': self.cache.get_stats(),
            'rate_limiter': self.rate_limiter.get_stats()
        }

@performance_monitor
def run_optimized_crypto_screening(timeframe: str, days: int, max_workers: int = None) -> Dict[str, Any]:
    """Run optimized crypto screening with performance monitoring"""
    
    # Initialize components
    processor = OptimizedCryptoProcessor()
    parallel_processor = OptimizedParallelProcessor(
        max_workers=max_workers, 
        memory_manager=processor.memory_manager
    )
    
    # Get all crypto symbols
    logger.info("Fetching crypto symbols...")
    all_cryptos = processor.downloader.get_all_symbols()
    logger.info(f"Found {len(all_cryptos)} crypto symbols to process")
    
    # Create processing function
    def process_single_crypto(symbol):
        return processor.process_crypto_optimized(symbol, timeframe, days)
    
    # Process with optimized parallel processing
    logger.info(f"Starting parallel processing with timeframe={timeframe}, days={days}")
    results = parallel_processor.process_with_rate_limiting(
        func=process_single_crypto,
        items=all_cryptos,
        use_threads=False,  # Use processes for CPU-bound work
        chunk_size=max(1, len(all_cryptos) // (max_workers or 4))
    )
    
    # Process results
    successful_results = []
    failed_results = []
    target_scores = {}
    
    for result in results:
        if isinstance(result, dict) and result.get("status") == "success":
            symbol = result["crypto"]
            rs_score = result["rs_score"]
            successful_results.append(result)
            target_scores[symbol] = rs_score
        else:
            failed_results.append(result)
    
    # Sort by RS score
    sorted_symbols = sorted(target_scores.keys(), key=lambda x: target_scores[x], reverse=True)
    
    # Performance statistics
    performance_stats = processor.get_performance_stats()
    
    return {
        'successful_count': len(successful_results),
        'failed_count': len(failed_results),
        'total_processed': len(all_cryptos),
        'top_performers': sorted_symbols[:20],
        'target_scores': target_scores,
        'failed_symbols': [r.get('crypto', 'unknown') for r in failed_results],
        'performance_stats': performance_stats
    }

def save_results_optimized(results: Dict[str, Any], timeframe: str, days: int):
    """Save results with optimized file operations"""
    
    # Create output directory
    output_dir = "output"
    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    full_output_dir = os.path.join(output_dir, date_str)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Generate filenames
    base_filename = f"{timestamp_str}_crypto_{timeframe}_optimized"
    txt_filename = os.path.join(full_output_dir, f"{base_filename}_strong_targets.txt")
    stats_filename = os.path.join(full_output_dir, f"{base_filename}_stats.txt")
    
    # Save TradingView format file
    top_performers = results['top_performers']
    target_scores = results['target_scores']
    
    with open(txt_filename, 'w') as f:
        f.write("BINANCE:BTCUSDT\n")  # Header for TradingView
        for symbol in top_performers:
            if symbol.endswith('USDT'):
                f.write(f"BINANCE:{symbol}\n")
    
    # Save detailed statistics
    with open(stats_filename, 'w') as f:
        f.write(f"Optimized Crypto Screening Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timeframe: {timeframe}\n")
        f.write(f"Days: {days}\n")
        f.write(f"Timestamp: {timestamp_str}\n\n")
        
        f.write(f"Processing Summary:\n")
        f.write(f"Total symbols: {results['total_processed']}\n")
        f.write(f"Successfully processed: {results['successful_count']}\n")
        f.write(f"Failed: {results['failed_count']}\n")
        f.write(f"Success rate: {results['successful_count']/results['total_processed']*100:.1f}%\n\n")
        
        f.write(f"Top 20 Performers:\n")
        f.write(f"{'-'*30}\n")
        for i, symbol in enumerate(top_performers, 1):
            score = target_scores[symbol]
            f.write(f"{i:2d}. {symbol:15s}: {score:8.4f}\n")
        
        # Performance statistics
        perf_stats = results['performance_stats']
        f.write(f"\nPerformance Statistics:\n")
        f.write(f"{'-'*30}\n")
        f.write(f"Memory usage: {perf_stats['memory']['process_mb']:.1f} MB\n")
        f.write(f"Cache hit rate: {perf_stats['cache']['hit_rate']*100:.1f}%\n")
        f.write(f"API delay: {perf_stats['rate_limiter']['current_delay']:.2f}s\n")
    
    logger.info(f"Results saved to {txt_filename}")
    logger.info(f"Statistics saved to {stats_filename}")

def main():
    """Main function with optimized argument parsing and execution"""
    parser = argparse.ArgumentParser(
        description='Optimized Crypto Screener with Performance Enhancements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default: 15m timeframe, 3 days
  %(prog)s -t 1h -d 7              # 1 hour timeframe, 7 days
  %(prog)s -t 4h -d 14 -w 8        # 4 hour timeframe, 14 days, 8 workers
  %(prog)s --performance-mode      # Enable detailed performance monitoring
        """
    )
    
    parser.add_argument('-t', '--timeframe', type=str, 
                       choices=['5m', '15m', '30m', '1h', '2h', '4h', '8h', '1d'],
                       default="15m", 
                       help='Time frame (default: 15m)')
    
    parser.add_argument('-d', '--days', type=int, default=3,
                       help='Calculation duration in days (default: 3)')
    
    parser.add_argument('-w', '--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    
    parser.add_argument('--performance-mode', action='store_true',
                       help='Enable detailed performance monitoring')
    
    parser.add_argument('--memory-limit', type=float, default=75.0,
                       help='Memory usage limit percentage (default: 75.0)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.performance_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Performance monitoring enabled")
    
    # Determine optimal worker count
    if args.workers is None:
        # Conservative approach for API rate limits
        args.workers = min(4, mp.cpu_count())
    
    logger.info(f"Starting optimized crypto screening...")
    logger.info(f"Configuration: timeframe={args.timeframe}, days={args.days}, workers={args.workers}")
    
    try:
        # Run optimized screening
        start_time = time.time()
        results = run_optimized_crypto_screening(
            timeframe=args.timeframe,
            days=args.days,
            max_workers=args.workers
        )
        end_time = time.time()
        
        # Display results
        print(f"\n{'='*60}")
        print(f"🚀 OPTIMIZED CRYPTO SCREENING RESULTS")
        print(f"{'='*60}")
        print(f"⏱️  Execution time: {end_time - start_time:.1f} seconds")
        print(f"📊 Total processed: {results['total_processed']}")
        print(f"✅ Successful: {results['successful_count']}")
        print(f"❌ Failed: {results['failed_count']}")
        print(f"📈 Success rate: {results['successful_count']/results['total_processed']*100:.1f}%")
        
        print(f"\n🏆 TOP 20 PERFORMERS:")
        print(f"{'='*40}")
        target_scores = results['target_scores']
        for i, symbol in enumerate(results['top_performers'], 1):
            score = target_scores[symbol]
            print(f"{i:2d}. {symbol:15s}: {score:8.4f}")
        
        # Performance statistics
        if args.performance_mode:
            perf_stats = results['performance_stats']
            print(f"\n📊 PERFORMANCE STATISTICS:")
            print(f"{'='*40}")
            print(f"Memory usage: {perf_stats['memory']['process_mb']:.1f} MB")
            print(f"Cache hits: {perf_stats['cache']['hits']}")
            print(f"Cache hit rate: {perf_stats['cache']['hit_rate']*100:.1f}%")
            print(f"Current API delay: {perf_stats['rate_limiter']['current_delay']:.2f}s")
        
        # Save results
        save_results_optimized(results, args.timeframe, args.days)
        
        print(f"\n✅ Analysis completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")
        raise

if __name__ == '__main__':
    main() 