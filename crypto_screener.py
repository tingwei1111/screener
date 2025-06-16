import argparse
import time
import os
import numpy as np
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
from src.downloader import CryptoDownloader


@lru_cache(maxsize=32)
def calc_total_bars(time_interval: str, days: int) -> Optional[int]:
    """Calculate total bars for given time interval and days with caching"""
    bars_dict = {
        "5m": 12 * 24 * days,
        "15m": 4 * 24 * days,
        "30m": 2 * 24 * days,
        "1h":  24 * days,
        "2h": 12 * days,
        "4h": 6 * days,
        "8h": 3 * days,
        "1d": days,
    }
    return bars_dict.get(time_interval)


def calculate_rs_score(crypto_data: np.ndarray, required_bars: int) -> Tuple[bool, float, str]:
    """
    Calculate RS score for cryptocurrency using optimized numpy operations
    
    Args:
        crypto_data: Pre-processed numpy array with [close, sma_30, sma_45, sma_60, atr]
        required_bars: Number of bars required for calculation
        
    Returns:
        tuple[bool, float, str]: Success flag, RS score, error message
    """
    # Check if we have enough data
    if len(crypto_data) < required_bars:
        return False, 0, f"Insufficient data: {len(crypto_data)} < {required_bars}"
    
    # Take the most recent required_bars data points
    data = crypto_data[-required_bars:]
    
    # Extract columns using vectorized operations
    close_prices = data[:, 0]
    sma_30 = data[:, 1]
    sma_45 = data[:, 2]
    sma_60 = data[:, 3]
    atr_values = data[:, 4]
    
    # Vectorized numerator calculation
    numerator = ((close_prices - sma_30) +
                 (close_prices - sma_45) +
                 (close_prices - sma_60) +
                 (sma_30 - sma_45) +
                 (sma_30 - sma_60) +
                 (sma_45 - sma_60))
    
    # Vectorized denominator with epsilon to avoid division by zero
    denominator = atr_values + 1e-18
    
    # Vectorized relative strength calculation
    relative_strength = numerator / denominator
    
    # Vectorized weight calculation (exponential decay)
    k = 2 * np.log(2) / required_bars
    indices = np.arange(required_bars)
    weights = np.exp(k * indices)
    
    # Weighted average using numpy dot product
    rs_score = np.dot(relative_strength, weights) / np.sum(weights)

    return True, rs_score, ""


def process_crypto(symbol: str, timeframe: str, days: int) -> Dict[str, any]:
    """Process a single cryptocurrency and calculate its RS score with optimized data handling"""
    max_retries = 3
    retry_delay = 1  # Reduced delay for faster processing
    
    for attempt in range(max_retries):
        try:
            cd = CryptoDownloader()
            
            # Calculate required bars
            required_bars = calc_total_bars(timeframe, days)
            
            # Calculate start timestamp with some buffer (20% more time to ensure we get enough data)
            buffer_factor = 1.2
            now = int(time.time())
            
            # Estimate interval seconds based on timeframe
            if "m" in timeframe:
                minutes = int(timeframe.replace("m", ""))
                interval_seconds = minutes * 60
            elif "h" in timeframe:
                hours = int(timeframe.replace("h", ""))
                interval_seconds = hours * 3600
            elif "d" in timeframe:
                days = int(timeframe.replace("d", ""))
                interval_seconds = days * 24 * 3600
            else:
                # Default to 1h if unknown format
                interval_seconds = 3600
            
            start_ts = now - int(required_bars * interval_seconds * buffer_factor)
            
            # Get crypto data
            success, data = cd.get_data(symbol, start_ts=start_ts, end_ts=now, timeframe=timeframe, atr=True)
            
            if not success or data.empty:
                error_msg = "Failed to get data or empty dataset"
                print(f"{symbol} -> Error: {error_msg}")
                return {"crypto": symbol, "status": "failed", "reason": error_msg}
            
            # Pre-process data for vectorized calculation
            required_columns = ['close', 'sma_30', 'sma_45', 'sma_60', 'atr']
            if not all(col in data.columns for col in required_columns):
                error_msg = f"Missing required columns: {required_columns}"
                print(f"{symbol} -> Error: {error_msg}")
                return {"crypto": symbol, "status": "failed", "reason": error_msg}
            
            # Convert to numpy array for faster processing
            crypto_array = data[required_columns].to_numpy(dtype=np.float64)
            
            # Calculate RS score
            success, rs_score, error = calculate_rs_score(crypto_array, required_bars)
            if not success:
                print(f"{symbol} -> Error: {error}")
                return {"crypto": symbol, "status": "failed", "reason": error}
            
            print(f"{symbol} -> Successfully calculated RS Score: {rs_score}")
            return {
                "crypto": symbol,
                "status": "success",
                "rs_score": rs_score
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's an API rate limit error
            if "Too many requests" in error_msg or "1003" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"{symbol} -> Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"{symbol} -> Rate limit exceeded, max retries reached")
            else:
                print(f"{symbol} -> Error: {error_msg}")
            
            return {"crypto": symbol, "status": "failed", "reason": error_msg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timeframe', type=str, help='Time frame (5m, 15m, 30m, 1h, 2h, 4h, 8h, 1d)', default="15m")
    parser.add_argument('-d', '--days', type=int, help='Calculation duration in days (default 3 days)', default=3)
    args = parser.parse_args()
    timeframe = args.timeframe
    days = args.days
    
    # Initialize crypto downloader
    crypto_downloader = CryptoDownloader()
    
    # Get list of all symbols
    all_cryptos = crypto_downloader.get_all_symbols()
    print(f"Total cryptos to process: {len(all_cryptos)}")
    
    # Process all cryptos using ThreadPoolExecutor for I/O bound operations
    num_workers = min(8, mp.cpu_count() * 2)  # Use more threads for I/O operations
    print(f"Using {num_workers} workers")
    
    # Batch processing to reduce memory usage
    batch_size = 50
    all_results = []
    
    for i in range(0, len(all_cryptos), batch_size):
        batch = all_cryptos[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(all_cryptos) + batch_size - 1)//batch_size}")
        
            futures = {executor.submit(process_crypto, crypto, timeframe, days): crypto for crypto in batch}
            batch_results = []
            
            for future in as_completed(futures):
                crypto = futures[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    print(f"{crypto} -> Error: {str(e)}")
                    batch_results.append({"crypto": crypto, "status": "failed", "reason": str(e)})
            
            all_results.extend(batch_results)
            
            # Small delay between batches to respect API limits
            if i + batch_size < len(all_cryptos):
                time.sleep(2)
    
    # Process results
    failed_targets = []     # Failed to download data or error happened
    target_score = {}
    
    for result in all_results:
        if result["status"] == "success":
            target_score[result["crypto"]] = result["rs_score"]
        else:
            failed_targets.append((result["crypto"], result["reason"]))
    
    # Sort by RS score
    targets = [x for x in target_score.keys()]
    targets.sort(key=lambda x: target_score[x], reverse=True)
    
    # Print results
    print(f"\nAnalysis Results:")
    print(f"Total cryptos processed: {len(all_cryptos)}")
    print(f"Failed cryptos: {len(failed_targets)}")
    print(f"Successfully calculated: {len(targets)}")
    
    print("\n=========================== Target : Score (TOP 20) ===========================")
    for idx, crypto in enumerate(targets[:20], 1):
        score = target_score[crypto]
        print(f"{idx}. {crypto}: {score:.6f}")
    print("===============================================================================")
    
    # Save results
    full_date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    date_str = datetime.now().strftime("%Y-%m-%d")
    txt_content = "###BTCETH\nBINANCE:BTCUSDT.P,BINANCE:ETHUSDT\n###Targets (Sort by score)\n"
    
    # Add all targets
    if targets:
        txt_content += ",".join([f"BINANCE:{crypto}.P" for crypto in targets])
    
    # Create output/<date> directory structure
    base_folder = "output"
    date_folder = os.path.join(base_folder, date_str)
    os.makedirs(date_folder, exist_ok=True)
    
    # Save the file with full timestamp in filename
    output_file = f"{full_date_str}_crypto_{timeframe}_strong_targets.txt"
    file_path = os.path.join(date_folder, output_file)
    with open(file_path, "w") as f:
        f.write(txt_content)
    
    # Save failed cryptos for analysis
    # failed_file = f"{full_date_str}_failed_cryptos_{timeframe}.txt"
    # failed_path = os.path.join(date_folder, failed_file)
    # with open(failed_path, "w") as f:
    #     for crypto, reason in failed_targets:
    #         f.write(f"{crypto}: {reason}\n")
    
    print(f"\nResults saved to {file_path}")
    # print(f"Failed cryptos saved to {failed_path}")
