"""
Performance Optimization Module
==============================

This module provides performance optimizations for the Screener project including:
- Memory management and garbage collection
- Intelligent caching strategies
- Optimized parallel processing
- API rate limiting with adaptive backoff
- Data processing pipeline optimization
"""

import gc
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    start_time: float
    memory_usage_mb: float
    cpu_percent: float
    active_threads: int
    cache_hits: int = 0
    cache_misses: int = 0

class MemoryManager:
    """Intelligent memory management"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.memory_threshold = psutil.virtual_memory().total * (max_memory_percent / 100)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'process_mb': process.memory_info().rss / (1024**2)
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        return psutil.virtual_memory().percent > self.max_memory_percent
    
    def force_garbage_collection(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        logger.info(f"Garbage collection completed. Memory usage: {self.get_memory_usage()['process_mb']:.1f} MB")
    
    def memory_efficient_dataframe_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        if df.empty:
            return df
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df

class AdaptiveRateLimiter:
    """Adaptive API rate limiting with exponential backoff"""
    
    def __init__(self, initial_delay: float = 0.5, max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.initial_delay = initial_delay
        self.current_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.success_count = 0
        self.failure_count = 0
        self.lock = threading.Lock()
    
    def wait(self):
        """Wait according to current rate limit"""
        time.sleep(self.current_delay)
    
    def on_success(self):
        """Called when API request succeeds"""
        with self.lock:
            self.success_count += 1
            # Gradually reduce delay on consecutive successes
            if self.success_count >= 5:
                self.current_delay = max(self.initial_delay, self.current_delay * 0.8)
                self.success_count = 0
            self.failure_count = 0
    
    def on_failure(self, error_type: str = "unknown"):
        """Called when API request fails"""
        with self.lock:
            self.failure_count += 1
            self.success_count = 0
            
            # Increase delay based on error type
            if "rate limit" in error_type.lower() or "too many requests" in error_type.lower():
                self.current_delay = min(self.max_delay, self.current_delay * self.backoff_factor)
            elif "timeout" in error_type.lower():
                self.current_delay = min(self.max_delay, self.current_delay * 1.5)
            
            logger.warning(f"API failure ({error_type}). New delay: {self.current_delay:.2f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            'current_delay': self.current_delay,
            'success_count': self.success_count,
            'failure_count': self.failure_count
        }

class IntelligentCache:
    """Intelligent caching with TTL and memory management"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.cache:
            return True
        return time.time() - self.cache[key]['timestamp'] > self.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used items"""
        if len(self.cache) < self.max_size:
            return
        
        # Find least recently used key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache and not self._is_expired(key):
                self.access_times[key] = time.time()
                self.hits += 1
                return self.cache[key]['data']
            else:
                self.misses += 1
                if key in self.cache:  # Expired
                    del self.cache[key]
                    del self.access_times[key]
                return None
    
    def put(self, key: str, data: Any):
        """Put item in cache"""
        with self.lock:
            self._evict_lru()
            self.cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
            self.access_times[key] = time.time()
    
    def clear_expired(self):
        """Clear expired entries"""
        with self.lock:
            expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
            for key in expired_keys:
                del self.cache[key]
                del self.access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

class OptimizedParallelProcessor:
    """Optimized parallel processing with dynamic worker adjustment"""
    
    def __init__(self, max_workers: Optional[int] = None, memory_manager: Optional[MemoryManager] = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.memory_manager = memory_manager or MemoryManager()
        self.rate_limiter = AdaptiveRateLimiter()
    
    def _adjust_workers_for_memory(self, current_workers: int) -> int:
        """Dynamically adjust worker count based on memory usage"""
        if self.memory_manager.check_memory_pressure():
            return max(1, current_workers // 2)
        return current_workers
    
    def process_with_rate_limiting(self, 
                                 func: Callable, 
                                 items: List[Any], 
                                 use_threads: bool = False,
                                 chunk_size: Optional[int] = None) -> List[Any]:
        """Process items with rate limiting and memory management"""
        
        if not items:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 4))
        
        # Adjust workers based on memory
        workers = self._adjust_workers_for_memory(self.max_workers)
        
        results = []
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        logger.info(f"Processing {len(items)} items with {workers} workers (chunks of {chunk_size})")
        
        with executor_class(max_workers=workers) as executor:
            # Submit jobs in chunks
            futures = []
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                future = executor.submit(self._process_chunk_with_rate_limiting, func, chunk)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_results = future.result(timeout=300)  # 5 minute timeout
                    results.extend(chunk_results)
                    
                    # Check memory pressure periodically
                    if self.memory_manager.check_memory_pressure():
                        self.memory_manager.force_garbage_collection()
                        
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    self.rate_limiter.on_failure(str(e))
        
        return results
    
    def _process_chunk_with_rate_limiting(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items with rate limiting"""
        results = []
        
        for item in chunk:
            try:
                # Apply rate limiting
                self.rate_limiter.wait()
                
                # Process item
                result = func(item)
                results.append(result)
                
                # Report success
                self.rate_limiter.on_success()
                
            except Exception as e:
                logger.error(f"Item processing failed: {e}")
                self.rate_limiter.on_failure(str(e))
                results.append({"status": "failed", "error": str(e), "item": item})
        
        return results

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start monitoring
        start_time = time.time()
        memory_manager = MemoryManager()
        initial_memory = memory_manager.get_memory_usage()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate metrics
            end_time = time.time()
            final_memory = memory_manager.get_memory_usage()
            
            metrics = {
                'function': func.__name__,
                'execution_time': end_time - start_time,
                'memory_delta_mb': final_memory['process_mb'] - initial_memory['process_mb'],
                'peak_memory_mb': final_memory['process_mb']
            }
            
            logger.info(f"Performance: {func.__name__} took {metrics['execution_time']:.2f}s, "
                       f"memory delta: {metrics['memory_delta_mb']:+.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            raise
    
    return wrapper

class DataPipeline:
    """Optimized data processing pipeline"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.cache = IntelligentCache()
        self.processor = OptimizedParallelProcessor(memory_manager=self.memory_manager)
    
    @performance_monitor
    def process_dataframes_efficiently(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process multiple DataFrames efficiently"""
        processed = {}
        
        for symbol, df in dataframes.items():
            # Check cache first
            cache_key = f"processed_{symbol}_{hash(str(df.shape))}"
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                processed[symbol] = cached_result
                continue
            
            # Process DataFrame
            if not df.empty:
                optimized_df = self.memory_manager.memory_efficient_dataframe_processing(df)
                processed[symbol] = optimized_df
                
                # Cache result
                self.cache.put(cache_key, optimized_df)
            
            # Memory management
            if self.memory_manager.check_memory_pressure():
                self.memory_manager.force_garbage_collection()
                self.cache.clear_expired()
        
        return processed
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'memory': self.memory_manager.get_memory_usage(),
            'cache': self.cache.get_stats(),
            'rate_limiter': self.processor.rate_limiter.get_stats()
        }

# Utility functions for common optimizations
def optimize_numpy_operations():
    """Optimize NumPy operations for better performance"""
    # Set optimal thread count for NumPy
    import os
    os.environ['OMP_NUM_THREADS'] = str(min(4, psutil.cpu_count() or 1))
    os.environ['MKL_NUM_THREADS'] = str(min(4, psutil.cpu_count() or 1))

def batch_process_with_memory_limit(items: List[Any], 
                                  process_func: Callable, 
                                  batch_size: int = 100,
                                  memory_limit_mb: float = 1000) -> List[Any]:
    """Process items in batches with memory limit"""
    memory_manager = MemoryManager()
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Check memory before processing batch
        if memory_manager.get_memory_usage()['process_mb'] > memory_limit_mb:
            memory_manager.force_garbage_collection()
        
        # Process batch
        batch_results = [process_func(item) for item in batch]
        results.extend(batch_results)
        
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")
    
    return results

# Initialize global optimizations
optimize_numpy_operations() 