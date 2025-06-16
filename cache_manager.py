#!/usr/bin/env python3
"""
緩存管理模組 - 智能緩存系統
提供數據緩存、過期管理、記憶體優化等功能
"""

import time
import pickle
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json
import gzip
import os

@dataclass
class CacheEntry:
    """緩存條目"""
    data: Any
    timestamp: float
    ttl: float  # 生存時間（秒）
    size_bytes: int
    access_count: int = 0
    last_access: float = 0
    
    def __post_init__(self):
        if self.last_access == 0:
            self.last_access = self.timestamp
    
    @property
    def is_expired(self) -> bool:
        """檢查是否過期"""
        if self.ttl <= 0:  # 永不過期
            return False
        return time.time() - self.timestamp > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """獲取緩存年齡（秒）"""
        return time.time() - self.timestamp

class InMemoryCache:
    """內存緩存"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self._current_size = 0
        
    def _calculate_size(self, data: Any) -> int:
        """計算數據大小"""
        try:
            return len(pickle.dumps(data))
        except:
            return 1024  # 默認1KB
    
    def _evict_lru(self, needed_space: int):
        """LRU淘汰策略"""
        if not self.cache:
            return
            
        # 按最後訪問時間排序
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_access
        )
        
        freed_space = 0
        keys_to_remove = []
        
        for key, entry in sorted_entries:
            keys_to_remove.append(key)
            freed_space += entry.size_bytes
            
            if freed_space >= needed_space:
                break
        
        # 刪除選中的條目
        for key in keys_to_remove:
            if key in self.cache:
                self._current_size -= self.cache[key].size_bytes
                del self.cache[key]
    
    def _cleanup_expired(self):
        """清理過期條目"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            if key in self.cache:
                self._current_size -= self.cache[key].size_bytes
                del self.cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """獲取緩存數據"""
        with self.lock:
            if key not in self.cache:
                return None
                
            entry = self.cache[key]
            
            # 檢查過期
            if entry.is_expired:
                self._current_size -= entry.size_bytes
                del self.cache[key]
                return None
            
            # 更新訪問統計
            entry.access_count += 1
            entry.last_access = time.time()
            
            return entry.data
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """設置緩存數據"""
        with self.lock:
            # 計算數據大小
            data_size = self._calculate_size(data)
            
            # 檢查大小限制
            if data_size > self.max_size_bytes:
                return False
            
            # 清理過期條目
            self._cleanup_expired()
            
            # 如果需要，進行LRU淘汰
            available_space = self.max_size_bytes - self._current_size
            if key in self.cache:
                available_space += self.cache[key].size_bytes
                
            if data_size > available_space:
                needed_space = data_size - available_space
                self._evict_lru(needed_space)
            
            # 如果是更新，先減去舊的大小
            if key in self.cache:
                self._current_size -= self.cache[key].size_bytes
            
            # 添加新條目
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=data_size
            )
            
            self.cache[key] = entry
            self._current_size += data_size
            
            return True
    
    def delete(self, key: str) -> bool:
        """刪除緩存條目"""
        with self.lock:
            if key in self.cache:
                self._current_size -= self.cache[key].size_bytes
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """清空緩存"""
        with self.lock:
            self.cache.clear()
            self._current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        with self.lock:
            total_entries = len(self.cache)
            expired_count = sum(1 for entry in self.cache.values() if entry.is_expired)
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_count,
                'current_size_mb': self._current_size / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'usage_percent': (self._current_size / self.max_size_bytes) * 100
            }

class DiskCache:
    """磁盤緩存"""
    
    def __init__(self, cache_dir: str = ".cache", compress: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.compress = compress
        self.index_file = self.cache_dir / "index.json"
        self.lock = threading.RLock()
        self._load_index()
    
    def _load_index(self):
        """載入索引"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {}
        except:
            self.index = {}
    
    def _save_index(self):
        """保存索引"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            print(f"保存緩存索引失敗: {e}")
    
    def _get_cache_file(self, key: str) -> Path:
        """獲取緩存文件路徑"""
        # 使用MD5避免文件名過長或包含特殊字符
        key_hash = hashlib.md5(key.encode()).hexdigest()
        filename = f"{key_hash}.pkl"
        if self.compress:
            filename += ".gz"
        return self.cache_dir / filename
    
    def _is_expired(self, metadata: Dict) -> bool:
        """檢查是否過期"""
        ttl = metadata.get('ttl', 0)
        if ttl <= 0:
            return False
        timestamp = metadata.get('timestamp', 0)
        return time.time() - timestamp > ttl
    
    def get(self, key: str) -> Optional[Any]:
        """獲取緩存數據"""
        with self.lock:
            if key not in self.index:
                return None
            
            metadata = self.index[key]
            
            # 檢查過期
            if self._is_expired(metadata):
                self.delete(key)
                return None
            
            # 讀取數據
            cache_file = self._get_cache_file(key)
            if not cache_file.exists():
                self.delete(key)
                return None
            
            try:
                if self.compress:
                    with gzip.open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                
                # 更新訪問統計
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                metadata['last_access'] = time.time()
                self._save_index()
                
                return data
                
            except Exception as e:
                print(f"讀取緩存文件失敗: {e}")
                self.delete(key)
                return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """設置緩存數據"""
        with self.lock:
            cache_file = self._get_cache_file(key)
            
            try:
                # 寫入數據
                if self.compress:
                    with gzip.open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                else:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                
                # 更新索引
                self.index[key] = {
                    'timestamp': time.time(),
                    'ttl': ttl or 0,
                    'file_size': cache_file.stat().st_size,
                    'access_count': 0,
                    'last_access': time.time()
                }
                
                self._save_index()
                return True
                
            except Exception as e:
                print(f"寫入緩存文件失敗: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """刪除緩存條目"""
        with self.lock:
            if key not in self.index:
                return False
            
            # 刪除文件
            cache_file = self._get_cache_file(key)
            try:
                if cache_file.exists():
                    cache_file.unlink()
            except:
                pass
            
            # 從索引中移除
            del self.index[key]
            self._save_index()
            
            return True
    
    def clear(self):
        """清空緩存"""
        with self.lock:
            # 刪除所有緩存文件
            for cache_file in self.cache_dir.glob("*.pkl*"):
                try:
                    cache_file.unlink()
                except:
                    pass
            
            # 清空索引
            self.index = {}
            self._save_index()
    
    def cleanup_expired(self):
        """清理過期緩存"""
        with self.lock:
            expired_keys = [
                key for key, metadata in self.index.items()
                if self._is_expired(metadata)
            ]
            
            for key in expired_keys:
                self.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        with self.lock:
            total_entries = len(self.index)
            expired_count = sum(
                1 for metadata in self.index.values()
                if self._is_expired(metadata)
            )
            
            total_size = sum(
                metadata.get('file_size', 0)
                for metadata in self.index.values()
            )
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_count,
                'total_size_mb': total_size / 1024 / 1024,
                'cache_dir': str(self.cache_dir)
            }

class CacheManager:
    """統一緩存管理器"""
    
    def __init__(self, 
                 memory_cache_mb: int = 50,
                 disk_cache_dir: str = ".cache",
                 default_ttl: int = 3600,
                 use_compression: bool = True):
        
        self.memory_cache = InMemoryCache(memory_cache_mb, default_ttl)
        self.disk_cache = DiskCache(disk_cache_dir, use_compression)
        self.default_ttl = default_ttl
        
    def get(self, key: str, default: Any = None) -> Any:
        """獲取緩存數據（先檢查內存，再檢查磁盤）"""
        # 先檢查內存緩存
        data = self.memory_cache.get(key)
        if data is not None:
            return data
        
        # 再檢查磁盤緩存
        data = self.disk_cache.get(key)
        if data is not None:
            # 將熱數據載入內存
            self.memory_cache.set(key, data)
            return data
        
        return default
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None, 
            memory_only: bool = False) -> bool:
        """設置緩存數據"""
        ttl = ttl or self.default_ttl
        
        # 設置內存緩存
        memory_success = self.memory_cache.set(key, data, ttl)
        
        # 設置磁盤緩存（除非指定僅內存）
        disk_success = True
        if not memory_only:
            disk_success = self.disk_cache.set(key, data, ttl)
        
        return memory_success or disk_success
    
    def delete(self, key: str) -> bool:
        """刪除緩存數據"""
        memory_deleted = self.memory_cache.delete(key)
        disk_deleted = self.disk_cache.delete(key)
        return memory_deleted or disk_deleted
    
    def clear(self):
        """清空所有緩存"""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def cleanup(self):
        """清理過期緩存"""
        self.disk_cache.cleanup_expired()
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        return {
            'memory_cache': self.memory_cache.get_stats(),
            'disk_cache': self.disk_cache.get_stats()
        }

# 緩存裝飾器
def cached(ttl: int = 3600, key_func: Optional[Callable] = None, 
           memory_only: bool = False):
    """
    緩存裝飾器
    
    Args:
        ttl: 緩存生存時間（秒）
        key_func: 自定義key生成函數
        memory_only: 是否僅使用內存緩存
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # 生成緩存key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默認key生成策略
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # 檢查緩存
            cache_manager = get_default_cache_manager()
            cached_result = cache_manager.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # 執行函數並緩存結果
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, memory_only)
            
            return result
        
        return wrapper
    return decorator

# 全域緩存管理器
_default_cache_manager: Optional[CacheManager] = None

def get_default_cache_manager() -> CacheManager:
    """獲取默認緩存管理器"""
    global _default_cache_manager
    if _default_cache_manager is None:
        _default_cache_manager = CacheManager()
    return _default_cache_manager

def configure_cache(memory_cache_mb: int = 50, 
                   disk_cache_dir: str = ".cache",
                   default_ttl: int = 3600):
    """配置全域緩存管理器"""
    global _default_cache_manager
    _default_cache_manager = CacheManager(
        memory_cache_mb, disk_cache_dir, default_ttl
    )

def clear_all_cache():
    """清空所有緩存"""
    get_default_cache_manager().clear()

def print_cache_stats():
    """打印緩存統計"""
    stats = get_default_cache_manager().get_stats()
    print("緩存統計:")
    print(f"內存緩存: {stats['memory_cache']}")
    print(f"磁盤緩存: {stats['disk_cache']}")

# 使用示例
if __name__ == "__main__":
    import numpy as np
    
    # 測試緩存功能
    @cached(ttl=10)
    def expensive_calculation(n: int) -> float:
        """模擬耗時計算"""
        print(f"執行耗時計算: n={n}")
        data = np.random.rand(n, n)
        return np.sum(data)
    
    print("測試緩存功能...")
    
    # 第一次調用（會執行計算）
    result1 = expensive_calculation(1000)
    print(f"第一次結果: {result1:.4f}")
    
    # 第二次調用（使用緩存）
    result2 = expensive_calculation(1000)
    print(f"第二次結果: {result2:.4f}")
    
    # 打印緩存統計
    print_cache_stats()