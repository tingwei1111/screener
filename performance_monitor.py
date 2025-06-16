#!/usr/bin/env python3
"""
性能監控模組 - 監控和優化系統性能
提供性能分析、記憶體監控、執行時間統計等功能
"""

import time
import psutil
import threading
import functools
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np

@dataclass
class PerformanceMetrics:
    """性能指標數據類"""
    function_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    cpu_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def memory_delta(self) -> float:
        """記憶體變化量（MB）"""
        return self.memory_after - self.memory_before

class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.metrics: deque = deque(maxlen=max_records)
        self.function_stats: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self, interval: float = 1.0):
        """開始系統監控"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """停止系統監控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            
    def _monitor_system(self, interval: float):
        """系統監控線程"""
        while self._monitoring:
            try:
                # 記錄系統狀態
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                with self.lock:
                    system_metric = PerformanceMetrics(
                        function_name="__system__",
                        execution_time=0,
                        memory_before=memory_info.used / 1024 / 1024,  # MB
                        memory_after=memory_info.used / 1024 / 1024,
                        cpu_percent=cpu_percent
                    )
                    self.metrics.append(system_metric)
                    
                time.sleep(interval)
            except Exception as e:
                print(f"監控線程錯誤: {e}")
                
    def record_metric(self, metric: PerformanceMetrics):
        """記錄性能指標"""
        with self.lock:
            self.metrics.append(metric)
            self.function_stats[metric.function_name].append(metric.execution_time)
            
            # 限制每個函數的統計記錄數量
            if len(self.function_stats[metric.function_name]) > 100:
                self.function_stats[metric.function_name] = \
                    self.function_stats[metric.function_name][-100:]
    
    def get_function_stats(self, function_name: str) -> Dict[str, float]:
        """獲取函數性能統計"""
        with self.lock:
            if function_name not in self.function_stats:
                return {}
                
            times = self.function_stats[function_name]
            if not times:
                return {}
                
            return {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times),
                'p95_time': np.percentile(times, 95),
                'p99_time': np.percentile(times, 99)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """獲取所有函數的性能統計"""
        with self.lock:
            return {
                func_name: self.get_function_stats(func_name)
                for func_name in self.function_stats.keys()
                if func_name != "__system__"
            }
    
    def get_system_stats(self) -> Dict[str, float]:
        """獲取系統性能統計"""
        with self.lock:
            system_metrics = [m for m in self.metrics if m.function_name == "__system__"]
            
            if not system_metrics:
                return {}
                
            recent_metrics = system_metrics[-60:]  # 最近60個記錄
            
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_before for m in recent_metrics]
            
            return {
                'avg_cpu_percent': np.mean(cpu_values),
                'max_cpu_percent': max(cpu_values),
                'current_memory_mb': memory_values[-1] if memory_values else 0,
                'avg_memory_mb': np.mean(memory_values),
                'max_memory_mb': max(memory_values)
            }
    
    def get_slow_functions(self, threshold_ms: float = 100) -> List[Dict[str, Any]]:
        """獲取執行緩慢的函數"""
        threshold_s = threshold_ms / 1000
        slow_functions = []
        
        for func_name, stats in self.get_all_stats().items():
            if stats.get('avg_time', 0) > threshold_s:
                slow_functions.append({
                    'function': func_name,
                    'avg_time_ms': stats['avg_time'] * 1000,
                    'max_time_ms': stats['max_time'] * 1000,
                    'call_count': stats['count']
                })
        
        return sorted(slow_functions, key=lambda x: x['avg_time_ms'], reverse=True)
    
    def clear_stats(self):
        """清除統計數據"""
        with self.lock:
            self.metrics.clear()
            self.function_stats.clear()
    
    def print_report(self):
        """打印性能報告"""
        print("=" * 60)
        print("性能監控報告")
        print("=" * 60)
        
        # 系統統計
        system_stats = self.get_system_stats()
        if system_stats:
            print(f"\n系統狀態:")
            print(f"  平均CPU使用率: {system_stats.get('avg_cpu_percent', 0):.1f}%")
            print(f"  最大CPU使用率: {system_stats.get('max_cpu_percent', 0):.1f}%")
            print(f"  當前記憶體使用: {system_stats.get('current_memory_mb', 0):.1f} MB")
            print(f"  平均記憶體使用: {system_stats.get('avg_memory_mb', 0):.1f} MB")
            print(f"  最大記憶體使用: {system_stats.get('max_memory_mb', 0):.1f} MB")
        
        # 函數統計
        all_stats = self.get_all_stats()
        if all_stats:
            print(f"\n函數性能統計:")
            for func_name, stats in sorted(all_stats.items(), 
                                         key=lambda x: x[1].get('avg_time', 0), 
                                         reverse=True):
                print(f"  {func_name}:")
                print(f"    調用次數: {stats.get('count', 0)}")
                print(f"    平均執行時間: {stats.get('avg_time', 0)*1000:.2f} ms")
                print(f"    最大執行時間: {stats.get('max_time', 0)*1000:.2f} ms")
                print(f"    P95執行時間: {stats.get('p95_time', 0)*1000:.2f} ms")
        
        # 慢函數警告
        slow_functions = self.get_slow_functions()
        if slow_functions:
            print(f"\n⚠️  執行緩慢的函數 (>100ms):")
            for func_info in slow_functions[:5]:  # 只顯示前5個
                print(f"  {func_info['function']}: {func_info['avg_time_ms']:.2f} ms 平均")
        
        print("=" * 60)

# 全域性能監控器實例
_global_monitor = PerformanceMonitor()

def performance_monitor(func: Callable = None, *, monitor_memory: bool = True):
    """
    性能監控裝飾器
    
    Args:
        func: 被裝飾的函數
        monitor_memory: 是否監控記憶體使用
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # 記錄開始狀態
            start_time = time.time()
            start_memory = 0
            
            if monitor_memory:
                try:
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except:
                    pass
            
            try:
                # 執行函數
                result = f(*args, **kwargs)
                
                # 記錄結束狀態
                end_time = time.time()
                end_memory = start_memory
                
                if monitor_memory:
                    try:
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    except:
                        pass
                
                # 記錄性能指標
                execution_time = end_time - start_time
                cpu_percent = 0
                
                try:
                    cpu_percent = psutil.cpu_percent()
                except:
                    pass
                
                metric = PerformanceMetrics(
                    function_name=f"{f.__module__}.{f.__name__}",
                    execution_time=execution_time,
                    memory_before=start_memory,
                    memory_after=end_memory,
                    cpu_percent=cpu_percent
                )
                
                _global_monitor.record_metric(metric)
                
                return result
                
            except Exception as e:
                # 即使發生異常也記錄性能數據
                end_time = time.time()
                execution_time = end_time - start_time
                
                metric = PerformanceMetrics(
                    function_name=f"{f.__module__}.{f.__name__}",
                    execution_time=execution_time,
                    memory_before=start_memory,
                    memory_after=start_memory,
                    cpu_percent=0
                )
                
                _global_monitor.record_metric(metric)
                raise
                
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

def get_monitor() -> PerformanceMonitor:
    """獲取全域性能監控器"""
    return _global_monitor

def start_monitoring(interval: float = 1.0):
    """開始性能監控"""
    _global_monitor.start_monitoring(interval)

def stop_monitoring():
    """停止性能監控"""
    _global_monitor.stop_monitoring()

def print_performance_report():
    """打印性能報告"""
    _global_monitor.print_report()

def clear_performance_stats():
    """清除性能統計"""
    _global_monitor.clear_stats()

# 使用示例
if __name__ == "__main__":
    # 示例：監控一個函數
    @performance_monitor
    def example_function():
        """示例函數"""
        import time
        import numpy as np
        
        # 模擬一些計算
        data = np.random.rand(100000)
        result = np.sum(data)
        time.sleep(0.1)  # 模擬I/O操作
        return result
    
    # 測試性能監控
    print("開始性能監控測試...")
    start_monitoring()
    
    # 執行多次函數調用
    for i in range(5):
        result = example_function()
        print(f"第{i+1}次調用結果: {result:.4f}")
        time.sleep(0.5)
    
    # 打印報告
    print_performance_report()
    
    stop_monitoring()