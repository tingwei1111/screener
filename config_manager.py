#!/usr/bin/env python3
"""
配置管理器 - 優化版本
統一管理所有模塊的配置文件，支持緩存和並發安全
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from functools import lru_cache
import threading
import time

@dataclass
class ScreenerConfig:
    """主配置類"""
    # API配置
    binance_api_key: str = ""
    binance_secret_key: str = ""
    polygon_api_key: str = ""
    
    # 數據配置
    default_timeframe: str = "1d"
    default_days: int = 365
    cache_enabled: bool = True
    cache_duration: int = 3600  # 秒
    
    # 篩選配置
    min_rs_score: float = 7.0
    top_results: int = 20
    min_volume: float = 1000000  # 最小成交量
    
    # 機器學習配置
    ml_model_path: str = "models/"
    lstm_epochs: int = 100
    rf_n_estimators: int = 100
    
    # 投資組合配置
    risk_free_rate: float = 0.02
    max_weight: float = 0.4
    min_weight: float = 0.0
    
    # 監控配置
    monitor_interval: int = 60  # 秒
    alert_cooldown: int = 3600  # 秒
    
    # 交易配置
    paper_trading: bool = True
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10%
    stop_loss: float = 0.05  # 5%
    take_profit: float = 0.15  # 15%
    
    # Web配置
    web_port: int = 8501
    web_host: str = "localhost"
    
    # 通知配置
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: list = None
    
    webhook_enabled: bool = False
    webhook_url: str = ""
    
    desktop_enabled: bool = True

class ConfigManager:
    """配置管理器 - 線程安全版本，支持緩存"""
    
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(cls, config_file: str = "screener_config.json"):
        """單例模式，每個配置文件一個實例"""
        config_path = str(Path(config_file).resolve())
        
        if config_path not in cls._instances:
            with cls._lock:
                if config_path not in cls._instances:
                    cls._instances[config_path] = super().__new__(cls)
        
        return cls._instances[config_path]
    
    def __init__(self, config_file: str = "screener_config.json"):
        if hasattr(self, '_initialized'):
            return
            
        self.config_file = Path(config_file)
        self._config = None
        self._last_modified = 0
        self._lock = threading.RLock()
        self._cache_duration = 60  # 緩存60秒
        self._initialized = True
    
    @property 
    def config(self) -> ScreenerConfig:
        """線程安全的配置屬性，支持自動重載"""
        with self._lock:
            now = time.time()
            file_modified = self.config_file.stat().st_mtime if self.config_file.exists() else 0
            
            # 檢查是否需要重載配置
            if (self._config is None or 
                file_modified > self._last_modified or 
                now - self._last_modified > self._cache_duration):
                
                self._config = self._load_config_from_file()
                self._last_modified = now
            
            return self._config
    
    def _load_config_from_file(self) -> ScreenerConfig:
        """從文件載入配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 處理列表字段
                if data.get('email_recipients') is None:
                    data['email_recipients'] = []
                
                return ScreenerConfig(**data)
            except Exception as e:
                print(f"載入配置文件失敗: {e}")
                print("使用默認配置")
        
        return ScreenerConfig()
    
    def load_config(self) -> ScreenerConfig:
        """向後兼容的載入配置方法"""
        return self.config
    
    def save_config(self):
        """保存配置"""
        try:
            config_dict = asdict(self.config)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"保存配置失敗: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """獲取配置值"""
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any):
        """設置配置值"""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save_config()
        else:
            raise ValueError(f"未知的配置項: {key}")
    
    def update(self, **kwargs):
        """批量更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"警告: 未知的配置項 {key}")
        self.save_config()
    
    def reset_to_default(self):
        """重置為默認配置"""
        self.config = ScreenerConfig()
        self.save_config()
    
    def setup_api_keys(self):
        """交互式設置API密鑰"""
        print("=== API密鑰設置 ===")
        
        # Binance API
        print("\n1. Binance API (用於加密貨幣數據)")
        api_key = input(f"API Key [{self.config.binance_api_key[:10]}...]: ").strip()
        if api_key:
            self.config.binance_api_key = api_key
        
        secret_key = input(f"Secret Key [{self.config.binance_secret_key[:10]}...]: ").strip()
        if secret_key:
            self.config.binance_secret_key = secret_key
        
        # Polygon API
        print("\n2. Polygon API (用於股票數據)")
        polygon_key = input(f"API Key [{self.config.polygon_api_key[:10]}...]: ").strip()
        if polygon_key:
            self.config.polygon_api_key = polygon_key
        
        self.save_config()
        print("\nAPI密鑰設置完成！")
    
    def setup_notifications(self):
        """交互式設置通知"""
        print("=== 通知設置 ===")
        
        # 郵件通知
        print("\n1. 郵件通知")
        enable_email = input(f"啟用郵件通知? (y/n) [{self.config.email_enabled}]: ").strip().lower()
        if enable_email in ['y', 'yes', 'true']:
            self.config.email_enabled = True
            
            smtp_server = input(f"SMTP服務器 [{self.config.email_smtp_server}]: ").strip()
            if smtp_server:
                self.config.email_smtp_server = smtp_server
            
            smtp_port = input(f"SMTP端口 [{self.config.email_smtp_port}]: ").strip()
            if smtp_port:
                self.config.email_smtp_port = int(smtp_port)
            
            username = input(f"郵箱用戶名 [{self.config.email_username}]: ").strip()
            if username:
                self.config.email_username = username
            
            password = input("郵箱密碼 (不顯示): ").strip()
            if password:
                self.config.email_password = password
            
            recipients = input("收件人列表 (逗號分隔): ").strip()
            if recipients:
                self.config.email_recipients = [r.strip() for r in recipients.split(',')]
        
        # Webhook通知
        print("\n2. Webhook通知")
        enable_webhook = input(f"啟用Webhook通知? (y/n) [{self.config.webhook_enabled}]: ").strip().lower()
        if enable_webhook in ['y', 'yes', 'true']:
            self.config.webhook_enabled = True
            
            webhook_url = input(f"Webhook URL [{self.config.webhook_url}]: ").strip()
            if webhook_url:
                self.config.webhook_url = webhook_url
        
        # 桌面通知
        print("\n3. 桌面通知")
        enable_desktop = input(f"啟用桌面通知? (y/n) [{self.config.desktop_enabled}]: ").strip().lower()
        if enable_desktop in ['y', 'yes', 'true']:
            self.config.desktop_enabled = True
        elif enable_desktop in ['n', 'no', 'false']:
            self.config.desktop_enabled = False
        
        self.save_config()
        print("\n通知設置完成！")
    
    def print_config(self):
        """打印當前配置"""
        print("=== 當前配置 ===")
        config_dict = asdict(self.config)
        
        # 隱藏敏感信息
        sensitive_keys = ['binance_secret_key', 'email_password']
        for key in sensitive_keys:
            if key in config_dict and config_dict[key]:
                config_dict[key] = "*" * 10
        
        for key, value in config_dict.items():
            print(f"{key}: {value}")

def main():
    """配置管理器命令行界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Screener 配置管理器")
    parser.add_argument('--setup-api', action='store_true', help='設置API密鑰')
    parser.add_argument('--setup-notifications', action='store_true', help='設置通知')
    parser.add_argument('--show', action='store_true', help='顯示當前配置')
    parser.add_argument('--reset', action='store_true', help='重置為默認配置')
    parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='設置配置項')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    if args.setup_api:
        config_manager.setup_api_keys()
    elif args.setup_notifications:
        config_manager.setup_notifications()
    elif args.show:
        config_manager.print_config()
    elif args.reset:
        confirm = input("確定要重置所有配置嗎? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            config_manager.reset_to_default()
            print("配置已重置為默認值")
    elif args.set:
        key, value = args.set
        try:
            # 嘗試轉換數據類型
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)
            
            config_manager.set(key, value)
            print(f"已設置 {key} = {value}")
        except Exception as e:
            print(f"設置失敗: {e}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 