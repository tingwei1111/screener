"""
Configuration Management System
==============================

Centralized configuration management for the Screener project.
Provides type-safe configuration with validation and environment variable support.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration settings"""
    stocksymbol_key: str = ""
    polygon_key: str = ""
    binance_timeout: int = 300
    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 60.0
    backoff_factor: float = 2.0

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    max_workers: Optional[int] = None
    memory_limit_percent: float = 75.0
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    chunk_size: Optional[int] = None
    enable_caching: bool = True
    enable_memory_optimization: bool = True

@dataclass
class AnalysisConfig:
    """Analysis parameters"""
    # Timeframe settings
    default_timeframe: str = "15m"
    default_days: int = 3
    supported_timeframes: List[str] = field(default_factory=lambda: [
        "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d"
    ])
    
    # RS Score calculation
    sma_periods: List[int] = field(default_factory=lambda: [30, 45, 60])
    atr_period: int = 60
    required_bars_multiplier: float = 1.2
    
    # Stock screening
    stock_sma_periods: List[int] = field(default_factory=lambda: [20, 30, 45, 50, 60, 150, 200])
    min_turnover: int = 10000000
    days_traceback_1d: int = 252
    days_traceback_1h: int = 126

@dataclass
class DTWConfig:
    """DTW analysis configuration"""
    window_ratio: float = 0.2
    window_ratio_diff: float = 0.1
    max_point_distance: float = 0.66
    max_point_distance_diff: float = 0.5
    
    # ShapeDTW parameters
    shapedtw_balance_pd_ratio: float = 4.0
    price_weight: float = 0.4
    diff_weight: float = 0.6
    slope_window_size: int = 5
    paa_window_size: int = 5
    
    # Window scaling factors
    window_scale_factors: List[float] = field(default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])
    min_similarity_score: float = 0.25

@dataclass
class OutputConfig:
    """Output and logging configuration"""
    output_dir: str = "output"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    save_detailed_stats: bool = True
    tradingview_format: bool = True

@dataclass
class ScreenerConfig:
    """Main configuration class combining all settings"""
    api: APIConfig = field(default_factory=APIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    dtw: DTWConfig = field(default_factory=DTWConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate timeframes
        if self.analysis.default_timeframe not in self.analysis.supported_timeframes:
            raise ValueError(f"Default timeframe {self.analysis.default_timeframe} not in supported timeframes")
        
        # Validate performance settings
        if self.performance.memory_limit_percent <= 0 or self.performance.memory_limit_percent > 100:
            raise ValueError("Memory limit percent must be between 0 and 100")
        
        # Validate DTW weights
        if abs(self.dtw.price_weight + self.dtw.diff_weight - 1.0) > 0.001:
            logger.warning(f"DTW weights don't sum to 1.0: {self.dtw.price_weight + self.dtw.diff_weight}")
        
        # Validate directories
        Path(self.output.output_dir).mkdir(parents=True, exist_ok=True)

class ConfigManager:
    """Configuration manager with file loading and environment variable support"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.yaml"
        self.config = ScreenerConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables"""
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                self._load_from_file()
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # Override with environment variables
        self._load_from_env()
        
        # Load API keys from separate file
        self._load_api_keys()
    
    def _load_from_file(self):
        """Load configuration from YAML or JSON file"""
        with open(self.config_file, 'r') as f:
            if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Update configuration with loaded data
        self._update_config_from_dict(data)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'SCREENER_API_STOCKSYMBOL_KEY': ('api', 'stocksymbol_key'),
            'SCREENER_API_POLYGON_KEY': ('api', 'polygon_key'),
            'SCREENER_PERFORMANCE_MAX_WORKERS': ('performance', 'max_workers'),
            'SCREENER_PERFORMANCE_MEMORY_LIMIT': ('performance', 'memory_limit_percent'),
            'SCREENER_ANALYSIS_DEFAULT_TIMEFRAME': ('analysis', 'default_timeframe'),
            'SCREENER_ANALYSIS_DEFAULT_DAYS': ('analysis', 'default_days'),
            'SCREENER_OUTPUT_DIR': ('output', 'output_dir'),
            'SCREENER_LOG_LEVEL': ('output', 'log_level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value, section, key)
                setattr(getattr(self.config, section), key, converted_value)
                logger.debug(f"Set {section}.{key} = {converted_value} from {env_var}")
    
    def _load_api_keys(self):
        """Load API keys from api_keys.json"""
        api_keys_file = "api_keys.json"
        if os.path.exists(api_keys_file):
            try:
                with open(api_keys_file, 'r') as f:
                    api_keys = json.load(f)
                
                if 'stocksymbol' in api_keys:
                    self.config.api.stocksymbol_key = api_keys['stocksymbol']
                if 'polygon' in api_keys:
                    self.config.api.polygon_key = api_keys['polygon']
                
                logger.info("API keys loaded from api_keys.json")
            except Exception as e:
                logger.warning(f"Failed to load API keys: {e}")
    
    def _convert_env_value(self, value: str, section: str, key: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Get the current value to determine type
        current_value = getattr(getattr(self.config, section), key)
        
        if isinstance(current_value, bool):
            return value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(current_value, int):
            return int(value)
        elif isinstance(current_value, float):
            return float(value)
        elif isinstance(current_value, list):
            return value.split(',')
        else:
            return value
    
    def _update_config_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section_name, section_data in data.items():
            if hasattr(self.config, section_name) and isinstance(section_data, dict):
                section = getattr(self.config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def save_config(self, filename: Optional[str] = None):
        """Save current configuration to file"""
        filename = filename or self.config_file
        
        # Convert to dictionary
        config_dict = asdict(self.config)
        
        # Save as YAML
        with open(filename, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {filename}")
    
    def get_config(self) -> ScreenerConfig:
        """Get the current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
            if '.' in key:
                section, attr = key.split('.', 1)
                if hasattr(self.config, section):
                    setattr(getattr(self.config, section), attr, value)
            else:
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        return self.config.api
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        return self.config.performance
    
    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration"""
        return self.config.analysis
    
    def get_dtw_config(self) -> DTWConfig:
        """Get DTW configuration"""
        return self.config.dtw
    
    def get_output_config(self) -> OutputConfig:
        """Get output configuration"""
        return self.config.output
    
    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print("=" * 50)
        
        sections = [
            ("API", self.config.api),
            ("Performance", self.config.performance),
            ("Analysis", self.config.analysis),
            ("DTW", self.config.dtw),
            ("Output", self.config.output)
        ]
        
        for section_name, section in sections:
            print(f"\n{section_name}:")
            print("-" * 20)
            for key, value in asdict(section).items():
                # Hide sensitive information
                if 'key' in key.lower() and value:
                    value = '*' * 8
                print(f"  {key}: {value}")

def create_default_config_file(filename: str = "config.yaml"):
    """Create a default configuration file"""
    config = ScreenerConfig()
    
    config_dict = asdict(config)
    
    with open(filename, 'w') as f:
        f.write("# Screener Configuration File\n")
        f.write("# This file contains all configuration settings for the Screener project\n\n")
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration file created: {filename}")

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> ScreenerConfig:
    """Get the current configuration"""
    return get_config_manager().get_config()

# Convenience functions for specific configurations
def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_config().api

def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return get_config().performance

def get_analysis_config() -> AnalysisConfig:
    """Get analysis configuration"""
    return get_config().analysis

def get_dtw_config() -> DTWConfig:
    """Get DTW configuration"""
    return get_config().dtw

def get_output_config() -> OutputConfig:
    """Get output configuration"""
    return get_config().output 