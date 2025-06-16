"""
Real-time Market Monitor and Alert System
----------------------------------------
This module provides real-time monitoring of cryptocurrency and stock markets,
sending alerts when predefined conditions are met.

Features:
- Real-time price monitoring
- RS Score threshold alerts
- Pattern breakout detection
- Multi-channel notifications (email, webhook, desktop)
- Customizable alert conditions
"""

import time
import json
import smtplib
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.downloader import CryptoDownloader
from src.common import TrendAnalysisConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlertCondition:
    """Alert condition configuration"""
    symbol: str
    condition_type: str  # 'rs_score', 'price_breakout', 'volume_spike', 'pattern_match'
    threshold: float
    timeframe: str = "15m"
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 60  # Minimum time between alerts

@dataclass
class NotificationConfig:
    """Notification configuration"""
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    
    webhook_enabled: bool = False
    webhook_url: str = ""
    
    desktop_enabled: bool = True

class RealTimeMonitor:
    """Real-time market monitoring system"""
    
    def __init__(self, config_file: str = "monitor_config.json"):
        """Initialize monitor with configuration"""
        self.config_file = config_file
        self.alert_conditions: List[AlertCondition] = []
        self.notification_config = NotificationConfig()
        self.downloader = CryptoDownloader()
        self.is_running = False
        self.monitor_thread = None
        
        # Load configuration
        self.load_config()
        
        # Data cache
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                
            # Load alert conditions
            for condition_data in config_data.get('alert_conditions', []):
                condition = AlertCondition(**condition_data)
                self.alert_conditions.append(condition)
                
            # Load notification config
            notification_data = config_data.get('notification_config', {})
            self.notification_config = NotificationConfig(**notification_data)
            
            logger.info(f"Loaded {len(self.alert_conditions)} alert conditions")
            
        except FileNotFoundError:
            logger.info("Config file not found, creating default configuration")
            self.create_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            
    def create_default_config(self):
        """Create default configuration file"""
        default_config = {
            "alert_conditions": [
                {
                    "symbol": "BTCUSDT",
                    "condition_type": "rs_score",
                    "threshold": 8.0,
                    "timeframe": "15m",
                    "enabled": True,
                    "cooldown_minutes": 60
                },
                {
                    "symbol": "ETHUSDT", 
                    "condition_type": "price_breakout",
                    "threshold": 0.05,  # 5% breakout
                    "timeframe": "1h",
                    "enabled": True,
                    "cooldown_minutes": 30
                }
            ],
            "notification_config": {
                "email_enabled": False,
                "email_smtp_server": "smtp.gmail.com",
                "email_smtp_port": 587,
                "email_username": "",
                "email_password": "",
                "email_recipients": [],
                "webhook_enabled": False,
                "webhook_url": "",
                "desktop_enabled": True
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2, default=str)
            
        logger.info(f"Created default config file: {self.config_file}")
        
    def add_alert_condition(self, symbol: str, condition_type: str, threshold: float, 
                          timeframe: str = "15m", cooldown_minutes: int = 60):
        """Add new alert condition"""
        condition = AlertCondition(
            symbol=symbol,
            condition_type=condition_type,
            threshold=threshold,
            timeframe=timeframe,
            cooldown_minutes=cooldown_minutes
        )
        self.alert_conditions.append(condition)
        self.save_config()
        logger.info(f"Added alert condition: {symbol} {condition_type} {threshold}")
        
    def remove_alert_condition(self, symbol: str, condition_type: str):
        """Remove alert condition"""
        self.alert_conditions = [
            c for c in self.alert_conditions 
            if not (c.symbol == symbol and c.condition_type == condition_type)
        ]
        self.save_config()
        logger.info(f"Removed alert condition: {symbol} {condition_type}")
        
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            "alert_conditions": [
                {
                    "symbol": c.symbol,
                    "condition_type": c.condition_type,
                    "threshold": c.threshold,
                    "timeframe": c.timeframe,
                    "enabled": c.enabled,
                    "cooldown_minutes": c.cooldown_minutes,
                    "last_triggered": c.last_triggered.isoformat() if c.last_triggered else None
                }
                for c in self.alert_conditions
            ],
            "notification_config": {
                "email_enabled": self.notification_config.email_enabled,
                "email_smtp_server": self.notification_config.email_smtp_server,
                "email_smtp_port": self.notification_config.email_smtp_port,
                "email_username": self.notification_config.email_username,
                "email_password": self.notification_config.email_password,
                "email_recipients": self.notification_config.email_recipients or [],
                "webhook_enabled": self.notification_config.webhook_enabled,
                "webhook_url": self.notification_config.webhook_url,
                "desktop_enabled": self.notification_config.desktop_enabled
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
            
    def calculate_rs_score(self, df: pd.DataFrame) -> float:
        """Calculate RS score for given dataframe"""
        if len(df) < 60:  # Need minimum data
            return 0.0
            
        # Calculate SMAs
        df['SMA_30'] = df['Close'].rolling(30).mean()
        df['SMA_45'] = df['Close'].rolling(45).mean() 
        df['SMA_60'] = df['Close'].rolling(60).mean()
        
        # Calculate ATR
        df['TR'] = np.maximum(df['High'] - df['Low'],
                             np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                      abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(60).mean()
        
        # Calculate RS components
        bars = len(df)
        weights = np.exp(2 * np.log(2) * np.arange(bars) / bars)
        
        rs_values = []
        for i in range(len(df)):
            if pd.isna(df['ATR'].iloc[i]) or df['ATR'].iloc[i] == 0:
                continue
                
            price = df['Close'].iloc[i]
            sma30 = df['SMA_30'].iloc[i]
            sma45 = df['SMA_45'].iloc[i]
            sma60 = df['SMA_60'].iloc[i]
            atr = df['ATR'].iloc[i]
            
            if pd.isna(sma30) or pd.isna(sma45) or pd.isna(sma60):
                continue
                
            rs_value = ((price - sma30) + (price - sma45) + (price - sma60) + 
                       (sma30 - sma45) + (sma30 - sma60) + (sma45 - sma60)) / atr
            rs_values.append(rs_value)
            
        if not rs_values:
            return 0.0
            
        # Apply weights to recent values
        recent_values = rs_values[-min(len(rs_values), len(weights)):]
        recent_weights = weights[-len(recent_values):]
        
        return np.average(recent_values, weights=recent_weights)
        
    def check_price_breakout(self, df: pd.DataFrame, threshold: float) -> bool:
        """Check for price breakout"""
        if len(df) < 20:
            return False
            
        # Calculate recent high/low
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()
        current_price = df['Close'].iloc[-1]
        
        # Check for breakout
        breakout_up = current_price > recent_high * (1 + threshold)
        breakout_down = current_price < recent_low * (1 - threshold)
        
        return breakout_up or breakout_down
        
    def check_volume_spike(self, df: pd.DataFrame, threshold: float) -> bool:
        """Check for volume spike"""
        if len(df) < 20 or 'Volume' not in df.columns:
            return False
            
        avg_volume = df['Volume'].tail(20).mean()
        current_volume = df['Volume'].iloc[-1]
        
        return current_volume > avg_volume * (1 + threshold)
        
    def update_price_data(self, symbol: str, timeframe: str):
        """Update price data for symbol"""
        try:
            # Calculate time range (last 24 hours)
            end_ts = int(datetime.now().timestamp())
            start_ts = end_ts - 86400  # 24 hours
            
            success, df = self.downloader.get_data(symbol, start_ts, end_ts, timeframe=timeframe)
            
            if success and not df.empty:
                self.price_cache[f"{symbol}_{timeframe}"] = df
                self.last_update[f"{symbol}_{timeframe}"] = datetime.now()
                return True
            else:
                logger.warning(f"Failed to update data for {symbol} {timeframe}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating price data for {symbol}: {e}")
            return False
            
    def check_alert_conditions(self):
        """Check all alert conditions"""
        for condition in self.alert_conditions:
            if not condition.enabled:
                continue
                
            # Check cooldown
            if (condition.last_triggered and 
                datetime.now() - condition.last_triggered < timedelta(minutes=condition.cooldown_minutes)):
                continue
                
            # Update data if needed
            cache_key = f"{condition.symbol}_{condition.timeframe}"
            if (cache_key not in self.last_update or 
                datetime.now() - self.last_update[cache_key] > timedelta(minutes=5)):
                self.update_price_data(condition.symbol, condition.timeframe)
                
            # Get cached data
            if cache_key not in self.price_cache:
                continue
                
            df = self.price_cache[cache_key]
            
            # Check condition
            alert_triggered = False
            alert_message = ""
            
            if condition.condition_type == "rs_score":
                rs_score = self.calculate_rs_score(df.copy())
                if rs_score >= condition.threshold:
                    alert_triggered = True
                    alert_message = f"RS Score Alert: {condition.symbol} RS={rs_score:.2f} (threshold: {condition.threshold})"
                    
            elif condition.condition_type == "price_breakout":
                if self.check_price_breakout(df, condition.threshold):
                    alert_triggered = True
                    current_price = df['Close'].iloc[-1]
                    alert_message = f"Price Breakout Alert: {condition.symbol} Price=${current_price:.4f} (threshold: {condition.threshold*100:.1f}%)"
                    
            elif condition.condition_type == "volume_spike":
                if self.check_volume_spike(df, condition.threshold):
                    alert_triggered = True
                    current_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
                    alert_message = f"Volume Spike Alert: {condition.symbol} Volume={current_volume:.0f} (threshold: {condition.threshold*100:.1f}%)"
                    
            # Send alert if triggered
            if alert_triggered:
                self.send_alert(alert_message, condition)
                condition.last_triggered = datetime.now()
                
    def send_alert(self, message: str, condition: AlertCondition):
        """Send alert through configured channels"""
        logger.info(f"ALERT: {message}")
        
        # Desktop notification
        if self.notification_config.desktop_enabled:
            self.send_desktop_notification("Market Alert", message)
            
        # Email notification
        if self.notification_config.email_enabled:
            self.send_email_notification(message)
            
        # Webhook notification
        if self.notification_config.webhook_enabled:
            self.send_webhook_notification(message, condition)
            
    def send_desktop_notification(self, title: str, message: str):
        """Send desktop notification"""
        try:
            import plyer
            plyer.notification.notify(
                title=title,
                message=message,
                timeout=10
            )
        except ImportError:
            logger.warning("plyer not installed, desktop notifications disabled")
        except Exception as e:
            logger.error(f"Error sending desktop notification: {e}")
            
    def send_email_notification(self, message: str):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.notification_config.email_username
            msg['To'] = ", ".join(self.notification_config.email_recipients)
            msg['Subject'] = "Screener Market Alert"
            
            body = f"""
Market Alert from Screener Tool

{message}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is an automated message from your Screener monitoring system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.notification_config.email_smtp_server, 
                                self.notification_config.email_smtp_port)
            server.starttls()
            server.login(self.notification_config.email_username, 
                        self.notification_config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.notification_config.email_username, 
                          self.notification_config.email_recipients, text)
            server.quit()
            
            logger.info("Email notification sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            
    def send_webhook_notification(self, message: str, condition: AlertCondition):
        """Send webhook notification"""
        try:
            payload = {
                "text": message,
                "symbol": condition.symbol,
                "condition_type": condition.condition_type,
                "threshold": condition.threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(self.notification_config.webhook_url, 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Webhook notification sent successfully")
            else:
                logger.error(f"Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            
    def start_monitoring(self, check_interval: int = 60):
        """Start real-time monitoring"""
        if self.is_running:
            logger.warning("Monitor is already running")
            return
            
        self.is_running = True
        logger.info(f"Starting real-time monitoring with {check_interval}s interval")
        
        def monitor_loop():
            while self.is_running:
                try:
                    self.check_alert_conditions()
                    time.sleep(check_interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in monitor loop: {e}")
                    time.sleep(check_interval)
                    
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.is_running:
            logger.warning("Monitor is not running")
            return
            
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        logger.info("Real-time monitoring stopped")
        
    def get_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            "is_running": self.is_running,
            "alert_conditions": len(self.alert_conditions),
            "enabled_conditions": len([c for c in self.alert_conditions if c.enabled]),
            "cached_symbols": len(self.price_cache),
            "last_updates": {k: v.isoformat() for k, v in self.last_update.items()}
        }

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Market Monitor')
    parser.add_argument('--config', default='monitor_config.json', help='Configuration file')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--add-alert', nargs=4, metavar=('SYMBOL', 'TYPE', 'THRESHOLD', 'TIMEFRAME'),
                       help='Add alert condition: symbol type threshold timeframe')
    parser.add_argument('--list-alerts', action='store_true', help='List all alert conditions')
    parser.add_argument('--status', action='store_true', help='Show monitoring status')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = RealTimeMonitor(args.config)
    
    # Handle commands
    if args.add_alert:
        symbol, condition_type, threshold, timeframe = args.add_alert
        monitor.add_alert_condition(symbol, condition_type, float(threshold), timeframe)
        print(f"Added alert: {symbol} {condition_type} {threshold} {timeframe}")
        return
        
    if args.list_alerts:
        print("Alert Conditions:")
        for i, condition in enumerate(monitor.alert_conditions):
            status = "✓" if condition.enabled else "✗"
            print(f"{i+1}. {status} {condition.symbol} {condition.condition_type} {condition.threshold} ({condition.timeframe})")
        return
        
    if args.status:
        status = monitor.get_status()
        print(f"Monitor Status: {'Running' if status['is_running'] else 'Stopped'}")
        print(f"Alert Conditions: {status['enabled_conditions']}/{status['alert_conditions']} enabled")
        print(f"Cached Symbols: {status['cached_symbols']}")
        return
    
    # Start monitoring
    try:
        monitor.start_monitoring(args.interval)
        print(f"Real-time monitoring started. Press Ctrl+C to stop.")
        
        # Keep main thread alive
        while monitor.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop_monitoring()
        print("Monitor stopped.")

if __name__ == "__main__":
    main() 