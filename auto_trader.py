"""
Automated Trading Execution System
---------------------------------
This module implements an automated trading system that can execute trades
based on analysis results from the screener tools.

Features:
- Multiple exchange support (Binance, paper trading)
- Risk management and position sizing
- Order management (market, limit, stop-loss)
- Trade logging and performance tracking
- Safety mechanisms and circuit breakers
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from functools import lru_cache
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class TradingSignal:
    """Trading signal from analysis"""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    price: float
    timestamp: datetime
    source: str  # crypto_screener, ml_predictor, etc.
    metadata: Dict[str, Any] = None

@dataclass
class Position:
    """Trading position"""
    symbol: str
    side: str  # long, short
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Order:
    """Trading order"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    size: float
    price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    filled_size: float = 0.0
    filled_price: Optional[float] = None
    fill_time: Optional[datetime] = None

@dataclass
class TradingConfig:
    """Trading configuration"""
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio per position
    max_total_exposure: float = 0.8  # 80% maximum total exposure
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    
    # Position sizing
    risk_per_trade: float = 0.02  # 2% risk per trade
    position_sizing_method: str = "fixed_risk"  # fixed_risk, fixed_amount, kelly
    
    # Trading rules
    min_confidence: float = 0.7  # Minimum signal confidence
    max_positions: int = 10  # Maximum concurrent positions
    cooldown_period: int = 3600  # Seconds between trades for same symbol
    
    # Safety mechanisms
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    max_drawdown_limit: float = 0.15  # 15% maximum drawdown
    circuit_breaker_enabled: bool = True
    
    # Exchange settings
    exchange: str = "paper"  # paper, binance
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True

class PaperTradingExchange:
    """Paper trading exchange simulator"""
    
    def __init__(self, initial_balance: float = 10000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []
        self.order_counter = 0
        
    def get_balance(self) -> float:
        """Get current balance"""
        return self.balance
        
    @lru_cache(maxsize=64)
    def get_portfolio_value(self, prices_tuple: tuple) -> float:
        """Calculate total portfolio value with caching"""
        prices = dict(prices_tuple)
        total_value = self.balance
        
        # Vectorized calculation for better performance
        symbols = list(self.positions.keys())
        if symbols and all(symbol in prices for symbol in symbols):
            sizes = np.array([self.positions[symbol].size for symbol in symbols])
            symbol_prices = np.array([prices[symbol] for symbol in symbols])
            total_value += np.sum(sizes * symbol_prices)
        else:
            # Fallback to loop if not all prices available
            for symbol, position in self.positions.items():
                if symbol in prices:
                    total_value += position.size * prices[symbol]
                
        return total_value
        
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                   size: float, price: Optional[float] = None) -> str:
        """Place an order"""
        self.order_counter += 1
        order_id = f"paper_{self.order_counter}"
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            size=size,
            price=price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        
        self.orders[order_id] = order
        logger.info(f"Placed order: {order_id} {side.value} {size} {symbol} @ {price}")
        
        return order_id
        
    def fill_order(self, order_id: str, fill_price: float) -> bool:
        """Fill an order (simulate execution)"""
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False
            
        # Calculate cost/proceeds
        if order.side == OrderSide.BUY:
            cost = order.size * fill_price
            if cost > self.balance:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order {order_id} rejected: insufficient balance")
                return False
                
            self.balance -= cost
            
            # Update or create position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_size = pos.size + order.size
                avg_price = (pos.size * pos.entry_price + order.size * fill_price) / total_size
                pos.size = total_size
                pos.entry_price = avg_price
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    side="long",
                    size=order.size,
                    entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_time=datetime.now()
                )
                
        else:  # SELL
            if order.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order {order_id} rejected: no position to sell")
                return False
                
            pos = self.positions[order.symbol]
            if order.size > pos.size:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order {order_id} rejected: insufficient position size")
                return False
                
            proceeds = order.size * fill_price
            self.balance += proceeds
            
            # Calculate realized PnL
            realized_pnl = (fill_price - pos.entry_price) * order.size
            pos.realized_pnl += realized_pnl
            
            # Update position
            pos.size -= order.size
            if pos.size <= 0:
                del self.positions[order.symbol]
                
        # Mark order as filled
        order.status = OrderStatus.FILLED
        order.filled_size = order.size
        order.filled_price = fill_price
        order.fill_time = datetime.now()
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': order.symbol,
            'side': order.side.value,
            'size': order.size,
            'price': fill_price,
            'order_id': order_id
        })
        
        logger.info(f"Filled order: {order_id} at {fill_price}")
        return True
        
    def update_positions(self, prices: Dict[str, float]):
        """Update position values with current prices"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                position.unrealized_pnl = (prices[symbol] - position.entry_price) * position.size
                
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self.positions.copy()
        
    def get_orders(self) -> Dict[str, Order]:
        """Get all orders"""
        return self.orders.copy()

class RiskManager:
    """Risk management system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.last_reset = datetime.now().date()
        
    def reset_daily_metrics(self, current_balance: float):
        """Reset daily metrics"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.daily_start_balance = current_balance
            self.last_reset = today
            
    def update_metrics(self, current_balance: float):
        """Update risk metrics"""
        self.reset_daily_metrics(current_balance)
        
        # Update daily PnL
        self.daily_pnl = current_balance - self.daily_start_balance
        
        # Update drawdown
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            
        current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
    def check_risk_limits(self, current_balance: float) -> Tuple[bool, str]:
        """Check if risk limits are breached"""
        self.update_metrics(current_balance)
        
        # Daily loss limit
        daily_loss_pct = self.daily_pnl / self.daily_start_balance if self.daily_start_balance > 0 else 0
        if daily_loss_pct < -self.config.daily_loss_limit:
            return False, f"Daily loss limit breached: {daily_loss_pct:.2%}"
            
        # Maximum drawdown limit
        if self.max_drawdown > self.config.max_drawdown_limit:
            return False, f"Maximum drawdown limit breached: {self.max_drawdown:.2%}"
            
        return True, "OK"
        
    def calculate_position_size(self, signal: TradingSignal, current_balance: float, 
                              current_price: float) -> float:
        """Calculate position size based on risk management"""
        if self.config.position_sizing_method == "fixed_risk":
            # Risk-based position sizing
            risk_amount = current_balance * self.config.risk_per_trade
            stop_loss_distance = current_price * self.config.stop_loss_pct
            position_size = risk_amount / stop_loss_distance
            
        elif self.config.position_sizing_method == "fixed_amount":
            # Fixed percentage of balance
            position_value = current_balance * self.config.max_position_size
            position_size = position_value / current_price
            
        else:
            # Default to fixed amount
            position_value = current_balance * self.config.max_position_size
            position_size = position_value / current_price
            
        # Apply maximum position size limit
        max_position_value = current_balance * self.config.max_position_size
        max_position_size = max_position_value / current_price
        
        return min(position_size, max_position_size)

class AutoTrader:
    """Main automated trading system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.exchange = self._initialize_exchange()
        self.signal_history: List[TradingSignal] = []
        self.last_trade_time: Dict[str, datetime] = {}
        self.is_running = False
        self.circuit_breaker_active = False
        
    def _initialize_exchange(self):
        """Initialize exchange connection"""
        if self.config.exchange == "paper":
            return PaperTradingExchange()
        elif self.config.exchange == "binance":
            # TODO: Implement Binance exchange
            raise NotImplementedError("Binance exchange not implemented yet")
        else:
            raise ValueError(f"Unsupported exchange: {self.config.exchange}")
            
    def add_signal(self, signal: TradingSignal):
        """Add a trading signal"""
        self.signal_history.append(signal)
        logger.info(f"Received signal: {signal.symbol} {signal.signal} (confidence: {signal.confidence:.2f})")
        
        if self.is_running and not self.circuit_breaker_active:
            self.process_signal(signal)
            
    def process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        # Check signal confidence
        if signal.confidence < self.config.min_confidence:
            logger.info(f"Signal confidence too low: {signal.confidence:.2f} < {self.config.min_confidence:.2f}")
            return
            
        # Check cooldown period
        if signal.symbol in self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time[signal.symbol]).total_seconds()
            if time_since_last < self.config.cooldown_period:
                logger.info(f"Cooldown period active for {signal.symbol}")
                return
                
        # Check risk limits
        current_balance = self.exchange.get_balance()
        risk_ok, risk_message = self.risk_manager.check_risk_limits(current_balance)
        
        if not risk_ok:
            logger.warning(f"Risk limit breached: {risk_message}")
            if self.config.circuit_breaker_enabled:
                self.activate_circuit_breaker()
            return
            
        # Check maximum positions
        current_positions = self.exchange.get_positions()
        if len(current_positions) >= self.config.max_positions:
            logger.info(f"Maximum positions reached: {len(current_positions)}")
            return
            
        # Execute trade based on signal
        if signal.signal == "BUY":
            self.execute_buy_signal(signal)
        elif signal.signal == "SELL":
            self.execute_sell_signal(signal)
            
    def execute_buy_signal(self, signal: TradingSignal):
        """Execute buy signal"""
        current_balance = self.exchange.get_balance()
        position_size = self.risk_manager.calculate_position_size(
            signal, current_balance, signal.price
        )
        
        if position_size * signal.price > current_balance:
            logger.warning(f"Insufficient balance for {signal.symbol}")
            return
            
        # Place buy order
        order_id = self.exchange.place_order(
            symbol=signal.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=position_size
        )
        
        # Simulate immediate fill for paper trading
        if isinstance(self.exchange, PaperTradingExchange):
            self.exchange.fill_order(order_id, signal.price)
            
        # Set stop loss and take profit
        self.set_risk_management_orders(signal.symbol, signal.price, OrderSide.BUY)
        
        # Update last trade time
        self.last_trade_time[signal.symbol] = datetime.now()
        
    def execute_sell_signal(self, signal: TradingSignal):
        """Execute sell signal"""
        positions = self.exchange.get_positions()
        
        if signal.symbol not in positions:
            logger.info(f"No position to sell for {signal.symbol}")
            return
            
        position = positions[signal.symbol]
        
        # Place sell order
        order_id = self.exchange.place_order(
            symbol=signal.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=position.size
        )
        
        # Simulate immediate fill for paper trading
        if isinstance(self.exchange, PaperTradingExchange):
            self.exchange.fill_order(order_id, signal.price)
            
        # Update last trade time
        self.last_trade_time[signal.symbol] = datetime.now()
        
    def set_risk_management_orders(self, symbol: str, entry_price: float, side: OrderSide):
        """Set stop loss and take profit orders"""
        if side == OrderSide.BUY:
            stop_loss_price = entry_price * (1 - self.config.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.config.take_profit_pct)
            
            # Note: In a real implementation, these would be actual orders
            logger.info(f"Risk management for {symbol}: SL={stop_loss_price:.4f}, TP={take_profit_price:.4f}")
            
    def activate_circuit_breaker(self):
        """Activate circuit breaker"""
        self.circuit_breaker_active = True
        logger.warning("CIRCUIT BREAKER ACTIVATED - Trading halted")
        
        # Close all positions
        positions = self.exchange.get_positions()
        for symbol, position in positions.items():
            self.exchange.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                size=position.size
            )
            
    def deactivate_circuit_breaker(self):
        """Deactivate circuit breaker"""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker deactivated")
        
    def start_trading(self):
        """Start automated trading"""
        self.is_running = True
        logger.info("Automated trading started")
        
    def stop_trading(self):
        """Stop automated trading"""
        self.is_running = False
        logger.info("Automated trading stopped")
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        current_balance = self.exchange.get_balance()
        initial_balance = self.exchange.initial_balance
        
        # Calculate returns
        total_return = (current_balance - initial_balance) / initial_balance
        
        # Get positions
        positions = self.exchange.get_positions()
        
        # Calculate unrealized PnL
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        
        # Trade statistics
        trades = self.exchange.trade_history if hasattr(self.exchange, 'trade_history') else []
        
        return {
            'initial_balance': initial_balance,
            'current_balance': current_balance,
            'total_return': total_return,
            'total_unrealized_pnl': total_unrealized_pnl,
            'active_positions': len(positions),
            'total_trades': len(trades),
            'daily_pnl': self.risk_manager.daily_pnl,
            'max_drawdown': self.risk_manager.max_drawdown,
            'circuit_breaker_active': self.circuit_breaker_active,
            'is_running': self.is_running
        }
        
    def save_state(self, filepath: str):
        """Save trading state"""
        state = {
            'config': asdict(self.config),
            'signal_history': [asdict(signal) for signal in self.signal_history],
            'last_trade_time': {k: v.isoformat() for k, v in self.last_trade_time.items()},
            'performance': self.get_performance_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
    def load_state(self, filepath: str):
        """Load trading state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        # Restore last trade times
        self.last_trade_time = {
            k: datetime.fromisoformat(v) for k, v in state.get('last_trade_time', {}).items()
        }

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Trading System')
    parser.add_argument('--config', default='trading_config.json', help='Configuration file')
    parser.add_argument('--start', action='store_true', help='Start automated trading')
    parser.add_argument('--stop', action='store_true', help='Stop automated trading')
    parser.add_argument('--status', action='store_true', help='Show trading status')
    parser.add_argument('--test-signal', nargs=3, metavar=('SYMBOL', 'SIGNAL', 'CONFIDENCE'),
                       help='Test signal: symbol signal confidence')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = TradingConfig(**config_data)
    except FileNotFoundError:
        print(f"Config file {args.config} not found, using defaults")
        config = TradingConfig()
        
    # Initialize trader
    trader = AutoTrader(config)
    
    # Handle commands
    if args.start:
        trader.start_trading()
        print("Automated trading started")
        
        # Keep running
        try:
            while trader.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            trader.stop_trading()
            print("Trading stopped")
            
    elif args.stop:
        trader.stop_trading()
        print("Trading stopped")
        
    elif args.status:
        report = trader.get_performance_report()
        print("Trading Status:")
        print(f"  Running: {report['is_running']}")
        print(f"  Balance: ${report['current_balance']:.2f}")
        print(f"  Total Return: {report['total_return']:.2%}")
        print(f"  Active Positions: {report['active_positions']}")
        print(f"  Total Trades: {report['total_trades']}")
        print(f"  Circuit Breaker: {report['circuit_breaker_active']}")
        
    elif args.test_signal:
        symbol, signal_type, confidence = args.test_signal
        
        test_signal = TradingSignal(
            symbol=symbol,
            signal=signal_type.upper(),
            confidence=float(confidence),
            price=100.0,  # Mock price
            timestamp=datetime.now(),
            source="test"
        )
        
        trader.start_trading()
        trader.add_signal(test_signal)
        
        report = trader.get_performance_report()
        print(f"Test signal processed. New balance: ${report['current_balance']:.2f}")

if __name__ == "__main__":
    main() 