"""
Portfolio Optimization and Risk Management
-----------------------------------------
This module implements modern portfolio theory and risk management techniques
for optimal asset allocation and portfolio construction.

Features:
- Mean-variance optimization
- Risk parity allocation
- Black-Litterman model
- Monte Carlo simulation
- Risk metrics calculation
- Backtesting framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf

warnings.filterwarnings('ignore')

from src.downloader import CryptoDownloader

@dataclass
class PortfolioConfig:
    """Portfolio optimization configuration"""
    # Optimization parameters
    risk_free_rate: float = 0.02  # Annual risk-free rate
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    max_weight: float = 0.4  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    
    # Risk parameters
    confidence_level: float = 0.95  # For VaR calculation
    lookback_days: int = 252  # Days for historical data
    
    # Rebalancing
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    
    # Transaction costs
    transaction_cost: float = 0.001  # 0.1% per trade

class RiskMetrics:
    """Risk metrics calculation"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
        
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
        
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
        
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility > 0 else 0
        
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        return excess_returns / downside_volatility if downside_volatility > 0 else 0
        
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = returns.mean() * 252
        prices = (1 + returns).cumprod()
        max_dd = RiskMetrics.calculate_max_drawdown(prices)
        return annual_return / abs(max_dd) if max_dd != 0 else 0

class DataManager:
    """Data management for portfolio optimization"""
    
    def __init__(self):
        self.crypto_downloader = CryptoDownloader()
        
    def get_crypto_data(self, symbols: List[str], days: int = 365, timeframe: str = "1d") -> pd.DataFrame:
        """Get cryptocurrency price data"""
        price_data = {}
        
        for symbol in symbols:
            try:
                end_ts = int(datetime.now().timestamp())
                start_ts = end_ts - (days * 24 * 3600)
                
                success, df = self.crypto_downloader.get_data(
                    symbol if symbol.endswith('USDT') else f"{symbol}USDT",
                    start_ts, end_ts, timeframe=timeframe
                )
                
                if success and not df.empty:
                    # Handle different column names
                    if 'Close' in df.columns:
                        price_data[symbol] = df['Close']
                    elif 'close' in df.columns:
                        price_data[symbol] = df['close']
                    else:
                        print(f"✗ {symbol}: No price column found")
                        continue
                    print(f"✓ {symbol}: {len(df)} data points")
                else:
                    print(f"✗ {symbol}: Failed to get data")
                    
            except Exception as e:
                print(f"✗ {symbol}: Error - {e}")
                
        if not price_data:
            raise ValueError("No data retrieved for any symbol")
            
        # Combine into DataFrame
        prices_df = pd.DataFrame(price_data)
        prices_df = prices_df.dropna()
        
        return prices_df
        
    def get_stock_data(self, symbols: List[str], days: int = 365) -> pd.DataFrame:
        """Get stock price data using yfinance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
            
            if isinstance(data, pd.Series):
                data = data.to_frame(symbols[0])
                
            return data.dropna()
            
        except Exception as e:
            print(f"Error getting stock data: {e}")
            return pd.DataFrame()

class PortfolioOptimizer:
    """Main portfolio optimization class"""
    
    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        self.data_manager = DataManager()
        self.prices = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def load_data(self, symbols: List[str], asset_type: str = "crypto", days: int = None):
        """Load price data for optimization"""
        days = days or self.config.lookback_days
        
        print(f"Loading {asset_type} data for {len(symbols)} symbols...")
        
        if asset_type == "crypto":
            self.prices = self.data_manager.get_crypto_data(symbols, days)
        elif asset_type == "stock":
            self.prices = self.data_manager.get_stock_data(symbols, days)
        else:
            raise ValueError("asset_type must be 'crypto' or 'stock'")
            
        # Calculate returns
        self.returns = self.prices.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        print(f"Loaded data: {len(self.prices)} days, {len(self.prices.columns)} assets")
        if hasattr(self.prices.index[0], 'date'):
            print(f"Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
        else:
            print(f"Data points: {len(self.prices)} rows")
        
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio metrics for given weights"""
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
        
    def optimize_max_sharpe(self) -> Dict[str, Any]:
        """Optimize for maximum Sharpe ratio"""
        n_assets = len(self.mean_returns)
        
        def objective(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            return -metrics['sharpe_ratio']  # Minimize negative Sharpe ratio
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((self.config.min_weight, self.config.max_weight) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            metrics = self.calculate_portfolio_metrics(optimal_weights)
            
            return {
                'weights': dict(zip(self.prices.columns, optimal_weights)),
                'metrics': metrics,
                'optimization_result': result
            }
        else:
            raise ValueError(f"Optimization failed: {result.message}")
            
    def optimize_min_volatility(self) -> Dict[str, Any]:
        """Optimize for minimum volatility"""
        n_assets = len(self.mean_returns)
        
        def objective(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            return metrics['volatility']
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = tuple((self.config.min_weight, self.config.max_weight) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            metrics = self.calculate_portfolio_metrics(optimal_weights)
            
            return {
                'weights': dict(zip(self.prices.columns, optimal_weights)),
                'metrics': metrics,
                'optimization_result': result
            }
        else:
            raise ValueError(f"Optimization failed: {result.message}")
            
    def optimize_target_return(self, target_return: float) -> Dict[str, Any]:
        """Optimize for target return with minimum risk"""
        n_assets = len(self.mean_returns)
        
        def objective(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            return metrics['volatility']
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.calculate_portfolio_metrics(x)['return'] - target_return}
        ]
        
        # Bounds
        bounds = tuple((self.config.min_weight, self.config.max_weight) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            metrics = self.calculate_portfolio_metrics(optimal_weights)
            
            return {
                'weights': dict(zip(self.prices.columns, optimal_weights)),
                'metrics': metrics,
                'optimization_result': result
            }
        else:
            raise ValueError(f"Optimization failed: {result.message}")
            
    def risk_parity_allocation(self) -> Dict[str, Any]:
        """Calculate risk parity allocation"""
        n_assets = len(self.mean_returns)
        
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = tuple((self.config.min_weight, self.config.max_weight) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            metrics = self.calculate_portfolio_metrics(optimal_weights)
            
            return {
                'weights': dict(zip(self.prices.columns, optimal_weights)),
                'metrics': metrics,
                'optimization_result': result
            }
        else:
            raise ValueError(f"Risk parity optimization failed: {result.message}")
            
    def equal_weight_allocation(self) -> Dict[str, Any]:
        """Calculate equal weight allocation"""
        n_assets = len(self.mean_returns)
        weights = np.array([1/n_assets] * n_assets)
        metrics = self.calculate_portfolio_metrics(weights)
        
        return {
            'weights': dict(zip(self.prices.columns, weights)),
            'metrics': metrics
        }
        
    def generate_efficient_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """Generate efficient frontier"""
        # Calculate return range
        min_ret = self.mean_returns.min() * 252
        max_ret = self.mean_returns.max() * 252
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                result = self.optimize_target_return(target_ret)
                efficient_portfolios.append({
                    'target_return': target_ret,
                    'return': result['metrics']['return'],
                    'volatility': result['metrics']['volatility'],
                    'sharpe_ratio': result['metrics']['sharpe_ratio'],
                    'weights': result['weights']
                })
            except:
                continue
                
        return pd.DataFrame(efficient_portfolios)
        
    def monte_carlo_simulation(self, weights: Dict[str, float], n_simulations: int = 10000, 
                             time_horizon: int = 252) -> Dict[str, Any]:
        """Monte Carlo simulation for portfolio"""
        # Convert weights to array
        weight_array = np.array([weights[col] for col in self.prices.columns])
        
        # Portfolio statistics
        portfolio_mean = np.sum(self.mean_returns * weight_array)
        portfolio_std = np.sqrt(np.dot(weight_array.T, np.dot(self.cov_matrix, weight_array)))
        
        # Simulate returns
        simulated_returns = np.random.normal(
            portfolio_mean, portfolio_std, (n_simulations, time_horizon)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
        final_values = cumulative_returns[:, -1]
        
        # Calculate statistics
        results = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'var_95': np.percentile(final_values, 5),
            'var_99': np.percentile(final_values, 1),
            'probability_loss': np.mean(final_values < 1),
            'max_gain': np.max(final_values) - 1,
            'max_loss': 1 - np.min(final_values),
            'simulated_paths': cumulative_returns
        }
        
        return results
        
    def backtest_portfolio(self, weights: Dict[str, float], rebalance_freq: str = "monthly") -> Dict[str, Any]:
        """Backtest portfolio performance"""
        # Convert weights to array
        weight_array = np.array([weights[col] for col in self.prices.columns])
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * weight_array).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = RiskMetrics.calculate_sharpe_ratio(portfolio_returns, self.config.risk_free_rate)
        sortino_ratio = RiskMetrics.calculate_sortino_ratio(portfolio_returns, self.config.risk_free_rate)
        max_drawdown = RiskMetrics.calculate_max_drawdown(cumulative_returns)
        var_95 = RiskMetrics.calculate_var(portfolio_returns, 0.95)
        cvar_95 = RiskMetrics.calculate_cvar(portfolio_returns, 0.95)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns
        }
        
    def plot_efficient_frontier(self, efficient_frontier: pd.DataFrame, 
                              optimal_portfolios: Dict[str, Dict] = None):
        """Plot efficient frontier"""
        plt.figure(figsize=(12, 8))
        
        # Plot efficient frontier
        plt.scatter(efficient_frontier['volatility'], efficient_frontier['return'], 
                   c=efficient_frontier['sharpe_ratio'], cmap='viridis', alpha=0.6)
        plt.colorbar(label='Sharpe Ratio')
        
        # Plot optimal portfolios
        if optimal_portfolios:
            for name, portfolio in optimal_portfolios.items():
                metrics = portfolio['metrics']
                plt.scatter(metrics['volatility'], metrics['return'], 
                          marker='*', s=200, label=name)
                
        plt.xlabel('Volatility (Annual)')
        plt.ylabel('Expected Return (Annual)')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_portfolio_composition(self, weights: Dict[str, float]):
        """Plot portfolio composition"""
        # Filter out zero weights
        non_zero_weights = {k: v for k, v in weights.items() if v > 0.001}
        
        plt.figure(figsize=(10, 8))
        plt.pie(non_zero_weights.values(), labels=non_zero_weights.keys(), autopct='%1.1f%%')
        plt.title('Portfolio Composition')
        plt.axis('equal')
        plt.show()
        
    def plot_monte_carlo_results(self, mc_results: Dict[str, Any]):
        """Plot Monte Carlo simulation results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot simulation paths
        paths = mc_results['simulated_paths']
        for i in range(min(100, len(paths))):  # Plot first 100 paths
            ax1.plot(paths[i], alpha=0.1, color='blue')
        ax1.set_title('Monte Carlo Simulation Paths')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)
        
        # Plot final value distribution
        final_values = paths[:, -1]
        ax2.hist(final_values, bins=50, alpha=0.7, density=True)
        ax2.axvline(mc_results['mean_final_value'], color='red', linestyle='--', label='Mean')
        ax2.axvline(mc_results['var_95'], color='orange', linestyle='--', label='VaR 95%')
        ax2.set_title('Final Portfolio Value Distribution')
        ax2.set_xlabel('Final Portfolio Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Portfolio Optimization Tool')
    parser.add_argument('--symbols', nargs='+', required=True, help='Asset symbols')
    parser.add_argument('--asset-type', choices=['crypto', 'stock'], default='crypto', help='Asset type')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data')
    parser.add_argument('--method', choices=['max_sharpe', 'min_vol', 'risk_parity', 'equal_weight'], 
                       default='max_sharpe', help='Optimization method')
    parser.add_argument('--target-return', type=float, help='Target return for optimization')
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo simulation')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    config = PortfolioConfig()
    optimizer = PortfolioOptimizer(config)
    
    # Load data
    print(f"Loading data for {len(args.symbols)} symbols...")
    optimizer.load_data(args.symbols, args.asset_type, args.days)
    
    # Optimize portfolio
    print(f"Optimizing portfolio using {args.method} method...")
    
    if args.method == 'max_sharpe':
        result = optimizer.optimize_max_sharpe()
    elif args.method == 'min_vol':
        result = optimizer.optimize_min_volatility()
    elif args.method == 'risk_parity':
        result = optimizer.risk_parity_allocation()
    elif args.method == 'equal_weight':
        result = optimizer.equal_weight_allocation()
    elif args.target_return:
        result = optimizer.optimize_target_return(args.target_return)
    else:
        result = optimizer.optimize_max_sharpe()
        
    # Display results
    print("\n" + "="*50)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("="*50)
    
    print(f"Method: {args.method}")
    print(f"Expected Annual Return: {result['metrics']['return']:.2%}")
    print(f"Annual Volatility: {result['metrics']['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['metrics']['sharpe_ratio']:.3f}")
    
    print("\nOptimal Weights:")
    for symbol, weight in result['weights'].items():
        if weight > 0.001:  # Only show significant weights
            print(f"  {symbol}: {weight:.1%}")
            
    # Monte Carlo simulation
    if args.monte_carlo:
        print("\nRunning Monte Carlo simulation...")
        mc_results = optimizer.monte_carlo_simulation(result['weights'])
        
        print(f"Expected Value (1 year): {mc_results['mean_final_value']:.3f}")
        print(f"VaR 95% (1 year): {mc_results['var_95']:.3f}")
        print(f"Probability of Loss: {mc_results['probability_loss']:.1%}")
        
    # Backtest
    if args.backtest:
        print("\nRunning backtest...")
        backtest_results = optimizer.backtest_portfolio(result['weights'])
        
        print(f"Total Return: {backtest_results['total_return']:.2%}")
        print(f"Annual Return: {backtest_results['annual_return']:.2%}")
        print(f"Annual Volatility: {backtest_results['annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        
    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        
        # Portfolio composition
        optimizer.plot_portfolio_composition(result['weights'])
        
        # Efficient frontier
        if args.method != 'equal_weight':
            efficient_frontier = optimizer.generate_efficient_frontier()
            optimizer.plot_efficient_frontier(efficient_frontier, {'Optimal': result})
            
        # Monte Carlo results
        if args.monte_carlo:
            optimizer.plot_monte_carlo_results(mc_results)

if __name__ == "__main__":
    main() 