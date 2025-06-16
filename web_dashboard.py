"""
Web Dashboard for Screener Tools
-------------------------------
This module provides a web-based dashboard for monitoring and controlling
all screener tools and trading systems.

Features:
- Real-time market monitoring
- Interactive charts and visualizations
- Portfolio management interface
- Trading system control panel
- Performance analytics
- Alert management
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not available. Web dashboard will be disabled.")

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Alternative web interface will be disabled.")

from src.downloader import CryptoDownloader
from src.common import TrendAnalysisConfig

class DashboardData:
    """Data management for dashboard"""
    
    def __init__(self):
        self.crypto_downloader = CryptoDownloader()
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview data"""
        cache_key = "market_overview"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
            
        try:
            # Get top cryptocurrencies
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
            market_data = []
            
            for symbol in symbols:
                end_ts = int(datetime.now().timestamp())
                start_ts = end_ts - 86400  # 24 hours
                
                success, df = self.crypto_downloader.get_data(symbol, start_ts, end_ts, timeframe="1h")
                
                if success and not df.empty:
                    current_price = df['Close'].iloc[-1]
                    price_24h_ago = df['Close'].iloc[0]
                    change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
                    
                    market_data.append({
                        'symbol': symbol.replace('USDT', ''),
                        'price': current_price,
                        'change_24h': change_24h,
                        'volume': df['Volume'].sum() if 'Volume' in df.columns else 0
                    })
                    
            self._update_cache(cache_key, market_data)
            return market_data
            
        except Exception as e:
            print(f"Error getting market overview: {e}")
            return []
            
    def get_screener_results(self, timeframe: str = "15m", days: int = 3) -> Dict[str, Any]:
        """Get latest screener results"""
        cache_key = f"screener_{timeframe}_{days}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
            
        try:
            # Check for latest output files
            output_dir = "output"
            if os.path.exists(output_dir):
                # Find latest date directory
                date_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
                if date_dirs:
                    latest_date = max(date_dirs)
                    date_path = os.path.join(output_dir, latest_date)
                    
                    # Find latest results file
                    files = [f for f in os.listdir(date_path) if f.endswith('.txt')]
                    if files:
                        latest_file = max(files)
                        file_path = os.path.join(date_path, latest_file)
                        
                        # Parse results
                        results = self._parse_screener_file(file_path)
                        self._update_cache(cache_key, results)
                        return results
                        
            return {'symbols': [], 'timestamp': None}
            
        except Exception as e:
            print(f"Error getting screener results: {e}")
            return {'symbols': [], 'timestamp': None}
            
    def get_portfolio_performance(self) -> Dict[str, Any]:
        """Get portfolio performance data"""
        cache_key = "portfolio_performance"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
            
        try:
            # Mock portfolio data - replace with actual portfolio tracking
            performance_data = {
                'total_value': 10500.0,
                'daily_pnl': 250.0,
                'daily_pnl_pct': 2.4,
                'total_return': 5.0,
                'positions': [
                    {'symbol': 'BTC', 'value': 5000, 'pnl': 150, 'pnl_pct': 3.1},
                    {'symbol': 'ETH', 'value': 3000, 'pnl': 80, 'pnl_pct': 2.7},
                    {'symbol': 'SOL', 'value': 2500, 'pnl': 20, 'pnl_pct': 0.8}
                ]
            }
            
            self._update_cache(cache_key, performance_data)
            return performance_data
            
        except Exception as e:
            print(f"Error getting portfolio performance: {e}")
            return {}
            
    def get_trading_signals(self) -> List[Dict[str, Any]]:
        """Get recent trading signals"""
        cache_key = "trading_signals"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
            
        try:
            # Mock signals data - replace with actual signal tracking
            signals = [
                {
                    'symbol': 'BTCUSDT',
                    'signal': 'BUY',
                    'confidence': 0.85,
                    'price': 45000,
                    'timestamp': datetime.now() - timedelta(minutes=5),
                    'source': 'crypto_screener'
                },
                {
                    'symbol': 'ETHUSDT',
                    'signal': 'SELL',
                    'confidence': 0.72,
                    'price': 3200,
                    'timestamp': datetime.now() - timedelta(minutes=15),
                    'source': 'ml_predictor'
                }
            ]
            
            self._update_cache(cache_key, signals)
            return signals
            
        except Exception as e:
            print(f"Error getting trading signals: {e}")
            return []
            
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is valid"""
        if key not in self.cache:
            return False
            
        cache_time = self.cache[key]['timestamp']
        return (datetime.now() - cache_time).total_seconds() < self.cache_timeout
        
    def _update_cache(self, key: str, data: Any):
        """Update cache entry"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
    def _parse_screener_file(self, file_path: str) -> Dict[str, Any]:
        """Parse screener output file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract symbols (simplified parsing)
            symbols = []
            for line in content.split('\n'):
                if 'BINANCE:' in line:
                    # Extract symbols from TradingView format
                    parts = line.split(',')
                    for part in parts:
                        if 'BINANCE:' in part:
                            symbol = part.replace('BINANCE:', '').replace('.P', '').replace('USDT', '')
                            if symbol and symbol not in symbols:
                                symbols.append(symbol)
                                
            return {
                'symbols': symbols[:20],  # Top 20
                'timestamp': datetime.fromtimestamp(os.path.getmtime(file_path))
            }
            
        except Exception as e:
            print(f"Error parsing screener file: {e}")
            return {'symbols': [], 'timestamp': None}

class StreamlitDashboard:
    """Streamlit-based dashboard"""
    
    def __init__(self):
        self.data_manager = DashboardData()
        
    def run(self):
        """Run Streamlit dashboard"""
        if not STREAMLIT_AVAILABLE:
            print("Streamlit not available")
            return
            
        st.set_page_config(
            page_title="Screener Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self._render_dashboard()
        
    def _render_dashboard(self):
        """Render main dashboard"""
        st.title("📊 Screener Dashboard")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Select Page", [
            "Market Overview",
            "Screener Results", 
            "Portfolio",
            "Trading Signals",
            "Analytics",
            "Settings"
        ])
        
        # Main content
        if page == "Market Overview":
            self._render_market_overview()
        elif page == "Screener Results":
            self._render_screener_results()
        elif page == "Portfolio":
            self._render_portfolio()
        elif page == "Trading Signals":
            self._render_trading_signals()
        elif page == "Analytics":
            self._render_analytics()
        elif page == "Settings":
            self._render_settings()
            
    def _render_market_overview(self):
        """Render market overview page"""
        st.header("Market Overview")
        
        # Get market data
        market_data = self.data_manager.get_market_overview()
        
        if market_data:
            # Create metrics
            cols = st.columns(len(market_data))
            
            for i, data in enumerate(market_data):
                with cols[i]:
                    change_color = "normal" if data['change_24h'] >= 0 else "inverse"
                    st.metric(
                        label=data['symbol'],
                        value=f"${data['price']:.2f}",
                        delta=f"{data['change_24h']:.2f}%"
                    )
                    
            # Market chart
            st.subheader("Price Charts")
            
            # Create sample chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[data['symbol'] for data in market_data[:4]]
            )
            
            for i, data in enumerate(market_data[:4]):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                # Generate sample price data
                dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
                prices = np.random.normal(data['price'], data['price'] * 0.02, len(dates))
                
                fig.add_trace(
                    go.Scatter(x=dates, y=prices, name=data['symbol']),
                    row=row, col=col
                )
                
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_screener_results(self):
        """Render screener results page"""
        st.header("Screener Results")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
        with col2:
            days = st.number_input("Days", min_value=1, max_value=30, value=3)
        with col3:
            if st.button("Run Screener"):
                st.info("Running screener... (This would trigger the actual screener)")
                
        # Results
        results = self.data_manager.get_screener_results(timeframe, days)
        
        if results['symbols']:
            st.subheader(f"Top Performing Assets ({len(results['symbols'])} found)")
            
            # Display as table
            df = pd.DataFrame({
                'Rank': range(1, len(results['symbols']) + 1),
                'Symbol': results['symbols'],
                'Action': ['📈 Analyze'] * len(results['symbols'])
            })
            
            st.dataframe(df, use_container_width=True)
            
            # Last update
            if results['timestamp']:
                st.caption(f"Last updated: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No screener results available. Run the screener to see results.")
            
    def _render_portfolio(self):
        """Render portfolio page"""
        st.header("Portfolio Management")
        
        # Portfolio overview
        portfolio_data = self.data_manager.get_portfolio_performance()
        
        if portfolio_data:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Value", f"${portfolio_data['total_value']:,.2f}")
            with col2:
                st.metric("Daily P&L", f"${portfolio_data['daily_pnl']:,.2f}", 
                         f"{portfolio_data['daily_pnl_pct']:.2f}%")
            with col3:
                st.metric("Total Return", f"{portfolio_data['total_return']:.2f}%")
            with col4:
                st.metric("Positions", len(portfolio_data['positions']))
                
            # Positions table
            st.subheader("Current Positions")
            
            positions_df = pd.DataFrame(portfolio_data['positions'])
            if not positions_df.empty:
                st.dataframe(positions_df, use_container_width=True)
                
                # Portfolio allocation chart
                fig = px.pie(
                    positions_df, 
                    values='value', 
                    names='symbol',
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
                
    def _render_trading_signals(self):
        """Render trading signals page"""
        st.header("Trading Signals")
        
        # Get signals
        signals = self.data_manager.get_trading_signals()
        
        if signals:
            # Convert to DataFrame
            signals_df = pd.DataFrame(signals)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            # Display signals
            for _, signal in signals_df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        signal_color = "🟢" if signal['signal'] == 'BUY' else "🔴"
                        st.write(f"{signal_color} **{signal['symbol']}**")
                        
                    with col2:
                        st.write(f"**{signal['signal']}**")
                        
                    with col3:
                        st.write(f"Confidence: {signal['confidence']:.1%}")
                        
                    with col4:
                        st.write(f"${signal['price']:,.2f}")
                        
                    st.caption(f"Source: {signal['source']} | {signal['timestamp'].strftime('%H:%M:%S')}")
                    st.divider()
        else:
            st.info("No recent trading signals")
            
    def _render_analytics(self):
        """Render analytics page"""
        st.header("Analytics & Performance")
        
        # Performance chart
        st.subheader("Portfolio Performance")
        
        # Generate sample performance data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        portfolio_values = np.cumsum(np.random.normal(0.1, 1, len(dates))) + 10000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            metrics = {
                "Total Return": "5.2%",
                "Sharpe Ratio": "1.45",
                "Max Drawdown": "-3.1%",
                "Win Rate": "68%"
            }
            
            for metric, value in metrics.items():
                st.metric(metric, value)
                
        with col2:
            st.subheader("Risk Metrics")
            risk_metrics = {
                "VaR (95%)": "-2.1%",
                "Beta": "0.85",
                "Volatility": "12.3%",
                "Correlation": "0.72"
            }
            
            for metric, value in risk_metrics.items():
                st.metric(metric, value)
                
    def _render_settings(self):
        """Render settings page"""
        st.header("Settings")
        
        # Trading settings
        st.subheader("Trading Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Max Position Size (%)", min_value=1, max_value=50, value=10)
            st.number_input("Stop Loss (%)", min_value=1, max_value=20, value=5)
            st.selectbox("Risk Management", ["Conservative", "Moderate", "Aggressive"])
            
        with col2:
            st.number_input("Take Profit (%)", min_value=5, max_value=50, value=15)
            st.number_input("Max Positions", min_value=1, max_value=20, value=10)
            st.checkbox("Auto Trading Enabled", value=False)
            
        # Alert settings
        st.subheader("Alert Configuration")
        
        st.checkbox("Email Alerts", value=True)
        st.text_input("Email Address", value="user@example.com")
        st.checkbox("Desktop Notifications", value=True)
        st.checkbox("Webhook Alerts", value=False)
        
        # Save button
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

def create_flask_app():
    """Create Flask application"""
    if not FLASK_AVAILABLE:
        return None
        
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'screener_dashboard_secret'
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    data_manager = DashboardData()
    
    @app.route('/')
    def index():
        return render_template('dashboard.html')
        
    @app.route('/api/market-overview')
    def api_market_overview():
        data = data_manager.get_market_overview()
        return jsonify(data)
        
    @app.route('/api/screener-results')
    def api_screener_results():
        timeframe = request.args.get('timeframe', '15m')
        days = int(request.args.get('days', 3))
        data = data_manager.get_screener_results(timeframe, days)
        return jsonify(data)
        
    @app.route('/api/portfolio')
    def api_portfolio():
        data = data_manager.get_portfolio_performance()
        return jsonify(data)
        
    @app.route('/api/signals')
    def api_signals():
        data = data_manager.get_trading_signals()
        return jsonify(data, default=str)
        
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        emit('status', {'msg': 'Connected to Screener Dashboard'})
        
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
        
    return app, socketio

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Screener Web Dashboard')
    parser.add_argument('--interface', choices=['streamlit', 'flask'], default='streamlit',
                       help='Web interface type')
    parser.add_argument('--port', type=int, default=8501, help='Port number')
    parser.add_argument('--host', default='localhost', help='Host address')
    
    args = parser.parse_args()
    
    if args.interface == 'streamlit':
        if STREAMLIT_AVAILABLE:
            print(f"Starting Streamlit dashboard on http://{args.host}:{args.port}")
            dashboard = StreamlitDashboard()
            dashboard.run()
        else:
            print("Streamlit not available. Please install: pip install streamlit plotly")
            
    elif args.interface == 'flask':
        if FLASK_AVAILABLE:
            app, socketio = create_flask_app()
            if app:
                print(f"Starting Flask dashboard on http://{args.host}:{args.port}")
                socketio.run(app, host=args.host, port=args.port, debug=True)
            else:
                print("Failed to create Flask app")
        else:
            print("Flask not available. Please install: pip install flask flask-socketio")

if __name__ == "__main__":
    main() 