"""
Machine Learning Predictor for Market Analysis
---------------------------------------------
This module implements various ML models to predict price movements and identify trading opportunities.

Features:
- LSTM neural networks for time series prediction
- Random Forest for pattern classification
- Feature engineering from technical indicators
- Model training and backtesting
- Real-time prediction capabilities
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be disabled.")

from src.downloader import CryptoDownloader
from src.common import TrendAnalysisConfig

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    # Feature engineering
    lookback_periods: List[int] = None
    technical_indicators: List[str] = None
    
    # Model parameters
    lstm_units: int = 50
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    
    # Prediction parameters
    prediction_horizon: int = 24  # Hours ahead to predict
    classification_threshold: float = 0.02  # 2% price change threshold
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50]
        if self.technical_indicators is None:
            self.technical_indicators = ['RSI', 'MACD', 'BB', 'ATR', 'Volume_MA']

class FeatureEngineer:
    """Feature engineering for ML models"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # RSI  
        if 'RSI' in self.config.technical_indicators:
            prices_tuple = tuple(df['Close'].values)
            rsi_values = self._calculate_rsi(prices_tuple)
            df['RSI'] = rsi_values
            
        # MACD
        if 'MACD' in self.config.technical_indicators:
            df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
            
        # Bollinger Bands
        if 'BB' in self.config.technical_indicators:
            df['BB_Upper'], df['BB_Lower'], df['BB_Middle'] = self._calculate_bollinger_bands(df['Close'])
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
        # ATR
        if 'ATR' in self.config.technical_indicators:
            df['ATR'] = self._calculate_atr(df)
            
        # Volume indicators
        if 'Volume_MA' in self.config.technical_indicators and 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
        return df
        
    @lru_cache(maxsize=128)
    def _calculate_rsi(self, prices_tuple: tuple, period: int = 14) -> np.ndarray:
        """Calculate RSI using optimized numpy operations"""
        prices = np.array(prices_tuple)
        delta = np.diff(prices, prepend=prices[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Use exponential moving average for efficiency
        alpha = 2.0 / (period + 1.0)
        avg_gain = np.zeros_like(gains)
        avg_loss = np.zeros_like(losses)
        
        avg_gain[period-1] = np.mean(gains[:period])
        avg_loss[period-1] = np.mean(losses[:period])
        
        for i in range(period, len(gains)):
            avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i-1]
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
        
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, lower, middle
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        df = self.calculate_technical_indicators(df)
        
        # Price-based features
        for period in self.config.lookback_periods:
            df[f'Return_{period}'] = df['Close'].pct_change(period)
            df[f'MA_{period}'] = df['Close'].rolling(period).mean()
            df[f'Price_MA_Ratio_{period}'] = df['Close'] / df[f'MA_{period}']
            df[f'Volatility_{period}'] = df['Close'].rolling(period).std()
            
        # Momentum features
        df['Price_Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Price_Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Price_Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volume features (if available)
        if 'Volume' in df.columns:
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Price_Volume_Trend'] = df['Close'].pct_change() * df['Volume']
            
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['Hour'] = df['timestamp'].dt.hour
            df['DayOfWeek'] = df['timestamp'].dt.dayofweek
            df['IsWeekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        else:
            df['Hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
            df['DayOfWeek'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
            df['IsWeekend'] = 0
        
        return df
        
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for classification"""
        df = df.copy()
        
        # Future return
        future_return = df['Close'].shift(-self.config.prediction_horizon) / df['Close'] - 1
        
        # Classification labels
        df['Target_Return'] = future_return
        df['Target_Class'] = 0  # Neutral
        df.loc[future_return > self.config.classification_threshold, 'Target_Class'] = 1  # Up
        df.loc[future_return < -self.config.classification_threshold, 'Target_Class'] = -1  # Down
        
        return df

class LSTMPredictor:
    """LSTM-based price prediction model"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_lstm_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # Select features
        feature_cols = [col for col in df.columns if not col.startswith('Target_') and col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']]
        feature_cols = [col for col in feature_cols if not df[col].isna().all()]
        
        self.feature_columns = feature_cols
        features = df[feature_cols].ffill().fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(df['Target_Return'].iloc[i])
            
        return np.array(X), np.array(y)
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model"""
        model = Sequential([
            LSTM(self.config.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.config.lstm_dropout),
            BatchNormalization(),
            
            LSTM(self.config.lstm_units // 2, return_sequences=False),
            Dropout(self.config.lstm_dropout),
            BatchNormalization(),
            
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
        
    def train(self, df: pd.DataFrame, sequence_length: int = 60, validation_split: float = 0.2):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
            
        print("Preparing LSTM training data...")
        X, y = self.prepare_lstm_data(df, sequence_length)
        
        # Remove NaN values
        valid_indices = ~np.isnan(y)
        X, y = X[valid_indices], y[valid_indices]
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.config.lstm_epochs,
            batch_size=self.config.lstm_batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def predict(self, df: pd.DataFrame, sequence_length: int = 60) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Prepare data
        features = df[self.feature_columns].ffill().fillna(0)
        features_scaled = self.scaler.transform(features)
        
        # Create sequences
        X = []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            
        if len(X) == 0:
            return np.array([])
            
        X = np.array(X)
        predictions = self.model.predict(X)
        
        return predictions.flatten()
        
    def save_model(self, filepath: str):
        """Save model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(f"{filepath}_lstm.h5")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.feature_columns, f"{filepath}_features.pkl")
        
    def load_model(self, filepath: str):
        """Load model and scaler"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
            
        self.model = load_model(f"{filepath}_lstm.h5")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.feature_columns = joblib.load(f"{filepath}_features.pkl")

class RandomForestPredictor:
    """Random Forest-based classification model"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_classification_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for classification"""
        # Select features
        feature_cols = [col for col in df.columns if not col.startswith('Target_') and col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']]
        feature_cols = [col for col in feature_cols if not df[col].isna().all()]
        
        self.feature_columns = feature_cols
        
        # Prepare features and target
        X = df[feature_cols].ffill().fillna(0)
        y = df['Target_Class']
        
        # Remove rows with NaN target
        valid_indices = ~y.isna()
        X, y = X[valid_indices], y[valid_indices]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
        
    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        """Train Random Forest model"""
        print("Preparing classification training data...")
        X, y = self.prepare_classification_data(df)
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Class distribution: {np.bincount(y + 1)}")  # +1 to handle -1, 0, 1 classes
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance
        }
        
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if self.feature_columns is None:
            raise ValueError("Model not trained yet")
            
        # Prepare features
        X = df[self.feature_columns].fillna(method='ffill').fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
        
    def save_model(self, filepath: str):
        """Save model and scaler"""
        joblib.dump(self.model, f"{filepath}_rf.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.feature_columns, f"{filepath}_features.pkl")
        
    def load_model(self, filepath: str):
        """Load model and scaler"""
        self.model = joblib.load(f"{filepath}_rf.pkl")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.feature_columns = joblib.load(f"{filepath}_features.pkl")

class MLPredictor:
    """Main ML predictor class"""
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.feature_engineer = FeatureEngineer(self.config)
        self.lstm_model = LSTMPredictor(self.config) if TENSORFLOW_AVAILABLE else None
        self.rf_model = RandomForestPredictor(self.config)
        self.downloader = CryptoDownloader()
        
    def prepare_training_data(self, symbol: str, timeframe: str = "1h", days: int = 365) -> pd.DataFrame:
        """Prepare training data for a symbol"""
        print(f"Downloading data for {symbol}...")
        
        # Calculate time range
        end_ts = int(datetime.now().timestamp())
        start_ts = end_ts - (days * 24 * 3600)
        
        # Get data
        success, df = self.downloader.get_data(symbol, start_ts, end_ts, timeframe=timeframe)
        
        if not success or df.empty:
            raise ValueError(f"Failed to get data for {symbol}")
            
        print(f"Downloaded {len(df)} data points")
        
        # Check and standardize column names
        print(f"Data columns: {df.columns.tolist()}")
        
        # Standardize column names
        column_mapping = {
            'close': 'Close',
            'open': 'Open', 
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Feature engineering
        print("Creating features...")
        df = self.feature_engineer.create_features(df)
        df = self.feature_engineer.create_target_variable(df)
        
        # Remove rows with insufficient data
        df = df.dropna(subset=['Target_Class'])
        
        print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        
        return df
        
    def train_models(self, symbol: str, timeframe: str = "1h", days: int = 365):
        """Train both LSTM and Random Forest models"""
        # Prepare data
        df = self.prepare_training_data(symbol, timeframe, days)
        
        # Train Random Forest
        print("\n" + "="*50)
        print("Training Random Forest Model")
        print("="*50)
        rf_results = self.rf_model.train(df)
        
        # Train LSTM if available
        if self.lstm_model is not None:
            print("\n" + "="*50)
            print("Training LSTM Model")
            print("="*50)
            lstm_history = self.lstm_model.train(df)
        else:
            print("\nLSTM model not available (TensorFlow not installed)")
            lstm_history = None
            
        return {
            'rf_results': rf_results,
            'lstm_history': lstm_history,
            'data_shape': df.shape
        }
        
    def predict_symbol(self, symbol: str, timeframe: str = "1h", hours_back: int = 168) -> Dict:
        """Make predictions for a symbol"""
        print(f"Making predictions for {symbol}...")
        
        # Get recent data
        end_ts = int(datetime.now().timestamp())
        start_ts = end_ts - (hours_back * 3600)
        
        success, df = self.downloader.get_data(symbol, start_ts, end_ts, timeframe=timeframe)
        
        if not success or df.empty:
            raise ValueError(f"Failed to get data for {symbol}")
            
        # Feature engineering
        df = self.feature_engineer.create_features(df)
        
        # Random Forest predictions
        rf_pred, rf_prob = self.rf_model.predict(df)
        
        # LSTM predictions
        lstm_pred = None
        if self.lstm_model is not None and self.lstm_model.model is not None:
            lstm_pred = self.lstm_model.predict(df)
            
        # Current market data
        current_price = df['Close'].iloc[-1]
        current_time = df.index[-1]
        
        # Prepare results
        results = {
            'symbol': symbol,
            'current_price': current_price,
            'current_time': current_time,
            'rf_prediction': rf_pred[-1] if len(rf_pred) > 0 else None,
            'rf_probability': rf_prob[-1] if len(rf_prob) > 0 else None,
            'lstm_prediction': lstm_pred[-1] if lstm_pred is not None and len(lstm_pred) > 0 else None,
            'prediction_horizon_hours': self.config.prediction_horizon
        }
        
        # Interpret predictions
        if results['rf_prediction'] is not None:
            rf_class = results['rf_prediction']
            if rf_class == 1:
                results['rf_signal'] = 'BUY'
            elif rf_class == -1:
                results['rf_signal'] = 'SELL'
            else:
                results['rf_signal'] = 'HOLD'
                
        if results['lstm_prediction'] is not None:
            lstm_return = results['lstm_prediction']
            if lstm_return > self.config.classification_threshold:
                results['lstm_signal'] = 'BUY'
            elif lstm_return < -self.config.classification_threshold:
                results['lstm_signal'] = 'SELL'
            else:
                results['lstm_signal'] = 'HOLD'
                
        return results
        
    def batch_predict(self, symbols: List[str], timeframe: str = "1h") -> pd.DataFrame:
        """Make predictions for multiple symbols"""
        results = []
        
        for symbol in symbols:
            try:
                pred_result = self.predict_symbol(symbol, timeframe)
                results.append(pred_result)
                print(f"✓ {symbol}: RF={pred_result.get('rf_signal', 'N/A')}, LSTM={pred_result.get('lstm_signal', 'N/A')}")
            except Exception as e:
                print(f"✗ {symbol}: Error - {e}")
                
        return pd.DataFrame(results)
        
    def save_models(self, filepath: str):
        """Save all trained models"""
        self.rf_model.save_model(filepath)
        if self.lstm_model is not None and self.lstm_model.model is not None:
            self.lstm_model.save_model(filepath)
            
    def load_models(self, filepath: str):
        """Load all trained models"""
        self.rf_model.load_model(filepath)
        if self.lstm_model is not None:
            try:
                self.lstm_model.load_model(filepath)
            except:
                print("LSTM model not found or failed to load")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Predictor for Market Analysis')
    parser.add_argument('--train', type=str, help='Train models for symbol')
    parser.add_argument('--predict', type=str, help='Make prediction for symbol')
    parser.add_argument('--batch-predict', type=str, help='File with symbols for batch prediction')
    parser.add_argument('--timeframe', default='1h', help='Timeframe for analysis')
    parser.add_argument('--days', type=int, default=365, help='Days of training data')
    parser.add_argument('--save', type=str, help='Save models to file')
    parser.add_argument('--load', type=str, help='Load models from file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    config = MLConfig()
    predictor = MLPredictor(config)
    
    # Load models if specified
    if args.load:
        print(f"Loading models from {args.load}...")
        predictor.load_models(args.load)
        
    # Train models
    if args.train:
        print(f"Training models for {args.train}...")
        results = predictor.train_models(args.train, args.timeframe, args.days)
        print("\nTraining completed!")
        
        # Save models if specified
        if args.save:
            print(f"Saving models to {args.save}...")
            predictor.save_models(args.save)
            
    # Single prediction
    if args.predict:
        try:
            result = predictor.predict_symbol(args.predict, args.timeframe)
            print(f"\nPrediction for {args.predict}:")
            print(f"Current Price: ${result['current_price']:.4f}")
            print(f"RF Signal: {result.get('rf_signal', 'N/A')}")
            print(f"LSTM Signal: {result.get('lstm_signal', 'N/A')}")
        except Exception as e:
            print(f"Error making prediction: {e}")
            
    # Batch prediction
    if args.batch_predict:
        try:
            with open(args.batch_predict, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
                
            print(f"Making predictions for {len(symbols)} symbols...")
            results_df = predictor.batch_predict(symbols, args.timeframe)
            
            # Save results
            output_file = f"ml_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error in batch prediction: {e}")

if __name__ == "__main__":
    main() 