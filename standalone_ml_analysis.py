#!/usr/bin/env python3
"""
Standalone ML Analysis for 07-03 Screening Results
獨立機器學習分析 - 不依賴額外庫
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.downloader import CryptoDownloader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class StandaloneMLAnalysis:
    """獨立ML分析類 - 使用基本技術指標"""
    
    def __init__(self):
        self.top_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT',
            'AVAXUSDT', 'ATOMUSDT', 'FILUSDT', 'LINKUSDT', 'UNIUSDT'
        ]
        self.downloader = CryptoDownloader()
        
    def calculate_rsi(self, prices, period=14):
        """計算RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rs = np.zeros_like(prices)
        rsi = np.zeros_like(prices)
        
        for i in range(period, len(prices)):
            if i == period:
                rs[i] = avg_gain / avg_loss if avg_loss != 0 else 0
            else:
                avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
                rs[i] = avg_gain / avg_loss if avg_loss != 0 else 0
            
            rsi[i] = 100 - (100 / (1 + rs[i])) if rs[i] != 0 else 50
            
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """計算MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """計算布林帶"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma
    
    def prepare_data_with_features(self, symbol, timeframe='1h', days=90):
        """準備帶有技術指標的數據"""
        print(f"📊 準備 {symbol} 的數據...")
        
        # 獲取數據
        end_ts = int(datetime.now().timestamp())
        start_ts = end_ts - (days * 24 * 3600)
        
        success, df = self.downloader.get_data(symbol, start_ts, end_ts, timeframe=timeframe)
        
        if not success or df.empty:
            raise ValueError(f"無法獲取 {symbol} 的數據")
        
        print(f"✅ 獲取到 {len(df)} 個數據點")
        
        # 確保我們有正確的列名
        if 'close' not in df.columns:
            df = df.rename(columns={'Close': 'close'})
        if 'high' not in df.columns:
            df = df.rename(columns={'High': 'high'})
        if 'low' not in df.columns:
            df = df.rename(columns={'Low': 'low'})
        if 'open' not in df.columns:
            df = df.rename(columns={'Open': 'open'})
        if 'volume' not in df.columns:
            df = df.rename(columns={'Volume': 'volume'})
        
        # 計算技術指標
        df = self.calculate_technical_indicators(df)
        
        # 創建目標變量
        df = self.create_target_variable(df)
        
        # 移除空值
        df = df.dropna()
        
        return df
    
    def calculate_technical_indicators(self, df):
        """計算技術指標"""
        print("🔧 計算技術指標...")
        
        # 移動平均線
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'].values)
        
        # MACD
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 布林帶
        df['bb_upper'], df['bb_lower'], df['bb_middle'] = self.calculate_bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(14).mean()
        
        # 價格變化
        df['price_change_1'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        # 成交量指標
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        else:
            df['volume_ratio'] = 1.0
        
        # 相對位置
        df['price_position'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        
        # 動量指標
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # 波動率
        df['volatility'] = df['close'].rolling(20).std()
        
        return df
    
    def create_target_variable(self, df, horizon=24, threshold=0.025):
        """創建目標變量"""
        # 未來收益率
        future_return = df['close'].shift(-horizon) / df['close'] - 1
        
        # 分類標籤
        df['target_return'] = future_return
        df['target_class'] = 0  # 持有
        df.loc[future_return > threshold, 'target_class'] = 1  # 買入
        df.loc[future_return < -threshold, 'target_class'] = -1  # 賣出
        
        return df
    
    def train_random_forest(self, df, symbol):
        """訓練隨機森林模型"""
        print(f"🌲 訓練 {symbol} 的隨機森林模型...")
        
        # 選擇特徵
        feature_cols = ['ma_5', 'ma_10', 'ma_20', 'ma_50', 'rsi', 'macd', 'macd_signal',
                       'macd_histogram', 'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
                       'bb_position', 'atr', 'price_change_1', 'price_change_5', 'price_change_10',
                       'volume_ratio', 'price_position', 'momentum_5', 'momentum_10', 'volatility']
        
        # 過濾存在的特徵
        available_features = [col for col in feature_cols if col in df.columns]
        
        # 準備數據
        X = df[available_features].fillna(method='ffill').fillna(0)
        y = df['target_class']
        
        # 移除無效行
        valid_indices = ~y.isna()
        X, y = X[valid_indices], y[valid_indices]
        
        if len(X) < 100:
            raise ValueError(f"數據量不足: {len(X)} 行")
        
        # 檢查類別分佈
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("目標變量只有一個類別")
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 訓練模型
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        
        # 評估
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        
        print(f"✅ 訓練完成 - 訓練準確率: {train_score:.3f}, 測試準確率: {test_score:.3f}")
        print(f"   類別分佈: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # 特徵重要性
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   前5個重要特徵: {feature_importance.head(5)['feature'].tolist()}")
        
        # 預測最新數據
        latest_X = X_scaled[-1:] if len(X_scaled) > 0 else None
        prediction = rf.predict(latest_X)[0] if latest_X is not None else 0
        probability = rf.predict_proba(latest_X)[0] if latest_X is not None else [0.33, 0.33, 0.34]
        
        return {
            'model': rf,
            'scaler': scaler,
            'features': available_features,
            'prediction': prediction,
            'probability': probability,
            'train_score': train_score,
            'test_score': test_score,
            'current_price': df['close'].iloc[-1],
            'feature_importance': feature_importance
        }
    
    def analyze_symbol(self, symbol, timeframe='1h', days=90):
        """分析單個符號"""
        try:
            # 準備數據
            df = self.prepare_data_with_features(symbol, timeframe, days)
            
            # 訓練模型
            results = self.train_random_forest(df, symbol)
            
            # 生成信號
            prediction = results['prediction']
            probability = results['probability']
            confidence = max(probability)
            
            # 映射預測到信號
            class_to_signal = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
            signal = class_to_signal.get(prediction, 'HOLD')
            
            # 生成建議
            if signal == 'BUY' and confidence > 0.6:
                recommendation = '強烈買入'
                risk_level = 'MEDIUM'
            elif signal == 'BUY':
                recommendation = '買入'
                risk_level = 'MEDIUM'
            elif signal == 'SELL' and confidence > 0.6:
                recommendation = '強烈賣出'
                risk_level = 'HIGH'
            elif signal == 'SELL':
                recommendation = '賣出'
                risk_level = 'HIGH'
            else:
                recommendation = '持有'
                risk_level = 'LOW'
            
            # 計算價格變化
            recent_data = df.tail(24)
            if len(recent_data) > 1:
                price_change_24h = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0] * 100
            else:
                price_change_24h = 0
            
            # 計算額外統計信息
            recent_volatility = recent_data['close'].std() / recent_data['close'].mean() * 100
            
            return {
                'symbol': symbol,
                'current_price': results['current_price'],
                'rf_prediction': prediction,
                'rf_signal': signal,
                'rf_confidence': confidence,
                'price_change_24h': price_change_24h,
                'volatility': recent_volatility,
                'recommendation': recommendation,
                'risk_level': risk_level,
                'train_score': results['train_score'],
                'test_score': results['test_score'],
                'top_features': results['feature_importance'].head(3)['feature'].tolist(),
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            print(f"❌ {symbol} 分析失敗: {str(e)}")
            return {
                'symbol': symbol,
                'current_price': 0,
                'rf_prediction': 0,
                'rf_signal': 'ERROR',
                'rf_confidence': 0,
                'price_change_24h': 0,
                'volatility': 0,
                'recommendation': 'ERROR',
                'risk_level': 'HIGH',
                'train_score': 0,
                'test_score': 0,
                'top_features': [],
                'status': 'ERROR'
            }
    
    def run_analysis(self, timeframe='1h', days=90):
        """運行完整分析"""
        print("🚀 開始獨立機器學習分析")
        print(f"📊 基於07-03篩選結果的Top 10加密貨幣")
        print(f"🌲 使用隨機森林模型進行預測")
        print(f"⏰ 時間框架: {timeframe}, 訓練天數: {days}")
        print("="*70)
        
        results = []
        
        for i, symbol in enumerate(self.top_symbols, 1):
            print(f"\n📈 [{i}/{len(self.top_symbols)}] 分析 {symbol}")
            print("-" * 50)
            
            result = self.analyze_symbol(symbol, timeframe, days)
            results.append(result)
            
            if result['status'] == 'SUCCESS':
                print(f"✅ {symbol} 分析完成")
                print(f"   當前價格: ${result['current_price']:.4f}")
                print(f"   信號: {result['rf_signal']}")
                print(f"   置信度: {result['rf_confidence']:.3f}")
                print(f"   建議: {result['recommendation']}")
                print(f"   24h變化: {result['price_change_24h']:+.2f}%")
                print(f"   重要特徵: {', '.join(result['top_features'])}")
        
        # 創建結果DataFrame
        df_results = pd.DataFrame(results)
        
        # 生成報告
        self.generate_report(df_results)
        
        # 保存結果
        filename = f"standalone_ml_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        filepath = f"/Users/ting/技術分析/screener/output/{filename}"
        df_results.to_csv(filepath, index=False)
        print(f"💾 結果已保存到: {filepath}")
        
        return df_results, filepath
    
    def generate_report(self, df_results):
        """生成分析報告"""
        print("\n" + "="*80)
        print("📊 獨立機器學習分析報告")
        print("="*80)
        
        # 過濾成功的結果
        successful_results = df_results[df_results['status'] == 'SUCCESS']
        
        if len(successful_results) == 0:
            print("❌ 沒有成功的分析結果")
            return
        
        print(f"\n🔍 分析概況:")
        print(f"   成功分析: {len(successful_results)}/{len(df_results)} 個符號")
        print(f"   成功率: {len(successful_results)/len(df_results)*100:.1f}%")
        print(f"   分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 按置信度排序
        sorted_results = successful_results.sort_values('rf_confidence', ascending=False)
        
        print(f"\n🏆 按置信度排名:")
        for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
            print(f"   {i}. {row['symbol']}: {row['rf_confidence']:.3f} - {row['rf_signal']} ({row['recommendation']})")
        
        # 信號統計
        signal_counts = successful_results['rf_signal'].value_counts()
        print(f"\n🎯 信號統計:")
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} ({count/len(successful_results)*100:.1f}%)")
        
        # 買入建議
        buy_signals = successful_results[successful_results['rf_signal'] == 'BUY'].sort_values('rf_confidence', ascending=False)
        if len(buy_signals) > 0:
            print(f"\n🔥 買入建議 (按置信度排序):")
            for i, (_, row) in enumerate(buy_signals.iterrows(), 1):
                print(f"   {i}. {row['symbol']}: ${row['current_price']:.4f} "
                      f"(置信度: {row['rf_confidence']:.3f}, 24h: {row['price_change_24h']:+.2f}%)")
        
        # 賣出警告
        sell_signals = successful_results[successful_results['rf_signal'] == 'SELL'].sort_values('rf_confidence', ascending=False)
        if len(sell_signals) > 0:
            print(f"\n❄️ 賣出警告 (按置信度排序):")
            for i, (_, row) in enumerate(sell_signals.iterrows(), 1):
                print(f"   {i}. {row['symbol']}: ${row['current_price']:.4f} "
                      f"(置信度: {row['rf_confidence']:.3f}, 24h: {row['price_change_24h']:+.2f}%)")
        
        # 模型性能統計
        avg_train_score = successful_results['train_score'].mean()
        avg_test_score = successful_results['test_score'].mean()
        avg_confidence = successful_results['rf_confidence'].mean()
        
        print(f"\n📊 模型性能統計:")
        print(f"   平均訓練準確率: {avg_train_score:.3f}")
        print(f"   平均測試準確率: {avg_test_score:.3f}")
        print(f"   平均預測置信度: {avg_confidence:.3f}")
        
        # 市場概況
        avg_price_change = successful_results['price_change_24h'].mean()
        avg_volatility = successful_results['volatility'].mean()
        
        print(f"\n📈 市場概況:")
        print(f"   平均24小時變化: {avg_price_change:+.2f}%")
        print(f"   平均波動率: {avg_volatility:.2f}%")
        print(f"   高風險符號: {len(successful_results[successful_results['risk_level'] == 'HIGH'])}")
        
        # 風險評估
        risk_counts = successful_results['risk_level'].value_counts()
        print(f"\n⚠️ 風險等級分布:")
        for risk, count in risk_counts.items():
            print(f"   {risk}: {count} ({count/len(successful_results)*100:.1f}%)")
        
        print(f"\n💡 投資建議摘要:")
        strong_buy = len(successful_results[successful_results['recommendation'] == '強烈買入'])
        buy = len(successful_results[successful_results['recommendation'] == '買入'])
        hold = len(successful_results[successful_results['recommendation'] == '持有'])
        sell = len(successful_results[successful_results['recommendation'] == '賣出'])
        strong_sell = len(successful_results[successful_results['recommendation'] == '強烈賣出'])
        
        print(f"   強烈買入: {strong_buy}, 買入: {buy}, 持有: {hold}")
        print(f"   賣出: {sell}, 強烈賣出: {strong_sell}")
        
        print(f"\n⚠️ 免責聲明:")
        print(f"   • 本分析基於機器學習模型，僅供參考")
        print(f"   • 加密貨幣市場高度波動，存在投資風險")
        print(f"   • 建議進行進一步研究和風險評估")
        print(f"   • 請根據自身風險承受能力做出投資決策")
        
        print("="*80)

def main():
    """主函數"""
    analyzer = StandaloneMLAnalysis()
    results, filepath = analyzer.run_analysis(timeframe='1h', days=90)
    
    print(f"\n🎉 分析完成！")
    print(f"📄 詳細結果已保存到: {filepath}")
    print("🔍 建議查看CSV文件獲取完整數據和進一步分析")

if __name__ == "__main__":
    main()