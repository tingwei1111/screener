#!/usr/bin/env python3
"""
Comprehensive ML Analysis for ALL 07-03 Screening Results
全面機器學習分析 - 覆蓋07-03篩選的所有幣種
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
warnings.filterwarnings('ignore')

from src.downloader import CryptoDownloader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ComprehensiveMLAnalysis:
    """全面機器學習分析類 - 處理200+幣種"""
    
    def __init__(self):
        self.downloader = CryptoDownloader()
        self.results = []
        self.failed_symbols = []
        self.lock = threading.Lock()
        
        # 從07-03結果提取所有符號
        self.all_symbols = self.extract_all_symbols()
        print(f"📊 提取到 {len(self.all_symbols)} 個交易對")
        
    def extract_all_symbols(self):
        """從07-03篩選結果提取所有符號"""
        with open("/Users/ting/技術分析/screener/output/2025-07-03/2025-07-03_23-17_crypto_1h_optimized_targets.txt", 'r') as f:
            content = f.read()
        
        # 提取所有BINANCE:符號
        pattern = r'BINANCE:([^,\s]+)'
        matches = re.findall(pattern, content)
        
        # 轉換為現貨交易對格式
        symbols = []
        for match in matches:
            # 移除.P後綴並保留USDT結尾的符號
            if match.endswith('.P'):
                symbol = match[:-2]  # 移除.P
            else:
                symbol = match
            
            if symbol.endswith('USDT') and symbol not in symbols:
                symbols.append(symbol)
        
        # 過濾掉一些可能有問題的符號
        filtered_symbols = []
        for symbol in symbols:
            # 跳過一些特殊或不常見的符號
            skip_patterns = [
                'BTCDOM',  # BTC dominance
                '1000000',  # 可能有精度問題
                'JELLYJELLY',  # 特殊符號
                'BROCCOLI',  # 特殊符號
                'BANANAS31',  # 特殊符號
                'BROCCOLIF3B'  # 特殊符號
            ]
            
            if not any(pattern in symbol for pattern in skip_patterns):
                filtered_symbols.append(symbol)
        
        return sorted(filtered_symbols)
    
    def calculate_technical_indicators(self, df):
        """計算技術指標"""
        try:
            # 移動平均線
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_10'] = df['close'].rolling(10).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['ma_50'] = df['close'].rolling(50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # 布林帶
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR
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
        except Exception as e:
            print(f"技術指標計算錯誤: {e}")
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
    
    def analyze_single_symbol(self, symbol, timeframe='1h', days=60):
        """分析單個符號"""
        try:
            # 獲取數據
            end_ts = int(datetime.now().timestamp())
            start_ts = end_ts - (days * 24 * 3600)
            
            success, df = self.downloader.get_data(symbol, start_ts, end_ts, timeframe=timeframe)
            
            if not success or df.empty or len(df) < 100:
                return None
            
            # 標準化列名
            df = df.rename(columns={
                'Close': 'close', 'High': 'high', 'Low': 'low', 
                'Open': 'open', 'Volume': 'volume'
            })
            
            # 計算技術指標
            df = self.calculate_technical_indicators(df)
            df = self.create_target_variable(df)
            df = df.dropna()
            
            if len(df) < 50:
                return None
            
            # 準備特徵
            feature_cols = ['ma_5', 'ma_10', 'ma_20', 'ma_50', 'rsi', 'macd', 'macd_signal',
                           'macd_histogram', 'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
                           'bb_position', 'atr', 'price_change_1', 'price_change_5', 'price_change_10',
                           'volume_ratio', 'price_position', 'momentum_5', 'momentum_10', 'volatility']
            
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) < 10:
                return None
            
            X = df[available_features].fillna(method='ffill').fillna(0)
            y = df['target_class']
            
            valid_indices = ~y.isna()
            X, y = X[valid_indices], y[valid_indices]
            
            if len(X) < 30 or len(np.unique(y)) < 2:
                return None
            
            # 訓練模型
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            rf.fit(X_train, y_train)
            
            train_score = rf.score(X_train, y_train)
            test_score = rf.score(X_test, y_test)
            
            # 預測
            latest_X = X_scaled[-1:] if len(X_scaled) > 0 else None
            prediction = rf.predict(latest_X)[0] if latest_X is not None else 0
            probability = rf.predict_proba(latest_X)[0] if latest_X is not None else [0.33, 0.33, 0.34]
            confidence = max(probability)
            
            # 信號映射
            signal_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
            signal = signal_map.get(prediction, 'HOLD')
            
            # 建議映射
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
            
            # 計算統計信息
            current_price = df['close'].iloc[-1]
            recent_data = df.tail(24)
            if len(recent_data) > 1:
                price_change_24h = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0] * 100
            else:
                price_change_24h = 0
            
            volatility = recent_data['close'].std() / recent_data['close'].mean() * 100 if len(recent_data) > 1 else 0
            
            # 特徵重要性
            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance.head(3)['feature'].tolist()
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'rf_prediction': prediction,
                'rf_signal': signal,
                'rf_confidence': confidence,
                'price_change_24h': price_change_24h,
                'volatility': volatility,
                'recommendation': recommendation,
                'risk_level': risk_level,
                'train_score': train_score,
                'test_score': test_score,
                'top_features': top_features,
                'data_points': len(df),
                'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'status': 'ERROR'
            }
    
    def run_batch_analysis(self, max_workers=5, batch_size=50):
        """運行批量分析"""
        print("🚀 開始全面機器學習分析")
        print(f"📊 總共 {len(self.all_symbols)} 個交易對")
        print(f"🔧 使用 {max_workers} 個並行線程")
        print("="*80)
        
        start_time = time.time()
        
        # 分批處理
        for i in range(0, len(self.all_symbols), batch_size):
            batch_symbols = self.all_symbols[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(self.all_symbols) + batch_size - 1) // batch_size
            
            print(f"\n🔄 處理批次 {batch_num}/{total_batches} ({len(batch_symbols)} 個符號)")
            print("-" * 60)
            
            # 並行處理當前批次
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.analyze_single_symbol, symbol): symbol 
                    for symbol in batch_symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result(timeout=30)  # 30秒超時
                        if result:
                            with self.lock:
                                if result['status'] == 'SUCCESS':
                                    self.results.append(result)
                                    print(f"✅ {symbol}: {result['rf_signal']} (置信度: {result['rf_confidence']:.2f})")
                                else:
                                    self.failed_symbols.append(symbol)
                                    print(f"❌ {symbol}: {result.get('error', 'Unknown error')}")
                        else:
                            with self.lock:
                                self.failed_symbols.append(symbol)
                                print(f"⚠️ {symbol}: 數據不足")
                    except Exception as e:
                        with self.lock:
                            self.failed_symbols.append(symbol)
                            print(f"💥 {symbol}: 處理失敗 - {str(e)}")
            
            # 批次間短暫休息
            if i + batch_size < len(self.all_symbols):
                print(f"⏱️ 批次完成，休息3秒...")
                time.sleep(3)
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 分析完成！耗時: {elapsed_time:.1f}秒")
        print(f"✅ 成功: {len(self.results)} 個")
        print(f"❌ 失敗: {len(self.failed_symbols)} 個")
        print(f"📊 成功率: {len(self.results)/(len(self.results)+len(self.failed_symbols))*100:.1f}%")
        
        return self.results
    
    def generate_comprehensive_report(self):
        """生成全面分析報告"""
        if not self.results:
            print("❌ 沒有可用的分析結果")
            return
        
        df_results = pd.DataFrame(self.results)
        
        print("\n" + "="*100)
        print("📊 全面機器學習分析報告 - 07-03篩選結果所有幣種")
        print("="*100)
        
        # 基本統計
        print(f"\n🔍 分析概況:")
        print(f"   成功分析: {len(df_results)} 個交易對")
        print(f"   失敗分析: {len(self.failed_symbols)} 個交易對")
        print(f"   總體成功率: {len(df_results)/(len(df_results)+len(self.failed_symbols))*100:.1f}%")
        print(f"   分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 信號分佈
        signal_counts = df_results['rf_signal'].value_counts()
        print(f"\n🎯 信號分佈:")
        for signal, count in signal_counts.items():
            percentage = count / len(df_results) * 100
            print(f"   {signal}: {count} 個 ({percentage:.1f}%)")
        
        # 置信度統計
        avg_confidence = df_results['rf_confidence'].mean()
        high_confidence = len(df_results[df_results['rf_confidence'] > 0.7])
        print(f"\n📊 置信度統計:")
        print(f"   平均置信度: {avg_confidence:.3f}")
        print(f"   高置信度(>70%): {high_confidence} 個 ({high_confidence/len(df_results)*100:.1f}%)")
        
        # 模型性能
        avg_train_score = df_results['train_score'].mean()
        avg_test_score = df_results['test_score'].mean()
        print(f"\n🤖 模型性能:")
        print(f"   平均訓練準確率: {avg_train_score:.3f}")
        print(f"   平均測試準確率: {avg_test_score:.3f}")
        
        # Top買入信號
        buy_signals = df_results[df_results['rf_signal'] == 'BUY'].sort_values('rf_confidence', ascending=False)
        print(f"\n🔥 買入信號 (按置信度排序, Top 20):")
        for i, (_, row) in enumerate(buy_signals.head(20).iterrows(), 1):
            print(f"   {i:2d}. {row['symbol']:15s}: ${row['current_price']:>10.4f} (置信度: {row['rf_confidence']:.3f})")
        
        # Top賣出信號
        sell_signals = df_results[df_results['rf_signal'] == 'SELL'].sort_values('rf_confidence', ascending=False)
        print(f"\n❄️ 賣出信號 (按置信度排序, Top 20):")
        for i, (_, row) in enumerate(sell_signals.head(20).iterrows(), 1):
            print(f"   {i:2d}. {row['symbol']:15s}: ${row['current_price']:>10.4f} (置信度: {row['rf_confidence']:.3f})")
        
        # 高置信度標的
        high_conf_signals = df_results[df_results['rf_confidence'] > 0.8].sort_values('rf_confidence', ascending=False)
        print(f"\n⭐ 高置信度信號 (>80%, Top 15):")
        for i, (_, row) in enumerate(high_conf_signals.head(15).iterrows(), 1):
            print(f"   {i:2d}. {row['symbol']:15s}: {row['rf_signal']:4s} (置信度: {row['rf_confidence']:.3f}) - {row['recommendation']}")
        
        # 風險統計
        risk_counts = df_results['risk_level'].value_counts()
        print(f"\n⚠️ 風險等級分佈:")
        for risk, count in risk_counts.items():
            print(f"   {risk}: {count} 個 ({count/len(df_results)*100:.1f}%)")
        
        # 異常波動
        high_volatility = df_results[df_results['volatility'] > 5].sort_values('volatility', ascending=False)
        if len(high_volatility) > 0:
            print(f"\n🌊 高波動標的 (>5%, Top 10):")
            for i, (_, row) in enumerate(high_volatility.head(10).iterrows(), 1):
                print(f"   {i:2d}. {row['symbol']:15s}: {row['volatility']:5.1f}% (信號: {row['rf_signal']})")
        
        # 異常價格變化
        big_movers = df_results[abs(df_results['price_change_24h']) > 10].sort_values('price_change_24h', ascending=False)
        if len(big_movers) > 0:
            print(f"\n📈 大幅波動標的 (24h >10%):")
            for i, (_, row) in enumerate(big_movers.iterrows(), 1):
                print(f"   {i:2d}. {row['symbol']:15s}: {row['price_change_24h']:+6.1f}% (信號: {row['rf_signal']})")
        
        return df_results
    
    def save_results(self, df_results):
        """保存結果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # 保存成功結果
        success_file = f"/Users/ting/技術分析/screener/output/comprehensive_ml_analysis_{timestamp}.csv"
        df_results.to_csv(success_file, index=False)
        print(f"\n💾 成功結果已保存: {success_file}")
        
        # 保存失敗列表
        if self.failed_symbols:
            failed_file = f"/Users/ting/技術分析/screener/output/failed_symbols_{timestamp}.txt"
            with open(failed_file, 'w') as f:
                f.write("# 分析失敗的交易對\n")
                for symbol in self.failed_symbols:
                    f.write(f"{symbol}\n")
            print(f"❌ 失敗列表已保存: {failed_file}")
        
        # 生成摘要報告
        summary_file = f"/Users/ting/技術分析/screener/output/analysis_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"07-03篩選結果全面ML分析摘要\n")
            f.write(f"="*50 + "\n")
            f.write(f"分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"成功分析: {len(df_results)} 個\n")
            f.write(f"失敗分析: {len(self.failed_symbols)} 個\n")
            f.write(f"成功率: {len(df_results)/(len(df_results)+len(self.failed_symbols))*100:.1f}%\n\n")
            
            # 信號統計
            signal_counts = df_results['rf_signal'].value_counts()
            f.write("信號分佈:\n")
            for signal, count in signal_counts.items():
                f.write(f"  {signal}: {count} 個 ({count/len(df_results)*100:.1f}%)\n")
        
        print(f"📋 摘要報告已保存: {summary_file}")
        
        return success_file, summary_file
    
    def run_complete_analysis(self):
        """運行完整分析"""
        # 運行批量分析
        results = self.run_batch_analysis()
        
        if results:
            # 生成報告
            df_results = self.generate_comprehensive_report()
            
            # 保存結果
            success_file, summary_file = self.save_results(df_results)
            
            return df_results, success_file, summary_file
        else:
            print("❌ 沒有成功的分析結果")
            return None, None, None

def main():
    """主函數"""
    print("🚀 啟動07-03篩選結果全面機器學習分析")
    print("📊 這將分析200+個加密貨幣交易對")
    print("⏱️ 預計需要15-30分鐘完成")
    print("="*80)
    
    analyzer = ComprehensiveMLAnalysis()
    df_results, success_file, summary_file = analyzer.run_complete_analysis()
    
    if df_results is not None:
        print(f"\n🎉 全面分析完成！")
        print(f"📄 詳細結果: {success_file}")
        print(f"📋 摘要報告: {summary_file}")
        print(f"🔍 共分析了 {len(df_results)} 個交易對")
    else:
        print("❌ 分析失敗")

if __name__ == "__main__":
    main()