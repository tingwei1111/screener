#!/usr/bin/env python3
"""
Simplified ML Analysis for 07-03 Screening Results
簡化版機器學習分析 - 專注於隨機森林模型
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 禁用TensorFlow以避免Metal問題
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ml_predictor import MLPredictor, MLConfig

class SimplifiedMLAnalysis:
    """簡化版ML分析類 - 只使用隨機森林"""
    
    def __init__(self):
        # 從07-03篩選結果中提取top 10加密貨幣
        self.top_symbols = [
            'BTCUSDT',  # 從BINANCE:BTCUSDT.P轉換
            'ETHUSDT',  # 從BINANCE:ETHUSDT.P轉換  
            'BNBUSDT',  # BNB是較穩定的選擇
            'ADAUSDT',  # ADA是較穩定的選擇
            'DOTUSDT',  # DOT是較穩定的選擇
            'AVAXUSDT', # AVAX是較穩定的選擇
            'ATOMUSDT', # ATOM是較穩定的選擇
            'FILUSDT',  # FIL是較穩定的選擇
            'LINKUSDT', # LINK是較穩定的選擇
            'UNIUSDT'   # UNI是較穩定的選擇
        ]
        
        # 配置ML參數 - 只使用隨機森林
        self.ml_config = MLConfig(
            rf_n_estimators=100,
            rf_max_depth=10,
            prediction_horizon=24,  # 預測24小時後
            classification_threshold=0.025  # 2.5%價格變化閾值
        )
        
        self.ml_predictor = MLPredictor(self.ml_config)
        
    def analyze_with_rf_only(self, timeframe='1h', training_days=90):
        """只使用隨機森林進行分析"""
        print("🌲 開始隨機森林分析")
        print(f"📊 分析符號: {', '.join(self.top_symbols)}")
        print(f"⏰ 時間框架: {timeframe}")
        print(f"📅 訓練天數: {training_days}")
        print("="*70)
        
        results = {
            'symbol': [],
            'current_price': [],
            'rf_prediction': [],
            'rf_signal': [],
            'rf_confidence': [],
            'price_change_24h': [],
            'recommendation': [],
            'risk_level': []
        }
        
        for i, symbol in enumerate(self.top_symbols, 1):
            print(f"\n📈 [{i}/{len(self.top_symbols)}] 分析 {symbol}")
            print("-" * 50)
            
            try:
                # 準備訓練數據
                print(f"📊 準備 {symbol} 的訓練數據...")
                df = self.ml_predictor.prepare_training_data(
                    symbol=symbol, 
                    timeframe=timeframe, 
                    days=training_days
                )
                
                # 訓練隨機森林模型
                print(f"🌲 訓練 {symbol} 的隨機森林模型...")
                rf_results = self.ml_predictor.rf_model.train(df)
                
                # 生成預測
                print(f"🎯 為 {symbol} 生成預測...")
                pred_results = self.ml_predictor.predict_symbol(
                    symbol=symbol, 
                    timeframe=timeframe,
                    hours_back=72  # 使用72小時的數據進行預測
                )
                
                # 計算24小時價格變化
                recent_data = df.tail(24)
                close_col = 'Close' if 'Close' in df.columns else 'close'
                price_change_24h = (recent_data[close_col].iloc[-1] - recent_data[close_col].iloc[0]) / recent_data[close_col].iloc[0] * 100
                
                # 提取結果
                results['symbol'].append(symbol)
                results['current_price'].append(pred_results.get('current_price', 0))
                results['rf_prediction'].append(pred_results.get('rf_prediction', 0))
                results['rf_signal'].append(pred_results.get('rf_signal', 'HOLD'))
                results['price_change_24h'].append(price_change_24h)
                
                # 計算隨機森林置信度
                rf_prob = pred_results.get('rf_probability', [0.33, 0.33, 0.34])
                rf_confidence = max(rf_prob) if rf_prob is not None else 0.5
                results['rf_confidence'].append(rf_confidence)
                
                # 生成建議和風險等級
                recommendation, risk_level = self.generate_recommendation_and_risk(
                    pred_results, rf_confidence, price_change_24h
                )
                results['recommendation'].append(recommendation)
                results['risk_level'].append(risk_level)
                
                print(f"✅ {symbol} 分析完成")
                print(f"   當前價格: ${pred_results.get('current_price', 0):.4f}")
                print(f"   RF信號: {pred_results.get('rf_signal', 'N/A')}")
                print(f"   置信度: {rf_confidence:.2f}")
                print(f"   24h變化: {price_change_24h:+.2f}%")
                print(f"   建議: {recommendation}")
                print(f"   風險等級: {risk_level}")
                
            except Exception as e:
                print(f"❌ {symbol} 分析失敗: {str(e)}")
                # 添加空值
                results['symbol'].append(symbol)
                results['current_price'].append(0)
                results['rf_prediction'].append(0)
                results['rf_signal'].append('ERROR')
                results['rf_confidence'].append(0)
                results['price_change_24h'].append(0)
                results['recommendation'].append('ERROR')
                results['risk_level'].append('HIGH')
        
        return pd.DataFrame(results)
    
    def generate_recommendation_and_risk(self, pred_results, confidence, price_change_24h):
        """生成投資建議和風險等級"""
        rf_signal = pred_results.get('rf_signal', 'HOLD')
        
        # 基於信號和置信度生成建議
        if rf_signal == 'BUY' and confidence > 0.7:
            recommendation = "強烈買入"
            risk_level = "MEDIUM"
        elif rf_signal == 'BUY' and confidence > 0.5:
            recommendation = "買入"
            risk_level = "MEDIUM"
        elif rf_signal == 'SELL' and confidence > 0.7:
            recommendation = "強烈賣出"
            risk_level = "HIGH"
        elif rf_signal == 'SELL' and confidence > 0.5:
            recommendation = "賣出"
            risk_level = "HIGH"
        else:
            recommendation = "持有"
            risk_level = "LOW"
        
        # 根據24小時價格變化調整風險等級
        if abs(price_change_24h) > 10:
            risk_level = "HIGH"
        elif abs(price_change_24h) > 5:
            risk_level = "MEDIUM"
        
        return recommendation, risk_level
    
    def generate_report(self, results_df):
        """生成分析報告"""
        print("\n" + "="*80)
        print("🌲 隨機森林機器學習分析報告")
        print("="*80)
        
        print(f"\n🔍 分析概況:")
        print(f"   分析符號數量: {len(results_df)}")
        print(f"   分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   預測時間範圍: {self.ml_config.prediction_horizon}小時")
        
        # 過濾掉錯誤的結果
        valid_results = results_df[results_df['rf_signal'] != 'ERROR']
        
        if len(valid_results) == 0:
            print("❌ 沒有有效的分析結果")
            return results_df
        
        # 按置信度排序
        sorted_df = valid_results.sort_values('rf_confidence', ascending=False)
        
        print(f"\n🏆 按置信度排名:")
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            print(f"   {i}. {row['symbol']}: {row['rf_confidence']:.3f} - {row['rf_signal']} ({row['recommendation']})")
        
        # 信號統計
        signal_counts = valid_results['rf_signal'].value_counts()
        print(f"\n🎯 信號統計:")
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} ({count/len(valid_results)*100:.1f}%)")
        
        # 建議統計
        rec_counts = valid_results['recommendation'].value_counts()
        print(f"\n💡 建議統計:")
        for rec, count in rec_counts.items():
            print(f"   {rec}: {count} ({count/len(valid_results)*100:.1f}%)")
        
        # 風險等級統計
        risk_counts = valid_results['risk_level'].value_counts()
        print(f"\n⚠️ 風險等級統計:")
        for risk, count in risk_counts.items():
            print(f"   {risk}: {count} ({count/len(valid_results)*100:.1f}%)")
        
        # 買入建議
        buy_signals = valid_results[valid_results['rf_signal'] == 'BUY'].sort_values('rf_confidence', ascending=False)
        if len(buy_signals) > 0:
            print(f"\n🔥 買入建議 (按置信度排序):")
            for i, (_, row) in enumerate(buy_signals.iterrows(), 1):
                print(f"   {i}. {row['symbol']}: ${row['current_price']:.4f} "
                      f"(置信度: {row['rf_confidence']:.3f}, 24h: {row['price_change_24h']:+.2f}%)")
        
        # 賣出警告
        sell_signals = valid_results[valid_results['rf_signal'] == 'SELL'].sort_values('rf_confidence', ascending=False)
        if len(sell_signals) > 0:
            print(f"\n❄️ 賣出警告 (按置信度排序):")
            for i, (_, row) in enumerate(sell_signals.iterrows(), 1):
                print(f"   {i}. {row['symbol']}: ${row['current_price']:.4f} "
                      f"(置信度: {row['rf_confidence']:.3f}, 24h: {row['price_change_24h']:+.2f}%)")
        
        # 市場概況
        avg_confidence = valid_results['rf_confidence'].mean()
        avg_price_change = valid_results['price_change_24h'].mean()
        
        print(f"\n📊 市場概況:")
        print(f"   平均置信度: {avg_confidence:.3f}")
        print(f"   平均24小時變化: {avg_price_change:+.2f}%")
        print(f"   高風險符號數量: {len(valid_results[valid_results['risk_level'] == 'HIGH'])}")
        
        print(f"\n⚠️ 免責聲明:")
        print(f"   • 本分析基於隨機森林機器學習模型")
        print(f"   • 預測結果僅供參考，不構成投資建議")
        print(f"   • 加密貨幣投資有風險，請謹慎決策")
        print(f"   • 建議設定止損點並分散投資")
        
        print("="*80)
        
        return sorted_df
    
    def save_results(self, results_df, filename=None):
        """保存結果到文件"""
        if filename is None:
            filename = f"rf_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        # 確保輸出目錄存在
        output_dir = "/Users/ting/技術分析/screener/output"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = f"{output_dir}/{filename}"
        results_df.to_csv(filepath, index=False)
        print(f"💾 結果已保存到: {filepath}")
        
        return filepath
    
    def run_analysis(self, timeframe='1h', training_days=90):
        """運行完整分析"""
        print("🚀 開始隨機森林機器學習分析")
        print(f"📊 基於07-03篩選結果優化的Top 10加密貨幣")
        print(f"🌲 使用隨機森林模型進行分類預測")
        print(f"⏰ 時間框架: {timeframe}, 訓練天數: {training_days}")
        
        # 執行分析
        results_df = self.analyze_with_rf_only(timeframe, training_days)
        
        # 生成報告
        sorted_results = self.generate_report(results_df)
        
        # 保存結果
        filepath = self.save_results(sorted_results)
        
        return sorted_results, filepath

def main():
    """主函數"""
    analyzer = SimplifiedMLAnalysis()
    results, filepath = analyzer.run_analysis(timeframe='1h', training_days=90)
    
    print(f"\n🎉 分析完成！")
    print(f"📄 詳細結果已保存到: {filepath}")
    print("🔍 可以打開CSV文件查看完整數據")

if __name__ == "__main__":
    main()