#!/usr/bin/env python3
"""
Enhanced ML Analysis for 07-03 Screening Results
增強型機器學習分析 - 基於07-03篩選結果
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ml_predictor import MLPredictor, MLConfig

class EnhancedMLAnalysis:
    """增強型ML分析類"""
    
    def __init__(self):
        # 從07-03篩選結果中提取top 10加密貨幣
        self.top_symbols = [
            'BTCUSDT',  # 從BINANCE:BTCUSDT.P轉換
            'ETHUSDT',  # 從BINANCE:ETHUSDT.P轉換
            'BUSDT',    # 從BINANCE:BUSDT.P轉換
            'HFTUSDT',  # 從BINANCE:HFTUSDT.P轉換
            'GUNUSDT',  # 從BINANCE:GUNUSDT.P轉換
            'HUSDT',    # 從BINANCE:HUSDT.P轉換
            '1000BONKUSDT',  # 從BINANCE:1000BONKUSDT.P轉換
            'FIDAUSDT', # 從BINANCE:FIDAUSDT.P轉換
            'DEEPUSDT', # 從BINANCE:DEEPUSDT.P轉換
            'CHESSUSDT' # 從BINANCE:CHESSUSDT.P轉換
        ]
        
        # 配置ML參數
        self.ml_config = MLConfig(
            lstm_units=64,
            lstm_dropout=0.2,
            lstm_epochs=50,
            lstm_batch_size=32,
            rf_n_estimators=200,
            rf_max_depth=15,
            prediction_horizon=24,  # 預測24小時後
            classification_threshold=0.03  # 3%價格變化閾值
        )
        
        self.ml_predictor = MLPredictor(self.ml_config)
        
    def analyze_top_symbols(self, timeframe='1h', training_days=180):
        """分析top符號的ML預測"""
        print("🤖 開始增強型機器學習分析")
        print(f"📊 分析符號: {', '.join(self.top_symbols)}")
        print(f"⏰ 時間框架: {timeframe}")
        print(f"📅 訓練天數: {training_days}")
        print("="*70)
        
        results = {
            'symbol': [],
            'current_price': [],
            'lstm_prediction': [],
            'lstm_signal': [],
            'rf_prediction': [],
            'rf_signal': [],
            'rf_confidence': [],
            'combined_score': [],
            'recommendation': []
        }
        
        for i, symbol in enumerate(self.top_symbols, 1):
            print(f"\n📈 [{i}/{len(self.top_symbols)}] 分析 {symbol}")
            print("-" * 50)
            
            try:
                # 訓練模型
                print(f"🔧 為 {symbol} 訓練ML模型...")
                train_results = self.ml_predictor.train_models(
                    symbol=symbol, 
                    timeframe=timeframe, 
                    days=training_days
                )
                
                # 生成預測
                print(f"🎯 為 {symbol} 生成預測...")
                pred_results = self.ml_predictor.predict_symbol(
                    symbol=symbol, 
                    timeframe=timeframe
                )
                
                # 提取結果
                results['symbol'].append(symbol)
                results['current_price'].append(pred_results.get('current_price', 0))
                results['lstm_prediction'].append(pred_results.get('lstm_prediction', 0))
                results['lstm_signal'].append(pred_results.get('lstm_signal', 'N/A'))
                results['rf_prediction'].append(pred_results.get('rf_prediction', 0))
                results['rf_signal'].append(pred_results.get('rf_signal', 'N/A'))
                
                # 計算隨機森林置信度
                rf_prob = pred_results.get('rf_probability', [0.33, 0.33, 0.34])
                rf_confidence = max(rf_prob) if rf_prob is not None else 0.5
                results['rf_confidence'].append(rf_confidence)
                
                # 計算綜合評分
                combined_score = self.calculate_combined_score(pred_results)
                results['combined_score'].append(combined_score)
                
                # 生成建議
                recommendation = self.generate_recommendation(pred_results, combined_score)
                results['recommendation'].append(recommendation)
                
                print(f"✅ {symbol} 分析完成")
                print(f"   LSTM信號: {pred_results.get('lstm_signal', 'N/A')}")
                print(f"   RF信號: {pred_results.get('rf_signal', 'N/A')}")
                print(f"   綜合評分: {combined_score:.2f}")
                print(f"   建議: {recommendation}")
                
            except Exception as e:
                print(f"❌ {symbol} 分析失敗: {str(e)}")
                # 添加空值
                results['symbol'].append(symbol)
                results['current_price'].append(0)
                results['lstm_prediction'].append(0)
                results['lstm_signal'].append('ERROR')
                results['rf_prediction'].append(0)
                results['rf_signal'].append('ERROR')
                results['rf_confidence'].append(0)
                results['combined_score'].append(0)
                results['recommendation'].append('ERROR')
        
        return pd.DataFrame(results)
    
    def calculate_combined_score(self, pred_results):
        """計算綜合評分"""
        score = 0
        
        # LSTM信號評分
        lstm_signal = pred_results.get('lstm_signal', 'HOLD')
        if lstm_signal == 'BUY':
            score += 0.4
        elif lstm_signal == 'SELL':
            score -= 0.4
        
        # RF信號評分
        rf_signal = pred_results.get('rf_signal', 'HOLD')
        if rf_signal == 'BUY':
            score += 0.3
        elif rf_signal == 'SELL':
            score -= 0.3
        
        # 置信度評分
        rf_prob = pred_results.get('rf_probability', [0.33, 0.33, 0.34])
        if rf_prob is not None:
            confidence = max(rf_prob)
            score += (confidence - 0.5) * 0.3
        
        # 標準化到-1到1之間
        return max(-1, min(1, score))
    
    def generate_recommendation(self, pred_results, combined_score):
        """生成投資建議"""
        if combined_score > 0.3:
            return "強烈買入"
        elif combined_score > 0.1:
            return "買入"
        elif combined_score > -0.1:
            return "持有"
        elif combined_score > -0.3:
            return "賣出"
        else:
            return "強烈賣出"
    
    def generate_detailed_report(self, results_df):
        """生成詳細報告"""
        print("\n" + "="*80)
        print("📊 增強型機器學習分析報告")
        print("="*80)
        
        print(f"\n🔍 分析概況:")
        print(f"   分析符號數量: {len(results_df)}")
        print(f"   分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   預測時間範圍: {self.ml_config.prediction_horizon}小時")
        
        # 按綜合評分排序
        sorted_df = results_df.sort_values('combined_score', ascending=False)
        
        print(f"\n🏆 綜合評分排名 (Top 5):")
        for i, (_, row) in enumerate(sorted_df.head(5).iterrows(), 1):
            print(f"   {i}. {row['symbol']}: {row['combined_score']:.3f} - {row['recommendation']}")
        
        # 信號統計
        lstm_signals = results_df['lstm_signal'].value_counts()
        rf_signals = results_df['rf_signal'].value_counts()
        
        print(f"\n🎯 LSTM信號統計:")
        for signal, count in lstm_signals.items():
            print(f"   {signal}: {count} ({count/len(results_df)*100:.1f}%)")
        
        print(f"\n🎯 隨機森林信號統計:")
        for signal, count in rf_signals.items():
            print(f"   {signal}: {count} ({count/len(results_df)*100:.1f}%)")
        
        # 一致性分析
        consistent_signals = results_df[
            (results_df['lstm_signal'] == results_df['rf_signal']) & 
            (results_df['lstm_signal'] != 'HOLD')
        ]
        
        print(f"\n🤝 模型一致性:")
        print(f"   一致信號數量: {len(consistent_signals)}")
        print(f"   一致性比例: {len(consistent_signals)/len(results_df)*100:.1f}%")
        
        if len(consistent_signals) > 0:
            print(f"\n🎯 一致性信號:")
            for _, row in consistent_signals.iterrows():
                print(f"   {row['symbol']}: {row['lstm_signal']} (評分: {row['combined_score']:.3f})")
        
        # 投資建議
        print(f"\n💡 投資建議:")
        
        strong_buy = sorted_df[sorted_df['recommendation'] == '強烈買入']
        if len(strong_buy) > 0:
            print(f"   🔥 強烈買入推薦:")
            for _, row in strong_buy.iterrows():
                print(f"      {row['symbol']}: ${row['current_price']:.4f} (評分: {row['combined_score']:.3f})")
        
        buy = sorted_df[sorted_df['recommendation'] == '買入']
        if len(buy) > 0:
            print(f"   📈 買入推薦:")
            for _, row in buy.iterrows():
                print(f"      {row['symbol']}: ${row['current_price']:.4f} (評分: {row['combined_score']:.3f})")
        
        strong_sell = sorted_df[sorted_df['recommendation'] == '強烈賣出']
        if len(strong_sell) > 0:
            print(f"   ❄️ 強烈賣出警告:")
            for _, row in strong_sell.iterrows():
                print(f"      {row['symbol']}: ${row['current_price']:.4f} (評分: {row['combined_score']:.3f})")
        
        print(f"\n⚠️  風險提示:")
        print(f"   • 機器學習預測僅供參考，不構成投資建議")
        print(f"   • 加密貨幣市場波動劇烈，請謹慎投資")
        print(f"   • 建議結合其他技術分析和基本面分析")
        print(f"   • 請設定適當的止損和止盈點")
        
        print("="*80)
        
        return sorted_df
    
    def save_results(self, results_df, filename=None):
        """保存結果到文件"""
        if filename is None:
            filename = f"ml_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        filepath = f"/Users/ting/技術分析/screener/output/{filename}"
        results_df.to_csv(filepath, index=False)
        print(f"💾 結果已保存到: {filepath}")
        
        return filepath
    
    def run_complete_analysis(self, timeframe='1h', training_days=180):
        """運行完整分析"""
        print("🚀 開始完整的增強型ML分析")
        print(f"📊 基於07-03篩選結果的Top 10加密貨幣")
        print(f"🤖 使用LSTM + 隨機森林雙模型預測")
        print(f"⏰ 時間框架: {timeframe}, 訓練天數: {training_days}")
        
        # 執行分析
        results_df = self.analyze_top_symbols(timeframe, training_days)
        
        # 生成詳細報告
        sorted_results = self.generate_detailed_report(results_df)
        
        # 保存結果
        self.save_results(sorted_results)
        
        return sorted_results

def main():
    """主函數"""
    analyzer = EnhancedMLAnalysis()
    results = analyzer.run_complete_analysis(timeframe='1h', training_days=180)
    
    print("\n🎉 分析完成！")
    print("請查看生成的CSV文件以獲取完整結果")

if __name__ == "__main__":
    main()