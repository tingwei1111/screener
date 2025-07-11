#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前50強勢幣種機器學習分析
基於2025-07-11 RS分數排名的Top 50加密貨幣
"""

import sys
import os
sys.path.append('.')
sys.path.append('./src')

from src.downloader import CryptoDownloader
from ml_predictor import MLPredictor
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_indicators(data):
    """計算技術指標"""
    # 移動平均線
    data['ma_5'] = data['close'].rolling(window=5).mean()
    data['ma_10'] = data['close'].rolling(window=10).mean()
    data['ma_20'] = data['close'].rolling(window=20).mean()
    data['ma_50'] = data['close'].rolling(window=50).mean()
    
    # 布林帶
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_width'] = data['bb_upper'] - data['bb_lower']
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = data['close'].ewm(span=12).mean()
    ema_26 = data['close'].ewm(span=26).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    # ATR
    data['high_low'] = data['high'] - data['low']
    data['high_close'] = np.abs(data['high'] - data['close'].shift())
    data['low_close'] = np.abs(data['low'] - data['close'].shift())
    data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
    data['atr'] = data['tr'].rolling(window=14).mean()
    
    # 波動率
    data['volatility'] = data['close'].pct_change().rolling(window=20).std()
    
    return data

def analyze_top_50_symbols():
    """分析前50個強勢幣種"""
    print('🚀 開始前50強勢幣種機器學習分析')
    print('📊 基於2025-07-11 RS分數排名的Top 50加密貨幣')
    print('🌲 使用隨機森林模型進行預測')
    print('⏰ 時間框架: 1h, 訓練天數: 90')
    print('='*70)
    
    # 讀取前50個符號
    try:
        with open('top_50_symbols.txt', 'r') as f:
            symbols = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print('❌ 找不到 top_50_symbols.txt 文件')
        return
    
    results = []
    successful_analyses = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f'\n📈 [{i}/50] 分析 {symbol}')
        print('-' * 50)
        
        try:
            # 下載數據
            downloader = CryptoDownloader()
            success, data = downloader.get_data(symbol, timeframe='1h', dropna=False)
            
            if not success or len(data) < 100:
                print(f'❌ {symbol}: 數據不足 ({len(data) if success else 0} 行)')
                continue
            
            print(f'✅ 獲取到 {len(data)} 個數據點')
            
            # 計算技術指標
            print('🔧 計算技術指標...')
            data = calculate_technical_indicators(data)
            
            # 訓練ML模型
            print(f'🌲 訓練 {symbol} 的隨機森林模型...')
            predictor = MLPredictor()
            
            # 準備特徵
            feature_columns = [
                'ma_5', 'ma_10', 'ma_20', 'ma_50',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'atr', 'volatility'
            ]
            
            # 檢查特徵列是否存在
            available_features = [col for col in feature_columns if col in data.columns]
            if len(available_features) < 5:
                print(f'❌ {symbol}: 技術指標不足')
                continue
            
            X = data[available_features].dropna()
            if len(X) < 50:
                print(f'❌ {symbol}: 有效數據不足')
                continue
            
            # 創建標籤
            returns = data['close'].pct_change().shift(-1)
            labels = np.where(returns > 0.02, 1, np.where(returns < -0.02, -1, 0))
            
            # 對齊數據
            min_len = min(len(X), len(labels))
            X = X.iloc[:min_len]
            y = labels[:min_len]
            
            # 訓練模型
            model, train_acc, test_acc, class_dist = predictor.train_model(X, y)
            
            # 預測
            prediction = predictor.predict(X.iloc[-1:])
            confidence = predictor.predict_proba(X.iloc[-1:])
            
            # 獲取當前價格
            current_price = data['close'].iloc[-1]
            # 計算24小時變化 (1小時 x 24 = 24小時前)
            if len(data) > 24:
                price_24h_ago = data['close'].iloc[-25]  # 24小時前
                price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
            else:
                price_change_24h = 0
            
            # 解釋信號
            signal_map = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}
            signal = signal_map[prediction[0]]
            
            # 計算置信度
            if len(confidence[0]) > 1:
                max_confidence = max(confidence[0])
            else:
                max_confidence = 0.5
            
            # 投資建議
            if signal == 'BUY' and max_confidence > 0.7:
                recommendation = '強烈買入'
            elif signal == 'BUY' and max_confidence > 0.5:
                recommendation = '買入'
            elif signal == 'HOLD':
                recommendation = '持有'
            elif signal == 'SELL' and max_confidence > 0.7:
                recommendation = '強烈賣出'
            elif signal == 'SELL' and max_confidence > 0.5:
                recommendation = '賣出'
            else:
                recommendation = '觀望'
            
            # 獲取重要特徵
            feature_importance = model.feature_importances_
            top_features = [available_features[i] for i in np.argsort(feature_importance)[-3:]]
            
            print(f'✅ 訓練完成 - 訓練準確率: {train_acc:.3f}, 測試準確率: {test_acc:.3f}')
            print(f'   類別分佈: {class_dist}')
            print(f'   前3個重要特徵: {top_features}')
            print(f'✅ {symbol} 分析完成')
            print(f'   當前價格: ${current_price:.4f}')
            print(f'   信號: {signal}')
            print(f'   置信度: {max_confidence:.3f}')
            print(f'   建議: {recommendation}')
            print(f'   24h變化: {price_change_24h:+.2f}%')
            print(f'   重要特徵: {", ".join(top_features)}')
            
            results.append({
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal,
                'confidence': max_confidence,
                'recommendation': recommendation,
                'price_change_24h': price_change_24h,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'top_features': ', '.join(top_features)
            })
            
            successful_analyses += 1
            
        except Exception as e:
            print(f'❌ {symbol}: 分析失敗 - {str(e)}')
            continue
    
    # 生成報告
    print('\n' + '='*70)
    print('📊 前50強勢幣種機器學習分析報告')
    print('='*70)
    
    if results:
        df = pd.DataFrame(results)
        
        # 按置信度排序
        df = df.sort_values('confidence', ascending=False)
        
        print(f'\n🔍 分析概況:')
        print(f'   成功分析: {successful_analyses}/50 個符號')
        print(f'   成功率: {successful_analyses/50*100:.1f}%')
        print(f'   分析時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # 按置信度排名
        print(f'\n🏆 按置信度排名:')
        for i, (_, row) in enumerate(df.head(15).iterrows(), 1):
            print(f'   {i}. {row["symbol"]}: {row["confidence"]:.3f} - {row["signal"]} ({row["recommendation"]})')
        
        # 信號統計
        signal_counts = df['signal'].value_counts()
        print(f'\n🎯 信號統計:')
        for signal, count in signal_counts.items():
            print(f'   {signal}: {count} ({count/len(df)*100:.1f}%)')
        
        # 買入建議
        buy_signals = df[df['signal'] == 'BUY'].sort_values('confidence', ascending=False)
        if len(buy_signals) > 0:
            print(f'\n🔥 買入建議 (按置信度排序):')
            for i, (_, row) in enumerate(buy_signals.head(15).iterrows(), 1):
                print(f'   {i}. {row["symbol"]}: ${row["current_price"]:.4f} (置信度: {row["confidence"]:.3f}, 24h: {row["price_change_24h"]:+.2f}%)')
        
        # 模型性能統計
        print(f'\n📊 模型性能統計:')
        print(f'   平均訓練準確率: {df["train_accuracy"].mean():.3f}')
        print(f'   平均測試準確率: {df["test_accuracy"].mean():.3f}')
        print(f'   平均預測置信度: {df["confidence"].mean():.3f}')
        
        # 保存結果
        output_file = f'output/top_50_ml_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        df.to_csv(output_file, index=False)
        print(f'\n💾 結果已保存到: {output_file}')
        
    else:
        print('❌ 沒有成功分析的符號')

if __name__ == '__main__':
    analyze_top_50_symbols() 