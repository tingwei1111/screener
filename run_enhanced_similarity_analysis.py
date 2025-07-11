#!/usr/bin/env python3
"""
運行增強版相似性分析
==============================
使用DTW算法和多維度相似性分析來識別加密貨幣的相似模式

主要功能:
1. 動態時間扭曲 (DTW) 算法
2. 多維度相似性分析 (價格+成交量+波動率)
3. 價量關係分析
4. 批量分析和可視化

使用方法:
python run_enhanced_similarity_analysis.py [options]

作者: Claude AI Assistant
版本: 1.0
日期: 2025-07-03
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_similarity_analyzer import EnhancedSimilarityAnalyzer, create_sample_reference_patterns

def load_strong_targets_from_file(file_path: str) -> List[str]:
    """從文件加載強勢目標符號"""
    symbols = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 提取符號名稱
                    if '\t' in line:
                        symbol = line.split('\t')[0]
                    elif ' ' in line:
                        symbol = line.split(' ')[0]
                    else:
                        symbol = line
                    
                    # 清理符號名稱
                    symbol = symbol.replace('✓', '').replace('❌', '').strip()
                    if symbol and symbol.endswith('USDT'):
                        symbols.append(symbol)
        
        print(f"✅ 從文件加載了 {len(symbols)} 個符號")
        return symbols
        
    except Exception as e:
        print(f"❌ 加載文件失敗: {e}")
        return []

def get_default_strong_targets() -> List[str]:
    """獲取默認的強勢目標符號"""
    return [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT',
        'LINKUSDT', 'MATICUSDT', 'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT',
        'FILUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT', 'CHZUSDT',
        'ENJUSDT', 'FLOWUSDT', 'ICPUSDT', 'VETUSDT', 'XLMUSDT'
    ]

def create_reference_patterns_from_recent_data() -> List[Dict]:
    """從最近的強勢數據創建參考模式"""
    patterns = [
        {
            'name': 'BTC_Recent_Bull_Pattern',
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'days': 7,
            'description': 'BTC 最近牛市模式'
        },
        {
            'name': 'ETH_Breakout_Pattern',
            'symbol': 'ETHUSDT',
            'timeframe': '1h',
            'days': 7,
            'description': 'ETH 突破模式'
        },
        {
            'name': 'SOL_Recovery_Pattern',
            'symbol': 'SOLUSDT',
            'timeframe': '1h',
            'days': 7,
            'description': 'SOL 復甦模式'
        },
        {
            'name': 'ADA_Uptrend_Pattern',
            'symbol': 'ADAUSDT',
            'timeframe': '1h',
            'days': 7,
            'description': 'ADA 上升趨勢模式'
        },
        {
            'name': 'DOT_Momentum_Pattern',
            'symbol': 'DOTUSDT',
            'timeframe': '1h',
            'days': 7,
            'description': 'DOT 動量模式'
        }
    ]
    
    return patterns

def run_single_analysis(analyzer: EnhancedSimilarityAnalyzer, symbol: str, 
                       reference_patterns: List[Dict], timeframe: str, days: int):
    """運行單個符號的分析"""
    print(f"\n{'='*60}")
    print(f"🎯 單個符號分析: {symbol}")
    print(f"{'='*60}")
    
    # 執行分析
    result = analyzer.analyze_pattern_similarity(symbol, reference_patterns, timeframe, days)
    
    if 'error' in result:
        print(f"❌ 分析失敗: {result['error']}")
        return None
    
    # 創建可視化
    output_dir = "enhanced_similarity_output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_path = os.path.join(output_dir, f"{symbol}_similarity_analysis_{timestamp}.png")
    
    analyzer.create_similarity_visualization(result, viz_path)
    
    return result

def run_batch_analysis(analyzer: EnhancedSimilarityAnalyzer, symbols: List[str], 
                      reference_patterns: List[Dict], timeframe: str, days: int):
    """運行批量分析"""
    print(f"\n{'='*60}")
    print(f"🚀 批量相似性分析")
    print(f"{'='*60}")
    
    # 執行批量分析
    batch_results = analyzer.batch_analyze_symbols(symbols, reference_patterns, timeframe, days)
    
    # 保存結果
    output_dir = "enhanced_similarity_output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f"batch_similarity_analysis_{timestamp}.txt")
    
    # 保存詳細結果到文件
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("增強版相似性分析結果\n")
        f.write("=" * 50 + "\n")
        f.write(f"分析時間: {batch_results['analysis_date']}\n")
        f.write(f"時間框架: {batch_results['timeframe']}\n")
        f.write(f"分析天數: {batch_results['days']}\n")
        f.write(f"總符號數: {batch_results['total_symbols']}\n")
        f.write(f"參考模式數: {batch_results['reference_patterns']}\n")
        f.write(f"成功分析: {batch_results['summary']['successful_analyses']}\n")
        f.write(f"失敗分析: {batch_results['summary']['failed_analyses']}\n")
        f.write(f"成功率: {batch_results['summary']['success_rate']:.1%}\n\n")
        
        # 詳細結果
        for result in batch_results['symbols_analyzed']:
            f.write(f"\n符號: {result['symbol']}\n")
            f.write("-" * 30 + "\n")
            
            if result['similarities']:
                f.write("相似性排名:\n")
                for i, sim in enumerate(result['similarities'][:5], 1):
                    f.write(f"  {i}. {sim['pattern_symbol']} - {sim['composite_similarity']:.3f} ({sim['similarity_level']})\n")
                    f.write(f"     價格: {sim['price_similarity']:.3f} | 成交量: {sim['volume_similarity']:.3f}\n")
                    f.write(f"     波動率: {sim['volatility_similarity']:.3f} | 價量關係: {sim['price_volume_similarity']:.3f}\n")
            else:
                f.write("  沒有找到相似模式\n")
    
    print(f"📄 詳細結果已保存到: {results_file}")
    
    return batch_results

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='增強版相似性分析工具')
    parser.add_argument('-s', '--symbol', type=str, help='單個符號分析')
    parser.add_argument('-f', '--file', type=str, help='從文件加載符號列表')
    parser.add_argument('-t', '--timeframe', type=str, default='1h', 
                       choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'],
                       help='時間框架 (默認: 1h)')
    parser.add_argument('-d', '--days', type=int, default=7, help='分析天數 (默認: 7)')
    parser.add_argument('--pattern-days', type=int, default=7, help='參考模式天數 (默認: 7)')
    parser.add_argument('-o', '--output', type=str, default='enhanced_similarity_output', 
                       help='輸出目錄 (默認: enhanced_similarity_output)')
    parser.add_argument('--demo', action='store_true', help='運行演示模式')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output, exist_ok=True)
    
    print("🚀 增強版相似性分析工具")
    print("=" * 50)
    print(f"時間框架: {args.timeframe}")
    print(f"分析天數: {args.days}")
    print(f"輸出目錄: {args.output}")
    
    # 創建分析器
    analyzer = EnhancedSimilarityAnalyzer()
    
    # 創建參考模式
    reference_patterns = create_reference_patterns_from_recent_data()
    
    # 更新參考模式的天數
    for pattern in reference_patterns:
        pattern['days'] = args.pattern_days
    
    print(f"\n📊 參考模式: {len(reference_patterns)} 個")
    for pattern in reference_patterns:
        print(f"   - {pattern['name']}: {pattern['symbol']} ({pattern['days']}天)")
    
    if args.demo:
        # 演示模式
        print("\n🎭 演示模式")
        demo_symbols = ['ADAUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT', 'ATOMUSDT']
        run_batch_analysis(analyzer, demo_symbols, reference_patterns, args.timeframe, args.days)
        
    elif args.symbol:
        # 單個符號分析
        run_single_analysis(analyzer, args.symbol, reference_patterns, args.timeframe, args.days)
        
    elif args.file:
        # 從文件加載符號
        symbols = load_strong_targets_from_file(args.file)
        if symbols:
            run_batch_analysis(analyzer, symbols, reference_patterns, args.timeframe, args.days)
        else:
            print("❌ 沒有從文件中加載到有效符號")
            
    else:
        # 使用默認符號列表
        print("\n📋 使用默認符號列表")
        symbols = get_default_strong_targets()
        run_batch_analysis(analyzer, symbols, reference_patterns, args.timeframe, args.days)
    
    print(f"\n✅ 分析完成！結果已保存到 {args.output} 目錄")

if __name__ == "__main__":
    main() 