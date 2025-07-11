"""
Enhanced Similarity Analyzer
=====================================
高級相似性分析器 - 實現DTW算法和多維度相似性分析

主要功能:
1. Dynamic Time Warping (DTW) 算法 - 處理時間軸拉伸/壓縮的相似模式
2. 多維度相似性分析 - 價格+成交量+波動率的複合模式匹配
3. 價量關係分析 - 識別"價跌量縮、價漲量增"等經典模式
4. 進階相似度算法 - 超越傳統歐式距離和相關係數

作者: Claude AI Assistant
版本: 1.0
日期: 2025-07-03
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr
from dtaidistance import dtw
from dtaidistance.dtw import distance as dtw_distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# 導入現有模塊
from src.downloader import CryptoDownloader
from src.common import DataNormalizer, format_dt_with_tz, TrendAnalysisConfig

class EnhancedSimilarityAnalyzer:
    """增強版相似性分析器"""
    
    def __init__(self, config: TrendAnalysisConfig = None):
        """初始化分析器"""
        self.config = config or TrendAnalysisConfig()
        self.downloader = CryptoDownloader()
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        
        # DTW參數
        self.dtw_window_ratio = 0.15  # DTW窗口比例
        self.dtw_step_pattern = "symmetric2"  # DTW步進模式
        self.dtw_distance_only = False  # 是否只計算距離
        
        # 相似性閾值
        self.similarity_thresholds = {
            'high': 0.8,      # 高相似性
            'medium': 0.6,    # 中等相似性
            'low': 0.4        # 低相似性
        }
        
        print("🚀 增強版相似性分析器已初始化")
        print(f"   DTW窗口比例: {self.dtw_window_ratio}")
        print(f"   相似性閾值: {self.similarity_thresholds}")

    def calculate_dtw_similarity(self, series1: np.ndarray, series2: np.ndarray, 
                                window_ratio: float = None) -> Dict[str, float]:
        """
        計算DTW相似性
        
        Args:
            series1: 第一個時間序列
            series2: 第二個時間序列
            window_ratio: DTW窗口比例
            
        Returns:
            包含DTW距離和相似性分數的字典
        """
        if window_ratio is None:
            window_ratio = self.dtw_window_ratio
            
        try:
            # 計算DTW窗口大小
            window_size = int(max(len(series1), len(series2)) * window_ratio)
            
            # 計算DTW距離
            dtw_dist = dtw_distance(series1, series2, window=window_size)
            
            # 轉換為相似性分數 (0-1，1為最相似)
            similarity = 1 / (1 + dtw_dist)
            
            return {
                'dtw_distance': dtw_dist,
                'dtw_similarity': similarity,
                'window_size': window_size
            }
            
        except Exception as e:
            print(f"DTW計算錯誤: {e}")
            return {
                'dtw_distance': float('inf'),
                'dtw_similarity': 0.0,
                'window_size': 0
            }

    def calculate_multidimensional_similarity(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, float]:
        """
        計算多維度相似性 (價格+成交量+波動率)
        
        Args:
            data1: 第一個數據集 (包含OHLCV)
            data2: 第二個數據集 (包含OHLCV)
            
        Returns:
            多維度相似性分析結果
        """
        results = {}
        
        # 確保數據長度一致
        min_len = min(len(data1), len(data2))
        if min_len < 10:
            return {'error': 'Data too short for analysis'}
        
        df1 = data1.tail(min_len).copy()
        df2 = data2.tail(min_len).copy()
        
        # 1. 價格相似性 (使用DTW)
        price1 = df1['Close'].values
        price2 = df2['Close'].values
        
        # 標準化價格
        price1_norm = self.normalize_series(price1)
        price2_norm = self.normalize_series(price2)
        
        price_similarity = self.calculate_dtw_similarity(price1_norm, price2_norm)
        results['price_dtw_similarity'] = price_similarity['dtw_similarity']
        results['price_dtw_distance'] = price_similarity['dtw_distance']
        
        # 2. 成交量相似性
        if 'Volume' in df1.columns and 'Volume' in df2.columns:
            volume1 = df1['Volume'].values
            volume2 = df2['Volume'].values
            
            # 標準化成交量
            volume1_norm = self.normalize_series(volume1)
            volume2_norm = self.normalize_series(volume2)
            
            volume_similarity = self.calculate_dtw_similarity(volume1_norm, volume2_norm)
            results['volume_dtw_similarity'] = volume_similarity['dtw_similarity']
            results['volume_dtw_distance'] = volume_similarity['dtw_distance']
        else:
            results['volume_dtw_similarity'] = 0.0
            results['volume_dtw_distance'] = float('inf')
        
        # 3. 波動率相似性
        volatility1 = self.calculate_volatility(df1)
        volatility2 = self.calculate_volatility(df2)
        
        volatility_similarity = self.calculate_dtw_similarity(volatility1, volatility2)
        results['volatility_dtw_similarity'] = volatility_similarity['dtw_similarity']
        results['volatility_dtw_distance'] = volatility_similarity['dtw_distance']
        
        # 4. 價量關係相似性
        price_volume_relationship1 = self.calculate_price_volume_relationship(df1)
        price_volume_relationship2 = self.calculate_price_volume_relationship(df2)
        
        if price_volume_relationship1 is not None and price_volume_relationship2 is not None:
            pv_similarity = self.calculate_dtw_similarity(price_volume_relationship1, price_volume_relationship2)
            results['price_volume_dtw_similarity'] = pv_similarity['dtw_similarity']
            results['price_volume_dtw_distance'] = pv_similarity['dtw_distance']
        else:
            results['price_volume_dtw_similarity'] = 0.0
            results['price_volume_dtw_distance'] = float('inf')
        
        # 5. 綜合相似性分數 (加權平均)
        weights = {
            'price': 0.4,
            'volume': 0.2,
            'volatility': 0.2,
            'price_volume': 0.2
        }
        
        composite_similarity = (
            results['price_dtw_similarity'] * weights['price'] +
            results['volume_dtw_similarity'] * weights['volume'] +
            results['volatility_dtw_similarity'] * weights['volatility'] +
            results['price_volume_dtw_similarity'] * weights['price_volume']
        )
        
        results['composite_similarity'] = composite_similarity
        results['similarity_level'] = self.classify_similarity_level(composite_similarity)
        
        return results

    def normalize_series(self, series: np.ndarray) -> np.ndarray:
        """標準化時間序列"""
        if len(series) == 0:
            return series
        
        # 使用Z-score標準化
        mean_val = np.mean(series)
        std_val = np.std(series)
        
        if std_val == 0:
            return np.zeros_like(series)
        
        return (series - mean_val) / std_val

    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> np.ndarray:
        """計算滾動波動率"""
        if len(df) < window:
            window = len(df)
        
        returns = df['Close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std().fillna(0)
        
        return volatility.values

    def calculate_price_volume_relationship(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        計算價量關係指標
        
        Returns:
            價量關係指標數組，正值表示價漲量增，負值表示價跌量增
        """
        if 'Volume' not in df.columns or len(df) < 2:
            return None
        
        # 計算價格變化
        price_change = df['Close'].pct_change().fillna(0)
        
        # 計算成交量變化
        volume_change = df['Volume'].pct_change().fillna(0)
        
        # 價量關係 = 價格變化 * 成交量變化
        # 正值: 價漲量增或價跌量減 (健康)
        # 負值: 價漲量減或價跌量增 (不健康)
        price_volume_relationship = price_change * volume_change
        
        return price_volume_relationship.values

    def classify_similarity_level(self, similarity: float) -> str:
        """分類相似性等級"""
        if similarity >= self.similarity_thresholds['high']:
            return 'HIGH'
        elif similarity >= self.similarity_thresholds['medium']:
            return 'MEDIUM'
        elif similarity >= self.similarity_thresholds['low']:
            return 'LOW'
        else:
            return 'VERY_LOW'

    def analyze_pattern_similarity(self, current_symbol: str, reference_patterns: List[Dict], 
                                 timeframe: str = '1h', days: int = 30) -> Dict:
        """
        分析當前符號與參考模式的相似性
        
        Args:
            current_symbol: 當前分析的符號
            reference_patterns: 參考模式列表
            timeframe: 時間框架
            days: 分析天數
            
        Returns:
            相似性分析結果
        """
        print(f"\n🔍 分析 {current_symbol} 的模式相似性...")
        
        # 獲取當前數據
        current_data = self.get_symbol_data(current_symbol, timeframe, days)
        if current_data is None:
            return {'error': f'無法獲取 {current_symbol} 的數據'}
        
        results = {
            'symbol': current_symbol,
            'timeframe': timeframe,
            'analysis_date': datetime.now().isoformat(),
            'current_data_points': len(current_data),
            'similarities': []
        }
        
        # 與每個參考模式比較
        for i, pattern in enumerate(reference_patterns):
            print(f"   比較模式 {i+1}/{len(reference_patterns)}: {pattern.get('name', 'Unknown')}")
            
            # 獲取參考數據
            ref_data = self.get_reference_data(pattern)
            if ref_data is None:
                continue
            
            # 計算多維度相似性
            similarity_result = self.calculate_multidimensional_similarity(current_data, ref_data)
            
            if 'error' not in similarity_result:
                similarity_info = {
                    'pattern_name': pattern.get('name', 'Unknown'),
                    'pattern_symbol': pattern.get('symbol', 'Unknown'),
                    'pattern_timeframe': pattern.get('timeframe', 'Unknown'),
                    'composite_similarity': similarity_result['composite_similarity'],
                    'similarity_level': similarity_result['similarity_level'],
                    'price_similarity': similarity_result['price_dtw_similarity'],
                    'volume_similarity': similarity_result['volume_dtw_similarity'],
                    'volatility_similarity': similarity_result['volatility_dtw_similarity'],
                    'price_volume_similarity': similarity_result['price_volume_dtw_similarity']
                }
                
                results['similarities'].append(similarity_info)
                
                print(f"      綜合相似性: {similarity_result['composite_similarity']:.3f} ({similarity_result['similarity_level']})")
        
        # 按相似性排序
        results['similarities'].sort(key=lambda x: x['composite_similarity'], reverse=True)
        
        # 生成分析報告
        self.generate_similarity_report(results)
        
        return results

    def get_symbol_data(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        """獲取符號數據"""
        try:
            end_ts = int(datetime.now().timestamp())
            start_ts = end_ts - (days * 24 * 3600)
            
            success, df = self.downloader.get_data(symbol, start_ts, end_ts, timeframe=timeframe)
            
            if not success or df.empty:
                print(f"❌ 無法獲取 {symbol} 的數據")
                return None
            
            # 標準化列名 - 處理大小寫差異
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # 應用列名映射
            df = df.rename(columns=column_mapping)
            
            # 確保包含必要的列
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_columns):
                print(f"❌ {symbol} 數據缺少必要的OHLC列")
                print(f"   可用列: {list(df.columns)}")
                return None
            
            return df
            
        except Exception as e:
            print(f"❌ 獲取 {symbol} 數據時發生錯誤: {e}")
            return None

    def get_reference_data(self, pattern: Dict) -> Optional[pd.DataFrame]:
        """獲取參考模式數據"""
        try:
            if 'data' in pattern:
                return pattern['data']
            elif 'symbol' in pattern:
                return self.get_symbol_data(
                    pattern['symbol'], 
                    pattern.get('timeframe', '1h'), 
                    pattern.get('days', 30)
                )
            else:
                print("❌ 參考模式缺少必要的數據信息")
                return None
                
        except Exception as e:
            print(f"❌ 獲取參考數據時發生錯誤: {e}")
            return None

    def generate_similarity_report(self, results: Dict):
        """生成相似性分析報告"""
        print(f"\n📊 {results['symbol']} 相似性分析報告")
        print("=" * 60)
        
        if not results['similarities']:
            print("❌ 沒有找到相似的模式")
            return
        
        print(f"✅ 分析完成，找到 {len(results['similarities'])} 個相似模式")
        print(f"📈 數據點數: {results['current_data_points']}")
        print(f"⏰ 分析時間: {results['analysis_date']}")
        
        print(f"\n🏆 相似性排名 (Top 10):")
        for i, sim in enumerate(results['similarities'][:10], 1):
            level_emoji = {
                'HIGH': '🔥',
                'MEDIUM': '⚡',
                'LOW': '💡',
                'VERY_LOW': '❄️'
            }.get(sim['similarity_level'], '❓')
            
            print(f"   {i:2d}. {sim['pattern_symbol']:12s} {level_emoji} {sim['composite_similarity']:.3f}")
            print(f"       價格: {sim['price_similarity']:.3f} | "
                  f"成交量: {sim['volume_similarity']:.3f} | "
                  f"波動率: {sim['volatility_similarity']:.3f} | "
                  f"價量關係: {sim['price_volume_similarity']:.3f}")
        
        # 統計分析
        similarities = [s['composite_similarity'] for s in results['similarities']]
        high_count = len([s for s in similarities if s >= self.similarity_thresholds['high']])
        medium_count = len([s for s in similarities if self.similarity_thresholds['medium'] <= s < self.similarity_thresholds['high']])
        low_count = len([s for s in similarities if self.similarity_thresholds['low'] <= s < self.similarity_thresholds['medium']])
        
        print(f"\n📈 相似性統計:")
        print(f"   高相似性 (≥{self.similarity_thresholds['high']}): {high_count} 個")
        print(f"   中等相似性 ({self.similarity_thresholds['medium']}-{self.similarity_thresholds['high']}): {medium_count} 個")
        print(f"   低相似性 ({self.similarity_thresholds['low']}-{self.similarity_thresholds['medium']}): {low_count} 個")
        print(f"   平均相似性: {np.mean(similarities):.3f}")
        
        # 投資建議
        self.generate_investment_advice(results)

    def generate_investment_advice(self, results: Dict):
        """生成投資建議"""
        print(f"\n💡 基於相似性分析的投資建議:")
        
        if not results['similarities']:
            print("   ❓ 沒有足夠的相似模式進行建議")
            return
        
        best_similarity = results['similarities'][0]
        avg_similarity = np.mean([s['composite_similarity'] for s in results['similarities'][:5]])
        
        if best_similarity['composite_similarity'] >= self.similarity_thresholds['high']:
            print(f"   🔥 發現高度相似的歷史模式！")
            print(f"   📊 最佳匹配: {best_similarity['pattern_symbol']} (相似度: {best_similarity['composite_similarity']:.3f})")
            print(f"   🎯 建議: 密切關注，可能重現相似走勢")
        elif avg_similarity >= self.similarity_thresholds['medium']:
            print(f"   ⚡ 發現中等相似的歷史模式")
            print(f"   📊 平均相似度: {avg_similarity:.3f}")
            print(f"   🎯 建議: 謹慎觀察，結合其他指標分析")
        else:
            print(f"   💡 相似性較低，歷史模式參考價值有限")
            print(f"   🎯 建議: 依賴基本面和技術面分析")
        
        # 價量關係分析
        pv_similarities = [s['price_volume_similarity'] for s in results['similarities'][:5]]
        avg_pv_similarity = np.mean(pv_similarities)
        
        if avg_pv_similarity >= 0.6:
            print(f"   📈 價量關係相似性高 ({avg_pv_similarity:.3f})，模式可靠性較強")
        elif avg_pv_similarity >= 0.4:
            print(f"   📊 價量關係相似性中等 ({avg_pv_similarity:.3f})，需要額外驗證")
        else:
            print(f"   📉 價量關係相似性低 ({avg_pv_similarity:.3f})，可能存在虛假信號")

    def create_similarity_visualization(self, results: Dict, save_path: str = None) -> Optional[str]:
        """創建相似性分析可視化圖表"""
        if not results['similarities']:
            print("❌ 沒有相似性數據可供可視化")
            return None
        
        # 創建圖表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 綜合相似性排名
        top_similarities = results['similarities'][:10]
        symbols = [s['pattern_symbol'] for s in top_similarities]
        similarities = [s['composite_similarity'] for s in top_similarities]
        
        bars = ax1.barh(symbols, similarities, color=['red' if s < 0.4 else 'orange' if s < 0.6 else 'green' for s in similarities])
        ax1.set_xlabel('綜合相似性分數')
        ax1.set_title(f'{results["symbol"]} - 綜合相似性排名')
        ax1.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for i, (bar, similarity) in enumerate(zip(bars, similarities)):
            ax1.text(similarity + 0.01, i, f'{similarity:.3f}', va='center', fontsize=10)
        
        # 2. 多維度相似性雷達圖
        if len(top_similarities) > 0:
            best_match = top_similarities[0]
            categories = ['價格', '成交量', '波動率', '價量關係']
            values = [
                best_match['price_similarity'],
                best_match['volume_similarity'],
                best_match['volatility_similarity'],
                best_match['price_volume_similarity']
            ]
            
            # 創建雷達圖
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 閉合圖形
            angles += angles[:1]
            
            ax2.plot(angles, values, 'o-', linewidth=2, color='blue')
            ax2.fill(angles, values, alpha=0.25, color='blue')
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories)
            ax2.set_ylim(0, 1)
            ax2.set_title(f'最佳匹配多維度分析\n{best_match["pattern_symbol"]}')
            ax2.grid(True)
        
        # 3. 相似性分佈直方圖
        all_similarities = [s['composite_similarity'] for s in results['similarities']]
        ax3.hist(all_similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(all_similarities), color='red', linestyle='--', label=f'平均值: {np.mean(all_similarities):.3f}')
        ax3.set_xlabel('相似性分數')
        ax3.set_ylabel('頻率')
        ax3.set_title('相似性分佈')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 各維度相似性比較
        dimensions = ['價格', '成交量', '波動率', '價量關係']
        dim_similarities = [
            np.mean([s['price_similarity'] for s in top_similarities[:5]]),
            np.mean([s['volume_similarity'] for s in top_similarities[:5]]),
            np.mean([s['volatility_similarity'] for s in top_similarities[:5]]),
            np.mean([s['price_volume_similarity'] for s in top_similarities[:5]])
        ]
        
        bars = ax4.bar(dimensions, dim_similarities, color=['red', 'green', 'blue', 'orange'])
        ax4.set_ylabel('平均相似性分數')
        ax4.set_title('各維度相似性比較 (Top 5)')
        ax4.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bar, similarity in zip(bars, dim_similarities):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{similarity:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存圖表
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"similarity_analysis_{results['symbol']}_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 相似性分析圖表已保存: {save_path}")
        
        plt.close()
        return save_path

    def batch_analyze_symbols(self, symbols: List[str], reference_patterns: List[Dict], 
                            timeframe: str = '1h', days: int = 30) -> Dict:
        """批量分析多個符號的相似性"""
        print(f"\n🚀 開始批量相似性分析...")
        print(f"   目標符號: {len(symbols)} 個")
        print(f"   參考模式: {len(reference_patterns)} 個")
        print(f"   時間框架: {timeframe}")
        print(f"   分析天數: {days}")
        
        batch_results = {
            'analysis_date': datetime.now().isoformat(),
            'timeframe': timeframe,
            'days': days,
            'symbols_analyzed': [],
            'total_symbols': len(symbols),
            'reference_patterns': len(reference_patterns),
            'summary': {}
        }
        
        successful_analyses = 0
        failed_analyses = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] 分析 {symbol}...")
            
            try:
                result = self.analyze_pattern_similarity(symbol, reference_patterns, timeframe, days)
                
                if 'error' not in result:
                    batch_results['symbols_analyzed'].append(result)
                    successful_analyses += 1
                else:
                    print(f"❌ {symbol} 分析失敗: {result['error']}")
                    failed_analyses += 1
                    
            except Exception as e:
                print(f"❌ {symbol} 分析時發生錯誤: {e}")
                failed_analyses += 1
        
        # 生成批量分析摘要
        batch_results['summary'] = {
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'success_rate': successful_analyses / len(symbols) if symbols else 0
        }
        
        # 生成批量報告
        self.generate_batch_report(batch_results)
        
        return batch_results

    def generate_batch_report(self, batch_results: Dict):
        """生成批量分析報告"""
        print(f"\n📊 批量相似性分析報告")
        print("=" * 80)
        
        summary = batch_results['summary']
        print(f"✅ 成功分析: {summary['successful_analyses']} 個符號")
        print(f"❌ 失敗分析: {summary['failed_analyses']} 個符號")
        print(f"📈 成功率: {summary['success_rate']:.1%}")
        
        if not batch_results['symbols_analyzed']:
            print("❌ 沒有成功的分析結果")
            return
        
        # 找出最佳匹配
        best_matches = []
        for result in batch_results['symbols_analyzed']:
            if result['similarities']:
                best_sim = result['similarities'][0]
                best_matches.append({
                    'symbol': result['symbol'],
                    'best_match': best_sim['pattern_symbol'],
                    'similarity': best_sim['composite_similarity'],
                    'level': best_sim['similarity_level']
                })
        
        # 按相似性排序
        best_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"\n🏆 最佳相似性匹配 (Top 15):")
        for i, match in enumerate(best_matches[:15], 1):
            level_emoji = {
                'HIGH': '🔥',
                'MEDIUM': '⚡',
                'LOW': '💡',
                'VERY_LOW': '❄️'
            }.get(match['level'], '❓')
            
            print(f"   {i:2d}. {match['symbol']:12s} → {match['best_match']:12s} "
                  f"{level_emoji} {match['similarity']:.3f}")
        
        # 統計相似性等級分佈
        level_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'VERY_LOW': 0}
        for match in best_matches:
            level_counts[match['level']] += 1
        
        print(f"\n📈 相似性等級分佈:")
        for level, count in level_counts.items():
            percentage = count / len(best_matches) * 100 if best_matches else 0
            emoji = {'HIGH': '🔥', 'MEDIUM': '⚡', 'LOW': '💡', 'VERY_LOW': '❄️'}[level]
            print(f"   {emoji} {level:8s}: {count:3d} 個 ({percentage:5.1f}%)")
        
        # 平均相似性
        avg_similarity = np.mean([m['similarity'] for m in best_matches]) if best_matches else 0
        print(f"\n📊 平均相似性: {avg_similarity:.3f}")
        
        # 投資建議摘要
        high_similarity_count = level_counts['HIGH']
        if high_similarity_count > 0:
            print(f"\n💡 投資建議摘要:")
            print(f"   🔥 發現 {high_similarity_count} 個高相似性符號，值得重點關注")
            print(f"   📊 建議優先分析這些符號的歷史模式和未來走勢")
        elif level_counts['MEDIUM'] > 0:
            print(f"\n💡 投資建議摘要:")
            print(f"   ⚡ 發現 {level_counts['MEDIUM']} 個中等相似性符號")
            print(f"   📊 建議結合其他技術指標進行綜合分析")
        else:
            print(f"\n💡 投資建議摘要:")
            print(f"   💡 整體相似性較低，建議依賴基本面和技術面分析")

# 使用示例和測試函數
def create_sample_reference_patterns():
    """創建示例參考模式"""
    patterns = [
        {
            'name': 'BTC_Bull_Run_2024',
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'days': 30,
            'description': 'BTC 2024年牛市模式'
        },
        {
            'name': 'ETH_Breakout_Pattern',
            'symbol': 'ETHUSDT',
            'timeframe': '1h',
            'days': 30,
            'description': 'ETH 突破模式'
        },
        {
            'name': 'SOL_Recovery_Pattern',
            'symbol': 'SOLUSDT',
            'timeframe': '1h',
            'days': 30,
            'description': 'SOL 復甦模式'
        }
    ]
    
    return patterns

def test_enhanced_similarity_analyzer():
    """測試增強版相似性分析器"""
    print("🧪 測試增強版相似性分析器...")
    
    # 創建分析器
    analyzer = EnhancedSimilarityAnalyzer()
    
    # 創建參考模式
    reference_patterns = create_sample_reference_patterns()
    
    # 測試單個符號分析
    test_symbol = 'ADAUSDT'
    result = analyzer.analyze_pattern_similarity(test_symbol, reference_patterns)
    
    # 創建可視化
    if 'error' not in result:
        analyzer.create_similarity_visualization(result)
    
    print("✅ 測試完成")

if __name__ == "__main__":
    test_enhanced_similarity_analyzer() 