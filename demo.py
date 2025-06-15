#!/usr/bin/env python3
"""
Screener 工具演示腳本
===================

這個腳本展示了如何使用 Screener 工具的基本功能。
運行這個腳本來快速體驗所有主要功能。

使用方法：
python3 demo.py
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def print_header(title):
    """打印標題"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """打印步驟"""
    print(f"\n📋 步驟 {step}: {description}")
    print("-" * 40)

def run_command(command, description):
    """運行命令並顯示結果"""
    print(f"💻 執行命令: {command}")
    print(f"📝 說明: {description}")
    print("\n⏳ 執行中...")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 執行成功！")
            # 只顯示最後幾行輸出
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                print("📊 輸出摘要（最後 10 行）:")
                for line in lines[-10:]:
                    print(f"   {line}")
            else:
                print("📊 完整輸出:")
                print(result.stdout)
        else:
            print("❌ 執行失敗！")
            print(f"錯誤信息: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 執行超時（5分鐘）")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
    
    print("\n" + "-" * 40)

def check_dependencies():
    """檢查依賴項目"""
    print_step(0, "檢查環境和依賴項目")
    
    # 檢查 Python 版本
    python_version = sys.version_info
    print(f"🐍 Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ 需要 Python 3.8 或更高版本")
        return False
    
    # 檢查必要文件
    required_files = [
        "crypto_screener.py",
        "stock_screener.py", 
        "crypto_trend_screener.py",
        "crypto_historical_trend_finder.py",
        "requirements.txt",
        "api_keys.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 所有必要文件都存在")
    
    # 檢查依賴包
    try:
        import pandas
        import numpy
        import matplotlib
        import requests
        print("✅ 主要依賴包已安裝")
    except ImportError as e:
        print(f"❌ 缺少依賴包: {e}")
        print("請運行: pip3 install -r requirements.txt")
        return False
    
    return True

def demo_crypto_screener():
    """演示加密貨幣篩選器"""
    print_step(1, "加密貨幣篩選器演示")
    
    print("🔍 這個工具會分析數百個加密貨幣，找出相對強勢的標的")
    print("📊 使用相對強度分析，結合移動平均線和 ATR 指標")
    print("⏱️  預計執行時間: 2-5 分鐘")
    
    # 使用較短的時間框架和天數來加快演示
    command = "python3 crypto_screener.py -t 1h -d 2"
    description = "分析加密貨幣（1小時時間框架，2天數據）"
    
    run_command(command, description)

def demo_stock_screener():
    """演示股票篩選器"""
    print_step(2, "股票篩選器演示")
    
    print("📈 這個工具會分析美股，使用相對強度和 Minervini 趨勢模板")
    print("🏦 如果沒有 API keys，會使用內建的熱門股票列表")
    print("⏱️  預計執行時間: 1-3 分鐘")
    
    command = "python3 stock_screener.py -g"
    description = "分析股票（只計算 RS 分數，使用備用股票列表）"
    
    run_command(command, description)

def demo_trend_screener():
    """演示趨勢篩選器"""
    print_step(3, "趨勢篩選器演示")
    
    print("🔄 這個工具使用 DTW 算法找到相似的價格模式")
    print("📊 會生成視覺化圖表顯示相似趨勢")
    print("⏱️  預計執行時間: 3-8 分鐘")
    
    # 檢查是否有之前的輸出文件
    output_dir = "output"
    if os.path.exists(output_dir):
        # 尋找最新的加密貨幣篩選結果
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if "crypto" in file and "strong_targets.txt" in file:
                    file_path = os.path.join(root, file)
                    command = f"python3 crypto_trend_screener.py -f {file_path} -k 5 -s 1.0"
                    description = f"使用篩選結果進行趨勢分析: {file}"
                    run_command(command, description)
                    return
    
    # 如果沒有找到文件，使用默認設置
    command = "python3 crypto_trend_screener.py -k 5 -s 1.0"
    description = "趨勢分析（使用所有可用符號，限制結果數量）"
    
    run_command(command, description)

def demo_historical_finder():
    """演示歷史趨勢分析器"""
    print_step(4, "歷史趨勢分析器演示")
    
    print("📚 這個工具搜索歷史數據中的相似模式")
    print("📈 分析模式完成後的未來價格走勢")
    print("⏱️  預計執行時間: 10-20 分鐘（數據量大）")
    print("⚠️  這是最耗時的功能，建議在有充足時間時運行")
    
    response = input("\n是否要運行歷史趨勢分析器？這可能需要很長時間 (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        command = "python3 crypto_historical_trend_finder.py -k 50 -s 5"
        description = "歷史趨勢分析（限制結果數量和增加延遲以加快速度）"
        run_command(command, description)
    else:
        print("⏭️  跳過歷史趨勢分析器演示")

def show_results():
    """顯示生成的結果文件"""
    print_step(5, "查看生成的結果")
    
    result_dirs = [
        ("output", "篩選器結果"),
        ("similarity_output", "趨勢分析結果"),
        ("past_similar_trends_report", "歷史趨勢分析結果")
    ]
    
    for dir_name, description in result_dirs:
        if os.path.exists(dir_name):
            print(f"\n📁 {description} ({dir_name}):")
            for root, dirs, files in os.walk(dir_name):
                for file in files[:5]:  # 只顯示前5個文件
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"   📄 {file} ({file_size} bytes)")
                if len(files) > 5:
                    print(f"   ... 還有 {len(files) - 5} 個文件")
        else:
            print(f"\n📁 {description}: 沒有生成文件")

def main():
    """主函數"""
    print_header("Screener 工具演示")
    
    print("🎯 這個演示將展示 Screener 工具的所有主要功能：")
    print("   1. 加密貨幣篩選器 - 找出強勢加密貨幣")
    print("   2. 股票篩選器 - 分析美股相對強度") 
    print("   3. 趨勢篩選器 - 找出相似價格模式")
    print("   4. 歷史趨勢分析器 - 分析歷史模式和未來走勢")
    
    print("\n⚠️  注意事項：")
    print("   • 加密貨幣功能完全免費，無需 API keys")
    print("   • 股票功能會使用備用列表（如果沒有 API keys）")
    print("   • 某些功能可能需要較長時間執行")
    print("   • 請確保網路連接穩定")
    
    response = input("\n是否要開始演示？(Y/n): ")
    if response.lower() in ['n', 'no']:
        print("👋 演示已取消")
        return
    
    # 檢查環境
    if not check_dependencies():
        print("\n❌ 環境檢查失敗，請解決上述問題後重新運行")
        return
    
    start_time = time.time()
    
    try:
        # 運行各個演示
        demo_crypto_screener()
        demo_stock_screener()
        demo_trend_screener()
        demo_historical_finder()
        show_results()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用戶中斷")
    except Exception as e:
        print(f"\n\n❌ 演示過程中發生錯誤: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("演示完成")
    print(f"⏱️  總執行時間: {duration:.1f} 秒")
    print("📚 更多詳細信息請查看:")
    print("   • 使用教學.md - 詳細使用指南")
    print("   • API_Keys_設置指南.md - API 設置說明")
    print("   • README.md - 英文說明文檔")
    print("\n🎉 感謝使用 Screener 工具！")

if __name__ == "__main__":
    main() 