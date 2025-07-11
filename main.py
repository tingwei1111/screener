#!/usr/bin/env python3
"""
Screener 主啟動腳本
統一的命令行界面，整合所有功能模塊
"""

import argparse
import sys
import os
from pathlib import Path

# 添加當前目錄到Python路徑
sys.path.append(str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(
        description="Screener - 加密貨幣與股票趨勢分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py crypto --top 20                    # 加密貨幣篩選
  python main.py trend --symbol BTCUSDT             # 趨勢分析
  python main.py ml --train BTCUSDT                 # 機器學習訓練
  python main.py ml --symbol BTC                    # 機器學習預測

更多幫助: python main.py <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 1. 加密貨幣篩選
    crypto_parser = subparsers.add_parser('crypto', help='加密貨幣篩選')
    crypto_parser.add_argument('--top', type=int, default=20, help='顯示前N名 (默認: 20)')
    crypto_parser.add_argument('--min-score', type=float, default=7.0, help='最低RS分數 (默認: 7.0)')
    crypto_parser.add_argument('--timeframe', default='1d', help='時間框架 (默認: 1d)')
    
    # 2. 趨勢分析
    trend_parser = subparsers.add_parser('trend', help='趨勢分析')
    trend_parser.add_argument('--symbol', required=True, help='交易對符號')
    trend_parser.add_argument('--timeframe', default='1h', help='時間框架 (默認: 1h)')
    trend_parser.add_argument('--days', type=int, default=30, help='分析天數 (默認: 30)')
    
    # 3. 機器學習
    ml_parser = subparsers.add_parser('ml', help='機器學習預測')
    ml_parser.add_argument('--train', help='訓練模型的交易對')
    ml_parser.add_argument('--predict', help='預測的交易對')
    ml_parser.add_argument('--load', help='載入模型路徑')
    ml_parser.add_argument('--save', help='保存模型路徑')
    ml_parser.add_argument('--timeframe', default='1h', help='時間框架')
    ml_parser.add_argument('--days', type=int, default=365, help='訓練天數')
    

    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'crypto':
            run_crypto_screener(args)
        elif args.command == 'trend':
            run_trend_analysis(args)
        elif args.command == 'ml':
            run_ml_predictor(args)
    except KeyboardInterrupt:
        print("\n程序被用戶中斷")
    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)

def run_crypto_screener(args):
    """運行加密貨幣篩選器 - 直接導入模組避免subprocess開銷"""
    try:
        # 直接導入模組而非使用subprocess
        import crypto_screener
        import sys
        
        # 設置命令行參數
        original_argv = sys.argv
        sys.argv = ['crypto_screener.py', '--timeframe', args.timeframe, '--days', str(getattr(args, 'days', 3))]
        
        # 直接調用main函數
        crypto_screener.main() if hasattr(crypto_screener, 'main') else exec(open('crypto_screener.py').read())
        
        # 恢復原始argv
        sys.argv = original_argv
    except Exception as e:
        print(f"Error running crypto screener: {e}")
        # 回退到subprocess
        import subprocess
        cmd = ['python', 'crypto_screener.py', '--timeframe', args.timeframe]
        subprocess.run(cmd)

def run_trend_analysis(args):
    """運行趨勢分析"""
    import subprocess
    cmd = [
        'python', 'crypto_trend_screener.py',
        '--symbol', args.symbol,
        '--timeframe', args.timeframe,
        '--days', str(args.days)
    ]
    subprocess.run(cmd)

def run_ml_predictor(args):
    """運行機器學習預測"""
    import subprocess
    cmd = ['python', 'ml_predictor.py']
    
    if args.train:
        cmd.extend(['--train', args.train])
        if args.save:
            cmd.extend(['--save', args.save])
    elif args.predict:
        cmd.extend(['--predict', args.predict])
        if args.load:
            cmd.extend(['--load', args.load])
    
    cmd.extend(['--timeframe', args.timeframe, '--days', str(args.days)])
    subprocess.run(cmd)





if __name__ == '__main__':
    main() 