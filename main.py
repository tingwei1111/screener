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
  python main.py portfolio --symbols BTC ETH ADA    # 投資組合優化
  python main.py monitor --add BTCUSDT rs_score 8.0 # 實時監控
  python main.py trade --start                      # 自動交易
  python main.py web                                # Web儀表板

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
    
    # 4. 投資組合優化
    portfolio_parser = subparsers.add_parser('portfolio', help='投資組合優化')
    portfolio_parser.add_argument('--symbols', nargs='+', required=True, help='資產符號列表')
    portfolio_parser.add_argument('--method', default='max_sharpe', 
                                choices=['max_sharpe', 'min_vol', 'risk_parity', 'equal_weight'],
                                help='優化方法')
    portfolio_parser.add_argument('--days', type=int, default=252, help='歷史數據天數')
    portfolio_parser.add_argument('--monte-carlo', action='store_true', help='執行蒙特卡羅模擬')
    portfolio_parser.add_argument('--plot', action='store_true', help='繪製圖表')
    
    # 5. 實時監控
    monitor_parser = subparsers.add_parser('monitor', help='實時監控')
    monitor_parser.add_argument('--add', nargs=3, metavar=('SYMBOL', 'TYPE', 'THRESHOLD'),
                               help='添加警報: 符號 類型 閾值')
    monitor_parser.add_argument('--list', action='store_true', help='列出所有警報')
    monitor_parser.add_argument('--start', action='store_true', help='開始監控')
    monitor_parser.add_argument('--stop', action='store_true', help='停止監控')
    
    # 6. 自動交易
    trade_parser = subparsers.add_parser('trade', help='自動交易')
    trade_parser.add_argument('--start', action='store_true', help='開始交易')
    trade_parser.add_argument('--stop', action='store_true', help='停止交易')
    trade_parser.add_argument('--status', action='store_true', help='查看狀態')
    trade_parser.add_argument('--test', nargs=3, metavar=('SYMBOL', 'SIGNAL', 'CONFIDENCE'),
                             help='測試信號: 符號 信號 信心度')
    
    # 7. Web儀表板
    web_parser = subparsers.add_parser('web', help='Web儀表板')
    web_parser.add_argument('--port', type=int, default=8501, help='端口號 (默認: 8501)')
    web_parser.add_argument('--host', default='localhost', help='主機地址')
    
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
        elif args.command == 'portfolio':
            run_portfolio_optimizer(args)
        elif args.command == 'monitor':
            run_monitor(args)
        elif args.command == 'trade':
            run_trader(args)
        elif args.command == 'web':
            run_web_dashboard(args)
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

def run_portfolio_optimizer(args):
    """運行投資組合優化器"""
    import subprocess
    cmd = [
        'python', 'portfolio_optimizer.py',
        '--symbols'
    ] + args.symbols + [
        '--method', args.method,
        '--days', str(args.days)
    ]
    
    if args.monte_carlo:
        cmd.append('--monte-carlo')
    if args.plot:
        cmd.append('--plot')
    
    subprocess.run(cmd)

def run_monitor(args):
    """運行實時監控"""
    import subprocess
    cmd = ['python', 'real_time_monitor.py']
    
    if args.add:
        cmd.extend(['--add-alert'] + args.add)
    elif args.list:
        cmd.append('--list-alerts')
    elif args.start:
        cmd.append('--start')
    elif args.stop:
        cmd.append('--stop')
    
    subprocess.run(cmd)

def run_trader(args):
    """運行自動交易器"""
    import subprocess
    cmd = ['python', 'auto_trader.py']
    
    if args.start:
        cmd.append('--start')
    elif args.stop:
        cmd.append('--stop')
    elif args.status:
        cmd.append('--status')
    elif args.test:
        cmd.extend(['--test-signal'] + args.test)
    
    subprocess.run(cmd)

def run_web_dashboard(args):
    """運行Web儀表板"""
    import subprocess
    cmd = [
        'streamlit', 'run', 'web_dashboard.py',
        '--server.port', str(args.port),
        '--server.address', args.host
    ]
    subprocess.run(cmd)

if __name__ == '__main__':
    main() 