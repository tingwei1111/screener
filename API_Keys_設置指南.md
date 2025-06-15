# 🔑 API Keys 設置指南

這個指南將幫助你獲取和設置所需的 API keys，以充分利用 Screener 工具的所有功能。

## 📋 API Keys 概覽

| API 服務 | 用途 | 是否必需 | 免費額度 |
|---------|------|---------|----------|
| Polygon | 美股數據 | 股票功能需要 | 5 calls/min |
| StockSymbol | 股票符號列表 | 可選 | 有限制 |
| Binance | 加密貨幣數據 | 不需要 | 無限制 |

## 🚀 重要提醒

**加密貨幣功能完全免費！** 
- `crypto_screener.py` ✅ 無需 API key
- `crypto_trend_screener.py` ✅ 無需 API key  
- `crypto_historical_trend_finder.py` ✅ 無需 API key

**只有股票功能需要 API keys**

## 📈 Polygon API 設置（推薦）

### 1. 註冊 Polygon 帳戶

1. 前往 [Polygon.io](https://polygon.io/)
2. 點擊 "Sign Up" 註冊免費帳戶
3. 驗證你的電子郵件地址

### 2. 獲取 API Key

1. 登入後，前往 [Dashboard](https://polygon.io/dashboard)
2. 在左側選單點擊 "API Keys"
3. 複製你的 API key（格式類似：`YOUR_API_KEY_HERE`）

### 3. 免費方案限制

**免費方案包含：**
- ✅ 5 calls/minute
- ✅ 歷史股票數據
- ✅ 實時數據（延遲 15 分鐘）
- ✅ 基本市場數據

**付費方案優勢：**
- 🚀 更高的請求限制
- 🚀 實時數據
- 🚀 更多數據類型

## 📊 StockSymbol API 設置（可選）

### 1. 註冊 StockSymbol 帳戶

1. 前往 [StockSymbol API](https://stocksymbol.com/)
2. 註冊免費帳戶
3. 獲取你的 API key

### 2. 注意事項

- StockSymbol API 有時不穩定
- 程式已內建備用股票列表
- 如果 API 失敗，會自動使用備用列表

## ⚙️ 設置 API Keys

### 1. 編輯配置文件

打開 `api_keys.json` 文件：

```json
{
  "stocksymbol": "你的_STOCKSYMBOL_API_KEY",
  "polygon": "你的_POLYGON_API_KEY"
}
```

### 2. 替換 API Keys

將 `你的_POLYGON_API_KEY` 替換為你從 Polygon 獲得的實際 API key：

```json
{
  "stocksymbol": "YOUR_STOCKSYMBOL_API_KEY_HERE",
  "polygon": "abcdef123456789"
}
```

### 3. 如果沒有某個 API Key

如果你只有 Polygon API key，可以這樣設置：

```json
{
  "stocksymbol": "",
  "polygon": "你的_POLYGON_API_KEY"
}
```

## 🧪 測試 API Keys

### 測試 Polygon API

```bash
python3 -c "
import json
from polygon import RESTClient

# 讀取 API key
with open('api_keys.json', 'r') as f:
    keys = json.load(f)

# 測試 Polygon API
try:
    client = RESTClient(keys['polygon'])
    tickers = client.list_tickers(market='stocks', limit=5)
    print('✅ Polygon API 測試成功！')
    print(f'獲得 {len(list(tickers))} 個股票符號')
except Exception as e:
    print(f'❌ Polygon API 測試失敗: {e}')
"
```

### 測試股票篩選器

```bash
# 測試股票篩選器（使用備用列表）
python3 stock_screener.py -g
```

如果看到類似輸出，表示設置成功：
```
Found 99 unique stock symbols
Total tickers to process: 99
```

## 🔒 安全注意事項

### 1. 保護你的 API Keys

- ❌ 不要將 API keys 提交到 Git
- ❌ 不要在公開場所分享 API keys
- ✅ 定期更換 API keys
- ✅ 使用環境變量（進階用戶）

### 2. Git 忽略設置

確保 `.gitignore` 文件包含：
```
api_keys.json
*.log
output/
similarity_output/
past_similar_trends_report/
```

### 3. 環境變量設置（進階）

你也可以使用環境變量：

```bash
export POLYGON_API_KEY="你的_API_KEY"
export STOCKSYMBOL_API_KEY="你的_API_KEY"
```

然後修改程式讀取環境變量。

## 🆓 免費使用方案

### 只使用加密貨幣功能

如果你只想分析加密貨幣，完全不需要任何 API keys：

```bash
# 這些命令都不需要 API keys
python3 crypto_screener.py
python3 crypto_trend_screener.py
python3 crypto_historical_trend_finder.py
```

### 使用備用股票列表

即使沒有 API keys，股票篩選器也能工作：

```bash
# 使用內建的 99 個熱門美股
python3 stock_screener.py -g
```

## 🔧 故障排除

### 常見錯誤

**1. "authorization header was malformed"**
```
原因：Polygon API key 無效或格式錯誤
解決：檢查 API key 是否正確複製
```

**2. "Something is wrong. Please contact the creator"**
```
原因：StockSymbol API 服務問題
解決：程式會自動使用備用股票列表
```

**3. "API key not found"**
```
原因：api_keys.json 文件格式錯誤
解決：檢查 JSON 格式是否正確
```

### 檢查 API 狀態

你可以檢查 API 服務狀態：
- [Polygon Status](https://status.polygon.io/)
- [Binance Status](https://www.binance.com/en/support/announcement)

## 💡 使用建議

### 新手建議

1. **先試用加密貨幣功能**（無需 API keys）
2. **如果需要股票分析，申請免費 Polygon API**
3. **熟悉工具後再考慮付費方案**

### 進階用戶

1. **考慮 Polygon 付費方案**（更高請求限制）
2. **設置環境變量**（更安全）
3. **自定義 API 請求間隔**（避免限制）

## 🎯 快速開始

如果你想立即開始使用：

```bash
# 1. 無需 API keys - 分析加密貨幣
python3 crypto_screener.py

# 2. 使用備用列表 - 分析股票
python3 stock_screener.py -g

# 3. 申請免費 Polygon API 後 - 完整股票功能
# 編輯 api_keys.json，然後：
python3 stock_screener.py
```

現在你已經準備好充分利用 Screener 工具了！🚀 