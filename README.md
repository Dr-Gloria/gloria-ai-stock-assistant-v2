
# Gloria AI Stock Assistant V5 Final

這是 V5 Final 穩定版。

## 重要更新

此版本已移除：

- scikit-learn
- sklearn
- RandomForest

因此 Streamlit Cloud 部署會比較穩定，不會再出現：

```text
ModuleNotFoundError: No module named 'sklearn'
```

## 核心功能

首頁直接回答：

- 今天是否應該買？
- 建議多少錢買？
- 不要追高超過多少？
- 停損是多少？
- 第一賣點是多少？
- 第二賣點是多少？
- 若已持有，目前應該續抱、停利、停損或賣出？

## 使用者可以輸入

- 股票代號
- 是否已持有
- 持有成本價
- 持有張數
- 分析週期：日線 1D / 小時線 1H
- 回測期間

## 使用的邏輯

此版本是 Rule-based Trading Assistant，使用：

- PSAR
- MA20 / MA60
- RSI
- MACD
- KD
- Bollinger Bands
- ATR
- 三大法人買賣超
- PSAR 策略回測
- Buy-and-Hold 比較

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub 更新

上傳以下檔案並取代舊版：

- app.py
- requirements.txt
- README.md

然後 Commit changes。

Streamlit Cloud 會重新部署。

## 注意

本工具只供教學與研究使用，不是投資建議，也不保證獲利。
