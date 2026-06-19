# Gloria AI Stock Assistant v5

v5 是 AI 預測版。

## v5 新增

- Random Forest AI 預測引擎
- 未來 5 / 10 / 20 期上漲機率
- AI 特徵重要性
- AI 股票排行榜
- 策略回測頁
- PSAR 策略與 Buy-and-Hold 比較
- 保留 v4 的買點、賣點、停損、停利與持股成本分析

## 頁面

### 我的股票
輸入股票代號，系統直接回答：今天是否應該買、多少錢買、多少錢賣、停損多少；若已持有，要續抱、停利、停損或賣出。

### AI排行榜
對多檔股票進行快速排序，顯示 AI 上漲機率與綜合分數。

### 策略回測
顯示 PSAR 策略報酬、Buy-and-Hold 報酬、勝率、平均每筆報酬與最大回撤。

## 安裝

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub 更新

上傳並取代舊版：app.py、requirements.txt、README.md，然後 Commit changes。Streamlit Cloud 會重新部署。

## 注意

v5 的 AI 模型是在 App 查詢時使用該股票歷史資料訓練 Random Forest。這仍然是研究與教學用工具，不是投資建議，也不保證獲利。
