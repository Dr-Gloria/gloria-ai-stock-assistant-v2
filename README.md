
# Gloria AI Stock Assistant v4

v4 是「決策版」股票 APP。

## 核心功能

首頁直接回答：

- 今天是否應該買？
- 建議多少錢買？
- 不要追高超過多少？
- 停損是多少？
- 第一賣點是多少？
- 第二賣點是多少？
- 若已持有，目前應該續抱、停利、停損，還是賣出？

## 使用者可以輸入

- 股票代號
- 是否已持有
- 持有成本價
- 持有張數
- 分析週期：日線 1D / 小時線 1H
- 回測期間

## 新增功能

- 決策式首頁
- 買點價格
- 賣點價格
- 停損價格
- 持股成本分析
- 未實現損益
- PSAR 回測
- Buy-and-Hold 比較
- 未來 20 期上漲機率估計
- Trading Mode / Investment Mode 分頁

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub 更新

將新版：

- app.py
- README.md
- requirements.txt

上傳到原本 GitHub repository，選 Replace，Commit changes。

Streamlit Cloud 通常會自動重新部署。

## 注意

本工具只供教學與研究使用，不是投資建議，也不保證獲利。
