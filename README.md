
# Gloria AI Stock Assistant v3

這是 Gloria AI Stock Assistant 的第三版。

## v3 新增功能

- 新增交易模式
- 顯示今日交易判斷
- 顯示建議買點區間
- 顯示停損點
- 顯示第一目標價
- 顯示第二目標價
- 顯示風險與報酬比例
- 美化 Dashboard 畫面
- 圖表加入買點、停損與目標價參考線

## 指標內容

### 基本面
- EPS
- 本益比 PE
- ROE
- 殖利率

### 技術面
- PSAR
- RSI
- MACD
- KD
- MA20
- MA60
- Bollinger Bands
- ATR

### 籌碼面
- 三大法人
- 外資
- 投信
- 自營商
- 融資融券

## 執行方式

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub 更新方式

如果你已經有 GitHub repository：

1. 打開 repository
2. 點 `app.py`
3. 點右上角鉛筆圖示
4. 全部刪掉，貼上新版 app.py
5. Commit changes

或直接：

1. 點 `Add file`
2. 點 `Upload files`
3. 上傳新版 `app.py` 和 `README.md`
4. 選 Replace
5. Commit changes

Streamlit Cloud 會自動重新部署。

## 注意

本工具只供教學與研究，不是投資建議，也不保證獲利。
買點、停損與目標價是根據技術指標和波動度推估，不代表真實市場一定會到達。
