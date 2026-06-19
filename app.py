
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Gloria AI Stock Assistant V5 Final",
    page_icon="💰",
    layout="wide"
)

# =============================
# Style
# =============================
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.hero {
    background: linear-gradient(135deg, #111827 0%, #374151 42%, #8b6f47 100%);
    padding: 28px 32px;
    border-radius: 26px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 12px 30px rgba(15,23,42,0.22);
}
.hero h1 {margin: 0; font-size: 36px;}
.hero p {margin-top: 10px; font-size: 16px; opacity: 0.92;}
.big-signal {
    padding: 28px;
    border-radius: 26px;
    color: #111827;
    box-shadow: 0 10px 26px rgba(15,23,42,0.10);
    border: 1px solid rgba(255,255,255,0.8);
    margin-bottom: 16px;
}
.buy {background: linear-gradient(135deg, #dcfce7 0%, #86efac 100%); border-left: 10px solid #16a34a;}
.wait {background: linear-gradient(135deg, #fef9c3 0%, #fde68a 100%); border-left: 10px solid #ca8a04;}
.sell {background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%); border-left: 10px solid #dc2626;}
.hold {background: linear-gradient(135deg, #dbeafe 0%, #93c5fd 100%); border-left: 10px solid #2563eb;}
.signal-title {font-size: 15px; color: #374151; font-weight: 700;}
.signal-main {font-size: 44px; font-weight: 900; margin-top: 6px;}
.signal-sub {font-size: 16px; margin-top: 8px; color: #374151;}
.price-card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.08);
    border: 1px solid #eef0f4;
    min-height: 125px;
}
.price-label {font-size: 14px; color: #6b7280; font-weight: 700;}
.price-value {font-size: 34px; font-weight: 900; color: #111827; margin-top: 6px;}
.price-note {font-size: 13px; color: #6b7280; margin-top: 5px;}
.section-title {font-size: 24px; font-weight: 900; margin-top: 24px; margin-bottom: 12px;}
.reason-box {
    background: #ffffff;
    padding: 18px 20px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 16px rgba(15,23,42,0.06);
}
.alert-box {
    background: #fff7ed;
    border-left: 7px solid #f97316;
    padding: 16px 18px;
    border-radius: 15px;
    color: #7c2d12;
    margin-top: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>💰 Gloria AI Stock Assistant V5 Final</h1>
<p>決策版：今天買不買？多少錢買？多少錢賣？如果已持有，現在要續抱、停利還是停損？</p>
</div>
""", unsafe_allow_html=True)

# =============================
# Helpers
# =============================
def normalize_ticker(raw: str):
    raw = raw.strip().upper()
    if raw.isdigit():
        return f"{raw}.TW", raw
    return raw, raw

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

@st.cache_data(ttl=600)
def fetch_price_data(ticker: str, period: str, interval: str):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["High", "Low", "Close"])
    return df

# =============================
# Indicators
# =============================
def parabolic_sar(df: pd.DataFrame, step=0.02, max_step=0.2):
    high = df["High"].values
    low = df["Low"].values
    n = len(df)
    sar = np.zeros(n)

    trend_up = True
    af = step
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        sar[i] = prev_sar + af * (ep - prev_sar)

        if trend_up:
            sar[i] = min(sar[i], low[i - 1])
            if i > 1:
                sar[i] = min(sar[i], low[i - 2])

            if low[i] < sar[i]:
                trend_up = False
                sar[i] = ep
                ep = low[i]
                af = step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            sar[i] = max(sar[i], high[i - 1])
            if i > 1:
                sar[i] = max(sar[i], high[i - 2])

            if high[i] > sar[i]:
                trend_up = True
                sar[i] = ep
                ep = high[i]
                af = step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)

    return pd.Series(sar, index=df.index, name="PSAR")

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50).rename("RSI")

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.rename("MACD"), signal_line.rename("MACD_signal"), hist.rename("MACD_hist")

def stochastic_kd(df, n=9):
    low_min = df["Low"].rolling(n).min()
    high_max = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    return k.rename("K"), d.rename("D")

def bollinger(series, window=20, n_std=2):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (mid + n_std * std).rename("BB_upper"), mid.rename("BB_mid"), (mid - n_std * std).rename("BB_lower")

def atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean().rename("ATR")

def prepare_indicators(df):
    df = df.copy()
    df["PSAR"] = parabolic_sar(df)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    df["K"], df["D"] = stochastic_kd(df)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = bollinger(df["Close"])
    df["ATR"] = atr(df)
    df["Return_20"] = df["Close"].pct_change(20) * 100
    df["Volume_MA20"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan
    return df.dropna()

# =============================
# Fundamentals and TWSE chips
# =============================
@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    roe = safe_float(info.get("returnOnEquity"))
    dy = safe_float(info.get("dividendYield"))
    return {
        "Company": info.get("longName") or info.get("shortName") or ticker,
        "EPS": safe_float(info.get("trailingEps")),
        "PE": safe_float(info.get("trailingPE")),
        "ROE (%)": roe * 100 if not np.isnan(roe) else np.nan,
        "Dividend Yield (%)": dy * 100 if not np.isnan(dy) else np.nan,
        "Sector": info.get("sector", "N/A"),
    }

@st.cache_data(ttl=3600)
def fetch_twse_institutional(stock_code):
    result = {"foreign_net": np.nan, "trust_net": np.nan, "dealer_net": np.nan, "total_net": np.nan, "note": "No TWSE data."}
    if not stock_code.isdigit():
        return result
    for days_back in range(1, 15):
        date_str = (datetime.today() - timedelta(days=days_back)).strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json"
        try:
            data = requests.get(url, timeout=8).json()
            fields, rows = data.get("fields", []), data.get("data", [])
            if not fields or not rows:
                continue
            d = pd.DataFrame(rows, columns=fields)
            row = d[d["證券代號"].astype(str).str.strip() == stock_code]
            if row.empty:
                continue
            row = row.iloc[0]
            def parse(v):
                try:
                    return float(str(v).replace(",", "").replace("--", "0"))
                except Exception:
                    return np.nan
            return {
                "foreign_net": parse(row.get("外陸資買賣超股數(不含外資自營商)", np.nan)),
                "trust_net": parse(row.get("投信買賣超股數", np.nan)),
                "dealer_net": parse(row.get("自營商買賣超股數", np.nan)),
                "total_net": parse(row.get("三大法人買賣超股數", np.nan)),
                "note": f"TWSE date: {date_str}",
            }
        except Exception:
            continue
    return result

# =============================
# Decision logic
# =============================
def score_market(df, chip):
    latest, prev = df.iloc[-1], df.iloc[-2]
    score = 0
    reasons_buy = []
    reasons_sell = []

    if latest["Close"] > latest["PSAR"]:
        score += 18; reasons_buy.append("價格在 PSAR 上方，短線偏多。")
    else:
        reasons_sell.append("價格跌破 PSAR，屬於轉弱訊號。")

    if latest["MA20"] > latest["MA60"]:
        score += 16; reasons_buy.append("MA20 高於 MA60，中期趨勢偏多。")
    else:
        reasons_sell.append("MA20 低於 MA60，中期趨勢還不夠強。")

    if latest["MACD"] > latest["MACD_signal"]:
        score += 16; reasons_buy.append("MACD 在訊號線上方，動能偏多。")
    else:
        reasons_sell.append("MACD 在訊號線下方，動能偏弱。")

    if latest["K"] > latest["D"] and latest["K"] < 85:
        score += 12; reasons_buy.append("KD 向上且未嚴重過熱。")
    elif latest["K"] > 85:
        score += 4; reasons_sell.append("KD 偏高，短線可能過熱。")
    else:
        reasons_sell.append("KD 尚未轉強。")

    if 45 <= latest["RSI"] <= 70:
        score += 12; reasons_buy.append("RSI 偏多但未明顯過熱。")
    elif latest["RSI"] > 75:
        score += 2; reasons_sell.append("RSI 過熱，追價風險較高。")
    elif latest["RSI"] < 35:
        reasons_sell.append("RSI 偏弱，尚未確認止跌。")
    else:
        score += 6

    if latest["Close"] > latest["BB_mid"]:
        score += 10; reasons_buy.append("價格位於布林中軌上方。")
    else:
        reasons_sell.append("價格在布林中軌下方。")

    if latest["Return_20"] > 0:
        score += 8; reasons_buy.append("近 20 期報酬為正。")
    else:
        reasons_sell.append("近 20 期報酬為負。")

    if not np.isnan(chip.get("total_net", np.nan)):
        if chip["total_net"] > 0:
            score += 8; reasons_buy.append("三大法人買超。")
        else:
            reasons_sell.append("三大法人賣超。")

    return min(score, 100), reasons_buy, reasons_sell

def decision_without_position(df, score):
    latest = df.iloc[-1]
    close, ma20, psar, atr_v = latest["Close"], latest["MA20"], latest["PSAR"], latest["ATR"]
    bb_mid, bb_upper, bb_lower = latest["BB_mid"], latest["BB_upper"], latest["BB_lower"]

    buy1 = max(min(ma20, psar, bb_mid), close - 0.9 * atr_v)
    buy2 = max(bb_lower, buy1 - 0.8 * atr_v)
    chase_limit = min(close * 1.015, close + 0.25 * atr_v)

    stop_loss = min(psar, ma20 - 0.6 * atr_v, close - 1.4 * atr_v)
    if stop_loss >= close:
        stop_loss = close - 1.2 * atr_v

    sell1 = max(bb_upper, close + 1.5 * atr_v)
    sell2 = close + 2.5 * atr_v

    confidence = max(5, min(95, 45 + (score - 50) * 0.7))

    if score >= 75 and close <= chase_limit:
        zh, cls = "🟢 買進", "buy"
        one_line = f"可考慮在 {buy1:.2f}–{chase_limit:.2f} 分批買進，不建議追高超過 {chase_limit:.2f}。"
    elif score >= 60:
        zh, cls = "🟡 觀望，等買點", "wait"
        one_line = f"目前不要追價，較合理買點在 {buy2:.2f}–{buy1:.2f}。"
    else:
        zh, cls = "🔴 不買 / 避開", "sell"
        one_line = f"訊號不足，不建議買進；若要觀察，等價格回到 {buy2:.2f} 附近再看。"

    return {
        "zh": zh, "cls": cls, "one_line": one_line,
        "buy1": buy1, "buy2": buy2, "chase_limit": chase_limit,
        "stop_loss": stop_loss, "sell1": sell1, "sell2": sell2,
        "confidence": confidence
    }

def decision_with_position(df, score, cost, shares):
    latest = df.iloc[-1]
    close, ma20, psar, atr_v, bb_upper = latest["Close"], latest["MA20"], latest["PSAR"], latest["ATR"], latest["BB_upper"]

    ret_pct = (close - cost) / cost * 100 if cost > 0 else np.nan
    profit_value = (close - cost) * shares * 1000 if shares > 0 else np.nan

    stop_loss = min(psar, ma20 - 0.6 * atr_v, close - 1.3 * atr_v)
    if stop_loss >= close:
        stop_loss = close - 1.2 * atr_v

    take_profit1 = max(bb_upper, cost * 1.06, close + 1.2 * atr_v)
    take_profit2 = max(cost * 1.10, close + 2.2 * atr_v)
    final_exit = max(cost * 1.15, close + 3.0 * atr_v)

    confidence = max(5, min(95, 45 + (score - 50) * 0.7))

    if close <= stop_loss or score < 45:
        zh, cls = "🔴 賣出 / 停損", "sell"
        one_line = f"目前已接近或跌破停損區，建議跌破 {stop_loss:.2f} 就停損離場。"
    elif close >= take_profit2 or latest["RSI"] > 75:
        zh, cls = "🟠 分批停利", "wait"
        one_line = f"目前已有停利條件，可考慮在 {take_profit1:.2f} 先賣一部分，{take_profit2:.2f} 再賣一部分。"
    elif score >= 60 and close > psar:
        zh, cls = "🔵 續抱", "hold"
        one_line = f"目前仍偏多，可續抱；跌破 {stop_loss:.2f} 再重新評估。"
    else:
        zh, cls = "🟡 保守續抱 / 不加碼", "wait"
        one_line = f"訊號普通，先不加碼；若跌破 {stop_loss:.2f} 應減碼或停損。"

    return {
        "zh": zh, "cls": cls, "one_line": one_line,
        "ret_pct": ret_pct, "profit_value": profit_value,
        "stop_loss": stop_loss,
        "take_profit1": take_profit1,
        "take_profit2": take_profit2,
        "final_exit": final_exit,
        "confidence": confidence
    }

def backtest_psar(df):
    d = df.copy()
    d["position"] = (d["Close"] > d["PSAR"]).astype(int).shift(1).fillna(0)
    d["ret"] = d["Close"].pct_change().fillna(0)
    d["strategy_ret"] = d["position"] * d["ret"]

    total_strategy = (1 + d["strategy_ret"]).prod() - 1
    total_hold = d["Close"].iloc[-1] / d["Close"].iloc[0] - 1

    trades = []
    pos = 0
    entry = None
    for i in range(1, len(d)):
        signal = 1 if d["Close"].iloc[i] > d["PSAR"].iloc[i] else 0
        if pos == 0 and signal == 1:
            pos = 1
            entry = d["Close"].iloc[i]
        elif pos == 1 and signal == 0:
            exit_price = d["Close"].iloc[i]
            trades.append((exit_price - entry) / entry)
            pos = 0
            entry = None
    if pos == 1 and entry is not None:
        trades.append((d["Close"].iloc[-1] - entry) / entry)

    wins = [x for x in trades if x > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else np.nan
    avg_trade = np.mean(trades) * 100 if trades else np.nan

    equity = (1 + d["strategy_ret"]).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1).min() * 100

    signals = d.copy()
    signals["Buy_Signal"] = ((signals["Close"] > signals["PSAR"]) & (signals["Close"].shift(1) <= signals["PSAR"].shift(1))).astype(int)
    signals["Sell_Signal"] = ((signals["Close"] < signals["PSAR"]) & (signals["Close"].shift(1) >= signals["PSAR"].shift(1))).astype(int)

    return {
        "strategy_return": total_strategy * 100,
        "buy_hold_return": total_hold * 100,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "max_drawdown": drawdown,
        "trades": len(trades),
        "equity": equity,
        "signals": signals
    }

# =============================
# Sidebar
# =============================
st.sidebar.header("股票與持股設定")
raw_ticker = st.sidebar.text_input("股票代號", value="2633")
mode = st.sidebar.radio("我目前是否持有？", ["尚未持有", "已持有"], index=0)
cost = 0.0
shares = 0.0
if mode == "已持有":
    cost = st.sidebar.number_input("持有成本價", min_value=0.0, value=25.60, step=0.05)
    shares = st.sidebar.number_input("持有張數", min_value=0.0, value=1.0, step=1.0)

st.sidebar.markdown("---")
timeframe = st.sidebar.selectbox("分析時間週期", ["日線 1D", "小時線 1H"], index=0)
if timeframe == "日線 1D":
    interval, period = "1d", st.sidebar.selectbox("回測時間", ["1y", "2y", "5y"], index=2)
else:
    interval, period = "60m", st.sidebar.selectbox("回測時間", ["1mo", "3mo", "6mo", "1y"], index=1)

page = st.sidebar.radio("頁面", ["我的股票", "策略回測", "技術細節"], index=0)
run = st.sidebar.button("開始分析", type="primary")
st.sidebar.caption("V5 Final 不使用 sklearn，因此 Streamlit 部署較穩定。")

# =============================
# Main
# =============================
if run or raw_ticker:
    ticker, stock_code = normalize_ticker(raw_ticker)

    with st.spinner("正在抓取股價資料..."):
        price = fetch_price_data(ticker, period, interval)

    if price.empty or len(price) < 80:
        st.error("資料不足，請換較長回測時間，或確認股票代號是否正確。")
        st.stop()

    df = prepare_indicators(price)
    fundamentals = fetch_fundamentals(ticker)
    chip = fetch_twse_institutional(stock_code)

    latest = df.iloc[-1]
    score, reasons_buy, reasons_sell = score_market(df, chip)
    bt = backtest_psar(df)

    st.markdown(f"## {fundamentals.get('Company', ticker)} ({ticker})")
    st.caption(f"分析週期：{timeframe}｜最後資料時間：{df.index[-1]}")

    if page == "策略回測":
        st.markdown('<div class="section-title">PSAR 策略回測</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PSAR策略報酬", f"{bt['strategy_return']:.1f}%")
        c2.metric("買進持有報酬", f"{bt['buy_hold_return']:.1f}%")
        c3.metric("勝率", f"{bt['win_rate']:.1f}%" if not np.isnan(bt["win_rate"]) else "N/A")
        c4.metric("平均每筆", f"{bt['avg_trade']:.1f}%" if not np.isnan(bt["avg_trade"]) else "N/A")
        c5.metric("最大回撤", f"{bt['max_drawdown']:.1f}%")
        st.line_chart(bt["equity"])
        st.dataframe(bt["signals"][["Close", "PSAR", "Buy_Signal", "Sell_Signal"]].tail(120).round(3), use_container_width=True)
        st.info("Buy_Signal=1：價格突破 PSAR 買進；Sell_Signal=1：價格跌破 PSAR 賣出。")

    elif page == "技術細節":
        tech = pd.DataFrame([{
            "Close": latest["Close"],
            "PSAR": latest["PSAR"],
            "MA20": latest["MA20"],
            "MA60": latest["MA60"],
            "RSI": latest["RSI"],
            "K": latest["K"],
            "D": latest["D"],
            "MACD": latest["MACD"],
            "MACD signal": latest["MACD_signal"],
            "ATR": latest["ATR"],
            "Rule score": score,
        }])
        st.dataframe(tech.round(3), hide_index=True, use_container_width=True)
        st.line_chart(df[["Close", "PSAR", "MA20", "MA60", "BB_upper", "BB_mid", "BB_lower"]].tail(180))
        st.line_chart(df[["MACD", "MACD_signal", "MACD_hist"]].tail(180))
        st.line_chart(df[["RSI", "K", "D"]].tail(180))

        st.subheader("基本面")
        st.dataframe(pd.DataFrame([fundamentals]).round(3), hide_index=True, use_container_width=True)
        st.subheader("籌碼面")
        st.dataframe(pd.DataFrame([chip]), hide_index=True, use_container_width=True)

    else:
        if mode == "尚未持有":
            decision = decision_without_position(df, score)
            st.markdown(f"""
            <div class="big-signal {decision['cls']}">
                <div class="signal-title">今日明確建議</div>
                <div class="signal-main">{decision['zh']}</div>
                <div class="signal-sub">{decision['one_line']}</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="price-card"><div class="price-label">目前價格</div><div class="price-value">{latest["Close"]:.2f}</div><div class="price-note">最新收盤/週期價格</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="price-card"><div class="price-label">第一買點</div><div class="price-value">{decision["buy1"]:.2f}</div><div class="price-note">接近支撐，可觀察</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="price-card"><div class="price-label">第二買點</div><div class="price-value">{decision["buy2"]:.2f}</div><div class="price-note">更保守的拉回價</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="price-card"><div class="price-label">不要追高超過</div><div class="price-value">{decision["chase_limit"]:.2f}</div><div class="price-note">高於此價風險較高</div></div>', unsafe_allow_html=True)

            c5, c6, c7, c8 = st.columns(4)
            c5.markdown(f'<div class="price-card"><div class="price-label">停損價格</div><div class="price-value">{decision["stop_loss"]:.2f}</div><div class="price-note">跌破應重新評估</div></div>', unsafe_allow_html=True)
            c6.markdown(f'<div class="price-card"><div class="price-label">第一賣點</div><div class="price-value">{decision["sell1"]:.2f}</div><div class="price-note">可分批停利</div></div>', unsafe_allow_html=True)
            c7.markdown(f'<div class="price-card"><div class="price-label">第二賣點</div><div class="price-value">{decision["sell2"]:.2f}</div><div class="price-note">較積極目標</div></div>', unsafe_allow_html=True)
            c8.markdown(f'<div class="price-card"><div class="price-label">信心分數</div><div class="price-value">{decision["confidence"]:.0f}%</div><div class="price-note">規則型估計</div></div>', unsafe_allow_html=True)

            chart = df[["Close", "PSAR", "MA20", "MA60", "BB_upper", "BB_mid", "BB_lower"]].tail(180).copy()
            chart["Buy 1"] = decision["buy1"]
            chart["Buy 2"] = decision["buy2"]
            chart["Stop loss"] = decision["stop_loss"]
            chart["Sell 1"] = decision["sell1"]
            chart["Sell 2"] = decision["sell2"]

        else:
            decision = decision_with_position(df, score, cost, shares)
            st.markdown(f"""
            <div class="big-signal {decision['cls']}">
                <div class="signal-title">我的持股建議</div>
                <div class="signal-main">{decision['zh']}</div>
                <div class="signal-sub">{decision['one_line']}</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="price-card"><div class="price-label">目前價格</div><div class="price-value">{latest["Close"]:.2f}</div><div class="price-note">最新收盤/週期價格</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="price-card"><div class="price-label">持有成本</div><div class="price-value">{cost:.2f}</div><div class="price-note">你輸入的成本</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="price-card"><div class="price-label">目前報酬</div><div class="price-value">{decision["ret_pct"]:.2f}%</div><div class="price-note">未實現報酬率</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="price-card"><div class="price-label">估計損益</div><div class="price-value">{decision["profit_value"]:,.0f}</div><div class="price-note">以每張1000股估算</div></div>', unsafe_allow_html=True)

            c5, c6, c7, c8 = st.columns(4)
            c5.markdown(f'<div class="price-card"><div class="price-label">停損賣出</div><div class="price-value">{decision["stop_loss"]:.2f}</div><div class="price-note">跌破就要小心</div></div>', unsafe_allow_html=True)
            c6.markdown(f'<div class="price-card"><div class="price-label">第一停利賣點</div><div class="price-value">{decision["take_profit1"]:.2f}</div><div class="price-note">可先賣 30%</div></div>', unsafe_allow_html=True)
            c7.markdown(f'<div class="price-card"><div class="price-label">第二停利賣點</div><div class="price-value">{decision["take_profit2"]:.2f}</div><div class="price-note">可再賣 30%</div></div>', unsafe_allow_html=True)
            c8.markdown(f'<div class="price-card"><div class="price-label">信心分數</div><div class="price-value">{decision["confidence"]:.0f}%</div><div class="price-note">規則型估計</div></div>', unsafe_allow_html=True)

            chart = df[["Close", "PSAR", "MA20", "MA60", "BB_upper", "BB_mid", "BB_lower"]].tail(180).copy()
            chart["Cost"] = cost
            chart["Stop loss"] = decision["stop_loss"]
            chart["Take profit 1"] = decision["take_profit1"]
            chart["Take profit 2"] = decision["take_profit2"]

        st.markdown('<div class="section-title">交易圖</div>', unsafe_allow_html=True)
        st.line_chart(chart)

        st.markdown('<div class="section-title">為什麼系統這樣判斷？</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown('<div class="reason-box"><b>偏多理由</b>', unsafe_allow_html=True)
            if reasons_buy:
                for r in reasons_buy[:6]:
                    st.write("✅ " + r)
            else:
                st.write("目前偏多理由不足。")
            st.markdown('</div>', unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="reason-box"><b>風險理由</b>', unsafe_allow_html=True)
            if reasons_sell:
                for r in reasons_sell[:6]:
                    st.write("⚠️ " + r)
            else:
                st.write("目前沒有明顯風險訊號。")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">回測摘要</div>', unsafe_allow_html=True)
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("PSAR策略報酬", f"{bt['strategy_return']:.1f}%")
        b2.metric("買進持有報酬", f"{bt['buy_hold_return']:.1f}%")
        b3.metric("勝率", f"{bt['win_rate']:.1f}%" if not np.isnan(bt["win_rate"]) else "N/A")
        b4.metric("平均每筆", f"{bt['avg_trade']:.1f}%" if not np.isnan(bt["avg_trade"]) else "N/A")
        b5.metric("最大回撤", f"{bt['max_drawdown']:.1f}%")

    st.markdown("""
    <div class="alert-box">
    <b>重要提醒：</b>這個 APP 是教學與研究用的交易決策輔助工具，不是投資建議，也不保證獲利。
    買點、賣點、停損與信心分數都是根據歷史價格與技術規則推估，實際交易仍要自行承擔風險。
    </div>
    """, unsafe_allow_html=True)
