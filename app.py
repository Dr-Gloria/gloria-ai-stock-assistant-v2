
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Gloria AI Stock Assistant v3",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f8fb;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(135deg, #1f2937 0%, #374151 45%, #6b4f3f 100%);
    padding: 26px 30px;
    border-radius: 24px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(31,41,55,0.18);
}
.hero h1 {
    margin: 0;
    font-size: 34px;
}
.hero p {
    margin-top: 10px;
    font-size: 16px;
    opacity: 0.9;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 6px 18px rgba(15,23,42,0.07);
    border: 1px solid #eef0f4;
    min-height: 120px;
}
.signal-buy {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border-left: 8px solid #16a34a;
}
.signal-hold {
    background: linear-gradient(135deg, #fef9c3 0%, #fde68a 100%);
    border-left: 8px solid #ca8a04;
}
.signal-sell {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border-left: 8px solid #dc2626;
}
.metric-title {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
}
.small-note {
    color: #6b7280;
    font-size: 13px;
}
.trade-box {
    background: #ffffff;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 16px rgba(15,23,42,0.06);
}
.trade-label {
    font-size: 13px;
    color: #6b7280;
}
.trade-value {
    font-size: 24px;
    font-weight: 800;
    color: #111827;
}
.warning-box {
    background: #fff7ed;
    border-left: 6px solid #f97316;
    padding: 16px 18px;
    border-radius: 14px;
    color: #7c2d12;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>📈 Gloria AI Stock Assistant v3</h1>
<p>整合基本面、技術面、籌碼面、趨勢面，並新增交易模式：買點、停損、停利、賣點與行動建議。</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def normalize_ticker(raw: str) -> tuple[str, str]:
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

@st.cache_data(ttl=900)
def fetch_price_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["High", "Low", "Close"])
    return df

# -----------------------------
# Technical indicators
# -----------------------------
def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
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

    return pd.Series(sar, index=df.index, name="SAR")

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return value.fillna(50).rename("RSI")

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.rename("MACD"), signal_line.rename("MACD_signal"), hist.rename("MACD_hist")

def stochastic_kd(df: pd.DataFrame, n: int = 9):
    low_min = df["Low"].rolling(n).min()
    high_max = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    return k.rename("K"), d.rename("D")

def bollinger_bands(series: pd.Series, window: int = 20, n_std: int = 2):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper.rename("BB_upper"), mid.rename("BB_mid"), lower.rename("BB_lower")

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean().rename("ATR")

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SAR"] = parabolic_sar(df)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    df["K"], df["D"] = stochastic_kd(df)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = bollinger_bands(df["Close"])
    df["ATR"] = atr(df)
    df["Return_20D"] = df["Close"].pct_change(20) * 100
    df["Volume_MA20"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan
    return df.dropna()

# -----------------------------
# Fundamental and chip data
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception:
        info = {}

    eps = safe_float(info.get("trailingEps"))
    pe = safe_float(info.get("trailingPE"))
    roe = safe_float(info.get("returnOnEquity"))
    dy = safe_float(info.get("dividendYield"))

    if not np.isnan(roe):
        roe *= 100
    if not np.isnan(dy):
        dy *= 100

    return {
        "EPS": eps,
        "PE": pe,
        "ROE (%)": roe,
        "Dividend Yield (%)": dy,
        "Company": info.get("longName") or info.get("shortName") or ticker,
        "Sector": info.get("sector", "N/A"),
    }

@st.cache_data(ttl=3600)
def fetch_twse_institutional(stock_code: str) -> dict:
    result = {
        "foreign_net": np.nan,
        "investment_trust_net": np.nan,
        "dealer_net": np.nan,
        "total_institution_net": np.nan,
        "source_note": "No institutional data available."
    }

    if not stock_code.isdigit():
        return result

    for days_back in range(1, 15):
        date_str = (datetime.today() - timedelta(days=days_back)).strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json"
        try:
            r = requests.get(url, timeout=8)
            data = r.json()
            fields = data.get("fields", [])
            rows = data.get("data", [])
            if not fields or not rows:
                continue
            df = pd.DataFrame(rows, columns=fields)
            row = df[df["證券代號"].astype(str).str.strip() == stock_code]
            if row.empty:
                continue
            row = row.iloc[0]

            def parse_num(v):
                try:
                    return float(str(v).replace(",", "").replace("--", "0"))
                except Exception:
                    return np.nan

            return {
                "foreign_net": parse_num(row.get("外陸資買賣超股數(不含外資自營商)", np.nan)),
                "investment_trust_net": parse_num(row.get("投信買賣超股數", np.nan)),
                "dealer_net": parse_num(row.get("自營商買賣超股數", np.nan)),
                "total_institution_net": parse_num(row.get("三大法人買賣超股數", np.nan)),
                "source_note": f"TWSE institutional data date: {date_str}"
            }
        except Exception:
            continue

    return result

@st.cache_data(ttl=3600)
def fetch_margin_data(stock_code: str) -> dict:
    result = {
        "margin_balance": np.nan,
        "short_balance": np.nan,
        "margin_short_note": "No margin/short data available."
    }

    if not stock_code.isdigit():
        return result

    for days_back in range(1, 15):
        date_str = (datetime.today() - timedelta(days=days_back)).strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={date_str}&selectType=ALL&response=json"
        try:
            r = requests.get(url, timeout=8)
            data = r.json()
            fields = data.get("fields", [])
            rows = data.get("data", [])
            if not fields or not rows:
                continue
            df = pd.DataFrame(rows, columns=fields)
            row = df[df["股票代號"].astype(str).str.strip() == stock_code]
            if row.empty:
                continue
            row = row.iloc[0]

            def parse_num(v):
                try:
                    return float(str(v).replace(",", "").replace("--", "0"))
                except Exception:
                    return np.nan

            return {
                "margin_balance": parse_num(row.get("融資今日餘額", np.nan)),
                "short_balance": parse_num(row.get("融券今日餘額", np.nan)),
                "margin_short_note": f"TWSE margin data date: {date_str}"
            }
        except Exception:
            continue

    return result

# -----------------------------
# Scoring
# -----------------------------
def score_fundamentals(f):
    score, notes = 0, []
    eps, pe, roe, dy = f["EPS"], f["PE"], f["ROE (%)"], f["Dividend Yield (%)"]

    if not np.isnan(eps) and eps > 0:
        score += 6; notes.append("EPS 為正，獲利能力加分。")
    else:
        notes.append("EPS 資料不足或為負。")

    if not np.isnan(pe):
        if 8 <= pe <= 25:
            score += 7; notes.append("本益比落在相對合理區間。")
        elif 0 < pe < 8:
            score += 5; notes.append("本益比較低，但需確認是否為景氣循環或一次性因素。")
        elif 25 < pe <= 40:
            score += 3; notes.append("本益比較高，可能反映成長期待，也可能偏貴。")
        else:
            notes.append("本益比偏極端或資料不足。")
    else:
        notes.append("本益比資料不足。")

    if not np.isnan(roe):
        if roe >= 15:
            score += 7; notes.append("ROE 高於 15%，股東權益報酬率佳。")
        elif roe >= 8:
            score += 4; notes.append("ROE 尚可。")
        else:
            notes.append("ROE 偏低。")
    else:
        notes.append("ROE 資料不足。")

    if not np.isnan(dy):
        if dy >= 3:
            score += 5; notes.append("殖利率具一定吸引力。")
        elif dy > 0:
            score += 3; notes.append("有配息，但殖利率普通。")
        else:
            notes.append("殖利率偏低。")
    else:
        notes.append("殖利率資料不足。")

    return min(score, 25), notes

def score_technicals(df):
    latest, prev = df.iloc[-1], df.iloc[-2]
    score, notes = 0, []

    if latest["Close"] > latest["SAR"]:
        score += 4; notes.append("PSAR 位於價格下方，短線偏多。")
    else:
        notes.append("PSAR 位於價格上方，短線偏弱。")

    if latest["MA20"] > latest["MA60"]:
        score += 5; notes.append("MA20 高於 MA60，中期趨勢偏多。")
    else:
        notes.append("MA20 低於 MA60，中期趨勢偏弱。")

    if 45 <= latest["RSI"] <= 70:
        score += 4; notes.append("RSI 偏多但未明顯過熱。")
    elif latest["RSI"] > 75:
        score += 1; notes.append("RSI 偏高，短線可能過熱。")
    elif latest["RSI"] < 35:
        score += 1; notes.append("RSI 偏弱，需觀察是否止跌。")
    else:
        score += 2; notes.append("RSI 中性。")

    if latest["MACD"] > latest["MACD_signal"]:
        score += 5; notes.append("MACD 在訊號線上方，動能偏多。")
    else:
        notes.append("MACD 在訊號線下方，動能偏弱。")

    if latest["K"] > latest["D"] and latest["K"] < 85:
        score += 3; notes.append("KD 偏多且尚未嚴重過熱。")
    elif latest["K"] > 85:
        score += 1; notes.append("KD 偏高，短線可能震盪。")
    else:
        notes.append("KD 尚未轉強。")

    if latest["Close"] > latest["BB_mid"] and latest["Close"] < latest["BB_upper"]:
        score += 4; notes.append("價格位於布林中軌上方，趨勢健康。")
    elif latest["Close"] >= latest["BB_upper"]:
        score += 2; notes.append("價格接近布林上軌，偏強但可能過熱。")
    else:
        notes.append("價格在布林中軌下方，技術面保守。")

    return min(score, 25), notes

def score_chips(chip, margin):
    score, notes = 0, []
    total, foreign, trust, dealer = chip["total_institution_net"], chip["foreign_net"], chip["investment_trust_net"], chip["dealer_net"]

    if not np.isnan(total):
        if total > 0:
            score += 8; notes.append("三大法人買超，籌碼偏多。")
        else:
            notes.append("三大法人賣超，籌碼偏弱。")
    else:
        notes.append("三大法人資料不足。")

    if not np.isnan(foreign):
        if foreign > 0:
            score += 6; notes.append("外資買超。")
        else:
            notes.append("外資賣超。")
    else:
        notes.append("外資資料不足。")

    if not np.isnan(trust):
        if trust > 0:
            score += 5; notes.append("投信買超。")
        else:
            notes.append("投信未買超。")
    else:
        notes.append("投信資料不足。")

    if not np.isnan(dealer):
        if dealer > 0:
            score += 3; notes.append("自營商買超。")
        else:
            notes.append("自營商未買超。")
    else:
        notes.append("自營商資料不足。")

    if not np.isnan(margin["margin_balance"]) and not np.isnan(margin["short_balance"]):
        score += 3; notes.append("融資融券資料可用，暫給中性分。")
    else:
        notes.append("融資融券資料不足。")

    return min(score, 25), notes

def score_trend(df):
    latest, prev = df.iloc[-1], df.iloc[-2]
    score, notes = 0, []

    if latest["Return_20D"] > 8:
        score += 8; notes.append("近20日漲幅明顯，趨勢強。")
    elif latest["Return_20D"] > 0:
        score += 5; notes.append("近20日為正報酬，趨勢偏多。")
    else:
        notes.append("近20日報酬為負，趨勢偏弱。")

    if latest["Close"] > latest["MA20"] > latest["MA60"]:
        score += 8; notes.append("價格、MA20、MA60 呈多頭排列。")
    elif latest["Close"] > latest["MA60"]:
        score += 5; notes.append("價格仍在 MA60 上方。")
    else:
        notes.append("價格在 MA60 下方，趨勢保守。")

    if "Volume" in df.columns and not np.isnan(latest.get("Volume_MA20", np.nan)):
        if latest["Volume"] > latest["Volume_MA20"] and latest["Close"] > prev["Close"]:
            score += 5; notes.append("價漲量增，趨勢加分。")
        elif latest["Volume"] < latest["Volume_MA20"]:
            score += 2; notes.append("量能未明顯放大。")
        else:
            score += 3; notes.append("量能中性。")

    if latest["MACD_hist"] > prev["MACD_hist"]:
        score += 4; notes.append("MACD histogram 改善，動能增強。")
    else:
        notes.append("MACD histogram 未改善。")

    return min(score, 25), notes

def recommendation(total_score):
    if total_score >= 80:
        return "🟢 Strong Buy / 強烈偏多"
    if total_score >= 70:
        return "🟢 Buy / 偏多"
    if total_score >= 55:
        return "🟡 Hold / 觀望或續抱"
    if total_score >= 40:
        return "🟠 Weak / 偏弱，保守觀察"
    return "🔴 Sell or Avoid / 偏空或暫避"

# -----------------------------
# Trading plan
# -----------------------------
def compute_trading_plan(df, total_score):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = latest["Close"]
    ma20 = latest["MA20"]
    ma60 = latest["MA60"]
    sar = latest["SAR"]
    atr_v = latest["ATR"]
    bb_upper = latest["BB_upper"]
    bb_mid = latest["BB_mid"]
    bb_lower = latest["BB_lower"]
    rsi_v = latest["RSI"]

    bullish_count = 0
    bullish_count += close > sar
    bullish_count += ma20 > ma60
    bullish_count += latest["MACD"] > latest["MACD_signal"]
    bullish_count += latest["K"] > latest["D"]
    bullish_count += close > bb_mid
    bullish_count += latest["MACD_hist"] > prev["MACD_hist"]

    if total_score >= 70 and bullish_count >= 4:
        action = "BUY"
        action_zh = "🟢 可考慮分批買進"
        css = "signal-buy"
    elif total_score < 45 or bullish_count <= 2 or close < sar:
        action = "SELL"
        action_zh = "🔴 避開或減碼"
        css = "signal-sell"
    else:
        action = "WAIT"
        action_zh = "🟡 等待更好買點"
        css = "signal-hold"

    # Buy zone
    if action == "BUY":
        buy_low = max(min(ma20, sar), close - 0.8 * atr_v)
        buy_high = min(close + 0.2 * atr_v, close * 1.015)
    else:
        buy_low = max(min(ma20, sar, bb_mid), close - 1.2 * atr_v)
        buy_high = min(max(ma20, sar, bb_mid), close * 1.01)

    # Stop loss
    stop_candidates = [sar, ma20 - 0.6 * atr_v, close - 1.5 * atr_v]
    stop_loss = min([x for x in stop_candidates if not np.isnan(x)])
    if stop_loss >= close:
        stop_loss = close - 1.2 * atr_v

    # Targets
    target1 = max(close + 1.5 * atr_v, bb_upper)
    target2 = close + 2.5 * atr_v

    # Sell / take profit zone
    if rsi_v >= 75:
        sell_note = "RSI 已偏熱，可考慮分批停利。"
    elif close >= bb_upper:
        sell_note = "價格接近布林上軌，若量能不足可分批停利。"
    elif close < sar:
        sell_note = "價格跌破 PSAR，屬於明確轉弱訊號。"
    else:
        sell_note = "尚未出現明確賣出訊號，可用停損與目標價管理部位。"

    risk_pct = (close - stop_loss) / close * 100
    reward1_pct = (target1 - close) / close * 100
    reward2_pct = (target2 - close) / close * 100
    rr1 = reward1_pct / risk_pct if risk_pct > 0 else np.nan

    return {
        "action": action,
        "action_zh": action_zh,
        "css": css,
        "buy_low": buy_low,
        "buy_high": buy_high,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "risk_pct": risk_pct,
        "reward1_pct": reward1_pct,
        "reward2_pct": reward2_pct,
        "rr1": rr1,
        "sell_note": sell_note,
        "bullish_count": bullish_count,
    }

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🔎 股票輸入")
raw_ticker = st.sidebar.text_input("股票代號", value="2633", help="台股可輸入 2633、2317、2330；美股可輸入 AAPL、NVDA。")
period = st.sidebar.selectbox("資料期間", ["6mo", "1y", "2y", "5y"], index=2)
st.sidebar.markdown("---")
st.sidebar.caption("資料來源：股價與基本面多來自 yfinance；台股法人資料嘗試讀取 TWSE 公開資料。")
run = st.sidebar.button("開始分析", type="primary")

if run or raw_ticker:
    ticker, stock_code = normalize_ticker(raw_ticker)

    with st.spinner("正在抓取股價與計算指標..."):
        price = fetch_price_data(ticker, period)

    if price.empty or len(price) < 80:
        st.error("抓不到足夠股價資料。請確認股票代號是否正確，或改用較長期間。")
        st.stop()

    df = prepare_indicators(price)

    with st.spinner("正在抓取基本面與籌碼資料..."):
        fundamentals = fetch_fundamentals(ticker)
        chip = fetch_twse_institutional(stock_code)
        margin = fetch_margin_data(stock_code)

    f_score, f_notes = score_fundamentals(fundamentals)
    t_score, t_notes = score_technicals(df)
    c_score, c_notes = score_chips(chip, margin)
    tr_score, tr_notes = score_trend(df)
    total_score = f_score + t_score + c_score + tr_score
    rec = recommendation(total_score)
    latest = df.iloc[-1]
    plan = compute_trading_plan(df, total_score)

    # Header cards
    st.markdown(f"### {fundamentals.get('Company', ticker)} ({ticker})")
    st.markdown(f"""
    <div class="card {plan['css']}">
        <div class="metric-title">今日交易判斷</div>
        <div class="metric-value">{plan['action_zh']}</div>
        <div class="small-note">總分 {total_score}/100｜多方訊號 {plan['bullish_count']}/6｜收盤價 {latest['Close']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div class="card"><div class="metric-title">最新收盤價</div><div class="metric-value">{latest["Close"]:.2f}</div><div class="small-note">Last close</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card"><div class="metric-title">AI 總分</div><div class="metric-value">{total_score}/100</div><div class="small-note">{rec}</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card"><div class="metric-title">RSI</div><div class="metric-value">{latest["RSI"]:.1f}</div><div class="small-note">45–70 較健康</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="card"><div class="metric-title">PSAR</div><div class="metric-value">{latest["SAR"]:.2f}</div><div class="small-note">跌破 PSAR 偏賣出</div></div>', unsafe_allow_html=True)

    st.markdown("## 🎯 交易計畫")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(f'<div class="trade-box"><div class="trade-label">建議買點區間</div><div class="trade-value">{plan["buy_low"]:.2f}–{plan["buy_high"]:.2f}</div><div class="small-note">接近支撐區較佳</div></div>', unsafe_allow_html=True)
    p2.markdown(f'<div class="trade-box"><div class="trade-label">停損點</div><div class="trade-value">{plan["stop_loss"]:.2f}</div><div class="small-note">風險約 {plan["risk_pct"]:.1f}%</div></div>', unsafe_allow_html=True)
    p3.markdown(f'<div class="trade-box"><div class="trade-label">第一目標價</div><div class="trade-value">{plan["target1"]:.2f}</div><div class="small-note">潛在報酬約 {plan["reward1_pct"]:.1f}%</div></div>', unsafe_allow_html=True)
    p4.markdown(f'<div class="trade-box"><div class="trade-label">第二目標價</div><div class="trade-value">{plan["target2"]:.2f}</div><div class="small-note">潛在報酬約 {plan["reward2_pct"]:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warning-box">
    <b>賣出判斷：</b>{plan['sell_note']}<br>
    <b>操作提醒：</b>若目前價格高於買點區間太多，不追價；若跌破停損點，應重新評估，不要只看 AI 分數。
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 總結", "📈 交易圖", "📊 指標分數", "💰 籌碼與基本面", "📄 原始資料"])

    with tab1:
        st.subheader("系統解讀")
        st.write(f"目前系統評分為 **{total_score}/100**，整體判斷為 **{rec}**。交易模式判斷為 **{plan['action_zh']}**。")
        st.write("這裡的買點不是保證會到的價格，而是系統依據 MA20、PSAR、布林中軌與 ATR 波動度估算出的較合理進場區。")

        st.markdown("#### 今天比較重要的訊號")
        if plan["action"] == "BUY":
            st.success("目前偏多訊號較多，可以考慮分批，而不是一次全買。")
        elif plan["action"] == "SELL":
            st.error("目前轉弱訊號較明顯，若已有持股，應注意停損或減碼。")
        else:
            st.warning("目前不是很漂亮的買點，較適合等待拉回或等待更明確突破。")

        st.markdown("#### 技術面")
        for n in t_notes:
            st.write("- " + n)

        st.markdown("#### 趨勢面")
        for n in tr_notes:
            st.write("- " + n)

    with tab2:
        chart_df = df[["Close", "SAR", "MA20", "MA60", "BB_upper", "BB_mid", "BB_lower"]].tail(180).copy()
        chart_df["Buy zone low"] = plan["buy_low"]
        chart_df["Buy zone high"] = plan["buy_high"]
        chart_df["Stop loss"] = plan["stop_loss"]
        chart_df["Target 1"] = plan["target1"]
        st.line_chart(chart_df)

        st.markdown("### MACD")
        st.line_chart(df[["MACD", "MACD_signal", "MACD_hist"]].tail(180))

        st.markdown("### RSI / KD")
        st.line_chart(df[["RSI", "K", "D"]].tail(180))

    with tab3:
        score_df = pd.DataFrame([
            {"項目": "基本面", "分數": f"{f_score}/25"},
            {"項目": "技術面", "分數": f"{t_score}/25"},
            {"項目": "籌碼面", "分數": f"{c_score}/25"},
            {"項目": "趨勢面", "分數": f"{tr_score}/25"},
            {"項目": "總分", "分數": f"{total_score}/100"},
            {"項目": "交易建議", "分數": plan["action_zh"]},
        ])
        st.dataframe(score_df, use_container_width=True, hide_index=True)

        st.markdown("#### 基本面重點")
        for n in f_notes:
            st.write("- " + n)

        st.markdown("#### 籌碼面重點")
        for n in c_notes:
            st.write("- " + n)

    with tab4:
        left, right = st.columns(2)

        with left:
            st.markdown("### 基本面")
            st.dataframe(pd.DataFrame([fundamentals]).round(3), use_container_width=True, hide_index=True)

        with right:
            st.markdown("### 籌碼面")
            chip_df = pd.DataFrame([{
                "外資買賣超股數": chip.get("foreign_net", np.nan),
                "投信買賣超股數": chip.get("investment_trust_net", np.nan),
                "自營商買賣超股數": chip.get("dealer_net", np.nan),
                "三大法人買賣超股數": chip.get("total_institution_net", np.nan),
                "資料來源說明": chip.get("source_note", "")
            }])
            st.dataframe(chip_df, use_container_width=True, hide_index=True)

            margin_df = pd.DataFrame([{
                "融資餘額": margin.get("margin_balance", np.nan),
                "融券餘額": margin.get("short_balance", np.nan),
                "資料來源說明": margin.get("margin_short_note", "")
            }])
            st.dataframe(margin_df, use_container_width=True, hide_index=True)

    with tab5:
        st.dataframe(df.tail(80).round(3), use_container_width=True)

    st.markdown("---")
    st.caption("免責聲明：本系統僅供教學與研究用，不構成投資建議，不保證獲利。股價資料可能有延遲，台股通常不是券商等級逐筆即時。")
