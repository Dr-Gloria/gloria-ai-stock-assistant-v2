
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from io import StringIO

st.set_page_config(
    page_title="Gloria AI Stock Assistant v2",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Gloria AI Stock Assistant v2")
st.caption("整合基本面、技術面、籌碼面與趨勢面，產生 AI-style 綜合評分。僅供研究與教學參考，不構成投資建議。")

# -----------------------------
# Utility
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

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SAR"] = parabolic_sar(df)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    df["K"], df["D"] = stochastic_kd(df)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = bollinger_bands(df["Close"])
    df["Return_20D"] = df["Close"].pct_change(20) * 100
    df["Volume_MA20"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan
    return df.dropna()

# -----------------------------
# Fundamental data
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
    div_yield = safe_float(info.get("dividendYield"))

    # yfinance often returns ROE and dividendYield in decimal form.
    if not np.isnan(roe):
        roe = roe * 100
    if not np.isnan(div_yield):
        div_yield = div_yield * 100

    return {
        "EPS": eps,
        "PE": pe,
        "ROE (%)": roe,
        "Dividend Yield (%)": div_yield,
        "Company": info.get("longName") or info.get("shortName") or ticker,
        "Sector": info.get("sector", "N/A"),
    }

# -----------------------------
# Taiwan chip data
# -----------------------------
def roc_date(dt: datetime) -> str:
    return f"{dt.year - 1911}/{dt.month:02d}/{dt.day:02d}"

@st.cache_data(ttl=3600)
def fetch_twse_institutional(stock_code: str) -> dict:
    """
    Try to fetch recent TWSE institutional data.
    This uses public TWSE API and may fail if the stock is OTC, market holiday, or API changes.
    """
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
        dt = datetime.today() - timedelta(days=days_back)
        date_str = dt.strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json"
        try:
            r = requests.get(url, timeout=8)
            data = r.json()
            fields = data.get("fields", [])
            rows = data.get("data", [])

            if not rows or not fields:
                continue

            df = pd.DataFrame(rows, columns=fields)
            code_col = "證券代號"
            row = df[df[code_col].astype(str).str.strip() == stock_code]
            if row.empty:
                continue

            row = row.iloc[0]

            def parse_num(v):
                try:
                    return float(str(v).replace(",", "").replace("--", "0"))
                except Exception:
                    return np.nan

            # Common TWSE columns
            foreign = parse_num(row.get("外陸資買賣超股數(不含外資自營商)", np.nan))
            trust = parse_num(row.get("投信買賣超股數", np.nan))
            dealer = parse_num(row.get("自營商買賣超股數", np.nan))
            total = parse_num(row.get("三大法人買賣超股數", np.nan))

            result = {
                "foreign_net": foreign,
                "investment_trust_net": trust,
                "dealer_net": dealer,
                "total_institution_net": total,
                "source_note": f"TWSE institutional data date: {date_str}"
            }
            return result
        except Exception:
            continue

    return result

@st.cache_data(ttl=3600)
def fetch_margin_data(stock_code: str) -> dict:
    """
    Try to fetch TWSE margin trading data.
    """
    result = {
        "margin_balance": np.nan,
        "short_balance": np.nan,
        "margin_short_note": "No margin/short data available."
    }

    if not stock_code.isdigit():
        return result

    for days_back in range(1, 15):
        dt = datetime.today() - timedelta(days=days_back)
        date_str = dt.strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={date_str}&selectType=ALL&response=json"
        try:
            r = requests.get(url, timeout=8)
            data = r.json()
            fields = data.get("fields", [])
            rows = data.get("data", [])
            if not rows or not fields:
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

            result = {
                "margin_balance": parse_num(row.get("融資今日餘額", np.nan)),
                "short_balance": parse_num(row.get("融券今日餘額", np.nan)),
                "margin_short_note": f"TWSE margin data date: {date_str}"
            }
            return result
        except Exception:
            continue

    return result

# -----------------------------
# Scoring
# -----------------------------
def score_fundamentals(f: dict) -> tuple[int, list[str]]:
    score = 0
    notes = []

    eps = f.get("EPS", np.nan)
    pe = f.get("PE", np.nan)
    roe = f.get("ROE (%)", np.nan)
    dy = f.get("Dividend Yield (%)", np.nan)

    if not np.isnan(eps) and eps > 0:
        score += 6
        notes.append("EPS 為正，獲利能力加分。")
    else:
        notes.append("EPS 資料不足或為負。")

    if not np.isnan(pe):
        if 8 <= pe <= 25:
            score += 7
            notes.append("本益比落在相對合理區間。")
        elif 0 < pe < 8:
            score += 5
            notes.append("本益比較低，但需確認是否為景氣循環或一次性因素。")
        elif 25 < pe <= 40:
            score += 3
            notes.append("本益比較高，可能反映成長期待，也可能偏貴。")
        else:
            notes.append("本益比偏極端或資料不足。")
    else:
        notes.append("本益比資料不足。")

    if not np.isnan(roe):
        if roe >= 15:
            score += 7
            notes.append("ROE 高於 15%，股東權益報酬率表現佳。")
        elif roe >= 8:
            score += 4
            notes.append("ROE 尚可。")
        else:
            notes.append("ROE 偏低或資料不足。")
    else:
        notes.append("ROE 資料不足。")

    if not np.isnan(dy):
        if dy >= 3:
            score += 5
            notes.append("殖利率具一定吸引力。")
        elif dy > 0:
            score += 3
            notes.append("有配息，但殖利率普通。")
        else:
            notes.append("殖利率偏低。")
    else:
        notes.append("殖利率資料不足。")

    return min(score, 25), notes

def score_technicals(df: pd.DataFrame) -> tuple[int, list[str]]:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    notes = []

    if latest["Close"] > latest["SAR"]:
        score += 4
        notes.append("SAR 位於價格下方，短線偏多。")
    else:
        notes.append("SAR 位於價格上方，短線偏弱。")

    if latest["MA20"] > latest["MA60"]:
        score += 5
        notes.append("MA20 高於 MA60，中期趨勢偏多。")
    else:
        notes.append("MA20 低於 MA60，中期趨勢偏弱。")

    if 45 <= latest["RSI"] <= 70:
        score += 4
        notes.append("RSI 位於偏多但未明顯過熱區。")
    elif latest["RSI"] > 75:
        score += 1
        notes.append("RSI 偏高，短線可能過熱。")
    elif latest["RSI"] < 35:
        score += 1
        notes.append("RSI 偏弱，需觀察是否止跌。")
    else:
        score += 2
        notes.append("RSI 中性。")

    if latest["MACD"] > latest["MACD_signal"]:
        score += 5
        notes.append("MACD 在訊號線上方，動能偏多。")
    else:
        notes.append("MACD 在訊號線下方，動能偏弱。")

    if latest["K"] > latest["D"] and latest["K"] < 85:
        score += 3
        notes.append("KD 偏多且尚未嚴重過熱。")
    elif latest["K"] > 85:
        score += 1
        notes.append("KD 偏高，短線可能震盪。")
    else:
        notes.append("KD 尚未轉強。")

    if latest["Close"] > latest["BB_mid"] and latest["Close"] < latest["BB_upper"]:
        score += 4
        notes.append("價格位於布林中軌上方，趨勢健康。")
    elif latest["Close"] >= latest["BB_upper"]:
        score += 2
        notes.append("價格接近或突破布林上軌，偏強但可能過熱。")
    else:
        notes.append("價格在布林中軌下方，技術面較保守。")

    return min(score, 25), notes

def score_chips(chip: dict, margin: dict) -> tuple[int, list[str]]:
    score = 0
    notes = []

    total = chip.get("total_institution_net", np.nan)
    foreign = chip.get("foreign_net", np.nan)
    trust = chip.get("investment_trust_net", np.nan)
    dealer = chip.get("dealer_net", np.nan)

    if not np.isnan(total):
        if total > 0:
            score += 8
            notes.append("三大法人買超，籌碼偏多。")
        else:
            notes.append("三大法人賣超，籌碼偏弱。")
    else:
        notes.append("三大法人資料不足。")

    if not np.isnan(foreign):
        if foreign > 0:
            score += 6
            notes.append("外資買超。")
        else:
            notes.append("外資賣超。")
    else:
        notes.append("外資資料不足。")

    if not np.isnan(trust):
        if trust > 0:
            score += 5
            notes.append("投信買超。")
        else:
            notes.append("投信未買超。")
    else:
        notes.append("投信資料不足。")

    if not np.isnan(dealer):
        if dealer > 0:
            score += 3
            notes.append("自營商買超。")
        else:
            notes.append("自營商未買超。")
    else:
        notes.append("自營商資料不足。")

    margin_balance = margin.get("margin_balance", np.nan)
    short_balance = margin.get("short_balance", np.nan)

    if not np.isnan(margin_balance) and not np.isnan(short_balance):
        if short_balance > 0 and margin_balance > 0:
            score += 3
            notes.append("融資融券資料可用，但需搭配變化量判斷，暫給中性分。")
    else:
        notes.append("融資融券資料不足。")

    return min(score, 25), notes

def score_trend(df: pd.DataFrame) -> tuple[int, list[str]]:
    latest = df.iloc[-1]
    score = 0
    notes = []

    if latest["Return_20D"] > 8:
        score += 8
        notes.append("近20日漲幅明顯，趨勢強。")
    elif latest["Return_20D"] > 0:
        score += 5
        notes.append("近20日為正報酬，趨勢偏多。")
    else:
        notes.append("近20日報酬為負，趨勢偏弱。")

    if latest["Close"] > latest["MA20"] > latest["MA60"]:
        score += 8
        notes.append("價格、MA20、MA60 呈多頭排列。")
    elif latest["Close"] > latest["MA60"]:
        score += 5
        notes.append("價格仍在 MA60 上方。")
    else:
        notes.append("價格在 MA60 下方，趨勢保守。")

    if "Volume" in df.columns and not np.isnan(latest.get("Volume_MA20", np.nan)):
        if latest["Volume"] > latest["Volume_MA20"] and latest["Close"] > df.iloc[-2]["Close"]:
            score += 5
            notes.append("價漲量增，趨勢加分。")
        elif latest["Volume"] < latest["Volume_MA20"]:
            score += 2
            notes.append("量能未明顯放大。")
        else:
            score += 3
            notes.append("量能中性。")

    if latest["MACD_hist"] > df.iloc[-2]["MACD_hist"]:
        score += 4
        notes.append("MACD histogram 改善，動能增強。")
    else:
        notes.append("MACD histogram 未改善。")

    return min(score, 25), notes

def recommendation(total_score: int) -> str:
    if total_score >= 80:
        return "🟢 Strong Buy / 強烈偏多"
    elif total_score >= 70:
        return "🟢 Buy / 偏多"
    elif total_score >= 55:
        return "🟡 Hold / 觀望或續抱"
    elif total_score >= 40:
        return "🟠 Weak / 偏弱，保守觀察"
    else:
        return "🔴 Sell or Avoid / 偏空或暫避"

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("輸入設定")
raw_ticker = st.sidebar.text_input("股票代號", value="2317", help="台股可輸入 2317；美股可輸入 AAPL、NVDA、TSLA。")
period = st.sidebar.selectbox("股價資料期間", ["6mo", "1y", "2y", "5y"], index=2)
run = st.sidebar.button("開始分析", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption("台股會自動轉為 Yahoo Finance 格式，例如 2317 → 2317.TW。")

if run or raw_ticker:
    ticker, stock_code = normalize_ticker(raw_ticker)

    with st.spinner("抓取股價與計算指標中..."):
        price = fetch_price_data(ticker, period=period)

    if price.empty or len(price) < 80:
        st.error("抓不到足夠股價資料。請確認股票代號是否正確，或改用較長期間。")
        st.stop()

    df = prepare_indicators(price)

    with st.spinner("抓取基本面與籌碼面資料中..."):
        fundamentals = fetch_fundamentals(ticker)
        chip = fetch_twse_institutional(stock_code)
        margin = fetch_margin_data(stock_code)

    fundamental_score, fundamental_notes = score_fundamentals(fundamentals)
    technical_score, technical_notes = score_technicals(df)
    chip_score, chip_notes = score_chips(chip, margin)
    trend_score, trend_notes = score_trend(df)

    total_score = fundamental_score + technical_score + chip_score + trend_score
    rec = recommendation(total_score)
    latest = df.iloc[-1]

    st.subheader(f"{fundamentals.get('Company', ticker)} ({ticker})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盤價", f"{latest['Close']:.2f}")
    c2.metric("總分", f"{total_score} / 100")
    c3.metric("AI 建議", rec)
    c4.metric("RSI", f"{latest['RSI']:.1f}")

    st.subheader("AI 評分總表")
    score_df = pd.DataFrame([
        {"項目": "基本面", "分數": f"{fundamental_score}/25"},
        {"項目": "技術面", "分數": f"{technical_score}/25"},
        {"項目": "籌碼面", "分數": f"{chip_score}/25"},
        {"項目": "趨勢面", "分數": f"{trend_score}/25"},
        {"項目": "總分", "分數": f"{total_score}/100"},
        {"項目": "建議", "分數": rec},
    ])
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 總結", "📊 技術面", "🏢 基本面", "💰 籌碼面", "📄 原始資料"])

    with tab1:
        st.markdown("### 系統解讀")
        st.write(f"目前系統評分為 **{total_score}/100**，判斷為：**{rec}**。")

        st.markdown("#### 基本面重點")
        for n in fundamental_notes:
            st.write("- " + n)

        st.markdown("#### 技術面重點")
        for n in technical_notes:
            st.write("- " + n)

        st.markdown("#### 籌碼面重點")
        for n in chip_notes:
            st.write("- " + n)

        st.markdown("#### 趨勢面重點")
        for n in trend_notes:
            st.write("- " + n)

    with tab2:
        st.markdown("### 技術指標")
        tech_latest = pd.DataFrame([{
            "Date": df.index[-1].strftime("%Y-%m-%d"),
            "Close": latest["Close"],
            "SAR": latest["SAR"],
            "MA20": latest["MA20"],
            "MA60": latest["MA60"],
            "RSI": latest["RSI"],
            "K": latest["K"],
            "D": latest["D"],
            "MACD": latest["MACD"],
            "MACD Signal": latest["MACD_signal"],
            "BB Upper": latest["BB_upper"],
            "BB Mid": latest["BB_mid"],
            "BB Lower": latest["BB_lower"],
        }])
        st.dataframe(tech_latest.round(3), use_container_width=True, hide_index=True)

        st.markdown("### 價格、SAR、均線、布林通道")
        chart_cols = ["Close", "SAR", "MA20", "MA60", "BB_upper", "BB_mid", "BB_lower"]
        st.line_chart(df[chart_cols].tail(180))

        st.markdown("### MACD")
        st.line_chart(df[["MACD", "MACD_signal", "MACD_hist"]].tail(180))

        st.markdown("### RSI / KD")
        st.line_chart(df[["RSI", "K", "D"]].tail(180))

    with tab3:
        st.markdown("### 基本面")
        fdf = pd.DataFrame([fundamentals])
        st.dataframe(fdf.round(3), use_container_width=True, hide_index=True)
        st.info("基本面資料主要來自 yfinance。不同市場與股票的資料完整度可能不同。")

    with tab4:
        st.markdown("### 三大法人")
        chip_df = pd.DataFrame([{
            "外資買賣超股數": chip.get("foreign_net", np.nan),
            "投信買賣超股數": chip.get("investment_trust_net", np.nan),
            "自營商買賣超股數": chip.get("dealer_net", np.nan),
            "三大法人買賣超股數": chip.get("total_institution_net", np.nan),
            "資料來源說明": chip.get("source_note", "")
        }])
        st.dataframe(chip_df, use_container_width=True, hide_index=True)

        st.markdown("### 融資融券")
        margin_df = pd.DataFrame([{
            "融資餘額": margin.get("margin_balance", np.nan),
            "融券餘額": margin.get("short_balance", np.nan),
            "資料來源說明": margin.get("margin_short_note", "")
        }])
        st.dataframe(margin_df, use_container_width=True, hide_index=True)

        st.warning("籌碼面目前以 TWSE 上市股票為主。若為上櫃股票、美股，或遇到休市/API調整，可能顯示資料不足。")

    with tab5:
        st.markdown("### 最近 60 筆指標資料")
        st.dataframe(df.tail(60).round(3), use_container_width=True)

    st.markdown("---")
    st.warning("免責聲明：本系統是研究與教學用的決策輔助工具，不是投資顧問，也不保證獲利。實際投資需自行判斷風險。")
