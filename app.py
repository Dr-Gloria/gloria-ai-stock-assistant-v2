import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Gloria AI Stock Assistant v5", page_icon="🤖", layout="wide")

st.markdown('''
<style>
.block-container{padding-top:1.2rem}.hero{background:linear-gradient(135deg,#020617,#1f2937 45%,#8b6f47);padding:28px 32px;border-radius:28px;color:white;margin-bottom:20px;box-shadow:0 12px 30px rgba(15,23,42,.24)}.hero h1{margin:0;font-size:38px}.hero p{margin-top:10px;font-size:16px;opacity:.92}.signal{padding:28px;border-radius:28px;color:#111827;box-shadow:0 10px 26px rgba(15,23,42,.1);border:1px solid rgba(255,255,255,.8);margin-bottom:16px}.buy{background:linear-gradient(135deg,#dcfce7,#86efac);border-left:10px solid #16a34a}.wait{background:linear-gradient(135deg,#fef9c3,#fde68a);border-left:10px solid #ca8a04}.sell{background:linear-gradient(135deg,#fee2e2,#fca5a5);border-left:10px solid #dc2626}.hold{background:linear-gradient(135deg,#dbeafe,#93c5fd);border-left:10px solid #2563eb}.signal-title{font-size:15px;color:#374151;font-weight:800}.signal-main{font-size:46px;font-weight:950;margin-top:6px}.signal-sub{font-size:16px;margin-top:8px;color:#374151}.card{background:white;padding:22px;border-radius:22px;box-shadow:0 8px 22px rgba(15,23,42,.08);border:1px solid #eef0f4;min-height:124px}.label{font-size:14px;color:#6b7280;font-weight:750}.value{font-size:34px;font-weight:950;color:#111827;margin-top:6px}.note{font-size:13px;color:#6b7280;margin-top:5px}.section-title{font-size:24px;font-weight:950;margin-top:24px;margin-bottom:12px}.reason-box{background:#fff;padding:18px 20px;border-radius:18px;border:1px solid #e5e7eb;box-shadow:0 5px 16px rgba(15,23,42,.06)}.alert-box{background:#fff7ed;border-left:7px solid #f97316;padding:16px 18px;border-radius:15px;color:#7c2d12;margin-top:14px}.ai-box{background:linear-gradient(135deg,#eef2ff,#dbeafe);border-left:8px solid #4f46e5;padding:20px;border-radius:20px;margin-bottom:16px}
</style>
<div class="hero"><h1>🤖 Gloria AI Stock Assistant v5</h1><p>AI 預測版：買進、賣出、停損、停利、PSAR 回測、Buy-and-Hold 比較、AI 上漲機率與股票排行榜。</p></div>
''', unsafe_allow_html=True)


def normalize_ticker(raw):
    raw = raw.strip().upper()
    return (f"{raw}.TW", raw) if raw.isdigit() else (raw, raw)

def safe_float(x, default=np.nan):
    try:
        return default if x is None else float(x)
    except Exception:
        return default

@st.cache_data(ttl=600)
def fetch_price_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["High", "Low", "Close"])


def parabolic_sar(df, step=0.02, max_step=0.2):
    high, low, n = df["High"].values, df["Low"].values, len(df)
    sar = np.zeros(n); trend_up=True; af=step; ep=high[0]; sar[0]=low[0]
    for i in range(1,n):
        sar[i] = sar[i-1] + af*(ep-sar[i-1])
        if trend_up:
            sar[i] = min(sar[i], low[i-1], low[i-2] if i>1 else low[i-1])
            if low[i] < sar[i]: trend_up=False; sar[i]=ep; ep=low[i]; af=step
            elif high[i] > ep: ep=high[i]; af=min(af+step,max_step)
        else:
            sar[i] = max(sar[i], high[i-1], high[i-2] if i>1 else high[i-1])
            if high[i] > sar[i]: trend_up=True; sar[i]=ep; ep=high[i]; af=step
            elif low[i] < ep: ep=low[i]; af=min(af+step,max_step)
    return pd.Series(sar, index=df.index, name="PSAR")

def rsi(series, period=14):
    delta=series.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(alpha=1/period, adjust=False).mean(); avg_loss=loss.ewm(alpha=1/period, adjust=False).mean()
    rs=avg_gain/avg_loss.replace(0,np.nan)
    return (100-(100/(1+rs))).fillna(50).rename("RSI")

def macd(series):
    ema12=series.ewm(span=12, adjust=False).mean(); ema26=series.ewm(span=26, adjust=False).mean()
    m=ema12-ema26; s=m.ewm(span=9, adjust=False).mean(); h=m-s
    return m.rename("MACD"), s.rename("MACD_signal"), h.rename("MACD_hist")

def stochastic_kd(df, n=9):
    low_min=df["Low"].rolling(n).min(); high_max=df["High"].rolling(n).max()
    rsv=(df["Close"]-low_min)/(high_max-low_min)*100
    k=rsv.ewm(alpha=1/3, adjust=False).mean(); d=k.ewm(alpha=1/3, adjust=False).mean()
    return k.rename("K"), d.rename("D")

def bollinger(series):
    mid=series.rolling(20).mean(); std=series.rolling(20).std()
    return (mid+2*std).rename("BB_upper"), mid.rename("BB_mid"), (mid-2*std).rename("BB_lower")

def atr(df, period=14):
    tr=pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean().rename("ATR")

def prepare_indicators(df):
    df=df.copy(); df["PSAR"]=parabolic_sar(df)
    df["MA5"]=df["Close"].rolling(5).mean(); df["MA20"]=df["Close"].rolling(20).mean(); df["MA60"]=df["Close"].rolling(60).mean()
    df["RSI"]=rsi(df["Close"]); df["MACD"],df["MACD_signal"],df["MACD_hist"]=macd(df["Close"])
    df["K"],df["D"]=stochastic_kd(df); df["BB_upper"],df["BB_mid"],df["BB_lower"]=bollinger(df["Close"]); df["ATR"]=atr(df)
    df["Return_1"]=df["Close"].pct_change(1)*100; df["Return_5"]=df["Close"].pct_change(5)*100; df["Return_20"]=df["Close"].pct_change(20)*100
    df["Volatility_20"]=df["Return_1"].rolling(20).std(); df["Volume_MA20"]=df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan
    df["Volume_Ratio"]=df["Volume"]/df["Volume_MA20"] if "Volume" in df.columns else np.nan
    return df.dropna()

@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker):
    try: info=yf.Ticker(ticker).info or {}
    except Exception: info={}
    roe=safe_float(info.get("returnOnEquity")); dy=safe_float(info.get("dividendYield"))
    return {"Company": info.get("longName") or info.get("shortName") or ticker, "EPS": safe_float(info.get("trailingEps")), "PE": safe_float(info.get("trailingPE")), "ROE (%)": roe*100 if not np.isnan(roe) else np.nan, "Dividend Yield (%)": dy*100 if not np.isnan(dy) else np.nan, "Sector": info.get("sector","N/A")}

@st.cache_data(ttl=3600)
def fetch_twse_institutional(stock_code):
    result={"foreign_net":np.nan,"trust_net":np.nan,"dealer_net":np.nan,"total_net":np.nan,"note":"No TWSE data."}
    if not stock_code.isdigit(): return result
    for days_back in range(1,15):
        date_str=(datetime.today()-timedelta(days=days_back)).strftime("%Y%m%d")
        url=f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json"
        try:
            data=requests.get(url,timeout=8).json(); fields, rows=data.get("fields",[]), data.get("data",[])
            if not fields or not rows: continue
            d=pd.DataFrame(rows, columns=fields); row=d[d["證券代號"].astype(str).str.strip()==stock_code]
            if row.empty: continue
            row=row.iloc[0]
            def parse(v):
                try: return float(str(v).replace(",","").replace("--","0"))
                except Exception: return np.nan
            return {"foreign_net":parse(row.get("外陸資買賣超股數(不含外資自營商)",np.nan)),"trust_net":parse(row.get("投信買賣超股數",np.nan)),"dealer_net":parse(row.get("自營商買賣超股數",np.nan)),"total_net":parse(row.get("三大法人買賣超股數",np.nan)),"note":f"TWSE date: {date_str}"}
        except Exception: continue
    return result


def score_market(df, chip):
    latest=df.iloc[-1]; score=0; buy=[]; risk=[]
    if latest.Close>latest.PSAR: score+=18; buy.append("價格在 PSAR 上方，短線偏多。")
    else: risk.append("價格跌破 PSAR，屬於轉弱訊號。")
    if latest.MA20>latest.MA60: score+=16; buy.append("MA20 高於 MA60，中期趨勢偏多。")
    else: risk.append("MA20 低於 MA60，中期趨勢還不夠強。")
    if latest.MACD>latest.MACD_signal: score+=16; buy.append("MACD 在訊號線上方，動能偏多。")
    else: risk.append("MACD 在訊號線下方，動能偏弱。")
    if latest.K>latest.D and latest.K<85: score+=12; buy.append("KD 向上且未嚴重過熱。")
    elif latest.K>85: score+=4; risk.append("KD 偏高，短線可能過熱。")
    else: risk.append("KD 尚未轉強。")
    if 45<=latest.RSI<=70: score+=12; buy.append("RSI 偏多但未過熱。")
    elif latest.RSI>75: score+=2; risk.append("RSI 過熱，追價風險較高。")
    elif latest.RSI<35: risk.append("RSI 偏弱，尚未確認止跌。")
    else: score+=6
    if latest.Close>latest.BB_mid: score+=10; buy.append("價格位於布林中軌上方。")
    else: risk.append("價格在布林中軌下方。")
    if latest.Return_20>0: score+=8; buy.append("近 20 期報酬為正。")
    else: risk.append("近 20 期報酬為負。")
    if not np.isnan(chip.get("total_net",np.nan)):
        if chip["total_net"]>0: score+=8; buy.append("三大法人買超。")
        else: risk.append("三大法人賣超。")
    return min(score,100), buy, risk

FEATURES=["Close","PSAR","MA5","MA20","MA60","RSI","MACD","MACD_signal","MACD_hist","K","D","BB_upper","BB_mid","BB_lower","ATR","Return_1","Return_5","Return_20","Volatility_20","Volume_Ratio"]

def train_ai_model(df,horizon=20):
    d=df.copy(); d["future_return"]=d["Close"].shift(-horizon)/d["Close"]-1; d["target"]=(d["future_return"]>0).astype(int)
    d=d.dropna(subset=FEATURES+["target"])
    if len(d)<180 or d["target"].nunique()<2: return None
    X=d[FEATURES].replace([np.inf,-np.inf],np.nan).dropna(); y=d.loc[X.index,"target"]
    if len(X)<160 or y.nunique()<2: return None
    split=int(len(X)*0.75); X_train,X_test=X.iloc[:split],X.iloc[split:]; y_train,y_test=y.iloc[:split],y.iloc[split:]
    model=RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=8,random_state=42,class_weight="balanced")
    model.fit(X_train,y_train); pred=model.predict(X_test); proba=model.predict_proba(X_test)[:,1]
    try: auc=roc_auc_score(y_test,proba)
    except Exception: auc=np.nan
    prob=model.predict_proba(X.iloc[[-1]])[0,1]*100
    imp=pd.DataFrame({"feature":FEATURES,"importance":model.feature_importances_}).sort_values("importance",ascending=False)
    return {"prob":prob,"accuracy":accuracy_score(y_test,pred)*100,"auc":auc,"samples":len(X),"positive_rate":y.mean()*100,"importances":imp}


def decision_without_position(df, rule_score, ai_prob):
    latest=df.iloc[-1]; close=latest.Close; atr_v=latest.ATR
    buy1=max(min(latest.MA20,latest.PSAR,latest.BB_mid), close-0.9*atr_v); buy2=max(latest.BB_lower,buy1-0.8*atr_v)
    chase_limit=min(close*1.015,close+0.25*atr_v); stop_loss=min(latest.PSAR,latest.MA20-0.6*atr_v,close-1.4*atr_v)
    if stop_loss>=close: stop_loss=close-1.2*atr_v
    sell1=max(latest.BB_upper,close+1.5*atr_v); sell2=close+2.5*atr_v; combined=0.55*rule_score+0.45*ai_prob
    if combined>=72 and close<=chase_limit: zh,cls="🟢 買進","buy"; one=f"可以考慮在 {buy1:.2f}–{chase_limit:.2f} 分批買進，不追高超過 {chase_limit:.2f}。"
    elif combined>=58: zh,cls="🟡 觀望，等買點","wait"; one=f"目前不要追價，較合理買點在 {buy2:.2f}–{buy1:.2f}。"
    else: zh,cls="🔴 不買 / 避開","sell"; one=f"目前訊號不足，不建議買進；若要觀察，等 {buy2:.2f} 附近再看。"
    return {"zh":zh,"cls":cls,"one":one,"combined":combined,"buy1":buy1,"buy2":buy2,"chase_limit":chase_limit,"stop_loss":stop_loss,"sell1":sell1,"sell2":sell2}

def decision_with_position(df, rule_score, ai_prob, cost, shares):
    latest=df.iloc[-1]; close=latest.Close; atr_v=latest.ATR; combined=0.55*rule_score+0.45*ai_prob
    ret_pct=(close-cost)/cost*100 if cost>0 else np.nan; profit_value=(close-cost)*shares*1000 if shares>0 else np.nan
    stop_loss=min(latest.PSAR,latest.MA20-0.6*atr_v,close-1.3*atr_v)
    if stop_loss>=close: stop_loss=close-1.2*atr_v
    tp1=max(latest.BB_upper,cost*1.06,close+1.2*atr_v); tp2=max(cost*1.10,close+2.2*atr_v); final=max(cost*1.15,close+3.0*atr_v)
    if close<=stop_loss or combined<45: zh,cls="🔴 賣出 / 停損","sell"; one=f"跌破 {stop_loss:.2f} 就不宜硬抱，建議停損或減碼。"
    elif close>=tp2 or latest.RSI>75: zh,cls="🟠 分批停利","wait"; one=f"可在 {tp1:.2f} 先賣一部分，{tp2:.2f} 再賣一部分。"
    elif combined>=58 and close>latest.PSAR: zh,cls="🔵 續抱","hold"; one=f"目前仍偏多，可續抱；跌破 {stop_loss:.2f} 再重新評估。"
    else: zh,cls="🟡 保守續抱 / 不加碼","wait"; one=f"訊號普通，先不加碼；跌破 {stop_loss:.2f} 應減碼。"
    return {"zh":zh,"cls":cls,"one":one,"combined":combined,"ret_pct":ret_pct,"profit_value":profit_value,"stop_loss":stop_loss,"take_profit1":tp1,"take_profit2":tp2,"final_exit":final}

def backtest_psar(df):
    d=df.copy(); d["position"]=(d["Close"]>d["PSAR"]).astype(int).shift(1).fillna(0); d["ret"]=d["Close"].pct_change().fillna(0); d["strategy_ret"]=d["position"]*d["ret"]
    total_strategy=(1+d["strategy_ret"]).prod()-1; total_hold=d["Close"].iloc[-1]/d["Close"].iloc[0]-1
    trades=[]; pos=0; entry=None
    for i in range(1,len(d)):
        signal=1 if d["Close"].iloc[i]>d["PSAR"].iloc[i] else 0
        if pos==0 and signal==1: pos=1; entry=d["Close"].iloc[i]
        elif pos==1 and signal==0: trades.append((d["Close"].iloc[i]-entry)/entry); pos=0; entry=None
    if pos==1 and entry is not None: trades.append((d["Close"].iloc[-1]-entry)/entry)
    wins=[x for x in trades if x>0]; win_rate=len(wins)/len(trades)*100 if trades else np.nan; avg_trade=np.mean(trades)*100 if trades else np.nan
    equity=(1+d["strategy_ret"]).cumprod(); drawdown=(equity/equity.cummax()-1).min()*100
    d["Buy_Signal"]=((d["Close"]>d["PSAR"]) & (d["Close"].shift(1)<=d["PSAR"].shift(1))).astype(int)
    d["Sell_Signal"]=((d["Close"]<d["PSAR"]) & (d["Close"].shift(1)>=d["PSAR"].shift(1))).astype(int)
    return {"strategy_return":total_strategy*100,"buy_hold_return":total_hold*100,"win_rate":win_rate,"avg_trade":avg_trade,"max_drawdown":drawdown,"trades":len(trades),"equity":equity,"signals":d}

DEFAULT_TW_TICKERS={"台積電":"2330","鴻海":"2317","聯發科":"2454","聯電":"2303","台達電":"2308","中華電":"2412","國泰金":"2882","富邦金":"2881","長榮":"2603","陽明":"2609","台灣高鐵":"2633","統一":"1216","台塑":"1301","中鋼":"2002","玉山金":"2884"}

def quick_rank(code,name,period="1y"):
    ticker,_=normalize_ticker(code); price=fetch_price_data(ticker,period,"1d")
    if price.empty or len(price)<120: return None
    df=prepare_indicators(price); rule_score,_,_=score_market(df,{"total_net":np.nan}); ai=train_ai_model(df,20); ai_prob=ai["prob"] if ai else rule_score
    combined=0.55*rule_score+0.45*ai_prob; latest=df.iloc[-1]
    return {"股票":name,"代號":code,"收盤價":latest.Close,"規則分數":rule_score,"AI上漲機率":ai_prob,"綜合分數":combined,"RSI":latest.RSI,"20期報酬%":latest.Return_20}

st.sidebar.header("設定")
page=st.sidebar.radio("頁面",["我的股票","AI排行榜","策略回測"],index=0)
raw_ticker=st.sidebar.text_input("股票代號",value="2633")
mode=st.sidebar.radio("是否已持有？",["尚未持有","已持有"],index=0)
cost=0.0; shares=0.0
if mode=="已持有":
    cost=st.sidebar.number_input("持有成本價",min_value=0.0,value=25.60,step=0.05)
    shares=st.sidebar.number_input("持有張數",min_value=0.0,value=1.0,step=1.0)
timeframe=st.sidebar.selectbox("分析時間週期",["日線 1D","小時線 1H"],index=0)
if timeframe=="日線 1D": interval,period="1d",st.sidebar.selectbox("資料與回測期間",["1y","2y","5y"],index=2)
else: interval,period="60m",st.sidebar.selectbox("資料與回測期間",["1mo","3mo","6mo","1y"],index=1)
horizon=st.sidebar.selectbox("AI預測未來幾期",[5,10,20],index=2)
run=st.sidebar.button("開始分析",type="primary")
st.sidebar.caption("v5 的 AI 為該股票歷史資料訓練的 Random Forest；資料不足時會退回規則分數。")

if page=="AI排行榜":
    st.markdown("## AI排行榜")
    st.write("用日線資料快速評估常見台股，依照綜合分數排序。")
    custom=st.text_area("自訂股票清單，可用逗號分隔，例如：2330,2317,2454", value=",".join(DEFAULT_TW_TICKERS.values()))
    if st.button("產生排行榜"):
        codes=[x.strip() for x in custom.split(",") if x.strip()]; name_map={v:k for k,v in DEFAULT_TW_TICKERS.items()}; rows=[]; progress=st.progress(0)
        for i,code in enumerate(codes):
            try:
                row=quick_rank(code,name_map.get(code,code),"1y")
                if row: rows.append(row)
            except Exception: pass
            progress.progress((i+1)/len(codes))
        if rows: st.dataframe(pd.DataFrame(rows).sort_values("綜合分數",ascending=False).round(2),use_container_width=True,hide_index=True)
        else: st.warning("沒有成功產生排行榜，可能是資料來源暫時無法取得。")
else:
    ticker,stock_code=normalize_ticker(raw_ticker)
    if run or raw_ticker:
        with st.spinner("正在抓取資料、計算指標與訓練 AI 模型..."):
            price=fetch_price_data(ticker,period,interval)
        if price.empty or len(price)<120:
            st.error("資料不足，請換較長期間，或確認股票代號是否正確。")
            st.stop()
        df=prepare_indicators(price); fundamentals=fetch_fundamentals(ticker); chip=fetch_twse_institutional(stock_code)
        rule_score,buy_reasons,risk_reasons=score_market(df,chip); ai_result=train_ai_model(df,horizon=horizon); ai_prob=ai_result["prob"] if ai_result else rule_score; bt=backtest_psar(df); latest=df.iloc[-1]
        if page=="策略回測":
            st.markdown(f"## 策略回測：{fundamentals.get('Company',ticker)} ({ticker})")
            c1,c2,c3,c4,c5=st.columns(5); c1.metric("PSAR策略報酬",f"{bt['strategy_return']:.1f}%"); c2.metric("買進持有報酬",f"{bt['buy_hold_return']:.1f}%"); c3.metric("勝率",f"{bt['win_rate']:.1f}%" if not np.isnan(bt['win_rate']) else "N/A"); c4.metric("平均每筆",f"{bt['avg_trade']:.1f}%" if not np.isnan(bt['avg_trade']) else "N/A"); c5.metric("最大回撤",f"{bt['max_drawdown']:.1f}%")
            st.line_chart(bt["equity"]); st.dataframe(bt["signals"][["Close","PSAR","Buy_Signal","Sell_Signal"]].tail(180).round(3),use_container_width=True); st.info("Buy_Signal=1 代表該期 PSAR 翻多；Sell_Signal=1 代表該期 PSAR 翻空。")
        else:
            st.markdown(f"## {fundamentals.get('Company',ticker)} ({ticker})"); st.caption(f"分析週期：{timeframe}｜最後資料時間：{df.index[-1]}｜AI 預測未來 {horizon} 期")
            if mode=="尚未持有":
                decision=decision_without_position(df,rule_score,ai_prob)
                st.markdown(f'<div class="signal {decision["cls"]}"><div class="signal-title">今日明確建議</div><div class="signal-main">{decision["zh"]}</div><div class="signal-sub">{decision["one"]}</div></div>',unsafe_allow_html=True)
                vals=[("目前價格",latest.Close,"最新收盤/週期價格"),("第一買點",decision["buy1"],"接近支撐，可觀察"),("第二買點",decision["buy2"],"更保守拉回價"),("不追高超過",decision["chase_limit"],"高於此價風險較高"),("停損價格",decision["stop_loss"],"跌破應重新評估"),("第一賣點",decision["sell1"],"可分批停利"),("第二賣點",decision["sell2"],"較積極目標"),("AI上漲機率",ai_prob,f"未來 {horizon} 期")]
                for chunk in [vals[:4],vals[4:]]:
                    cols=st.columns(4)
                    for col,(lab,val,note) in zip(cols,chunk): col.markdown(f'<div class="card"><div class="label">{lab}</div><div class="value">{val:.2f}{"%" if lab=="AI上漲機率" else ""}</div><div class="note">{note}</div></div>',unsafe_allow_html=True)
            else:
                decision=decision_with_position(df,rule_score,ai_prob,cost,shares)
                st.markdown(f'<div class="signal {decision["cls"]}"><div class="signal-title">我的持股建議</div><div class="signal-main">{decision["zh"]}</div><div class="signal-sub">{decision["one"]}</div></div>',unsafe_allow_html=True)
                vals=[("目前價格",latest.Close,"最新收盤/週期價格"),("持有成本",cost,"你輸入的成本"),("目前報酬",decision["ret_pct"],"未實現報酬率"),("估計損益",decision["profit_value"],"以每張1000股估算"),("停損賣出",decision["stop_loss"],"跌破就要小心"),("第一停利賣點",decision["take_profit1"],"可先賣30%"),("第二停利賣點",decision["take_profit2"],"可再賣30%"),("AI上漲機率",ai_prob,f"未來 {horizon} 期")]
                for chunk in [vals[:4],vals[4:]]:
                    cols=st.columns(4)
                    for col,(lab,val,note) in zip(cols,chunk): col.markdown(f'<div class="card"><div class="label">{lab}</div><div class="value">{val:,.2f}{"%" if lab in ["目前報酬","AI上漲機率"] else ""}</div><div class="note">{note}</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="section-title">AI 預測引擎</div>',unsafe_allow_html=True)
            if ai_result: st.markdown(f'<div class="ai-box"><b>Random Forest 預測：</b>未來 {horizon} 期上漲機率約 <b>{ai_prob:.1f}%</b><br><b>測試集 Accuracy：</b>{ai_result["accuracy"]:.1f}%　<b>AUC：</b>{ai_result["auc"]:.3f}　<b>訓練樣本：</b>{ai_result["samples"]}　<b>歷史上漲比例：</b>{ai_result["positive_rate"]:.1f}%</div>',unsafe_allow_html=True)
            else: st.warning("資料不足，AI 模型無法穩定訓練；目前使用規則分數作為替代。")
            st.markdown('<div class="section-title">為什麼這樣判斷？</div>',unsafe_allow_html=True)
            r1,r2=st.columns(2)
            with r1:
                st.markdown('<div class="reason-box"><b>偏多理由</b>',unsafe_allow_html=True)
                for r in buy_reasons[:7]: st.write("✅ "+r)
                st.markdown('</div>',unsafe_allow_html=True)
            with r2:
                st.markdown('<div class="reason-box"><b>風險理由</b>',unsafe_allow_html=True)
                for r in risk_reasons[:7]: st.write("⚠️ "+r)
                st.markdown('</div>',unsafe_allow_html=True)
            tabs=st.tabs(["交易圖","AI特徵重要性","回測摘要","投資模式","原始資料"])
            with tabs[0]:
                chart=df[["Close","PSAR","MA20","MA60","BB_upper","BB_mid","BB_lower"]].tail(180).copy()
                if mode=="尚未持有": chart["Buy 1"]=decision["buy1"]; chart["Buy 2"]=decision["buy2"]; chart["Stop loss"]=decision["stop_loss"]; chart["Sell 1"]=decision["sell1"]; chart["Sell 2"]=decision["sell2"]
                else: chart["Cost"]=cost; chart["Stop loss"]=decision["stop_loss"]; chart["Take profit 1"]=decision["take_profit1"]; chart["Take profit 2"]=decision["take_profit2"]
                st.line_chart(chart)
            with tabs[1]:
                if ai_result: st.dataframe(ai_result["importances"].round(4),use_container_width=True,hide_index=True); st.bar_chart(ai_result["importances"].set_index("feature"))
                else: st.write("AI 模型未建立，因此沒有特徵重要性。")
            with tabs[2]:
                c1,c2,c3,c4,c5=st.columns(5); c1.metric("PSAR策略報酬",f"{bt['strategy_return']:.1f}%"); c2.metric("買進持有報酬",f"{bt['buy_hold_return']:.1f}%"); c3.metric("勝率",f"{bt['win_rate']:.1f}%" if not np.isnan(bt['win_rate']) else "N/A"); c4.metric("平均每筆",f"{bt['avg_trade']:.1f}%" if not np.isnan(bt['avg_trade']) else "N/A"); c5.metric("最大回撤",f"{bt['max_drawdown']:.1f}%")
            with tabs[3]: st.subheader("基本面"); st.dataframe(pd.DataFrame([fundamentals]).round(3),use_container_width=True,hide_index=True); st.subheader("籌碼面"); st.dataframe(pd.DataFrame([chip]),use_container_width=True,hide_index=True)
            with tabs[4]: st.dataframe(df.tail(160).round(3),use_container_width=True)
            st.markdown('<div class="alert-box"><b>重要提醒：</b>v5 的 AI 模型是在使用者查詢時，以該股票歷史資料訓練 Random Forest。這是研究與教學用工具，不是投資建議，也不保證獲利。</div>',unsafe_allow_html=True)
