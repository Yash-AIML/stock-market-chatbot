import os
import io
import json
import time
import math
import sqlite3
import base64
import requests
import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

from datetime import datetime, timedelta
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# Optional features (auto-disable if libs missing)
try:
    import streamlit_authenticator as stauth
    AUTH_AVAILABLE = True
except Exception:
    AUTH_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# -------------------- Config --------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.warning("⚠️ Set HF_TOKEN env variable for LLM access.")

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

st.set_page_config(page_title="📈 Multi-Model Trading Assistant", page_icon="📈", layout="wide")
st.title("📈 Multi-Model Trading Assistant")

# -------------------- Auth (Feature 9) --------------------
import streamlit_authenticator as stauth

def ensure_auth():
    # Example plain text passwords
    passwords = ["abc123", "mypassword", "test123"]

    # Initialize hasher
    hasher = stauth.Hasher()

    # Hash all passwords properly
    hashed_passwords = [hasher.hash(pw) for pw in passwords]

    # Now you can use hashed_passwords in your authenticator
    return hashed_passwords

if not ensure_auth():
    st.stop()

# -------------------- State & Memory --------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "watchlist" not in st.session_state:
    # Portfolio format: [{"symbol":"RELIANCE.NS","qty":10,"avg_price":2500.0}]
    st.session_state.watchlist = []

if "last_agent_summary" not in st.session_state:
    st.session_state.last_agent_summary = ""

# -------------------- Models & Routing --------------------
MODELS = {
    "finance_qa": "Qwen/Qwen3-Coder-30B-A3B-Instruct:fireworks-ai",  # general finance Q&A
    "trade_signals": "Qwen/Qwen3-Coder-30B-A3B-Instruct:fireworks-ai",  # entry/target/SL
    "general": "Qwen/Qwen3-Coder-30B-A3B-Instruct:fireworks-ai",  # fallback
}

def route_model(user_input: str):
    trade_kw = ["entry", "entry price", "target", "target price", "stop", "stoploss", "stop loss", "exit", "buy", "sell"]
    finance_kw = ["stock", "market", "nifty", "sensex", "pe ratio", "dividend", "earnings", "balance sheet", "options", "futures", "ipo"]
    text = user_input.lower()
    if any(k in text for k in trade_kw):
        return "trade_signals"
    if any(k in text for k in finance_kw):
        return "finance_qa"
    return "general"

# -------------------- LLM Streaming --------------------
def query_stream(messages, model):
    if not HF_TOKEN:
        yield "LLM unavailable (HF_TOKEN not set)."
        return
    payload = {"messages": messages, "model": model, "stream": True}
    with requests.post(API_URL, headers=HEADERS, json=payload, stream=True) as response:
        full_reply = ""
        for line in response.iter_lines():
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            if line.strip() == b"data: [DONE]":
                break
            data = json.loads(line.decode("utf-8").lstrip("data:").rstrip("\n"))
            if "choices" in data and "delta" in data["choices"][0]:
                chunk = data["choices"][0]["delta"].get("content", "")
                if chunk:
                    full_reply += chunk
                    yield chunk

# -------------------- Live Market Data (Feature 1) --------------------
def fetch_price_fast(symbol: str):
    try:
        t = yf.Ticker(symbol)
        cp = getattr(t, "fast_info", None)
        if cp and "last_price" in cp.__dict__:
            return float(cp.last_price)
        # Fallback from history
        hist = t.history(period="1d", interval="1m")
        if len(hist) > 0:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None

# -------------------- News + Sentiment (Feature 2) --------------------
def fetch_news(symbol: str, limit: int = 8):
    # Google News RSS (no API key). Using English-India for finance context
    q = requests.utils.quote(f"{symbol} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:limit]:
            items.append({"title": e.title, "link": e.link, "published": getattr(e, "published", "")})
        return items
    except Exception:
        return []

def llm_news_sentiment(headlines: list, symbol: str):
    if not headlines:
        return "No recent headlines found."
    text = "\n".join([f"- {h['title']}" for h in headlines])
    sys = {
        "role": "system",
        "content": (
            "You analyze finance news. Given recent headlines for a stock, return:\n"
            "OverallSentiment: Positive|Neutral|Negative\n"
            "Confidence: 0-100\n"
            "Rationale: one short paragraph."
        ),
    }
    user = {"role": "user", "content": f"Stock: {symbol}\nHeadlines:\n{text}"}
    chunks = []
    for c in query_stream([sys, user], MODELS["finance_qa"]):
        chunks.append(c)
    return "".join(chunks).strip()

# -------------------- Charts (Feature 4) --------------------
def plot_candles_with_indicators(symbol: str, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        st.warning("No data for chart.")
        return

    # Moving Averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI Calculation
    delta = df["Close"].diff()
    up = np.where(delta > 0, delta, 0).flatten()     # flatten ensures 1D
    down = np.where(delta < 0, -delta, 0).flatten()

    roll_up = pd.Series(up, index=df.index).rolling(14).mean()
    roll_down = pd.Series(down, index=df.index).rolling(14).mean()

    rs = roll_up / (roll_down + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- Candlestick chart ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
    fig.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- RSI chart ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    fig2.add_hline(y=70, line_dash="dash", line_color="red")
    fig2.add_hline(y=30, line_dash="dash", line_color="green")
    fig2.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig2, use_container_width=True)


# -------------------- Trade Journal (Feature 5) --------------------
DB_PATH = "journal.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            user_input TEXT,
            model_used TEXT,
            response TEXT,
            symbol TEXT,
            entry REAL,
            target REAL,
            stop REAL
        );
    """)
    conn.commit()
    conn.close()

def add_journal_entry(user_input, model_used, response, symbol=None, entry=None, target=None, stop=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO journal (ts, user_input, model_used, response, symbol, entry, target, stop) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), user_input, model_used, response, symbol, entry, target, stop),
    )
    conn.commit()
    conn.close()

def get_journal_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM journal ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# -------------------- Risk Calculator (Feature 6) --------------------
def position_size(capital: float, risk_pct: float, entry: float, stop: float):
    if entry <= 0 or stop <= 0 or capital <= 0 or risk_pct <= 0:
        return 0, 0.0
    risk_amt = capital * (risk_pct / 100.0)
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        return 0, 0.0
    qty = math.floor(risk_amt / per_share_risk)
    exposure = qty * entry
    return qty, exposure

# -------------------- Multi-Agent (Feature 7) --------------------
def agent_news(symbol):
    headlines = fetch_news(symbol, limit=8)
    sentiment = llm_news_sentiment(headlines, symbol)
    return headlines, sentiment

def agent_tech(symbol):
    # Ask LLM to produce a concise technical read based on recent OHLC (we pass last 100 closes)
    df = yf.download(symbol, period="3mo", interval="1d", progress=False)
    if df.empty:
        return "No data for technical analysis."
    closes = df[["Close"]].squeeze().tail(60).round(2).astype(str).to_list()
    sys = {"role": "system", "content": "You are a technical analyst. Use the numbers to infer trend and levels."}
    user = {"role": "user", "content": f"Symbol: {symbol}\nLast 60 daily closes: {', '.join(closes)}\nSummarize trend and key levels briefly."}
    chunks = []
    for c in query_stream([sys, user], MODELS["finance_qa"]):
        chunks.append(c)
    return "".join(chunks).strip()

def agent_risk(symbol, entry, target, stop, capital=100000, risk_pct=1.0):
    qty, exposure = position_size(capital, risk_pct, entry, stop)
    rr = (target - entry) / max(entry - stop, 1e-9) if (entry and stop) else None
    return f"Position: {qty} shares, Exposure ≈ {exposure:,.2f}. Risk/Reward ≈ {rr:.2f}." if rr else "Provide valid entry/stop to compute risk."

def multi_agent_summary(symbol, entry=None, target=None, stop=None):
    headlines, sentiment = agent_news(symbol)
    tech = agent_tech(symbol)
    risk = agent_risk(symbol, entry or 0, target or 0, stop or 0)
    head_list = "\n".join([f"- {h['title']}" for h in headlines]) if headlines else "No headlines."
    final_prompt = {
        "role": "user",
        "content": (
            f"Summarize a trade view for {symbol} by combining:\n\n"
            f"NewsSentiment:\n{sentiment}\n\n"
            f"TechnicalView:\n{tech}\n\n"
            f"RiskSummary:\n{risk}\n\n"
            "Return: A short final take (bullish/neutral/bearish) with 2 bullets of justification."
        ),
    }
    sys = {"role": "system", "content": "You are a chief strategist. Be concise and actionable."}
    chunks = []
    for c in query_stream([sys, final_prompt], MODELS["finance_qa"]):
        chunks.append(c)
    result = "".join(chunks).strip()
    st.session_state.last_agent_summary = result
    return head_list, sentiment, tech, risk, result

# -------------------- Voice I/O (Feature 8) --------------------
def transcribe_audio(file):
    if not SR_AVAILABLE:
        return "SpeechRecognition not available."
    r = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        return f"(Transcription failed: {e})"

def tts_download_link(text, filename="reply.mp3"):
    if not TTS_AVAILABLE:
        return None
    tts = gTTS(text)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a download="{filename}" href="data:audio/mp3;base64,{b64}">🔊 Download voice reply</a>'
    return href

# -------------------- Sidebar: Watchlist / Portfolio (Feature 3) --------------------
st.sidebar.header("📜 Watchlist / Portfolio")
with st.sidebar.expander("Add to Watchlist"):
    sym = st.text_input("Symbol (e.g., RELIANCE.NS, TCS.NS, AAPL)", key="add_sym")
    qty = st.number_input("Quantity", min_value=0, value=0, step=1, key="add_qty")
    avg = st.number_input("Avg Buy Price", min_value=0.0, value=0.0, step=1.0, key="add_avg")
    if st.button("➕ Add/Update"):
        # update if exists
        updated = False
        for row in st.session_state.watchlist:
            if row["symbol"].upper() == sym.upper():
                row["qty"], row["avg_price"] = qty, avg
                updated = True
                break
        if not updated and sym:
            st.session_state.watchlist.append({"symbol": sym.upper(), "qty": qty, "avg_price": avg})
        st.success("Saved.")

uploaded = st.sidebar.file_uploader("Upload CSV (symbol,qty,avg_price)", type=["csv"])
if uploaded is not None:
    try:
        dfu = pd.read_csv(uploaded)
        for _, r in dfu.iterrows():
            st.session_state.watchlist.append({"symbol": str(r["symbol"]).upper(), "qty": int(r["qty"]), "avg_price": float(r["avg_price"])})
        st.sidebar.success("Watchlist merged.")
    except Exception as e:
        st.sidebar.error(f"CSV error: {e}")

# Show portfolio table with live P/L
if st.session_state.watchlist:
    rows = []
    for r in st.session_state.watchlist:
        live = fetch_price_fast(r["symbol"]) or 0.0
        pnl = (live - r["avg_price"]) * r["qty"] if r["qty"] else 0.0
        rows.append({"Symbol": r["symbol"], "Qty": r["qty"], "Avg": r["avg_price"], "Live": round(live, 2), "P/L": round(pnl, 2)})
    port_df = pd.DataFrame(rows)
    st.sidebar.dataframe(port_df, use_container_width=True)

# -------------------- Main: Chat + Tools --------------------
col_chat, col_tools = st.columns([0.6, 0.4])

with col_chat:
    st.subheader("💬 Chat")
    # Voice input (upload)
    if SR_AVAILABLE:
        audio_file = st.file_uploader("🎙️ Upload WAV/AIFF audio to ask", type=["wav", "aiff", "aif"])
        if audio_file is not None:
            text_in = transcribe_audio(audio_file)
            if text_in:
                st.info(f"Transcribed: {text_in}")
                # push into chat as user
                st.session_state.memory.chat_memory.add_message(HumanMessage(content=text_in))

    # Display history
    for m in st.session_state.memory.chat_memory.messages:
        role = "assistant" if isinstance(m, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(m.content)

    # Chat input
    user_prompt = st.chat_input("Ask about stocks, finance, entry/target/stop…")
    if user_prompt:
        st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_prompt))
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Routing
        model_key = route_model(user_prompt)
        model = MODELS.get(model_key, MODELS["general"])

        # System prompts
        if model_key == "trade_signals":
            sys = {
                "role": "system",
                "content": (
                    "You are a trading assistant. "
                    "If the user mentions a stock, respond ONLY:\n"
                    "Stock: <Name or Ticker>\n"
                    "Entry Price: <value>\n"
                    "Target Price: <value>\n"
                    "Stop Loss: <value>\n"
                    "If stock invalid: 'Stock not found'. No extra text."
                ),
            }
        elif model_key == "finance_qa":
            sys = {"role": "system", "content": "You are a finance expert. Answer clearly and concisely."}
        else:
            sys = {"role": "system", "content": "You are a helpful assistant."}

        # Messages → HF format
        msgs = [sys]
        for msg in st.session_state.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                msgs.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msgs.append({"role": "assistant", "content": msg.content})

        # Stream
        with st.chat_message("assistant"):
            ph = st.empty()
            reply = ""
            for chunk in query_stream(msgs, model):
                reply += chunk
                ph.markdown(reply + "▌")
            ph.markdown(reply)

        st.session_state.memory.chat_memory.add_message(AIMessage(content=reply))

        # Journal save if structured trade reply (Feature 5)
        try:
            if "Entry Price" in reply and "Target Price" in reply and "Stop Loss" in reply:
                sym = None
                entry = target = stop = None
                for line in reply.splitlines():
                    if line.lower().startswith("stock"):
                        sym = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("entry"):
                        entry = float(line.split(":", 1)[1].strip().split()[0].replace(",", ""))
                    elif line.lower().startswith("target"):
                        target = float(line.split(":", 1)[1].strip().split()[0].replace(",", ""))
                    elif line.lower().startswith("stop"):
                        stop = float(line.split(":", 1)[1].strip().split()[0].replace(",", ""))
                add_journal_entry(user_prompt, model, reply, sym, entry, target, stop)
        except Exception:
            pass

        # TTS link (Feature 8)
        if TTS_AVAILABLE:
            link = tts_download_link(reply)
            if link:
                st.markdown(link, unsafe_allow_html=True)

with col_tools:
    st.subheader("🛠️ Tools")

    # Live snapshot (Feature 1)
    with st.expander("📊 Live Snapshot"):
        sym = st.text_input("Symbol for snapshot (e.g., RELIANCE.NS, TCS.NS, AAPL)", value="RELIANCE.NS")
        if st.button("Get Live Price"):
            price = fetch_price_fast(sym)
            if price:
                st.success(f"**{sym}** live ≈ `{price}`")
            else:
                st.warning("Price not available.")
        if st.button("Show Chart"):
            plot_candles_with_indicators(sym)

    # News & Sentiment (Feature 2)
    with st.expander("📰 News & Sentiment"):
        sym2 = st.text_input("Symbol for news sentiment", value="RELIANCE.NS", key="news_sym")
        if st.button("Analyze News"):
            hd = fetch_news(sym2, 8)
            if hd:
                st.write("Recent headlines:")
                for h in hd:
                    st.markdown(f"- [{h['title']}]({h['link']}) · {h.get('published','')}")
            sent = llm_news_sentiment(hd, sym2)
            st.markdown("**LLM Sentiment Summary**")
            st.write(sent)

    # Risk Calculator (Feature 6)
    with st.expander("🧮 Risk Calculator"):
        cap = st.number_input("Capital", min_value=0.0, value=100000.0, step=1000.0)
        riskp = st.number_input("Risk % per trade", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        e = st.number_input("Entry", min_value=0.0, value=2500.0, step=1.0)
        sl = st.number_input("Stop Loss", min_value=0.0, value=2450.0, step=1.0)
        qty, exposure = position_size(cap, riskp, e, sl)
        st.info(f"Suggested Qty: **{qty}** | Exposure: **{exposure:,.2f}**")

    # Multi-Agent (Feature 7)
    with st.expander("🧠 Multi-Agent View"):
        sym3 = st.text_input("Symbol", value="RELIANCE.NS", key="agent_sym")
        e3 = st.number_input("Entry (optional)", min_value=0.0, value=0.0, step=1.0)
        t3 = st.number_input("Target (optional)", min_value=0.0, value=0.0, step=1.0)
        s3 = st.number_input("Stop (optional)", min_value=0.0, value=0.0, step=1.0)
        if st.button("Run Agents"):
            heads, sent, tech, risk, final = multi_agent_summary(sym3, e3 or None, t3 or None, s3 or None)
            st.markdown("**Headlines**")
            st.write(heads)
            st.markdown("**Sentiment**")
            st.write(sent)
            st.markdown("**Technical View**")
            st.write(tech)
            st.markdown("**Risk Summary**")
            st.write(risk)
            st.markdown("**Final Take**")
            st.success(final)

    # Trade Journal (Feature 5)
    with st.expander("🗂️ Trade Journal"):
        dfj = get_journal_df()
        st.dataframe(dfj, use_container_width=True)
        if st.button("Export Journal CSV"):
            st.download_button("Download CSV", dfj.to_csv(index=False).encode(), "journal.csv", "text/csv")

# --------------- Footer / Notes ---------------
st.caption("⚠️ Educational use only. Not financial advice.")

