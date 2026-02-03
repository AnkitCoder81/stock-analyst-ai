import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import os
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Annotated
import operator

# Load environment variables
load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY is missing. Please set it in your .env file.")
    st.stop()

# Initialize Groq Model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)


# --- Data Fetching Functions ---

def fetch_stock_data(ticker: str) -> dict:
    """Fetch price, volume, and calculate technical indicators."""
    stock = yf.Ticker(ticker)
    try:
        # Fetch 1 year of history for decent technical analysis
        history = stock.history(period='1y')
        info = stock.info
        
        if history.empty:
            return {}

        # --- Technical Indicators (Pandas TA) ---
        # 1. RSI
        history.ta.rsi(length=14, append=True)
        # 2. MACD
        history.ta.macd(append=True)
        # 3. Bollinger Bands
        history.ta.bbands(length=20, std=2, append=True)

        # Get latest values (dropping NaNs from calculation window)
        latest = history.iloc[-1]
        
        # Safe access to column names (pandas_ta naming convention)
        rsi = latest.get('RSI_14')
        macd = latest.get('MACD_12_26_9')
        macd_signal = latest.get('MACDs_12_26_9')
        bb_upper = latest.get('BBU_20_2.0')
        bb_lower = latest.get('BBL_20_2.0')

        return {
            'ticker': ticker,
            'current_price': latest['Close'],
            'history': history, # Store full dataframe for plotting
            'info': info,
            'technicals': {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'sma_50': info.get('fiftyDayAverage'),
                'sma_200': info.get('twoHundredDayAverage')
            }
        }
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return {}

def fetch_financials(ticker: str, period: str = 'yearly') -> dict:
    """Fetch financial statements."""
    stock = yf.Ticker(ticker)
    try:
        if period == 'quarterly':
            bal = stock.quarterly_balance_sheet
            inc = stock.quarterly_income_stmt
            cash = stock.quarterly_cashflow
        else:
            bal = stock.balance_sheet
            inc = stock.income_stmt
            cash = stock.cashflow

        def get_item(df, item):
            return df.loc[item].iloc[0] if item in df.index and not df.empty else 0

        return {
            'period': period,
            'revenue': get_item(inc, 'Total Revenue'),
            'net_income': get_item(inc, 'Net Income'),
            'eps': get_item(inc, 'Basic EPS'),
            'total_assets': get_item(bal, 'Total Assets'),
            'total_liabilities': get_item(bal, 'Total Liabilities Net Minority Interest'),
            'free_cash_flow': get_item(cash, 'Free Cash Flow')
        }
    except Exception as e:
        print(f"Error fetching financials: {e}")
        return {}

def fetch_news(ticker: str) -> List[str]:
    """Fetch top 10 recent news headlines using NewsAPI."""
    if not NEWS_API_KEY:
        return ["⚠️ NewsAPI Key not found. Skipping sentiment analysis."]
    
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize=10&language=en&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get('articles', [])
        headlines = [f"{art['title']} - {art['source']['name']}" for art in articles]
        return headlines if headlines else ["No recent news found."]
    except Exception as e:
        return [f"Error fetching news: {str(e)}"]

# --- LangGraph State & Nodes ---

class AgentState(TypedDict):
    ticker: str
    period: str
    stock_data: dict
    financial_data: dict
    news_headlines: List[str]
    sentiment_score: float
    sentiment_analysis: str
    technical_analysis: str
    final_report: str
    messages: Annotated[List[str], operator.add]


def data_aggregator_node(state: AgentState) -> AgentState:
    """Node 1: Gather all Data (Price, Techs, Fin, News)."""
    ticker = state['ticker']
    state['stock_data'] = fetch_stock_data(ticker)
    state['financial_data'] = fetch_financials(ticker, state['period'])
    state['news_headlines'] = fetch_news(ticker)
    state['messages'].append("✓ Fetched Market, Technical, Financial & News Data")

    return {
    "stock_data": state["stock_data"],
    "financial_data": state["financial_data"],
    "news_headlines": state["news_headlines"],
    "messages": state["messages"]
    }

def sentiment_agent_node(state: AgentState) -> AgentState:
    """Node 2: Analyze News Sentiment."""
    headlines = state['news_headlines']
    
    if not headlines or "⚠️" in headlines[0]:
        state['sentiment_score'] = 0.0
        state['sentiment_analysis'] = "Sentiment data unavailable."
        return state

    prompt = f"""
    You are a Sentiment Analysis Engine. Analyze these 10 headlines for {state['ticker']}:
    {json.dumps(headlines)}
    
    1. Calculate a sentiment score from -1.0 (Very Negative) to 1.0 (Very Positive).
    2. Provide a 2-sentence summary of the prevailing media narrative.
    
    Return STRICT JSON format: {{"score": float, "summary": "string"}}
    """
    try:
        response = llm.invoke(prompt).content
        # Basic cleanup to ensure JSON parsing
        response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(response)
        state['sentiment_score'] = data.get('score', 0.0)
        state['sentiment_analysis'] = data.get('summary', "Analysis failed.")
        state['messages'].append(f"✓ Calculated Sentiment Score: {state['sentiment_score']}")
    except:
        state['sentiment_score'] = 0.0
        state['sentiment_analysis'] = "Could not parse sentiment."
    
    return {
    "sentiment_score": state["sentiment_score"],
    "sentiment_analysis": state["sentiment_analysis"],
    "messages": state["messages"]
   }


def technical_analyst_node(state: AgentState) -> AgentState:
    """Node 3: Analyze Technical Indicators."""
    data = state['stock_data'].get('technicals', {})
    price = state['stock_data'].get('current_price', 0)
    
    # --- SAFETY HELPER ---
    # Ensures we never try to format 'None' as a float
    def safe_get(key, default=0.0):
        val = data.get(key)
        if val is None: 
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Use the helper to get clean numbers
    rsi = safe_get('rsi', 50.0)
    macd = safe_get('macd', 0.0)
    macd_signal = safe_get('macd_signal', 0.0)
    bb_upper = safe_get('bb_upper', 0.0)
    bb_lower = safe_get('bb_lower', 0.0)
    
    prompt = f"""
    You are a Technical Analyst. Analyze {state['ticker']} based on these metrics:
    - Price: ${price:.2f}
    - RSI (14): {rsi:.2f}
    - MACD: {macd:.4f} (Signal: {macd_signal:.4f})
    - Bollinger Bands: Upper ${bb_upper:.2f}, Lower ${bb_lower:.2f}
    
    Provide a concise paragraph analyzing the trend strength and momentum. 
    State if the stock is Overbought, Oversold, or Neutral.
    """
    state['technical_analysis'] = llm.invoke(prompt).content
    state['messages'].append("✓ Completed Technical Analysis")
    return {
    "technical_analysis": state["technical_analysis"],
    "messages": state["messages"]
     }


def master_analyst_node(state: AgentState) -> AgentState:
    """Node 4: Synthesize everything into a final report."""
    
    prompt = f"""
    You are a Wall Street Veteran. Synthesize the following for {state['ticker']}:

    1. **Fundamental Data**: 
       {json.dumps(state['financial_data'])}
    
    2. **Technical Analysis**: 
       {state['technical_analysis']}
    
    3. **Sentiment Analysis** (Score: {state['sentiment_score']}): 
       {state['sentiment_analysis']}

    Create a professional "Investment Memo". 
    Structure:
    - **Executive Summary** (The verdict)
    - **Bull Case vs Bear Case**
    - **Technical Setup** (Entry/Exit points)
    - **Final Recommendation** (Strong Buy/Buy/Hold/Sell)
    
    Use Markdown formatting.
    """
    state['final_report'] = llm.invoke(prompt).content
    state['messages'].append("✓ Generated Final Investment Memo")
    return {
    "final_report": state["final_report"],
    "messages": state["messages"]
}


# --- Graph Construction ---
def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("aggregator", data_aggregator_node)
    workflow.add_node("sentiment", sentiment_agent_node)
    workflow.add_node("technical", technical_analyst_node)
    workflow.add_node("master", master_analyst_node)
    
    workflow.set_entry_point("aggregator")
    workflow.add_edge("aggregator", "sentiment")
    workflow.add_edge("aggregator", "technical")
    workflow.add_edge("sentiment", "master")
    workflow.add_edge("technical", "master")
    workflow.add_edge("master", END)
    
    return workflow.compile()

# --- Plotting Functions ---
def plot_candlestick(df, ticker):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price'))

    # Bollinger Bands
    if 'BBU_20_2.0' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], 
                                 line=dict(color='gray', width=1, dash='dot'), name='Upper BB'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], 
                                 line=dict(color='gray', width=1, dash='dot'), name='Lower BB',
                                 fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))

    fig.update_layout(
        title=f'{ticker} Price Action & Volatility',
        yaxis_title='Stock Price (USD)',
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500
    )
    return fig

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Stock Analyst AI", page_icon="📈", layout="wide")
    
    st.title("📈 Stock Analyst AI")
    st.markdown("### Institutional-Grade Analysis with **LangGraph** + **Groq**")

    # Session State Initialization
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        ticker = st.text_input("Ticker Symbol", value="NVDA").upper()
        period = st.selectbox("Financial Period", ["yearly", "quarterly"])
        
        if st.button("🚀 Run Analysis", type="primary"):
            st.session_state.result = None # Reset
            st.session_state.chat_history = []
            
            with st.status("🤖 AI Agents at work...", expanded=True) as status:
                st.write("Initializing workflow...")
                graph = create_graph()
                inputs = {
                    "ticker": ticker, 
                    "period": period, 
                    "stock_data": {}, 
                    "financial_data": {},
                    "news_headlines": [],
                    "sentiment_score": 0,
                    "messages": []
                }
                
                # Stream updates
                final_state = graph.invoke(inputs)
                
                for msg in final_state['messages']:
                    st.write(msg)
                
                st.session_state.result = final_state
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                st.rerun()

    # --- Results Display ---
    if st.session_state.result:
        data = st.session_state.result
        stock = data['stock_data']
        techs = stock['technicals']
        
        # 1. KPI Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${stock['current_price']:.2f}")
        col2.metric("RSI (14)", f"{techs['rsi']:.1f}", 
                    delta="Overbought" if techs['rsi']>70 else "Oversold" if techs['rsi']<30 else "Neutral")
        col3.metric("Sentiment Score", f"{data['sentiment_score']:.2f}", 
                    delta="Bullish" if data['sentiment_score']>0.2 else "Bearish" if data['sentiment_score']<-0.2 else "Neutral")
        col4.metric("Recommendation", "See Report", help="Check the detailed memo below")

        # 2. Interactive Charts
        st.subheader("📊 Technical Chart (Interactive)")
        st.plotly_chart(plot_candlestick(stock['history'], ticker), use_container_width=True)

        # 3. Deep Dive Tabs
        tab1, tab2, tab3 = st.tabs(["📝 Investment Memo", "📰 News & Sentiment", "🔢 Raw Data"])
        
        with tab1:
            st.markdown(data['final_report'])
        
        with tab2:
            st.subheader("Sentiment Analysis")
            st.write(f"**AI Summary:** {data['sentiment_analysis']}")
            
            # Simple Sentiment Gauge using Progress Bar
            score_normalized = (data['sentiment_score'] + 1) / 2 # Convert -1..1 to 0..1
            st.progress(score_normalized, text=f"Sentiment Score: {data['sentiment_score']}")
            
            st.divider()
            st.caption("Recent Headlines")
            for news in data['news_headlines'][:5]:
                st.text(f"• {news}")

        with tab3:
            st.json(data['financial_data'])
            st.json(techs)

        # --- Chat Interface (Persistent) ---
        st.divider()
        st.subheader(f"💬 Chat about {ticker}")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
        
        if prompt := st.chat_input("Ask specifically about the RSI, Debt, or News..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                # Context-aware RAG
                context = f"""
                Ticker: {ticker}
                Technical Analysis: {data['technical_analysis']}
                Financials: {json.dumps(data['financial_data'])}
                Sentiment: {data['sentiment_analysis']} (Score: {data['sentiment_score']})
                Report: {data['final_report']}
                """
                response = llm.invoke(f"Context:\n{context}\n\nUser: {prompt}").content
                st.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

if __name__ == "__main__":
    main()