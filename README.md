# 📈 Stock Analyst AI

**Institutional-grade AI stock analysis** combining technical indicators, fundamentals, and news sentiment — built with **Streamlit**, **LangGraph**, and **Groq LLM**.  
The app generates an investment memo, interactive charts, and a chat interface for context-aware follow-up questions.

> **Tech Stack:** Python · Streamlit · yfinance · pandas_ta · Plotly · LangGraph · Groq LLM · NewsAPI

---

## 🌟 Features

- 🧠 **Multi-Agent AI Workflow** using LangGraph
- 📊 **Technical Analysis**
  - RSI, MACD, Bollinger Bands
  - 50 / 200 SMA
- 📉 **Interactive Candlestick Chart** (Plotly)
- 📰 **News Sentiment Analysis** using LLM
- 📑 **AI-Generated Investment Memo**
- 💬 **Chat Interface** for stock-specific Q&A
- ⚡ **Single-file Streamlit App** (easy to deploy)

---

## 🖥️ Demo Flow

1. Enter stock ticker (e.g. `AAPL`, `RELIANCE.NS`)
2. App fetches:
   - Historical price data
   - Financial fundamentals
   - Latest news
3. AI agents analyze:
   - Technical indicators
   - Market sentiment
4. Final output:
   - Investment memo
   - Charts & metrics
   - Interactive chat

---

## 📁 Project Structure
StockMarket/
├── stock_app_chat-v2.py
├── requirements.txt
├── .gitignore
├── README.md
└── myenv/ # local virtual environment (not pushed)

## 📁 Project Structure

