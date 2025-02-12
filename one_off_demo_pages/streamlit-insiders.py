import streamlit as st

#from finvizfinance.insider import Insider
from finvizfinance.quote import finvizfinance
import pandas as pd


st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="st-"] {
        text-align: center;
    }

    .stDataFrame {
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Insider Transactions")

# User selects a stock ticker
selected_ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()


# Fetch insider trades from Finviz
#finviz_insider = Insider()
#insider_trades = finviz_insider.get_insider().head(10)  # Get latest 10 insider trades
stock = finvizfinance(selected_ticker)
insider_trades = stock.ticker_inside_trader()

pd.set_option("display.max_columns", None)  # Show all columns

st.subheader(f"Latest Insider Transactions for {selected_ticker}")
if not insider_trades.empty:
    columns_to_exclude = ["SEC Form 4", "SEC Form 4 Link", "Insider_id"]
    insider_trades_filtered = insider_trades.drop(columns=[col for col in columns_to_exclude if col in insider_trades.columns])
    st.dataframe(insider_trades_filtered, hide_index=True)
else:
    st.error("No recent insider transactions found.")

news = stock.ticker_news()
first_res = 0
company_name = stock.ticker_description().split(",")[first_res].strip()
company_name_short = company_name.split(" ")[first_res].strip()
if not news.empty:
    news_filtered = \
                    news[
                          news["Link"].str.contains(company_name_short, case=False, na=False) \
                        | news["Title"].str.contains(company_name_short, case=False, na=False)
                        ]
else:
    news_filtered = pd.DataFrame()
st.subheader(f"Latest News for {company_name} seaching with {company_name_short}")
if not news.empty:
    st.dataframe(news_filtered, hide_index=True)
else:
    st.error("No recent insider transactions found.")
