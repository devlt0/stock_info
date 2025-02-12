import streamlit as st
from pygooglenews import GoogleNews
from dateutil import parser
import datetime

st.title("Google News: Latest News for Stocks")

# User selects a stock ticker
ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()

# Fetch news using pygooglenews
gn = GoogleNews()
news_data = gn.search(ticker, when='1y')

# Manually sort first 50 entries by published date
if "entries" in news_data and news_data["entries"]:
    def safe_parse_date(date_str):
        try:
            return parser.parse(date_str)
        except Exception:
            return datetime.datetime.min

    entries = news_data["entries"][:50]  # Limit to first 50 entries
    sorted_entries = sorted(
        [(safe_parse_date(entry.get("published", "")), entry) for entry in entries],
        key=lambda x: x[0],
        reverse=True
    )
    sorted_entries = [entry[1] for entry in sorted_entries]
else:
    sorted_entries = []

with st.container(height=600):
    st.header(f"Latest News for {ticker}")
    if sorted_entries:
        for entry in sorted_entries[:25]:
            published_date = safe_parse_date(entry.get("published", "Unknown"))
            formatted_date = published_date.strftime("%Y-%m-%d %H:%M:%S") if published_date != datetime.datetime.min else "Unknown"
            #st.subheader(entry.get("title", "No Title"))
            #st.write(f"**Published:** {formatted_date}")
            st.markdown(f"### {entry.get('title', 'No Title')}")
            st.markdown(f"**Published:** {formatted_date}")
            st.markdown(entry.get("summary", "No Summary Available"),unsafe_allow_html=True)
            st.markdown(f"[Read more]({entry.get('link')})")
            st.markdown("---")
    else:
        st.error("No news found for this ticker.")
