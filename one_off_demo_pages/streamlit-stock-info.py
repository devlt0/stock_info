import streamlit as st
import yfinance as yf
import math
from os import getenv

from pymongo import MongoClient



conn_str = getenv("MDB_CLOUD_CNN")
conn_str = conn_str.replace('"', '') # strip quotes from env variable being set
if not conn_str:
    raise ValueError("MDB_CLOUD_CNN environment variable is not set.")

if not conn_str.startswith(("mongodb://", "mongodb+srv://")):
    raise ValueError(f"Invalid MongoDB URI. It must start with 'mongodb://' or 'mongodb+srv://'.\n{conn_str}")

client = MongoClient(conn_str)
db = client["company_db"]
collection = db["company_info"]

available_tickers = [doc["symbol"] for doc in collection.find({}, {"symbol": 1})]
available_tickers.sort()

st.title("Stock Ticker Information")


ticker_symbol = st.selectbox("Enter or select a stock ticker symbol:", available_tickers) # selected_ticker

# Fetch data from yfinance
if ticker_symbol:
    company_info = collection.find_one({"symbol": ticker_symbol})
    if company_info:
        try:
            avg_age = company_info.get("average_age", 0)
            avg_pay = company_info.get("average_pay", 0)
            std_dev_age = company_info.get("std_dev_age", 0)
            std_dev_pay = company_info.get("std_dev_pay", 0)
            full_time_employees = company_info.get("fullTimeEmployees", "N/A")
            total_officers = company_info.get("total_officers", 0)
            # Display basic information
            st.subheader(f"Information for {company_info.get('longName', ticker_symbol)}")
            st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
            st.write(f"**Market Cap:** ${company_info.get('marketCap', 'N/A'):,}")
            st.write(f"**Current Price:** ${company_info.get('currentPrice', 'N/A'):.2f}")
            st.write(f"**52 Week High:** ${company_info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
            st.write(f"**52 Week Low:** ${company_info.get('fiftyTwoWeekLow', 'N/A'):.2f}")
            st.write(f"Average Age of Company Officers: {avg_age:.2f} years")
            st.write(f"Standard Deviation of Age: {std_dev_age:.2f} years")
            st.write(f"Average Pay of Company Officers: ${avg_pay:,.2f}")
            st.write(f"Standard Deviation of Pay: ${std_dev_pay:,.2f}")
            st.write(f"Total Number of Company Officers Listed: {total_officers}")
            st.write(f"Number of Full-Time Employees: {full_time_employees:,}")

            # Display additional info in an expandable section
            with st.expander("Show More Details"):
                st.write(company_info)

        except Exception as e:
            st.error(f"Error fetching data for {ticker_symbol}. Please check the ticker symbol and try again.")
            st.write(f"Error details: {e}")
    else:
        st.warning(f"No data found for {ticker_symbol}. Please try another ticker.")
else:
    st.warning("Please enter a valid stock ticker symbol.")