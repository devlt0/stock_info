from datetime import datetime
from os import getenv
from time import sleep
import math

from pymongo import MongoClient
import yfinance as yf

from shards.nyse_list_2025 import nyse_tickers
from shards.nasdaq_list_2025 import nasdaq_tickers

# db connection
conn_str = getenv('MDB_LOCAL_CNN') # ToDo, if local collection size < MongoDB free cloud (500mb), move db there
client = MongoClient(conn_str) #MongoClient("mongodb://localhost:27017/")
db = client["company_db"]
collection = db["company_info"]

yahoo_rate_limit_delay = 2  # technically can do 1.8s but a lil buffer doesn't hurt
# not counting processing time ~2hrs 47min on 5000 tickers
# going on bare min limit of 1.8sec would result in ~2hr 30min or ~17min saved (~10%)

def calculate_std_dev(values, mean):
    if not values:
        return 0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

tickers = []
tickers.extend(nyse_tickers)
tickers.extend(nasdaq_tickers)
tickers_to_dbl_chk = []
for ticker in tickers:
    if collection.find_one({"symbol": ticker}):
        print(f"Data for {ticker} already exists in MongoDB, skipping...")
        continue

    company_info = None
    try:
        company_info = yf.Ticker(ticker).info
    except Exception as e:
        print(e)
        tickers_to_dbl_chk.append(ticker)
        continue

    if company_info is not None:
        # Extract ages and pay, filtering out None values
        ages = [officer["age"] for officer in company_info.get("companyOfficers", []) if officer.get("age") is not None]
        pay = [officer["totalPay"] for officer in company_info.get("companyOfficers", []) if officer.get("totalPay") is not None]

        # Calculate statistics
        avg_age = sum(ages) / len(ages) if ages else 0
        avg_pay = sum(pay) / len(pay) if pay else 0
        std_dev_age = calculate_std_dev(ages, avg_age)
        std_dev_pay = calculate_std_dev(pay, avg_pay)
        total_officers = len(company_info.get("companyOfficers", []))

        # Add computed values to the company info
        company_info.update({
            "average_age": avg_age,
            "average_pay": avg_pay,
            "std_dev_age": std_dev_age,
            "std_dev_pay": std_dev_pay,
            "total_officers": total_officers,
            "last_updated_utc": datetime.utcnow(),
            "last_updated_utc_str": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        })
        try:
            # Save to MongoDB
            collection.update_one(
                {"symbol": company_info["symbol"]},
                {"$set": company_info},
                upsert=True,
            )
            print(f"Data for {ticker} saved to MongoDB")
        except Exception as e:
            print(e)
            tickers_to_dbl_chk.append(ticker)
            continue

    sleep(yahoo_rate_limit_delay)

client.close()
tickers_to_dbl_chk = list(set(tickers_to_dbl_chk))
print(f"double check following tickers;\n{tickers_to_dbl_chk}")