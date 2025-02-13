from pymongo import MongoClient
from os import getenv
from tabulate import tabulate

conn_str = getenv('MDB_LOCAL_CNN') # ToDo, if local collection size < MongoDB free cloud (500mb), move db there
client = MongoClient(conn_str) #MongoClient("mongodb://localhost:27017/")
db = client["company_db"]
collection = db["company_info"]

# Aggregation pipeline
pipeline = [
    {
        "$group": {
            "_id": {
                "sector": {"$ifNull": ["$sector", "Unknown"]},  # Handle missing Sector
                "industry": {"$ifNull": ["$industry", "Unknown"]}  # Handle missing Industry
            },
            "total_stocks": {"$sum": 1}
        }
    },
    {
        "$sort": {"total_stocks": -1}  # Sort results by count (descending)
    }
]

# Execute aggregation
result = list(collection.aggregate(pipeline))

'''
# Print results
print("Stock Summary by Sector and Industry:")
for entry in result:
    sector = entry["_id"].get("sector", "Unknown")  # Use .get() to avoid KeyError
    industry = entry["_id"].get("industry", "Unknown")
    count = entry["total_stocks"]
    print(f"Sector: {sector}, Industry: {industry}, Total Stocks: {count}")
'''

table_data = [
    [entry["_id"]["sector"], entry["_id"]["industry"], entry["total_stocks"]]
    for entry in result
]

# Print the table
print(tabulate(table_data, headers=["Sector", "Industry", "Total Stocks"], tablefmt="grid"))