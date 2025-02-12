import sqlite3


table_name = 'GOOG'
conn = sqlite3.connect(f'{table_name}.db')
cursor = conn.cursor()


# Get all column names in the table
cursor.execute(f"PRAGMA table_info({table_name})")
columns = [row[1] for row in cursor.fetchall()]
#columns = [col[0] for col in cursor.description] # meant for sql server
#columns = list(zip(*cursor.fetchall()))[0]


#  Process columns in batches to avoid error on too many elements in select statment
batch_size = 50
results = []

for i in range(0, len(columns), batch_size):
    batch_columns = columns[i:i + batch_size]

    # Construct the query for the current batch
    query_parts = []
    for column in batch_columns:
        query_parts.append(f"""
            SELECT '{column}' AS column_name,
                   SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) AS null_count
            FROM "{table_name}"
        """)
        # need the first column in single quote to get prper name

    dynamic_query = " UNION ALL ".join(query_parts) +# " ORDER BY null_count DESC;" # no need with sort at end

    # Execute the query for the current batch
    cursor.execute(dynamic_query)
    results.extend(cursor.fetchall())
conn.close()

sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

print("Column Name | Null Count")
print("------------------------")
for row in sorted_results[:100]:
    if row[0] is not None and row[1] > 0:
        print(f"{row[0]:<12} | {row[1]}")


