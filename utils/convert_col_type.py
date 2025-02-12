import os
import sqlite3

# Function to create a new table and copy the data with updated column types
def process_db_file(db_path, _debug=False):
    # Get the database name without extension
    db_name = os.path.splitext(os.path.basename(db_path))[0]

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the list of tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    new_table_name = f'{db_name}_{db_name}'
    if new_table_name in tables:
        drop_incomplete_table_sql   = f'DROP TABLE "{new_table_name}";'
        cursor.execute(drop_incomplete_table_sql)
        conn.commit()
    table_name = db_name

    # Fetch the schema for the table
    cursor.execute(f'PRAGMA table_info("{table_name}");')
    columns = cursor.fetchall()

    # Build the SQL statement for the new table creation
    new_table_columns = []
    for column in columns:
        col_name = column[1]
        col_type = column[2]
        cur_col_sql = f'"{col_name}"'
        if col_name in ['Ticker', 'Date']:
            #new_table_columns.append(f'"{col_name}" TEXT')  # Keep original type
            cur_col_sql += ' TEXT'
        # no noticable space saving diff using integer for candle stick indicators vs real
        elif col_name.upper().startswith("CDL_"):
            cur_col_sql += ' INTEGER'
        else:
            #new_table_columns.append(f'"{col_name}" REAL')  # Change type to REAL
            cur_col_sql += ' REAL'
        new_table_columns.append(cur_col_sql)

    try:
        # Create the new table with the updated column types

        create_table_sql = f'CREATE TABLE "{new_table_name}" ({", ".join(new_table_columns)});'
        if _debug: print(f"new table sql;\n {create_table_sql}")
        cursor.execute(create_table_sql)

        # Copy data from the old table to the new table
        column_names = [column[1] for column in columns]
        column_names = [f'"{col_name}"' for col_name in column_names]
        column_names_str = ", ".join(column_names)
        placeholders = ", ".join("?" * len(columns))
        insert_sql = f'INSERT INTO "{new_table_name}" ({column_names_str}) SELECT {column_names_str} FROM "{table_name}";'
        if _debug: print(f"copy table sql;\n {insert_sql}")
        cursor.execute(insert_sql)

        drop_cur_table_sql   = f'DROP TABLE "{table_name}";'
        rename_cur_table_sql = f'ALTER TABLE "{new_table_name}"\
                                 RENAME TO "{table_name}";'
        if _debug: print(f"drop table sql;\n {drop_cur_table_sql}")
        cursor.execute(drop_cur_table_sql)
        if _debug: print(f"rename table sql;\n {rename_cur_table_sql}")
        cursor.execute(rename_cur_table_sql)
        if _debug: print(f"clean up table sql;\n vacuum")

    except Exception as e:
        print(e)



        print(f"Created and copied data to new table {new_table_name} in {db_name}.")

    # Commit and close the connection
    conn.commit()
    # MASSIVELY IMPORTANT TO INCLUDE VACUUM
    # changes from resulting db file taking 50% more space to taking 50% less space
    cursor.execute('VACUUM;')
    conn.commit()
    conn.close()

# Function to process all databases in the given directory
def process_directory(directory_path, _dry_run=True, _debug=False):
    for filename in os.listdir(directory_path):
        if filename.endswith(".db"):
            db_path = os.path.join(directory_path, filename)
            print(f"Processing database: {db_path}")
            if not _dry_run:
                process_db_file(db_path, _debug)

if __name__ == "__main__":
    nasdaq_dir = r"C:\Users\human-c137\Documents\GitHub\stock_info\shards\nasdaq"
    nyse_dir = r"C:\Users\human-c137\Documents\GitHub\stock_info\shards\nyse"
    test_dir = r"C:\Users\human-c137\Documents\GitHub\stock_info\shards\test"

    process_directory(nyse_dir, _dry_run=False, _debug=False)
    process_directory(nasdaq_dir, _dry_run=False, _debug=False)
    #process_directory(test_dir, _dry_run=False, _debug=False)
