import csv
import sqlite3

def csv_to_sqlite(csv_file_path, sqlite_db_path, table_name):
    # Connect to SQLite database (it will create the file if it doesn't exist)
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Get the headers from the first row of the CSV file
        headers = next(csv_reader)
        
        # Create table with columns based on headers
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"CREATE TABLE {table_name} ({', '.join([f'{header} TEXT' for header in headers])})")

        # Insert each row of data into the table
        for row in csv_reader:
            placeholders = ', '.join('?' * len(row))
            cursor.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", row)

    conn.commit()
    conn.close()
    print(f"Data from {csv_file_path} has been inserted into {table_name} table in {sqlite_db_path} database.")

csv_file_path = 'fp-historical-wildfire-data-2006-2023.csv'
sqlite_db_path = 'wildfire-data.db'
table_name = 'wildfire'

csv_to_sqlite(csv_file_path, sqlite_db_path, table_name)
