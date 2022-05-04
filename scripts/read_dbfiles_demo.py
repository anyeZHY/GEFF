from pathlib import Path
import sqlite3 as sq
import pandas as pd

path=str(Path.cwd().parent.parent)+'/assets/ColumbiaGazeDataSet/0001'
dbfile = path+'/Thumbs.db'
# Create a SQL connection to our SQLite database
con = sq.connect(dbfile)
# creating cursor
cur = con.cursor()

# reading all table names
table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
# here is you table list

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(f"Table Name : {cur.fetchall()}")
print(table_list)
df = pd.read_sql_query('SELECT * FROM Table_Name', con)
# Be sure to close the connection
con.close()
