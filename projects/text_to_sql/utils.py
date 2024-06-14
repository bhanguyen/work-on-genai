import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
# TODO: Function to connect to database
# Set up the connection to PostgreSQL
def connect_to_db():
    conn = psycopg2.connect(
        user = os.environ.get("DB_USER"),                                     
        password = os.environ.get("DB_PASSWORD"),                                  
        host = os.environ.get("DB_HOST"),                                            
        port = os.environ.get("DB_PORT"),                                          
        database = os.environ.get("DB_DATABASE")
    )

    return conn

# TODO: Function to get tables 
def get_tables(conn):
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    with conn.cursor() as cur:
        cur.execute(query)
        tables = cur.fetchall()
    return [table[0] for table in tables]

# Function to load table from PostgreSQL database
def load_table(table_name, connection_params):
    try:
        conn = connection_params
        query = f'SELECT * FROM {table_name} LIMIT 5'
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        return str(e)