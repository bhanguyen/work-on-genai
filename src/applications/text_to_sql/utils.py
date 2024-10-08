import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()
# Function to connect to database
# Set up the connection to PostgreSQL
def connect_to_db(user, password, host, port, database):
    db_param = {
        "user": user,                                     
        "password": password,                                  
        "host": host,                                            
        "port": port,                                          
        "database": database
    }

    return psycopg2.connect(**db_param)

# Function to get tables 
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