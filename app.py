import os
import streamlit as st
import pandas as pd
import psycopg2
from anthropic import Anthropic
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Default values from environment variables
DEFAULT_RDS_HOST = os.environ.get("RDS_HOST")
DEFAULT_RDS_DATABASE = os.environ.get("RDS_DATABASE")
DEFAULT_RDS_USER = os.environ.get("RDS_USER")
DEFAULT_RDS_PASSWORD = os.environ.get("RDS_PASSWORD")
DEFAULT_RDS_PORT = os.environ.get("RDS_PORT", "5432")
DEFAULT_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Function to execute SQL queries and fetch data
def run_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)
    return cur.fetchall()

# Function to generate a response using the Claude AI model
def get_response(client, prompt, data=None):
    
    prompt += f"\n\nHuman: {prompt}\n\nAssistant: \n\n"

    response = client.completions.create(
        prompt=prompt,
        model="claude-v1.3",
        max_tokens_to_sample=1000,
    )

    return response.completion

# Streamlit app
def main():
    st.title("AI-Powered Data Exploration")
    st.write("Welcome to the AI-Powered Data Exploration app! Please choose whether you want to explore a single file or data in a database.")
    data_source = st.radio("Select data source", ["Single File", "Database"])

    if data_source == "Single File":
        # User input for Anthropic API key
        # api_key = st.text_input("Anthropic API Key", DEFAULT_API_KEY, type="password")

        # User input for file upload
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "parquet", "json"])

        if uploaded_file is not None:
            # Read the uploaded file into a Pandas DataFrame
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                df = pd.read_json(uploaded_file)

            # Display the DataFrame
            st.write(df)

            # User input for the prompt or query
            prompt = st.text_area("Enter your query or prompt:")

            if st.button("Submit"):
                try:
                    # Initialize the Anthropic client
                    client = Anthropic(api_key=DEFAULT_API_KEY)

                    # Get the response from the AI model
                    response = get_response(client, prompt, df)
                    st.write(response)
                except Exception as e:
                    st.write(f"Error: {e}")

    elif data_source == "Database":
        # User input for RDS connection details
        rds_host = st.text_input("RDS Host", DEFAULT_RDS_HOST)
        rds_database = st.text_input("Database Name", DEFAULT_RDS_DATABASE)
        rds_user = st.text_input("Database User", DEFAULT_RDS_USER)
        rds_password = st.text_input("Database Password", DEFAULT_RDS_PASSWORD, type="password")
        rds_port = st.text_input("Database Port", DEFAULT_RDS_PORT)

        # User input for Anthropic API key
        # api_key = st.text_input("Anthropic API Key", DEFAULT_API_KEY, type="password")

        # User input for the prompt or query
        prompt = st.text_area("Enter your query or prompt:")

        if st.button("Submit"):
            try:
                # Connect to the RDS instance
                conn = psycopg2.connect(
                    host=rds_host or DEFAULT_RDS_HOST,
                    database=rds_database or DEFAULT_RDS_DATABASE,
                    user=rds_user or DEFAULT_RDS_USER,
                    password=rds_password or DEFAULT_RDS_PASSWORD,
                    port=rds_port or DEFAULT_RDS_PORT
                )

                # Initialize the Anthropic client
                client = Anthropic(api_key=DEFAULT_API_KEY)

                # Get the response from the AI model
                response = get_response(client, prompt, df)

                # Check if the response requires fetching data from the database
                if "EXECUTE SQL:" in response:
                    query = response.split("EXECUTE SQL:")[1].strip()
                    data = run_query(conn, query)
                    st.write("Query Results:")
                    st.write(data)
                else:
                    st.write(response)
            except Exception as e:
                st.write(f"Error: {e}")

if __name__ == "__main__":
    main()