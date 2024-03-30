import os
import streamlit as st
import pandas as pd
import psycopg2
from anthropic import Anthropic
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from streamlit import session_state as st_state
from textwrap import dedent

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
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

# Function to generate a response using the Claude AI model
def get_response(client, prompt, data=None):
    system_prompt = """
    System: You are an AI assistant specialized in Exploratory Data Analysis (EDA). 
    Your goal is to help analysts and data scientists understand and gain insights from their data by following a structured EDA process. 
    When presented with a dataset, consider the following steps and techniques:

    1. Data Understanding:
    - Describe the dataset (number of rows, columns, data types, missing values, etc.)
    - Identify the target variable(s) and predictor variables
    - Understand the context and domain knowledge related to the data

    2. Data Cleaning and Preprocessing:
    - Handle missing values (imputation, removal, etc.)
    - Identify and handle outliers
    - Convert data types as needed
    - Encode categorical variables

    3. Univariate Analysis:
    - Analyze distributions of individual variables (numerical and categorical)
    - Visualize distributions using histograms, bar plots, box plots, etc.
    - Calculate summary statistics (mean, median, mode, standard deviation, etc.)

    4. Bivariate Analysis:
    - Analyze relationships between pairs of variables
    - Visualize relationships using scatter plots, line plots, bar plots, etc.
    - Calculate correlation coefficients (Pearson, Spearman, etc.)

    5. Multivariate Analysis:
    - Analyze relationships among multiple variables
    - Use dimensionality reduction techniques (PCA, t-SNE, etc.)
    - Visualize higher-dimensional data using scatter plot matrices, parallel coordinate plots, etc.

    6. Target Variable Analysis:
    - Analyze the relationship between predictor variables and the target variable(s)
    - Use techniques like correlation analysis, hypothesis testing, etc.

    7. Data Transformation and Feature Engineering:
    - Apply transformations (log, square root, etc.) to improve data quality
    - Create new features based on domain knowledge or statistical techniques

    8. Model Development and Evaluation:
    - Split data into training and testing sets
    - Build and evaluate machine learning models (regression, classification, clustering, etc.)
    - Interpret model results and provide insights

    Throughout the EDA process, use appropriate visualizations, statistical tests, and domain knowledge to gain a comprehensive understanding of the data. 
    Be prepared to handle different data types, distributions, and relationships. Clearly communicate your findings and recommendations.
    """
    if data is not None:
        data_str = data.to_string()
        prompt = f"{data_str}\n\nHuman: {prompt}\n\nAssistant:"
    else:
        prompt = f"\n\nHuman: {prompt}\n\nAssistant:"

    response = client.completions.create(
        prompt=prompt,
        model="claude-v1.3",
        max_tokens_to_sample=1000,
    )

    display_message(False, prompt.split('Human: ')[1].split('Assistant:')[0], "human_bg")
    display_message(True, response.completion, "assistant_bg")

# Function to display messages in a block format
def display_message(is_assistant, message, bg_color_class):
    spacer, utterance_pre = (
        "\n\n\n\n",
        dedent(
            f"""
            <div class="{bg_color_class}">
            <b>{"Assistant" if is_assistant else "You"}:</b>
            """
        ),
    )
    utterance = f"{utterance_pre} {message}</div>"
    st.markdown(spacer + utterance, unsafe_allow_html=True)

# Function to explore single file
def explore_single_file():
    st.subheader("Explore Data from Single File")

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

        # Display basic info about the uploaded file
        st.write("File Info:")
        st.write(df.info())

        # User input for the prompt or query
        prompt = st.text_area("Enter your query or prompt:", key="input_text")

        if st.button("Submit", key="submit_button") or prompt:
            if prompt:
                try:
                    # Initialize the Anthropic client
                    client = Anthropic(api_key=DEFAULT_API_KEY)
                    
                    # Get the response from the AI model
                    get_response(client, prompt, df)
                except Exception as e:
                    st.write(f"Error: {e}")

        conversation_container = st.container()

#Function to explore database tables
def explore_database_tables():
    st.subheader("Explore Data from Database")

    # User input for RDS connection details
    rds_host = st.text_input("RDS Host", DEFAULT_RDS_HOST)
    rds_database = st.text_input("Database Name", DEFAULT_RDS_DATABASE)
    rds_user = st.text_input("Database User", DEFAULT_RDS_USER)
    rds_password = st.text_input("Database Password", DEFAULT_RDS_PASSWORD, type="password")
    rds_port = st.text_input("Database Port", DEFAULT_RDS_PORT)

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
            response = get_response(client, prompt)

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

# Streamlit app
def main():
    st.sidebar.title("AI-Powered Exploratory Data Analysis")
    st.write("Welcome to the AI-Powered EDA app! Please choose whether you want to explore a single file or data in a database.")
    data_source = st.sidebar.radio("Select data source", ["Single File", "Database"])

    if data_source == "Single File":
        explore_single_file()

    elif data_source == "Database":
        explore_database_tables()

    st.markdown(
        """
    <style>
    .human_bg {
        background-color: #808080;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .assistant_bg {
        background-color: #808080;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()