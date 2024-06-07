import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd

from utils import connect_to_db, get_tables, load_table
from get_sql_answer import get_chain, get_agent

# TODO: Streamlit app
def main():
    st.set_page_config(
        page_title="Text-To-SQL",
        layout="wide")
    
    with st.sidebar:
        st.title("Text-To-SQL :computer:")
        st.markdown(
            """
            ### Instructions:
            1. Select table to query
            2. Ask question
            3. Choose if you want to return SQL query or direct answer
            4. Enter or Press 'Get Answer'
            """
        )
        conn = connect_to_db()
        table = get_tables(conn)
        selected_table = st.sidebar.selectbox("Select table to query", table)
    
    # Preview table
    df = load_table(selected_table, conn)
    st.dataframe(df, hide_index=True)

    question = st.text_input("Ask Question: ")

    PROMPT = """ 
Given an input question, first create a syntactically correct postgresql query to run,  
then look at the results of the query and return the answer.  
The question: {question}
"""
    return_sql = st.checkbox("Return SQL Query", value=True)
    if st.button("Get Answer"):
        if question:
            # Checkbox to toggle return_sql
            if return_sql:
                chain = get_chain(CONNECTION_STRING, selected_table, return_sql)
                response = chain.run(PROMPT.format(question=question))
                st.header("SQL Query")
                st.code(response, language='sql')
                
                result = pd.read_sql_query(response, conn)
                conn.close()
                st.write("Query executed successfully.")
                st.dataframe(result, hide_index=True)
            else:
                chain = get_chain(CONNECTION_STRING, selected_table, return_sql)
                response = chain.run(PROMPT.format(question=question))
                st.header("Answer")
                st.write(response)

    # mod_response = st.text_input("Enter your modified sql")
    # st.code(mod_response, language='sql')
    
    # if st.button("Execute"):
    #     conn = connect_to_db()
    #     new_result = pd.read_sql_query(mod_response, conn)
    #     conn.close()
    #     if isinstance(new_result, pd.DataFrame):
    #         st.write("Query executed successfully.")
    #         st.dataframe(new_result)
    #     else:
    #         st.write("Error executing query:")
    #         st.write(new_result)
    
    with st.sidebar:
        st.divider()

    st.sidebar.markdown(
    """
    ### Sample questions to get started:
    1. Question 1 
    2. Question 2
    3. Question 3
    """
)

if __name__ in '__main__':
    load_dotenv()

    host = os.environ.get("DB_HOST")                                          
    user = os.environ.get("DB_USER")                                      
    password = os.environ.get("DB_PASSWORD")
    database = os.environ.get("DB_DATABASE")                               

    CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{database}"
    
    main()