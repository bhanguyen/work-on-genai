import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd

from utils import connect_to_db, get_tables, load_table
from get_sql_answer import get_chain, get_agent

from langchain.callbacks import StreamlitCallbackHandler

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
    agent = st.checkbox("Use Agent?")
    if agent is False:
        return_sql = st.checkbox("Return SQL Query", value=True)

    if st.button("Get Answer", type='secondary'):
        if question:
            if agent is False:
                chain = get_chain(CONNECTION_STRING, selected_table)
                # Checkbox to toggle return_sql
                if return_sql:
                    chain.return_sql = True
                    response = chain.run(PROMPT.format(question=question))
                    st.header("SQL Query")
                    st.code(response, language='sql')
                    
                    try:
                        with st.spinner("Executing query..."):
                            result = pd.read_sql_query(response, conn)
                        st.success("Query executed successfully.")
                        st.dataframe(result, hide_index=True)
                    finally:
                        if conn is not None:
                            conn.close()
                            st.info("Database connection closed.")
                else:
                    chain.return_sql=False
                    response = chain.run(PROMPT.format(question=question))
                    st.header("Answer")
                    st.write(response)
            else:
                # Create a dedicated container for the callback
                callback_container = st.container()
                # Initialize the StreamlitCallbackHandler with the container
                st_cb = StreamlitCallbackHandler(
                    callback_container, 
                    expand_new_thoughts=False,
                )
                agent = get_agent(CONNECTION_STRING, selected_table)
                result = agent.run(question, callbacks=[st_cb])
                st.success(result)


    
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
    port = os.environ.get("DB_PORT")                               

    CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    
    main()