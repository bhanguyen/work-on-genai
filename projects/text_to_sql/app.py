import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine 

from utils import connect_to_db, get_tables, load_table
from get_sql_answer import get_chain, get_agent

from langchain_community.callbacks import StreamlitCallbackHandler

# TODO: Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Text-To-SQL",
        layout="wide",
    )
    
    with st.sidebar:
        st.title("Text-To-SQL :computer:")

        st.markdown(
            f"""
            ### Instructions:"""
        )
            # 1. Select `table` to preview data
            # `{selected_table}`
            # """
        # Set database connection
        database = st.sidebar.text_input('Enter your database name', 'brisbane2032')
        load_dotenv()

        host = os.environ.get("DB_HOST")                                          
        user = os.environ.get("DB_USER")                                      
        password = os.environ.get("DB_PASSWORD")

        CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{database}"
        conn = connect_to_db(database)
        table = get_tables(conn)
        selected_table = st.sidebar.selectbox("Select table to preview", table)
        st.markdown(
            """
            2. Ask question
            3. Select `Use Agent`, then select the agent type (Optional)
            4. Select `Return SQL` to return the executed query or direct answer
            5. Select `Use Table` to focus on limit context
            6. Enter or Press 'Get Answer'
            """
        )
    
    # Preview table
    df = load_table(selected_table, conn)
    st.dataframe(df, hide_index=True)

    question = st.text_input("Ask Question: ")

    PROMPT = """ 
Given an input question, first create a syntactically correct postgresql query to run,  
then look at the results of the query and return the answer.  
The question: {question}
"""
    agent = st.checkbox("Use Agent")
    if agent is False:
        # return_sql = st.checkbox("Return SQL", value=True)
        use_table = st.checkbox("Use Table", value=True)
    else:
        agent_type = st.selectbox(
                    'Choose your agent type',
                    ('openai-tools','openai-functions','zero-shot-react-description')
                )
        st.info("Each agent type will dictate the toolkit used")

    if st.button("Get Answer", type='secondary'):
        if question:
            if agent is False:
                if use_table:
                    table = selected_table
                    chain = get_chain(CONNECTION_STRING, selected_table)
                    st.write(f"Using {table}")

                    tab1, tab2 = st.tabs(["Answer", "Chain logs"])

                    # Get response
                    with st.spinner("Getting the answer..."):
                        response = chain(PROMPT.format(question=question))

                    with tab1:
                        # st.header("Answer")
                        st.success(response['result'])
                        final_query = response['intermediate_steps'][1]
                        st.subheader("Executed SQL")
                        st.code(final_query, language='sql')
                        
                        st.subheader("Result Table")
                        try:
                            with st.spinner("Executing query..."):
                                result = pd.read_sql_query(final_query, conn)
                            st.success("Query executed successfully.")
                            st.dataframe(result, hide_index=True)
                        finally:
                            if conn is not None:
                                conn.close()
                                st.info("Database connection closed.")
                    with tab2:
                        st.subheader("Chain logs")
                        # container = st.container(height=300, border=True)
                        # container.write(response)
                        response

                else:
                    st.info("Only support single table for now")
            
            else:
                # Create a dedicated container for the callback
                callback_container = st.container()
                # Initialize the StreamlitCallbackHandler with the container
                st_cb = StreamlitCallbackHandler(
                    callback_container, 
                    expand_new_thoughts=False,
                )
                agent = get_agent(CONNECTION_STRING, agent_type) #, selected_table)
                result = agent(question, callbacks=[st_cb])
                st.success(result['input'])
                # result_1 = agent.__call__(question, callbacks=[st_cb])
                # st.success(result_1)
    
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
        
    main()