import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine 

# Import langchain modules
from langchain_community.callbacks import StreamlitCallbackHandler

# Import custom modules
from text_to_sql.utils import connect_to_db, get_tables, load_table
from text_to_sql.get_sql_answer import get_chain, get_agent, get_llm


# Load environment variables
load_dotenv()

# Constants
HOST = os.environ.get("DB_HOST")
USER = os.environ.get("DB_USER")
PASSWORD = os.environ.get("DB_PASSWORD")

# Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Text-To-SQL",
        layout="wide",
    )
    
    st.title("Text-To-SQL :computer:")

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Configuration:")
        
        # Database connection setup
        database = st.sidebar.text_input('Enter your database name', '')
        CONNECTION_STRING = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:5432/{database}"
        
        try:
            conn = connect_to_db(database)
            all_tables = get_tables(conn)
            selected_tables = st.multiselect("Select tables to use", all_tables, default=all_tables[0] if all_tables else None)
        except Exception as e:
            st.warning('Please enter the correct database name')
            st.error(f"Error: {e}")
        
        # LLM selection
        llm_options = ['openai', 'anthropic', 'ollama']
        selected_llm = st.selectbox("Select Language Model", llm_options, index=2)  # Default to 'ollama'
        
        st.markdown(
            """
            ### Instructions:
            1. Select tables to use
            2. Choose a Language Model
            3. Ask a question
            4. Select `Use Agent` and agent type (Optional)
            5. Enter or Press 'Get Answer'
            """
        )

    # Main content area
    # Table preview
    if selected_tables:
        for table in selected_tables:
            st.subheader(f"Preview of {table}")
            try:
                df = load_table(table, conn)
                st.dataframe(df.head(), hide_index=True)
            except Exception as e:
                st.error(f"Error loading table {table}: {e}")

    # User input for question
    question = st.text_input("Ask Question: ")

    # SQL generation prompt
    PROMPT = """ 
    Given an input question, first create a syntactically correct postgresql query to run,  
    then look at the results of the query and return the answer.  
    The question: {question}
    """

    # Agent configuration
    use_agent = st.checkbox("Use Agent")

    if use_agent:
        agent_type = st.selectbox(
            'Choose your agent type',
            ('openai-tools', 'openai-functions', 'zero-shot-react-description')
        )
        st.info("Each agent type will dictate the toolkit used")

    # Get Answer button
    if st.button("Get Answer", type='secondary'):
        if question and selected_tables:
            # Initialize LLM
            llm = get_llm(selected_llm)

            if not use_agent:
                # Using chain for query
                chain = get_chain(CONNECTION_STRING, selected_tables, llm)

                tab1, tab2 = st.tabs(["Answer", "Chain logs"])

                # Get response
                with st.spinner("Getting the answer..."):
                    response = chain(PROMPT.format(question=question))

                with tab1:
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
                    st.json(response)
            
            else:
                # Using agent for more complex queries
                callback_container = st.container()
                st_cb = StreamlitCallbackHandler(
                    callback_container, 
                    expand_new_thoughts=False,
                )
                agent = get_agent(CONNECTION_STRING, agent_type, selected_tables, llm)
                result = agent(question, callbacks=[st_cb])
                st.success(result['output'])
        else:
            st.warning("Please enter a question and select at least one table.")

    # Sample questions in sidebar
    with st.sidebar:
        st.divider()
        st.markdown(
            """
            ### Sample questions to get started:
            1. Question 1 
            2. Question 2
            3. Question 3
            """
        )

if __name__ == '__main__':
    main()