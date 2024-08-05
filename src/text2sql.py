import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import sys

# Import langchain modules
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor

# Import custom modules
from applications.text_to_sql.utils import connect_to_db, get_tables, load_table
from applications.text_to_sql.get_sql import get_chain, get_agent, get_llm

# Load environment variables
load_dotenv()

# Constants
HOST = os.environ.get("DB_HOST")
USER = os.environ.get("DB_USER")
PASSWORD = os.environ.get("DB_PASSWORD")
PORT = os.environ.get("DB_PORT")

# Main Streamlit app function
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Text-To-SQL",
        layout="wide",
    )
    
    st.title("Text-To-SQL :computer:")

    # Sidebar configuration
    with st.sidebar:
        # Instructions for users
        st.markdown(
            """
            ### Instructions:
            1. Select tables to use
            2. Choose Language Model(s)
            3. Ask a question
            4. Select `Use Agent` (Optional)
            5. Enter or Press 'Get Answer'
            """
        )

        st.markdown("### Configuration:")
        st.write("Configure your database connection")
        # Database connection setup
        with st.expander("Database Configuration"):
            USER = st.text_input('Enter your username', '', placeholder="Username")
            PASSWORD = st.text_input('Enter your password', '', placeholder="Password", type="password")
            HOST =  st.text_input('Enter your host', '', placeholder="Host")
            PORT = st.text_input('Enter your port', '', placeholder="Port")
            database = st.text_input('Enter your database name', '', placeholder="Database name")
            CONNECTION_STRING = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{database}"

        # Attempt to connect to the database and get table information
        try:
            conn = connect_to_db(database)
            all_tables = get_tables(conn)
            selected_tables = st.multiselect("Select tables to use", all_tables, default=all_tables[0] if all_tables else None)
        except Exception as e:
            st.warning('Please enter the correct database name')
            st.error(f"Error: {e}")
    
    # Table preview
    if selected_tables:
        tabs = st.tabs(selected_tables)
        for i, tab in enumerate(tabs):
            with tab:
                table = selected_tables[i]
                st.subheader(f"Preview of {table}")
                try:
                    df = load_table(table, conn)
                    st.dataframe(df.head(), hide_index=True)
                except Exception as e:
                    st.error(f"Error loading table {table}: {e}")

    with st.sidebar:
        st.subheader("Application Configuration")
        # Table selection
        use_table = st.sidebar.toggle("Use Tables")
        st.sidebar.warning("""Select to use the preview tables for generating SQL query.
                If left unchecked, all tables will be used.""")
        
        if use_table:
            selected_tables = selected_tables
        else:
            selected_tables = all_tables
        
        # LLM selection
        llm_options = ['OpenAI', 'Anthropic', 'Ollama']
        selected_llms = st.multiselect("Select LLMs", llm_options, default=['Ollama'])
        
        # Model selection for each LLM
        selected_models = {}
        for i, llm in enumerate(selected_llms):
            st.subheader(llm)
            if llm == 'Ollama':
                model_options = ['phi3', 'llama3.1', 'mistral', 'mistral-nemo','gemma2']
                selected_models[llm] = st.selectbox(f"Select {llm} model", model_options)
            elif llm == 'OpenAI':
                model_options = ['gpt-3.5-turbo', 'gpt-4o-mini']
                selected_models[llm] = st.selectbox(f"Select {llm} model", model_options)
            elif llm == 'Anthropic':
                model_options = ['claude-3-5-sonnet-20240620', 'claude-3-haiku-20240307']
                selected_models[llm] = st.selectbox(f"Select {llm} model", model_options)

        # Agent configuration
        use_agent = st.toggle("Use Agent")
        st.info("""The agent calls a set of tools to process your plain English inputs, 
                transforming them into precise SQL queries and executing them seamlessly.""")

    # User input for question
    question = st.text_input("Ask Question: ")

    # Get Answer button
    if st.button("Get Answer", key='get_answer'):
        st.write(f"Debug: Question - {question}")
        st.write(f"Debug: Selected Tables - {selected_tables}")
        st.write(f"Debug: Selected Models - {selected_models}")
        if question and selected_tables and selected_models:
            # Create tabs for each selected LLM
            tabs = st.tabs(selected_models.keys())
            # cols = st.columns(len(selected_models.keys()))
            
            for tab, (llm_name, model) in zip(tabs, selected_models.items()):
                with tab:
                    # Initialize LLM
                    llm = get_llm(llm_name, model)

                    if not use_agent:
                        # Using chain for query
                        chain = get_chain(CONNECTION_STRING, selected_tables, llm)

                        answer_tab, logs_tab = st.tabs(["Answer", "Chain logs"])

                        # Get response
                        with st.spinner("Getting the answer..."):
                            response = chain.invoke(question)

                        with answer_tab:
                            st.success(response['result'])
                            final_query = response['intermediate_steps'][1]
                            # final_query = response['result']
                            st.subheader("Executed SQL")
                            st.code(final_query, language='sql')
                            
                            # if st.button("Execute SQL", key='execute_sql'):
                            # st.subheader("Result Table")
                            # try:
                            #     with st.spinner("Executing query..."):
                            #         result = pd.read_sql_query(final_query, conn)
                            #     st.success("Query executed successfully.")
                            #     st.dataframe(result, hide_index=True)
                            # finally:
                            #     if conn is not None:
                            #         conn.close()
                            #         st.info("Database connection closed.")
                        
                        with logs_tab:
                            st.subheader("Chain logs")
                            st.json(response)
                    
                    else:
                        # Using agent for more complex queries
                        callback_container = st.container()
                        st_cb = StreamlitCallbackHandler(
                            callback_container, 
                            expand_new_thoughts=False,
                        )
                        
                        # Assign agent type based on LLM
                        if llm_name == 'OpenAI':
                            agent_type = 'tool-calling'
                        else:  # For Ollama and Anthropic
                            agent_type = 'zero-shot-react-description'
                            # agent_type = 'openai-tools'
                        
                        # Create agent executor
                        agent_executor = get_agent(CONNECTION_STRING, agent_type, selected_tables, llm)
                        if agent_executor:
                            try:
                                with st.spinner("Agent is working on the query..."):
                                    if isinstance(agent_executor, AgentExecutor):
                                        result = agent_executor(question, callbacks=[st_cb])
                                    else:
                                        result = agent_executor({"input": question}, callbacks=[st_cb])
                                st.write(result["output"])
                                # Extract and display the SQL query
                                # Locate the final query
                                action = result["intermediate_steps"][-1]
                                # Extract the tool_input part from action tuple object
                                tool_input = action[0].tool_input
                                # If tool_input is a dictionary, extract the 'query' field
                                if isinstance(tool_input, dict) and 'query' in tool_input:
                                    tool_input = tool_input['query']
                                # Find the SELECT part in the tool_input if it's a string
                                if isinstance(tool_input, str) and tool_input.strip().upper().startswith('SELECT'):
                                    final_query = tool_input.strip()
                                
                                if final_query:
                                    st.subheader("Final Executed Query")
                                    st.code(final_query, language='sql')
                                else:
                                    st.info("""No SELECT query was found in the agent's output. 
                                            This might be due to the nature of the question or how the agent processed it.""")
                                
                            except Exception as e:
                                st.error(f"An error occurred: {str(e)}")
                                st.error("Agent Trace:")
                                st.json(vars(agent_executor))  # This will show the full agent trace
                        else:
                            st.error("Failed to create the agent. Please check your configuration.")
        else:
            st.warning("Please enter a question, select at least one table, and choose at least one LLM.")

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