import os

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain

from langchain.prompts import PromptTemplate

from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
# from langgraph.prebuilt import chat_agent_executor

from langchain_community.chat_models import ChatOllama

def get_db(CONNECTION_STRING, table):
    # TODO: Get database connection
    db = SQLDatabase.from_uri(
        CONNECTION_STRING,
        include_tables=[table],
        sample_rows_in_table_info=3
    )

    return db

def get_llm():
    # llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],temperature=0)
    llm = ChatOllama(model='phi3', format='json')

    return llm

def get_chain(CONNECTION_STRING, table):
    # TODO: Get chain
    chain = SQLDatabaseSequentialChain.from_llm(
        llm=get_llm(),
        db=get_db(CONNECTION_STRING, table),
        # top_k=10,
        verbose=True,
        return_intermediate_steps=True,
        use_query_checker=True,
    )

    return chain

def get_agent(CONNECTION_STRING, agent_type):

    db = SQLDatabase.from_uri(
        CONNECTION_STRING,
        # include_tables=[table],
        sample_rows_in_table_info=3
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=get_llm())
    tools = toolkit.get_tools()


    SQL_PREFIX = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""

    # Message for priming AI behavior
    system_message = SystemMessage(content=SQL_PREFIX)

    # agent_executor = chat_agent_executor.create_tool_calling_executor(
    #     llm, tools, messages_modifier=system_message
    # )

    # for s in agent_executor.stream(
    # {"messages": [HumanMessage(content=question)]}
    # ):  
    #     print(s)
    #     print("----")

    # Create the SQL agent with additional debugging
    try:
        agent_executor = create_sql_agent(
            llm=get_llm(), 
            toolkit=toolkit,
            agent_type=agent_type,  # Must be one of 'openai-tools', 'openai-functions', or 'zero-shot-react-description'
            prefix=SQL_PREFIX,
            verbose=True,
            agent_executor_kwargs={"return_intermediate_steps": True},
            handle_parsing_errors=True
        )
    except ValueError as e:
        print("Error creating SQL agent:", e)
    
    return agent_executor

