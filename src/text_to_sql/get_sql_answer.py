import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_core.messages import SystemMessage, HumanMessage
# from langgraph.prebuilt import chat_agent_executor

# Constants
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

def get_db(connection_string, tables=None):
    """
    Create and return a SQLDatabase object.

    This function establishes a connection to a SQL database using the provided connection string.
    It can optionally focus on a specific table and includes a sample of rows for table info.

    Args:
        connection_string (str): The connection string for the database.
        table (str, optional): The name of a specific table to include. Defaults to None.

    Returns:
        SQLDatabase: An object representing the connected database.

    Example:
        db = get_db("postgresql://user:password@localhost/dbname", "users")
    """
    db_kwargs = {
        "sample_rows_in_table_info": 3  # Include 3 sample rows in table info
    }

    if tables:
        if isinstance(tables, str):
            db_kwargs["include_tables"] = [tables]
        elif isinstance(tables, list):
            db_kwargs["include_tables"] = tables
        else:
            raise ValueError("tables must be a string or a list of strings")

    return SQLDatabase.from_uri(connection_string, **db_kwargs)

def get_llm(provider='ollama'):
    """
    Create and return a language model based on the specified provider.

    This function initializes and returns a language model object based on the chosen provider.
    It supports OpenAI, Anthropic, and Ollama models.

    Args:
        provider (str): The provider of the language model. 
                        Options: 'openai', 'anthropic', 'ollama'. Defaults to 'ollama'.

    Returns:
        object: An instance of the specified language model.

    Raises:
        ValueError: If an unsupported LLM provider is specified.

    Example:
        llm = get_llm('openai')
    """
    if provider == 'openai':
        return ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], temperature=0)
    elif provider == 'anthropic':
        return ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
    elif provider == 'ollama':
        return ChatOllama(model='phi3')
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def get_chain(connection_string, table, llm):
    """
    Create and return a SQLDatabaseSequentialChain.

    This function sets up a chain that can translate natural language queries into SQL,
    execute them, and return the results.

    Args:
        connection_string (str): The connection string for the database.
        table (str): The name of the table to query.
        llm_provider (str): The provider for the language model. Defaults to 'ollama'.

    Returns:
        SQLDatabaseSequentialChain: A chain object for processing SQL queries.

    Example:
        chain = get_chain("postgresql://user:password@localhost/dbname", "users", "openai")
    """
    db = get_db(connection_string, table)
    return SQLDatabaseSequentialChain.from_llm(
        llm=llm,
        db=db,
        verbose=True,  # For detailed output
        return_intermediate_steps=True,  # To see intermediate steps
        use_query_checker=True,  # To check and potentially correct SQL queries
    )

def get_agent(connection_string, agent_type, table, llm):
    """
    Create and return a SQL agent.

    This function creates an agent capable of interacting with a SQL database,
    understanding natural language queries, and returning results.

    Args:
        connection_string (str): The connection string for the database.
        agent_type (AgentType): The type of agent to create.
        llm_provider (str): The provider for the language model. Defaults to 'ollama'.

    Returns:
        AgentExecutor or None: An agent executor if successful, None if creation fails.

    Example:
        agent = get_agent("postgresql://user:password@localhost/dbname", 
                          AgentType.ZERO_SHOT_REACT_DESCRIPTION, "anthropic")
    """
    db = get_db(connection_string, table)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    try:
        return create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type=agent_type,
            prefix=SQL_PREFIX,
            verbose=True,  # For detailed output
            agent_executor_kwargs={"return_intermediate_steps": True},
            handle_parsing_errors=True  # To handle parsing errors
        )
    except ValueError as e:
        print(f"Error creating SQL agent: {e}")
        return None
