import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType, AgentExecutor
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler
from typing import Union, Dict, Any
import re

class FlexibleOutputParser(AgentOutputParser):
    """
    A flexible output parser that can interpret the output of an agent.
    
    This parser can detect "Final Answer" responses, interpret thought-action patterns,
    and handle unclear output by requesting clarification.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Parse the given text to determine the agent's response.
        
        This method identifies whether the text contains a final answer, a thought-action pattern,
        or an unclear output, and returns the appropriate response object.
        
        Args:
            text (str): The text output from the agent to be parsed.
        
        Returns:
            Union[AgentAction, AgentFinish]: An AgentAction or AgentFinish object based on the parsed text.
        """
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        
        # Check for thought-action pattern
        match = re.search(r"Thought:(.+?)Action:(.+?)(?:\n|$)", text, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            action = match.group(2).strip()
            
            # Extract action input if present
            action_input_match = re.search(r"Action Input:(.+?)(?:\n|$)", text[match.end():], re.DOTALL)
            action_input = action_input_match.group(1).strip() if action_input_match else ""
            
            return AgentAction(tool=action, tool_input=action_input, log=thought)
        
        # If no clear pattern is found, treat the whole text as a thought
        return AgentAction(tool="human", tool_input="Unclear output. Please clarify your next step.", log=text)

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

def get_llm(provider, model):
    """
    Create and return a language model based on the specified provider and model.

    This function initializes and returns a language model object based on the chosen provider and model.
    It supports OpenAI, Anthropic, and Ollama models.

    Args:
        provider (str): The provider of the language model. 
                        Options: 'OpenAI', 'Anthropic', 'Ollama'.
        model (str): The specific model to use for the chosen provider.

    Returns:
        object: An instance of the specified language model.

    Raises:
        ValueError: If an unsupported LLM provider is specified.

    Example:
        llm = get_llm('OpenAI', 'gpt-3.5-turbo')
    """
    if provider == 'OpenAI':
        return ChatOpenAI(
            model=model,
            api_key=os.environ["OPENAI_API_KEY"], 
            temperature=0
        )
    elif provider == 'Anthropic':
        return ChatAnthropic(
            model=model,
            temperature=0,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
    elif provider == 'Ollama':
        return ChatOllama(model=model)
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
        llm: The language model instance.

    Returns:
        SQLDatabaseChain: A chain object for processing SQL queries.

    Example:
        llm = get_llm("OpenAI", "gpt-3.5-turbo")
        chain = get_chain("postgresql://user:password@localhost/dbname", "users", llm)
    """
    
    custom_prompt = PromptTemplate(
        input_variables=["input", "table_info", "top_k"],
        template="""
        Given an input question, first create a syntactically correct {dialect} query to run, then return the query.
        Use the following format:

        Question: "Question here"
        SQLQuery: SQL Query to run

        Only use the following tables:
        {table_info}

        Question: {input}
        SQLQuery: 
        """
    )

    db = get_db(connection_string, table)
    return SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        prompt=custom_prompt,
        verbose=True,  # For detailed output
        # return_sql=True,
        return_intermediate_steps=True,  # To see intermediate steps
        use_query_checker=False,  # To check and potentially correct SQL queries
    )

def get_agent(connection_string, agent_type, table, llm):
    """
    Create and return a SQL agent with enhanced error handling and SQL query extraction.

    This function creates an agent capable of interacting with a SQL database,
    understanding natural language queries, and returning results. It includes
    additional error handling, a flexible output parser, and SQL query extraction.

    Args:
        connection_string (str): The connection string for the database.
        agent_type (AgentType): The type of agent to create.
        table (str): The name of the table to query.
        llm: The language model instance.

    Returns:
        tuple: (AgentExecutor, SQLQueryExtractor) if successful, (None, None) if creation fails.

    Example:
        llm = get_llm("OpenAI", "gpt-3.5-turbo")
        agent = get_agent("postgresql://user:password@localhost/dbname", 
                                           AgentType.ZERO_SHOT_REACT_DESCRIPTION, "users", llm)
    """

    SQL_PREFIX = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    DO NOT skip this step.
    Then you should query the schema of the most relevant tables.

    DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE.
    If the question does not seem related to the database, just return "I don't know" as the answer.
    ALWAYS, as part of your final answer, explain how you got to the answer on a section that starts with: 
    "Explanation:". Include the SQL query as part of the explanation section.

    Only use the below tools. Only use the information returned by the below tools to construct your query and final answer.
    DO NOT make up table names, only use the tables returned by any of the tools below.

    ## Tools: 
    """
    db = get_db(connection_string, table)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    try:
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type=agent_type,
            prefix=SQL_PREFIX,
            max_iterations=20,
            verbose=True,
            handle_parsing_errors=True,
            agent_executor_kwargs={
                "return_intermediate_steps": True,
                "handle_parsing_errors":True
            },
            agent_kwargs={
                "output_parser": FlexibleOutputParser(),
            }
        )
        return agent
    except ValueError as e:
        print(f"Error creating SQL agent: {e}")
        return None, None
