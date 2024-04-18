import os
from typing import Any

import pandas as pd
import httpx
from IPython.display import SVG, display
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor, create_openapi_agent
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ChatMessage,
)

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.llms import AzureOpenAI

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
import json

class XsuntAgent:
    def __init__(self, db_uri: str, initial_message: str, examples_message: str, examples: list[Any],
                 anonymizer: PresidioReversibleAnonymizer, verbose: bool = False):
        # Setup connection to OpenAI and database

        # Below code is to connect mitmproxt, comment out if no needed.
        # http_client = httpx.Client(verify=False, proxies="http://localhost:8080/")

        # llm = ChatOpenAI(
        #     model_name="gpt-3.5-turbo", temperature=0
        # )

        llm = AzureChatOpenAI(
            openai_api_version="2024-02-01",
            azure_deployment="xsunt-ai",
            temperature=0.7,
            # http_client=http_client
        )

        db = SQLDatabase.from_uri(db_uri)

        # Create prompt
        messages = [
            ("system", initial_message.format(table_info=db.get_table_info())),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            ("system", "{examples_message}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

        # Create example selector
        self.examples_message = examples_message
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            OpenAIEmbeddings(),
            FAISS,
            k=5,
            input_keys=["input"]
        )

        # Python functions used for running queries
        def run_query(sql_query: str) -> pd.DataFrame:
            # sql_query = anonymizer.deanonymize(sql_query)
            return pd.DataFrame(db._execute(sql_query))

        def run_query_str(sql_query: str) -> str:
            # sql_query = anonymizer.deanonymize(sql_query)
            return self.db.run_no_throw(sql_query)

        def plot_to_json(fig) -> json:
            json_data = fig.to_json()
            return json_data

        def save_plot_json(fig_json):
            # globals()["fig_json"] = fig_json
            self.fig_json = json.loads(fig_json)

        # Setup Python REPL for creating charts
        python_repl = PythonREPL()
        python_repl.run("import pandas as pd")
        python_repl.run("import numpy as np")
        python_repl.run("import plotly as plty")
        python_repl.globals["run_query"] = run_query
        python_repl.globals['plot_to_json'] = plot_to_json
        python_repl.globals['save_plot_json'] = save_plot_json

        # Setup tools
        tools = [
            Tool(
                name="sql_query_tool",
                description=(
                    "Input to this tool is a detailed and correct Microsoft SQL Server query. "
                    "Output is a response from the database."
                ),
                func=run_query_str
            ),
            PythonREPLTool(python_repl=python_repl),
        ]

        self.python_repl = python_repl
        self.fig_json = None

        # Create agent
        chat_history = ChatMessageHistory()
        chat_history.add_message(
            AIMessage("Hello, my name is XSUNT AI. How can I help you today?")
        )
        agent = RunnableWithMessageHistory(
            AgentExecutor(
                agent=create_openai_functions_agent(llm, tools, prompt),
                tools=tools,
                verbose=verbose,
            ),
            lambda _: chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # Assign to instance variables
        self.llm = llm
        self.db = db
        self.run_query = run_query
        self.chat_history = chat_history
        self.anonymizer = anonymizer
        self.agent = agent
        self.fig = None

    def invoke(self, input: str):
        # clear old fig_json
        if self.fig_json:
            self.fig_json = None

        examples = self.example_selector.select_examples({"input": input})
        examples_text = "\n\n".join([f"-- {e['input']}\n{e["query"]}" for e in examples])
        examples_message = self.examples_message.format(examples_text=examples_text)

        # input = self.anonymizer.anonymize(input)
        # print(input)
        self.fig = None
        config = {"configurable": {"session_id": "foo"}}
        res = self.agent.invoke({"input": input, "examples_message": examples_message}, config)
        # self.fig = self.python_repl.locals["fig_json"]

        # print("globals:")
        # print(self.python_repl.globals)
        # fig_json = self.python_repl.globals.get("fig_json", None)
        return res, self.fig_json


# plot a line chart of G0000000 2024 pomalyst monthly sales