import json

from datetime import datetime
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.tools import StructuredTool

from constants import PLOTLY_START_FLAG, PLOTLY_END_FLAG
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory, ChatMessageHistory
from langchain.tools import BaseTool, Tool, tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

with open('examples.json', 'r') as file:
    examples = json.load(file)

db_uri = "sqlite:///./mydb.db"
db = SQLDatabase.from_uri(db_uri)
# context = db.get_context()
# print('list(context)', list(context))
# print('context["table_info"]', context["table_info"])
# print('context["table_names"]', context["table_names"])

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    streaming=True
)

# llm = AzureChatOpenAI(
#     openai_api_version="2024-02-01",
#     azure_deployment="xsunt-ai",
#     temperature=0.7,
# )

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)
# test = example_selector.select_examples({"input": "How many accounts we have?"})
# print('\n test', test)

response_schemas = [
    ResponseSchema(name="answer", description="The natural language answer"),
    ResponseSchema(name="plotly", description="The JSON config of the Plotly chart, table, etc."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prefix = """
You are an agent designed to interact with an SQL database and visualize data.
You will receive questions about pharmaceutical prescription data. 
Create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table. Only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

If the query returns nothing or gives an error, simply say, "Sorry, no data available."

You should know about some common abbreviations people might use:
- "TRx": "total prescriptions". This is the total number of prescriptions sold in some time period.
- "NRx": "new prescriptions". This is the number of new prescriptions sold in some time period.
- "HCP": "health care provider". Also called "prescriber" or "doctor".
- "CPD": Calls Per Day
- "Adj CPD": Adjusted Calls Per Day

When the user asks for Geo a related question, at the end of the response, add the text 'For more information, please refer to' + https://www.bmsonelook.com/HEME30/GeoMetrics.
When the user asks for a specific account, at the end of the response,  add the text 'For more information, please refer to' + https://www.bmsonelook.com/HEME30/SingleView/+account_id. Example: https://www.bmsonelook.com/HEME30/SingleView/71109949.
When the user asks for a specific HCP, at the end of the response, add the text 'For more information, please refer to' + https://www.bmsonelook.com/HEME30/SingleView/+hcp_id. Example: https://www.bmsonelook.com/HEME30/SingleView/728139.
When the user asks for rank, at the end of the response, add the text 'For more information, please refer to' https://www.bmsonelook.com/HEME30/Ranking.

Some further notes:
- Unless stated otherwise, assume the user is referring to TRx, not NRx, when asking about the number of sales.
- The dates in the "week_date" column are stored as ISO-8601 strings.
- Any 8 characters starting with letter G is "geo_code" referring to some geographical unit, like "G1M12002". This may be either at one of four levels: "NATION", "REGION", "DISTRICT", or "TERRITORY". With the exception of Any 8 characters starting with the letter G carrying "_" is "geo_pod", like "G1M1_002".


If user request a chart, a table or visualize data, you will use the tool called "generate_plotly" to do this, but you don't need to return the data from this function, just use it.

Here are some examples of user inputs and their corresponding SQL queries:
"""

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

input_variables = ["input", "dialect", "top_k"]

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="",
    input_variables=input_variables,
    partial_variables={
        'format_instructions': format_instructions
    }
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


class GeneratePlotlyInput(BaseModel):
    query_data_result: Any = Field(description="The query result")


def generate_plotly_func(query_data_result: Any) -> str:
    """Generate a plotly data"""
    chart_data = {
        "data": [{
            "type": "bar",
            "x": ["A", "B", "C", "D"],
            "y": [3, 7, 2, 5]
        }],
        "layout": {
            "title": "Basic Bar Chart",
            "xaxis": {"title": "Categories"},
            "yaxis": {"title": "Values"}
        }
    }

    # Convert the chart data to JSON
    chart_json = json.dumps(chart_data, indent=4)
    return chart_json


generate_plotly = StructuredTool.from_function(
    func=generate_plotly_func,
    name="GeneratePlotly",
    description="Useful tool when user request a chart or a table, this tool will generate Plotly and return it as a JSON.",
    args_schema=GeneratePlotlyInput,
    # return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
    handle_parsing_errors=True,
    handle_sql_errors=True,
    extra_tools=[generate_plotly]
    # format_instructions=format_instructions,
)

store = {}

# def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
#     if (user_id, conversation_id) not in store:
#         store[(user_id, conversation_id)] = ChatMessageHistory()
#     return store[(user_id, conversation_id)]

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]


msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello, how can I help you today?")

agent_executor_with_message_history = RunnableWithMessageHistory(
    agent_executor,
    # get_session_history,
    # lambda session_id: message_history,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="history",
    # history_factory_config=[
    #     ConfigurableFieldSpec(
    #         id="user_id",
    #         annotation=str,
    #         name="User ID",
    #         description="Unique identifier for the user.",
    #         default="",
    #         is_shared=True,
    #     ),
    #     ConfigurableFieldSpec(
    #         id="conversation_id",
    #         annotation=str,
    #         name="Conversation ID",
    #         description="Unique identifier for the conversation.",
    #         default="",
    #         is_shared=True,
    #     ),
    # ],
)
