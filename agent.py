import json

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
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

load_dotenv()

with open('examples.json', 'r') as file:
    examples = json.load(file)

db_uri = "sqlite:///./mydb.db"
db = SQLDatabase.from_uri(db_uri)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

prefix = """
You are an agent designed to interact with an SQL database.
You will receive questions about pharmaceutical prescription data. 
Create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
If the user requests a chart or a table, you will return a JSON of Plotly in the answer, wrap it between "===Plotly===" and "===EndPlotly===".
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table. Only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

If the question does not seem related to the database (except require plot a chart, a table ...), just return "I don't know" as the answer.

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
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
    handle_parsing_errors=True,
    handle_sql_errors=True,
)
