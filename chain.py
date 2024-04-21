import json

from operator import itemgetter
from typing import List
from dotenv import load_dotenv
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate, MessagesPlaceholder,
)

load_dotenv()

history = ChatMessageHistory()

with open('examples.json', 'r') as file:
    examples = json.load(file)

db_uri = "sqlite:///./mydb.db"
db = SQLDatabase.from_uri(db_uri)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# generate_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)
# query = generate_query.invoke({"question": "How many items in table account`"})
# response = execute_query.invoke(query)

answer_prompt = PromptTemplate.from_template(
    """
    Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: 
    """
)

rephrase_answer = answer_prompt | llm | StrOutputParser()

# chain = (
#         RunnablePassthrough.assign(query=generate_query).assign(
#             result=itemgetter("query") | execute_query
#         )
#         | rephrase_answer
# )
# response = chain.invoke({"question": "How many items in table account?"})
# print(response)

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

vectorstore = Chroma()
vectorstore.delete_collection()
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    vectorstore,
    k=2,
    input_keys=["input"],
)
example_selector.select_examples({"input": "Show me top 10 team 1 territories by pomalyst trx volume growth"})
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input", "top_k"],
)

system = """
    You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specificed.
    Here is the relevant table info: {table_info}
    Below are a number of examples of questions and their corresponding SQL queries. Those examples are just for referecne and hsould be considered while answering follow up questions"
"""
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)
generate_query = create_sql_query_chain(llm, db, final_prompt)


# chain = (
#         RunnablePassthrough.assign(query=generate_query).assign(
#             result=itemgetter("query") | execute_query
#         )
#         | rephrase_answer
# )


# response = chain.invoke({"question": "How many territory under G1000000?"})
# print(response)


def get_table_details():
    table_names = db.get_usable_table_names()
    formatted_table_names = "\n".join([f"Table Name: {name}" for name in table_names])
    return formatted_table_names


table_details = get_table_details()


class Table(BaseModel):
    name: str = Field(description="Name of table in SQL database.")


table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:
{table_details}
Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""


# table_chain = create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)
# tables = table_chain.invoke({"input": "Show me top 10 team 1 territories by pomalyst trx volume growth"})
# print(tables)


def get_tables(tables: List[Table]) -> List[str]:
    tables = [table.name for table in tables]
    return tables


select_table = (
        {"input": itemgetter("question")}
        | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)
        | get_tables
)

chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table) |
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        )
        | rephrase_answer
)

response = chain.invoke({
    "question": "Give me address of them",
    "messages": history.messages
})

print(response)
