import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an helpful assistant"),
    ("user", "{input}")
])
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

st.title("Demo")
st.chat_message("assistant").write("Hello, can I help you!")
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    response = chain.invoke({"input": prompt})
    print(response)
    with st.chat_message("assistant"):
        st.write(response)
