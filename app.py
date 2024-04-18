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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Can I help you you?"},
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = chain.invoke({"input": prompt})
        st.write(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
