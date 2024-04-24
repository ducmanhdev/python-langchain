import json
from typing import Any, Optional
from uuid import UUID

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

from agent import agent_executor_with_message_history, msgs
from utils.split_plotly import split_plotly

TEMP_USER_ID = 1
TEMP_CONSERVATION_ID = 1

st.title("Demo")


# if "history" in agent_executor_with_message_history:
#     st.session_state.messages = []
#     for message in agent_executor_with_message_history.history:
#         content, plotly_json = split_plotly(message.content)
#         st.session_state.messages.append({
#             "role": message.type,
#             "content": content,
#             "plotly_json": plotly_json
#         })
#
# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {
#             "role": "ai",
#             "content": "Can I help you?"
#         },
#     ]
#
# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if ("plotly_json" in message) and (message["plotly_json"] is not None):
#             st.plotly_chart(message["plotly_json"], use_container_width=True)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, plotly_container, initial_text=""):
        self.container = container
        self.plotly_container = plotly_container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

    def on_tool_end(
            self,
            output: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        plotly_data = json.loads(output)
        self.plotly_container.plotly_chart(plotly_data, use_container_width=True)


for msg in msgs.messages:
    content, plotly_json = split_plotly(msg.content)
    with st.chat_message(msg.type):
        st.markdown(content)
        if plotly_json and plotly_json is not None:
            st.plotly_chart(plotly_json, use_container_width=True)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    # st.session_state.messages.append({
    #     "role": "user",
    #     "content": prompt
    # })
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)

    # Display assistant response in chat message container
    with st.chat_message("ai"):
        stream_handler = StreamHandler(
            container=st.empty(),
            plotly_container=st.empty()
        )
        agent_executor_with_message_history.invoke(
            {"input": prompt},
            {
                # "configurable": {
                #     "user_id": TEMP_USER_ID,
                #     "conversation_id": TEMP_CONSERVATION_ID
                # },
                "configurable": {
                    "session_id": "any"
                },
                "callbacks": [stream_handler]
            }
        )
        # output: str = response["output"]
        # content, plotly_json = split_plotly(output)
        # st.markdown(content)
        # if plotly_json and plotly_json is not None:
        #     st.plotly_chart(plotly_json, use_container_width=True)

        # st.session_state.messages.append({
        #     "role": "ai",
        #     "content": content,
        #     "plotly_json": plotly_json
        # })
