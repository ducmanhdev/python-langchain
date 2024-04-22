import streamlit as st

from agent import agent_executor_with_message_history
from utils.split_plotly import split_plotly

TEMP_USER_ID = 1
TEMP_CONSERVATION_ID = 1

st.title("Demo")

if "history" in agent_executor_with_message_history:
    st.session_state.messages = []
    for message in agent_executor_with_message_history.history:
        content, plotly_json = split_plotly(message.content)
        st.session_state.messages.append({
            "role": message.type,
            "content": content,
            "plotly_json": plotly_json
        })
# TODO: Transform history to session_state.messages

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "content": "Can I help you you?"
        },
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plotly_json" in message:
            st.plotly_chart(message["plotly_json"], use_container_width=True)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display spinner while waiting for response
    with st.spinner(text="Waiting for response..."):
        # Invoke agent_executor to get response
        response = agent_executor_with_message_history.invoke(
            {"input": prompt},
            {
                "configurable": {
                    "user_id": TEMP_USER_ID,
                    "conversation_id": TEMP_CONSERVATION_ID
                }
            }
        )

    output: str = response["output"]
    # Display assistant response in chat message container
    with st.chat_message("ai"):
        START_FLAG = "===Plotly==="
        ENG_FLAG = "===EndPlotly==="
        if START_FLAG in output and ENG_FLAG in output:
            try:
                content, plotly_json = split_plotly(output)
                st.markdown(content)
                st.plotly_chart(plotly_json, use_container_width=True)

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "ai",
                    "content": content,
                    "plotly_json": plotly_json
                })
            except Exception as e:
                print(e)
                st.write("Sorry, I couldn't find the plot!")

        else:
            st.markdown(output)
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "ai",
                "content": output,
            })
