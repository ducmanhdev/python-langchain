import streamlit as st
import json

from agent import agent_executor

st.title("Demo")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
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
        response = agent_executor.invoke({"input": prompt})

    output: str = response["output"]
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        START_FLAG = "===Plotly==="
        ENG_FLAG = "===EndPlotly==="
        if START_FLAG in output and ENG_FLAG in output:
            try:
                start = output.find(START_FLAG) + len(START_FLAG)
                end = output.find(ENG_FLAG)
                region = output[start:end].strip()

                content = region[:output.find(START_FLAG)].strip()
                st.markdown(content)

                plotly_json = json.loads(region)
                st.plotly_chart(plotly_json, use_container_width=True)

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": content,
                    "plotly_json": plotly_json
                })
            except Exception:
                st.write("Sorry, I couldn't find the plot!")
        else:
            st.markdown(output)
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": output,
            })
