__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import warnings

import streamlit as st

from tools import Tools

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    tools = Tools()

    st.title("Clinical Trials Research Q&A Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        history = "\n\n".join([f"{x['role']}: {x['content']}" for x in st.session_state.messages])
        response = tools.query_rag(query=prompt, history=history)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
