# python 3.8 (3.8.16) or it doesn't work
# pip install streamlit streamlit-chat langchain python-dotenv
import streamlit as st
from dotenv import load_dotenv
import os
import openai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

import utils 
def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    # setup streamlit page
    st.set_page_config(
        page_title="ChatBot",
    )

def main():
    init()
    llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name= "gpt-3.5-turbo")
    template = utils.template 

    with st.sidebar:
        user_input  = st.text_area("System Message", template, height  = 500)
        if user_input:
            template = user_input
    
    template = template + utils.template_end
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

    conversation = ConversationChain(prompt=PROMPT,
                                    llm=llm,
                                    verbose=True, 
                                    memory = ConversationBufferMemory())

    for i, msg in enumerate(st.session_state["chat_history"]):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.markdown(msg.content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

    conversation.memory.chat_memory.messages = st.session_state["chat_history"]

            
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        # Use the conversation chain to generate assistant responses
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                assistant_response  = conversation.predict(input = prompt)
            st.markdown(assistant_response)
        st.session_state["chat_history"] = conversation.memory.chat_memory.messages


if __name__ == '__main__':
    main()
