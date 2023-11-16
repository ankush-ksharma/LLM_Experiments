import os
from dotenv import load_dotenv
load_dotenv()

from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import load_tools

llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name= "gpt-3.5-turbo")


wikipedia = WikipediaAPIWrapper()
wiki  = Tool(
    name='Wikipedia Search',
    func= wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia."
)


search = DuckDuckGoSearchRun()
d_search  = Tool(
    name='DuckDuckgo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

python_repl = PythonREPL()
python_code_runner  =  Tool(
        name = "python repl",
        func=python_repl.run,
        description="useful for when you need to use python to answer a question. You should input only python code."
)


tools = [wiki, d_search, python_code_runner]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            You should check todays date and time before starting your research. Find the most relevant inforation based on todays date and then proceed.
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are things you want to search for latest information use search tool.
            3/ You should use wikipedia tool to gather more information
            4/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            5/ You should not make things up, you should only write facts & data that you have gathered
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            7/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

try:
    agent = initialize_agent(
        tools=tools,
        agent="zero-shot-react-description",
        llm=llm,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
        max_iterations=3
    )
except Exception as e:
    print(f"error in loading agent {e}")

def main():
    content = agent({"input": "Find out the score of latest cricket world cup match."})
    actual_content = content['output']
    print(actual_content)

if __name__ == '__main__':
    main()
