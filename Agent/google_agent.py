import os
from dotenv import load_dotenv
load_dotenv()

from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import load_tools

llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name= "gpt-3.5-turbo")


tools = load_tools(
    ['llm-math'],
    llm=llm
)
search = GoogleSearchAPIWrapper()
google_search  = Tool(
    name='Google Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

tools.append(google_search)

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are things you want to search for latest information use langchain search tool.
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should use Calculator tool to do calculations.
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

agent = initialize_agent(
    tools=tools,
    agent="zero-shot-react-description",
    llm=llm,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    max_iterations=3
)

def main():
    content = agent({"input": "Find out the score of latest cricket world cup match"})
    actual_content = content['output']
    print(actual_content)

if __name__ == '__main__':
    main()
