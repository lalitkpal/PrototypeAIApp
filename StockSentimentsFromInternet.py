import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import load_tools
import configparser
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# setup API keys
config = configparser.ConfigParser()
config.read("config.ini")
os.environ["SERPAPI_API_KEY"] = config["Keys"]["serp_api_key"]
os.environ["GROQ_API_KEY"] = config["Keys"]["groq_api_key"]

# setup LLM API
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# Initialize the agent with custom tools
tools = load_tools(["serpapi"])

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
