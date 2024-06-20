from langchain.agents import AgentExecutor,create_react_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import  PromptTemplate
from langchain.tools import Tool
from vector import kg_qa
from cypher import cypher_qa
from app import llm


# agent_prompt=hub.pull("hwchase17/react-chat")
agent_prompt = PromptTemplate.from_template("""You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

Do not answer any questions that do not relate to movies, actors or directors.


TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}""")

memory=ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)
def run_retriver(query):
    results=kg_qa.invoke({"query":query})
    return results['result']

def run_Graphretriver(query):
    results=cypher_qa.invoke({"query":query})
    return results['result']

tools=[
    Tool.from_function(
        name="Genral Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
    ),
    Tool.from_function(
            name="Vector Search Index",  # (1)
            description="Provides information about movie plots using Vector Search", # (2)
            func = run_retriver, # (3)
            return_direct=True
        ),
Tool.from_function(
        name="Graph Cypher QA Chain",  # (1)
        description="Provides information about Movies including their Actors, Directors and User reviews", # (2)
        func = run_Graphretriver, # (3)
        return_direct=True
    )
]



agent=create_react_agent(llm,tools,agent_prompt)

agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

def generate_response(prompt):
    response=agent_executor.invoke({"input":prompt})
    return response['output']