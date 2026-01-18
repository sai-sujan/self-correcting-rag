from graph import agent_graph
from langchain_core.messages import HumanMessage

# Create a conversation thread
config = {"configurable": {"thread_id": "test-1"}}

# Ask a question
result = agent_graph.invoke(
    {"messages": [HumanMessage(content="Give some example projects on it ")]},
    config
)

# Print the answer
print("\n" + "="*50)
print("ANSWER:")
print("="*50)
print(result["messages"][-1].content)