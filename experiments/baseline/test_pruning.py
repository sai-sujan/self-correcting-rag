from graph import agent_graph
from langchain_core.messages import HumanMessage,AIMessage

config = {"configurable": {"thread_id": "prune-test"}}

questions = [
    "What is JavaScript?",
    "How do I install it?",
    "What are variables?",
    "What is blockchain?",
    "How does mining work?",
    "What is a smart contract?"
]

for i, question in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"Question {i}: {question}")
    print('='*60)
    
    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config
    )
    
    # Check message count
    state = agent_graph.get_state(config)
    msg_count = len([m for m in state.values["messages"] if isinstance(m, (HumanMessage, AIMessage))])
    
    print(f"\n📊 Current message count: {msg_count}")
    print(f"📝 Summary: {state.values.get('conversation_summary', 'None')[:100]}...")