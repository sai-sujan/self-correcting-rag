import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from experiments.baseline.graph import agent_graph

# Page config
st.set_page_config(
    page_title="Self-Correcting RAG",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("🤖 Self-Correcting RAG System")
st.markdown("""
Ask questions about your documents! The system will:
- Search relevant chunks  
- Retrieve full context
- Generate accurate answers
""")

# Sidebar
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Try asking:")
    st.markdown("- What is JavaScript?")
    st.markdown("- How do I install it?")
    st.markdown("- What is blockchain?")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is JavaScript?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            try:
                result = agent_graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config
                )
                
                response = result["messages"][-1].content
                st.markdown(response)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})