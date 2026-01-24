import streamlit as st
import sys
import os
import time

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import StreamlitCallbackHandler

# Import strict v7 graph construction
from experiments.opt_v7_temperature_tuning.graph import create_graph, set_temperatures

# Page Configuration
st.set_page_config(
    page_title="Self-Correcting RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #333;
    }
    
    .stChatMessage.user {
        background-color: #2B313E;
        border-left: 4px solid #4CAF50;
    }
    
    .stChatMessage.assistant {
        background-color: #1E1E1E;
        border-left: 4px solid #2196F3;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #333;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #2196F3;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1976D2;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"user-{int(time.time())}"

if "graph" not in st.session_state:
    # Initialize graph with Config D (Winner)
    # LLM=0.3, Small=0.3
    # We call create_graph without arguments because set_temperatures handles the global state
    # But wait, create_graph in v7 doesn't take arguments, it uses global state.
    # We must explicitly set temperatures first.
    set_temperatures(llm_temp=0.3, llm_small_temp=0.3, config_name="D (Winner)")
    st.session_state.graph = create_graph()

# Sidebar
with st.sidebar:
    st.title("⚙️ RAG Settings")
    st.markdown("---")
    
    st.markdown("### Model Configuration")
    st.info("**Config D (Winner)** Active")
    st.caption("Main LLM: `qwen2.5:7b` (Temp: 0.3)")
    st.caption("Small LLM: `qwen2.5:3b` (Temp: 0.3)")
    
    st.markdown("---")
    st.markdown("### Debug Metrics")
    if st.checkbox("Show Latency"):
        st.session_state.show_latency = True
    else:
        st.session_state.show_latency = False
        
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"user-{int(time.time())}"
        st.rerun()

# Main Chat Interface
st.title("🤖 Self-Correcting RAG")
st.markdown("Ask questions about your documents. The system uses **multi-query search** and **self-correction**.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "latency" in msg and st.session_state.get("show_latency"):
            st.caption(f"⏱️ {msg['latency']:.2f}s")

# Chat Input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        start_time = time.time()
        
        try:
            # Config for graph execution
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Run graph
            response = st.session_state.graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config
            )
            
            # Extract answer
            answer = response["messages"][-1].content
            
            # Update history and UI
            end_time = time.time()
            latency = end_time - start_time
            
            message_placeholder.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "latency": latency
            })
            
            # Show debug info in sidebar if available
            with st.sidebar:
                st.markdown("### Last Retrieval Info")
                if "extracted_content" in response and response["extracted_content"]:
                    with st.expander("View Extracted Context"):
                        st.text(response["extracted_content"][:500] + "...")
                        
        except Exception as e:
            st.error(f"Error generating response: {e}")
            
