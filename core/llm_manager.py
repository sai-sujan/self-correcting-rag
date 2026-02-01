import os
from langchain_ollama import ChatOllama
from core.config import llm as default_llm, llm_small as default_llm_small

# Global instances that can be updated at runtime
_current_llm = default_llm
_current_llm_small = default_llm_small

def get_llm():
    """Get the current main LLM instance."""
    global _current_llm
    if _current_llm is None:
        return default_llm
    return _current_llm

def get_llm_small():
    """Get the current small LLM instance."""
    global _current_llm_small
    if _current_llm_small is None:
        return default_llm_small
    return _current_llm_small

def set_temperatures(llm_temp: float, llm_small_temp: float):
    """Dynamically update LLM temperatures (used for experiments)."""
    global _current_llm, _current_llm_small
    
    print(f"[LLM Manager] Setting temperatures: Main={llm_temp}, Small={llm_small_temp}")
    
    _current_llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        temperature=llm_temp
    )
    
    _current_llm_small = ChatOllama(
        model="qwen2.5:3b",
        temperature=llm_small_temp
    )

def reset_llm_config():
    """Reset to defaults defined in core.config."""
    global _current_llm, _current_llm_small
    _current_llm = default_llm
    _current_llm_small = default_llm_small
