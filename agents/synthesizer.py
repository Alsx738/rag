from langchain_core.messages import SystemMessage
from agents.state import AgentState
from utility.llm import chat_llm

llm_synthesizer = chat_llm

SYNTHESIZER_PROMPT = """You are the Synthesizer Agent. Your task is to address the end user directly, setting aside all technicalities.
Read the conversation and the information gathered by the Finder and the Recommender.

Behaviour:
- If the conversation contains product findings from the Finder (Product ID lines or tool outputs), produce the mandatory product recommendation format.
- If the user's message is conversational or asks about personal info (e.g., 'what is my name', 'who am I', 'remember that I'), use the conversation history and any saved checkpoints to answer directly and warmly.

Mandatory format example for products:
"The product I recommend is [Primary Product Name/Summary] (ID: [Insert_Product_ID_Here]) because [qualities identified by the Finder]... Some users also purchased [Related Product] (ID: [Insert_Related_Product_ID_Here]) because [analysis from the Recommender]."

You MUST always explicitly include the full Product ID codes for every product you mention, so the user knows what to buy. Avoid the word 'agent' and never mention the use of tools.
"""

def synthesizer(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYNTHESIZER_PROMPT)] + state.messages
    response = llm_synthesizer.invoke(messages)
    return {"messages": [response]}
