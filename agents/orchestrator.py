from langchain_core.messages import SystemMessage
from agents.state import AgentState
from utility.llm import chat_llm


ORCHESTRATOR_PROMPT = """You are the Orchestrator. Your job is to read the user's last message and decide whether
it is a product search request or a conversational/memory request.

If the user is clearly asking to find or recommend products (contains words like 'product', 'find', 'looking for', 'buy', 'search'),
return the single token: PRODUCT_SEARCH

If the user is asking about personal info, memory, or general conversation (e.g., 'what is my name', 'remember that I', 'who am I', 'hello'),
return the single token: CONVERSATION

If unsure, default to CONVERSATION.
"""


def orchestrator(state: AgentState) -> dict:
    messages = [SystemMessage(content=ORCHESTRATOR_PROMPT)] + state.messages
    response = chat_llm.invoke(messages)
    # We expect the model to return either PRODUCT_SEARCH or CONVERSATION
    token = response.content.strip().upper()
    if "PRODUCT" in token:
        # Redirect to product finder
        # We encode the routing decision as a message that the graph can inspect
        routing_msg = SystemMessage(content="ROUTE:PRODUCT_SEARCH")
    else:
        routing_msg = SystemMessage(content="ROUTE:CONVERSATION")
    return {"messages": [routing_msg]}
