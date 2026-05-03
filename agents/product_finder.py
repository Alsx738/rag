from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from agents.state import AgentState
from agents.tools import get_similar_reviews
from utility.llm import chat_llm

finder_tools = [get_similar_reviews]
llm_finder = chat_llm.bind_tools(finder_tools)

FINDER_PROMPT = """You are the Finder Agent. Your primary job is to use the provided tool to find
relevant Amazon product reviews when the user's request is a product search.

Behaviour:
- If the most recent conversation messages already contain tool results, do NOT call the tool again.
- If no tool results are present, call the tool exactly once and then produce an internal
    technical report indicating the chosen Product ID, the reviewer's User ID, and a logical
    summary of why it is a good match.

Only use the tool when necessary. If the conversation is about personal info or general chat,
do not call the tool and instead leave the message for the Synthesizer.
"""

def product_finder(state: AgentState) -> dict:
    messages = [SystemMessage(content=FINDER_PROMPT)] + state.messages
    response = llm_finder.invoke(messages)
    return {"messages": [response]}

finder_tools_node = ToolNode(tools=finder_tools)
