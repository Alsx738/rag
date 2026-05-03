from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from agents.state import AgentState
from agents.tools import get_other_user_reviews
from utility.llm import chat_llm

recommender_tools = [get_other_user_reviews]
llm_recommender = chat_llm.bind_tools(recommender_tools)

RECOMMENDER_PROMPT = """You are the Recommender Agent. Your task is to read the Finder Agent's findings.
Behaviour:
- If the the Finder has already provided tool results in the conversation, use your tool once to find related products reviewed by the same user (excluding the primary product).
- If there are no Finder results, do not call your tool.

Take the User ID of the primary product found, the user's search context (e.g. the original query), and the Product ID to exclude. Stay within the domain of the request (do not suggest unrelated categories).
Internally report the related results and the reason why they are relevant."""

def recommender(state: AgentState) -> dict:
    messages = [SystemMessage(content=RECOMMENDER_PROMPT)] + state.messages
    response = llm_recommender.invoke(messages)
    return {"messages": [response]}

recommender_tools_node = ToolNode(tools=recommender_tools)
