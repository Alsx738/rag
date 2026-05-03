from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.product_finder import product_finder, finder_tools_node
from agents.recommender import recommender, recommender_tools_node
from agents.synthesizer import synthesizer
from agents.orchestrator import orchestrator


def _build_graph_builder():
    graph_builder = StateGraph(AgentState)

    # Orchestrator routes queries to the appropriate subgraph
    graph_builder.add_node("orchestrator", orchestrator)

    graph_builder.add_node("product_finder", product_finder)
    graph_builder.add_node("finder_tools", finder_tools_node)
    graph_builder.add_node("recommender", recommender)
    graph_builder.add_node("recommender_tools", recommender_tools_node)
    graph_builder.add_node("synthesizer", synthesizer)

    graph_builder.add_edge(START, "orchestrator")

    # Conditional routing after orchestrator: inspect the last SystemMessage
    # produced by the orchestrator (expected content: "ROUTE:PRODUCT_SEARCH" or
    # "ROUTE:CONVERSATION") and return the next node name. We pop the routing
    # message so it does not leak into downstream agent prompts.
    def orchestrator_route(state: AgentState):
        if not state.messages:
            return "synthesizer"
        last = state.messages[-1]
        content = getattr(last, "content", "")
        token = content.strip().upper()
        if token.startswith("ROUTE:PRODUCT"):
            # remove routing token from history
            try:
                state.messages.pop()
            except Exception:
                pass
            return "product_finder"
        if token.startswith("ROUTE:CONVERSATION"):
            try:
                state.messages.pop()
            except Exception:
                pass
            return "synthesizer"
        # default to conversation
        return "synthesizer"

    graph_builder.add_conditional_edges("orchestrator", orchestrator_route)

    # Finder routing
    def finder_route(state: AgentState):
        if hasattr(state.messages[-1], "tool_calls") and state.messages[-1].tool_calls:
            return "finder_tools"
        return "recommender"

    graph_builder.add_conditional_edges("product_finder", finder_route)
    graph_builder.add_edge("finder_tools", "product_finder")

    # Recommender routing
    def recommender_route(state: AgentState):
        if hasattr(state.messages[-1], "tool_calls") and state.messages[-1].tool_calls:
            return "recommender_tools"
        return "synthesizer"

    graph_builder.add_conditional_edges("recommender", recommender_route)
    graph_builder.add_edge("recommender_tools", "recommender")

    graph_builder.add_edge("synthesizer", END)

    return graph_builder


def create_agent(checkpointer=None):
    return _build_graph_builder().compile(checkpointer=checkpointer)
