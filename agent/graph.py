from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import planner_node, retriever_node, should_continue, synthesizer_node


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_conditional_edges(
        "retriever",
        should_continue,
        {"retrieve": "retriever", "synthesize": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()


agent = build_agent()
