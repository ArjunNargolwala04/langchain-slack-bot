# LangGraph agent definition.
# ReAct loop: agent reasons and calls tools until it has enough info to answer.

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import MODEL_NAME
from agent.state import AgentState
from agent.tools import ALL_TOOLS
from agent.prompts import SYSTEM_PROMPT

# Build a lookup so the tool node can find tools by name
_tools_by_name = {t.name: t for t in ALL_TOOLS}


def _create_model():
    """Initialize the LLM with tools bound."""
    model = init_chat_model(MODEL_NAME, temperature=0)
    return model.bind_tools(ALL_TOOLS)


def _agent_node(state: AgentState) -> dict:
    """Invoke the LLM with conversation history and available tools."""
    model = _create_model()
    response = model.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


def _tool_node(state: AgentState) -> dict:
    """Execute all tool calls from the last AI message."""
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        tool = _tools_by_name[call["name"]]
        result = tool.invoke(call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )
    return {"messages": results}


def _should_continue(state: AgentState) -> str:
    """Route after the agent node.
    If the LLM made tool calls, go to the tools node.
    If it produced a final response, end the graph."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


def build_graph(checkpointer=None):
    """Construct and compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# Default compiled graph with in-memory checkpointing
agent = build_graph()