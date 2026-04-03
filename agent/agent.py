# LangGraph agent definition.
# ReAct loop: agent reasons and calls tools until it has enough info to answer.

import logging
import time

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
import sqlite3 as _sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 15

from app.config import MODEL_NAME
from agent.state import AgentState
from agent.tools import ALL_TOOLS
from agent.prompts import SYSTEM_PROMPT, VERIFY_PROMPT

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
        args_str = str(call["args"])[:100]
        t0 = time.perf_counter()
        result = tool.invoke(call["args"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result_str = str(result)
        logger.info(
            "Tool call: %s | args: %s | %.0fms | %d chars",
            call["name"], args_str, elapsed_ms, len(result_str),
        )
        results.append(
            ToolMessage(content=result_str, tool_call_id=call["id"], name=call["name"])
        )
    return {
        "messages": results,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
    }


def _limit_node(state: AgentState) -> dict:
    """Inject a final message when the tool call limit is reached."""
    return {
        "messages": [
            AIMessage(
                content="I've reached the maximum number of tool calls for "
                "this query. Here's what I found so far based on the "
                "information I've gathered above."
            )
        ]
    }


def _verify_node(state: AgentState) -> dict:
    """Score confidence of the agent's answer against retrieved evidence."""
    model = init_chat_model(MODEL_NAME, temperature=0)
    response = model.invoke(
        [SystemMessage(content=VERIFY_PROMPT)] + state["messages"]
    )
    # Extract the confidence line and append it as a footer to the answer
    confidence_line = response.content.strip()
    last_answer = state["messages"][-1].content
    updated = f"{last_answer}\n\n---\n_{confidence_line}_"
    return {
        "messages": [AIMessage(content=updated)],
        "verified": True,
    }


def _should_continue(state: AgentState) -> str:
    """Route after the agent node.
    If the LLM made tool calls, go to the tools node.
    If it produced a final response, go to verify (once).
    If we hit the tool call limit, go to the limit node."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
            return "limit"
        return "tools"
    if not state.get("verified", False):
        return "verify"
    return END


def build_graph(checkpointer=None):
    """Construct and compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tool_node)
    graph.add_node("limit", _limit_node)
    graph.add_node("verify", _verify_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", _should_continue, ["tools", "limit", "verify", END]
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("verify", END)
    graph.add_edge("limit", END)

    if checkpointer is None:
        conn = _sqlite3.connect("data/checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)


# Default compiled graph with in-memory checkpointing
agent = build_graph()