# Schema that flows through every node in LangGraph agent

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Conversation history
    messages: Annotated[list[BaseMessage], add_messages]
    # Number of tool call rounds executed (safety limit)
    tool_call_count: int