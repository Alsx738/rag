from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
