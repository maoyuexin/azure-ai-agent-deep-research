
from pydantic import BaseModel
from typing import List, Optional


class ChatHistoryInput(BaseModel):
    question: str

class ChatHistoryOutput(BaseModel):
    answer: str

class SingleChatHistory(BaseModel):
    inputs: ChatHistoryInput
    outputs: ChatHistoryOutput

class ChatHistory(BaseModel):
    content: str
    name: str
    role: str

class ChatRequest(BaseModel):
    question: str
    agent_id: str
    thread_id: Optional[str] = None
    tool_name: Optional[List[str]] = []
    chatHistory: Optional[List[SingleChatHistory]] = []
 

class ChatResponse(BaseModel):
    content: str
    chatHistory: List[ChatHistory]
    thread_id:str
    followupQuestions: Optional[List[str]] = None

 
class CreateAgentRequest(BaseModel):
    agent_name: str
    instructions: str
    agent_description: Optional[str] = None
    agent_instructions: Optional[str] = None

class CreateAgentResponse(BaseModel):
    status: str
    agent_name: str
    agent_id: str
     