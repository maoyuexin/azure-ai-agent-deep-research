from fastapi import FastAPI
# Initialize FastAPI app
from fastapi import FastAPI, HTTPException, status
 
from dotenv import load_dotenv
global toolset
 
from models import *

import asyncio
from typing import Annotated
 

 
import os, time
from typing import Optional
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import DeepResearchTool, MessageRole, ThreadMessage,  ListSortOrder
 

def fetch_and_print_new_agent_response(
    thread_id: str,
    agents_client: AgentsClient,
    last_message_id: Optional[str] = None,
) -> Optional[str]:
    response = agents_client.messages.get_last_message_by_role(
        thread_id=thread_id,
        role=MessageRole.AGENT,
    )
    if not response or response.id == last_message_id:
        return last_message_id  # No new content

    print("\nAgent response:")
    print("\n".join(t.text.value for t in response.text_messages))

    for ann in response.url_citation_annotations:
        print(f"URL Citation: [{ann.url_citation.title}]({ann.url_citation.url})")

    return response.id


def create_research_summary(
        message : ThreadMessage,
        filepath: str
         
) -> None:
    if not message:
        print("No message content provided, cannot create research summary.")
        return

    with open(filepath, "w", encoding="utf-8") as fp:
        # Write text summary
        text_summary = "\n\n".join([t.text.value.strip() for t in message.text_messages])
        fp.write(text_summary)

        # Write unique URL citations, if present
        if message.url_citation_annotations:
            fp.write("\n\n## References\n")
            seen_urls = set()
            for ann in message.url_citation_annotations:
                url = ann.url_citation.url
                title = ann.url_citation.title or url
                if url not in seen_urls:
                    fp.write(f"- [{title}]({url})\n")
                    seen_urls.add(url)

    print(f"Research summary written to '{filepath}'.")



description = """
This is an API for configuring and calling deep research agent.
"""
app = FastAPI(title = "Agent API",
              root_path = "/api",
              description=description)

load_dotenv()

 


@app.post("/chat_deep_research_agent", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_deep_research_agent(request: ChatRequest) -> ChatResponse:
    try:
        project_client = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(),
        )
        # Create Agent with the Deep Research tool and process Agent run
        with project_client:

            with project_client.agents as agents_client:

    
                agent = agents_client.get_agent(request.agent_id)
            
                # Create thread for communication
                if not request.thread_id:
                    # If no thread ID is provided, create a new thread
                    thread = agents_client.threads.create()
                    print(f"Created thread, ID: {thread.id}")
                    thread_id = thread.id
                else:
                    # If a thread ID is provided, use it
                    thread_id = request.thread_id
                    print(f"Using existing thread, ID: {thread_id}")

                # Create message to thread
                message = agents_client.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=(request.question),
                )
                print(f"Created message, ID: {message.id}")

                print(f"Start processing the message... this may take a few minutes to finish. Be patient!")
                # Poll the run as long as run status is queued or in progress
                run = agents_client.runs.create(thread_id=thread_id, agent_id=agent.id)

                last_message_id = None
                while run.status in ("queued", "in_progress"):
                    time.sleep(5)
                    run = agents_client.runs.get(thread_id=thread_id, run_id=run.id)

                    last_message_id = fetch_and_print_new_agent_response(
                        thread_id=thread_id,
                        agents_client=agents_client,
                        last_message_id=last_message_id,
                    )
                    print(f"Run status: {run.status}")

                print(f"Run finished with status: {run.status}, ID: {run.id}")
                
                if run.status == "failed":
                    raise HTTPException(status_code=422, detail=str(f"Run failed: {run.last_error}"))

                # Fetch the final message from the agent in the thread and create a research summary
                final_message = agents_client.messages.get_last_message_by_role(
                    thread_id=thread_id, role=MessageRole.AGENT
                )
                print(final_message)
                if final_message:
                    create_research_summary(final_message, f"report/research_report_{thread_id}.md")
        
            
        return ChatResponse(content="\n\n".join([t.text.value.strip() for t in final_message.text_messages]), chatHistory=[], thread_id=thread_id, followupQuestions=None)

    except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))


@app.post("/create_deep_research_agent", response_model=CreateAgentResponse, status_code=status.HTTP_200_OK)
def create_ai_agent(request: CreateAgentRequest) -> CreateAgentResponse:
    
    try:
        project_client = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(),
        )
        #conn_id = project_client.connections.get(name=os.environ["BING_RESOURCE_NAME"]).id
        conn_id = os.environ["AZURE_BING_CONNECTION_ID"]


        # Initialize a Deep Research tool with Bing Connection ID and Deep Research model deployment name
        deep_research_tool = DeepResearchTool(
            bing_grounding_connection_id=conn_id,
            deep_research_model=os.environ["DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"],
        )

        # Create Agent with the Deep Research tool and process Agent run
        with project_client:

            with project_client.agents as agents_client:

                agent = agents_client.create_agent(
                    model=os.environ["MODEL_DEPLOYMENT_NAME"],
                    name=request.agent_name,
                    instructions= request.instructions,
                    tools=deep_research_tool.definitions,
                )

                print(f"Created agent with ID: {agent.id}")
                return CreateAgentResponse(status= "success", agent_id=agent.id, agent_name=agent.name)
     
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


 