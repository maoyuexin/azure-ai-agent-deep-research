from fastapi import FastAPI
# Initialize FastAPI app
from fastapi import FastAPI, HTTPException, status
 
from dotenv import load_dotenv
global toolset
 
from models import *

import asyncio
from typing import Annotated
 
 
import os, time,re
from typing import Optional
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import DeepResearchTool, MessageRole, ThreadMessage,  ListSortOrder
from azure.storage.blob import BlobServiceClient


def upload_blob_file(container_name, file_name, connection_string):

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container=container_name)
    try:
        with open(file=file_name, mode="rb") as data:
            container_client.upload_blob(name=file_name, data=data, overwrite=True)
        return {
            "status": "success",
            "message": f"File {file_name} uploaded to container {container_name} successfully."
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": f"Failed to upload {file_name} to {container_name}: {str(e)}"
        }


def convert_citations_to_superscript(markdown_content):
    """
    Convert citation markers in markdown content to HTML superscript format.
    
    This function finds citation patterns like 【78:12†source】 and converts them to 
    HTML superscript tags <sup>12</sup> for better formatting in markdown documents.
    
    Args:
        markdown_content (str): The markdown content containing citation markers
        
    Returns:
        str: The markdown content with citations converted to HTML superscript format"
    """
    # Pattern to match 【number:number†source】
    pattern = r'【\d+:(\d+)†source】'
    
    # Replace with <sup>captured_number</sup>
    def replacement(match):
        citation_number = match.group(1)
        return f'<sup>{citation_number}</sup>'
    
    return re.sub(pattern, replacement, markdown_content)


def fetch_and_print_new_agent_response(
    thread_id: str,
    agents_client: AgentsClient,
    last_message_id: Optional[str] = None,
    progress_filename: str = "research_progress.txt",
) -> Optional[str]:
    """
    Fetch the interim agent responses and citations from a thread and write them to a file.
    
    Args:
        thread_id (str): The ID of the thread to fetch messages from
        agents_client (AgentsClient): The Azure AI agents client instance
        last_message_id (Optional[str], optional): ID of the last processed message 
            to avoid duplicates. Defaults to None.
        progress_filename (str, optional): Name of the file to write progress to. 
            Defaults to "run_progress.txt".
            
    Returns:
        Optional[str]: The ID of the latest message if new content was found, 
            otherwise returns the last_message_id
    """
    response = agents_client.messages.get_last_message_by_role(
        thread_id=thread_id,
        role=MessageRole.AGENT,
    )
    if not response or response.id == last_message_id:
        return last_message_id  # No new content

    # if not a "cot_summary" return
    if not any(t.text.value.startswith("cot_summary:") for t in response.text_messages):
        return last_message_id    

    with open(progress_filename, "a", encoding="utf-8") as fp:
        fp.write("\nAGENT>\n")
        fp.write("\n".join(t.text.value.replace("cot_summary:", "Reasoning:") for t in response.text_messages))
        fp.write("\n")

        for ann in response.url_citation_annotations:
            fp.write(f"Citation: [{ann.url_citation.title}]({ann.url_citation.url})\n")

    return response.id




def create_research_summary(
        message : ThreadMessage,
        filepath: str = "research_report.md"
) -> None:
    """
    Create a formatted research report from an agent's thread message with numbered citations 
    and a references section.
    
    Args:
        message (ThreadMessage): The thread message containing the agent's research response
        filepath (str, optional): Path where the research summary will be saved. 
            Defaults to "research_report.md".
            
    Returns:
        None: This function doesn't return a value, it writes to a file
    """
    if not message:
        print("No message content provided, cannot create research report.")
        return

    with open(filepath, "w", encoding="utf-8") as fp:
        # Write text summary
        text_summary = "\n\n".join([t.text.value.strip() for t in message.text_messages])
        # Convert citations to superscript format
        text_summary = convert_citations_to_superscript(text_summary)
        fp.write(text_summary)

        # Write unique URL citations with numbered bullets, if present
        if message.url_citation_annotations:
            fp.write("\n\n## Citations\n")
            seen_urls = set()
            citation_dict = {}
            
            for ann in message.url_citation_annotations:
                url = ann.url_citation.url
                title = ann.url_citation.title or url
                
                if url not in seen_urls:
                    # Extract citation number from annotation text like "【58:1†...】"
                    citation_number = None
                    if ann.text and ":" in ann.text:
                        match = re.search(r'【\d+:(\d+)', ann.text)
                        if match:
                            citation_number = int(match.group(1))
                    
                    if citation_number is not None:
                        citation_dict[citation_number] = f"[{title}]({url})"
                    else:
                        # Fallback for citations without proper format
                        citation_dict[len(citation_dict) + 1] = f"[{title}]({url})"
                    
                    seen_urls.add(url)
            
            # Write citations in numbered order
            for num in sorted(citation_dict.keys()):
                fp.write(f"{num}. {citation_dict[num]}\n")

    print(f"Research report written to '{filepath}'.")



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
                        progress_filename=f"report/research_progre_{thread_id}_{run.id}.txt",
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
                    file_path = f"report/research_report_{thread_id}_{run.id}.md"
                    create_research_summary(final_message, file_path)
                    # Upload the research report to Azure Blob Storage 
                    upload_result = upload_blob_file(container_name= os.environ.get("AZURE_STORAGE_ACCOUNT_CONTAINER_NAME"), file_name = file_path, connection_string= f"DefaultEndpointsProtocol=https;AccountName={os.environ.get('AZURE_STORAGE_ACCOUNT_NAME')};AccountKey={os.environ.get('AZURE_STORAGE_ACCOUNT_KEY')};EndpointSuffix=core.windows.net")
                    report_blob_path = None
                    if upload_result["status"] == "success":
                        report_blob_path = f"https://{os.environ.get('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net/{os.environ.get('AZURE_STORAGE_ACCOUNT_CONTAINER_NAME')}/{file_path}"    
                    
        return ChatResponse(content="\n\n".join([t.text.value.strip() for t in final_message.text_messages]),  thread_id=thread_id, report_blob_path = report_blob_path)

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


 