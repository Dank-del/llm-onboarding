from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory=".")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCHEMA_PATH = "schema.json"
DATA_PATH = "collected_data.json"
TRANSCRIPT_PATH = "conversation_transcript.json"

for path in [DATA_PATH, TRANSCRIPT_PATH]:
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f) if path == DATA_PATH else json.dump([], f)

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)


class ConversationMessage(BaseModel):
    role: str
    content: str

class SaveDataRequest(BaseModel):
    data: dict
    transcript: list[ConversationMessage] = []


def generate_save_data_tool(schema_fields: list) -> dict:
    """
    Dynamically generate the save_user_data tool based on schema fields.
    
    Args:
        schema_fields: List of field definitions from schema.json
        
    Returns:
        Tool definition dict for OpenAI Realtime API
    """
    # Build properties dict from schema fields
    properties = {}
    required_fields = []
    field_descriptions = []
    
    for field in schema_fields:
        field_name = field["name"]
        field_type = field["type"]
        field_prompt = field.get("prompt", f"The user's {field_name}")
        
        # Map schema types to JSON schema types
        json_type = "string"
        if field_type == "number":
            json_type = "number"
        elif field_type == "integer":
            json_type = "integer"
        elif field_type == "boolean":
            json_type = "boolean"
        
        properties[field_name] = {
            "type": json_type,
            "description": field_prompt
        }
        
        required_fields.append(field_name)
        field_descriptions.append(f"  - {field_name}: {field_prompt}")
    
    # Add transcript field
    properties["transcript"] = {
        "type": "array",
        "description": "The full conversation transcript",
        "items": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["assistant", "user"],
                    "description": "Who spoke (assistant or user)"
                },
                "content": {
                    "type": "string",
                    "description": "What was said"
                }
            },
            "required": ["role", "content"]
        }
    }
    required_fields.append("transcript")
    
    # Generate dynamic description
    fields_list = ", ".join(required_fields[:-1])  # Exclude transcript
    description = f"Save the collected user information and conversation transcript to the database. Call this function only after you have collected ALL required fields ({fields_list}) and the user has confirmed the information is correct."
    
    return {
        "type": "function",
        "name": "save_user_data",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required_fields
        }
    }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("client.html", {"request": request})

@app.post("/session")
async def create_session(request: Request):
    """
    Creates a Realtime API session using the unified interface.
    Accepts SDP from the client and forwards it to OpenAI with session config.
    """
    import aiohttp
    
    # Get the SDP offer from the client
    sdp_offer = await request.body()
    
    # Generate the save_user_data tool dynamically from schema
    save_data_tool = generate_save_data_tool(schema['fields'])
    
    # Session configuration with tool calling
    session_config = {
        "type": "realtime",
        "model": "gpt-4o-realtime-preview-2024-12-17",
        "audio": {
            "output": {
                "voice": "alloy"
            }
        },
        "tools": [save_data_tool],
        "tool_choice": "auto",
        "instructions": f"""You are a friendly voice assistant conducting an onboarding interview.

Your task:
1. Greet the user warmly and explain you'll be collecting some information
2. Ask for each piece of information one at a time:
{json.dumps(schema['fields'], indent=2)}

3. After collecting all information, summarize everything back to the user
4. Ask for confirmation that all details are correct
5. If confirmed, call the 'save_user_data' function with all collected data AND the complete conversation transcript
6. After successfully saving, thank the user and end the conversation

Important:
- Be conversational and friendly
- If user wants to correct something, allow them to do so
- Keep track of the entire conversation for the transcript
- Only call save_user_data after user confirms all information is correct
- Include the full conversation in the transcript parameter when calling save_user_data"""
    }
    
    # Create multipart form data using aiohttp
    form_data = aiohttp.FormData()
    form_data.add_field('sdp', sdp_offer.decode('utf-8'), content_type='text/plain')
    form_data.add_field('session', json.dumps(session_config), content_type='application/json')
    
    # Forward to OpenAI Realtime API
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/realtime/calls",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            },
            data=form_data
        ) as response:
            # Log the error if there's a problem
            if response.status != 200:
                error_text = await response.text()
                print(f"OpenAI API Error: {response.status}")
                print(f"Response: {error_text}")
                response.raise_for_status()
            
            # Return the SDP answer to the client
            sdp_answer = await response.text()
            return Response(content=sdp_answer, media_type="application/sdp")


@app.post("/save")
async def save_data(request: SaveDataRequest):
    """Save collected data and conversation transcript to JSON files."""
    
    with open(DATA_PATH, "w") as f:
        json.dump(request.data, f, indent=2)
    
    transcript_data = {
        "user_data": request.data,
        "conversation": [msg.dict() for msg in request.transcript],
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }
    
    with open(TRANSCRIPT_PATH, "w") as f:
        json.dump(transcript_data, f, indent=2)
    
    print(f"✓ Saved user data: {request.data}")
    print(f"✓ Saved transcript with {len(request.transcript)} messages")
    
    return {
        "message": "Data and transcript saved successfully!",
        "data": request.data,
        "transcript_messages": len(request.transcript)
    }
