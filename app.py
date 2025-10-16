from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, List
from uuid import uuid4
import json
from pydantic import BaseModel, EmailStr, Field
from agents import Agent, Runner, function_tool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory=".")


class UserProfileSchema(BaseModel):
    """Schema defining what profile information to collect from user"""
    name: str = Field(..., description="User's full name")
    email: EmailStr = Field(..., description="User's email address")
    company: str = Field(..., description="User's company or organization")
    role: str = Field(..., description="User's job title or role")
    experience_years: int = Field(..., description="Years of professional experience", ge=0)
    team: Optional[str] = Field(None, description="User's team or department")
    skills: Optional[List[str]] = Field(None, description="Key technical skills")

sessions: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_class=HTMLResponse)
async def root():
    return templates.TemplateResponse("client.html", {"request": {}})


@function_tool
def save_answer(session_id: str, key: str, raw_text: str) -> str:
    """Save a user's answer to the session."""
    s = sessions.get(session_id)
    if s is None:
        return "error:session_not_found"
    s["answers"][key] = raw_text
    s["history"].append({"key": key, "value": raw_text})
    return json.dumps({"saved_key": key, "status": "saved"})

@function_tool
def get_session_data(session_id: str) -> str:
    """Retrieve current session data for context."""
    s = sessions.get(session_id)
    if s is None:
        return "error:session_not_found"
    return json.dumps({"answers": s["answers"], "history": s["history"]})

agent = Agent(
    name="ConversationalOnboarder",
    instructions=(
        "You are a friendly, conversational onboarding assistant engaged in a natural dialogue. "
        "You must collect the following information from the user in a conversational way:\n"
        f"Required fields: {', '.join([name for name, field in UserProfileSchema.model_fields.items() if field.is_required()])}\n"
        f"Optional fields: {', '.join([name for name, field in UserProfileSchema.model_fields.items() if not field.is_required()])}\n"
        "Ask clear, one question at a time. When the user provides an answer, use save_answer(session_id, key, raw_text) "
        "to persist their response with the appropriate field name as the key. "
        "Accept only the user's words - no special formatting needed from them. "
        "Maintain a conversational tone and gradually build their profile through natural back-and-forth dialogue. "
        "When you have gathered all required information, summarize what you've learned and conclude the conversation."
    ),
    tools=[save_answer, get_session_data],
)
@app.websocket("/ws/onboard")
async def websocket_onboard(ws: WebSocket):
    await ws.accept()
    session_id = None
    
    try:
        # Wait for initial connection message
        init_msg = await ws.receive_text()
        parsed = json.loads(init_msg)
        
        # Create new session
        if isinstance(parsed, dict) and parsed.get("action") == "start":
            session_id = str(uuid4())
            sessions[session_id] = {
                "answers": {},
                "history": [],
                "schema": parsed.get("schema", {}),
                "turn_count": 0
            }
            await ws.send_text(json.dumps({"type": "session_started", "session_id": session_id}))
        else:
            await ws.send_text(json.dumps({"type": "error", "message": "invalid_start_payload"}))
            await ws.close()
            return
    except (json.JSONDecodeError, WebSocketDisconnect):
        await ws.send_text(json.dumps({"type": "error", "message": "connection_failed"}))
        await ws.close()
        return
    
    # Initial greeting from LLM
    try:
        context = {
            "session_id": session_id,
            "action": "start_conversation",
            "state": sessions[session_id]
        }
        result = Runner.run_streamed(agent, input=json.dumps(context))
        
        full_message = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                try:
                    delta = event.data.delta
                    full_message += delta
                    await ws.send_text(json.dumps({"type": "delta", "delta": delta}))
                except Exception:
                    pass
            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "tool_call_item":
                    await ws.send_text(json.dumps({"type": "tool_call", "tool": getattr(item, "tool_name", None)}))
                elif item.type == "tool_call_output_item":
                    try:
                        output = json.loads(getattr(item, "output", "{}"))
                        await ws.send_text(json.dumps({"type": "tool_output", "status": "saved"}))
                    except Exception:
                        pass
        
        await ws.send_text(json.dumps({"type": "message_complete"}))
        sessions[session_id]["last_llm_message"] = full_message
        sessions[session_id]["turn_count"] += 1
        
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        await ws.close()
        return
    
    # Conversation loop: receive user text, send to LLM, stream response
    while True:
        try:
            user_input = await ws.receive_text()
            
            # Check if it's a close signal
            if user_input.lower() in ["exit", "quit", "done", "bye"]:
                await ws.send_text(json.dumps({"type": "conversation_ended", "message": "Thank you for chatting. Your profile has been saved."}))
                break
            
            # Build context for LLM with user's text response
            context = {
                "session_id": session_id,
                "action": "process_user_message",
                "user_message": user_input,
                "state": sessions[session_id]
            }
            
            result = Runner.run_streamed(agent, input=json.dumps(context))
            
            full_message = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    try:
                        delta = event.data.delta
                        full_message += delta
                        await ws.send_text(json.dumps({"type": "delta", "delta": delta}))
                    except Exception:
                        pass
                elif event.type == "run_item_stream_event":
                    item = event.item
                    if item.type == "tool_call_item":
                        tool_name = getattr(item, "tool_name", None)
                        await ws.send_text(json.dumps({"type": "tool_call", "tool": tool_name}))
                    elif item.type == "tool_call_output_item":
                        try:
                            output = json.loads(getattr(item, "output", "{}"))
                            if "saved_key" in output:
                                await ws.send_text(json.dumps({"type": "answer_saved", "key": output.get("saved_key")}))
                            else:
                                await ws.send_text(json.dumps({"type": "tool_output", "status": "success"}))
                        except Exception:
                            pass
            
            await ws.send_text(json.dumps({"type": "message_complete"}))
            sessions[session_id]["last_llm_message"] = full_message
            sessions[session_id]["turn_count"] += 1
            
        except WebSocketDisconnect:
            break
        except json.JSONDecodeError:
            await ws.send_text(json.dumps({"type": "error", "message": "invalid_input"}))
            continue
        except Exception as e:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            continue

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve session profile data."""
    if session_id not in sessions:
        return {"error": "session not found"}
    s = sessions[session_id]
    return {
        "session_id": session_id,
        "answers": s["answers"],
        "history": s["history"],
        "turns": s["turn_count"]
    }
