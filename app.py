"""
FastAPI application for Book RAG Agent
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
import logging
from connection import ConnectionManager
from agent import initialize_agent, rag_query_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Book RAG Agent API",
    description="Serverless RAG API for book content queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import asyncio
from datetime import datetime, timedelta

# Simple in-memory session storage (for MVP)
# In production, use a proper database or Redis
sessions: Dict[str, Dict[str, Any]] = {}

# Session cleanup settings
SESSION_TIMEOUT_MINUTES = 30

async def cleanup_expired_sessions():
    """Background task to remove expired sessions"""
    while True:
        try:
            current_time = datetime.now()
            expired_sessions = []

            for session_id, session_data in sessions.items():
                last_activity_str = session_data.get("last_activity", session_data.get("created_at"))
                if last_activity_str:
                    last_activity = datetime.fromisoformat(last_activity_str.replace('Z', '+00:00').split('.')[0] + '.000000+00:00')
                    if (current_time - last_activity).total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
                        expired_sessions.append(session_id)

            for session_id in expired_sessions:
                del sessions[session_id]

            await asyncio.sleep(60)  # Run cleanup every minute
        except Exception as e:
            print(f"Error during session cleanup: {e}")

@app.on_event("startup")
async def startup_event():
    """Start background session cleanup task"""
    asyncio.create_task(cleanup_expired_sessions())

class ChatRequest(BaseModel):
    message: str
    selected_text: Optional[str] = None
    session_id: Optional[str] = None

class Citation(BaseModel):
    chunk_id: str
    text: str
    similarity_score: float
    source_info: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    session_id: str
    timestamp: str
    mode: str = "rag"

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Input validation
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Log the incoming request
        logger.info(f"Received chat request for session: {request.session_id}")

        # Determine session
        session_id = request.session_id or str(uuid.uuid4())

        # Initialize session if not exists
        if session_id not in sessions:
            sessions[session_id] = {
                "mode": "rag" if request.selected_text is None else "selected",
                "selected_text": request.selected_text,
                "message_history": [],
                "created_at": datetime.now().isoformat()
            }

        # Update session mode based on request
        if request.selected_text is not None:
            sessions[session_id]["mode"] = "selected"
            sessions[session_id]["selected_text"] = request.selected_text
        elif request.selected_text is None and sessions[session_id]["mode"] == "selected":
            sessions[session_id]["mode"] = "rag"
            sessions[session_id]["selected_text"] = None

        # Get the current session
        session = sessions[session_id]

        # Determine rag_mode
        rag_mode = session["mode"]

        # Initialize agent (in a real implementation, this would be done once at startup)
        agent = initialize_agent()

        # In a full implementation with OpenAI Agents SDK, we would use Runner.run() here
        # For now, we'll use rag_query tool directly to get context
        if rag_mode == "rag":
            retrieved_context = rag_query_tool(request.message, "rag", 3)  # top_k = 3
        else:
            # For selected text mode, we need to pass the selected text as chunks
            selected_chunks = [session["selected_text"]] if session["selected_text"] else []
            retrieved_context = rag_query_tool(request.message, "selected", 3)  # top_k = 3

        # Create citations from retrieved context
        citations = []
        for item in retrieved_context:
            citation = Citation(
                chunk_id=item.get("chunk_id", ""),
                text=item.get("text", ""),
                similarity_score=item.get("similarity_score", 0.0),
                source_info=item.get("metadata", {})
            )
            citations.append(citation)

        # In a real implementation, we would use Runner.run() with the agent to generate the response
        # For now, we'll create a simple response based on the retrieved context
        if retrieved_context:
            response_text = f"Based on the book content, here's what I found: {retrieved_context[0].get('text', 'No content found')[:200]}..."
        else:
            response_text = "I couldn't find relevant information in the book content to answer your question."

        # Update last activity timestamp
        session["last_activity"] = datetime.now().isoformat()

        # Add to message history
        session["message_history"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        session["message_history"].append({
            "role": "assistant",
            "content": response_text,
            "citations": [c.dict() for c in citations],
            "timestamp": datetime.now().isoformat()
        })

        response = ChatResponse(
            response=response_text,
            citations=citations,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            mode=rag_mode
        )

        # Log successful response
        logger.info(f"Successfully processed chat request for session: {session_id}")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        # Raise a generic error to avoid exposing internal details
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing the request")