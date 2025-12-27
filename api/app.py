"""
Vercel-optimized FastAPI application for Book RAG Agent
This file is optimized for Vercel serverless deployment
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
import logging

from agent import initialize_agent, rag_query_tool
from connection import ConnectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Book RAG Agent API",
    description="Serverless RAG API for book content queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://localhost:3000"
        ],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session storage (for MVP)
# In production, use a proper database or Redis
sessions: Dict[str, Dict[str, Any]] = {}

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

@app.get("/")
async def root():
    return {"message": "Book RAG Agent API - Ready to serve RAG queries"}

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

        # Initialize agent
        agent = initialize_agent()

        # Use rag_query tool to get context
        if rag_mode == "rag":
            retrieved_context = rag_query_tool(request.message, "rag", 3)  # top_k = 3
        else:
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

        # Create response based on retrieved context
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

# Additional endpoints to match what your frontend is looking for
class AskRequest(BaseModel):
    message: str
    selected_text: Optional[str] = None
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    response: str
    citations: List[Citation]
    session_id: str
    timestamp: str

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """Endpoint that matches your frontend's /ask request"""
    # Convert AskRequest to ChatRequest format
    chat_request = ChatRequest(
        message=request.message,
        selected_text=request.selected_text,
        session_id=request.session_id
    )

    # Call the main chat endpoint logic
    chat_response = await chat_endpoint(chat_request)

    # Convert to AskResponse format
    ask_response = AskResponse(
        response=chat_response.response,
        citations=chat_response.citations,
        session_id=chat_response.session_id,
        timestamp=chat_response.timestamp
    )

    return ask_response

@app.post("/api/v1/rag", response_model=AskResponse)
async def rag_endpoint(request: AskRequest):
    """Endpoint that matches your frontend's /api/v1/rag request"""
    # Convert AskRequest to ChatRequest format
    chat_request = ChatRequest(
        message=request.message,
        selected_text=request.selected_text,
        session_id=request.session_id
    )

    # Call the main chat endpoint logic
    chat_response = await chat_endpoint(chat_request)

    # Convert to AskResponse format
    ask_response = AskResponse(
        response=chat_response.response,
        citations=chat_response.citations,
        session_id=chat_response.session_id,
        timestamp=chat_response.timestamp
    )

    return ask_response

@app.post("/api/v1/ask-agent", response_model=AskResponse)
async def ask_agent_endpoint(request: AskRequest):
    """Endpoint that matches your frontend's /api/v1/ask-agent request"""
    # Convert AskRequest to ChatRequest format
    chat_request = ChatRequest(
        message=request.message,
        selected_text=request.selected_text,
        session_id=request.session_id
    )

    # Call the main chat endpoint logic
    chat_response = await chat_endpoint(chat_request)

    # Convert to AskResponse format
    ask_response = AskResponse(
        response=chat_response.response,
        citations=chat_response.citations,
        session_id=chat_response.session_id,
        timestamp=chat_response.timestamp
    )

    return ask_response

# The ASGI application that Vercel will serve
# The variable name 'app' is what Vercel looks for by default