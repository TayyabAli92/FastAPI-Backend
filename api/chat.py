import json
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from .agent import get_agent
from .qdrant_rag import get_qdrant_rag

# Define request model
class ChatRequest(BaseModel):
    message: str
    selected_text: Optional[str] = None

# Define response model
class ChatResponse(BaseModel):
    response: str

def retrieve_from_qdrant(query: str, top_k: int = 5) -> str:
    """Retrieve relevant documents from Qdrant based on the query"""
    try:
        qdrant_rag = get_qdrant_rag()
        return qdrant_rag.get_relevant_content(query, top_k)
    except Exception as e:
        logger.error(f"Error retrieving from Qdrant: {str(e)}")
        return "Error retrieving content from the book."

def retrieve_from_selected_text(selected_text: str, query: str) -> str:
    """Process and return context from selected text"""
    # In a real implementation, you might process the selected_text to extract relevant parts
    # For now, we'll return the selected text with query context
    return f"Selected Text Context:\n{selected_text}\n\nQuery: {query}\n\nPlease answer based on this selected text."

def run_rag_chat(message: str, selected_text: Optional[str] = None) -> str:
    """Main function to run the RAG chat"""
    try:
        # Validate environment variables
        if not os.getenv("GEMINI_API_KEY"):
            raise Exception("GEMINI_API_KEY not configured")

        if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
            raise Exception("Qdrant configuration not set")

        # Determine which RAG mode to use
        if selected_text:
            # Use Selected Text RAG mode
            context = retrieve_from_selected_text(selected_text, message)
        else:
            # Use normal Qdrant RAG mode
            context = retrieve_from_qdrant(message)

        # Get the agent instance
        agent = get_agent()

        # Create a thread for the conversation
        thread = agent.create_thread()

        # Run the assistant with context and user message
        # Reduced timeout parameters for serverless environment
        # Using shorter max_wait_time to avoid timeout issues
        response = agent.run_assistant(thread.id, context, message, max_wait_time=25, wait_interval=1)

        # Clean up the thread
        try:
            agent.cleanup_thread(thread.id)
        except:
            pass  # Ignore cleanup errors in serverless environment

        return response.strip() if response else "I couldn't find relevant information to answer your question."

    except Exception as e:
        logger.error(f"Error in RAG chat: {str(e)}")
        # Return a user-friendly error message
        if "timeout" in str(e).lower() or "time" in str(e).lower():
            return "Request timed out. Please try again with a shorter query."
        elif "api" in str(e).lower() or "key" in str(e).lower():
            return "API configuration error. Please check your API keys."
        else:
            return "An error occurred while processing your request. Please try again."

# FastAPI app for local testing (Vercel will use the handler directly)
app = FastAPI()

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint that handles user queries with RAG"""
    try:
        response = run_rag_chat(request.message, request.selected_text)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Vercel serverless function handler
def handler(event, context):
    """Vercel serverless function handler"""
    try:
        # Parse the incoming request
        body = json.loads(event.get('body', '{}'))
        message = body.get('message', '')
        selected_text = body.get('selected_text', None)

        if not message:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': json.dumps({'error': 'Message is required'})
            }

        # Run the RAG chat
        response = run_rag_chat(message, selected_text)

        # Return the response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({'response': response})
        }
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({'error': f'Error processing request: {str(e)}'})
        }

# For Vercel Python runtime, we also provide a standard entry point
def main():
    """Standard entry point for local testing"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    main()