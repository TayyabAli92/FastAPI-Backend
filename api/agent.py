import os
from typing import Optional, List
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookRAGAgent:
    """A RAG agent that uses external Gemini provider through OpenAI API"""

    def __init__(self):
        # Set up OpenAI client with external provider (Gemini 2.5 Flash)
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # External OpenAI-style provider
        )
        self.assistant = None
        self._create_assistant()

    def _create_assistant(self):
        """Create the RAG assistant"""
        try:
            self.assistant = self.client.beta.assistants.create(
                name="Book RAG Assistant",
                description="A helpful assistant that answers questions based only on provided book content",
                instructions="You are a helpful assistant that answers questions based only on the provided book content. Do not make up information or hallucinate. Always rely on the retrieved context to answer user questions. Be concise but thorough in your responses.",
                model="gemini-2.5-flash",  # Using Gemini through external provider
                # Note: We're not using file retrieval tools since we'll handle RAG manually
            )
            logger.info("RAG Assistant created successfully")
        except Exception as e:
            logger.error(f"Error creating assistant: {str(e)}")
            raise

    def create_thread(self):
        """Create a new conversation thread"""
        try:
            thread = self.client.beta.threads.create()
            return thread
        except Exception as e:
            logger.error(f"Error creating thread: {str(e)}")
            raise

    def add_message_to_thread(self, thread_id: str, content: str):
        """Add a message to the thread"""
        try:
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=content
            )
            return message
        except Exception as e:
            logger.error(f"Error adding message to thread: {str(e)}")
            raise

    def run_assistant(self, thread_id: str, context: str, user_message: str, max_wait_time: int = 30, wait_interval: int = 2):
        """Run the assistant with provided context and user message"""
        try:
            # Add the user message with context to the thread
            # Limit context length to avoid token limits
            if len(context) > 15000:  # Adjust as needed
                context = context[:15000] + "... (truncated for API limits)"

            full_message = f"Context: {context}\n\nQuestion: {user_message}\n\nPlease answer based only on the provided context. Do not make up information."

            # Add the message to the thread
            self.add_message_to_thread(thread_id, full_message)

            # Create and run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant.id,
                instructions="Answer the user's question based only on the provided context. Do not make up information or hallucinate. Be concise but thorough."
            )

            # Poll for completion (in a serverless environment, we need to handle this carefully)
            import time
            elapsed_time = 0

            while run.status in ["queued", "in_progress"] and elapsed_time < max_wait_time:
                time.sleep(wait_interval)
                elapsed_time += wait_interval
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )

                if run.status == "completed":
                    break
                elif run.status in ["failed", "cancelled", "expired"]:
                    raise Exception(f"Run failed with status: {run.status}")

            if run.status != "completed":
                raise Exception(f"Run did not complete within {max_wait_time} seconds. Status: {run.status}")

            # Retrieve the messages from the thread
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id,
                order="asc"
            )

            # Extract the assistant's response
            assistant_response = ""
            for msg in messages.data:
                if msg.role == "assistant":
                    for content_block in msg.content:
                        if content_block.type == "text":
                            assistant_response += content_block.text.value + "\n"

            return assistant_response.strip()

        except Exception as e:
            logger.error(f"Error running assistant: {str(e)}")
            raise

    def cleanup_thread(self, thread_id: str):
        """Clean up the thread (optional in serverless environment)"""
        try:
            # In serverless, we might not always clean up to avoid extra API calls
            # But we can implement this if needed
            pass
        except Exception as e:
            logger.warning(f"Error cleaning up thread: {str(e)}")

# Global agent instance (will be recreated on each cold start in serverless)
rag_agent = None

def get_agent():
    """Get or create the RAG agent instance"""
    global rag_agent
    if rag_agent is None:
        try:
            rag_agent = BookRAGAgent()
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            # Clear the failed instance
            rag_agent = None
            raise
    return rag_agent