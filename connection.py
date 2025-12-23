"""
Connection layer for Book RAG Agent
Handles external service connections and utility functions
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SearchParams
from qdrant_client.models import PointStruct
import numpy as np
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class Chunk(BaseModel):
    chunk_id: str
    text: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    book_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class ConnectionManager:
    def __init__(self):
        # Initialize Qdrant client
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
        else:
            # Fallback to local Qdrant if environment variables are not set
            self.qdrant_client = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", 6333))
            )

        # Initialize Gemini clients
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')  # Using gemini-2.0-flash as specified
        self.embedding_model = genai.embed_content(
            model="models/embedding-001",
            content=["sample text"],
            task_type="retrieval_document"
        )

    def get_gemini_client(self):
        """Get configured Gemini client for chat completion"""
        return self.gemini_model

    def get_embedding_client(self):
        """Get configured client for embeddings"""
        return self.gemini_model

    def get_qdrant_client(self):
        """Get configured Qdrant client"""
        return self.qdrant_client

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text using Gemini
        """
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=[text],
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise

    def qdrant_search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Search Qdrant for relevant chunks
        """
        try:
            # Assuming we have a collection named "book_content" as per the data model
            search_results = self.qdrant_client.search(
                collection_name="book_content",
                query_vector=query_embedding,
                limit=top_k
            )

            results = []
            for result in search_results:
                results.append({
                    "chunk_id": result.id,
                    "text": result.payload.get("text", ""),
                    "similarity_score": result.score,
                    "metadata": result.payload
                })

            return results
        except Exception as e:
            logging.error(f"Error searching Qdrant: {e}")
            raise

    def selected_text_search(self, query_embedding: List[float], selected_text_chunks: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        Search within selected text chunks
        This is a simplified implementation that would normally involve vector similarity
        computation between the query and the selected text chunks
        """
        try:
            # For now, this is a placeholder that returns the selected chunks with some score
            # In a real implementation, we would compute embeddings for the selected chunks
            # and calculate similarity with the query embedding
            results = []
            for i, chunk_text in enumerate(selected_text_chunks[:top_k]):
                results.append({
                    "chunk_id": f"selected_chunk_{i}",
                    "text": chunk_text,
                    "similarity_score": 0.8,  # Placeholder score
                    "metadata": {"source": "selected_text"}
                })

            return results
        except Exception as e:
            logging.error(f"Error searching selected text: {e}")
            raise