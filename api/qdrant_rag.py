import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import google.generativeai as genai
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantRAG:
    """Qdrant integration for RAG functionality"""

    def __init__(self):
        # Initialize Google Generative AI first
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                logger.info("Google Generative AI configured")
            except Exception as e:
                logger.error(f"Failed to configure Google Generative AI: {str(e)}")
                raise
        else:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Initialize Qdrant client
        try:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            if not qdrant_url:
                raise ValueError("QDRANT_URL environment variable not set")
            if not qdrant_api_key:
                raise ValueError("QDRANT_API_KEY environment variable not set")

            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise

        # Collection name
        self.collection_name = "books"

        # Create collection if it doesn't exist
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the collection exists with proper configuration"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with appropriate vector size
                # Google's embedding-001 model produces 768-dimensional vectors
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Convert text to embedding vector using Google's embedding service"""
        try:
            # Use Google's embedding API
            response = genai.embed_content(
                model="models/embedding-001",
                content=[text],
                task_type="semantic_similarity"
            )

            # The response should have an 'embedding' attribute
            # Try both attribute access and dictionary access
            if hasattr(response, 'embedding'):
                return response.embedding
            elif isinstance(response, dict) and 'embedding' in response:
                return response['embedding']
            else:
                logger.error(f"Unexpected API response format: {response}")
                raise ValueError(f"Unexpected API response format: {response}")
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a document to the Qdrant collection"""
        try:
            # Create embedding for the content
            vector = self.embed_text(content)

            # Prepare the point
            point = PointStruct(
                id=doc_id,
                vector=vector,
                payload={
                    "content": content,
                    "metadata": metadata or {}
                }
            )

            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Added document {doc_id} to collection")
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents based on query"""
        try:
            # Create embedding for the query
            query_vector = self.embed_text(query)

            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )

            # Extract content from results
            results = []
            for hit in search_results:
                if hit.payload and 'content' in hit.payload:
                    results.append({
                        'id': hit.id,
                        'content': hit.payload['content'],
                        'score': hit.score,
                        'metadata': hit.payload.get('metadata', {})
                    })

            logger.info(f"Found {len(results)} results for query")
            return results

        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            return []

    def get_relevant_content(self, query: str, top_k: int = 5) -> str:
        """Get relevant content as a formatted string for RAG"""
        try:
            results = self.search(query, top_k)

            if not results:
                return "No relevant content found in the book."

            # Format the results as a context string
            formatted_results = []
            for result in results:
                formatted_results.append(f"Relevant Content:\n{result['content']}\n")

            return "\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error getting relevant content: {str(e)}")
            return "Error retrieving content from the book."

# Global Qdrant instance (will be recreated on each cold start in serverless)
qdrant_rag = None

def get_qdrant_rag():
    """Get or create the Qdrant RAG instance"""
    global qdrant_rag
    if qdrant_rag is None:
        qdrant_rag = QdrantRAG()
    return qdrant_rag