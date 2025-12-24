#!/usr/bin/env python3
"""
Script to run the Book RAG Agent application
"""

import os
import subprocess
import sys
import time
import threading
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import google.generativeai
        import qdrant_client
        import pydantic
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_server():
    """Start the FastAPI server"""
    if not check_dependencies():
        sys.exit(1)

    print("Starting Book RAG Agent server...")
    print("Loading configuration...")

    # Start the Uvicorn server
    try:
        import uvicorn
        from app import app

        print("Server starting on http://localhost:8000")
        print("Press Ctrl+C to stop the server")

        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

def open_frontend():
    """Open the frontend in a web browser after a delay"""
    time.sleep(3)  # Wait for server to start
    frontend_path = Path("../frontend/book-rag-interface.html")
    if frontend_path.exists():
        webbrowser.open(f"file://{frontend_path.absolute()}")
        print("Frontend opened in your default browser")
    else:
        print(f"Frontend file not found: {frontend_path}")
        print("You can manually open frontend/book-rag-interface.html in your browser")

def main():
    print("Book RAG Agent Startup Script")
    print("=" * 40)

    # Check if backend is running
    if not check_dependencies():
        print("❌ Dependencies not met. Please install requirements.txt")
        return

    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Open frontend in browser
    frontend_thread = threading.Thread(target=open_frontend, daemon=True)
    frontend_thread.start()

    try:
        # Keep the main thread alive
        while server_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Book RAG Agent...")
        sys.exit(0)

if __name__ == "__main__":
    main()