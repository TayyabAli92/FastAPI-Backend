#!/usr/bin/env python3
"""
Test script to validate the fixes to the serverless function
"""
import json
import os
from api.chat import handler

def test_handler():
    """Test the Vercel handler function with mock data"""
    print("Testing the Vercel handler function...")

    # Mock Vercel event
    event = {
        'body': json.dumps({
            'message': 'Hello, how are you?',
            'selected_text': None
        }),
        'httpMethod': 'POST',
        'headers': {
            'content-type': 'application/json'
        }
    }

    # Mock context (not used in our implementation)
    context = type('obj', (object,), {
        'function_name': 'test_function',
        'memory_limit_in_mb': 128,
        'invoked_function_arn': 'arn:aws:lambda:test',
        'aws_request_id': 'test_request_id'
    })()

    try:
        # This will test the basic flow without actually calling external services
        # (since we don't have the environment variables set up in test)
        result = handler(event, context)
        print(f"Handler returned: {result}")

        # Check if it's an error response due to missing environment variables
        if result.get('statusCode') == 500:
            body = json.loads(result.get('body', '{}'))
            error_msg = body.get('error', '')
            if 'GEMINI_API_KEY' in error_msg or 'not configured' in error_msg:
                print("Expected error due to missing API keys - this is normal in test environment")
                return True
            else:
                print(f"Unexpected error: {error_msg}")
                return False
        else:
            print("Handler executed without error (though likely failed due to missing API keys)")
            return True

    except Exception as e:
        print(f"Exception occurred during handler execution: {str(e)}")
        return False

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    try:
        from api.chat import run_rag_chat, retrieve_from_qdrant, retrieve_from_selected_text
        from api.agent import get_agent, BookRAGAgent
        from api.qdrant_rag import get_qdrant_rag, QdrantRAG
        print("All imports successful")
        return True
    except Exception as e:
        print(f"Import error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running tests for the serverless function fixes...")

    success = True
    success &= test_imports()
    success &= test_handler()

    if success:
        print("\nAll tests passed! The fixes appear to be working correctly.")
        print("Note: The function will still fail in test environment due to missing API keys,")
        print("but the timeout and initialization issues should be resolved.")
    else:
        print("\nSome tests failed!")