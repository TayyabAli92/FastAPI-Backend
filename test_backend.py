"""
Test script to verify the Book RAG Agent backend is working properly
"""
import requests
import json
import time

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✓ Health endpoint: OK")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health endpoint: Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health endpoint: Error - {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint with a simple query"""
    try:
        # Test data
        test_data = {
            "message": "Hello, can you help me?",
            "selected_text": None,
            "session_id": None
        }

        response = requests.post(
            "http://localhost:8000/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_data)
        )

        if response.status_code == 200:
            print("✓ Chat endpoint: OK")
            response_data = response.json()
            print(f"  Response: {response_data}")
            return True
        else:
            print(f"✗ Chat endpoint: Failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Chat endpoint: Error - {e}")
        return False

def test_selected_text_mode():
    """Test the selected text mode"""
    try:
        # Test data with selected text
        test_data = {
            "message": "Summarize this text",
            "selected_text": "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others.",
            "session_id": None
        }

        response = requests.post(
            "http://localhost:8000/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_data)
        )

        if response.status_code == 200:
            print("✓ Selected text mode: OK")
            response_data = response.json()
            print(f"  Response: {response_data}")
            return True
        else:
            print(f"✗ Selected text mode: Failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Selected text mode: Error - {e}")
        return False

def main():
    print("Testing Book RAG Agent Backend...")
    print("=" * 50)

    # Wait a moment for the server to be ready if just started
    time.sleep(2)

    all_tests_passed = True

    print("\n1. Testing health endpoint...")
    if not test_health_endpoint():
        all_tests_passed = False

    print("\n2. Testing chat endpoint...")
    if not test_chat_endpoint():
        all_tests_passed = False

    print("\n3. Testing selected text mode...")
    if not test_selected_text_mode():
        all_tests_passed = False

    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All tests passed! Backend is working correctly.")
    else:
        print("✗ Some tests failed. Please check the backend configuration.")

    return all_tests_passed

if __name__ == "__main__":
    main()