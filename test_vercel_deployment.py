"""
Test script to validate the Vercel deployment configuration
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all necessary modules can be imported"""
    print("Testing imports...")

    try:
        import fastapi
        print("[OK] FastAPI imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import FastAPI: {e}")
        return False

    try:
        from agent import initialize_agent, rag_query_tool
        print("[OK] Agent module imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import agent: {e}")
        return False

    try:
        from connection import ConnectionManager
        print("[OK] Connection module imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import connection: {e}")
        return False

    try:
        from api.app import app
        print("[OK] API app imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import API app: {e}")
        return False

    return True

def test_api_structure():
    """Test the API structure"""
    print("\nTesting API structure...")

    try:
        from api.app import ChatRequest, ChatResponse, Citation
        print("[OK] Pydantic models defined correctly")
    except Exception as e:
        print(f"[ERROR] Failed to define Pydantic models: {e}")
        return False

    try:
        from api.app import app
        # Check if the required routes exist
        routes = [route.path for route in app.routes]
        required_routes = ["/", "/health", "/chat"]

        for route in required_routes:
            if route in routes or f"{route}:0" in routes:
                print(f"[OK] Route {route} exists")
            else:
                print(f"[ERROR] Route {route} missing")
                return False

        return True
    except Exception as e:
        print(f"[ERROR] Failed to check API routes: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\nTesting environment configuration...")

    # Check if required environment variables are available
    required_vars = ["GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]

    for var in required_vars:
        if os.getenv(var):
            print(f"[OK] Environment variable {var} is set")
        else:
            print(f"[WARN] Environment variable {var} is not set (will be required for production)")

    return True

def main():
    """Main validation function"""
    print("Validating Vercel deployment configuration...\n")

    success = True

    # Test imports
    success &= test_imports()

    # Test API structure
    success &= test_api_structure()

    # Test environment
    success &= test_environment()

    print(f"\n{'='*50}")
    if success:
        print("[SUCCESS] All validations passed! The Vercel deployment should work correctly.")
        print("\nTo deploy to Vercel:")
        print("1. Make sure your environment variables are set in Vercel dashboard")
        print("2. Run: vercel --prod")
        print("3. Your API endpoints will be available at:")
        print("   - GET / (root)")
        print("   - GET /health")
        print("   - POST /chat")
    else:
        print("[FAILURE] Some validations failed. Please fix the issues before deploying.")
        print("Check the error messages above for details.")

    print(f"{'='*50}")
    return success

if __name__ == "__main__":
    main()