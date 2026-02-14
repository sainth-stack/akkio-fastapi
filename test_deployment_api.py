"""
Test script for deployment API endpoints
Run this on your FastAPI server to verify endpoints are working
"""
import requests
import json

# Update this to your FastAPI URL
BASE_URL = "http://18.143.150.140:3001"

def test_health():
    """Test if FastAPI is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"✅ FastAPI is running: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ FastAPI not accessible: {e}")
        return False

def test_deploy_endpoint():
    """Test deployment endpoint"""
    url = f"{BASE_URL}/api/deployment/deploy"
    payload = {
        "app_id": None,
        "project_name": "generated_app_1770640849576",
        "user_email": "admin@gmail.com"
    }
    
    print(f"\n🔍 Testing: POST {url}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Deploy endpoint working!")
            return True
        else:
            print(f"❌ Deploy endpoint returned error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error calling deploy endpoint: {e}")
        return False

def test_status_endpoint():
    """Test status endpoint"""
    url = f"{BASE_URL}/api/deployment/status"
    params = {
        "project_name": "generated_app_1770640849576",
        "user_email": "admin@gmail.com"
    }
    
    print(f"\n🔍 Testing: GET {url}")
    print(f"📦 Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Status endpoint working!")
            return True
        else:
            print(f"❌ Status endpoint returned error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error calling status endpoint: {e}")
        return False

def list_all_routes():
    """List all available routes from FastAPI"""
    try:
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code == 200:
            openapi = response.json()
            print("\n📋 Available Routes:")
            for path, methods in openapi.get('paths', {}).items():
                if 'deployment' in path.lower():
                    print(f"  {path}")
                    for method in methods.keys():
                        print(f"    - {method.upper()}")
        else:
            print("❌ Could not fetch OpenAPI spec")
    except Exception as e:
        print(f"❌ Error fetching routes: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Deployment API Test Suite")
    print("=" * 60)
    
    # Test 1: FastAPI health
    if not test_health():
        print("\n❌ FastAPI is not running. Please start it first.")
        exit(1)
    
    # Test 2: List routes
    list_all_routes()
    
    # Test 3: Deploy endpoint
    test_deploy_endpoint()
    
    # Test 4: Status endpoint
    test_status_endpoint()
    
    print("\n" + "=" * 60)
    print("🏁 Test Complete")
    print("=" * 60)
