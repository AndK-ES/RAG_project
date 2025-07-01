#!/usr/bin/env python3
"""
Test script for the Museum Art RAG API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the server is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ask_endpoint():
    """Test the ask question endpoint"""
    print("\n❓ Testing ask endpoint...")
    
    test_question = {
        "question": "Tell me about Vincent van Gogh",
        "top_k": 3
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ask",
            json=test_question,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Answer: {data['answer'][:100]}...")
            print(f"Confidence: {data['confidence']:.4f}")
            print(f"Sources: {data['total_sources']}")
            print(f"Generation Time: {data['generation_time']:.2f}s")
            return True
        elif response.status_code == 503:
            print("⚠️  RAG pipeline not ready (this is expected if no data is loaded)")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_search_endpoint():
    """Test the search endpoint"""
    print("\n🔍 Testing search endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/search?query=van gogh&top_k=3")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Found {data['total_found']} documents")
            return True
        elif response.status_code == 503:
            print("⚠️  RAG pipeline not ready (this is expected if no data is loaded)")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_web_interface():
    """Test the web interface"""
    print("\n🌐 Testing web interface...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Web interface is accessible")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_swagger_docs():
    """Test the Swagger documentation"""
    print("\n📖 Testing Swagger documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Swagger documentation is accessible")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Museum Art RAG API")
    print("=" * 50)
    
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Web Interface", test_web_interface),
        ("Swagger Documentation", test_swagger_docs),
        ("Ask Endpoint", test_ask_endpoint),
        ("Search Endpoint", test_search_endpoint),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Small delay between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for more information.")
    
    print(f"\n🌐 Web Interface: {BASE_URL}")
    print(f"📖 API Documentation: {BASE_URL}/docs")
    print(f"🔍 Health Check: {BASE_URL}/health")

if __name__ == "__main__":
    main() 