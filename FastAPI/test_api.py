#!/usr/bin/env python3
"""
Test script for the FastAPI Dockerized application
Run this to verify that the API is working correctly
"""

import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_health_endpoint():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print("🤖 Testing prediction endpoint...")
    try:
        test_data = {
            "text": "Hello Safaricom, I'm having issues with my MPESA transaction",
            "user_id": "test_user_123"
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Text: {data['text']}")
            print(f"   Prediction: {data['prediction']}")
            print(f"   Confidence: {data['confidence']:.4f}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_transformer_endpoint():
    """Test the transformer prediction endpoint"""
    print("🤖 Testing transformer prediction endpoint...")
    try:
        test_data = {
            "text": "Safaricom network is down and I'm angry about it!",
            "user_id": "test_user_456"
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/transformer",
            json=test_data,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Transformer prediction successful:")
            print(f"   Text: {data['text']}")
            print(f"   Prediction: {data['prediction']}")
            print(f"   Confidence: {data['confidence']:.4f}")
            return True
        else:
            print(f"❌ Transformer prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Transformer prediction error: {e}")
        return False

def test_batch_endpoint():
    """Test the batch prediction endpoint"""
    print("📦 Testing batch prediction endpoint...")
    try:
        test_data = [
            {"text": "Safaricom customer care is terrible"},
            {"text": "MPESA is not working properly"},
            {"text": "Great service from Safaricom today"}
        ]
        
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=test_data,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            print(f"✅ Batch prediction successful for {len(predictions)} texts")
            for i, pred in enumerate(predictions):
                print(f"   {i+1}: {pred['prediction']} (confidence: {pred['confidence']:.4f})")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def test_api_docs():
    """Test that API documentation is accessible"""
    print("📚 Testing API documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=TIMEOUT)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API docs failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API docs error: {e}")
        return False

def wait_for_api(max_attempts=10):
    """Wait for the API to be ready"""
    print("⏳ Waiting for API to be ready...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(3)
    
    print("❌ API is not responding")
    return False

def main():
    """Run all tests"""
    print("🚀 FastAPI Docker Test Suite")
    print("=" * 40)
    
    # Wait for API to be ready
    if not wait_for_api():
        print("❌ API is not accessible. Is the Docker container running?")
        print("   Try: ./deploy.sh run")
        sys.exit(1)
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_api_docs,
        test_prediction_endpoint,
        test_transformer_endpoint,
        test_batch_endpoint,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Empty line for readability
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    # Summary
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
