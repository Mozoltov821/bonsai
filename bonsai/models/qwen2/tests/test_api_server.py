#!/usr/bin/env python3
"""
Test script for Qwen2 JAX API server.

Usage:
    1. Start the server in another terminal:
       python -m bonsai.models.qwen2.tests.serve_api

    2. Run this test script:
       python bonsai/models/qwen2/tests/test_api_server.py
"""

import requests
import time
import sys


def test_health_check(base_url: str = "http://localhost:8000"):
    """Test health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_list_models(base_url: str = "http://localhost:8000"):
    """Test list models endpoint."""
    print("\nTesting list models...")
    try:
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()
        data = response.json()
        print(f"✓ List models passed: {data['data'][0]['id']}")
        return True
    except Exception as e:
        print(f"✗ List models failed: {e}")
        return False


def test_chat_completion(base_url: str = "http://localhost:8000"):
    """Test chat completion endpoint."""
    print("\nTesting chat completion...")
    try:
        payload = {
            "model": "qwen2-0.5b",
            "messages": [
                {"role": "user", "content": "你好，请用一句话介绍一下你自己。"}
            ],
            "max_tokens": 50,
            "temperature": 0.7,
        }

        start_time = time.time()
        response = requests.post(f"{base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        elapsed_time = time.time() - start_time

        data = response.json()
        message = data["choices"][0]["message"]["content"]
        usage = data["usage"]

        print(f"✓ Chat completion passed ({elapsed_time:.2f}s)")
        print(f"  Response: {message}")
        print(f"  Usage: {usage}")
        return True
    except Exception as e:
        print(f"✗ Chat completion failed: {e}")
        if hasattr(e, 'response'):
            print(f"  Response: {e.response.text}")
        return False


def test_text_completion(base_url: str = "http://localhost:8000"):
    """Test text completion endpoint."""
    print("\nTesting text completion...")
    try:
        payload = {
            "model": "qwen2-0.5b",
            "prompt": "人工智能是",
            "max_tokens": 30,
            "temperature": 0.7,
        }

        start_time = time.time()
        response = requests.post(f"{base_url}/v1/completions", json=payload)
        response.raise_for_status()
        elapsed_time = time.time() - start_time

        data = response.json()
        text = data["choices"][0]["text"]
        usage = data["usage"]

        print(f"✓ Text completion passed ({elapsed_time:.2f}s)")
        print(f"  Response: {text}")
        print(f"  Usage: {usage}")
        return True
    except Exception as e:
        print(f"✗ Text completion failed: {e}")
        if hasattr(e, 'response'):
            print(f"  Response: {e.response.text}")
        return False


def test_batch_completion(base_url: str = "http://localhost:8000"):
    """Test batch text completion."""
    print("\nTesting batch completion...")
    try:
        payload = {
            "model": "qwen2-0.5b",
            "prompt": ["北京是", "上海是"],
            "max_tokens": 20,
            "temperature": 0.7,
        }

        start_time = time.time()
        response = requests.post(f"{base_url}/v1/completions", json=payload)
        response.raise_for_status()
        elapsed_time = time.time() - start_time

        data = response.json()
        print(f"✓ Batch completion passed ({elapsed_time:.2f}s)")
        for i, choice in enumerate(data["choices"]):
            print(f"  Response {i}: {choice['text']}")
        return True
    except Exception as e:
        print(f"✗ Batch completion failed: {e}")
        if hasattr(e, 'response'):
            print(f"  Response: {e.response.text}")
        return False


def wait_for_server(base_url: str = "http://localhost:8000", timeout: int = 30):
    """Wait for server to be ready."""
    print(f"Waiting for server at {base_url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                print("✓ Server is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    print("✗ Server did not start within timeout")
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Qwen2 JAX API server")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000",
                        help="Base URL of the API server")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for server to be ready before testing")
    args = parser.parse_args()

    if args.wait:
        if not wait_for_server(args.base_url):
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Testing Qwen2 JAX API Server at {args.base_url}")
    print(f"{'='*60}\n")

    tests = [
        test_health_check,
        test_list_models,
        test_chat_completion,
        test_text_completion,
        test_batch_completion,
    ]

    results = []
    for test in tests:
        try:
            result = test(args.base_url)
            results.append(result)
        except Exception as e:
            print(f"Unexpected error in {test.__name__}: {e}")
            results.append(False)

    print(f"\n{'='*60}")
    print(f"Test Summary: {sum(results)}/{len(results)} passed")
    print(f"{'='*60}\n")

    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
