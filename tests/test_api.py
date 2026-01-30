import requests
import json
import time
import subprocess
import sys
import os

def test_api():
    print("--- Testing Sakha Search Optimization API ---")
    
    # Start the server in the background
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "api_server.py"],
        cwd=os.path.join(os.getcwd(), "Projects/Sakha"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))} # simplified logic
    )
    
    # Wait for server to boot
    time.sleep(10)
    
    try:
        url = "http://localhost:8000/optimize"
        
        test_queries = [
            "how to reach banke bihari",
            "order sattvic lunch",
            "buy krishna painting"
        ]
        
        for q in test_queries:
            print(f"\nQuery: {q}")
            response = requests.post(url, json={"query": q})
            if response.status_code == 200:
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"FAILED: {response.status_code} - {response.text}")
                
    finally:
        print("\nShutting down server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_api()
