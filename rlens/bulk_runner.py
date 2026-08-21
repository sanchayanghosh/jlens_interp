import requests
import json

# Example of your dataset format
dataset = [
    {
        "id": "conv_001",
        "user_turns": [
            "Hello, what is the capital of France?",
            "And what is their most famous tower called?"
        ]
    },
    {
        "id": "conv_002",
        "user_turns": [
            "Write a two line poem about the ocean.",
            "Make it rhyme with 'blue'."
        ]
    }
]

def run_dataset():
    for data in dataset:
        print(f"Processing conversation: {data['id']}")
        
        # Send the list of user turns to your local API
        response = requests.post(
            "http://127.0.0.1:8080/api/run_multiturn",
            json={"user_turns": data["user_turns"], "max_new_tokens": 50}
        )
        
        if response.status_code == 200:
            print(f"Success! Saved to local disk: {response.json()['run_id']}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    run_dataset()
