import os
import json
import requests

class LLMAPIHandler:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.URL = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.0-flash:generateContent"
            f"?key={self.api_key}"
        )

    def request(self, content: str) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{content}"
                        }
                    ]
                }
            ]
        }
        headers = {
            "Content-Type": "application/json"
        }

        # Make the POST request
        response = requests.post(self.URL, headers=headers, data=json.dumps(payload), timeout=30)

        # Raise error if something went wrong (HTTP ≠ 200)
        response.raise_for_status()

        # Parse JSON
        data = response.json()

        # Pretty-print response
        # print(json.dumps(data, indent=2))
        # print(data["candidates"][0]["content"]["parts"][0]["text"])
        return data["candidates"][0]["content"]["parts"][0]["text"]
            

if __name__ == "__main__":
    API_KEY = json.load(open("api_keys.json"))["google_api_key"]

    llm = LLMAPIHandler(API_KEY)

    print(llm.request("Explain how AI works in a few words"))
