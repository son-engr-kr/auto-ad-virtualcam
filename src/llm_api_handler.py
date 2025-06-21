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
        company_list = [
                    ("BlendED", "A company focused on blended learning solutions that combine online and offline education."),
                    ("IIElevenLabs", "A cutting-edge AI voice synthesis company known for realistic speech generation."),
                    ("BIZ CRUSH", "A platform that helps startups and businesses connect, pitch, and get funding."),
                    ("Glimpse", "A tool that provides instant user behavior insights through short video captures or analytics."),
                    ("clix", "An adtech company specializing in real-time, personalized digital advertisements."),
                    ("UClone", "A service for creating realistic virtual avatars or deepfake-like video content."),
                    ("typecast", "A synthetic voice generation service tailored for creators, often used in video production."),
                    ("Codetree", "An online coding platform or IDE designed for collaborative software development."),
                    ("Signaling", "A communication framework or API for real-time audio/video signaling between users."),
                    ("Z Order", "A company specializing in graphics rendering, 3D ordering, or UI layering solutions.")
                ]
        
        prompt1 = """

                You are an intelligent assistant helping to match companies to user conversations.
                Below is a list of companies. Each entry contains the company name and a short explanation of what the company does:

                """
        prompt2 = """

                Now read the following conversation script and determine which single company is most relevant to the
                context of the dialogue. Base your decision on the content, goals, and keywords in the conversation
                """
        prompt3 = """

                Return only the name of the most relevant company, and nothing else. Do not explain your reasoning.
                """
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{prompt1}\n{company_list}\n{prompt2}\n{content}\n{prompt3}"
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
