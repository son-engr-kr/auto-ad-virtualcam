# 2025 Dream AI Hackathon: EchoBoard

# Scope

This project aims to develop an AI-powered virtual camera system for Zoom meetings that dynamically displays context-relevant advertisements and branded backgrounds using real-time face detection and speech recognition. By integrating speech-to-text recognition with a Large Language Model (LLM), the system detects the topic of conversation (e.g., web hosting) and displays tailored ads/backgrounds (e.g., Amazon AWS) or company logos behind participants via pyvirtualcam, all while making sure a seamless user experience through Image processing-based face detection and background overlay.

# Environment setting

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Get Google AI API Key
- https://aistudio.google.com/prompts/new_chat
put it in the `api_keys.json`
```
{
    "google_api_key": "YOUR API KEY HERE"
}
```

## Install OBS Studio
- Start virtual camera once and close it. It will set up the virtual camera system on your PC.


## System Requirements
- Windows 10/11
- macOS 13 or lower (macOS 14+ not supported)
- Python 3.11
- Webcam
- Microphone
