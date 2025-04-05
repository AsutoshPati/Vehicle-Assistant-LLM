# 🚗🧠 AI-Powered Car Assistant 

An intelligent, voice-activated assistant built using **Generative AI** and 
**Retrieval-Augmented Generation (RAG)**, designed to enhance the in-vehicle 
experience for Tata Motor vehicle owners.

This project allows drivers to troubleshoot issues in real-time, receive 
voice-guided instructions on car features, and interact naturally through the 
car’s speaker and microphone systems.

> 🕊️ Built as a heartfelt tribute to the late Mr. Ratan Tata, a true visionary 
> and a national treasure whose legacy continues to inspire generations.

## 🚀 Features
> Instead of waiting for customer support like your crush’s reply 💔📞... get 
> instant answers from an AI 🤖 who actually knows your car inside out 🔧.

- **🎙️ Voice Input/Output**: Interact using natural language through the car's 
built-in mic and speaker.
- **🔧 Real-Time Troubleshooting**: Diagnose vehicle issues instantly with 
contextual, AI-powered answers.
- **📘 Feature Guidance**: Get detailed voice explanations of car 
functionalities and settings.
- **🔄 RAG-Powered Retrieval**: Combines LLM capabilities with a curated 
knowledge base for accurate, relevant answers.
- **🛠️ Extendable Architecture**: Built for future updates—like full voice 
control over car electronics.

## 🧠 Tech Stack
- Generative AI: Ollama (Alternatives - OpenAI, Gemini or others)
- RAG (Retrieval-Augmented Generation) for dynamic information access
- Speech Recognition:  Whisper (Alternatives - Vosk, Deepgram or others)
- Text-to-Speech: Coqui TTS (Alternatives - poly, gTTS, pyttsx3 or others)
- Python
- Microcontroller Integration (optional – for future voice-controlled actions)

## 📦 Installation
So, you want your car to talk back? Not in anger, but with helpful, AI-powered 
advice? Here's how you set it up:

### Clone the repo
```commandline
git clone https://github.com/yourusername/tata-ai-assistant.git
cd tata-ai-assistant
```

### Install Required Python Modules
I recommend to use Python 3.10 (tested and working beautifully). Then, 
install all dependencies:
```commandline
pip install -r requirements.txt
```

### Load Up the Knowledge Base
I built this for TATA Altroz—because well, that’s the car I own. But you can 
train the assistant for any vehicle or gadget!
#### 🛠️ Here’s how:
- Download the official user manual (PDF) of your vehicle/device from the manufacturer's website.
- Place the PDF(s) into the /knowledge_base/ directory.
- Then, run this command to convert your manual into a searchable brain:
```commandline
python generate_knowledgebase.py
```
> 🤖 No PhD needed. Just your user manual and one command to rule them all.

### Set Up the LLM with Ollama
I am using LLaMA 3.2:3B via Ollama
#### Steps:
- Download & install Ollama from: https://ollama.com/download
- Then, download the model from https://ollama.com/library/llama3.2 or using:
```commandline
ollama run llama3.2:3b
```

### Run the Assistant
```commandline
python main.py
```

## ⭐ A Note of Gratitude
This project is a humble tribute to Mr. Ratan Tata—a visionary whose integrity, 
compassion, and dedication have profoundly shaped India’s industrial and 
humanitarian landscape.

His legacy continues to inspire creators, dreamers, and doers like us. 🙏
If this project resonates with you, please consider giving it a ⭐ and sharing 
your thoughts or ideas to help it grow further. Let’s build something 
meaningful—together. 💛



With a mix of pride and a heavy heart, I present my latest project — a Generative AI & Retrieval-Augmented Generation (RAG) powered assistant for Tata Motors (not an official one). This project is more than just technology; it's a tribute to the late Mr. Ratan Tata, a visionary whose impact on India's automotive and industrial landscape is immeasurable.


