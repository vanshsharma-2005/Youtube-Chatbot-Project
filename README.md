<div align="center">

# 🎬 TubeChat AI

### Chat with any YouTube video using AI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-00A86B?style=for-the-badge)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **An AI-powered RAG chatbot that lets you have intelligent conversations with any YouTube video — powered by Google Gemini, LangChain, FAISS, and Sentence Transformers.**

[🚀 Live Demo](#) · [📖 Documentation](#how-it-works) · [🐛 Report Bug](https://github.com/vanshsharma-2005/Youtube-Chatbot-Project/issues) · [✨ Request Feature](https://github.com/vanshsharma-2005/Youtube-Chatbot-Project/issues)

</div>

---

## 📌 Overview

**TubeChat AI** is an end-to-end Retrieval-Augmented Generation (RAG) application that transforms any YouTube video into an interactive knowledge base. Simply paste a video URL, and start asking questions — the AI answers strictly from the video's transcript, so you always get grounded, accurate responses.

Whether you want to quickly extract key insights from a lecture, understand a tutorial without watching the whole thing, or quiz yourself on a documentary — TubeChat AI has you covered.

---

## ✨ Features

- 🎯 **Semantic Search** — Uses vector embeddings to find the most relevant parts of the transcript for each question
- 🤖 **Gemini 2.5 Flash** — Google's latest and fastest LLM for generating high-quality, context-aware answers
- ⚡ **FAISS Vector Store** — Lightning-fast similarity search across chunked transcript data
- 🧠 **RAG Pipeline** — Answers are grounded in the video transcript, not hallucinated
- 🎬 **Inline Video Preview** — Watch the video directly inside the app
- 💬 **Persistent Chat History** — Full conversation memory within a session
- 🗑️ **Clear Chat** — Reset conversation anytime
- 🎨 **Custom Dark UI** — Sleek, modern interface with animated elements
- 📱 **Wide Layout** — Sidebar with tech stack info and step-by-step guide

---

## 🏗️ Architecture

```
YouTube URL
     │
     ▼
┌─────────────────────┐
│  YouTube Transcript │  ◄── youtube-transcript-api
│       API           │
└────────┬────────────┘
         │  Raw Transcript Text
         ▼
┌─────────────────────┐
│  Text Splitter      │  ◄── RecursiveCharacterTextSplitter
│  (chunk_size=1000)  │       chunk_overlap=200
└────────┬────────────┘
         │  Document Chunks
         ▼
┌─────────────────────┐
│  Sentence           │  ◄── all-MiniLM-L6-v2
│  Transformers       │       (HuggingFace Embeddings)
└────────┬────────────┘
         │  Vector Embeddings
         ▼
┌─────────────────────┐
│   FAISS             │  ◄── In-memory vector store
│   Vector Store      │       k=4 similarity retrieval
└────────┬────────────┘
         │  Relevant Chunks (on query)
         ▼
┌─────────────────────┐
│  LangChain RAG      │  ◄── RunnableParallel + PromptTemplate
│  Chain              │
└────────┬────────────┘
         │  Prompt with Context + Question
         ▼
┌─────────────────────┐
│  Google Gemini      │  ◄── gemini-2.5-flash
│  2.5 Flash LLM      │       temperature=0.3
└────────┬────────────┘
         │  Answer
         ▼
    Streamlit UI
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive web UI |
| **LLM** | Google Gemini 2.5 Flash | Answer generation |
| **Orchestration** | LangChain | RAG pipeline management |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | Text vectorization |
| **Vector Store** | FAISS | Semantic similarity search |
| **Transcript** | YouTube Transcript API | Fetching video captions |
| **Environment** | python-dotenv | API key management |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Google AI Studio](https://aistudio.google.com) API key (free)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/vanshsharma-2005/Youtube-Chatbot-Project.git
cd Youtube-Chatbot-Project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

> 💡 Get your free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)

### 5. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## ☁️ Deploy on Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New App** → select your forked repo
4. Set **Main file path** to `app.py`
5. Under **Advanced Settings → Secrets**, add:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

6. Click **Deploy** — your app will be live in ~2 minutes!

---

## 📖 How It Works

```
1. 📥  Paste YouTube URL  →  App extracts the video ID
2. 📝  Fetch Transcript   →  YouTube Transcript API pulls English captions
3. ✂️  Chunk Text         →  Transcript split into 1000-char overlapping chunks
4. 🧠  Embed Chunks       →  Sentence Transformers convert chunks to vectors
5. ⚡  Index Vectors      →  FAISS stores all vectors for fast retrieval
6. 💬  User Asks          →  Question is embedded and matched to top-4 chunks
7. 🤖  LLM Answers        →  Gemini generates answer from retrieved context only
```

---

## 📁 Project Structure

```
Youtube-Chatbot-Project/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # API keys (do NOT commit this)
├── .gitignore                      # Git ignore rules
├── YT_Chatbot_Project.ipynb.ipynb  # Jupyter notebook (experimentation)
└── README.md                       # You are here
```

---

## ⚙️ Configuration

You can tweak these parameters in `app.py` to change behaviour:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `1000` | Characters per transcript chunk |
| `chunk_overlap` | `200` | Overlap between chunks |
| `search_kwargs k` | `4` | Number of chunks retrieved per query |
| `temperature` | `0.3` | LLM creativity (0 = factual, 1 = creative) |
| `model` | `gemini-2.5-flash` | Gemini model variant |

---

## ⚠️ Limitations

- Only works with YouTube videos that have **English captions/subtitles** enabled
- FAISS index is **in-memory** — resets when you process a new video
- Very long videos (3+ hours) may take longer to process
- Auto-generated captions may contain transcription errors

---

## 🔮 Future Improvements

- [ ] Support for multiple languages
- [ ] Persistent vector store with ChromaDB or Pinecone
- [ ] Multi-video knowledge base (chat across several videos)
- [ ] Timestamp citations in answers (jump to the exact moment)
- [ ] Export chat history as PDF or JSON
- [ ] YouTube playlist support

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

**Vansh Sharma**

- GitHub: [@vanshsharma-2005](https://github.com/vanshsharma-2005)

---

<div align="center">

Made with ❤️ by **Vansh Sharma**

*If you found this project useful, please consider giving it a ⭐ on GitHub!*

</div>
