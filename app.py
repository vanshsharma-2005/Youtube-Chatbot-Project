import os
import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---------------- LOAD ENV ----------------
import streamlit as st

# Works on both local (.env) and Streamlit Cloud (secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TubeChat AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #050A0E !important;
    color: #E8F4F8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── ANIMATED GRID BACKGROUND ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,62,0,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,62,0,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stHeader"] { display: none; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #080D11 !important;
    border-right: 1px solid rgba(255,62,0,0.15) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── MAIN CONTENT AREA ── */
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
}

/* ── TYPOGRAPHY ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em !important;
}

/* ── HERO HEADER ── */
.hero-header {
    position: relative;
    padding: 2.5rem 0 1.5rem;
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #FF3E00;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.0;
    color: #FFFFFF;
    margin: 0;
}
.hero-title span {
    color: #FF3E00;
    position: relative;
    display: inline-block;
}
.hero-title span::after {
    content: '';
    position: absolute;
    bottom: 2px;
    left: 0;
    width: 100%;
    height: 3px;
    background: #FF3E00;
    border-radius: 2px;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: rgba(232,244,248,0.5);
    margin-top: 0.75rem;
    font-weight: 300;
}

/* ── STAT PILLS ── */
.stat-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,62,0,0.08);
    border: 1px solid rgba(255,62,0,0.2);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: rgba(232,244,248,0.7);
    letter-spacing: 0.05em;
}
.stat-pill .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #FF3E00;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

/* ── INPUT CARD ── */
.input-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,62,0,0.15);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.input-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(to bottom, #FF3E00, transparent);
    border-radius: 3px 0 0 3px;
}
.input-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: #FF3E00;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── STREAMLIT TEXT INPUT ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,62,0,0.2) !important;
    border-radius: 10px !important;
    color: #E8F4F8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #FF3E00 !important;
    box-shadow: 0 0 0 2px rgba(255,62,0,0.12) !important;
}
[data-testid="stTextInput"] label {
    display: none !important;
}

/* ── PROCESS BUTTON ── */
[data-testid="stButton"] > button {
    background: #FF3E00 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stButton"] > button:hover {
    background: #E83600 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(255,62,0,0.3) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── SUCCESS / ERROR / INFO MESSAGES ── */
[data-testid="stSuccess"] {
    background: rgba(0,200,100,0.08) !important;
    border: 1px solid rgba(0,200,100,0.25) !important;
    border-radius: 10px !important;
    color: #00C864 !important;
}
[data-testid="stError"] {
    background: rgba(255,62,0,0.08) !important;
    border: 1px solid rgba(255,62,0,0.3) !important;
    border-radius: 10px !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,62,0,0.12) !important;
    margin: 2rem 0 !important;
}

/* ── CHAT SECTION HEADER ── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.chat-header-icon {
    width: 36px; height: 36px;
    background: rgba(255,62,0,0.12);
    border: 1px solid rgba(255,62,0,0.25);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.chat-header-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #E8F4F8;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    border-color: rgba(255,62,0,0.2) !important;
    background: rgba(255,62,0,0.04) !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,62,0,0.2) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #E8F4F8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] button {
    background: #FF3E00 !important;
    border-radius: 8px !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] {
    color: #FF3E00 !important;
}

/* ── SIDEBAR CONTENT ── */
.sidebar-logo {
    background: linear-gradient(135deg, #FF3E00 0%, #FF7043 100%);
    padding: 1.5rem;
    margin: -1rem -1rem 1.5rem -1rem;
    position: relative;
    overflow: hidden;
}
.sidebar-logo::before {
    content: '▶';
    position: absolute;
    right: -0.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
    color: white;
}
.sidebar-logo h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    color: white !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}
.sidebar-logo p {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: rgba(255,255,255,0.7);
    margin: 0.25rem 0 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.sidebar-section {
    margin-bottom: 1.5rem;
}
.sidebar-section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: rgba(232,244,248,0.35);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.tech-badge {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.75rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    color: rgba(232,244,248,0.7);
}
.tech-badge .tech-icon {
    font-size: 1rem;
    flex-shrink: 0;
}

.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}
.step-num {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: rgba(255,62,0,0.15);
    border: 1px solid rgba(255,62,0,0.4);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #FF3E00;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text {
    font-size: 0.82rem;
    color: rgba(232,244,248,0.6);
    line-height: 1.4;
}

/* ── FOOTER ── */
.footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,62,0,0.1);
    text-align: center;
}
.footer-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    color: rgba(232,244,248,0.25);
}
.footer-text strong {
    color: #FF3E00;
    font-weight: 500;
}
.footer-heart {
    color: #FF3E00;
    animation: heartbeat 1.5s infinite;
    display: inline-block;
}
@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.2); }
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>TubeChat AI</h2>
        <p>Powered by Gemini + LangChain</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">How it works</div>
        <div class="step-item">
            <div class="step-num">1</div>
            <div class="step-text">Paste a YouTube video URL above</div>
        </div>
        <div class="step-item">
            <div class="step-num">2</div>
            <div class="step-text">Click Process Video to extract the transcript</div>
        </div>
        <div class="step-item">
            <div class="step-num">3</div>
            <div class="step-text">Ask any question about the video content</div>
        </div>
        <div class="step-item">
            <div class="step-num">4</div>
            <div class="step-text">Get AI-powered answers grounded in the video</div>
        </div>
    </div>

    <div class="sidebar-section">
        <div class="sidebar-section-title">Tech Stack</div>
        <div class="tech-badge"><span class="tech-icon">✦</span> Google Gemini 2.5 Flash</div>
        <div class="tech-badge"><span class="tech-icon">⛓</span> LangChain RAG Pipeline</div>
        <div class="tech-badge"><span class="tech-icon">⚡</span> FAISS Vector Store</div>
        <div class="tech-badge"><span class="tech-icon">🤗</span> Sentence Transformers</div>
        <div class="tech-badge"><span class="tech-icon">▶</span> YouTube Transcript API</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero-header">
    <div class="hero-eyebrow">AI · RAG · YouTube</div>
    <h1 class="hero-title">Chat with any<br><span>YouTube video</span></h1>
    <p class="hero-subtitle">
        Paste a link. Ask anything. Get answers grounded in the video transcript.
    </p>
    <div class="stat-row">
        <div class="stat-pill"><span class="dot"></span> Semantic Search</div>
        <div class="stat-pill"><span class="dot"></span> Context-Aware Answers</div>
        <div class="stat-pill"><span class="dot"></span> Gemini 2.5 Flash</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
st.markdown("""
<div class="input-card">
    <div class="input-label">▶ YouTube Video URL</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    youtube_url = st.text_input(
        "url",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )
with col2:
    process_btn = st.button("⚡ Process")

# ---------------- EXTRACT VIDEO ID ----------------
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# ---------------- PROCESS VIDEO ----------------
if process_btn:
    if not youtube_url:
        st.error("Please enter a YouTube URL first.")
        st.stop()

    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("⚠️ Invalid YouTube URL. Please check and try again.")
        st.stop()

    try:
        with st.spinner("📥 Fetching transcript from YouTube..."):
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id, languages=["en"])
            transcript = " ".join(chunk.text for chunk in transcript_list)

        with st.spinner("✂️ Chunking transcript..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

        with st.spinner("🧠 Building vector store..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

        prompt = PromptTemplate(
            template="""You are a helpful AI assistant analyzing a YouTube video transcript.
Answer ONLY from the provided transcript context. Be concise and precise.
If the answer is not in the transcript, say: "I couldn't find this in the video transcript."

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        st.session_state.chain = chain
        st.session_state.chat_history = []
        st.session_state.video_id = video_id

        st.success(f"✅ Video processed! {len(chunks)} chunks indexed. Ready to chat.")

    except TranscriptsDisabled:
        st.error("⚠️ This video has no captions/transcripts available.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---------------- VIDEO PREVIEW ----------------
if "video_id" in st.session_state:
    with st.expander("🎬 Video Preview", expanded=False):
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

# ---------------- CHAT SECTION ----------------
if "chain" in st.session_state:
    st.divider()

    st.markdown("""
    <div class="chat-header">
        <div class="chat-header-icon">💬</div>
        <div class="chat-header-text">Ask anything about the video</div>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.chat_input("What would you like to know about this video?")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("🤖 Generating answer..."):
            answer = st.session_state.chain.invoke(user_input)
        st.session_state.chat_history.append(("bot", answer))

    for role, message in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user"):
                st.markdown(message)
        else:
            with st.chat_message("assistant"):
                st.markdown(message)

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <p class="footer-text">
        Made with <span class="footer-heart">♥</span> by <strong>Vansh Sharma</strong>
        &nbsp;·&nbsp; TubeChat AI &nbsp;·&nbsp; Gemini + LangChain + FAISS
    </p>
</div>
""", unsafe_allow_html=True)
