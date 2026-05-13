import os
import re
import streamlit as st

from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import (
    StrOutputParser
)

# ---------------- LOAD ENV ----------------

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv(
    "GOOGLE_API_KEY"
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="YouTube AI Chatbot",
    page_icon="🎥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #FF4B4B;
    color: white;
    font-size: 16px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("🎥 YouTube AI Chatbot")

    st.markdown("""
    ### Features
    - Chat with YouTube videos
    - AI-powered answers
    - Transcript search
    - Gemini + LangChain + FAISS
    """)

# ---------------- TITLE ----------------

st.title("🎥 Chat with YouTube Videos")

st.markdown(
    "Paste a YouTube video link and ask questions."
)

# ---------------- INPUT ----------------

youtube_url = st.text_input(
    "Enter YouTube Video URL"
)

# ---------------- EXTRACT VIDEO ID ----------------

def extract_video_id(url):

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"

    match = re.search(pattern, url)

    if match:
        return match.group(1)

    return None

# ---------------- PROCESS VIDEO ----------------

if st.button("🚀 Process Video"):

    video_id = extract_video_id(youtube_url)

    if not video_id:

        st.error("Invalid YouTube URL")
        st.stop()

    try:

        with st.spinner("📥 Fetching transcript..."):

            ytt_api = YouTubeTranscriptApi()

            transcript_list = ytt_api.fetch(
                video_id,
                languages=["en"]
            )

            transcript = " ".join(
                chunk.text
                for chunk in transcript_list
            )

        with st.spinner("✂️ Splitting transcript..."):

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.create_documents(
                [transcript]
            )

        with st.spinner("🧠 Creating embeddings..."):

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vector_store = FAISS.from_documents(
                chunks,
                embeddings
            )

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        prompt = PromptTemplate(
            template="""
            You are a helpful AI assistant.

            Answer ONLY from the transcript context.

            If the answer is not present,
            say:
            "I could not find this information in the video."

            Context:
            {context}

            Question:
            {question}
            """,

            input_variables=[
                "context",
                "question"
            ]
        )

        def format_docs(retrieved_docs):

            return "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

        parallel_chain = RunnableParallel({

            "context":
                retriever
                | RunnableLambda(format_docs),

            "question":
                RunnablePassthrough()
        })

        parser = StrOutputParser()

        main_chain = (
            parallel_chain
            | prompt
            | llm
            | parser
        )

        st.session_state.chain = main_chain
        st.session_state.chat_history = []

        st.success(
            "✅ Video processed successfully!"
        )

    except TranscriptsDisabled:

        st.error(
            "No captions available for this video."
        )

    except Exception as e:

        st.error(f"Error: {e}")

# ---------------- CHAT SECTION ----------------

if "chain" in st.session_state:

    st.divider()

    st.subheader("💬 Chat with Video")

    user_input = st.chat_input(
        "Ask anything about the video..."
    )

    if user_input:

        st.session_state.chat_history.append(
            ("user", user_input)
        )

        with st.spinner("🤖 Thinking..."):

            answer = st.session_state.chain.invoke(
                user_input
            )

        st.session_state.chat_history.append(
            ("bot", answer)
        )

    for role, message in (
        st.session_state.chat_history
    ):

        if role == "user":

            with st.chat_message("user"):
                st.markdown(message)

        else:

            with st.chat_message("assistant"):
                st.markdown(message)