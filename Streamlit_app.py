import os
import streamlit as st

from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="🎥",
    layout="wide"
)


# ============================================================
# 2. LOAD API KEYS
# ============================================================

load_dotenv()

# First try local .env
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# If running on Streamlit Cloud, use Streamlit Secrets
if not YOUTUBE_API_KEY:
    try:
        YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY")
    except Exception:
        YOUTUBE_API_KEY = None


if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        GEMINI_API_KEY = None


# ============================================================
# 3. CHECK API KEYS
# ============================================================

if not YOUTUBE_API_KEY:
    st.error("❌ YouTube API key is missing.")
    st.info(
        "If you are using Streamlit Cloud, add "
        "YOUTUBE_API_KEY in App Settings → Secrets."
    )
    st.stop()


if not GEMINI_API_KEY:
    st.error("❌ Gemini API key is missing.")
    st.info(
        "If you are using Streamlit Cloud, add "
        "GEMINI_API_KEY in App Settings → Secrets."
    )
    st.stop()


# ============================================================
# 4. TITLE / HEADER
# ============================================================

st.title("🎥 YouTube AI Assistant")

st.write(
    "Enter a YouTube video URL and ask questions "
    "about its transcript using LangChain + RAG."
)


# ============================================================
# 5. CONNECT TO YOUTUBE API
# ============================================================

try:

    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY
    )

except Exception as e:

    st.error("❌ Could not connect to YouTube API.")
    st.write(str(e))
    st.stop()


# ============================================================
# 6. GET VIDEO ID
# ============================================================

def get_video_id(url):

    url = url.strip()

    if "v=" in url:

        return url.split("v=")[1].split("&")[0]

    elif "youtu.be/" in url:

        return url.split("youtu.be/")[1].split("?")[0]

    elif "youtube.com/shorts/" in url:

        return url.split("youtube.com/shorts/")[1].split("?")[0]

    else:

        return None


# ============================================================
# 7. GET VIDEO INFORMATION
# ============================================================

def get_video_info(video_id):

    try:

        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )

        response = request.execute()

        if not response.get("items"):

            return None

        video = response["items"][0]

        return {
            "title": video["snippet"]["title"],
            "channel": video["snippet"]["channelTitle"],
            "description": video["snippet"]["description"],
            "views": video["statistics"].get(
                "viewCount",
                "N/A"
            )
        }

    except Exception as e:

        st.error("❌ Could not fetch video information.")
        st.write(str(e))

        return None


# ============================================================
# 8. GET VIDEO TRANSCRIPT
# ============================================================

def get_transcript(video_id):

    api = YouTubeTranscriptApi()

    try:

        # Try Hindi and English
        transcript = api.fetch(
            video_id,
            languages=["hi", "en"]
        )

        text = " ".join(
            snippet.text
            for snippet in transcript
        )

        return text

    except Exception as e:

        st.warning(
            "⚠️ Could not fetch the transcript automatically."
        )

        st.write(str(e))

        return None


# ============================================================
# 9. CREATE EMBEDDINGS
# ============================================================

@st.cache_resource
def create_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )


# ============================================================
# 10. CREATE GEMINI LLM
# ============================================================

@st.cache_resource
def create_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )


# ============================================================
# 11. PROCESS VIDEO
# ============================================================

def process_video(url):

    # --------------------------------------------------------
    # VIDEO ID
    # --------------------------------------------------------

    video_id = get_video_id(url)

    if not video_id:

        st.error("❌ Invalid YouTube URL.")

        return None


    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    video_info = get_video_info(video_id)

    if not video_info:

        st.error("❌ Video not found.")

        return None


    # --------------------------------------------------------
    # TRANSCRIPT
    # --------------------------------------------------------

    with st.spinner("Fetching transcript..."):

        transcript = get_transcript(video_id)

    if not transcript:

        return None


    # --------------------------------------------------------
    # LANGCHAIN DOCUMENT
    # --------------------------------------------------------

    document = Document(
        page_content=transcript,
        metadata={
            "video_id": video_id,
            "title": video_info["title"],
            "channel": video_info["channel"]
        }
    )


    # --------------------------------------------------------
    # TEXT SPLITTING
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        [document]
    )


    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    with st.spinner("Creating embeddings..."):

        embeddings = create_embeddings()


    # --------------------------------------------------------
    # CREATE FAISS VECTOR DATABASE
    # --------------------------------------------------------

    with st.spinner("Building vector database..."):

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )


    # --------------------------------------------------------
    # RETURN DATA
    # --------------------------------------------------------

    return {
        "video_id": video_id,
        "video_info": video_info,
        "vectorstore": vectorstore,
        "chunks": chunks
    }


# ============================================================
# 12. YOUTUBE URL INPUT
# ============================================================

youtube_url = st.text_input(
    "🔗 YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


# ============================================================
# 13. PROCESS VIDEO BUTTON
# ============================================================

if st.button(
    "🚀 Process Video",
    type="primary"
):

    if not youtube_url:

        st.warning(
            "⚠️ Please enter a YouTube video URL."
        )

    else:

        with st.spinner(
            "Processing video... Please wait."
        ):

            data = process_video(
                youtube_url
            )

        if data:

            # Store data in session
            st.session_state["vectorstore"] = (
                data["vectorstore"]
            )

            st.session_state["video_info"] = (
                data["video_info"]
            )

            st.session_state["chunks"] = (
                data["chunks"]
            )

            st.session_state["processed"] = True

            st.success(
                "✅ Video processed successfully!"
            )


# ============================================================
# 14. SHOW VIDEO INFORMATION
# ============================================================

if "video_info" in st.session_state:

    video_info = st.session_state[
        "video_info"
    ]

    st.divider()

    st.subheader("📺 Video Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🎬 Title")

        st.write(
            video_info["title"]
        )

    with col2:

        st.markdown("### 👤 Channel")

        st.write(
            video_info["channel"]
        )

    with col3:

        st.markdown("### 👁️ Views")

        st.write(
            video_info["views"]
        )


# ============================================================
# 15. SHOW RAG INFORMATION
# ============================================================

if "chunks" in st.session_state:

    st.info(
        f"📚 Transcript split into "
        f"{len(st.session_state['chunks'])} chunks."
    )


# ============================================================
# 16. QUESTION SECTION
# ============================================================

if "vectorstore" in st.session_state:

    st.divider()

    st.subheader(
        "💬 Ask Questions About The Video"
    )

    query = st.text_input(
        "Your Question",
        placeholder=(
            "What are prompts in LangChain?"
        )
    )


    # --------------------------------------------------------
    # ASK BUTTON
    # --------------------------------------------------------

    if st.button("🔍 Ask"):

        if not query:

            st.warning(
                "⚠️ Please enter a question."
            )

        else:

            vectorstore = (
                st.session_state["vectorstore"]
            )


            # ------------------------------------------------
            # RETRIEVE RELEVANT CHUNKS
            # ------------------------------------------------

            with st.spinner(
                "Searching the transcript..."
            ):

                results = (
                    vectorstore.similarity_search(
                        query,
                        k=3
                    )
                )


            # ------------------------------------------------
            # CREATE CONTEXT
            # ------------------------------------------------

            context = "\n\n".join(
                result.page_content
                for result in results
            )


            # ------------------------------------------------
            # VIDEO INFORMATION
            # ------------------------------------------------

            title = (
                st.session_state["video_info"]
                ["title"]
            )

            channel = (
                st.session_state["video_info"]
                ["channel"]
            )


            # ------------------------------------------------
            # RAG PROMPT
            # ------------------------------------------------

            prompt = f"""
You are a helpful assistant answering questions
about a YouTube video.

The transcript may be in Hindi, English,
or Hinglish.

The user's question may also be in Hindi,
English, or Hinglish.

Understand the meaning of the question
regardless of language.

Answer the user's question using ONLY
the video transcript context provided below.

Do not make up information.

If the answer is not present in the context,
say:

"I could not find the answer in the video transcript."

Video Title:
{title}

Video Channel:
{channel}

Context:
{context}

Question:
{query}

Answer:
"""


            # ------------------------------------------------
            # GEMINI
            # ------------------------------------------------

            llm = create_llm()

            with st.spinner(
                "🤖 Generating answer..."
            ):

                try:

                    response = llm.invoke(
                        prompt
                    )

                except Exception as e:

                    st.error(
                        "❌ Gemini could not generate an answer."
                    )

                    st.write(str(e))

                    response = None


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            if response:

                st.subheader(
                    "🤖 Answer"
                )

                st.write(
                    response.content
                )


                # ------------------------------------------------
                # SHOW SOURCES
                # ------------------------------------------------

                with st.expander(
                    "📚 View Retrieved Transcript Chunks"
                ):

                    for i, result in enumerate(
                        results
                    ):

                        st.markdown(
                            f"### Chunk {i + 1}"
                        )

                        st.write(
                            result.page_content
                        )

                        st.divider()


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • LangChain • Gemini • FAISS • Streamlit"
)