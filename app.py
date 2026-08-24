import os

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
# 1. LOAD API KEYS
# ============================================================

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not YOUTUBE_API_KEY:
    YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY")

if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

print("YouTube API Key loaded:", YOUTUBE_API_KEY is not None)
print("Gemini API Key loaded:", GEMINI_API_KEY is not None)


# ============================================================
# 2. CONNECT TO YOUTUBE API
# ============================================================

youtube = build(
    "youtube",
    "v3",
    developerKey=YOUTUBE_API_KEY
)

print("YouTube API connected!")


# ============================================================
# 3. GET VIDEO ID FROM URL
# ============================================================

def get_video_id(url):

    if "v=" in url:
        return url.split("v=")[1].split("&")[0]

    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    else:
        return None


# ============================================================
# 4. GET VIDEO INFORMATION
# ============================================================

def get_video_info(video_id):

    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )

    response = request.execute()

    if not response["items"]:
        return None

    video = response["items"][0]

    title = video["snippet"]["title"]
    channel = video["snippet"]["channelTitle"]
    description = video["snippet"]["description"]
    views = video["statistics"].get("viewCount", "N/A")

    return {
        "title": title,
        "channel": channel,
        "description": description,
        "views": views
    }


# ============================================================
# 5. GET VIDEO TRANSCRIPT
# ============================================================

def get_transcript(video_id):

    api = YouTubeTranscriptApi()

    try:

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

        print("\nTranscript error:")
        print(e)

        return None


# ============================================================
# 6. GET YOUTUBE URL
# ============================================================

url = input("\nEnter YouTube URL: ")

video_id = get_video_id(url)

if not video_id:

    print("\nInvalid YouTube URL.")
    exit()


print("\nVideo ID:", video_id)


# ============================================================
# 7. GET VIDEO INFORMATION
# ============================================================

video_info = get_video_info(video_id)

if not video_info:

    print("\nVideo not found.")
    exit()


title = video_info["title"]
channel = video_info["channel"]
description = video_info["description"]
views = video_info["views"]


print("\nVideo Information")
print("------------------")
print("Title:", title)
print("Channel:", channel)
print("Views:", views)


# ============================================================
# 8. GET TRANSCRIPT
# ============================================================

transcript = get_transcript(video_id)

if not transcript:

    print("\nSorry, transcript is not available right now.")
    exit()


print("\nTranscript fetched successfully!")


# ============================================================
# 9. CREATE LANGCHAIN DOCUMENT
# ============================================================

document = Document(
    page_content=transcript,
    metadata={
        "video_id": video_id,
        "title": title,
        "channel": channel
    }
)

print("\nLangChain Document created!")


# ============================================================
# 10. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents([document])

print("\nNumber of chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0])


# ============================================================
# 11. CREATE EMBEDDING MODEL
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

print("\nEmbedding model ready!")


# ============================================================
# 12. CREATE FAISS VECTOR DATABASE
# ============================================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector database created!")


# ============================================================
# 13. CREATE GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

print("Gemini LLM ready!")


# ============================================================
# 14. CHAT LOOP
# ============================================================

print("\n======================================")
print(" YouTube AI Chatbot is Ready!")
print(" Type 'exit' to quit.")
print("======================================")


while True:

    query = input("\nAsk a question about the video: ")

    if query.lower() == "exit":

        print("\nGoodbye!")

        break


    # --------------------------------------------------------
    # 15. RETRIEVE RELEVANT CHUNKS
    # --------------------------------------------------------

    results = vectorstore.similarity_search(
        query,
        k=3
    )


    # --------------------------------------------------------
    # 16. COMBINE RETRIEVED CHUNKS
    # --------------------------------------------------------

    context = "\n\n".join(
        result.page_content
        for result in results
    )


    # --------------------------------------------------------
    # 17. CREATE RAG PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a helpful assistant answering questions about a YouTube video.

Answer the user's question using ONLY the video transcript
provided in the context below.

Do not make up information.

If the answer is not present in the context, say:

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


    # --------------------------------------------------------
    # 18. GENERATE ANSWER
    # --------------------------------------------------------

    response = llm.invoke(prompt)


    # --------------------------------------------------------
    # 19. PRINT ANSWER
    # --------------------------------------------------------

    print("\nAnswer:")
    print(response.content)