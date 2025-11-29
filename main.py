import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "robot-annotation-bot-index")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"


# Request/Response Models
class ChatQueryRequest(BaseModel):
    query: str
    include_sources: Optional[bool] = False


class ChatQueryResponse(BaseModel):
    answer: str
    sources: Optional[list] = None
    confidence: Optional[str] = None


# Load RAG Chain
def load_rag_chain():
    print("🤖 Initializing Robot Annotation Q&A Bot...")

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        print("❌ Missing API Keys")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"✓ Embeddings loaded: {EMBEDDING_MODEL}")

        vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        print(f"✓ Connected to Pinecone index: {PINECONE_INDEX_NAME}")

        retriever = vector_store.as_retriever(search_kwargs={"k": 8})

        llm = ChatOpenAI(
            model_name=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.1
        )
        print(f"✓ LLM initialized: {LLM_MODEL}")

        # ======= Prompt =======
        prompt_template = """
You are a senior expert in robot-workflow annotation. Your job is to help annotators correctly apply the bearings-and-hopper annotation guidelines.

Your answers must ALWAYS:
- Be detailed but clear
- Be structured, direct, and rule-driven
- Explain WHAT to mark, WHEN to mark it, and HOW to decide reliability
- Use bullet points where helpful
- Never mention question numbers
- Never invent rules outside the provided context
- If the answer isn't covered: say "This scenario is not covered in the current guidelines. Please consult your Manager."

**Context from Guidelines and FAQ:**
{context}

**Annotator's Question:**
{query}

**Provide a clear, direct answer:**
"""

        prompt = PromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # IMPORTANT FIX: your prompt uses {query}, not {question}
        rag_chain = (
            {
                "context": retriever | format_docs,
                "query": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("✅ RAG Chain initialized successfully!")
        return rag_chain

    except Exception as e:
        print(f"❌ CRITICAL ERROR initializing RAG chain: {e}")
        import traceback
        traceback.print_exc()
        return None


# FastAPI App
app = FastAPI(
    title="Robot Annotation Q&A Bot",
    description="AI assistant for robot annotation guidelines",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG_CHAIN = load_rag_chain()


def get_chain():
    if not RAG_CHAIN:
        raise HTTPException(
            status_code=503,
            detail="RAG Chain not initialized. Check API keys & index."
        )
    return RAG_CHAIN


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "Robot Annotation Q&A Bot",
        "version": "1.0.0",
        "usage": "/api/v1/query"
    }


@app.get("/health", tags=["Monitoring"])
async def health():
    return {
        "status": "healthy" if RAG_CHAIN else "degraded",
        "rag_ready": RAG_CHAIN is not None,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "pinecone_index": PINECONE_INDEX_NAME
    }


@app.post("/api/v1/query", response_model=ChatQueryResponse)
async def handle_query(request: ChatQueryRequest, rag_chain: Runnable = Depends(get_chain)):
    try:
        print(f"📝 Query: {request.query[:80]}...")
        answer = await rag_chain.ainvoke(request.query)

        return ChatQueryResponse(answer=answer)

    except Exception as e:
        print(f"❌ Error handling query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
