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

class ChatQueryRequest(BaseModel):
    query: str
    include_sources: Optional[bool] = False

class ChatQueryResponse(BaseModel):
    answer: str
    sources: Optional[list] = None
    confidence: Optional[str] = None

def load_rag_chain():
    """Initialize the RAG chain for robot annotation guidelines."""
    print("🤖 Initializing Robot Annotation Q&A Bot...")
    
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        print("❌ CRITICAL ERROR: Missing API keys in .env file")
        print("   Required: OPENAI_API_KEY, PINECONE_API_KEY")
        return None
        
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"✓ Embeddings model loaded: {EMBEDDING_MODEL}")
        
        # Connect to Pinecone vector store
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        print(f"✓ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Create retriever with more relevant results
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 8
            }
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name=LLM_MODEL, 
            openai_api_key=OPENAI_API_KEY,
            temperature=0.1
        )
        print(f"✓ LLM initialized: {LLM_MODEL}")
        
                # Custom prompt for robot annotation guidelines
        prompt_template = """You are a senior expert in robot-workflow annotation. Your job is to help annotators correctly apply the bearings-and-hopper annotation guidelines.

Your answers must ALWAYS:
- Be detailed but clear
- Be structured, direct, and rule-driven
- Explain EXACTLY what to mark (milestones, action points, screening)
- Explain WHEN to mark (precise moment rules)
- Explain HOW to mark (reliability, simultaneous arm rules, healthy/unhealthy)
- Use bullet points for complex scenarios
- Never mention question numbers or FAQ references
- Never invent rules outside the given context
- If the context doesn't contain the answer, say: "This scenario is not covered in the current guidelines. Please consult your Manager."

**Context from Guidelines and FAQ:**
{context}

**Annotator's Question:**
{query}

**CRITICAL INSTRUCTIONS:**
1. Answer ONLY based on the provided context above
2. Give direct, clear answers WITHOUT mentioning question numbers (like Q1, Q20, etc.)
3. Be specific about WHAT to mark, WHEN to mark it, and WHETHER it's reliable/unreliable
4. Use simple, natural language as if explaining to a colleague
5. For screening decisions, clearly state whether to mark as healthy/unhealthy and whether to submit/skip
6. If the answer isn't in the context, say: "This scenario is not covered in the current guidelines. Please consult with your Manager."

**Provide a clear, direct answer:**"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create a simple RAG chain using LCEL
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✅ RAG Chain Initialized Successfully!")
        print("=" * 60)
        return rag_chain
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to initialize RAG Chain: {e}")
        import traceback
        traceback.print_exc()
        return None

# Initialize FastAPI app
app = FastAPI(
    title="Robot Annotation Q&A Bot",
    description="AI-powered assistant for robot annotation guidelines and FAQ",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG chain at startup
RAG_CHAIN = load_rag_chain()

def get_chain():
    """Dependency to ensure RAG chain is available."""
    if not RAG_CHAIN:
        raise HTTPException(
            status_code=503, 
            detail="Service Unavailable: RAG Chain not initialized. Check API keys and Pinecone connection."
        )
    return RAG_CHAIN

@app.get("/", tags=["Info"])
async def root():
    """Welcome endpoint with usage information."""
    return {
        "service": "Robot Annotation Q&A Bot",
        "version": "1.0.0",
        "description": "Ask questions about robot annotation guidelines and get instant answers",
        "endpoints": {
            "health": "/health",
            "query": "/api/v1/query"
        },
        "example_questions": [
            "How should I mark frames when the robot hits the camera?",
            "What makes a 'box on table' milestone reliable?",
            "When should I mark screening as unhealthy?",
            "How to handle simultaneous plastic and paper disposal?",
            "What if the robot completes only 2 milestones?"
        ]
    }

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint to confirm the service is running."""
    return {
        "status": "healthy" if RAG_CHAIN else "degraded",
        "rag_chain_ready": RAG_CHAIN is not None,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "index_name": PINECONE_INDEX_NAME
    }

@app.post("/api/v1/query", response_model=ChatQueryResponse, tags=["Q&A"])
async def handle_query(
    request: ChatQueryRequest,
    rag_chain: Runnable = Depends(get_chain)
):
    """
    Handle annotation-related questions and return answers based on guidelines.
    
    **Example Questions:**
    - "How do I mark frames when the lid opens accidentally?"
    - "What is considered reliable for bearings in hopper?"
    - "Should I skip the task if only 2 milestones are achieved?"
    - "How to handle human intervention in the video?"
    """
    try:
        print(f"📝 Query received: {request.query[:100]}...")
        
        # Invoke RAG chain with the query
        answer = await rag_chain.ainvoke(request.query)
        
        # Prepare response
        result = ChatQueryResponse(answer=answer)
        
        print(f"✅ Answer generated successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error during query processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing your question: {str(e)}"
        )

@app.get("/api/v1/guidelines/summary", tags=["Reference"])
async def get_guidelines_summary():
    """Get a quick summary of key annotation rules."""
    return {
        "milestones": [
            "1. Box on the table",
            "2. Pull the tab",
            "3. Lid open",
            "4. Bearings in the hopper",
            "5. Box disposed",
            "6. Plastic bag disposed",
            "7. Paper disposed"
        ],
        "screening_rules": {
            "healthy": "Mark as healthy when 4+ milestones are completed",
            "unhealthy_cases": [
                "0-3 milestones completed",
                "Robot doesn't move",
                "Robot hits cameras",
                "Frozen or black camera views",
                "Mechanical failures",
                "Human intervention",
                "Arm shaking/vibration > 100 frames"
            ]
        },
        "submission_rules": {
            "skip_task": [
                "No milestones achieved",
                "Only 1-2 milestones achieved",
                "Human intervention occurs",
                "Unrelated task appears",
                "Video incomplete/frozen"
            ],
            "submit_task": [
                "3-4 milestones: Mark unhealthy, still submit",
                "4+ milestones: Mark healthy, submit"
            ]
        },
        "critical_reminders": [
            "Mark frames at exact moments (no delay)",
            "Simultaneous actions: mark by arm (right/left segment)",
            "Skip unrelated tasks (non-bearings workflow)",
            "Both grippers must release before marking 'box on table'",
            "Assess reliability independently of task order"
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 Starting Robot Annotation Q&A Bot on port {port}")
    print(f"📖 Access documentation at: http://localhost:{port}/docs")
    print(f"🏥 Health check at: http://localhost:8000/health\n")
    uvicorn.run(app, host="0.0.0.0", port=port)