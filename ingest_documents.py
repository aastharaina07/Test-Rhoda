import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()

# Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "robot-annotation-bot-index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Document paths
DOCUMENTS_DIR = "./documents"
ANNOTATION_GUIDELINE_PDF = "[Encord _-_ Rhoda] Annotation Guideline (1).pdf"
FAQ_PDF = "FAQ Sheet Rhoda.pdf"

def create_pinecone_index():
    """Create Pinecone index if it doesn't exist."""
    print("🔧 Initializing Pinecone...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME in existing_indexes:
        print(f"✓ Index '{PINECONE_INDEX_NAME}' already exists")
        return pc
    
    print(f"📦 Creating new index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    # Wait for index to be ready
    print("⏳ Waiting for index to be ready...")
    while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
        time.sleep(1)
    
    print("✅ Index created successfully!")
    return pc

def load_documents():
    """Load PDF and text documents."""
    print("\n📚 Loading documents...")
    documents = []
    
    doc_files = [
        {
            "path": os.path.join(DOCUMENTS_DIR, ANNOTATION_GUIDELINE_PDF),
            "type": "pdf",
            "metadata": {"source": "Annotation Guidelines", "type": "guideline"}
        },
        {
            "path": os.path.join(DOCUMENTS_DIR, FAQ_PDF),
            "type": "pdf",
            "metadata": {"source": "FAQ", "type": "faq"}
        }
    ]
    
    for doc_info in doc_files:
        doc_path = doc_info["path"]
        
        if not os.path.exists(doc_path):
            print(f"⚠️  Warning: File not found: {doc_path}")
            continue
        
        try:
            print(f"📖 Loading: {os.path.basename(doc_path)}")
            
            if doc_info["type"] == "pdf":
                loader = PyPDFLoader(doc_path)
            else:
                loader = TextLoader(doc_path)
            
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata.update(doc_info["metadata"])
            
            documents.extend(loaded_docs)
            print(f"   ✓ Loaded {len(loaded_docs)} pages")
            
        except Exception as e:
            print(f"   ❌ Error loading {doc_path}: {e}")
    
    print(f"\n✅ Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    print("\n✂️  Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    if chunks:
        print(f"\n📝 Sample chunk (first 200 chars):")
        print(f"   {chunks[0].page_content[:200]}...")
    
    return chunks

def upload_to_pinecone(chunks):
    """Upload document chunks to Pinecone."""
    print("\n🚀 Uploading to Pinecone...")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"✓ Embeddings model loaded: {EMBEDDING_MODEL}")
        
        print(f"📤 Uploading {len(chunks)} chunks to index '{PINECONE_INDEX_NAME}'...")
        
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        
        print("✅ Upload complete!")
        return vector_store
        
    except Exception as e:
        print(f"❌ Error uploading to Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_retrieval(vector_store):
    """Test the retrieval system with sample queries."""
    print("\n🧪 Testing retrieval system...")
    
    test_queries = [
        "How should I mark frames when the robot hits the camera?",
        "What makes a box on table milestone reliable?",
        "When should screening be marked as unhealthy?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        results = vector_store.similarity_search(query, k=2)
        
        if results:
            print(f"   ✓ Found {len(results)} relevant chunks")
            print(f"   📄 Top result: {results[0].page_content[:150]}...")
        else:
            print("   ⚠️  No results found")

def main():
    """Main ingestion pipeline."""
    print("=" * 70)
    print("🤖 ROBOT ANNOTATION BOT - DOCUMENT INGESTION")
    print("=" * 70)
    
    if not PINECONE_API_KEY:
        print("❌ ERROR: PINECONE_API_KEY not found in .env file")
        return
    
    create_pinecone_index()
    
    documents = load_documents()
    if not documents:
        print("❌ No documents loaded. Check your document paths.")
        return
    
    chunks = split_documents(documents)
    if not chunks:
        print("❌ No chunks created.")
        return
    
    vector_store = upload_to_pinecone(chunks)
    if not vector_store:
        print("❌ Upload failed.")
        return
    
    test_retrieval(vector_store)
    
    print("\n" + "=" * 70)
    print("✅ INGESTION COMPLETE!")
    print("=" * 70)
    print(f"📊 Summary:")
    print(f"   • Documents processed: {len(documents)}")
    print(f"   • Chunks created: {len(chunks)}")
    print(f"   • Index name: {PINECONE_INDEX_NAME}")
    print(f"   • Embedding model: {EMBEDDING_MODEL}")
    print(f"\n🎯 Your bot is now ready to answer questions!")
    print(f"   Run: python main.py")

if __name__ == "__main__":
    main()