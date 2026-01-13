from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional
import time
import gradio as gr

from src.config import PDF_DIR
from src.data_loader import EpokaDataLoader
from src.embedding_manager import EmbeddingManager
from src.retrieval import HybridRetriever
from src.response_generator import ResponseGenerator

# FastAPI app
app = FastAPI(title="Epoka University Chatbot API")

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    use_hybrid: Optional[bool] = True

class ChatResponse(BaseModel):
    answer: str
    citations: list
    performance: dict
    university_info: dict

# Global chatbot instance
class EpokaChatbot:
    def __init__(self):
        self.data_loader = EpokaDataLoader()
        self.embedding_manager = EmbeddingManager()
        self.retriever = None
        self.response_gen = ResponseGenerator()
        self.initialized = False
    
    def initialize(self, rebuild_index: bool = False):
        """Initialize or rebuild the chatbot"""
        print("Initializing Epoka University Chatbot...")
        
        # Try to load existing index
        if not rebuild_index and self.embedding_manager.load_index():
            chunks = self.embedding_manager.chunks
            print("✓ Loaded existing index")
        else:
            print("Building new index...")
            # Load and process documents
            pdf_docs = self.data_loader.load_pdfs()
            faq_docs = self.data_loader.load_faqs()
            all_docs = pdf_docs + faq_docs
            
            if not all_docs:
                raise Exception("No documents found! Please add PDFs to data/raw_pdfs/")
            
            # Chunk documents
            chunks = self.data_loader.chunk_documents(all_docs)
            
            # Create embeddings and index
            embeddings = self.embedding_manager.create_embeddings(chunks)
            self.embedding_manager.build_faiss_index(embeddings, chunks)
            print("✓ Built new index")
        
        # Initialize retriever
        self.retriever = HybridRetriever(self.embedding_manager, chunks)
        self.initialized = True
        
        print(f"✓ Chatbot initialized with {len(chunks)} document chunks")
        return True
    
    def query(self, question: str, use_hybrid: bool = True) -> dict:
        """Process a query"""
        if not self.initialized:
            raise Exception("Chatbot not initialized!")
        
        start_time = time.time()
        
        try:
            # Retrieve relevant context
            print(f"Retrieving context for: {question}")
            context, citations = self.retriever.get_relevant_context(question)
            print(f"Retrieved context: {len(context) if context else 0} chars")
            print(f"Citations: {citations}")
            
            # Generate response
            latency = time.time() - start_time
            response = self.response_gen.generate_response(
                question, context, citations, latency
            )
            
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()  # This will show the full error
            latency = time.time() - start_time
            return {
                'query': question,
                'answer': f"Error processing your query: {str(e)}",
                'citations': [],
                'performance': {'response_time': round(latency, 3)},
                'university_info': {
                    'name': 'Epoka University',
                    'suggested_contact': 'admissions@epoka.edu.al'
                }
            }

# Initialize chatbot
chatbot = EpokaChatbot()

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    try:
        chatbot.initialize(rebuild_index=False)
        print("🚀 Epoka University Chatbot is ready!")
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/")
async def root():
    return {
        "message": "Epoka University Chatbot API",
        "status": "running",
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health (GET)",
            "info": "/info (GET)"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if chatbot.initialized else "initializing",
        "initialized": chatbot.initialized
    }

@app.get("/info")
async def get_info():
    if chatbot.initialized and chatbot.retriever:
        return {
            "chunks_count": len(chatbot.embedding_manager.chunks),
            "sources": list(chatbot.embedding_manager.metadata.get('sources', {}).keys()),
            "document_types": list(chatbot.embedding_manager.metadata.get('types', set())),
            "index_size": chatbot.embedding_manager.index.ntotal if chatbot.embedding_manager.index else 0
        }
    return {"error": "Chatbot not initialized"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if not chatbot.initialized:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    response = chatbot.query(request.question, request.use_hybrid)
    return ChatResponse(**response)

# Gradio Interface
def create_gradio_interface():
    """Create a user-friendly Gradio interface"""
    
    def gradio_query(question, history):
        if not question.strip():
            return ""
        
        response = chatbot.query(question)
        
        # Format answer with citations
        answer = response['answer']
        
        # Add citations if available
        if response['citations']:
            answer += "\n\n**Sources:**\n"
            for i, cite in enumerate(response['citations'][:3], 1):
                answer += f"{i}. {cite['source']} (Page {cite['page']})\n"
        
        # Add performance info
        perf = response['performance']
        answer += f"\n⏱️ Response time: {perf['response_time']}s"
        
        return answer
    
    # Create interface (removed theme parameter)
    interface = gr.ChatInterface(
        fn=gradio_query,
        title="🎓 Epoka University Assistant",
        description="Ask me anything about Epoka University programs, admissions, fees, campus life, and more!",
        examples=[
            "What are the tuition fees for international students?",
            "How do I apply for a Master's program?",
            "What scholarships are available?",
            "When is the application deadline?",
            "Tell me about Computer Science programs"
        ]
    )
    
    return interface

# Main execution
if __name__ == "__main__":
    # Initialize chatbot
    print("Starting Epoka University Chatbot...")
    
    # Check if PDFs exist
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDF files found in {PDF_DIR}")
        print("Please add your Epoka University PDFs to the data/raw_pdfs/ directory")
        print("Example: data/raw_pdfs/admission_guide.pdf")
    
    # Initialize
    chatbot.initialize(rebuild_index=False)
    
    # Create Gradio interface
    print("Launching Gradio interface...")
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False # Set to True for temporary public link
    )