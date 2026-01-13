import numpy as np
import pickle
import faiss
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL, EMBEDDING_DIM, FAISS_INDEX_PATH, METADATA_PATH

class EmbeddingManager:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        # Use LangChain's HuggingFaceEmbeddings - no sentence-transformers needed!
        self.model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.index = None
        self.chunks = []
    
    def create_embeddings(self, chunks: List[Document]) -> np.ndarray:
        """Create embeddings for all chunks"""
        print("Creating embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        
        # Use LangChain's embed_documents method
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.embed_documents(texts)
        
        # Convert to float32 numpy array
        embeddings = np.array(embeddings, dtype=np.float32)
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, chunks: List[Document]):
        """Build and save FAISS index"""
        print("Building FAISS index...")
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.array(embeddings, dtype=np.float32)
        if not embeddings.flags['C_CONTIGUOUS']:
            embeddings = np.ascontiguousarray(embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)
        self.chunks = chunks
        
        self.save_index()
        print(f"Index built with {len(chunks)} chunks")
        return self.index
    
    def save_index(self):
        """Save FAISS index and metadata"""
        print("Saving index to disk...")
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump({'chunks': self.chunks}, f)
        
        print(f"Index saved to {FAISS_INDEX_PATH}")
    
    def load_index(self) -> bool:
        """Load existing FAISS index"""
        try:
            if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
                print("Loading existing index...")
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                
                with open(METADATA_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data['chunks']
                
                print(f"Loaded index with {len(self.chunks)} chunks")
                return True
            return False
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        return np.array(self.model.embed_query(query))