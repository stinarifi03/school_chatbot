import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import hashlib
from src.config import PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP

class EpokaDataLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdfs(self) -> List[Document]:
        """Load all PDFs and TXT files from the raw_pdfs directory"""
        documents = []
        
        # Load PDF files
        for pdf_file in PDF_DIR.glob("*.pdf"):
            print(f"Loading {pdf_file.name}...")
            try:
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                
                # Add source metadata
                for doc in pdf_docs:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["type"] = "pdf"
                    doc.metadata["doc_id"] = hashlib.md5(
                        f"{pdf_file.name}_{doc.metadata.get('page', 0)}".encode()
                    ).hexdigest()[:8]
                
                documents.extend(pdf_docs)
                print(f"  → Loaded {len(pdf_docs)} pages from {pdf_file.name}")
            except Exception as e:
                print(f"  → Error loading {pdf_file.name}: {e}")
        
        # Load TXT files
        for txt_file in PDF_DIR.glob("*.txt"):
            print(f"Loading {txt_file.name}...")
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                txt_docs = loader.load()
                
                # Add source metadata
                for doc in txt_docs:
                    doc.metadata["source"] = txt_file.name
                    doc.metadata["type"] = "txt"
                    doc.metadata["doc_id"] = hashlib.md5(
                        f"{txt_file.name}".encode()
                    ).hexdigest()[:8]
                
                documents.extend(txt_docs)
                print(f"  → Loaded {len(txt_docs)} document(s) from {txt_file.name}")
            except Exception as e:
                print(f"  → Error loading {txt_file.name}: {e}")
        
        return documents
    
    def load_faqs(self) -> List[Document]:
        """Load ALL FAQ files from faqs directory"""
        faq_docs = []
        faq_dir = PDF_DIR.parent / "faqs"  # Gets data/faqs directory
        
        if not faq_dir.exists():
            return []
        
        # Load all .txt files in the faqs directory
        for faq_file in faq_dir.glob("*.txt"):
            print(f"Loading FAQ file: {faq_file.name}")
            try:
                with open(faq_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split by question-answer pairs (separated by blank lines)
                sections = content.split("\n\n")
                
                for i, section in enumerate(sections):
                    if section.strip() and not section.strip().startswith("==="):
                        doc = Document(
                            page_content=section.strip(),
                            metadata={
                                "source": faq_file.name,
                                "type": "faq",
                                "doc_id": f"faq_{faq_file.stem}_{i}"
                            }
                        )
                        faq_docs.append(doc)
            except Exception as e:
                print(f"Error loading {faq_file.name}: {e}")
        
        print(f"Loaded {len(faq_docs)} FAQ entries")
        return faq_docs
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for embedding"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_metadata_index(self, chunks: List[Document]) -> Dict[str, Any]:
        """Create a searchable metadata index"""
        metadata_index = {
            "sources": {},
            "types": set(),
            "chunk_ids": []
        }
        
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            doc_type = chunk.metadata.get("type", "unknown")
            chunk_id = chunk.metadata.get("doc_id", "")
            
            metadata_index["types"].add(doc_type)
            metadata_index["chunk_ids"].append(chunk_id)
            
            if source not in metadata_index["sources"]:
                metadata_index["sources"][source] = 0
            metadata_index["sources"][source] += 1
        
        return metadata_index