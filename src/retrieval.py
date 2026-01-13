import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import re
from src.config import TOP_K, SCORE_THRESHOLD

class HybridRetriever:
    def __init__(self, embedding_manager, chunks: List[Document]):
        self.embedding_manager = embedding_manager
        self.chunks = chunks
        
        # Prepare BM25 for keyword search
        self.bm25_index = None
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        tokenized_corpus = [
            re.findall(r'\w+', chunk.page_content.lower()) 
            for chunk in self.chunks
        ]
        self.bm25_index = BM25Okapi(tokenized_corpus)
    
    def semantic_search(self, query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
        """Semantic search using FAISS"""
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Convert to 2D array with correct shape (1, embedding_dim) and dtype
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Ensure it's contiguous in memory
        if not query_embedding.flags['C_CONTIGUOUS']:
            query_embedding = np.ascontiguousarray(query_embedding)
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.embedding_manager.index.search(query_embedding, k * 2)  # Get extra for filtering
        
        results = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                # Convert inner product to cosine similarity
                cosine_sim = (score + 1) / 2
                if cosine_sim >= SCORE_THRESHOLD:
                    results.append((self.chunks[idx], cosine_sim))
                if len(results) >= k:
                    break
        
        return results
    
    def keyword_search(self, query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
        """Keyword search using BM25"""
        tokenized_query = re.findall(r'\w+', query.lower())
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                # Normalize BM25 score to 0-1 range
                normalized_score = min(scores[idx] / 10, 1.0)
                results.append((self.chunks[idx], normalized_score))
        
        return results
    
    def hybrid_search(self, query: str, k: int = TOP_K, 
                     semantic_weight: float = 0.5) -> List[Tuple[Document, float, str]]:
        """
        Combine semantic and keyword search
        Returns: List of (document, score, search_type)
        """
        # Enhance query for better matching
        enhanced_query = self._enhance_query(query)
        
        # Get results from both methods
        semantic_results = self.semantic_search(enhanced_query, k * 2)
        keyword_results = self.keyword_search(enhanced_query, k * 2)
        
        # Combine results
        scored_docs = {}
        
        # Add semantic results
        for doc, score in semantic_results:
            doc_id = doc.metadata.get('doc_id', '')
            if doc_id not in scored_docs:
                scored_docs[doc_id] = {
                    'doc': doc,
                    'semantic_score': score * semantic_weight,
                    'keyword_score': 0,
                    'type': 'semantic'
                }
        
        # Add keyword results
        for doc, score in keyword_results:
            doc_id = doc.metadata.get('doc_id', '')
            if doc_id in scored_docs:
                scored_docs[doc_id]['keyword_score'] = score * (1 - semantic_weight)
                scored_docs[doc_id]['type'] = 'hybrid'
            else:
                scored_docs[doc_id] = {
                    'doc': doc,
                    'semantic_score': 0,
                    'keyword_score': score * (1 - semantic_weight),
                    'type': 'keyword'
                }
        
        # Calculate combined scores
        combined_results = []
        for doc_data in scored_docs.values():
            combined_score = doc_data['semantic_score'] + doc_data['keyword_score']
            if combined_score > 0:
                combined_results.append((
                    doc_data['doc'],
                    combined_score,
                    doc_data['type']
                ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results[:k]
    
    def get_relevant_context(self, query: str, max_chars: int = 3000) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant context for a query
        Returns: (context_string, citations_list)
        """
        results = self.hybrid_search(query, k=TOP_K)
        
        context_parts = []
        citations = []
        total_chars = 0
        
        for i, (doc, score, search_type) in enumerate(results):
            # Truncate if needed
            content = doc.page_content
            if total_chars + len(content) > max_chars:
                content = content[:max_chars - total_chars]
            
            context_parts.append(f"[{i+1}] {content}")
            
            # Build citation
            citation = {
                'id': i + 1,
                'content': content[:150] + ("..." if len(content) > 150 else ""),
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'type': doc.metadata.get('type', 'unknown'),
                'score': round(score, 3),
                'search_type': search_type
            }
            citations.append(citation)
            
            total_chars += len(content)
            if total_chars >= max_chars:
                break
        
        return "\n\n".join(context_parts), citations
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with synonyms and related terms"""
        query_lower = query.lower()
        
        # Date/calendar related
        if any(word in query_lower for word in ['when', 'date', 'deadline']):
            if 'winter break' in query_lower or 'winter holiday' in query_lower:
                query += " vacation december january"
            elif 'summer break' in query_lower or 'summer holiday' in query_lower:
                query += " vacation june july august"
            elif 'exam' in query_lower:
                query += " examination test assessment schedule"
            elif 'registration' in query_lower:
                query += " enroll enrollment course selection"
        
        # Attendance related
        if 'attendance' in query_lower or 'absent' in query_lower:
            query += " presence class participation requirement policy"
        
        # Fee related
        if any(word in query_lower for word in ['fee', 'cost', 'tuition', 'price']):
            query += " payment euro cost tuition financial"
        
        # Requirement related
        if 'requirement' in query_lower or 'required' in query_lower:
            query += " must need mandatory necessary prerequisite"
        
        return query