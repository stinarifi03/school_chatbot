import time
from typing import Dict, Any, List
import os
import re

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not installed. Run: pip install openai")

class ResponseGenerator:
    def __init__(self):
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        
        if OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key)
            self.use_gpt = True
            print("✅ OpenAI GPT is ready!")
        else:
            self.client = None
            self.use_gpt = False
            print("⚠️  OpenAI API key not set. Using fallback mode.")
        
        self.model = "gpt-4o-mini"  # Fast, cheap, and good quality
        
        # Fallback templates
        self.templates = {
            'greeting': "Hello! I'm the Epoka University assistant. I can help you with information about programs, admissions, fees, scholarships, and more. What would you like to know?",
            'farewell': "You're welcome! If you have more questions, feel free to ask. For additional assistance, contact admissions@epoka.edu.al",
            'no_info': "I couldn't find specific information about that in the available documents. I recommend:\n\n• Contacting the admissions office: admissions@epoka.edu.al\n• Visiting the official website: www.epoka.edu.al\n• Calling the main office for direct assistance",
        }
    
    def generate_response(self, query: str, context: str, citations: list, 
                         latency: float) -> Dict[str, Any]:
        """Generate a response using OpenAI GPT or fallback"""
        
        query_lower = query.lower()
        
        # Handle greetings
        if any(word in query_lower.split()[:2] for word in ['hello', 'hi', 'hey']):
            return self._format_response(
                answer=self.templates['greeting'],
                citations=[],
                query=query,
                latency=latency
            )
        
        # Handle farewells/thanks
        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks', 'thank you']) and len(query.split()) < 5:
            return self._format_response(
                answer=self.templates['farewell'],
                citations=[],
                query=query,
                latency=latency
            )
        
        # Check if we have context
        if not context or len(context.strip()) < 30:
            return self._format_response(
                answer=self.templates['no_info'],
                citations=[],
                query=query,
                latency=latency
            )
        
        # Generate answer with GPT or fallback
        if self.use_gpt:
            answer = self._generate_with_gpt(query, context)
        else:
            answer = self._generate_fallback(query, context)
        
        return self._format_response(answer, citations, query, latency)
    
    def _generate_with_gpt(self, query: str, context: str) -> str:
        """Generate answer using OpenAI GPT"""
        
        # Build the system prompt
        system_prompt = """You are an assistant for Epoka University. Your job is to answer student questions accurately based on official university documents.

Rules:
1. Answer directly and concisely (2-4 sentences usually)
2. Use ONLY information from the provided context
3. If context has dates, fees, or numbers, state them exactly
4. Be factual - never make assumptions or add information not in the context
5. If the context doesn't answer the question, say: "I don't have that specific information. Please contact admissions@epoka.edu.al"
6. Keep answers under 150 words unless more detail is needed"""

        user_prompt = f"""Question: {query}

Context from official documents:
{context[:3000]}

Provide a clear, accurate answer:"""

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,  # Limit response length
                temperature=0.3,  # Lower for more factual responses
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return self._generate_fallback(query, context)
    
    def _generate_fallback(self, query: str, context: str) -> str:
        """Enhanced fallback method when GPT is not available"""
        
        query_lower = query.lower()
        sentences = [s.strip() + '.' for s in context.split('.') if len(s.strip()) > 20]
        
        # Date/Calendar queries
        if any(word in query_lower for word in ['when', 'date', 'deadline', 'schedule', 'calendar']):
            date_keywords = ['january', 'february', 'march', 'april', 'may', 'june', 
                           'july', 'august', 'september', 'october', 'november', 'december',
                           'break', 'vacation', 'exam', 'semester', 'deadline', 'start', 'end',
                           'from', 'to', 'until', '2024', '2025']
            relevant = [s for s in sentences if any(kw in s.lower() for kw in date_keywords)]
            if relevant:
                return ' '.join(relevant[:3])
        
        # Fee/Cost queries
        elif any(word in query_lower for word in ['fee', 'cost', 'tuition', 'price', 'pay']):
            fee_keywords = ['euro', '€', 'fee', 'tuition', 'cost', 'price', 'payment', 'eur']
            relevant = [s for s in sentences if any(kw in s.lower() for kw in fee_keywords)]
            if relevant:
                return ' '.join(relevant[:3])
        
        # Admission/Application queries
        elif any(word in query_lower for word in ['admission', 'apply', 'application', 'enroll']):
            adm_keywords = ['admission', 'application', 'apply', 'requirement', 'document', 
                           'transcript', 'certificate', 'submit', 'eligible']
            relevant = [s for s in sentences if any(kw in s.lower() for kw in adm_keywords)]
            if relevant:
                return ' '.join(relevant[:4])
        
        # Requirement queries
        elif any(word in query_lower for word in ['requirement', 'required', 'need', 'must']):
            req_keywords = ['requirement', 'required', 'must', 'need', 'necessary', 
                          'prerequisite', 'document', 'mandatory']
            relevant = [s for s in sentences if any(kw in s.lower() for kw in req_keywords)]
            if relevant:
                return ' '.join(relevant[:4])
        
        # Attendance queries
        elif 'attendance' in query_lower or 'absent' in query_lower:
            att_keywords = ['attendance', 'absent', 'presence', 'class', 'participate', 
                          'percentage', '%', 'miss']
            relevant = [s for s in sentences if any(kw in s.lower() for kw in att_keywords)]
            if relevant:
                return ' '.join(relevant[:3])
        
        # Program/Course queries
        elif any(word in query_lower for word in ['program', 'course', 'major', 'degree', 'bachelor', 'master']):
            prog_keywords = ['program', 'course', 'degree', 'bachelor', 'master', 'major',
                           'faculty', 'department', 'study']
            relevant = [s for s in sentences if any(kw in s.lower() for kw in prog_keywords)]
            if relevant:
                return ' '.join(relevant[:4])
        
        # Scholarship queries
        elif 'scholarship' in query_lower or 'financial aid' in query_lower:
            sch_keywords = ['scholarship', 'financial', 'aid', 'grant', 'discount', 'assistance']
            relevant = [s for s in sentences if any(kw in s.lower() for kw in sch_keywords)]
            if relevant:
                return ' '.join(relevant[:4])
        
        # General: Score sentences by query word overlap
        query_words = set(query_lower.split())
        query_words.discard('what')
        query_words.discard('when')
        query_words.discard('where')
        query_words.discard('how')
        query_words.discard('is')
        query_words.discard('are')
        query_words.discard('the')
        
        scored = []
        for s in sentences:
            s_words = set(s.lower().split())
            score = len(query_words.intersection(s_words))
            if score > 0:
                scored.append((s, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            return ' '.join([s[0] for s in scored[:4]])
        
        return ' '.join(sentences[:3]) if sentences else context[:400]
    
    def _format_response(self, answer: str, citations: list, 
                        query: str, latency: float) -> Dict[str, Any]:
        """Format the final response"""
        
        # Format citations
        formatted_citations = []
        for cite in citations[:5]:
            formatted_citations.append({
                'source': cite['source'],
                'page': cite.get('page', 'N/A'),
                'excerpt': cite['content'][:150] + '...' if len(cite['content']) > 150 else cite['content'],
                'relevance_score': cite['score']
            })
        
        return {
            'query': query,
            'answer': answer,
            'citations': formatted_citations,
            'performance': {
                'response_time': round(latency, 3),
                'sources_used': len(set(c['source'] for c in citations)),
                'total_citations': len(citations)
            },
            'university_info': {
                'name': 'Epoka University',
                'contact': 'admissions@epoka.edu.al',
                'website': 'www.epoka.edu.al'
            }
        }