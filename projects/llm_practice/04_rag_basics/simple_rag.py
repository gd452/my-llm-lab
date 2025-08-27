"""
RAG (Retrieval-Augmented Generation) 기초
벡터 DB + LLM을 활용한 지식 기반 응답 시스템
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import requests
import json
import os

class SimpleRAG:
    """간단한 RAG 구현"""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 llm_model: str = 'qwen3:8b',
                 collection_name: str = 'knowledge_base'):
        
        # 임베딩 모델
        self.embedder = SentenceTransformer(embedding_model)
        
        # ChromaDB 초기화 (로컬 벡터 DB)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # 컬렉션 생성/로드
        try:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.chroma_client.get_collection(collection_name)
        
        self.llm_model = llm_model
        self.llm_base_url = "http://localhost:11434"
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """문서를 벡터 DB에 추가"""
        
        # 임베딩 생성
        embeddings = self.embedder.encode(documents).tolist()
        
        # ID 생성
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # ChromaDB에 저장
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids
        )
        
        print(f"✅ {len(documents)}개 문서 추가 완료")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """관련 문서 검색"""
        
        # 쿼리 임베딩
        query_embedding = self.embedder.encode(query).tolist()
        
        # 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # 결과 포맷팅
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else 0
            })
        
        return retrieved_docs
    
    def generate_response(self, query: str, context: str) -> str:
        """컨텍스트를 활용한 응답 생성"""
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If you cannot answer based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        # Ollama API 호출
        try:
            response = requests.post(
                f"{self.llm_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: LLM API returned {response.status_code}"
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """RAG 파이프라인 실행"""
        
        # 1. 관련 문서 검색
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        # 2. 컨텍스트 생성
        context = "\n\n".join([doc['document'] for doc in retrieved_docs])
        
        # 3. 응답 생성
        answer = self.generate_response(question, context)
        
        return {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs,
            'context_used': context
        }

class AdvancedRAG:
    """고급 RAG 기법"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def add_documents_with_chunking(self, 
                                   text: str, 
                                   chunk_size: int = 200,
                                   overlap: int = 50):
        """텍스트를 청크로 나누어 저장"""
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            self.documents.append(chunk)
            self.metadata.append({
                'chunk_id': len(chunks) - 1,
                'start_word': i,
                'end_word': min(i + chunk_size, len(words))
            })
        
        # 임베딩 생성
        new_embeddings = self.embedder.encode(chunks)
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        return chunks
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """하이브리드 검색 (의미 + 키워드)"""
        
        # 의미 검색
        query_embedding = self.embedder.encode(query)
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 키워드 검색 (BM25 간단 구현)
        query_words = set(query.lower().split())
        keyword_scores = []
        
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            keyword_scores.append(overlap / max(len(query_words), 1))
        
        keyword_scores = np.array(keyword_scores)
        
        # 점수 결합 (가중 평균)
        combined_scores = 0.7 * semantic_scores + 0.3 * keyword_scores
        
        # 상위 k개 선택
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(combined_scores[idx])))
        
        return results
    
    def rerank_results(self, query: str, initial_results: List[str]) -> List[Tuple[str, float]]:
        """결과 재순위 지정"""
        
        # Cross-encoder 스타일 재순위 (간단한 버전)
        query_doc_pairs = [(query, doc) for doc in initial_results]
        
        # 쿼리와 문서를 함께 임베딩
        combined_texts = [f"Query: {q} Document: {d}" for q, d in query_doc_pairs]
        scores = self.embedder.encode(combined_texts)
        
        # 점수 기반 정렬
        score_magnitudes = np.linalg.norm(scores, axis=1)
        sorted_indices = np.argsort(score_magnitudes)[::-1]
        
        reranked = []
        for idx in sorted_indices:
            reranked.append((initial_results[idx], float(score_magnitudes[idx])))
        
        return reranked
    
    def query_expansion(self, query: str) -> List[str]:
        """쿼리 확장 (유사 용어 추가)"""
        
        # 간단한 쿼리 확장 (실제로는 동의어 사전이나 LLM 활용)
        expansions = {
            "AI": ["artificial intelligence", "machine learning", "deep learning"],
            "programming": ["coding", "software development", "development"],
            "data": ["information", "dataset", "records"]
        }
        
        expanded_queries = [query]
        
        for term, synonyms in expansions.items():
            if term.lower() in query.lower():
                for synonym in synonyms:
                    expanded_queries.append(query.replace(term, synonym))
        
        return expanded_queries

def demo_simple_rag():
    """간단한 RAG 데모"""
    
    print("📚 Simple RAG 데모")
    print("=" * 60)
    
    rag = SimpleRAG()
    
    # 지식 베이스 구축
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Docker is a containerization platform that packages applications with their dependencies.",
        "React is a JavaScript library for building user interfaces, especially single-page applications.",
        "Kubernetes is an orchestration platform for managing containerized applications at scale.",
        "Git is a distributed version control system for tracking changes in source code.",
        "SQL is a domain-specific language used for managing and querying relational databases.",
        "REST API is an architectural style for building web services using HTTP methods."
    ]
    
    metadatas = [
        {"topic": "programming", "category": "language"},
        {"topic": "AI", "category": "technology"},
        {"topic": "DevOps", "category": "tool"},
        {"topic": "web", "category": "framework"},
        {"topic": "DevOps", "category": "orchestration"},
        {"topic": "development", "category": "tool"},
        {"topic": "database", "category": "language"},
        {"topic": "web", "category": "architecture"}
    ]
    
    print("\n1. 문서 추가:")
    rag.add_documents(documents, metadatas)
    
    # 질문하기
    questions = [
        "What is Python used for?",
        "How does containerization work?",
        "What are the benefits of version control?"
    ]
    
    print("\n2. RAG 질의응답:")
    for question in questions:
        print(f"\n  Q: {question}")
        result = rag.query(question, top_k=2)
        print(f"  A: {result['answer'][:200]}...")
        print(f"  Sources used: {len(result['sources'])} documents")

def demo_advanced_rag():
    """고급 RAG 기법 데모"""
    
    print("\n🚀 Advanced RAG 데모")
    print("=" * 60)
    
    adv_rag = AdvancedRAG()
    
    # 긴 텍스트 (청킹 데모)
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural 
    intelligence displayed by humans. Leading AI textbooks define the field as the study of intelligent agents: 
    any device that perceives its environment and takes actions that maximize its chance of successfully achieving 
    its goals. Colloquially, the term artificial intelligence is often used to describe machines that mimic 
    cognitive functions that humans associate with the human mind, such as learning and problem solving.
    
    Machine learning is a subset of AI that provides systems the ability to automatically learn and improve 
    from experience without being explicitly programmed. Machine learning focuses on the development of computer 
    programs that can access data and use it to learn for themselves. The process of learning begins with 
    observations or data, such as examples, direct experience, or instruction, in order to look for patterns 
    in data and make better decisions in the future based on the examples that we provide.
    
    Deep learning is part of a broader family of machine learning methods based on artificial neural networks 
    with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning 
    architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional 
    neural networks have been applied to fields including computer vision, speech recognition, natural language 
    processing, and machine translation.
    """
    
    print("\n1. 문서 청킹:")
    chunks = adv_rag.add_documents_with_chunking(long_text, chunk_size=50, overlap=10)
    print(f"  생성된 청크 수: {len(chunks)}")
    print(f"  첫 번째 청크: {chunks[0][:100]}...")
    
    print("\n2. 하이브리드 검색:")
    query = "machine learning and neural networks"
    results = adv_rag.hybrid_search(query, top_k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. [Score: {score:.3f}] {doc[:80]}...")
    
    print("\n3. 쿼리 확장:")
    original_query = "AI programming"
    expanded = adv_rag.query_expansion(original_query)
    print(f"  원본 쿼리: {original_query}")
    print(f"  확장된 쿼리:")
    for exp_query in expanded[:3]:
        print(f"    - {exp_query}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "advanced":
            demo_advanced_rag()
        else:
            print("Usage: python simple_rag.py [advanced]")
    else:
        demo_simple_rag()
        
        print("\n" + "=" * 60)
        print("💡 더 많은 데모:")
        print("  python simple_rag.py advanced  # 고급 RAG 기법")