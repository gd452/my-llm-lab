"""
RAG (Retrieval-Augmented Generation) 기초 - Metadata 강화 버전
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
from datetime import datetime

class EnhancedRAG:
    """Metadata를 활용한 향상된 RAG 구현"""
    
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
            # 기존 컬렉션 삭제 (테스트용)
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.llm_model = llm_model
        self.llm_base_url = "http://localhost:11434"
    
    def add_documents(self, 
                      documents: List[str], 
                      metadatas: Optional[List[Dict]] = None,
                      auto_metadata: bool = True):
        """
        문서를 벡터 DB에 추가 (향상된 메타데이터 처리)
        
        Args:
            documents: 저장할 문서 리스트
            metadatas: 각 문서의 메타데이터
            auto_metadata: 자동 메타데이터 생성 여부
        """
        
        # 임베딩 생성
        embeddings = self.embedder.encode(documents).tolist()
        
        # 메타데이터 처리
        if metadatas is None:
            metadatas = []
        
        # 메타데이터 보강
        enhanced_metadatas = []
        for i, doc in enumerate(documents):
            # 기본 메타데이터
            base_metadata = metadatas[i] if i < len(metadatas) else {}
            
            # 자동 메타데이터 추가
            if auto_metadata:
                enhanced_metadata = {
                    **base_metadata,  # 원본 메타데이터 유지
                    "doc_id": f"doc_{i}_{hash(doc) % 100000}",
                    "char_length": len(doc),
                    "word_count": len(doc.split()),
                    "added_at": datetime.now().isoformat(),
                    "embedding_model": "all-MiniLM-L6-v2"
                }
                
                # 문서 타입 자동 감지
                if "코드" in doc or "def " in doc or "class " in doc:
                    enhanced_metadata["doc_type"] = "code"
                elif "?" in doc:
                    enhanced_metadata["doc_type"] = "question"
                else:
                    enhanced_metadata["doc_type"] = "text"
                    
            else:
                # auto_metadata가 False일 때도 최소한의 메타데이터는 필요
                enhanced_metadata = base_metadata if base_metadata else {"doc_id": f"doc_{i}"}
            
            # ChromaDB는 문자열, 숫자, 불린만 지원
            # 복잡한 타입은 JSON 문자열로 변환
            clean_metadata = {}
            for key, value in enhanced_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                elif value is None:
                    clean_metadata[key] = "null"
                else:
                    clean_metadata[key] = str(value)
            
            enhanced_metadatas.append(clean_metadata)
        
        # ID 생성 (유니크하게)
        ids = [meta.get("doc_id", f"doc_{i}") for i, meta in enumerate(enhanced_metadatas)]
        
        # ChromaDB에 저장
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=enhanced_metadatas,
            ids=ids
        )
        
        print(f"✅ {len(documents)}개 문서 추가 완료 (메타데이터 포함)")
        return enhanced_metadatas
    
    def retrieve(self, 
                query: str, 
                top_k: int = 3,
                filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        관련 문서 검색 (메타데이터 필터링 지원)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            filter_metadata: 메타데이터 필터 조건
        """
        
        # 쿼리 임베딩
        query_embedding = self.embedder.encode(query).tolist()
        
        # 검색 파라미터
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }
        
        # 메타데이터 필터 적용
        if filter_metadata:
            query_params["where"] = filter_metadata
        
        # 검색
        results = self.collection.query(**query_params)
        
        # 결과 포맷팅
        retrieved_docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc_info = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0,
                    'relevance_score': 1.0 - (results['distances'][0][i] if results['distances'] else 0)
                }
                retrieved_docs.append(doc_info)
        
        return retrieved_docs
    
    def generate_response(self, query: str, context: str, metadata_context: str = "") -> str:
        """컨텍스트와 메타데이터를 활용한 응답 생성"""
        
        # 메타데이터 정보를 프롬프트에 포함
        metadata_info = f"\nSource Information:\n{metadata_context}\n" if metadata_context else ""
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If you cannot answer based on the context, say so.
{metadata_info}
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
    
    def query(self, 
             question: str, 
             top_k: int = 3,
             use_metadata: bool = True,
             filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        향상된 RAG 파이프라인 실행
        
        Args:
            question: 질문
            top_k: 검색할 문서 수
            use_metadata: 메타데이터 사용 여부
            filter_metadata: 메타데이터 필터
        """
        
        # 1. 관련 문서 검색
        retrieved_docs = self.retrieve(question, top_k=top_k, filter_metadata=filter_metadata)
        
        # 2. 컨텍스트 생성
        context = "\n\n".join([doc['document'] for doc in retrieved_docs])
        
        # 3. 메타데이터 컨텍스트 생성
        metadata_context = ""
        if use_metadata and retrieved_docs:
            metadata_info = []
            for i, doc in enumerate(retrieved_docs, 1):
                meta = doc.get('metadata', {})
                source_info = f"Source {i}: "
                if meta.get('source'):
                    source_info += f"from {meta['source']}"
                if meta.get('added_at'):
                    source_info += f" (added: {meta['added_at'][:10]})"
                if meta.get('topic'):
                    source_info += f" [Topic: {meta['topic']}]"
                metadata_info.append(source_info)
            metadata_context = "\n".join(metadata_info)
        
        # 4. 응답 생성
        answer = self.generate_response(question, context, metadata_context)
        
        return {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs,
            'context_used': context,
            'metadata_used': metadata_context,
            'num_sources': len(retrieved_docs)
        }

class MetadataExamples:
    """메타데이터 활용 예제"""
    
    @staticmethod
    def demonstrate_metadata_importance():
        """메타데이터의 중요성 시연"""
        
        print("\n" + "="*60)
        print("📊 메타데이터가 RAG 시스템에서 중요한 이유")
        print("="*60)
        
        rag = EnhancedRAG()
        
        # 1. 다양한 소스의 문서 추가 (메타데이터 포함)
        documents = [
            "Python은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어입니다.",
            "Python 3.12는 2023년 10월에 출시되었으며 성능이 크게 향상되었습니다.",
            "파이썬은 데이터 과학에서 가장 인기 있는 언어입니다.",
            "Django는 Python 기반의 웹 프레임워크입니다.",
            "머신러닝에서 Python이 널리 사용되는 이유는 풍부한 라이브러리 때문입니다."
        ]
        
        metadatas = [
            {
                "source": "Wikipedia",
                "year": 2020,
                "topic": "history",
                "reliability": "high"
            },
            {
                "source": "Python.org",
                "year": 2023,
                "topic": "release",
                "reliability": "official"
            },
            {
                "source": "Survey2023",
                "year": 2023,
                "topic": "statistics",
                "reliability": "high"
            },
            {
                "source": "Django Docs",
                "year": 2022,
                "topic": "framework",
                "reliability": "official"
            },
            {
                "source": "ML Blog",
                "year": 2023,
                "topic": "machine_learning",
                "reliability": "medium"
            }
        ]
        
        # 문서 추가
        rag.add_documents(documents, metadatas)
        
        print("\n### 1. 필터링 없는 일반 검색")
        result1 = rag.query("Python의 최신 버전은?", top_k=3, use_metadata=True)
        print(f"질문: Python의 최신 버전은?")
        print(f"답변: {result1['answer'][:200]}...")
        print(f"사용된 소스:")
        for source in result1['sources']:
            print(f"  - {source['metadata'].get('source', 'Unknown')}: {source['document'][:50]}...")
        
        print("\n### 2. 공식 소스만 필터링")
        result2 = rag.query(
            "Python의 최신 버전은?", 
            top_k=3,
            filter_metadata={"reliability": "official"}
        )
        print(f"질문: Python의 최신 버전은? (공식 소스만)")
        print(f"답변: {result2['answer'][:200]}...")
        print(f"사용된 소스:")
        for source in result2['sources']:
            print(f"  - {source['metadata'].get('source', 'Unknown')} [{source['metadata'].get('reliability', '')}]")
        
        print("\n### 3. 최신 정보만 필터링 (2023년)")
        result3 = rag.query(
            "Python의 현재 상황은?",
            top_k=3,
            filter_metadata={"year": 2023}
        )
        print(f"질문: Python의 현재 상황은? (2023년 정보만)")
        print(f"답변: {result3['answer'][:200]}...")
        
        return rag

def explain_metadata_best_practices():
    """메타데이터 베스트 프랙티스"""
    
    print("\n" + "="*60)
    print("🎯 RAG 시스템에서 메타데이터 활용 베스트 프랙티스")
    print("="*60)
    
    practices = {
        "1. 필수 메타데이터": [
            "source: 출처 (신뢰도 평가)",
            "timestamp: 생성/수정 시간 (최신성 평가)",
            "doc_type: 문서 유형 (처리 방식 결정)",
            "language: 언어 (다국어 지원)"
        ],
        
        "2. 도메인별 메타데이터": [
            "학술: author, journal, doi, citations",
            "뉴스: publisher, category, region",
            "코드: language, version, license",
            "제품: price, category, brand, rating"
        ],
        
        "3. 성능 최적화 메타데이터": [
            "chunk_id: 청크 식별자",
            "parent_doc: 원본 문서 참조",
            "embedding_version: 임베딩 모델 버전",
            "quality_score: 품질 점수"
        ],
        
        "4. 메타데이터 활용 시나리오": [
            "시간 필터링: 최신 정보만 검색",
            "신뢰도 필터링: 공식 소스 우선",
            "언어 필터링: 특정 언어 문서만",
            "권한 관리: 사용자별 접근 제어"
        ]
    }
    
    for category, items in practices.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n### 💡 핵심 포인트:")
    print("""
    1. 메타데이터는 검색 정확도를 크게 향상시킵니다
    2. 필터링으로 관련성 높은 결과만 선택 가능
    3. 출처 추적과 버전 관리가 가능
    4. 컨텍스트 윈도우를 효율적으로 사용
    5. 답변의 신뢰도와 투명성 향상
    """)

def compare_with_without_metadata():
    """메타데이터 있음/없음 비교"""
    
    print("\n" + "="*60)
    print("⚖️ 메타데이터 있음 vs 없음 비교")
    print("="*60)
    
    # 메타데이터 없는 RAG
    basic_rag = EnhancedRAG(collection_name="basic_rag")
    docs = [
        "AI는 미래 기술입니다.",
        "AI는 위험할 수 있습니다.",
        "AI는 일자리를 대체합니다."
    ]
    basic_rag.add_documents(docs, auto_metadata=False)
    
    # 메타데이터 있는 RAG  
    enhanced_rag = EnhancedRAG(collection_name="enhanced_rag")
    docs_with_meta = [
        ("AI는 미래 기술입니다.", {"source": "Tech Report 2024", "sentiment": "positive", "reliability": "high"}),
        ("AI는 위험할 수 있습니다.", {"source": "Ethics Paper 2023", "sentiment": "negative", "reliability": "medium"}),
        ("AI는 일자리를 대체합니다.", {"source": "Economic Study 2023", "sentiment": "neutral", "reliability": "high"})
    ]
    
    for doc, meta in docs_with_meta:
        enhanced_rag.add_documents([doc], [meta])
    
    print("\n### 검색 결과 비교:")
    
    query = "AI의 영향은?"
    
    print(f"\n질문: {query}")
    print("\n1. 메타데이터 없음:")
    basic_result = basic_rag.query(query, use_metadata=False)
    print(f"   답변: {basic_result['answer'][:150]}...")
    print(f"   소스 구분: 불가능")
    
    print("\n2. 메타데이터 있음:")
    enhanced_result = enhanced_rag.query(query, use_metadata=True)
    print(f"   답변: {enhanced_result['answer'][:150]}...")
    print(f"   소스 정보:")
    for source in enhanced_result['sources']:
        meta = source['metadata']
        print(f"     - {meta.get('source', 'Unknown')} [{meta.get('reliability', 'unknown')} reliability]")
    
    print("\n3. 신뢰도 높은 소스만 필터링:")
    filtered_result = enhanced_rag.query(
        query, 
        filter_metadata={"reliability": "high"}
    )
    print(f"   답변: {filtered_result['answer'][:150]}...")
    print(f"   사용된 고신뢰도 소스: {len(filtered_result['sources'])}개")

if __name__ == "__main__":
    # 메타데이터 중요성 시연
    rag_system = MetadataExamples.demonstrate_metadata_importance()
    
    # 베스트 프랙티스 설명
    explain_metadata_best_practices()
    
    # 비교 분석
    compare_with_without_metadata()
    
    print("\n" + "="*60)
    print("✅ 메타데이터를 활용한 RAG 시스템 구축 완료!")
    print("="*60)