# RAGSystem VectorStore 수정 코드

## 문제점
- VectorStore의 `add_documents` 함수가 메타데이터를 받지 않음
- RAGSystem에서 메타데이터와 함께 호출하여 에러 발생
- ChromaDB 호환성 문제

## 해결 방법

### 1. VectorStore 클래스 수정

```python
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, collection_name="knowledge_base"):
        # 새로운 ChromaDB 클라이언트 (호환성 문제 해결)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # 임베딩 모델
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 컬렉션 생성/로드
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"기존 컬렉션 '{collection_name}' 로드됨")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"새 컬렉션 '{collection_name}' 생성됨")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """문서 추가 (메타데이터 지원)"""
        embeddings = self.embedding_model.encode(documents)
        
        # 메타데이터가 없으면 빈 딕셔너리로 초기화
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # ChromaDB에 추가
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        print(f"{len(documents)}개 문서 추가됨")
    
    def search(self, query: str, n_results: int = 5):
        """유사도 검색"""
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        return results
    
    def delete_collection(self):
        """컬렉션 삭제"""
        try:
            self.client.delete_collection(self.collection.name)
            print(f"컬렉션 '{self.collection.name}' 삭제됨")
        except Exception as e:
            print(f"컬렉션 삭제 실패: {e}")
    
    def get_collection_info(self):
        """컬렉션 정보 조회"""
        count = self.collection.count()
        print(f"컬렉션 '{self.collection.name}'에 {count}개 문서가 저장됨")
        return count
```

### 2. RAGSystem 클래스 수정

```python
class RAGSystem:
    """완전한 RAG 시스템"""
    
    def __init__(self, llm_model="qwen3:8b", embedding_model="all-MiniLM-L6-v2"):
        # LLM 초기화
        self.llm = Ollama(model=llm_model)
        
        # 벡터 스토어
        self.vector_store = VectorStore("rag_system")
        
        # 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def add_document(self, text: str, source: str = "unknown"):
        """문서 추가 (자동 청킹)"""
        # 텍스트를 청크로 분할
        chunks = self.text_splitter.split_text(text)
        
        # 메타데이터 생성
        metadatas = [
            {
                "source": source,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]
        
        # 벡터 DB에 추가 (메타데이터와 함께)
        self.vector_store.add_documents(chunks, metadatas)
        return len(chunks)
    
    def query(self, question: str, top_k: int = 3, use_thinking: bool = False):
        """RAG 기반 질의응답"""
        
        # 1. 관련 문서 검색
        search_results = self.vector_store.search(question, n_results=top_k)
        
        # 2. 컨텍스트 생성
        context_docs = search_results['documents'][0] if search_results['documents'] else []
        context = "\n\n".join(context_docs)
        
        # 3. 프롬프트 구성
        if use_thinking:
            prompt = f"""
당신은 지식 베이스를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

참고할 수 있는 정보:
{context}

질문: {question}

위의 정보를 바탕으로 질문에 답변해주세요. 
정보가 부족하거나 관련이 없는 경우 "제공된 정보로는 답변할 수 없습니다"라고 말해주세요.

답변:
"""
        else:
            prompt = f"""
다음 정보를 바탕으로 질문에 답변해주세요:

{context}

질문: {question}

답변:
"""
        
        # 4. LLM 호출
        try:
            response = self.llm(prompt)
            answer = response.strip()
        except Exception as e:
            answer = f"LLM 호출 중 오류가 발생했습니다: {e}"
        
        # 5. 결과 반환
        return {
            "answer": answer,
            "sources": context_docs,
            "num_sources": len(context_docs),
            "question": question
        }
```

### 3. 사용 예시

```python
# RAG 시스템 초기화
rag = RAGSystem()

# 문서 추가
documents = [
    """
    회사 규정 문서
    
    1. 근무 시간: 오전 9시 - 오후 6시 (점심시간 12시-1시)
    2. 재택근무: 주 2회 가능 (월/금 권장)
    3. 휴가: 연차 15일, 병가 10일
    4. 교육 지원: 연간 200만원 한도
    5. 회의: 매주 월요일 10시 팀 미팅
    """,
    
    """
    프로젝트 가이드라인
    
    1. 코드 리뷰: 모든 PR은 2명 이상의 리뷰 필요
    2. 테스트: 코드 커버리지 80% 이상 유지
    3. 문서화: 모든 공개 API는 문서화 필수
    4. 브랜치: feature/*, bugfix/*, hotfix/* 규칙 준수
    5. 배포: 매주 화요일, 목요일 정기 배포
    """
]

sources = ["회사규정", "프로젝트가이드"]

# 문서 추가
for i, doc in enumerate(documents):
    chunks = rag.add_document(doc, source=sources[i])
    print(f"문서 {i+1}: {chunks}개 청크로 분할")

# 질문 테스트
questions = [
    "재택근무는 언제 가능한가요?",
    "코드 리뷰 규칙은 무엇인가요?",
    "점심시간은 언제인가요?"
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"❓ 질문: {question}")
    
    result = rag.query(question)
    
    print(f"\n💡 답변: {result['answer']}")
    print(f"\n📚 참고한 소스 ({result['num_sources']}개):")
    for i, source in enumerate(result['sources'][:2]):
        print(f"  [{i+1}] {source[:100]}...")
```

## 주요 변경사항

1. **VectorStore.add_documents()**: 메타데이터 매개변수 추가
2. **ChromaDB 클라이언트**: `PersistentClient` 사용으로 호환성 문제 해결
3. **RAGSystem.add_document()**: 메타데이터와 함께 VectorStore 호출
4. **에러 처리**: 메타데이터가 없는 경우 빈 딕셔너리로 초기화

## 테스트 방법

1. 위의 VectorStore 클래스를 노트북에 복사
2. RAGSystem 클래스를 노트북에 복사
3. 사용 예시 코드 실행
4. 에러 없이 작동하는지 확인
