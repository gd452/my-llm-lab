"""
임베딩 기초 - 텍스트를 벡터로 변환
Sentence Transformers를 사용한 실용적인 예제
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json

class EmbeddingBasics:
    """임베딩 기초 클래스"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        가벼운 임베딩 모델 초기화
        all-MiniLM-L6-v2: 빠르고 효율적 (384 차원)
        all-mpnet-base-v2: 더 정확함 (768 차원)
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}
        
    def get_embedding(self, text: str) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        embedding = self.model.encode(text)
        self.embeddings_cache[text] = embedding
        return embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """여러 텍스트를 한번에 임베딩"""
        return self.model.encode(texts)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트의 유사도 계산"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # 코사인 유사도 계산
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    
    def find_most_similar(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """가장 유사한 문서 찾기"""
        query_emb = self.get_embedding(query)
        doc_embs = self.get_embeddings_batch(documents)
        
        # 유사도 계산
        similarities = cosine_similarity([query_emb], doc_embs)[0]
        
        # 상위 k개 선택
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((documents[idx], float(similarities[idx])))
        
        return results
    
    def semantic_search(self, query: str, knowledge_base: Dict[str, str]) -> List[Tuple[str, str, float]]:
        """의미 기반 검색"""
        # 모든 텍스트 추출
        titles = list(knowledge_base.keys())
        contents = list(knowledge_base.values())
        
        # 제목과 내용을 결합한 텍스트로 임베딩
        combined_texts = [f"{title}: {content}" for title, content in zip(titles, contents)]
        
        # 유사한 문서 찾기
        similar_docs = self.find_most_similar(query, combined_texts, top_k=3)
        
        results = []
        for doc, score in similar_docs:
            # 원본 제목 찾기
            for title, content in knowledge_base.items():
                if f"{title}: {content}" == doc:
                    results.append((title, content, score))
                    break
        
        return results
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> Dict[int, List[str]]:
        """텍스트 클러스터링"""
        from sklearn.cluster import KMeans
        
        # 임베딩 생성
        embeddings = self.get_embeddings_batch(texts)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # 클러스터별로 그룹화
        clusters = {}
        for text, label in zip(texts, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text)
        
        return clusters
    
    def detect_duplicates(self, texts: List[str], threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """중복 또는 매우 유사한 텍스트 찾기"""
        duplicates = []
        embeddings = self.get_embeddings_batch(texts)
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity >= threshold:
                    duplicates.append((texts[i], texts[j], float(similarity)))
        
        return duplicates
    
    def zero_shot_classification(self, text: str, labels: List[str]) -> Dict[str, float]:
        """Zero-shot 분류 (라벨 없이 분류)"""
        text_emb = self.get_embedding(text)
        label_embs = self.get_embeddings_batch(labels)
        
        # 각 라벨과의 유사도 계산
        similarities = cosine_similarity([text_emb], label_embs)[0]
        
        # 소프트맥스로 확률 변환
        exp_scores = np.exp(similarities)
        probabilities = exp_scores / exp_scores.sum()
        
        results = {}
        for label, prob in zip(labels, probabilities):
            results[label] = float(prob)
        
        return results

class AdvancedEmbeddings:
    """고급 임베딩 기법"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def weighted_embedding(self, texts: List[str], weights: List[float]) -> np.ndarray:
        """가중치가 적용된 임베딩 평균"""
        embeddings = self.model.encode(texts)
        weights = np.array(weights).reshape(-1, 1)
        weighted_emb = np.sum(embeddings * weights, axis=0) / np.sum(weights)
        return weighted_emb
    
    def hierarchical_embedding(self, paragraphs: List[str]) -> Dict[str, Any]:
        """계층적 임베딩 (문단 -> 문장 -> 단어)"""
        result = {
            "document_embedding": None,
            "paragraph_embeddings": [],
            "sentence_embeddings": []
        }
        
        paragraph_embs = []
        all_sentences = []
        
        for para in paragraphs:
            # 문장 분리 (간단한 방법)
            sentences = para.split('. ')
            all_sentences.extend(sentences)
            
            # 문단 임베딩
            para_emb = self.model.encode(para)
            paragraph_embs.append(para_emb)
        
        # 전체 문서 임베딩 (문단 임베딩의 평균)
        result["document_embedding"] = np.mean(paragraph_embs, axis=0)
        result["paragraph_embeddings"] = paragraph_embs
        result["sentence_embeddings"] = self.model.encode(all_sentences)
        
        return result
    
    def sliding_window_embedding(self, text: str, window_size: int = 100, stride: int = 50) -> List[np.ndarray]:
        """슬라이딩 윈도우 임베딩 (긴 텍스트 처리)"""
        words = text.split()
        embeddings = []
        
        for i in range(0, len(words) - window_size + 1, stride):
            window_text = ' '.join(words[i:i + window_size])
            embedding = self.model.encode(window_text)
            embeddings.append(embedding)
        
        return embeddings

def demo_basic_embeddings():
    """기본 임베딩 데모"""
    
    print("🔤 임베딩 기초 데모")
    print("=" * 60)
    
    emb = EmbeddingBasics()
    
    # 1. 텍스트 유사도
    print("\n1. 텍스트 유사도 계산:")
    pairs = [
        ("cat", "dog"),
        ("cat", "kitten"),
        ("car", "automobile"),
        ("car", "banana")
    ]
    
    for text1, text2 in pairs:
        similarity = emb.calculate_similarity(text1, text2)
        print(f"  '{text1}' vs '{text2}': {similarity:.3f}")
    
    # 2. 의미 검색
    print("\n2. 의미 기반 검색:")
    documents = [
        "Python is a programming language",
        "Machine learning uses algorithms to learn from data",
        "Dogs are loyal pets",
        "Neural networks are inspired by the brain",
        "Coffee is a popular morning beverage"
    ]
    
    query = "artificial intelligence and deep learning"
    results = emb.find_most_similar(query, documents, top_k=3)
    
    print(f"  Query: '{query}'")
    print("  Top matches:")
    for doc, score in results:
        print(f"    - {doc[:50]}... (score: {score:.3f})")
    
    # 3. Zero-shot 분류
    print("\n3. Zero-shot Classification:")
    text = "The new smartphone has an amazing camera and long battery life"
    categories = ["Technology", "Sports", "Food", "Politics"]
    
    classification = emb.zero_shot_classification(text, categories)
    print(f"  Text: '{text[:50]}...'")
    print("  Categories:")
    for category, prob in sorted(classification.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {category}: {prob:.2%}")
    
    # 4. 중복 감지
    print("\n4. 중복 텍스트 감지:")
    texts = [
        "The weather is nice today",
        "Today the weather is nice",
        "It's a beautiful day",
        "Python is great for data science",
        "Data science is great with Python"
    ]
    
    duplicates = emb.detect_duplicates(texts, threshold=0.8)
    if duplicates:
        print("  유사한 텍스트 쌍:")
        for text1, text2, score in duplicates:
            print(f"    - '{text1[:30]}...' ↔ '{text2[:30]}...' (유사도: {score:.3f})")
    else:
        print("  중복 없음")

def demo_semantic_search():
    """의미 검색 데모"""
    
    print("\n🔍 의미 기반 검색 데모")
    print("=" * 60)
    
    emb = EmbeddingBasics()
    
    # 지식 베이스
    knowledge_base = {
        "Python 기초": "Python은 배우기 쉬운 프로그래밍 언어로, 간결한 문법과 강력한 기능을 제공합니다.",
        "머신러닝": "머신러닝은 데이터에서 패턴을 학습하여 예측이나 결정을 내리는 AI 기술입니다.",
        "웹 개발": "웹 개발은 HTML, CSS, JavaScript를 사용하여 웹사이트와 웹 애플리케이션을 만드는 과정입니다.",
        "데이터베이스": "데이터베이스는 구조화된 데이터를 저장하고 관리하는 시스템으로, SQL로 데이터를 조회합니다.",
        "클라우드 컴퓨팅": "클라우드 컴퓨팅은 인터넷을 통해 컴퓨팅 리소스를 제공하는 서비스입니다."
    }
    
    queries = [
        "프로그래밍 언어 배우기",
        "AI와 딥러닝",
        "데이터 저장 방법"
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = emb.semantic_search(query, knowledge_base)
        print("  Results:")
        for title, content, score in results:
            print(f"    [{score:.3f}] {title}: {content[:50]}...")

def demo_clustering():
    """클러스터링 데모"""
    
    print("\n📊 텍스트 클러스터링 데모")
    print("=" * 60)
    
    emb = EmbeddingBasics()
    
    texts = [
        "Python programming tutorial",
        "Machine learning with Python",
        "How to cook pasta",
        "Italian cuisine recipes",
        "Deep learning fundamentals",
        "Neural network architecture",
        "Best pasta restaurants",
        "Data science with R",
        "Traditional Italian dishes"
    ]
    
    clusters = emb.cluster_texts(texts, n_clusters=3)
    
    print("Clusters:")
    for cluster_id, cluster_texts in clusters.items():
        print(f"\n  Cluster {cluster_id + 1}:")
        for text in cluster_texts:
            print(f"    - {text}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "search":
            demo_semantic_search()
        elif mode == "cluster":
            demo_clustering()
        else:
            print("Usage: python embedding_basics.py [search|cluster]")
    else:
        demo_basic_embeddings()
        
        print("\n" + "=" * 60)
        print("💡 더 많은 데모:")
        print("  python embedding_basics.py search   # 의미 검색")
        print("  python embedding_basics.py cluster  # 클러스터링")