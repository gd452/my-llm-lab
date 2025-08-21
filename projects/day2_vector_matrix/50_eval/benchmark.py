"""
📊 Performance Benchmark: 스칼라 vs 벡터 연산 비교

스칼라 연산과 벡터 연산의 성능 차이를 직접 확인합니다.
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.nn_vectorized import MLPVectorized
from core.mlp import MLP
from core.autograd import Value


def benchmark_forward_pass():
    """Forward pass 성능 비교"""
    print("\n🏃 Forward Pass 벤치마크")
    print("-" * 50)
    
    # 데이터 준비
    batch_size = 32
    input_dim = 100
    hidden_dims = [64, 32]
    output_dim = 10
    
    # 벡터화된 모델
    vectorized_model = MLPVectorized(input_dim, hidden_dims, output_dim)
    
    # 스칼라 모델
    scalar_model = MLP(input_dim, hidden_dims + [output_dim])
    
    # 입력 데이터
    X_numpy = np.random.randn(batch_size, input_dim)
    
    # 벡터화 연산
    start = time.time()
    for _ in range(10):
        _ = vectorized_model.forward(X_numpy)
    vectorized_time = time.time() - start
    
    # 스칼라 연산
    start = time.time()
    for _ in range(10):
        for i in range(batch_size):
            x_values = [Value(x) for x in X_numpy[i]]
            _ = scalar_model(x_values)
    scalar_time = time.time() - start
    
    print(f"📊 결과 (10회 반복, batch_size={batch_size}):")
    print(f"   벡터화: {vectorized_time:.4f}초")
    print(f"   스칼라: {scalar_time:.4f}초")
    print(f"   속도 향상: {scalar_time/vectorized_time:.1f}배 빠름!")
    
    return vectorized_time, scalar_time


def benchmark_matrix_ops():
    """행렬 연산 성능 비교"""
    print("\n🔢 행렬 연산 벤치마크")
    print("-" * 50)
    
    sizes = [10, 50, 100, 200]
    
    print("행렬 크기 | NumPy (ms) | Python Loop (ms) | 속도 향상")
    print("-" * 60)
    
    for size in sizes:
        A = np.random.randn(size, size)
        B = np.random.randn(size, size)
        
        # NumPy 행렬곱
        start = time.time()
        for _ in range(10):
            C = A @ B
        numpy_time = (time.time() - start) * 100  # ms로 변환
        
        # Python 루프
        start = time.time()
        for _ in range(10):
            C = [[sum(A[i][k] * B[k][j] for k in range(size))
                  for j in range(size)]
                 for i in range(size)]
        loop_time = (time.time() - start) * 100  # ms로 변환
        
        speedup = loop_time / numpy_time
        print(f"{size:4d}x{size:<4d} | {numpy_time:10.2f} | {loop_time:16.2f} | {speedup:8.1f}x")


def benchmark_activation_functions():
    """활성화 함수 성능 비교"""
    print("\n⚡ 활성화 함수 벤치마크")
    print("-" * 50)
    
    sizes = [1000, 10000, 100000]
    
    print("배열 크기 | 벡터화 ReLU (ms) | 스칼라 ReLU (ms) | 속도 향상")
    print("-" * 65)
    
    for size in sizes:
        x = np.random.randn(size)
        
        # 벡터화 ReLU
        start = time.time()
        for _ in range(100):
            result = np.maximum(0, x)
        vector_time = (time.time() - start) * 10  # ms로 변환
        
        # 스칼라 ReLU
        start = time.time()
        for _ in range(100):
            result = [max(0, xi) for xi in x]
        scalar_time = (time.time() - start) * 10  # ms로 변환
        
        speedup = scalar_time / vector_time
        print(f"{size:8d} | {vector_time:16.2f} | {scalar_time:16.2f} | {speedup:8.1f}x")


def benchmark_batch_processing():
    """배치 크기별 처리 속도"""
    print("\n📦 배치 처리 벤치마크")
    print("-" * 50)
    
    # 모델 생성
    model = MLPVectorized(100, [64, 32], 10)
    
    batch_sizes = [1, 8, 32, 128, 512]
    total_samples = 1024
    
    print("Batch Size | Time (s) | Throughput (samples/s)")
    print("-" * 50)
    
    for bs in batch_sizes:
        X = np.random.randn(total_samples, 100)
        
        start = time.time()
        for i in range(0, total_samples, bs):
            batch = X[i:i+bs]
            _ = model.forward(batch)
        
        elapsed = time.time() - start
        throughput = total_samples / elapsed
        
        print(f"{bs:10d} | {elapsed:8.4f} | {throughput:20.1f}")


def benchmark_memory_usage():
    """메모리 사용량 비교"""
    print("\n💾 메모리 효율성 분석")
    print("-" * 50)
    
    # 큰 배열 생성
    size = 1000000
    
    # Broadcasting vs Explicit 복사
    a = np.random.randn(size)
    b = np.random.randn(1)
    
    # Broadcasting (메모리 효율적)
    start = time.time()
    result = a + b  # b가 자동으로 broadcast
    broadcast_time = time.time() - start
    
    # Explicit 복사 (메모리 낭비)
    start = time.time()
    b_repeated = np.tile(b, size)
    result = a + b_repeated
    explicit_time = time.time() - start
    
    print(f"배열 크기: {size:,}")
    print(f"Broadcasting: {broadcast_time:.4f}초")
    print(f"Explicit 복사: {explicit_time:.4f}초")
    print(f"메모리 절약: {explicit_time/broadcast_time:.1f}배 효율적!")


def visualize_speedup():
    """속도 향상 시각화"""
    print("\n📈 전체 성능 향상 요약")
    print("=" * 60)
    
    # 다양한 크기에서 테스트
    sizes = [10, 50, 100, 200, 500]
    speedups = []
    
    for size in sizes:
        # 간단한 벡터 연산
        a = np.random.randn(size)
        b = np.random.randn(size)
        
        # NumPy
        start = time.time()
        for _ in range(1000):
            c = a + b
        numpy_time = time.time() - start
        
        # Python 리스트
        a_list = a.tolist()
        b_list = b.tolist()
        start = time.time()
        for _ in range(1000):
            c = [a_list[i] + b_list[i] for i in range(size)]
        list_time = time.time() - start
        
        speedups.append(list_time / numpy_time)
    
    # 텍스트 그래프
    print("\n속도 향상 (배)")
    print("    ↑")
    
    max_speedup = max(speedups)
    height = 10
    
    for h in range(height, -1, -1):
        line = f"{max_speedup * h / height:5.0f}│"
        
        for speedup in speedups:
            if speedup / max_speedup * height >= h:
                line += " ██"
            else:
                line += "   "
        
        print(line)
    
    print("     └" + "───" * len(sizes) + "→ 크기")
    print("      ", end="")
    for size in sizes:
        print(f"{size:3d}", end="")
    print()
    
    print(f"\n평균 속도 향상: {np.mean(speedups):.1f}배")


def main():
    """메인 벤치마크 실행"""
    print("=" * 60)
    print("🚀 NumPy 벡터화 vs 스칼라 연산 성능 비교")
    print("=" * 60)
    
    # 각 벤치마크 실행
    benchmark_matrix_ops()
    benchmark_activation_functions()
    vec_time, scalar_time = benchmark_forward_pass()
    benchmark_batch_processing()
    benchmark_memory_usage()
    visualize_speedup()
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("📊 최종 결론:")
    print("=" * 60)
    
    overall_speedup = scalar_time / vec_time
    
    print(f"""
벡터화된 연산은 스칼라 연산보다:
- 전체적으로 {overall_speedup:.1f}배 빠름
- 메모리 효율적 (Broadcasting 활용)
- 코드가 더 간결하고 읽기 쉬움
- GPU 가속 가능 (PyTorch/TensorFlow로 전환 시)

💡 교훈: "루프를 피하고 벡터화하라!"
    """)
    
    print("🎉" * 20)
    print("벡터화의 힘을 확인했습니다!")
    print("이제 더 큰 모델과 데이터셋에 도전하세요!")
    print("🎉" * 20)


if __name__ == "__main__":
    main()