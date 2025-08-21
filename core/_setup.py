"""
Path 설정 헬퍼

모든 core 모듈과 프로젝트에서 사용하는 공통 path 설정
"""
import sys
from pathlib import Path

def setup_path():
    """my-llm-lab 루트를 sys.path에 추가
    
    Returns:
        Path: my-llm-lab 루트 경로
    """
    current = Path(__file__).resolve()
    
    # my-llm-lab 폴더를 찾을 때까지 상위로 이동
    while current.name != 'my-llm-lab' and current.parent != current:
        current = current.parent
    
    if current.name == 'my-llm-lab':
        if str(current) not in sys.path:
            sys.path.insert(0, str(current))
        return current
    
    raise RuntimeError("my-llm-lab 폴더를 찾을 수 없습니다")

# 모듈 import 시 자동 실행
setup_path()