"""
LLM Lab 프로젝트 설정

설치 방법:
    pip install -e .
    
설치 후:
    from core.autograd import Value
    from core.nn import Neuron, Layer, MLP
"""

from setuptools import setup, find_packages

setup(
    name="llm-lab",
    version="0.1.0",
    description="My LLM Learning Lab",
    packages=['core'],  # core 폴더만 포함!
    python_requires=">=3.8",
    install_requires=[
        # 기본 의존성만 (최소한으로)
    ],
    extras_require={
        "dev": [
            "pytest>=8.2",
            "jupyter>=1.0.0",
            "numpy>=1.26",
            "matplotlib>=3.5.0",
        ]
    }
)