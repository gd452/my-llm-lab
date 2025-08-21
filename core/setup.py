"""
Core 패키지 설정

설치:
    cd core
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="core",
    version="0.1.0",
    description="LLM Lab Core Library",
    packages=find_packages(),
    python_requires=">=3.8",
)