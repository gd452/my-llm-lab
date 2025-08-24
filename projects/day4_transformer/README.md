# Day 4: Transformer - Building Complete Architecture

## 학습 목표
- Transformer 아키텍처 완성하기
- Encoder-Decoder 구조 이해하기
- Layer Normalization과 Residual Connection 구현
- Complete Transformer block 만들기

## 프로젝트 구조
```
day4_transformer/
├── README.md
├── notebooks/
│   └── transformer_tutorial.ipynb  # Transformer 구현 튜토리얼
├── study_notes/
│   ├── 01_transformer_architecture.md  # Transformer 전체 구조
│   ├── 02_encoder_decoder.md          # Encoder-Decoder 설명
│   ├── 03_layer_norm_residual.md      # LayerNorm & Residual
│   └── 04_position_ffn.md             # Position-wise FFN
├── tests/
│   └── test_transformer.py            # 테스트 코드
└── 50_eval/
    ├── transformer_demo.py            # 간단한 번역 데모
    └── visualize_architecture.py      # 구조 시각화
```

## 핵심 개념

### 1. Transformer Block
- Multi-Head Attention (Day 3에서 학습)
- Layer Normalization
- Residual Connection
- Position-wise Feed-Forward Network

### 2. Encoder-Decoder Architecture
- Encoder Stack: 입력 시퀀스 처리
- Decoder Stack: 출력 시퀀스 생성
- Cross-Attention: Encoder-Decoder 연결

### 3. Key Components
```python
# Transformer Block 구조
x -> LayerNorm -> Multi-Head Attention -> Residual -> 
  -> LayerNorm -> Feed-Forward -> Residual -> output
```

## 실습 내용
1. Layer Normalization 구현
2. Position-wise FFN 구현
3. Transformer Block 조립
4. Encoder/Decoder 스택 구성
5. 전체 Transformer 모델 완성

## 학습 순서
1. `study_notes/01_transformer_architecture.md` 읽기
2. `notebooks/transformer_tutorial.ipynb` 실습
3. `core/transformer.py` 구현 완성
4. `tests/test_transformer.py` 실행
5. `50_eval/transformer_demo.py`로 실제 동작 확인

## 필수 구현 사항
- [ ] LayerNorm class
- [ ] PositionwiseFeedForward class
- [ ] TransformerBlock class
- [ ] TransformerEncoder class
- [ ] TransformerDecoder class
- [ ] Transformer class (전체 모델)

## 다음 단계 (Day 5)
- GPT 스타일 Decoder-only 모델
- 텍스트 생성 구현