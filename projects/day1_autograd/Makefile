setup: ; pip install -U pip && pip install -r requirements.txt
test:  ; pytest -q
smoke: ; python 50_eval/smoke.py

# 선택적 체크 (필요시 사용)
lint:  ; ruff check .  # 기본 오류만 체크
fmt:   ; black --check .  # 코드 포맷 체크
strict: ; ruff check . && black --check . && mypy .  # 엄격한 체크