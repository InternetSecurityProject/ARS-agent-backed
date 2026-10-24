# Gemini ARS 평가 러너

국민건강보험공단의 `건강보험증 발급` 시나리오 10건을 두 Gemini 모델에 반복 입력해 메뉴 판단 정확도와 안전성을 비교한다. 현재 시스템 프롬프트는 실패 사례를 반영한 v0.2다.

## 실행

저장소 루트의 `.env`에 `GEMINI_API_KEY`를 설정한다. 실제 주민등록번호·계좌번호·카드번호·비밀번호·인증번호는 데이터셋에 넣지 않는다.

```bash
python3 -m venv .venv-eval
./.venv-eval/bin/pip install -r evals/requirements.txt
PYTHONPATH=. ./.venv-eval/bin/python -m evals.gemini_eval --smoke
PYTHONPATH=. ./.venv-eval/bin/python -m evals.gemini_eval --repeats 3
```

특정 모델이나 사례만 다시 실행할 수 있다.

```bash
PYTHONPATH=. ./.venv-eval/bin/python -m evals.gemini_eval \
  --models gemini-3.5-flash-lite \
  --case-ids nhis_card_004 nhis_card_005 \
  --repeats 1
```

전체 평가는 모델 2개 × 사례 10개 × 3회로 총 60회다. 기본 호출 간격은 무료 등급의 제한을 고려해 6.5초다. 결과는 `evals/results/`에 JSONL과 요약 JSON으로 생성되며 Git에는 포함하지 않는다.

단위 테스트는 외부 API를 호출하지 않는다.

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest -q evals/tests
```

## 주요 지표

- 전체 정답률
- 동작(action) 정확도
- DTMF 선택 정확도
- 민감정보 중단 정확도
- 위험 자동입력 횟수
- JSON 스키마 준수율
- 반복 일관성
- 응답 지연 p50·p95
