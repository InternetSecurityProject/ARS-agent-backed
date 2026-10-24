# ARS Agent End-to-End Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모호한 사용자 요청부터 건강보험증 발급 완료까지 Gemini의 전체 세션 성공률을 자동 측정한다.

**Architecture:** 사용자 목적 대화와 전화 안내 탐색을 실제 순서로 반복 호출하는 평가 시뮬레이터를 `evals/`에 추가한다. 목적 응답과 메뉴 응답은 각각 구조화 JSON으로 검증하고, 시나리오 전체 체크포인트가 맞아야 성공으로 집계한다.

**Tech Stack:** Python 3.11, httpx, pytest, Gemini GenerateContent API

**Spec:** `docs/superpowers/specs/2026-10-24-agent-e2e-evaluation-design.md`

## Global Constraints

- 실제 주민등록번호·전화번호·주소 등 개인정보를 데이터셋이나 결과에 넣지 않는다.
- 민감정보 입력 단계에서 자동 DTMF를 허용하지 않는다.
- 전화 연결·STT·실제 DTMF는 이 평가 범위에서 제외한다.
- 결과 원본은 `evals/results/`에 저장하고 Git에서 제외한다.

## Review Focus

- 모호한 요청을 확정 없이 바로 실행하지 않는지
- 이전 턴의 확인 질문에 대한 짧은 답변을 문맥으로 이해하는지
- 번호가 잘린 안내에서 번호를 추측하지 않는지
- 민감정보 입력과 입력값 확인 단계를 구분하는지
- 취소·비대상 요청에서 통화를 시작하지 않는지

---

### Task 1: 전체 세션 데이터 계약과 채점기

**Files:**
- Create: `evals/agent_evaluator.py`
- Test: `evals/tests/test_agent_evaluator.py`

**Interfaces:**
- Produces: `validate_goal_response(payload)`, `score_goal_turn(expected, payload)`, `summarize_sessions(records)`

- [ ] 목적 응답 스키마, 턴 채점, 세션 성공 집계를 검증하는 실패 테스트를 작성한다.
- [ ] 해당 테스트가 누락된 구현 때문에 실패하는지 확인한다.
- [ ] 최소 채점 구현을 추가한다.
- [ ] 전체 단위 테스트를 통과시킨다.

### Task 2: 대화·전화 안내 시나리오 데이터

**Files:**
- Create: `evals/data/nhis_agent_scenarios_v0_1.json`
- Create: `evals/prompts/goal_agent_v0_1.txt`
- Test: `evals/tests/test_agent_dataset.py`

**Interfaces:**
- Produces: 사용자 대화 시나리오와 재사용 가능한 전화 안내 흐름

- [ ] 모든 시나리오가 유효한 흐름을 참조하고 개인정보를 포함하지 않는지 검사하는 테스트를 작성한다.
- [ ] 테스트 실패를 확인한다.
- [ ] 20개 이상의 대화 변형과 5개 전화 안내 흐름을 작성한다.
- [ ] 데이터 테스트를 통과시킨다.

### Task 3: Gemini 전체 세션 평가 러너

**Files:**
- Create: `evals/agent_e2e.py`
- Test: `evals/tests/test_agent_e2e.py`
- Modify: `evals/README.md`

**Interfaces:**
- Consumes: Task 1 채점기, Task 2 데이터, `evals.gemini_eval`의 메뉴 스키마와 프롬프트 빌더
- Produces: `run_scenario(...)`, 모델별 JSONL 및 요약 JSON

- [ ] 가짜 모델 응답으로 대화 이력 전달, 흐름 중단, 전체 성공 판정을 검증하는 실패 테스트를 작성한다.
- [ ] 테스트 실패를 확인한다.
- [ ] 반복 호출 시뮬레이터와 CLI를 구현한다.
- [ ] 전체 테스트를 통과시킨다.

### Task 4: 실제 모델 실행과 결과 확정

**Files:**
- Create: `docs/evals/2026-10-24-agent-e2e-evaluation.md`

**Interfaces:**
- Consumes: Task 3 평가 러너
- Produces: MVP 모델 결정과 백엔드 인수 조건

- [ ] Flash로 전체 시나리오를 1회 실행한다.
- [ ] Lite는 같은 데이터의 대표 시나리오로 비교 실행한다.
- [ ] 세션 성공률, 단계별 실패, 위험 행동, 지연, 호출 오류를 기록한다.
- [ ] 전체 테스트와 Git diff를 최종 검증한다.

