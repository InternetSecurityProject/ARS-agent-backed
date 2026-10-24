# KB Balance Agent Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 국민은행 계좌 잔액조회의 목적 확정, ARS 경로 탐색, 오탐 방지를 전체 세션으로 자동 검증한다.

**Architecture:** 기존 `agent_e2e` 러너와 메뉴 채점기를 재사용하고, 다중 업무 목적 프롬프트와 국민은행 전용 대화·ARS 흐름을 추가한다. 데이터셋의 `prompt_version`으로 목적 프롬프트를 선택하여 기존 건강보험증 검증을 보존한다.

**Tech Stack:** Python 3.11, pytest, Gemini GenerateContent API

**Spec:** `docs/superpowers/specs/2026-10-24-kb-balance-agent-evaluation-design.md`

## Global Constraints

- 실제 주민등록번호·계좌번호·비밀번호·잔액을 데이터셋에 넣지 않는다.
- 민감정보 입력 단계에서 DTMF를 자동 전송하지 않는다.
- 기존 건강보험증 데이터셋과 실행 방식을 깨뜨리지 않는다.
- 실제 전화·STT·DTMF 전송은 이번 검증에서 제외한다.

## Review Focus

- “통장 확인”을 확인 없이 잔액조회로 시작하지 않는지
- 거래내역·송금·카드 잔액을 계좌 잔액으로 오인하지 않는지
- 잔액조회 번호가 누락되면 `3`을 추측하지 않는지
- 잘못된 카드 메뉴에서 `#`로 복귀하는지
- 세 개의 민감정보 입력 단계에서 모두 멈추는지

---

### Task 1: 다중 목적 프롬프트 선택

**Files:**
- Create: `evals/prompts/goal_agent_v0_2.txt`
- Modify: `evals/agent_evaluator.py`
- Modify: `evals/agent_e2e.py`
- Test: `evals/tests/test_agent_evaluator.py`
- Test: `evals/tests/test_agent_e2e.py`

**Interfaces:**
- Consumes: 데이터셋의 `prompt_version`
- Produces: `KB_ACCOUNT_BALANCE_INQUIRY` 의도 검증과 프롬프트 자동 선택

- [ ] 국민은행 의도와 알 수 없는 프롬프트 버전을 검증하는 실패 테스트를 작성한다.
- [ ] 테스트가 기대한 이유로 실패하는지 확인한다.
- [ ] 다중 의도 계약과 프롬프트 선택을 최소 구현한다.
- [ ] 전체 평가 단위 테스트를 통과시킨다.
- [ ] 변경을 커밋한다.

### Task 2: 국민은행 대화·ARS 데이터셋

**Files:**
- Create: `evals/data/kb_balance_agent_scenarios_v0_1.json`
- Create: `evals/tests/test_kb_agent_dataset.py`

**Interfaces:**
- Consumes: Task 1의 `KB_ACCOUNT_BALANCE_INQUIRY`와 `goal-agent-v0.2`
- Produces: 20개 대화, 5개 ARS 흐름, 오탐·복귀·민감정보 체크포인트

- [ ] 시나리오 수, 흐름 참조, 민감정보 3단계, 오탐 금지, 복귀를 검증하는 실패 테스트를 작성한다.
- [ ] 데이터셋이 없어 테스트가 실패하는지 확인한다.
- [ ] 실제 전사문을 익명화한 20개 대화와 5개 흐름을 작성한다.
- [ ] 데이터셋과 전체 평가 테스트를 통과시킨다.
- [ ] 변경을 커밋한다.

### Task 3: 실제 모델 실행과 보고서

**Files:**
- Create: `docs/evals/2026-10-24-kb-balance-agent-evaluation.md`
- Modify: `evals/README.md`

**Interfaces:**
- Consumes: Task 2 데이터셋과 `evals.agent_e2e`
- Produces: 모델별 세션 성공률, 메뉴 정확도, 오탐, 민감정보 인계, 복귀 결과

- [ ] Flash-Lite와 Flash에 동일한 데이터셋을 실행한다.
- [ ] 정답 없음, 번호 누락, 잘못된 경로 복귀, 민감정보 인계를 별도 집계한다.
- [ ] 실행 결과와 모델 선정 판단을 보고서에 기록한다.
- [ ] 전체 테스트와 Git diff를 검증한다.
- [ ] 변경을 커밋하고 원격 기능 브랜치에 푸시한다.
