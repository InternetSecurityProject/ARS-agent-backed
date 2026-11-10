# Web Large Text Implementation Plan

> Execute inline with executing-plans; user approved implementation and local verification.

**Goal:** 웹 전체에서 큰글씨 보기를 전환하고 로컬에서 안전하게 확인한다.
**Architecture:** 기존 Preact Root에 공통 preference hook과 toggle component를 둔다. CSS 크기는 의미별 custom property로 관리한다. 기존 Python 웹 빌드를 유지하고 npm 의존성/검증 명령만 추가한다.
**Tech Stack:** Preact + htm, CSS custom properties, Node test runner + Playwright, existing Python builder.
**Spec:** `docs/superpowers/specs/2026-11-10-web-large-text.md`

## Global Constraints
- 서버 코드·API·음성·전화 로직 변경 및 실제 발신 없음.
- 생성물 직접 편집 없음, 공유 브랜치 직접 push/merge 없음.
- 별도 worktree에서 작업, 사용 중인 8011 서버·에뮬레이터 재시작 없음.

## Review Focus
- 좁은 화면/긴 선택지: 잘림 없이 스크롤, 종료 버튼 접근 가능.
- 확대 중 입력: 값·포커스·현재 화면 유지.
- 저장소 예외: 설정 저장이 불가능해도 전환 가능.
- 대화상자: 필수 확인 버튼이 viewport 안에 유지.
- 설정 전환/데모: 서버 세션 생성 요청 없음.

## Task 1 — 재현 가능한 웹 검증
- [x] 웹 dev dependency lock, 브라우저 행동 테스트를 추가한다.
- [x] 기존 빌드 화면에서 toggle 부재로 테스트가 실패하는지 확인한다.
- [x] 안전한 정적 preview server를 제공한다. API는 차단하며 proxy하지 않는다.

## Task 2 — 큰글씨 기능
- [x] `web/app.src.html`: 공통 preference hook/component 및 typography tokens 추가.
- [x] 기존 build.py는 npm 설치 경로도 지원한다. 기존 vendor 방식은 유지한다.
- [x] 공식 빌드로 `static/app.html`을 생성한다.
- [x] 모바일/데스크톱/저장/오류/입력/데모 브라우저 테스트 통과.

## Task 3 — 확인 및 전달
- [x] 수정 전·후 기존 pytest 결과를 비교하고 기존 실패는 별도로 기록한다.
- [x] 기본/큰글씨 screenshot 비교 및 diff/backend 미수정 확인.
- [x] 로컬 정적 preview 열기, 실행 방법과 제약 기록.
- [x] 최신 원격 브랜치 변동을 확인한다. 이번 전달은 로컬 확인 우선이다.


## Verification record
- 기준 원격 HEAD 재확인: 0871ac60 (작업 시작과 동일).
- 웹 전체 행동 테스트: 11 passed. 캡처 경로 처리 수정 후 390px 시나리오 재실행.
- 저장소 사용 불가·키보드 전환·설정 복원·입력 DOM 유지·연락처 제목·긴 목적 확인창·안전 영역·320/390/1440px 후보/보안입력 포함.
- 기존 Python suite: 변경 전후 모두 427 passed, 1 skipped, 1 failed. 동일한 기존 실패는 `test_flows.py::test_cut_paragraph_is_not_joined_across_a_key_press`; 서버 구현은 변경하지 않음.
- 별도 코드 검토: 상단 safe-area 누락을 찾아 수정하고 재현 테스트 RED→GREEN. 다른 Critical/Important 미해결 없음.
- 원본 재빌드 결과와 static/app.html의 byte equality 확인. git diff --check 통과.
- Ruling: 승인한 “이미 큰 전화번호 유지”는 통화 상단/발신 확인/다이얼러 숫자에 적용. 연락처 목록의 작은 번호 설명은 가독성을 위해 본문과 함께 확대.
- Ruling: 기존 웹에 별도 화면 갤러리는 없어 공통 데모 화면으로 검증. Android 미리보기와 서버/Android 코드는 수정하지 않음.
- Ruling: 현재 서버에만 있는 웹 변경은 추측하여 가져오지 않음. 기준 브랜치 원본에서 작은 수정으로 유지하며 실제 배포 전 담당자 대조 필요.
