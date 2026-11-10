# Response cues implementation plan

> **For agentic workers:** Use superpowers:executing-plans; local implementation approved, no push/merge/deploy.

**Goal:** 기다림과 사용자 응답을 시각적으로 구분한다.
**Architecture:** 기존 Preact/htm CallScreen에서 응답 가능 여부를 읽기 전용으로 도출한다. 기존 ProgressCard와 질문·입력 콘텐츠에 전달하며 상태·전송·음성 수명은 바꾸지 않는다.
**Tech Stack:** Preact, htm, CSS, Playwright.
**Spec:** docs/superpowers/specs/2026-11-10-response-cues-design.md

## Constraints / review focus
- 백엔드·Android·API·음성 코드는 수정하지 않는다. 실제 발신 없음.
- 새 request_id와 같은 요청의 기한 갱신 구분, 선택 전달·만료·연결 불가 시 강조 해제, 개인정보 입력 버퍼 유지, 기본/큰글씨 작은 화면, 모션 감소를 검증한다.

## Task: shared presentation and local verification
- [x] 테스트: 서버 데이터를 사용하는 실제 CallScreen을 테스트 전용 fixture에서 렌더링하여 질문 제목·중복 없음·강조 진입/해제·기한 갱신·입력 보존을 확인한다. 외부 API와 실제 발신 차단.
- [x] 새 동작 테스트의 RED를 확인한다.
- [x] web/app.src.html: 공통 표현 상태·짧은 상태 표시·응답 카드 테두리와 단발 모션, 서버 질문 제목과 데모 적용. 개인정보 키패드는 카드 밖 유지.
- [x] 기존 빌드로 static/app.html 재생성하고 새 테스트 및 npm test 전체 실행.
- [x] 기본·큰글씨 화면을 캡처하고 별도 검토 후 로컬 탭에 데모를 열어 전달. 변경은 커밋하지 않고 유지.

## Ledger
- 승인된 설계를 그대로 로컬 구현한다. 재승인 질문은 생략한다.
- 배포판은 큰글씨 변경 이전 HTML로 확인됨. 이번 작업은 머지된 c573932 원본 기준이며 배포 HTML을 덮어쓰지 않는다.
- RED: 새 기다림 표시가 없어 실패하는 것을 확인했다. 입력 보존 테스트에서 같은 질문의 기한 연장도 일반 답변을 지우는 기존 동작을 발견했다. 질문의 expiresAt만 바뀌면 유지하고, 실제 질문 변경은 paint 전에 초기화하도록 프론트 상태만 보완했다.
- 검증: 실제 폰트로 npm test 33개 통과, 원본 재빌드와 생성물 일치, diff --check 통과. 별도 소스 검토에서 차단 사항 없음. 마이크·실제 서버·실제 발신 테스트는 하지 않았다.
- 검토 한계: 텍스트 응답 전송 후 강조 해제는 기존 owner의 응답 영수증·busy 상태 갱신 경로를 소스에서 확인했고, UI fixture는 응답 불가 상태의 강조 해제를 검증한다. 실제 통화는 하지 않았다.
- 사용자 후속 요청: 중복되는 ‘시간 내 선택해 주세요’ 기본 안내는 제거하고 만료·전달·중요한 선택 안내는 유지한다. 기관명도 복원한다. 기본 글씨는 번호 아래 기관명·시간, 큰글씨는 번호·시간 아래 기관명 별도 줄로 배치한다. 여전히 로컬만 변경하며 푸시·머지하지 않는다.
