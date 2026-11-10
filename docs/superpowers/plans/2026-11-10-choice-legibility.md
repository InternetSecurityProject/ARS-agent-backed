# Candidate button legibility implementation plan

> **For agentic workers:** Use superpowers:executing-plans. User approved local implementation; no push, merge or deployment before visual confirmation.

**Goal:** 후보 선택 글씨를 기본 16px·큰글씨 20px로 키운다.
**Architecture:** 기존 서버·데모 후보 목록이 같은 CSS를 사용한다. 후보 목록 간격만 별도 컨테이너로 묶고 기존 요청·전송·기한·음성 로직은 유지한다.
**Tech Stack:** Existing Preact/htm, CSS, Playwright.
**Spec:** 사용자가 승인한 후보 선택 가독성 계획. 18px 제안을 16px로 조정했으며 큰글씨는 1.25배다.

## Constraints and review focus
- 버튼 최소 높이 56px / 70px, 글씨 16px / 20px, 줄 높이 24px / 30px. 긴 이름은 원문 그대로 자동 줄바꿈하며 높이를 늘린다.
- 번호는 오른쪽 연한 파란 칸, *·# 선택지는 원문 유지. 누름·포커스·비활성 스타일을 구분한다.
- 질문·다른 글씨·배경·기관명·하단 원형 버튼의 크기와 위치를 유지한다.
- 작은 화면에서는 카드 내부 스크롤로 모든 후보에 접근한다. 커진 후보 세 개의 동시 노출은 보장하지 않는다.
- 말하기·글자 입력은 후보 아래 유지. API/백엔드/Android/음성 로직은 변경하지 않는다. 실제 발신 없음.

## Task: shared candidate styling
- [x] 서버 데이터 fixture에서 후보 글씨/높이, 긴 이름, *·#, 큰글씨와 하단 위치를 검증하는 테스트를 추가하고 RED 확인.
- [x] web/app.src.html의 후보 전용 CSS와 서버·데모 목록에만 적용하고 생성 HTML 재빌드.
- [x] 기존 후보 세 개 동시 노출 검증을 승인된 스크롤 접근 검증으로 변경. 회귀 테스트 실행, 로컬 화면 캡처.
- [x] 원본/생성물 일치와 diff 확인 후 커밋·푸시·머지 없이 로컬 결과 전달.

## Verification ledger
- RED: 기존 13px 후보가 새 16px 읽기 크기 검증에서 실패함을 확인.
- 글씨·긴 메뉴·특수 키 테스트 통과 후 전체 테스트에서 688px 높이 두 규격의 스크롤 하단 여백 검증 실패. 좌표 확인 결과 마지막 후보의 아래쪽이 카드 경계 0.5px 안쪽에 붙음. scroll-margin-block 12px로 포커스·스크롤 도착 여백을 보완.
- 최종 실제 폰트 npm test 35개 모두 통과. 생성물 재빌드와 byte 일치, diff --check 통과. 별도 소스 검토에서 차단할 문제 없음.
- API·서버·음성·실제 발신 검증은 하지 않았고 관련 코드는 변경하지 않음. 로컬 변경은 커밋하지 않았으며 푸시·머지·배포 없음.

- 2026-11-11 통합 승인: 사용자가 최종 로컬 화면 확인 후 feat/model-select-concurrent-calls로 머지·푸시 요청. 이전 로컬 전용 제한은 구현 단계의 기록이며 이번 통합 승인이 우선한다. main·실제 서버 배포는 제외.
