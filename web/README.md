# 웹 화면 개발

기존 Preact + htm 구조를 유지한다. `app.src.html`이 원본이며 `static/app.html`을 직접 편집하지 않는다.

```sh
cd web
npm ci
npm run build -- ../../ARS-agent-frontend/app/src/main/assets/figma ../static/app.html
npm run preview
```

빌드 첫 인자는 기존 Android 리포의 SVG 아이콘 폴더다. 다른 위치에 체크아웃했다면 그 경로를 지정한다.
기존 `python web/build.py <아이콘 폴더> static/app.html` 명령과 `web/vendor` 방식도 유지한다.

## 로컬 확인

`http://127.0.0.1:8013/app`을 연다. 상단 오른쪽 **큰글씨 보기**를 누르면 안내·선택지·입력·버튼 설명이 25% 확대된다. 같은 버튼으로 기본 크기를 복원한다. 전화번호·키패드 숫자·아이콘 크기는 유지한다. 이 설정만 브라우저에 보관하며 저장소가 막혀 있어도 현재 화면의 전환은 가능하다.

상단 **실제 ARS → UI 데모** 또는 **후보 선택 / 민감정보 입력**을 선택한 다음 131을 입력하고 **ARS Agent 전화**로 데모에 들어간다. 목적을 입력하면 기존 데모 순서로 진행한다.

이 미리보기 서버는 정적 화면만 제공하고 모든 API를 403으로 차단한다. 실제 발신·마이크 테스트·서버 연동용이 아니다. 기존 8011 서버나 배포 서버를 재시작하지 않는다. UI 확인 뒤 서버 연동 검증은 별도 승인 범위에서 진행한다.

큰글씨 Agent 화면은 상단 모드와 설정을 한 줄로 묶고, 번호·기관·시간을 폭에 맞춰 함께 배치한다. 긴 번호는 줄바꿈한다. 상하·좌우 여백을 줄인 공간은 정보 카드에 사용하며 기본 글씨의 배치와 원형 버튼 크기는 유지한다.

## 검증

```sh
npx playwright install chromium
npm test
```

이미 설치된 Chromium을 쓰려면 `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`를 실행 파일 경로로 설정한다. 테스트는 임시 로컬 서버를 사용하며 외부 요청·서버 API를 차단한다.

실제 폰트의 줄바꿈까지 검증하려면 `WEB_TEST_NOTO_FONT`와 `WEB_TEST_GOWUN_FONT`를 각각 로컬 Noto Sans KR·Gowun Batang TTF 경로로 설정한다. 테스트 서버가 이 파일만 로컬에서 제공하므로 외부 폰트 서버나 실제 API에 연결하지 않는다. Android 리포의 `app/src/main/res/font/` 파일을 사용할 수 있다.

## 동시 작업

작업 기준은 `feat/model-select-concurrent-calls`이다. 스타일 변경은 의미별 CSS custom property, 동작은 `useTextSizePreference` / `TextSizeControl`에 모은다. 화면·세션 컴포넌트에 설정을 위한 key를 추가하거나 재생성하지 않는다.

공유 전 기준 브랜치를 fetch하고 원본에서 충돌을 해결한 뒤 생성물을 다시 빌드한다. 생성된 `static/app.html`만 덮어쓰거나 배포 서버 파일을 복사해 원본을 대체하지 않는다. 서버에 미푸시 변경이 있으면 배포 담당자와 원본부터 맞춘다.
