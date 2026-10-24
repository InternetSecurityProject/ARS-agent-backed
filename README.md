# ARS-agent-backed

ARS 자동화 API 서버

- 앱에서 전화번호·목적 전송, 서버가 ClawOps 070 번호로 기관 ARS 발신
- 기관 메뉴 정보 없이 통화 중 들은 안내만으로 메뉴 탐색
- 세션 생성·조작은 REST, 진행 상황(상태·자막·경로·재질문·결과) 수신은 WebSocket
- AI 슬롯필링(안내 분류·선택지 정규화·신뢰도·요약)은 팀원 담당, 현재 목 데이터(`mock_ai.json`)로 동작
- 상세 명세: [docs/ARS자동화-API명세서-v0.3.md](docs/ARS자동화-API명세서-v0.3.md)

## API 형태

| 구분 | 형식 |
| --- | --- |
| Base URL | `http://{host}/v1` (운영 시 `https`·`wss`) |
| 요청·응답 | REST, `application/json; charset=utf-8` |
| 이벤트 | WebSocket 텍스트 프레임(JSON), 서버 → 앱 단방향 |
| 오디오 | WebSocket 바이너리 프레임, G.711 μ-law 8kHz mono, 20ms(160바이트) 단위 |
| 인증 | 세션 생성은 `X-App-Key` 헤더, 그 외 REST 는 `Authorization: Bearer {session_token}`, WebSocket 은 `?token={session_token}` |
| 시간 | ISO 8601, KST |

## API 목록

| 메서드 | 경로 | 인증 | 용도 |
| --- | --- | --- | --- |
| `POST` | `/v1/sessions` | 앱 키 | 세션 생성 (생성 즉시 발신, 첫 메뉴부터 탐색) |
| `WS` | `/v1/sessions/{id}/events` | 세션 토큰 | 이벤트 수신 (연결 직후 `snapshot` 1회) |
| `POST` | `/v1/sessions/{id}/actions` | 세션 토큰 | 사용자 조작 (아래 `type` 표) |
| `GET` | `/v1/sessions/{id}/result` | 세션 토큰 | 결과 조회 (통화 중 `202`, 종료 후 `200`) |
| `WS` | `/v1/sessions/{id}/audio` | 세션 토큰 | ARS 소리 듣기·상담원 통화 오디오 (검증 필요) |
| `GET` | `/v1/health` | 없음 | 개발용 상태 확인 (시험 화면용) |
| `GET` | `/` | 없음 | 시험용 앱 화면 |

### 세션 생성

```http
POST /v1/sessions
X-App-Key: dev-app-key

{ "phone_number": "1588-9999", "goal": "통장 잔액 알고 싶어" }
```

```json
{
  "session_id": "ses_4b8e21",
  "session_token": "st_...",
  "events_url": "ws://{host}/v1/sessions/ses_4b8e21/events",
  "audio_url": "ws://{host}/v1/sessions/ses_4b8e21/audio",
  "phone_number": "15889999",
  "status": "dialing",
  "goal_summary": "계좌 잔액을 확인해 드릴게요."
}
```

- `goal` 필수 (누락 시 `400`)
- 목적 승인 단계 없음. 목적을 다르게 이해한 경우 첫 번호 입력 전까지 `revise_goal` 로 재입력
- 동시 통화 1건 (진행 중인 통화가 있으면 `409`)

### 사용자 조작 (`POST /actions` 의 `type`)

```json
{ "type": "answer", "request_id": "req_2", "text": "결제일을 바꾸고 싶어" }
```

| type | 필수 필드 | 용도 |
| --- | --- | --- |
| `answer` | `request_id`, `text` | 재질문(`clarification`) 음성 답변. 목적에 반영 후 재판단 |
| `select` | `request_id`, `key` | 재질문 후보 선택. 중요 선택(`critical_action`)은 `select` 만 허용 |
| `key` | `key` | 직접 선택. 키(`0-9*#`, 1~4자) 즉시 전송, AI 메뉴 판단 10초 보류 |
| `secure_input` | `request_id`, `digits` | 민감정보 직접 입력. 톤 전송 후 즉시 폐기 (저장·AI 전달 없음) |
| `revise_goal` | `text` | 다시 말하기. 첫 번호 입력 전까지만 허용, 이후 `409` |
| `cancel` | — | 통화 종료 |

### 이벤트

- 형식: `{ "event": "...", "at": "...", "data": { } }`
- 재연결 시 `snapshot` 으로 화면 재구성

| event | data | 전송 시점 |
| --- | --- | --- |
| `snapshot` | `status`, `message`, `goal`, `goal_summary`, `path`, `captions`, `pending_request`, `summary`, `end_reason` | 연결·재연결 직후 |
| `status_changed` | `status`, `message` | 상태·안내 문장 변경 시 |
| `caption` | `step`, `speaker`, `text`, `at` | ARS 안내 한 단락 종료 시 |
| `path_updated` | `step`, `text`, `options`, `selected`(`key`·`label`·`category`·`confidence`), `actor` | 키 전송 직후 |
| `request` | `request_id`, `kind`, `message`, `expires_at` 외 `kind` 별 필드 | 사용자 응답 필요 시 |
| `input_rejected` | `keys`, `actor`, `ars_prompt`, `retry_count`, `message` | 키 전송 후 ARS 입력 오류 안내 시 |
| `summary` | `kind`(`result`/`handoff`), `summary`, `key_info`, `handoff_script` | 결과 수신, 상담원 연결 시 |
| `ended` | `end_reason`, `message`, `detail`(선택) | 통화 종료 시 |

### 사용자 요청 (`request`)

| kind | reason | 앱 화면 | 응답 |
| --- | --- | --- | --- |
| `clarification` | `low_confidence` (최고 신뢰도 0.30~0.50), `ambiguous` (1·2순위 차이 0.10 미만), `loop` (지나온 메뉴 재도착 2회, 또는 AI 선택 5회 동안 미도달), `input_rejected` (같은 단계 입력 오류 2회 연속), `unknown_prompt` (목적만으로 답할 수 없는 안내) | 재질문 | `answer`, `select` |
| `clarification` | `critical_action` (결제·계약·해지 등) | 재질문 | `select` 만 |
| `secure_input` | — | 키패드 | `secure_input` |

### 상태 (`status`)

`dialing` → `listening` ⇄ `selecting` → … → `listening_result` → `agent_done`(업무 완료, 통화 유지) → `ended`

- 사용자 응답 대기: `awaiting_user`
- 상담원 연결: `waiting_agent` → `connected_to_agent`

### 종료 사유 (`end_reason`)

| outcome | end_reason |
| --- | --- |
| `success` | `completed` |
| `partial` | `agent_call_ended`, `message_sent` |
| `failed` | `user_input_timeout`, `out_of_hours`, `ars_rejected`, `not_found`, `loop_detected`, `user_canceled`, `no_answer`, `busy`, `failed`, `max_duration` |

### 오류

```json
{ "error": { "code": "invalid_state", "message": "이미 종료된 통화입니다." } }
```

| HTTP | code | 의미 |
| --- | --- | --- |
| 400 | `invalid_request` | 필수 값 누락, 형식 오류 |
| 401 | `unauthorized` | 앱 키·세션 토큰 오류 |
| 404 | `not_found` | 세션 없음 |
| 409 | `invalid_state` | 현재 상태에서 처리할 수 없는 조작 (종료된 통화, 만료된 요청, 다른 통화 진행 중, 첫 번호 이후 `revise_goal` 등) |
| 409 | `sensitive_input_blocked` | 민감정보 입력 요청 중 `key` 사용 |

## 실행

- Python 3.10 이상 필요
- macOS 기본 `python3` 는 3.9 인 경우가 많으므로 python.org 또는 Homebrew(`brew install python`)로 3.10 이상 설치

macOS · Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

PowerShell 스크립트 실행이 막힌 경우 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 1회 실행

서버 실행 (가상환경 활성화 후, 운영체제 공통)

| 명령 | 용도 |
| --- | --- |
| `python app.py --sim` | 가상 ARS (실통화·과금 없음) |
| `python app.py` | ClawOps 실통화 (`.env` 에 `CLAWOPS_API_KEY`, `CLAWOPS_ACCOUNT_ID`, `CLAWOPS_FROM` 입력) |

- 기본 주소: `http://localhost:8000` (`--port` 로 변경)
- 시험용 앱 화면: 브라우저에서 `http://localhost:8000` 접속

## API 호출 확인

- 서버 실행 상태에서 다른 터미널(가상환경 활성화)로 `api_check.py` 실행
- 앱과 같은 방식의 REST·WebSocket 호출, 요청(→)·응답(←)·이벤트(⇐) 시간순 출력

```bash
python api_check.py check
```

`check`: 가상 ARS 서버(`--sim`) 대상 전체 엔드포인트 순차 호출, 통과·실패 표 출력 (실패 시 종료 코드 1)

| 대상 | 확인 내용 |
| --- | --- |
| 세션 생성 | `201` 과 응답 필드, 앱 키 오류 `401`, 목적 누락 `400`, 통화 중 재요청 `409` |
| 이벤트 채널 | `snapshot`, `status_changed`, `caption`, `path_updated`, `request`, `summary`, `agent_done`, `ended` 수신과 형식 |
| 사용자 조작 | `secure_input` `200`, 민감정보 입력 중 `key` `409`, 첫 번호 이후 `revise_goal` `409`, 알 수 없는 `type` `400`, `cancel` `200`, 종료 후 조작 `409` |
| 결과 조회 | 통화 중 `202`, 종료 후 `200`, 토큰 오류 `401`, 없는 세션 `404` |
| 오디오 채널 | 프레임 수신, 형식(160바이트=20ms), 끊김, 소리 포함, WAV 저장 |

흐름별 확인

| 명령 | 흐름 |
| --- | --- |
| `python api_check.py goal` | 목적과 함께 발신 → 탐색 → 민감정보 직접 입력 → 결과 → 종료 |
| `python api_check.py lowconf` | 신뢰도 낮음 → 재질문(`low_confidence`) → 답변 → 결과 → 종료 |
| `python api_check.py revise` | 첫 메뉴 재질문 중 다시 말하기(`revise_goal`) → 결과 → 첫 번호 이후 다시 말하기 `409` |
| `python api_check.py real --phone 15889999 --goal "상담원 연결해줘" --sec 40` | 실통화 (서버는 `python app.py` 로 실행). `--keys "12:3,25:2"`: 12초에 3, 25초에 2 직접 선택 |

## 오디오 스트림 확인

```bash
python api_check.py goal --audio
python api_check.py real --phone 15889999 --goal "상담원 연결해줘" --sec 30 --audio
```

- `--audio`: 오디오 채널(`/audio`) 연결, 수신 음성 `audio_check.wav`(8kHz 16bit mono) 저장, 프레임 형식·수신 비율·끊김·소리 크기 출력
- `check` 는 옵션 없이도 오디오 채널 확인 포함
- 가상 ARS 는 음성 없음. 안내 문장을 텍스트로 직접 전달하므로 오디오 채널에는 안내 구간 동안 400Hz 시험음만 전송. 시험음이 끊김 없이 이어지면 정상
- 실제 ARS 음성 품질은 실통화로 확인 (`python app.py` 실행 후 `real ... --audio` 실행, 저장된 WAV 재생)
- 정상 기준: 프레임 160바이트, 수신 비율 95% 이상, 300ms 이상 멈춤 없음, 소리 구간 있음 (100ms 이상 지연 도착 횟수는 참고용 출력)
- 실시간 청취: 시험 화면(`http://localhost:8000`)의 "ARS 소리 켜기" (같은 오디오 채널 API 사용)
- 앱 → 서버(마이크) 방향은 상담원 연결(`connected_to_agent`) 중에만 통화로 전달되므로 실통화 상담원 연결에서 확인

## 실시간 듣기·디버깅 (`live.py`)

- 통화 음성 실시간 재생, 상태·ARS 자막·번호 선택·재질문 시간순 출력
- 앱 조작을 터미널 명령으로 사용, 음성 입력(목적·재질문 답변)은 텍스트로 대체
- 번호 입력 시 해당 버튼음(DTMF)을 재생음에 합성

```bash
python live.py
python live.py --goal "통장 잔액 알고 싶어"
python live.py --session ses_xxx --token st_xxx
```

| 실행 | 용도 |
| --- | --- |
| `python live.py` | 받는 번호·목적 입력 후 발신 |
| `python live.py --goal "..."` | 목적 지정 후 바로 발신 |
| `python live.py --session ... --token ...` | 앱이 만든 세션에 연결, 함께 듣기·조작 |

| 명령 | 앱 조작 |
| --- | --- |
| `a <문장>` | 재질문 답변 (`answer`) |
| `s <번호>` | 재질문 후보 선택 (`select`) |
| `k <키>` | 직접 선택 (`key`). 예: `k 3`, `k *` |
| `p` | 민감정보 직접 입력 (`secure_input`, 입력 숫자 화면 비표시) |
| `r <문장>` | 다시 말하기 (`revise_goal`, 첫 번호 입력 전까지) |
| `m` | 소리 켜기·끄기 (오디오 채널 연결·해제) |
| `q` / `x` | 통화 종료 (`cancel`) / 콘솔만 종료 (통화 유지) |

- 소리 출력: `sounddevice` 우선, 사용 불가 시 FFmpeg `ffplay` (`--player ffplay` 로 지정 가능)
- `sounddevice` 는 requirements 포함 (Windows·macOS 추가 설치 없음)
- FFmpeg 설치: Windows `winget install Gyan.FFmpeg`, macOS `brew install ffmpeg`
- 기타 옵션: `--wav 파일명` 재생음(버튼음 포함) 저장, `--no-sound` 소리 끈 상태로 시작, `--raw` 이벤트 원문(JSON) 출력
- 앱이 만든 세션 연결 시 세션 생성 응답의 `session_id`·`session_token` 사용 (이벤트·오디오 채널은 여러 클라이언트 동시 연결 가능)
- 시험 화면(`http://localhost:8000`)에서도 같은 기능 제공: "ARS 소리 켜기"로 실시간 청취, 번호 입력 시 상태 카드에 번호·선택지·신뢰도 표시와 키패드 버튼 강조 (소리 켠 경우 버튼음 재생)

## 자동 시험

```bash
python -m pytest -q test_flows.py
```

가상 ARS 기반 전체 흐름·`api_check.py check`·`live.py` 콘솔 시험 (실통화 없음)

## 구조

| 파일 | 역할 |
| --- | --- |
| `app.py` | REST·WebSocket 엔드포인트 |
| `session.py` | 통화 1건의 상태 머신 (상태·자막·경로·요청·종료, 키 전송) |
| `navigator.py` | 메뉴 탐색 규칙 (현재 계층 → 이전 계층 확인 → 종료) |
| `interpreter.py` | AI 슬롯필링 연동 지점 (`Interpreter` 프로토콜, 목 구현) |
| `mock_ai.json` | AI 자리의 목 데이터 |
| `telephony.py` | ClawOps 실통화 백엔드, 가상 ARS 백엔드 |
| `stt.py`, `audio.py` | 로컬 Whisper 음성 인식, μ-law·DTMF·발화 구간 처리 |
| `sim_kb.json` | 가상 ARS 메뉴 트리 (시험용, 실제 메뉴와 다름) |
| `api_check.py` | API 호출 확인 도구 (전체 점검, 흐름별 시나리오, 오디오 채널 확인) |
| `live.py` | 실시간 듣기·디버깅 콘솔 (앱 대역) |
| `test_flows.py` | 자동 시험 |
| `static/index.html` | 시험용 앱 화면 |

## AI 슬롯필링 연동

- `interpreter.py` 의 `Interpreter` 프로토콜 7개 함수(`summarize_goal`, `classify`, `score`, `answer_confirm`, `merge_answer`, `summarize_result`, `handoff_script`) 구현 후 `.env` 에 클래스 지정
- 입출력 형식: 명세서 9장

```
INTERPRETER=my_module:MyInterpreter
```

- 민감정보 입력값은 AI 미전달
- ARS 가 입력값을 다시 읽어 주는 문장은 숫자 마스킹 후 전달

### Gemini 해석기 (`gemini_interpreter.py`)

```
INTERPRETER=gemini_interpreter:GeminiInterpreter
GEMINI_API_KEY=...            # https://aistudio.google.com/apikey
GEMINI_MODEL=gemini-3.5-flash-lite
```

- 한국어로 말한 번호(`삼 번`, `별표`, `우물정자`)를 `3`·`*`·`#` 로 바꾼 뒤 판단 (`이번 달` 등은 제외)
- 안내에 실제로 나온 번호만 선택지로 인정 (번호 추측 금지)
- 재질문 답변의 번호(`삼 번이요`)·대명사(`그거`)는 직전 메뉴 선택지와 지나온 경로로 해석
- 확인 질문의 명확한 `네`/`아니요` 답변은 LLM 없이 규칙으로 처리 (중요 선택 제외)
- Gemini 실패·시간 초과 시 1회 재시도, 이후 번호를 고르지 않음 (`classify` 는 규칙 기반 분류, `score` 는 0.40 → 재질문)
- 민감정보 요청은 모델이 놓쳐도 규칙으로 한 번 더 확인
- 시험: `python -m pytest -q test_gemini_interpreter.py` (Gemini 호출 없음)
- `.env` 에서 Gemini 해석기를 켠 상태로 `test_flows.py` 를 돌리면 목 데이터 기준 시험이 실패하므로 `INTERPRETER=interpreter:MockInterpreter python -m pytest -q test_flows.py` 로 실행
- 건강보험증 발급 평가 10건(`feat/ai-evaluation`) × 2회: 정답 20/20, 위험 자동입력 0건, 메뉴 판단 p50 약 3.3초 (무료 키, 2026-10-24)

## 주의

- `.env` 는 ClawOps API 키 포함, 커밋 금지 (`.gitignore` 처리)
- ClawOps Trial 플랜: 동시 통화 1건, 발신 10분 제한
- 첫 실통화 실행 시 Whisper 모델 다운로드 (small 약 460MB)
