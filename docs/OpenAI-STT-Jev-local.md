# OpenAI STT + Jev 로컬 실험

`fix/caption-and-readback`의 `b454526`에서 시작한 별도 실험이다. Android 앱이나 운영 서버에 적용한 변경이 아니다. 코드·회귀 테스트·설정 예시만 공유한다. 실제 키, 통화 녹음·전사문·로그, 모델 가중치와 측정 원본은 포함하지 않는다.

현재 실행 프로필은 `jev_menu_interpreter:UnifiedJevInterpreter`다. 기존 `jev_phrase_interpreter:PhraseJevInterpreter`는 수정 전후 비교용으로 남아 있다. 기존 `.env`를 재사용한다면 INTERPRETER만 새 프로필로 바꾸거나 실행 명령 앞에 `INTERPRETER=jev_menu_interpreter:UnifiedJevInterpreter`를 지정한다. 키는 변경할 필요가 없다.

## 설치와 가상 ARS 실행

Python 3.11 권장. 기존 작업 폴더와 다른 디렉터리에서 실행한다.

```sh
git clone --branch experiment/openai-stt-jev --single-branch https://github.com/InternetSecurityProject/ARS-agent-backed.git ARS-agent-openai-jev
cd ARS-agent-openai-jev
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.openai-jev.example .env
```

`.env`에 본인 `APP_KEY`와 `TYPESAFE_API_KEY`를 입력한다. OpenAI STT에는 `OPENAI_API_KEY`, 실제 발신에는 ClawOps 키·계정·허가받은 발신번호도 필요하다. `APP_KEY`는 웹 화면에 입력하는 이 로컬 서버의 인증값이다. 쉘에 같은 이름의 환경변수가 있으면 `.env`보다 우선한다. 설정 변경 후 서버를 수동 재시작하되 통화 중에는 재시작하지 않는다.

```sh
.venv/bin/python app.py --sim --no-reload --host 127.0.0.1 --port 8012
```

`http://127.0.0.1:8012/app`을 연다. 기존 8011 서버와 겹치지 않도록 8012를 사용한다. `--sim`에서는 실제 발신 없이 텍스트 가상 안내를 사용하므로 **OpenAI STT 성능은 검증하지 않는다**. 메뉴 판단을 위한 Jev API 호출·과금은 발생할 수 있다. API 없이 코드만 검증하려면 아래 pytest를 실행한다.

## 개인정보 보호용 로컬 모델

개인정보 입력 후 안내는 별도 로컬 인식기로 처리한다. 모델이 없으면 해당 구간의 자동 인식·다음 키패드 안내가 제한된다. 실제 테스트 전에 공개 모델을 다운로드한다. 녹음을 업로드하는 명령은 아니다.

```sh
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-small", local_dir="models/faster-whisper-small")
PY
```

`.env`의 `PRIVATE_STT_MODEL=models/faster-whisper-small`을 설정한다. 실행 중에는 디스크 모델만 사용하며 자동 다운로드나 클라우드 대체 인식은 하지 않는다. CPU int8, 4 threads, beam 3이며 CPU에 따라 지연이 발생할 수 있다.

## 실제 통화 준비 — 발신은 사용자가 직접

`.env`의 `CALL_BACKEND=clawops`를 설정하고 아래 명령을 실행한다. **서버 실행만으로 전화가 걸리지는 않지만**, 웹에서 실제 발신을 확인하거나 `POST /v1/sessions`를 호출하면 발신이 시작된다. ClawOps 계정 권한·발신번호 승인이 필요하며 3자리 특수번호 연결은 보장하지 않는다.

```sh
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import uvicorn
from config import Settings
from app import create_app, setup_logging

st = Settings()
if st.backend != "clawops":
    raise SystemExit(".env의 CALL_BACKEND를 확인하세요.")
if st.stt != "openai" or st.interpreter != "jev_menu_interpreter:UnifiedJevInterpreter":
    raise SystemExit(".env.openai-jev.example의 STT·INTERPRETER를 확인하세요.")
model = Path(os.environ.get("PRIVATE_STT_MODEL") or "models/faster-whisper-small").resolve()
if not (model / "model.bin").is_file():
    raise SystemExit("개인정보 보호용 로컬 모델을 먼저 준비하세요.")
st.private_stt_model = str(model)
# 실제 전화번호 되읽기 음성을 OpenAI에 전송하는 데 명시적으로 동의한
# 테스트에 한해서만 True로 바꾼다. 계좌·주민번호 전송 허용이 아니다.
st.allow_openai_phone_readback = False
setup_logging(False)
uvicorn.run(create_app(st), host="127.0.0.1", port=8012)
PY
```

`app.py` CLI는 `private_stt_model`과 `allow_openai_phone_readback`을 환경변수에서 자동 설정하지 않는다. 따라서 보호용 모델이 필요한 실제 실험은 위 실행 코드를 사용한다. 비공개 진단 스크립트는 필요하지 않다. 자동 재시작은 사용하지 않는다.

웹에 APP_KEY를 입력하고 목적·전화번호를 확인한 뒤 사용자가 발신한다. 개인정보 입력 및 중요한 확인 선택은 사용자가 수행한다. 다음 입력 안내 시 자동 키패드, 확인 시 의미가 드러나는 버튼, 종료 후 결과를 확인한다. 녹음은 필요한 경우에만 사용하고 결과 파일은 저장소에 넣지 않는다. **이벤트 마스킹이 녹음 파일까지 비식별화하지는 않는다.**

## 처리 구조와 주요 변경

1. 8 kHz μ-law 음성을 20 ms 프레임으로 받는다. 음량 기반 발화 검출, 약 0.3초 쉼에서 전사 선행 처리, 1.2초 무음에서 단락 종료를 판정한다.
2. OpenAI `gpt-transcribe`로 한국어·영어를 전사한다. 최소 6초 단위의 선행 전사, 2초 중첩과 단락별 작업 합치기를 사용한다. 충돌이 있으면 전체 단락으로 복구한다. Realtime STT API 구현은 아니다.
3. `PhraseJevInterpreter`가 원문 구절 후보와 들은 번호의 연결을 Jev에 판단시킨다. 원문·번호·역할이 불확실한 항목은 선택 근거로 쓰지 않는다. 자막은 메뉴 추출 결과로 재조립하지 않고 전사문을 유지한다.
4. 사용자 목적·답변·선택 경로를 세션별로 유지한다. 추출 이후 하나의 Jev 요청에서 다음 행동과 아직 필요한 사용자 선택을 함께 결정한다. 직접 업무와 상위 업무는 모두 정상적인 진입 경로다. 후보 검증→상위 메뉴 검증→대상·행동별 재검증을 순차 호출하지 않는다. 불확실한 원문-번호 추출의 보완 호출은 별도로 남는다.
5. 시스템 판단 실패는 제한된 자체 재판단·실제로 안내된 다시듣기로 복구한다. 사용자만 아는 정보가 부족하면 질문하고, 명확한 목적에 진행할 메뉴가 없다는 근거가 있을 때 종료한다. 루프·오래된 판단·입력 중 선택·없는 번호를 코드로 방어한다.
6. 키 전송 때 음성 경계와 진행 중 전사를 초기화한다. DTMF 잔향이나 다음 단계 음성이 이전 단계 처리에 섞이는 문제를 방어한다.

Jev는 정해진 선택지/수치 응답을 반환한다. 자유로운 설명 생성이나 OpenAI 보조 LLM은 사용하지 않는다. 안내·결과 문구는 원문과 템플릿에 기반한다. Gemini 해석기의 공통 자료형·규칙을 가져오지만 이 프로필에서 Gemini API를 호출하는 것은 아니다. 실험용 `enable_general_intent`는 기본 꺼짐이며 활성화하지 않는다.

### 판단 보류 진단·다시듣기 복구

서버 로그의 `route_diagnostic` 중 `policy=unified`, `stage=menu_decision`이 현재 판단 결과다. 행동·사용자 선택 보호·원문 연결 점수와 고정 탈락 코드를 기록한다. 사용자 목적·전사문·옵션 이름·공급자 오류 본문은 이 진단에 넣지 않는다. 모델 점수는 실제 정답률을 보증하는 확률이 아니다. 로깅 오류는 실제 선택 판정을 바꾸지 않는다.

| 최종 행동 | 코드 검사 |
| --- | --- |
| 다음 메뉴 자동 선택 | 선택 확신·확률 각각 0.80 이상, 원문-번호 근거 0.70 이상, 필요한 사용자 선택의 강한 근거가 없어야 함 |
| 사용자 질문 | 현재 안내의 실제 선택지 쌍 또는 명시된 이용 조건, 확신·확률 각각 0.70 이상 |
| 경로 없음 | 확신·확률 각각 0.90 이상, 원문 추출이 완전하고 미확인·다른 경로가 없어야 함 |
| 불확실·오류 | 자체 복구. 사용자에게 메뉴 해석을 떠넘기거나 경로 없음으로 종료하지 않음 |

`decision`과 별도의 `user_choice` 보호 항목은 **같은 요청**에 들어간다. 신청 목적만으로 문자/상담원 등 진행 방법을 대신 고르지 않도록 한다. 현재 이용 조건은 선택지가 하나여도 확인하며, 이후 본인확인·신청 정보까지 미리 요구하지 않는다. 실제 선택지 쌍·명시된 조건의 확신과 확률이 모두 0.70 이상이면 질문으로 전환한다. 강한 시스템 불확실 신호도 자동 선택을 보류한다.

0.70 미만의 보조 추측만으로는 기준을 통과한 주 선택을 취소하지 않는다. `weak_user_choice_not_applied` 진단으로 구분하며 주 판단 0.80·원문 연결 0.70 기준을 낮추거나 약한 선택을 되살리는 처리는 하지 않는다. 경로 없음 종료는 여전히 강한 ‘필요한 사용자 정보 없음’과 완전한 원문 근거를 요구한다. 응답 형식 오류도 기존처럼 보류한다. 한 항목에 함께 묶인 독립 업무와 같은 업무의 실제 고객 유형·진행 방법 차이를 구분하도록 지시한다.

일반 확인 질문도 사용자에게 표시한다. 이전 질문에 한 ‘네’가 다음 단계의 확인을 승인하지 않도록 답변을 원문·추출 근거에 묶고 한 번만 사용한다. 개인정보 입력, 되읽기 확인, 중요한 최종 승인은 기존 별도 경로를 유지하며 중요한 승인은 버튼으로만 답한다.

제한된 복구 과정에서 다시듣기 번호가 성공적으로 전송된 경우에만 같은 목적·원문·추출 근거의 `unknown` 캐시를 무효화한다. 새 안내는 새 판단을 받을 수 있고, 단순 조회나 중복 함수 호출은 AI를 반복 호출하지 않는다. 성공 판정과 사용자 선택이 필요한 판정은 유지하며 기존 재판단·다시듣기 횟수 제한도 유지한다. 관련 테스트는 `test_route_diagnostics.py`, `test_replay_verdict_cache.py`에 있다.

## 개인정보·버튼 표시

- 개인정보 되읽기와 **다음 필드 입력 안내**를 구분한다. 계좌번호 등이 필요하면 기존 secure_input 요청으로 키패드를 표시한다. 원시 값은 Jev·일반 자막·이벤트 기록으로 전달하지 않는다.
- 보호 중 인식이 실패했다고 시간이 지나면 외부 STT로 재전송하지 않는다. 이전 비동기 결과가 새 입력 화면에 적용되지 않도록 세대를 검사한다.
- 전화번호 되읽기를 별도로 허용하면 제한된 기간·단일 요청에서 OpenAI 전송을 허용한다. 읽힌 숫자를 표시하되 누락 숫자를 기대값으로 채우지 않는다.
- 전화번호 요청의 앞 문구에서 `경과의`, `결과를`처럼 조사까지 붙은 단어 안의 `과`는 추가 입력 항목을 잇는 연결어로 처리하지 않는다. 실제 연결어·다른 개인정보 항목·미지원 입력 문장은 계속 보호하며, 띄어쓰기 없는 복합 입력이나 인식이 애매한 표현을 이유로 다른 개인정보를 OpenAI에 보내지 않는다.
- 단일 전화번호 요청에 한해 실제 전송한 번호를 한시적으로 메모리에서 대조한다. 화면은 `입력한 번호`와 `음성 인식 번호`를 따로 표시하며, 다르면 인식 오류 가능성과 소리 확인·재입력을 안내한다. 입력값으로 전사문을 고치거나 자동 승인하지 않는다. 비교 데이터는 Jev나 다른 외부 서비스로 보내지 않는다.
- 번호 대조는 입력 안내의 종료 기호가 없음/별표/샵으로 명확한 경우에만 적용한다. 전송 대기 중에는 대조 사본을 만들지 않는다. 전송 실패·새 입력 항목·취소·종료·되읽기 대기 30초 만료 시 해제하며, 확인 화면을 만든 뒤에는 별도 대조 사본을 소비한다. 현재 요청의 표시 메시지는 확인 응답·요청 교체·확인 시간 만료·종료 때 번호를 지운다. 계좌·주민번호 등은 이 비교 대상으로 보관하지 않는다.
- 개인정보 되읽기 확인은 버튼으로만 응답한다. 일반 목적·메뉴 답변의 음성 기능은 유지한다. 거절되거나 만료된 확인 요청의 음성 답변은 텍스트 기록에 넣지 않는다.
- 버튼은 의미가 명확할 때 `발급하기`, `발급하지 않기`, `취소하기`, `다시 입력하기`, `주민등록 주소로 받기`, `회사 주소로 받기`처럼 표시한다. 실제 번호·순서·판단 근거는 변경하지 않는다. 모호한 부정·복합 주소는 원래 문구를 유지한다. 추가 AI 호출은 없다.
- 위 보호는 인식된 입력·요청 상태에 의존하는 실험적 방어다. 모든 개인정보를 자동 탐지·차단한다고 보장하지 않는다. 음성·목적 자체에 예기치 않은 개인정보가 있을 수 있다.

## 테스트

```sh
.venv/bin/python -m pytest -q -rs
```

가짜 통화 백엔드, 합성 문장, 모의 API 응답으로 회귀 검증한다. 실제 발신·OpenAI/Jev API 호출은 하지 않는다. 일부 HTTP/WebSocket 테스트는 localhost 포트 허용이 필요하다. FFmpeg가 없으면 해당 오디오 테스트 1개를 건너뛴다.

2026-11-07 공유 준비 시 Python 3.11.18에서 전체 테스트 **674개 통과, FFmpeg 미설치로 1개 건너뜀**을 확인했다.

| 영역 | 주요 테스트 |
| --- | --- |
| STT 선행 처리·음성 경계 | `test_openai_prefetch.py`, `test_onset_resume.py`, `test_menu_transition.py` |
| 원문과 번호·중복 안내 | `test_jev_phrase.py`, `test_binding_validation.py`, `test_repeated_menu.py` |
| 목적·상위 메뉴·자체 복구 | `test_relation_policy.py`, `test_parent_menu.py`, `test_system_recovery.py`, `test_route_validation.py` |
| 통합 판단·보류 진단·다시듣기 캐시 | `test_unified_menu.py`, `test_route_diagnostics.py`, `test_replay_verdict_cache.py` |
| 개인정보·되읽기·다음 입력 | `test_protected_input.py`, `test_phone_readback_audio.py`, `test_readback_readability.py` |
| 버튼·주소 선택 | `test_choice_labels.py`, `test_address_choice_labels.py` |
| 번호 대조·임시값 정리·전체 텍스트 기록 가림 | `test_phone_comparison.py` |
| 전화번호 전용 OpenAI 허용·복합 입력 차단 | `test_phone_request_gate.py` |

테스트 통과는 실제 모델의 의미 판단 정확도를 보증하지 않는다. 과거 통화의 속도 측정 원본은 개인정보를 포함할 수 있어 공유하지 않으며 일반적인 기대 속도로 단정하지 않는다.

## 웹 빌드와 알려진 한계

`static/app.html`은 빌드된 파일이므로 실행할 때 프론트 저장소가 필요하지 않다. 소스는 `web/app.src.html`이며 로컬 테스트를 위한 loopback HTTP를 허용한다. 공개망 운영용 설정이 아니다. 재빌드는 `web/build.py` 안내대로 htm 3.1.1을 `web/vendor`에 준비하고 별도 프론트 복사본의 SVG assets 경로를 넘긴다. 프론트 원본은 수정하지 않는다.

- 건강보험증처럼 업무와 상위 범주 연결이 약한 요청에서는 Jev가 여전히 진행을 보류할 수 있다. 기관별 정답 번호표나 OpenAI 보조 판단으로 우회하지 않는다.
- 단어/번호 누락, 언어 혼합, 짧은 응답 대기, 긴 안내와 CPU 지연을 모든 기관에서 해결한 상태는 아니다.
- 보호 구간의 미지원 안내는 자동으로 외부에 보내지 않으므로 수동 입력이나 추가 진단이 필요할 수 있다.
- 3자리 특수번호의 ClawOps 발신 지원은 이 변경으로 해결되지 않는다.
- Android 반영, 운영 배포, main 병합, 메뉴 추출과 판단까지 한 호출로 합치는 추가 리팩터링은 범위에 없다.
