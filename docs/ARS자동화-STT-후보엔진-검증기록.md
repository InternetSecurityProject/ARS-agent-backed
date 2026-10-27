# ARS 자동화 — STT 후보 엔진 검증 기록

작성일: 2026-10-27
대상: `feat/stt-engines` 브랜치의 `stt_engines.py`, `stt.py`(OpenAI 제한 시간), `stt_bench.py`(엔진 선택)
관련 문서: 무음 환각·숫자 처리 개선은 `docs/ARS자동화-STT-무음환각-숫자처리-설계안.md` (구현 전 설계)

---

## 1. 대상 엔진

| `STT` 값 | 엔진 | 오디오 외부 전송 | 필요 설정·패키지 |
| --- | --- | --- | --- |
| `whisper` | faster-whisper (기존) | 없음 | `WHISPER_MODEL`. 후보 `large-v3-turbo`, batisay(변환 필요, `.env.example`) |
| `openai` | OpenAI 음성 인식 (기존, 제한 시간 추가) | 있음 | `OPENAI_API_KEY`, `pip install openai` |
| `rtzr` | 리턴제로 스트리밍 STT | 있음 | `RTZR_CLIENT_ID`, `RTZR_CLIENT_SECRET` |
| `clova` | CLOVA Speech 장문 인식(동기) | 있음 | `CLOVA_SPEECH_INVOKE_URL`, `CLOVA_SPEECH_SECRET` |
| `gemini` | Gemini 오디오 입력 | 있음 | `GEMINI_API_KEY` (`GEMINI_STT_MODEL` 비우면 `GEMINI_MODEL`) |
| `qwen` | Qwen3-ASR 로컬 | 없음 | qwen-asr 방식: `pip install qwen-asr torch` / hf 방식: `pip install "transformers>=5.13" torch` (+LoRA 는 `peft`) |

공통 규칙
- 외부 전송 엔진은 `is_local=False` 이며, 민감정보를 다시 읽어 주는 구간은 `session.py` 가 보내지 않는다.
- 외부 전송 엔진에는 미리 인식(`prefetch`)을 두지 않는다. 미리 인식 경로는 위 검사를 거치지 않기 때문이다.
- 인식 실패·시간 초과(`STT_TIMEOUT_S`, 기본 10초)는 빈 결과와 경고 로그(키 가림)로 처리한다.
- 필수 키나 패키지가 없으면 서버 시작 시 `RuntimeError` 로 알린다.

---

## 2. 검증 결과

### 2.1 자동 시험

- `python -m pytest -q`: 전체 157건 통과 (기존 113건 + 엔진 시험 44건). OpenAI SDK 가 없는 환경에서는 OpenAI 시험 3건이 건너뛰어진다.
- 엔진 시험(`test_stt_engines.py`)의 확인 범위
  - 요청 형식: 주소, 인증 헤더, 쿼리·본문 필드, 음성 형식(표본화 주파수·길이)
  - 응답 처리: 확정 문장만 사용, 빈 문장 제외, `stt._clean` 적용
  - 실패 처리: HTTP 오류, 오류 응답, 인증 실패(401) 후 토큰 재발급, 응답 지연, 토큰 발급 무응답, 서버가 연결을 닫지 않는 경우
  - 시작 시 검사: 키 누락, Qwen3-ASR 패키지 누락·transformers 버전
  - 통화 흐름: 가짜 서버로 인식 → 메뉴 판단 → 키 전송, 민감정보 되읽기 구간 미전송, 외부 엔진에 미리 인식 없음
  - 비교 도구: 저장소 시험 음성(`bench_clips`, 19개)을 엔진 이름으로 끝까지 인식하고 요약 표 출력

### 2.2 외부 API 요청 형식 대조

실제 키 없이 공개 자료와 대조하였다.

| 엔진 | 대조 자료 | 일치를 확인한 항목 | 확인하지 못한 항목 |
| --- | --- | --- | --- |
| 리턴제로 | `livekit-plugins-rtzr` 1.8.9 소스(리턴제로 연동 클라이언트), 리턴제로 공식 블로그 실시간 API 예제 | 인증 `POST /v1/authenticate` → `access_token`·`expire_at`, `wss://openapi.vito.ai/v1/transcribe:streaming`, `Authorization: bearer`, 쿼리 `model_name`·`domain`·`sample_rate`·`encoding`·`epd_time`·`use_punctuation`·`keywords`, 종료 문자열 `EOS`, 응답 `alternatives[0].text`·`final`, 오류 `error`·`type=error`, 토큰 유효 6시간 | 실시간보다 빠른 전송의 처리 속도 (공식 예제는 실시간 속도로 전송). 설계안 2.8 의 방법으로 첫 실측에서 확인 |
| CLOVA Speech | 네이버 클라우드 API 문서(로컬 파일 인식, 국문·영문) | `POST {Invoke URL}/recognizer/upload`, `X-CLOVASPEECH-API-KEY`, multipart `media`·`params`·`type`, `language`·`completion=sync`·`wordAlignment`·`fullText`·`noiseFiltering`·`diarization.enable`·`boostings`(`words`, `weight` 1~5 실수), 동기 응답 `result=COMPLETED`·`text` | 음성 표본화 주파수 요구사항(문서에 없음). `CLOVA_SAMPLE_RATE` 8000·16000 을 모두 측정 |
| Gemini | 저장소의 Gemini 해석기(`gemini_interpreter.py`)가 쓰는 호출 방식 | 같은 주소·헤더(`x-goog-api-key`)·필드(`responseJsonSchema`, `thinkingConfig.thinkingLevel`). 오디오는 `inlineData`(`audio/wav`) | 오디오 입력에서의 응답 시간 |
| OpenAI | openai 3.39 SDK | `AsyncOpenAI(timeout, max_retries)`, `audio.transcriptions.create(model, file, language, prompt)` | — |

### 2.3 로컬 엔진(Qwen3-ASR) 실행

작업 환경에서 Hugging Face 다운로드가 막혀 실제 가중치를 받을 수 없었다. 대신 같은 구조의 작은 모델을 무작위 가중치로 만들어 실제 라이브러리로 엔진 전체 경로를 실행하였다. 인식 문장은 의미가 없으며, 확인 대상은 호출 방식과 반환 형식이다.

| 방식 | 라이브러리 | 확인 결과 |
| --- | --- | --- |
| hf | torch 2.14.1, transformers 5.21.0, peft 0.21.2 | 모델·프로세서 로드, `apply_transcription_request` → `generate` → `decode(return_format="transcription_only")` 정상. `decode` 는 목록을 돌려주므로 `[0]` 사용이 맞다. 언어 지정 시 프롬프트 끝이 `assistant\nlanguage Korean<asr_text>`, 자동 판별 시 `assistant\n` 으로 transformers 시험의 기대 형식과 같다. LoRA 어댑터 적용(`PeftModel`) 확인. 시험 음성 4개 인식 중 오류 없음 |
| qwen-asr | qwen-asr 0.0.6, transformers 4.57.6 | `Qwen3ASRModel.from_pretrained`, `transcribe(audio=(배열, 16000), context, language)` 정상. 언어 지정·자동 판별 두 조건 모두 오류 없음 |

검증 중 발견하여 수정한 사항
1. hf 방식에서 `device_map` 을 쓰면 accelerate 가 없을 때 로드가 실패하였다. 장치 하나로 옮기는 방식(`.to(장치)`)으로 바꾸어 accelerate 없이 로드되는 것을 확인하였다.
2. 패키지가 없으면 통화마다 빈 결과가 나왔다. 서버 시작 시 설치 명령을 담은 오류로 알리도록 바꾸었다.
3. bfloat16 을 지원하지 않는 GPU 에서는 float16 을 쓰도록 하였다.

참고: qwen-asr 패키지를 불러오면 로그 출력 수준이 INFO 로 바뀌어 인식 로그가 많이 출력된다. 기능에는 영향이 없다.

### 2.4 Whisper 계열 후보

- `large-v3-turbo`: faster-whisper 1.2.1 의 모델 이름 별칭(`mobiuslabsgmbh/faster-whisper-large-v3-turbo`)을 확인하였다.
- batisay: CTranslate2 4.8.4 변환기의 인자(`--model`, `--output_dir`, `--copy_files`, `--quantization`)를 확인하였다. 변환 폴더에 `tokenizer.json` 이 없으면 faster-whisper 가 whisper-tiny 토크나이저로 대체하는 것을 코드에서 확인하였으며, large-v3 계열과 특수 토큰 번호가 달라 인식이 깨지므로 `.env.example` 에 대처 방법을 적었다.

---

## 3. 첫 실행 확인 순서 (실제 키·가중치)

이 환경에서 확인하지 못한 것은 실제 API 호출과 실제 가중치 로드이다. 저장소 시험 음성으로 다음 순서로 확인한다. 각 줄은 요약 표 한 행을 만든다.

| 순서 | 명령 | 확인할 것 |
| --- | --- | --- |
| 1 | `python stt_bench.py --models rtzr` | "RTZR STT 실패" 경고가 없는지. `clip14`(약 30초) 인식 시간이 2~3초 안인지 (음성 길이에 가까우면 설계안 2.8) |
| 2 | `python stt_bench.py --models clova` 후 `CLOVA_SAMPLE_RATE=16000` 으로 한 번 더 | 두 조건의 문자 오류율 비교 |
| 3 | `python stt_bench.py --models gemini` | 400 오류가 나면 모델 이름(`GEMINI_STT_MODEL`)과 `GEMINI_STT_THINKING`(2.5 계열은 비움) 확인 |
| 4 | `python stt_bench.py --models openai` | `pip install openai` 필요 |
| 5 | `python stt_bench.py --device cuda --models qwen` (qwen-asr 가상환경) | 기본 `TeamUNIVA/qwen3_asr_1.7b_ko`. 첫 실행은 가중치 다운로드(약 4GB) |
| 6 | 같은 명령, `QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B-hf`, `QWEN_ASR_ADAPTER=Playedwell03/qwen3-asr-1.7b-119ko-v5` (transformers 5 가상환경) | 119 LoRA. 어휘 힌트(`QWEN_ASR_CONTEXT`)는 비워 둔다 |
| 7 | `python stt_bench.py --device cuda --models whisper:large-v3-turbo <batisay 변환 폴더>` | Whisper 계열 비교 |

외부 엔진은 시험 음성을 외부 API 로 전송한다. 시험 음성에는 개인정보가 없다.

---

## 4. 브랜치 구성

`feat/stt-engines` = `main`(9f373e2) + 엔진 추가 커밋 + `feat/chaeyeon-stt`(35b0e10, 비교 도구 개선·시험 음성) 병합 + 보강 커밋.
비교 도구(`stt_bench.py`)는 채연 브랜치의 판을 바탕으로 엔진 이름 선택만 더하였다. 채연 브랜치를 먼저 합치면 이 브랜치는 충돌 없이 합쳐진다.
