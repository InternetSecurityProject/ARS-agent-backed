# ARS 자동화 — STT 무음 환각·숫자 처리 개선 설계안 (초안)

작성일: 2026-10-27
대상: 구현 작업 세션 인수인계
범위: STT 인터페이스(`stt.Recognizer`) 밖에서 새로 추가해야 하는 기능이다. 후보 STT 엔진 자체는 `stt_engines.py` 에 구현되어 있으며 본 문서의 범위가 아니다.
기준 코드: `main` 9f373e2 + `feat/stt-engines` (`feat/chaeyeon-stt` 35b0e10 의 비교 도구·시험 음성 병합 포함). 줄 번호는 이 기준이다

---

## 0. 요약

실통화에서 무음 구간에 없는 문장이 인식되고, 한 음절 숫자(일·이·영 등)를 잘못 인식하는 문제가 보고되었다. 코드를 점검한 결과 원인과 개선 항목은 다음과 같다. 우선순위는 위험도(잘못된 키 전송으로 이어지는지)와 구현 비용으로 정하였다.

| 우선 | 항목 | 문제 | 변경 위치 |
| --- | --- | --- | --- |
| P1 | 2.1 어휘 힌트 축소·반향 제거 | 힌트 문장 전체가 무음 구간 출력으로 새어 나올 수 있다. 가짜 선택지·가짜 민감정보 요청이 된다 | `stt.py` |
| P1 | 2.2 단락 끝 무음 제거 | 모든 단락이 끝에 약 1.2초 무음을 포함한 채 STT 로 전달된다 | 공통 래퍼(신규) |
| P1 | 2.6 "이번" 오변환 수정 | "이번 통화는" → "2번 통화는" 으로 바뀐다 | `gemini_interpreter.py` (해석기 담당 협의) |
| P2 | 2.3 발화 길이 대비 글자 수 검사 | 확신도가 높은 환각은 현재 필터를 통과한다 | 공통 래퍼(신규) |
| P2 | 2.4 짧은 단락 제거 | 0.1초 이상 소리면 단락이 되어 잡음 조각도 STT 로 간다 | 공통 래퍼(신규) |
| P2 | 2.5 Whisper 디코딩 옵션 | 단락 내부 무음이 그대로 입력되고 반복 출력 검사가 없다 | `stt.py` WhisperRecognizer |
| 반영됨 | 2.7 OpenAI 호출 제한 시간 | 제한 시간이 SDK 기본값(10분)이었다 | `stt.py` OpenAIRecognizer (`feat/stt-engines` 에 반영) |
| 조건부 | 2.8 외부 엔진의 말하는 동안 전송 | 리턴제로가 실시간 속도로만 처리하면 긴 안내의 인식이 늦어진다. 이때 필요한 구조가 민감정보 구간 검사를 우회한다 | `app.py`, `session.py` |
| P2 | 2.9 비교 도구의 무음 지표 | 시험 음성에 무음·잡음 조각이 없어 환각을 측정할 수 없다 | `stt_bench.py`, `bench_clips/` |
| 기타 | 2.6 오타 | `"여섯": "1"` (6 이어야 함) | `gemini_interpreter.py:43` |

공통 후처리(2.2~2.4)는 엔진마다 넣지 않고, 모든 엔진을 감싸는 래퍼 하나로 구현한다(3장). 그래야 Whisper·OpenAI·후보 엔진(rtzr·clova·gemini·qwen)에 같은 규칙이 적용되고, 엔진 비교 결과가 후처리 차이에 흔들리지 않는다.

---

## 1. 현재 구조와 확인된 문제

### 1.1 처리 경로

```
통화 음성(μ-law 8kHz, 20ms)
 → audio.Segmenter.feed()        RMS 500 기준. 100ms 이상 소리 → 단락 시작, 1.2초 무음 → 단락 종료
 → session._stt_worker()         외부 엔진이면 민감정보 되읽기 구간 건너뜀
 → Recognizer.recognize(ulaw8k)  엔진별 인식 → stt._clean() (환각 문장 블랙리스트, ARS 표기 교정)
 → session.on_paragraph(text)    해석기 classify → normalize_keys() → 메뉴 판단 → DTMF
```

### 1.2 무음 환각 관련

| # | 위치 | 현재 | 문제 |
| --- | --- | --- | --- |
| a | `stt.py:31` `ARS_PROMPT` | "안녕하십니까. KB국민은행입니다. 잔액조회는 1번 … 주민등록번호 앞 여섯 자리를 입력해 주십시오." 를 모든 통화의 `initial_prompt`(Whisper)·`prompt`(OpenAI)로 사용 | Whisper 는 무음·짧은 입력에서 프롬프트 문장을 그대로 출력하는 경향이 있다. 건강보험공단 통화에서 "잔액조회는 1번" 이 나오면 가짜 선택지가 되고, "주민등록번호 앞 여섯 자리를 입력" 이 나오면 민감정보 요청으로 오판된다. `_HALLUCINATIONS` 블랙리스트로는 걸러지지 않는다 |
| b | `audio.py:204-205` | 단락 종료 시 `self.buf` 전체를 반환 | 단락 끝의 무음 60프레임(1.2초)이 포함된다. Whisper 미리 인식 경로는 남은 꼬리가 조용하면 건너뛰지만(`stt.py` recognize), 미리 인식이 없는 엔진(OpenAI·후보 엔진)과 1.5초 미만 단락은 무음까지 인식한다 |
| c | `stt.py:114` | `no_speech_prob > 0.6 and avg_logprob < -0.8` 인 구간만 버림 | Whisper 기본 규칙과 같은 수준이다. 환각 문장은 대개 확신도가 높아(avg_logprob 가 높음) 통과한다 |
| d | `stt.py:110` | `vad_filter=False` | 단락 내부의 쉼(최대 1.2초 미만)이 그대로 입력된다. 무음 구간 삽입이 한국어 환각을 크게 늘린다는 측정 결과가 있다(WildASR, arXiv 2603.25727) |
| e | `audio.py:192` | 5프레임(100ms) 이상 소리면 단락 시작 | 딸깍음·잔향 같은 짧은 소리도 단락이 되어 STT 로 간다 |
| f | `stt.py` OpenAIRecognizer | `AsyncOpenAI(max_retries=1)` | 제한 시간 미지정이었다. `feat/stt-engines` 에서 `STT_TIMEOUT_S` 로 제한하도록 반영하였다 |
| f' | `app.py:38-48` | 엔진에 `prefetch` 가 있으면 Segmenter 가 단락 조각을 바로 넘긴다 | 이 경로는 `session._stt_worker` 의 민감정보 구간 검사(`secure_window`)를 거치지 않는다. 지금은 로컬 Whisper 만 `prefetch` 가 있어 문제가 없으나, 외부 엔진에 붙이면 되읽기 음성이 외부로 나간다 (`test_external_engines_have_no_prefetch` 로 막아 둠) |

### 1.3 숫자 관련

| # | 위치 | 현재 | 문제 |
| --- | --- | --- | --- |
| g | `gemini_interpreter.py:45-46` | `_SPOKEN_KEY` 가 "이 번"·"이번" 을 "2번" 으로 바꾸고, 제외어는 `달·주·년·에·엔·만·째·호·씩·정도·처럼` 뿐 | "이번 통화는 녹음됩니다" → "2번 통화는 녹음됩니다", "이번 안내를" → "2번 안내를" 로 바뀌는 것을 확인하였다. 선택지 추출 정규식과 겹치면 가짜 2번 선택지가 될 수 있다 |
| h | `gemini_interpreter.py:43` | `"여섯": "1"` | 오타. 현재 정규식이 한 음절만 쓰므로 실제 영향은 없으나 고쳐 둔다 |
| i | (구조) | 같은 번호가 다른 선택지에 두 번 붙으면 둘 다 제외 (`gemini_interpreter.py` `_to_prompt`) | 이미 구현되어 있다. 일/이 오인식으로 번호가 충돌하는 경우는 걸러진다. 충돌하지 않는 오인식(1번이 2번으로 들리고 2번 선택지가 없는 경우)은 텍스트로 구분할 수 없다 |

---

## 2. 개선 항목

### 2.1 어휘 힌트 축소와 힌트 반향 제거 (P1)

목적: 힌트 문장이 출력으로 새어 나오는 것을 막고, 새어 나와도 버린다.

방법
1. `ARS_PROMPT` 를 기관명·업무명·민감정보 문장이 없는 단어 나열로 바꾼다. 이름은 `ARS_HINT` 로 바꾸고 `.env` 의 `STT_HINT` 로 덮어쓸 수 있게 한다.
   - 기본값 예: `"1번, 2번, 3번, 4번, 5번, 0번, 별표, 우물정자, 샵 버튼, 상담원, 다시 듣기, 이전 메뉴"`
2. 반향 검출: 인식 결과가 힌트의 일부와 거의 같으면 버린다.
   - 공백·구두점을 지운 결과 문자열 `t` 와 힌트 `h` 에 대해, `len(t) >= 4` 이고 `t` 가 `h` 의 부분 문자열이면 반향으로 본다.
   - 이어 붙이는 문맥(미리 인식의 `context[-120:]`)도 같은 방식으로 검사한다. 새 조각 결과가 앞 조각 문장의 끝부분과 같으면 버린다.
3. OpenAI 경로(`prompt=ARS_PROMPT`)에도 같은 힌트와 검사를 적용한다. 후보 엔진 중 힌트를 받는 엔진(rtzr 키워드, clova 부스팅, qwen context)은 단어 단위 부스팅이므로 반향 위험이 낮으나 같은 검사를 거쳐도 해가 없다.

주의
- 힌트를 줄이면 메뉴 이름 인식률이 떨어질 수 있다. stt_bench 로 힌트 있음/없음/축소를 비교하여 정한다(5장).
- 119 신고전화 LoRA(Qwen3-ASR)는 힌트를 주면 정확도가 떨어진다고 모델 카드에 명시되어 있다. 엔진별로 힌트 사용 여부를 둔다.

### 2.2 단락 끝 무음 제거 (P1)

목적: 1.2초 꼬리 무음을 STT 에 넣지 않는다.

방법: 공통 래퍼에서 단락의 뒤쪽부터 20ms 프레임 RMS 를 보고, 마지막 유음 프레임 뒤 `STT_TRAIL_KEEP_MS`(기본 200ms)만 남기고 자른다. 앞쪽은 Segmenter 가 이미 시작 직전 200ms 만 남기므로 그대로 둔다.

주의
- 판정 기준은 Segmenter 와 같은 `VAD_RMS` 를 쓴다. 대기 음악 판정(`music`)으로 끝난 단락은 Segmenter 가 이미 버린다.
- 꼬리를 자르면 마지막 음절이 잘릴 수 있으므로 남기는 길이를 0.2초 이상으로 둔다.

### 2.3 발화 길이 대비 글자 수 검사 (P2)

목적: 확신도가 높은 환각 문장을 길이로 걸러낸다.

방법: 공통 래퍼에서 인식 후 다음 조건이면 결과를 버리고 경고 로그를 남긴다.
- 유음 시간 `v` (RMS 가 `VAD_RMS` 를 넘는 프레임 수 × 0.02초)
- 결과의 음절 수 `n` (한글 음절과 숫자 문자 수)
- `n > STT_MAX_SYLL_PER_S × v + STT_SYLL_SLACK` 이면 버린다 (초기값 9음절/초, 여유 4음절)

주의
- ARS 안내 속도는 실측하지 않았다. 초기값은 일반 낭독 속도(초당 5~7음절 내외, 추정)에 여유를 둔 값이므로, 실통화 녹음으로 단락별 `n / v` 분포를 구해 99백분위 이상으로 정한다.
- 일부만 환각인 단락(정상 문장 + 끝에 "감사합니다.")은 이 검사로 걸리지 않는다. 2.2 와 2.5 로 줄인다.
- 결과를 자르지 않고 통째로 버린다. 부분 삭제는 오히려 잘못된 메뉴 문장을 만들 수 있다.

### 2.4 짧은 단락 제거 (P2)

목적: 잡음 조각을 STT 로 보내지 않는다.

방법: 공통 래퍼에서 유음 시간 `v < STT_MIN_VOICED_MS`(초기값 400ms)이면 인식하지 않고 빈 문자열을 돌려준다.

동작 영향 확인
- "키 전송 뒤 ARS 가 소리를 냈는지"(`session.heard_since_press`)는 Segmenter 가 소리를 감지한 시점에 정해지며(`session.py:368-369`), 인식 결과와 무관하다. 따라서 이 변경은 무응답 처리(다시 듣기 키 전송)에 영향을 주지 않는다. 점검표 G 의 "키 전송 뒤 소리는 났으나 인식 문장이 없음" 행과 같은 동작이 된다.
- ARS 가 "1번" 한 단어만 말하는 단락은 거의 없으나, 실통화 녹음에서 400ms 미만 정상 단락이 있는지 확인한다.

### 2.5 Whisper 디코딩 옵션 (P2)

목적: 단락 내부 무음과 반복 출력을 줄인다.

방법 (`WhisperRecognizer._run`)
1. `vad_filter=True`, `vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200}` 로 단락 내부 무음을 제거한다.
2. 구간별 `compression_ratio > 2.4` 이면 그 구간을 버린다 (같은 말 반복 출력). 현재는 `temperature=0.0` 단일값이라 faster-whisper 의 재시도(fallback)가 일어나지 않으므로 직접 검사해야 한다.
3. `no_speech_prob`·`avg_logprob` 조건은 유지하되, 녹음 평가에서 OR 조건(`no_speech_prob > 0.6 or avg_logprob < -1.0`)과 비교한다.
4. (선택) `hallucination_silence_threshold` 는 `word_timestamps=True` 가 필요하여 라즈베리파이에서 느려진다. PC 시연 경로에서만 실험한다.

주의
- Silero VAD 는 16kHz 기준이다. 8kHz 를 업샘플링한 음성에서 짧은 "1번" 이 잘리지 않는지 확인한다.
- 미리 인식 조각에도 VAD 가 적용된다. 조각 경계에서 낱말이 잘리는지 확인한다.

### 2.6 숫자 처리 (P1: "이번", 기타: 오타)

담당: 해석기(`gemini_interpreter.py`)는 팀원 담당이므로 변경안을 공유하고 합의 후 반영한다.

1. "이번" 오변환
   - 한자어 한 음절 숫자 + "번" 을 키로 바꾸는 조건을 메뉴 문맥으로 한정한다.
     - 앞이 `은·는·면·으로·로` 이거나(예: "잔액조회는 이 번"), 뒤가 `을·를·,·.·문장 끝·누르` 인 경우만 바꾼다.
     - 공백 없이 붙은 "이번" 뒤에 명사가 오면 바꾸지 않는다.
   - 시험 문장

     | 입력 | 기대 |
     | --- | --- |
     | 이번 통화는 녹음됩니다 | 그대로 |
     | 이번 안내를 다시 들으시려면 | 그대로 |
     | 이번 달 결제 금액 | 그대로 (기존 시험) |
     | 잔액조회는 이 번, 거래내역은 삼 번 | 잔액조회는 2번, 거래내역은 3번 |
     | 이 번을 눌러 주세요 | 2번을 눌러 주세요 |
     | 상담원 연결은 영 번 | 상담원 연결은 0번 |

2. `"여섯": "1"` → `"여섯": "6"`.
3. 일·이 같은 음향 혼동은 텍스트 규칙으로 고칠 수 없다. 대응은 다음 두 가지로 한정한다.
   - 번호 충돌 검사(이미 구현)를 유지한다.
   - 엔진 선택 시 "메뉴 번호 정확도"(단락의 (번호, 메뉴명) 쌍 일치율)를 평가 지표에 넣는다(5장).
4. 되읽기 숫자열("구 영 영 일 일 일")은 민감정보 구간이므로 마스킹 대상이며 해석에 쓰지 않는다. 결과 안내의 전화번호·금액 숫자는 요약 문장에만 쓰이므로 본 설계 범위에서 제외한다.

### 2.7 OpenAI 호출 제한 시간 (반영됨)

`feat/stt-engines` 에서 `AsyncOpenAI(..., timeout=STT_TIMEOUT_S, max_retries=1)` 와 `asyncio.wait_for` 로 전체 시간을 제한하였다. 넘으면 빈 결과를 돌려준다. 시험: `test_openai_timeout_returns_empty`.

### 2.8 외부 엔진의 말하는 동안 전송 (조건부)

배경
- 리턴제로 엔진은 단락이 끝난 뒤 단락 전체를 한 번에 전송한다. 리턴제로 공식 예제는 실시간 속도로 전송하며, 실시간보다 빠른 전송의 처리 속도는 문서에서 확인하지 못하였다.
- 서버가 실시간 속도로만 처리하면 KB 첫 메뉴(약 29초) 같은 긴 안내는 단락이 끝난 뒤 수십 초가 지나야 결과가 나온다. 이때는 `STT_TIMEOUT_S` 까지 받은 확정 문장만 쓰게 되어 뒤쪽 선택지를 놓친다.

확인 방법 (실제 키로 한 번)
- `python stt_bench.py --models rtzr` 결과에서 `clip14`(약 30초) 의 인식 시간을 본다. 2~3초 안이면 문제없다. 음성 길이에 가까우면 아래 변경이 필요하다.

변경안 (필요할 때만)
1. 먼저 민감정보 구간 검사를 미리 인식 경로에도 적용한다. `Segmenter.on_chunk` 를 클래스 공통값으로 두지 않고 세션별로 연결하여, `secure_window` 이고 엔진이 외부(`is_local=False`)이면 조각을 넘기지 않는다. 이 단계 없이 2를 하면 되읽기 음성이 외부로 전송된다.
2. 리턴제로 엔진에 `prefetch` 를 구현한다. 단락 시작 후 첫 조각에서 WebSocket 을 열고, 조각마다 새 음성만 보내고, `recognize` 에서 나머지와 EOS 를 보낸 뒤 확정 문장을 모은다. 단락이 버려지면(대기 음악) 다음 단락 시작 시 이전 연결을 닫는다.
3. `test_external_engines_have_no_prefetch` 는 1 이 구현된 뒤에만 수정한다.

### 2.9 비교 도구의 무음 지표 (P2)

`bench_clips/` 는 안내 문장 19개뿐이라 무음 환각을 잴 수 없다. 다음을 추가한다.
- 시험 음성: 무음 3초, 회선 잡음 3초, 대기 음악 형태(일정한 음량) 5초, 0.2초 딸깍음, 메뉴 사이 1초 쉼을 넣은 안내. `clips.json` 의 정답 문장은 빈 문자열(무음류) 또는 원문(쉼 포함 안내)으로 둔다.
- 지표: 정답이 빈 문자열인 시험 음성에서 글자를 출력한 건수(무음 출력 건수). 요약 표에 열을 추가한다.
- 엔진별 옵션 비교: Qwen3-ASR 은 언어를 지정하면(`QWEN_ASR_LANGUAGE=Korean`) 모델이 '말소리 없음'을 출력할 수 없다. 지정하지 않으면(`QWEN_ASR_LANGUAGE=`) 무음에서 빈 결과가 나온다. 두 조건의 문자 오류율과 무음 출력 건수를 함께 잰다.

---

## 3. 구현 구조 제안: 공통 래퍼

`Recognizer` 프로토콜을 바꾸지 않고, 모든 엔진을 감싸는 래퍼를 추가한다.

```python
# stt.py (또는 stt_guard.py)
class GuardedRecognizer:
    """모든 엔진 공통 전후처리: 꼬리 무음 제거(2.2), 짧은 단락 제거(2.4), 길이 검사(2.3), 힌트 반향 제거(2.1)."""

    def __init__(self, inner, quiet_rms, min_voiced_ms=400, trail_keep_ms=200,
                 max_syll_per_s=9.0, syll_slack=4, hint=ARS_HINT):
        self.inner = inner
        self.is_local = inner.is_local
        ...

    def __getattr__(self, name):          # warmup, prefetch 등 엔진 고유 기능은 그대로 노출
        return getattr(self.inner, name)

    async def recognize(self, ulaw8k):
        seg = trim_trailing_silence(ulaw8k, self.quiet_rms, self.trail_keep_ms)
        v = voiced_seconds(seg, self.quiet_rms)
        if v * 1000 < self.min_voiced_ms:
            return ""
        text = await self.inner.recognize(seg)
        if not text or is_hint_echo(text, self.hint) or too_long(text, v, ...):
            log.warning(...)
            return ""
        return text
```

연결
- `stt.make_recognizer` 가 `none` 을 제외한 모든 엔진을 `GuardedRecognizer` 로 감싸 반환한다.
- `app.build_deps` 는 `getattr(recognizer, "prefetch")`·`warmup` 을 쓰므로 `__getattr__` 위임으로 그대로 동작한다. 단, 미리 인식(`prefetch`)은 단락 일부를 넘기므로 래퍼 검사를 거치지 않는다. 미리 인식 결과는 최종 `recognize` 에서 합쳐진 뒤 검사된다.
- `session._stt_worker` 의 `is_local` 검사는 래퍼의 `is_local` 로 그대로 동작한다.

설정값 (`config.py` 또는 `.env`)

| 이름 | 초기값 | 근거 |
| --- | --- | --- |
| `STT_HINT` | 2.1 의 단어 나열 | 반향 위험 제거 |
| `STT_MIN_VOICED_MS` | 400 | 실측으로 조정 |
| `STT_TRAIL_KEEP_MS` | 200 | 마지막 음절 보존 |
| `STT_MAX_SYLL_PER_S` | 9.0 | 실측 99백분위로 조정 |
| `STT_SYLL_SLACK` | 4 | 짧은 단락 보호 |

---

## 4. 경우의 수 (구현 후 엣지케이스 점검표에 반영)

| 상황 | 기대 처리 | 시험 이름(제안) |
| --- | --- | --- |
| 무음만 있는 단락이 STT 로 감 | 인식하지 않음 (유음 시간 미달) | `test_guard_skips_silence_only_segment` |
| 0.1~0.4초 딸깍음·잡음 단락 | 인식하지 않음. 무응답 처리는 소리 감지 기준이므로 동작 변화 없음 | `test_guard_skips_short_noise_and_keeps_heard_flag` |
| 정상 안내 + 1.2초 꼬리 무음 | 꼬리를 200ms 만 남기고 인식 | `test_guard_trims_trailing_silence` |
| 엔진이 힌트 문장을 그대로 출력 | 버림, 경고 로그 | `test_guard_drops_hint_echo` |
| 엔진이 음성 길이보다 훨씬 긴 문장 출력 | 버림, 경고 로그 | `test_guard_drops_overlong_text` |
| 빠른 안내(초당 8음절)의 정상 문장 | 통과 | `test_guard_keeps_fast_but_plausible_text` |
| 미리 인식 조각이 앞 조각 끝과 같은 문장 반복 | 새 조각 결과를 버림 | `test_prefetch_drops_repeated_context` |
| Whisper 반복 출력 (compression_ratio > 2.4) | 해당 구간 버림 | `test_whisper_drops_repetitive_segment` (가짜 구간 객체) |
| "이번 통화는 녹음됩니다" | 번호로 바꾸지 않음 | 기존 `test_normalize_keeps_non_key_words` 에 추가 |
| "잔액조회는 이 번" | "잔액조회는 2번" | 기존 `test_normalize_spoken_numbers` 에 추가 |
| 외부 엔진 + 민감정보 되읽기 구간 | 전송하지 않음 (래퍼 적용 후에도 유지) | `test_external_engine_skips_secure_readback` (구현됨) |
| OpenAI 응답 지연 | 제한 시간 후 빈 결과 | `test_openai_timeout_returns_empty` |

---

## 5. 검증 계획

1. 단위 시험: 4장의 시험을 추가하고 `python -m pytest -q` 전체 통과를 확인한다 (`feat/stt-engines` 기준 전체 통과, 건수는 커밋 메시지 참고).
2. 녹음 평가 (실통화 없이 반복 가능)
   - 실통화 녹음을 단락 단위로 저장한다. 메뉴 사이 대기 구간을 자르지 않은 단락, 2초 이하 짧은 단락, 숫자 포함 단락, 무음·대기음만 있는 구간을 포함한다.
   - `python stt_bench.py --models <엔진들>`(저장소 시험 음성) 또는 `--wav <녹음>` 으로 변경 전후를 비교한다. 지표: CER, 메뉴 번호 정확도, 무음 구간 출력 건수(0 이어야 함), 단락당 처리 시간.
   - 설정값(3장 표)은 이 결과로 확정한다.
3. 실통화 확인: 건강보험공단·KB 시나리오 각 5회. 로그에서 다음을 센다.
   - KB 외 통화에 "KB국민은행" 이 인식된 횟수 (2.1 전후 비교)
   - 래퍼가 버린 단락 수와 그 원문 (정상 안내를 버린 경우가 있는지)

---

## 6. 미결정·실측 필요 항목

1. ARS 안내의 초당 음절 수 분포 (2.3 기준값)
2. 400ms 미만 정상 단락의 존재 여부 (2.4 기준값)
3. 힌트 축소가 메뉴 이름 인식률에 주는 영향 (2.1)
4. Silero VAD 가 8kHz 업샘플링 음성의 짧은 낱말을 자르는지 (2.5)
5. "이번" 규칙 변경을 해석기 담당과 합의 (2.6)
6. 리턴제로가 실시간보다 빠른 전송을 바로 처리하는지 (2.8)
7. Qwen3-ASR 언어 지정 여부에 따른 무음 출력 차이 (2.9)
