"""
후보 STT 엔진 (stt.Recognizer 프로토콜 구현)

STT 모델 비교용 엔진 모음. stt.make_recognizer 가 STT 값에 따라 선택.

  STT=rtzr    리턴제로 스트리밍 STT (RTZR_CLIENT_ID, RTZR_CLIENT_SECRET)
              통화 음성 8kHz 를 업샘플링 없이 전송, 통화 도메인(domain=CALL)으로 인식.
  STT=clova   네이버 CLOVA Speech 장문 인식, 동기 방식 (CLOVA_SPEECH_INVOKE_URL, CLOVA_SPEECH_SECRET)
  STT=gemini  Gemini 오디오 입력 (GEMINI_API_KEY, GEMINI_STT_MODEL)
  STT=qwen    Qwen3-ASR 로컬 인식 (QWEN_ASR_MODEL, QWEN_ASR_ADAPTER). 별도 패키지 설치 필요 (아래 QwenASRRecognizer)

공통 규칙 (stt.OpenAIRecognizer 와 동일)
  - 외부 엔진(rtzr·clova·gemini)은 is_local=False. 민감정보를 다시 읽어 주는 구간은 session.py 에서 전송 제외.
  - 인식 실패·시간 초과 시 예외 대신 빈 문자열 반환 (한 단락 실패로 인한 통화 중단 방지). 원인은 경고 로그로 기록.
  - 인식 결과에 stt._clean 적용 (환각 문장 제거·ARS 표기 교정).
  - 필수 키 누락 시 서버 시작 단계에서 RuntimeError (통화마다 빈 결과가 나오는 상황 방지).
  - 무음 환각·숫자 처리 같은 공통 후처리 강화는 인터페이스 밖 기능으로 이 모듈의 범위 외.

Whisper 계열 후보는 별도 코드 없이 stt.WhisperRecognizer 사용.
  WHISPER_MODEL=large-v3-turbo
  WHISPER_MODEL=<CTranslate2 로 변환한 batiai/batisay-ko-turbo 폴더>
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import time
import wave
from urllib.parse import urlencode

from clawops.agent._audio import ulaw_to_pcm16

from audio import ulaw8k_to_pcm16k, voiced_s
from stt import _clean, stt_timeout_s

log = logging.getLogger("ars.stt")

DEFAULT_TIMEOUT_S = 10.0  # 단락 하나의 기본 인식 제한 시간. 초과 시 해당 단락은 빈 결과 처리


def wav_bytes(pcm16: bytes, rate: int) -> bytes:
    """16bit mono PCM 의 WAV 파일 바이트 변환."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def _hide(msg: object, *secrets: str | None) -> str:
    """로그용 문자열에서 키 가림 (로그에 키가 남는 상황 방지)."""
    s = str(msg) or type(msg).__name__
    for sec in secrets:
        if sec:
            s = s.replace(sec, "[KEY]")
    return s[:300]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _env_str(name: str, default: str = "") -> str:
    """빈 값(.env 에 'KEY=' 로만 적은 경우)도 기본값으로 처리."""
    return os.environ.get(name, "").strip() or default


def _require(*names: str) -> list[str]:
    vals = [os.environ.get(n, "").strip() for n in names]
    missing = [n for n, v in zip(names, vals) if not v]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} 가 .env 에 없습니다. (STT={os.environ.get('STT', '')} 사용 시 필요)")
    return vals


# ---------------------------------------------------------------------------
# 리턴제로 (RTZR) 스트리밍 STT
# ---------------------------------------------------------------------------
_RTZR_KEYWORD = re.compile(r"^[가-힣 ]{1,20}(?::-?\d+(?:\.\d+)?)?$")


def rtzr_keywords(raw: str) -> str:
    """'별표,우물정자:2' 형식의 키워드 목록 검사. 형식에 맞지 않는 항목은 제외하고 경고 (연결 거부 방지).
    20자 이하·100개 이하·가중치 -5~5 기준은 리턴제로 연동 클라이언트(livekit-plugins-rtzr)와 동일.
    한글·공백만 허용하는 것은 공식 문서로 확인되지 않은 보수적 제한 (영문·숫자 키워드 필요 시 동작 확인 후 완화)."""
    out = []
    for item in (x.strip() for x in raw.split(",")):
        if not item:
            continue
        ok = bool(_RTZR_KEYWORD.match(item))
        if ok and ":" in item:
            ok = -5.0 <= float(item.rsplit(":", 1)[1]) <= 5.0
        if ok and len(out) < 100:
            out.append(item)
        else:
            log.warning("RTZR 키워드 제외 (한글·공백 20자 이하, 가중치 -5~5, 최대 100개): %s", item)
    return ",".join(out)


class RtzrRecognizer:
    """리턴제로 스트리밍 STT. 단락 전체를 100ms 단위로 바로 전송, EOS 전송 뒤 확정 문장(final) 수집.

    인증: POST /v1/authenticate (client_id, client_secret) → access_token, expire_at(초). 토큰은 만료 10분 전까지 재사용.
    연결: wss://openapi.vito.ai/v1/transcribe:streaming?model_name=sommers_ko&domain=CALL&sample_rate=8000&encoding=LINEAR16
          헤더 Authorization: bearer {token}
    응답: {"alternatives": [{"text": ...}], "final": true|false, ...}, 오류는 {"error": ...} 또는 {"type": "error", "message": ...}
    키워드 부스팅(RTZR_KEYWORDS)은 sommers_ko 모델에서만 동작.
    """

    is_local = False
    CHUNK = 1600  # 100ms (8kHz 16bit)

    def __init__(self, client_id: str, client_secret: str, *, model: str = "sommers_ko", domain: str = "CALL",
                 keywords: str = "", epd_time: float = 0.5, timeout_s: float = DEFAULT_TIMEOUT_S,
                 api_base: str = "https://openapi.vito.ai", ws_url: str | None = None, http=None) -> None:
        self.client_id, self.client_secret = client_id, client_secret
        self.timeout_s = timeout_s
        self.api_base = api_base.rstrip("/")
        self.ws_url = ws_url or "wss://" + self.api_base.split("://", 1)[1] + "/v1/transcribe:streaming"
        self.params = {"model_name": model, "domain": domain, "sample_rate": "8000", "encoding": "LINEAR16",
                       "epd_time": str(epd_time), "use_punctuation": "false"}
        kw = rtzr_keywords(keywords)
        if kw:
            self.params["keywords"] = kw
        self._http = http
        self._token: str | None = None
        self._expire_at = 0.0
        self._lock = asyncio.Lock()

    async def warmup(self) -> None:
        """서버 시작 시 토큰 미리 발급 (첫 단락 인식 지연 감소). 실패 시 인식 단계에서 재시도."""
        try:
            await self._get_token()
        except Exception as e:
            log.warning("RTZR 토큰 발급 실패: %s", _hide(e, self.client_secret))

    async def _get_token(self) -> str:
        async with self._lock:
            if self._token and self._expire_at - 600 > time.time():
                return self._token
            import httpx

            if self._http is None:
                self._http = httpx.AsyncClient(timeout=self.timeout_s)
            r = await self._http.post(f"{self.api_base}/v1/authenticate",
                                      data={"client_id": self.client_id, "client_secret": self.client_secret})
            r.raise_for_status()
            d = r.json()
            self._token, self._expire_at = str(d["access_token"]), float(d["expire_at"])
            return self._token

    async def recognize(self, ulaw8k: bytes) -> str:
        if not ulaw8k:
            return ""
        t0 = time.monotonic()
        try:
            # 응답 대기: _stream 안에서 제한 시간까지 받은 확정 문장으로 종료.
            # 토큰 발급·연결·전송 정지 대비: 전체에 조금 더 긴 제한 시간(+2초)을 한 번 더 적용
            text = await asyncio.wait_for(self._stream(ulaw_to_pcm16(ulaw8k), t0 + self.timeout_s),
                                          self.timeout_s + 2)
        except Exception as e:
            from websockets.exceptions import InvalidStatus

            if isinstance(e, InvalidStatus) and e.response.status_code == 401:
                self._token = None  # 만료·폐기된 토큰. 다음 단락에서 재발급
            log.warning("RTZR STT 실패: %s", _hide(e, self.client_secret, self._token))
            return ""
        text = _clean(text)
        log.info("인식(rtzr) %.2fs (음성 %.1fs) %s", time.monotonic() - t0, len(ulaw8k) / 8000, text)
        return text

    async def _stream(self, pcm: bytes, deadline: float) -> str:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosedOK

        token = await self._get_token()
        url = f"{self.ws_url}?{urlencode(self.params)}"
        finals: list[str] = []
        left = lambda: max(0.01, deadline - time.monotonic())  # noqa: E731
        async with connect(url, additional_headers={"Authorization": f"bearer {token}"},
                           open_timeout=left(), max_size=2 ** 20) as ws:
            for i in range(0, len(pcm), self.CHUNK):
                await ws.send(pcm[i:i + self.CHUNK])
            await ws.send("EOS")
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), left())
                except ConnectionClosedOK:
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    # 서버가 마지막 응답 뒤 연결을 닫지 않는 경우: 그때까지 받은 확정 문장 사용
                    log.warning("RTZR 응답 대기 시간 초과. 받은 문장 %d개 사용", len(finals))
                    break
                if isinstance(msg, bytes):
                    continue
                try:
                    d = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if "error" in d or d.get("type") == "error":
                    raise RuntimeError(f"RTZR 오류 응답: {d.get('error') or d.get('message')}")
                alts = d.get("alternatives") or []
                if d.get("final") and alts and (alts[0].get("text") or "").strip():
                    finals.append(alts[0]["text"].strip())
        return " ".join(finals)


# ---------------------------------------------------------------------------
# 네이버 CLOVA Speech (장문 인식, 로컬 파일 업로드, 동기)
# ---------------------------------------------------------------------------
class ClovaSpeechRecognizer:
    """CLOVA Speech 장문 인식 API 의 로컬 파일 인식(POST {Invoke URL}/recognizer/upload, completion=sync).

    헤더 X-CLOVASPEECH-API-KEY: 시크릿 키. multipart 필드 media(WAV), params(JSON), type=application/json.
    응답 {"result": "COMPLETED", "text": "...", "segments": [...]} 의 text 사용.
    CLOVA_SAMPLE_RATE: 8000(기본, 통화 원본 그대로) 또는 16000(업샘플링). 정확도 비교 후 선택.
    CLOVA_BOOSTINGS: '별표,우물정자,상담원' 처럼 쉼표로 구분한 부스팅 단어 (가중치 CLOVA_BOOST_WEIGHT, 1~5).
    """

    is_local = False

    def __init__(self, invoke_url: str, secret: str, *, sample_rate: int = 8000, boostings: str = "",
                 boost_weight: float = 2.0, timeout_s: float = DEFAULT_TIMEOUT_S, http=None) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("CLOVA_SAMPLE_RATE 는 8000 또는 16000")
        self.url = invoke_url.rstrip("/") + "/recognizer/upload"
        self.secret, self.sample_rate, self.timeout_s = secret, sample_rate, timeout_s
        self.params: dict = {"language": "ko-KR", "completion": "sync", "wordAlignment": False, "fullText": True,
                             "noiseFiltering": True, "diarization": {"enable": False}}
        words = ",".join(w.strip() for w in boostings.split(",") if w.strip())
        if words:
            self.params["boostings"] = [{"words": words, "weight": min(5.0, max(1.0, boost_weight))}]
        self._http = http

    async def recognize(self, ulaw8k: bytes) -> str:
        if not ulaw8k:
            return ""
        import httpx

        t0 = time.monotonic()
        pcm = ulaw_to_pcm16(ulaw8k) if self.sample_rate == 8000 else ulaw8k_to_pcm16k(ulaw8k)
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout_s)
        try:
            # httpx 의 timeout 은 연결·읽기 단계별 제한 → 전체 시간은 wait_for 로 제한
            r = await asyncio.wait_for(self._http.post(
                self.url, headers={"X-CLOVASPEECH-API-KEY": self.secret}, timeout=self.timeout_s,
                files={"media": ("ars.wav", wav_bytes(pcm, self.sample_rate), "audio/wav")},
                data={"params": json.dumps(self.params, ensure_ascii=False), "type": "application/json"}),
                self.timeout_s)
            r.raise_for_status()
            d = r.json()
            if d.get("result") != "COMPLETED":
                raise RuntimeError(f"CLOVA 결과 {d.get('result')}: {d.get('message')}")
            text = str(d.get("text") or "")
        except Exception as e:
            log.warning("CLOVA STT 실패: %s", _hide(e, self.secret))
            return ""
        text = _clean(text)
        log.info("인식(clova) %.2fs (음성 %.1fs) %s", time.monotonic() - t0, len(ulaw8k) / 8000, text)
        return text


# ---------------------------------------------------------------------------
# Gemini 오디오 입력
# ---------------------------------------------------------------------------
GEMINI_STT_SYSTEM = (
    "전화 ARS 안내 녹음을 들리는 그대로 한국어로 받아 적는다.\n"
    "- 들린 말만 적는다. 들리지 않은 내용을 추측하거나 덧붙이지 않는다. 요약·설명·번역을 하지 않는다.\n"
    "- 말소리가 없거나 대기 음악·잡음·신호음뿐이면 text 를 빈 문자열로 둔다.\n"
    "- 누르라고 안내한 번호는 아라비아 숫자로 적는다 (예: 1번, 0번). 별표·샵·우물정자는 들린 대로 적는다.\n"
    "- 전화번호·금액 등 다른 숫자도 들린 그대로 숫자로 적는다.")
_SCHEMA_TEXT = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


class GeminiAudioRecognizer:
    """Gemini generateContent 에 단락 음성(WAV, inlineData)을 넣어 받아쓰기.

    GEMINI_STT_MODEL (기본 GEMINI_MODEL 값), GEMINI_STT_THINKING (기본 MINIMAL. 빈 값이면 thinkingConfig 미전송,
    Gemini 2.5 계열처럼 thinkingLevel 을 받지 않는 모델용). 재시도 없음 (ARS 입력 대기 시간 안에 종료 필요).
    """

    is_local = False
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, *, thinking: str = "MINIMAL", timeout_s: float = DEFAULT_TIMEOUT_S,
                 url: str | None = None, http=None) -> None:
        self.api_key, self.model, self.thinking, self.timeout_s = api_key, model, thinking, timeout_s
        self.url = (url or self.URL).format(model=model)
        self._http = http
        self.interp = None  # attach_interpreter 로 연결 시 받아쓰기와 분류·점수를 한 번에 수신 (GEMINI_STT_COMBINED=1)

    def attach_interpreter(self, interp) -> None:
        """받아쓰기 + 분류·점수를 Gemini 호출 한 번으로 처리 (안내 종료 → 판단 지연을 호출 한 번만큼 단축).
        결과를 해석기 캐시에 넣으므로 session 이 바로 이어 부르는 classify 는 Gemini 재호출 없음.
        연결하지 않으면 받아쓰기만 수행, 해석기가 따로 Gemini 를 부르는 기본 구조 유지."""
        self.interp = interp

    def combined_body(self, ulaw8k: bytes) -> dict | None:
        """해석기가 연결되어 있고 목적을 알면 받아쓰기+분류 요청 본문, 아니면 None (일반 받아쓰기)."""
        interp = self.interp
        if interp is None or not getattr(interp, "goal", ""):
            return None
        system, schema, blocks = interp.audio_request()
        audio = base64.b64encode(wav_bytes(ulaw8k_to_pcm16k(ulaw8k), 16000)).decode("ascii")
        gen: dict = {"temperature": 0, "maxOutputTokens": 1024, "responseMimeType": "application/json",
                     "responseJsonSchema": schema}
        if self.thinking:
            gen["thinkingConfig"] = {"thinkingLevel": self.thinking}
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": blocks}, {"inlineData": {"mimeType": "audio/wav", "data": audio}}]}],
                "generationConfig": gen}

    def body(self, ulaw8k: bytes) -> dict:
        audio = base64.b64encode(wav_bytes(ulaw8k_to_pcm16k(ulaw8k), 16000)).decode("ascii")
        gen: dict = {"temperature": 0, "maxOutputTokens": 512, "responseMimeType": "application/json",
                     "responseJsonSchema": _SCHEMA_TEXT}
        if self.thinking:
            gen["thinkingConfig"] = {"thinkingLevel": self.thinking}
        return {"systemInstruction": {"parts": [{"text": GEMINI_STT_SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": "이 녹음을 받아 적어 주세요."},
                                                        {"inlineData": {"mimeType": "audio/wav", "data": audio}}]}],
                "generationConfig": gen}

    async def recognize(self, ulaw8k: bytes) -> str:
        if not ulaw8k:
            return ""
        import httpx

        t0 = time.monotonic()
        if self._http is None:
            self._http = httpx.AsyncClient(headers={"x-goog-api-key": self.api_key}, timeout=self.timeout_s)
        combined = self.combined_body(ulaw8k)
        try:
            r = await asyncio.wait_for(self._http.post(self.url, json=combined or self.body(ulaw8k), timeout=self.timeout_s),
                                       self.timeout_s)  # 전체 응답 시간 제한 (httpx timeout 은 단계별)
            r.raise_for_status()
            cand = (r.json().get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            if not raw:
                raise RuntimeError(f"빈 응답 (finishReason={cand.get('finishReason')})")
            d = json.loads(raw)
            if combined:
                text = _clean(str(d.pop("transcript", "") or ""))
                if text:
                    self.interp.prime(text, d)  # 분류·점수를 캐시에 저장. 이어지는 classify 는 Gemini 재호출 없음
            else:
                text = _clean(str(d.get("text") or ""))
        except Exception as e:
            log.warning("Gemini STT 실패: %s", _hide(e, self.api_key))
            return ""
        log.info("인식(gemini%s) %.2fs (음성 %.1fs) %s", "+분류" if combined else "", time.monotonic() - t0,
                 len(ulaw8k) / 8000, text)
        return text


# ---------------------------------------------------------------------------
# Qwen3-ASR 로컬
# ---------------------------------------------------------------------------
class QwenASRRecognizer:
    """Qwen3-ASR 로컬 인식. 오디오가 서버 밖으로 나가지 않으므로 is_local=True.

    로드 방식(backend)은 모델 형식별로 다름. 두 방식의 transformers 버전 충돌로 가상환경 분리 필요.
      qwen-asr  원본 형식 체크포인트 (기본 TeamUNIVA/qwen3_asr_1.7b_ko).
                pip install qwen-asr torch  (qwen-asr 가 transformers==4.57.6 을 고정)
      hf        transformers 기본 형식(-hf) 체크포인트와 LoRA 어댑터 (예: Qwen/Qwen3-ASR-1.7B-hf
                + Playedwell03/qwen3-asr-1.7b-119ko-v5). pip install "transformers>=5.13" torch (어댑터 사용 시 peft 추가)
    필요한 패키지가 없으면 서버 시작 시 RuntimeError (설치 명령 포함).
      server    별도로 띄운 vLLM 서버(OpenAI 음성 인식 API /v1/audio/transcriptions)로 전송. 서버 가상환경에
                torch 불필요. QWEN_ASR_URL (기본 http://127.0.0.1:8001/v1), 모델 이름은 vLLM 에 올린 이름(QWEN_ASR_MODEL).
                vLLM 은 Windows 미지원 → WSL2 에서 실행.
                주소가 루프백(127.0.0.1·localhost·::1)이 아니면 외부 STT 로 간주, is_local=False (민감정보 재확인 구간 미전송).
    QWEN_ASR_ADAPTER 지정 또는 모델 이름이 -hf 로 끝나면 hf, 그 밖에는 qwen-asr 로 로드 (QWEN_ASR_BACKEND 로 지정 가능).
    QWEN_ASR_CONTEXT: 어휘 힌트(시스템 프롬프트). 기본값 없음. 119 LoRA 는 힌트 사용 시 정확도 저하가 모델 카드에 명시됨.
    QWEN_ASR_LANGUAGE: 기본 Korean (모델 카드 평가와 같은 조건). 비우면 언어 미지정. 이때 말소리가 없으면 모델이
      'language None' 을 출력해 빈 결과. 언어 지정 시 이 출력이 막혀 무음에서도 문장 생성 가능 →
      무음 환각 비교 시 두 조건 모두 확인 필요.
    WHISPER_DEVICE 와 같은 장치 설정(cpu/cuda) 사용. 1.7B 모델은 CPU 전용 소형 기기에서 실시간 인식 곤란.
    """

    is_local = True

    def __init__(self, model: str = "TeamUNIVA/qwen3_asr_1.7b_ko", *, adapter: str = "", device: str = "cpu",
                 backend: str = "", context: str = "", language: str | None = "Korean",
                 max_new_tokens: int = 256, quiet_rms: float = 500, url: str = "", timeout_s: float = 10.0) -> None:
        self.model_name, self.adapter, self.device, self.context = model, adapter, device, context
        self.language = language or None  # None: 자동 판별 (무음이면 빈 결과)
        self.backend = backend or ("hf" if adapter or model.endswith("-hf") else "qwen-asr")
        if self.backend not in ("hf", "qwen-asr", "server"):
            raise ValueError("QWEN_ASR_BACKEND 는 hf, qwen-asr, server 중 하나")
        if adapter and self.backend != "hf":
            raise ValueError("LoRA 어댑터(QWEN_ASR_ADAPTER)는 hf 방식에서만 쓸 수 있습니다")
        self.max_new_tokens = max_new_tokens
        self.quiet_rms = quiet_rms
        self.url = (url or "http://127.0.0.1:8001/v1").rstrip("/")
        self.timeout_s = timeout_s
        if self.backend == "server":
            from urllib.parse import urlsplit

            # 루프백 주소의 서버만 로컬로 간주. 다른 컴퓨터로 보내면 외부 STT 와 동일하게 처리 (민감정보 재확인 구간 미전송)
            self.is_local = (urlsplit(self.url).hostname or "") in ("127.0.0.1", "localhost", "::1")
        self._model = None
        self._processor = None
        self._lock = asyncio.Lock()
        self._prefix: list[tuple[int, bytes, str]] = []  # 미리 인식 결과 (끝 위치, 앞부분 해시, 누적 문장)
        self._prefetch_failed = False  # 이 단락의 미리 인식 실패 여부 (단락 인식이 끝나면 초기화)

    @staticmethod
    def missing_packages(backend: str, adapter: str = "") -> list[str]:
        """설치되지 않은 패키지 목록 (모듈을 불러오지 않고 확인). 서버 시작 시 make_engine 에서 사용."""
        import importlib.util
        from importlib.metadata import PackageNotFoundError, version

        if backend == "server":
            return [m for m in ("httpx",) if importlib.util.find_spec(m) is None]
        need = ["torch"] + (["qwen_asr"] if backend == "qwen-asr" else ["transformers"]) + (["peft"] if adapter else [])
        missing = [m for m in need if importlib.util.find_spec(m) is None]
        if backend == "hf" and "transformers" not in missing:
            try:  # Qwen3ASRForConditionalGeneration 은 transformers 5.13 부터 포함
                major, minor = (int(x) for x in version("transformers").split(".")[:2])
                if (major, minor) < (5, 13):
                    missing.append("transformers>=5.13")
            except (PackageNotFoundError, ValueError):
                pass
        return missing

    @staticmethod
    def _dtype(torch, device: str):
        """cuda 장치는 bfloat16(미지원 시 float16), 그 밖의 장치는 float32."""
        if not device.startswith("cuda"):
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _load(self):
        if self._model is not None:
            return self._model
        if self.backend == "server":
            import httpx

            # 루프백 주소의 서버로 보낼 때는 시스템·환경 프록시 미사용 (되읽기 음성의 프록시 경유 방지)
            self._model = httpx.Client(timeout=self.timeout_s, trust_env=not self.is_local)
            return self._model
        import torch

        dtype = self._dtype(torch, self.device)
        log.info("Qwen3-ASR 모델 로드: %s%s (%s, %s)", self.model_name,
                 f" + {self.adapter}" if self.adapter else "", self.backend, self.device)
        if self.backend == "qwen-asr":
            from qwen_asr import Qwen3ASRModel

            self._model = Qwen3ASRModel.from_pretrained(self.model_name, dtype=dtype, device_map=self.device,
                                                        max_inference_batch_size=1, max_new_tokens=self.max_new_tokens)
        else:
            from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            # device_map 은 accelerate 필요 → 사용하지 않고 단일 장치로 이동
            m = Qwen3ASRForConditionalGeneration.from_pretrained(self.model_name, dtype=dtype).to(self.device)
            self._device, self._dtype = m.device, m.dtype  # PEFT 로 감싸기 전 값 (입력 텐서 이동용)
            if self.adapter:
                from peft import PeftModel

                m = PeftModel.from_pretrained(m, self.adapter)
            m.eval()
            self._model = m
        return self._model

    def _run(self, ulaw8k: bytes) -> str:
        import numpy as np

        model = self._load()
        if self.backend == "server":
            return self._run_server(model, ulaw8k)
        pcm = np.frombuffer(ulaw8k_to_pcm16k(ulaw8k), dtype=np.int16).astype(np.float32) / 32768.0
        if self.backend == "qwen-asr":
            out = model.transcribe(audio=(pcm, 16000), context=self.context, language=self.language)
            return out[0].text if out else ""
        import torch

        proc = self._processor
        inputs = proc.apply_transcription_request(audio=pcm, language=self.language, prompt=self.context or None)
        inputs = inputs.to(self._device, self._dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen = ids[:, inputs["input_ids"].shape[1]:]
        return proc.decode(gen, return_format="transcription_only")[0]

    def _run_server(self, http, ulaw8k: bytes) -> str:
        """vLLM OpenAI 음성 인식 API 호출. 16kHz WAV 로 전송. 실패 시 예외 (recognize 가 빈 결과·앞부분으로 대체)."""
        data = {"model": self.model_name, "temperature": "0", "response_format": "json"}
        lang = {"korean": "ko", "english": "en"}.get((self.language or "").lower())
        if lang:
            data["language"] = lang
        if self.context:
            data["prompt"] = self.context
        r = http.post(f"{self.url}/audio/transcriptions", data=data,
                      files={"file": ("audio.wav", wav_bytes(ulaw8k_to_pcm16k(ulaw8k), 16000), "audio/wav")})
        r.raise_for_status()
        text = r.json().get("text", "")
        # 언어 미지정 시 Qwen3-ASR 출력 형식은 'language Korean<asr_text>…' → 본문만 추출
        return text.split("<asr_text>", 1)[1] if "<asr_text>" in text else text

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)
        if self.backend == "server":
            try:  # 서버 실행 여부 확인, 첫 요청 준비 시간을 시작 단계에서 미리 소모
                await asyncio.to_thread(self._run, bytes([0xFF]) * 8000)  # 1초 무음 (μ-law 0xFF)
                log.info("Qwen3-ASR 서버 연결 확인: %s (%s)", self.url, self.model_name)
            except Exception as e:
                log.warning("Qwen3-ASR 서버(%s)에 연결하지 못함: %s — 서버를 먼저 띄울 것", self.url, _hide(e))

    # 미리 인식 (STT_PREFETCH=1) ---------------------------------------------
    # stt.WhisperRecognizer 와 같은 방식: ARS 안내 도중 Segmenter 가 단락 안의 쉼(0.3초)마다 그때까지의 음성을 넘기면
    # 새 조각만 인식해 앞 조각 문장에 이어 붙여 보관. 단락 종료 시 남은 꼬리만 인식 → 단락 종료 뒤 인식 시간 단축.
    # 앞부분 일치는 오디오 바이트의 해시로 확인. 조각 경계의 낱말 잘림 방지를 위해 Segmenter 는 쉼에서만 전달.
    def _cached(self, ulaw8k: bytes) -> tuple[int, str]:
        """ulaw8k 의 앞부분과 일치하는 가장 긴 미리 인식 결과 (길이, 문장)."""
        best = (0, "")
        for end, digest, text in self._prefix:
            if best[0] < end <= len(ulaw8k) and hashlib.sha1(ulaw8k[:end]).digest() == digest:
                best = (end, text)
        return best

    def _run_piece(self, piece: bytes) -> str:
        """조각 하나 인식. 말소리가 0.1초 미만인 조각(잡음·딸깍음)은 모델 입력 제외 (무음에서 '응' 같은 글자를 지어내는 현상 방지).
        평균 RMS 대신 20ms 프레임 단위 유음 길이 기준 (짧은 말 + 쉼이 평균에 묻혀 빠지는 현상 방지)."""
        if voiced_s(piece, self.quiet_rms) < 0.1:
            return ""
        return _clean(self._run(piece), piece=True)

    async def prefetch(self, prefix: bytes, start: int) -> None:
        if not self.is_local:
            return  # 외부로 보내는 인식기는 미리 인식 없음 (민감정보 되읽기 구간 판정은 단락 단위로만 수행)
        async with self._lock:
            if self._prefetch_failed:
                return  # 이 단락에서 이미 실패 → 남은 조각마다 시간 초과를 기다리지 않음
            done, prev = self._cached(prefix)
            if done >= len(prefix):
                return
            if done < start:  # 앞 조각 없음 (단락 시작 직후 또는 앞 조각 인식 실패) → 처음부터 인식
                done, prev = 0, ""
            t0 = time.monotonic()
            try:
                text = await asyncio.to_thread(self._run_piece, prefix[done:])
            except Exception as e:
                log.warning("Qwen3-ASR 미리 인식 실패: %s", _hide(e))
                self._prefetch_failed = True
                return
            cum = f"{prev} {text}".strip()
            self._prefix = self._prefix[-15:] + [(len(prefix), hashlib.sha1(prefix).digest(), cum)]
            log.debug("미리 인식(qwen) %.2fs (%.1f~%.1fs) %s", time.monotonic() - t0, done / 8000, len(prefix) / 8000, text)

    async def recognize(self, ulaw8k: bytes) -> str:
        if not ulaw8k:
            return ""
        t0 = time.monotonic()
        async with self._lock:  # 모델 하나를 순차 사용 (진행 중인 미리 인식 종료까지 대기)
            done, prev = self._cached(ulaw8k)
            rest = ulaw8k[done:]
            try:
                if not done:
                    text = await asyncio.to_thread(self._run, ulaw8k)  # 미리 인식 결과가 없으면 단락 전체 인식
                elif voiced_s(rest, self.quiet_rms) < 0.1:
                    text = prev  # 남은 부분은 단락 끝 무음뿐
                else:
                    text = f"{prev} {await asyncio.to_thread(self._run_piece, rest)}".strip()
            except Exception as e:
                log.warning("Qwen3-ASR 인식 실패: %s", _hide(e))
                text = prev  # 꼬리 인식만 실패한 경우 미리 인식한 앞부분 유지
            finally:
                self._prefix.clear()
                self._prefetch_failed = False
        text = _clean(text)
        log.info("인식(qwen) %.2fs (음성 %.1fs, 미리 인식 %.1fs 사용) %s", time.monotonic() - t0, len(ulaw8k) / 8000,
                 done / 8000, text)
        return text


# ---------------------------------------------------------------------------
# 음성 입력 LLM (Qwen3-Omni 등) — 받아쓰기와 분류·점수를 한 번에 처리
# ---------------------------------------------------------------------------
class OmniRecognizer:
    """음성을 직접 듣는 LLM(예: Qwen3-Omni-30B-A3B)을 llama.cpp llama-server 의 OpenAI 호환 /v1/chat/completions 로 호출.

    해석기가 연결되어 있고(app.py 에서 STT=omni 이면 연결) 목적을 알면, 녹음과 함께 해석기 지시문·스키마를 보내 받아쓰기(transcript)와
    분류·점수를 한 번에 수신. 결과는 해석기 캐시에 저장(prime) → 이어지는 classify 의 모델 재호출 없음 (GeminiAudioRecognizer
    결합 모드와 같은 구조). 목적을 모르면 받아쓰기만 수행. 같은 서버를 해석기(LOCAL_LLM_API=openai)로도 쓰면 STT·해석을 모델 하나로 처리.
      OMNI_URL    기본 http://127.0.0.1:8080/v1 (llama-server --port 8080). 루프백 주소가 아니면 is_local=False (되읽기 구간 미전송)
      OMNI_MODEL  요청에 넣을 모델 이름 (llama-server 는 올린 모델 하나만 쓰므로 표시용)
    음성은 16kHz WAV(base64)를 input_audio 로 전송. 미리 인식(쉼마다 앞부분 인식) 없음: 분류에 단락 전체 필요.
    받아쓰기를 분류보다 먼저 생성하도록 스키마의 transcript 를 맨 앞에 배치 (문법 제한 디코딩은 속성 순서대로 생성).
    """

    def __init__(self, url: str = "", model: str = "qwen3-omni", *, timeout_s: float = DEFAULT_TIMEOUT_S, compact: bool = True,
                 quiet_rms: float = 500, http=None) -> None:
        from urllib.parse import urlsplit

        self.url = (url or "http://127.0.0.1:8080/v1").rstrip("/")
        self.model, self.timeout_s, self.compact, self.quiet_rms = model, timeout_s, compact, quiet_rms
        self.is_local = (urlsplit(self.url).hostname or "") in ("127.0.0.1", "localhost", "::1")
        self._http = http
        self.interp = None

    PLAIN_TEXT = "이 녹음을 받아 적어 주세요."

    def attach_interpreter(self, interp) -> None:
        self.interp = interp

    def request(self, ulaw8k: bytes) -> tuple[dict, bool]:
        """(요청 본문, 받아쓰기+분류 결합 여부)."""
        interp = self.interp
        combined = interp is not None and bool(getattr(interp, "goal", "")) and hasattr(interp, "audio_request")
        if combined:
            system, schema, text = interp.audio_request()
            props = dict(schema["properties"])
            schema = {"type": "object", "properties": {"transcript": props.pop("transcript", {"type": "string"}), **props},
                      "required": ["transcript", *[k for k in schema["required"] if k != "transcript"]]}
        else:
            # 말소리 없는 잡음 조각에서 모델이 이 지시문을 그대로 받아 적는 경우 있음 → recognize 에서 제거.
            # 지시문 생략 시에는 없는 안내 문장을 지어내는 경우가 있어 지시문 유지
            system, schema, text = GEMINI_STT_SYSTEM, _SCHEMA_TEXT, self.PLAIN_TEXT
        if self.compact:
            system += "\n- JSON 은 공백과 줄바꿈 없이 한 줄로 씁니다."
        audio = base64.b64encode(wav_bytes(ulaw8k_to_pcm16k(ulaw8k), 16000)).decode("ascii")
        parts = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}]
        body = {"model": self.model, "temperature": 0, "max_tokens": 1024,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": parts}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "result", "strict": True, "schema": schema}}}
        return body, combined

    async def warmup(self) -> None:
        try:  # 서버 실행 여부 확인 및 음성 처리 경로 예열 (1초 무음)
            await self._post(bytes([0xFF]) * 8000)
            log.info("음성 LLM 서버 연결 확인: %s", self.url)
        except Exception as e:
            log.warning("음성 LLM 서버(%s)에 연결하지 못함: %s — llama-server 를 먼저 띄울 것", self.url, _hide(e))

    async def _post(self, ulaw8k: bytes) -> tuple[dict, bool]:
        import httpx

        if self._http is None:
            # 루프백 주소의 서버로 보낼 때는 시스템·환경 프록시 미사용 (되읽기 음성의 프록시 경유 방지)
            self._http = httpx.AsyncClient(timeout=self.timeout_s, trust_env=not self.is_local)
        body, combined = self.request(ulaw8k)
        r = await asyncio.wait_for(self._http.post(f"{self.url}/chat/completions", json=body, timeout=self.timeout_s),
                                   self.timeout_s)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        raw = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if not raw.strip():
            raise RuntimeError(f"빈 응답 (finish_reason={(data.get('choices') or [{}])[0].get('finish_reason')})")
        from local_llm import parse_json

        return parse_json(raw), combined

    async def recognize(self, ulaw8k: bytes) -> str:
        if not ulaw8k or voiced_s(ulaw8k, self.quiet_rms) < 0.1:
            return ""
        t0 = time.monotonic()
        try:
            d, combined = await self._post(ulaw8k)
            if combined:
                text = _clean(str(d.pop("transcript", "") or ""))
                if text:
                    self.interp.prime(text, d)  # 이어지는 classify 는 모델 재호출 없음
            else:
                text = _clean(str(d.get("text") or ""))
                if re.sub(r"\W", "", text) == re.sub(r"\W", "", self.PLAIN_TEXT):
                    text = ""  # 말소리 없는 조각에서 지시문을 따라 적은 경우
        except Exception as e:
            log.warning("음성 LLM 인식 실패: %s", _hide(e))
            return ""
        log.info("인식(omni%s) %.2fs (음성 %.1fs) %s", "+분류" if combined else "", time.monotonic() - t0,
                 len(ulaw8k) / 8000, text)
        return text


# ---------------------------------------------------------------------------
KINDS = ("rtzr", "clova", "gemini", "qwen", "omni")


def make_engine(kind: str, device: str = "cpu"):
    """.env 값으로 후보 엔진 생성. stt.make_recognizer 에서 호출."""
    timeout = stt_timeout_s()
    if kind == "rtzr":
        cid, secret = _require("RTZR_CLIENT_ID", "RTZR_CLIENT_SECRET")
        return RtzrRecognizer(cid, secret, model=_env_str("RTZR_MODEL", "sommers_ko"),
                              domain=_env_str("RTZR_DOMAIN", "CALL"), keywords=_env_str("RTZR_KEYWORDS"),
                              epd_time=_env_float("RTZR_EPD_TIME", 0.5), timeout_s=timeout)
    if kind == "clova":
        url, secret = _require("CLOVA_SPEECH_INVOKE_URL", "CLOVA_SPEECH_SECRET")
        return ClovaSpeechRecognizer(url, secret, sample_rate=int(_env_str("CLOVA_SAMPLE_RATE", "8000")),
                                     boostings=_env_str("CLOVA_BOOSTINGS"),
                                     boost_weight=_env_float("CLOVA_BOOST_WEIGHT", 2.0), timeout_s=timeout)
    if kind == "gemini":
        (key,) = _require("GEMINI_API_KEY")
        model = _env_str("GEMINI_STT_MODEL") or _env_str("GEMINI_MODEL", "gemini-3.5-flash-lite")
        # 'GEMINI_STT_THINKING=' 처럼 비워 두면 thinkingConfig 미전송 (2.5 계열 모델용)
        thinking = os.environ.get("GEMINI_STT_THINKING", "MINIMAL").strip()
        return GeminiAudioRecognizer(key, model, thinking=thinking, timeout_s=timeout)
    if kind == "qwen":
        # 'QWEN_ASR_LANGUAGE=' 처럼 비워 두면 언어 미지정 (변수가 없으면 Korean)
        rec = QwenASRRecognizer(_env_str("QWEN_ASR_MODEL", "TeamUNIVA/qwen3_asr_1.7b_ko"),
                                adapter=_env_str("QWEN_ASR_ADAPTER"), device=device,
                                backend=_env_str("QWEN_ASR_BACKEND"), context=_env_str("QWEN_ASR_CONTEXT"),
                                language=os.environ.get("QWEN_ASR_LANGUAGE", "Korean").strip(),
                                quiet_rms=_env_float("VAD_RMS", 500), url=_env_str("QWEN_ASR_URL"), timeout_s=timeout)
        missing = rec.missing_packages(rec.backend, rec.adapter)
        if missing:
            cmd = ("pip install qwen-asr torch" if rec.backend == "qwen-asr" else "pip install httpx" if rec.backend == "server"
                   else 'pip install "transformers>=5.13" torch' + (" peft" if rec.adapter else ""))
            raise RuntimeError(f"STT=qwen ({rec.backend} 방식)에 필요한 패키지가 없습니다: {', '.join(missing)}. "
                               f"설치: {cmd} (qwen-asr 와 transformers 5 는 가상환경을 나눈다)")
        return rec
    if kind == "omni":
        return OmniRecognizer(_env_str("OMNI_URL"), _env_str("OMNI_MODEL", "qwen3-omni"), timeout_s=timeout,
                              compact=os.environ.get("LOCAL_LLM_COMPACT", "1").strip() != "0",
                              quiet_rms=_env_float("VAD_RMS", 500))
    raise ValueError(f"알 수 없는 STT: {kind}")
