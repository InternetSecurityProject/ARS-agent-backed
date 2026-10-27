"""
후보 STT 엔진 (stt.Recognizer 프로토콜 구현)

STT 모델 비교를 위해 추가한 엔진이다. stt.make_recognizer 가 STT 값으로 고른다.

  STT=rtzr    리턴제로 스트리밍 STT (RTZR_CLIENT_ID, RTZR_CLIENT_SECRET)
              통화 음성 8kHz 를 업샘플링 없이 전송하고 통화 도메인(domain=CALL)으로 인식한다.
  STT=clova   네이버 CLOVA Speech 장문 인식, 동기 방식 (CLOVA_SPEECH_INVOKE_URL, CLOVA_SPEECH_SECRET)
  STT=gemini  Gemini 오디오 입력 (GEMINI_API_KEY, GEMINI_STT_MODEL)
  STT=qwen    Qwen3-ASR 로컬 인식 (QWEN_ASR_MODEL, QWEN_ASR_ADAPTER). 별도 패키지 설치 필요 (아래 QwenASRRecognizer)

공통 규칙 (stt.OpenAIRecognizer 와 같다)
  - 외부 엔진(rtzr·clova·gemini)은 is_local=False 이다. 민감정보를 다시 읽어 주는 구간은 session.py 가 보내지 않는다.
  - 인식 실패·시간 초과는 예외 대신 빈 문자열을 돌려준다 (한 단락 실패로 통화가 멈추지 않게). 원인은 경고 로그로 남긴다.
  - 인식 결과는 stt._clean 으로 환각 문장 제거·ARS 표기 교정을 거친다.
  - 필수 키가 없으면 서버 시작 시 RuntimeError 로 바로 알린다 (통화마다 빈 결과가 나오는 것을 막는다).
  - 무음 환각·숫자 처리 같은 공통 후처리 강화는 인터페이스 밖의 신규 기능이므로 여기서 다루지 않는다
    (설계: docs/ARS자동화-STT-무음환각-숫자처리-설계안.md).

Whisper 계열 후보는 새 코드 없이 기존 WhisperRecognizer 로 시험한다.
  WHISPER_MODEL=large-v3-turbo
  WHISPER_MODEL=<CTranslate2 로 변환한 batiai/batisay-ko-turbo 폴더>
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
import wave
from urllib.parse import urlencode

from clawops.agent._audio import ulaw_to_pcm16

from audio import ulaw8k_to_pcm16k
from stt import _clean, stt_timeout_s

log = logging.getLogger("ars.stt")

DEFAULT_TIMEOUT_S = 10.0  # 단락 하나의 인식 제한 시간. 넘으면 그 단락은 빈 결과로 처리한다


def wav_bytes(pcm16: bytes, rate: int) -> bytes:
    """16bit mono PCM 을 WAV 파일 바이트로 감싼다."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def _hide(msg: object, *secrets: str | None) -> str:
    """로그에 키가 남지 않게 가린다."""
    s = str(msg) or type(msg).__name__
    for sec in secrets:
        if sec:
            s = s.replace(sec, "[KEY]")
    return s[:300]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _env_str(name: str, default: str = "") -> str:
    """빈 값(.env 에 'KEY=' 로만 적은 경우)도 기본값으로 본다."""
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
    """'별표,우물정자:2' 형식의 키워드 목록을 검사한다. 맞지 않는 항목은 버리고 경고한다 (연결 거부 방지).
    20자 이하·100개 이하·가중치 -5~5 는 리턴제로 연동 클라이언트(livekit-plugins-rtzr)와 같은 규칙이다.
    한글·공백만 허용하는 것은 공식 문서로 확인하지 못해 보수적으로 둔 제한이다 (영문·숫자 키워드가 필요하면 실측 후 완화)."""
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
    """리턴제로 스트리밍 STT. 단락 전체를 100ms 단위로 바로 보내고 EOS 를 보낸 뒤 확정 문장(final)을 모은다.

    인증: POST /v1/authenticate (client_id, client_secret) → access_token, expire_at(초). 토큰은 만료 10분 전까지 재사용한다.
    연결: wss://openapi.vito.ai/v1/transcribe:streaming?model_name=sommers_ko&domain=CALL&sample_rate=8000&encoding=LINEAR16
          헤더 Authorization: bearer {token}
    응답: {"alternatives": [{"text": ...}], "final": true|false, ...}, 오류는 {"error": ...} 또는 {"type": "error", "message": ...}
    키워드 부스팅(RTZR_KEYWORDS)은 sommers_ko 모델에서만 동작한다.
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
        """서버 시작 시 토큰을 미리 받아 둔다 (첫 단락 인식 지연 감소). 실패해도 인식 시 다시 시도한다."""
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
            # 응답 대기는 _stream 안에서 제한 시간까지 받은 확정 문장을 쓰고 끝낸다.
            # 토큰 발급·연결·전송이 멈추는 경우까지 막기 위해 전체를 조금 더 긴 시간으로 한 번 더 제한한다
            text = await asyncio.wait_for(self._stream(ulaw_to_pcm16(ulaw8k), t0 + self.timeout_s),
                                          self.timeout_s + 2)
        except Exception as e:
            from websockets.exceptions import InvalidStatus

            if isinstance(e, InvalidStatus) and e.response.status_code == 401:
                self._token = None  # 만료·폐기된 토큰. 다음 단락에서 다시 받는다
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
                    # 서버가 마지막 응답 뒤 연결을 닫지 않는 경우: 그때까지 받은 확정 문장을 쓴다
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
    응답 {"result": "COMPLETED", "text": "...", "segments": [...]} 의 text 를 쓴다.
    CLOVA_SAMPLE_RATE: 8000(기본, 통화 원본 그대로) 또는 16000(업샘플링). 어느 쪽이 정확한지는 실측으로 정한다.
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
            # httpx 의 timeout 은 연결·읽기 단계별 제한이므로 전체 시간은 wait_for 로 제한한다
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
    """Gemini generateContent 에 단락 음성(WAV, inlineData)을 넣어 받아쓴다.

    GEMINI_STT_MODEL (기본 GEMINI_MODEL 값), GEMINI_STT_THINKING (기본 MINIMAL. 빈 값이면 thinkingConfig 를 보내지 않는다.
    Gemini 2.5 계열처럼 thinkingLevel 을 받지 않는 모델에 쓴다). 재시도하지 않는다 (ARS 입력 대기 시간 안에 끝나야 한다).
    """

    is_local = False
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, *, thinking: str = "MINIMAL", timeout_s: float = DEFAULT_TIMEOUT_S,
                 url: str | None = None, http=None) -> None:
        self.api_key, self.model, self.thinking, self.timeout_s = api_key, model, thinking, timeout_s
        self.url = (url or self.URL).format(model=model)
        self._http = http
        self.interp = None  # attach_interpreter 로 연결되면 받아쓰기와 분류·점수를 한 번에 받는다 (GEMINI_STT_COMBINED=1)

    def attach_interpreter(self, interp) -> None:
        """받아쓰기 + 분류·점수를 Gemini 한 번으로 처리한다 (안내 종료 → 판단 지연을 호출 한 번만큼 줄인다).
        결과는 해석기 캐시에 넣어 두므로 session 이 바로 이어 부르는 classify 는 Gemini 를 다시 부르지 않는다.
        끄면(연결하지 않으면) 받아쓰기만 하고 해석기가 따로 Gemini 를 부르는 원래 구조 그대로다."""
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
                    self.interp.prime(text, d)  # 분류·점수를 캐시에 넣는다. 이어지는 classify 는 Gemini 를 다시 부르지 않는다
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

    로드 방식(backend)은 모델 형식에 따라 다르며, 두 방식의 transformers 버전이 서로 충돌하므로 가상환경을 나눈다.
      qwen-asr  원본 형식 체크포인트 (기본 TeamUNIVA/qwen3_asr_1.7b_ko).
                pip install qwen-asr torch  (qwen-asr 가 transformers==4.57.6 을 고정)
      hf        transformers 기본 형식(-hf) 체크포인트와 LoRA 어댑터 (예: Qwen/Qwen3-ASR-1.7B-hf
                + Playedwell03/qwen3-asr-1.7b-119ko-v5). pip install "transformers>=5.13" torch (어댑터를 쓰면 peft 추가)
    필요한 패키지가 없으면 서버 시작 시 RuntimeError 로 알린다 (설치 명령 포함).
    QWEN_ASR_ADAPTER 를 지정하거나 모델 이름이 -hf 로 끝나면 hf, 그 밖에는 qwen-asr 로 로드한다 (QWEN_ASR_BACKEND 로 지정 가능).
    QWEN_ASR_CONTEXT: 어휘 힌트(시스템 프롬프트). 기본은 없음. 119 LoRA 는 힌트를 주면 정확도가 떨어진다고 모델 카드에 명시되어 있다.
    QWEN_ASR_LANGUAGE: 기본 Korean (모델 카드 평가와 같은 조건). 비우면 언어를 지정하지 않는다. 이때 모델은 말소리가 없으면
      'language None' 을 출력하고 빈 결과가 된다. 언어를 지정하면 이 출력이 막혀 무음에서도 문장을 만들 수 있으므로,
      무음 환각 비교 실험에서 두 조건을 함께 잰다 (설계안 문서 참고).
    WHISPER_DEVICE 와 같은 장치 설정(cpu/cuda)을 쓴다. 1.7B 모델은 CPU(라즈베리파이)에서는 실시간 인식이 어렵다.
    """

    is_local = True

    def __init__(self, model: str = "TeamUNIVA/qwen3_asr_1.7b_ko", *, adapter: str = "", device: str = "cpu",
                 backend: str = "", context: str = "", language: str | None = "Korean",
                 max_new_tokens: int = 256) -> None:
        self.model_name, self.adapter, self.device, self.context = model, adapter, device, context
        self.language = language or None  # None: 자동 판별 (무음이면 빈 결과)
        self.backend = backend or ("hf" if adapter or model.endswith("-hf") else "qwen-asr")
        if self.backend not in ("hf", "qwen-asr"):
            raise ValueError("QWEN_ASR_BACKEND 는 hf 또는 qwen-asr")
        if adapter and self.backend != "hf":
            raise ValueError("LoRA 어댑터(QWEN_ASR_ADAPTER)는 hf 방식에서만 쓸 수 있습니다")
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._lock = asyncio.Lock()

    @staticmethod
    def missing_packages(backend: str, adapter: str = "") -> list[str]:
        """설치되지 않은 패키지 목록 (모듈을 불러오지 않고 확인). 서버 시작 시 make_engine 이 확인한다."""
        import importlib.util
        from importlib.metadata import PackageNotFoundError, version

        need = ["torch"] + (["qwen_asr"] if backend == "qwen-asr" else ["transformers"]) + (["peft"] if adapter else [])
        missing = [m for m in need if importlib.util.find_spec(m) is None]
        if backend == "hf" and "transformers" not in missing:
            try:  # Qwen3ASRForConditionalGeneration 은 transformers 5.13 부터 들어 있다
                major, minor = (int(x) for x in version("transformers").split(".")[:2])
                if (major, minor) < (5, 13):
                    missing.append("transformers>=5.13")
            except (PackageNotFoundError, ValueError):
                pass
        return missing

    @staticmethod
    def _dtype(torch, device: str):
        """GPU 는 bfloat16(지원하지 않는 GPU 는 float16), CPU 는 float32."""
        if not device.startswith("cuda"):
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _load(self):
        if self._model is not None:
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
            # device_map 은 accelerate 가 필요하므로 쓰지 않고 장치 하나로 옮긴다
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

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    async def recognize(self, ulaw8k: bytes) -> str:
        if not ulaw8k:
            return ""
        t0 = time.monotonic()
        async with self._lock:  # 모델 하나를 순차 사용
            try:
                text = await asyncio.to_thread(self._run, ulaw8k)
            except Exception as e:
                log.warning("Qwen3-ASR 인식 실패: %s", _hide(e))
                return ""
        text = _clean(text)
        log.info("인식(qwen) %.2fs (음성 %.1fs) %s", time.monotonic() - t0, len(ulaw8k) / 8000, text)
        return text


# ---------------------------------------------------------------------------
KINDS = ("rtzr", "clova", "gemini", "qwen")


def make_engine(kind: str, device: str = "cpu"):
    """.env 값으로 후보 엔진을 만든다. stt.make_recognizer 가 호출한다."""
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
        # 'GEMINI_STT_THINKING=' 처럼 비워 두면 thinkingConfig 를 보내지 않는다 (2.5 계열 모델용)
        thinking = os.environ.get("GEMINI_STT_THINKING", "MINIMAL").strip()
        return GeminiAudioRecognizer(key, model, thinking=thinking, timeout_s=timeout)
    if kind == "qwen":
        # 'QWEN_ASR_LANGUAGE=' 처럼 비워 두면 언어를 지정하지 않는다 (없으면 Korean)
        rec = QwenASRRecognizer(_env_str("QWEN_ASR_MODEL", "TeamUNIVA/qwen3_asr_1.7b_ko"),
                                adapter=_env_str("QWEN_ASR_ADAPTER"), device=device,
                                backend=_env_str("QWEN_ASR_BACKEND"), context=_env_str("QWEN_ASR_CONTEXT"),
                                language=os.environ.get("QWEN_ASR_LANGUAGE", "Korean").strip())
        missing = rec.missing_packages(rec.backend, rec.adapter)
        if missing:
            cmd = ("pip install qwen-asr torch" if rec.backend == "qwen-asr"
                   else 'pip install "transformers>=5.13" torch' + (" peft" if rec.adapter else ""))
            raise RuntimeError(f"STT=qwen ({rec.backend} 방식)에 필요한 패키지가 없습니다: {', '.join(missing)}. "
                               f"설치: {cmd} (qwen-asr 와 transformers 5 는 가상환경을 나눈다)")
        return rec
    raise ValueError(f"알 수 없는 STT: {kind}")
