"""
STT 후보 엔진 시험 (실제 API 호출 없음)

  python -m pytest -q test_stt_engines.py

- 리턴제로: 127.0.0.1 의 가짜 WebSocket 서버와 가짜 인증 응답으로 요청 형식·응답 처리·실패 처리 확인.
- CLOVA·Gemini: 가짜 HTTP 응답(httpx.MockTransport)으로 요청 형식·응답 처리·실패 처리 확인.
- Qwen3-ASR: 모델 대신 가짜 객체로 입력 변환·호출 인자 확인 (모델 가중치 없이 시험).
"""

from __future__ import annotations

import asyncio
import time
import base64
import contextlib
import io
import json
import logging
import sys
import types
import wave
from urllib.parse import parse_qs, urlsplit

import httpx
import numpy as np
import pytest

from audio import hum_ulaw
import stt_engines as E
from stt import make_recognizer

SECRET = "sec-SECRET-123"
AUDIO = hum_ulaw(1000)  # 1초 (8000바이트 μ-law)


def run(coro):
    return asyncio.run(coro)


def wav_info(data: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


# ---------------------------------------------------------------------------
# 엔진 선택 (stt.make_recognizer)
# ---------------------------------------------------------------------------
ENV_KEYS = ["RTZR_CLIENT_ID", "RTZR_CLIENT_SECRET", "RTZR_MODEL", "RTZR_KEYWORDS", "CLOVA_SPEECH_INVOKE_URL",
            "CLOVA_SPEECH_SECRET", "CLOVA_SAMPLE_RATE", "CLOVA_BOOSTINGS", "GEMINI_API_KEY", "GEMINI_MODEL",
            "GEMINI_STT_MODEL", "GEMINI_STT_THINKING", "QWEN_ASR_MODEL", "QWEN_ASR_ADAPTER", "QWEN_ASR_BACKEND",
            "QWEN_ASR_CONTEXT", "STT_TIMEOUT_S"]


@pytest.fixture
def clean_env(monkeypatch):
    for k in ENV_KEYS + ["QWEN_ASR_LANGUAGE"]:
        monkeypatch.delenv(k, raising=False)
    # Qwen3-ASR 패키지(torch 등) 미설치 환경에서도 실행 가능. 패키지 확인 자체는 test_qwen_missing_packages_* 에서 시험
    monkeypatch.setattr(E.QwenASRRecognizer, "missing_packages", staticmethod(lambda backend, adapter="": []))
    return monkeypatch


def test_make_recognizer_selects_engines(clean_env):
    env = clean_env
    env.setenv("RTZR_CLIENT_ID", "id")
    env.setenv("RTZR_CLIENT_SECRET", SECRET)
    env.setenv("CLOVA_SPEECH_INVOKE_URL", "https://clova.example/ext/v1/1/abc")
    env.setenv("CLOVA_SPEECH_SECRET", SECRET)
    env.setenv("GEMINI_API_KEY", "g-key")
    env.setenv("STT_TIMEOUT_S", "3.5")
    r = make_recognizer("rtzr")
    assert isinstance(r, E.RtzrRecognizer) and r.is_local is False and r.timeout_s == 3.5
    assert r.params["domain"] == "CALL" and r.params["sample_rate"] == "8000"
    c = make_recognizer("clova")
    assert isinstance(c, E.ClovaSpeechRecognizer) and c.is_local is False
    assert c.url == "https://clova.example/ext/v1/1/abc/recognizer/upload"
    g = make_recognizer("gemini")
    assert isinstance(g, E.GeminiAudioRecognizer) and g.is_local is False
    q = make_recognizer("qwen", device="cuda")
    assert isinstance(q, E.QwenASRRecognizer) and q.is_local is True and q.device == "cuda"
    assert q.backend == "qwen-asr" and q.model_name == "TeamUNIVA/qwen3_asr_1.7b_ko"


def test_blank_env_values_use_defaults(clean_env):
    """.env.example 를 그대로 복사해 'KEY=' 로 비어 있는 경우의 기본값 동작 확인."""
    env = clean_env
    for k in ["RTZR_MODEL", "RTZR_KEYWORDS", "CLOVA_SAMPLE_RATE", "CLOVA_BOOSTINGS", "GEMINI_STT_MODEL",
              "QWEN_ASR_MODEL", "QWEN_ASR_ADAPTER", "QWEN_ASR_BACKEND", "QWEN_ASR_CONTEXT", "STT_TIMEOUT_S"]:
        env.setenv(k, "")
    env.setenv("RTZR_CLIENT_ID", "id")
    env.setenv("RTZR_CLIENT_SECRET", SECRET)
    env.setenv("CLOVA_SPEECH_INVOKE_URL", "https://clova.example/x")
    env.setenv("CLOVA_SPEECH_SECRET", SECRET)
    env.setenv("GEMINI_API_KEY", "g-key")
    env.setenv("GEMINI_MODEL", "gemini-test")
    r = make_recognizer("rtzr")
    assert r.params["model_name"] == "sommers_ko" and "keywords" not in r.params
    assert r.timeout_s == E.DEFAULT_TIMEOUT_S
    c = make_recognizer("clova")
    assert c.sample_rate == 8000 and "boostings" not in c.params
    assert make_recognizer("gemini").model == "gemini-test"
    q = make_recognizer("qwen")
    assert q.model_name == "TeamUNIVA/qwen3_asr_1.7b_ko" and q.backend == "qwen-asr" and q.context == ""


@pytest.mark.parametrize("kind,need", [("rtzr", "RTZR_CLIENT_ID"), ("clova", "CLOVA_SPEECH_INVOKE_URL"),
                                       ("gemini", "GEMINI_API_KEY")])
def test_missing_keys_fail_at_startup(clean_env, kind, need):
    with pytest.raises(RuntimeError, match=need):
        make_recognizer(kind)


def test_build_deps_wires_engine_without_prefetch(clean_env):
    """미리 인식(prefetch)이 없는 후보 엔진에서 Segmenter 의 조각 전달 없음 확인."""
    from app import build_deps
    from audio import Segmenter
    from config import Settings

    clean_env.setenv("GEMINI_API_KEY", "g-key")
    st = Settings()
    st.backend, st.sim_audio, st.stt, st.interpreter = "clawops", False, "gemini", "interpreter:MockInterpreter"
    deps = build_deps(st)
    assert isinstance(deps.recognizer, E.GeminiAudioRecognizer)
    assert Segmenter.on_chunk is None


def test_bench_engine_names():
    from stt_bench import engine_of

    assert engine_of("rtzr") == ("rtzr", "")
    assert engine_of("openai") == ("openai", "")
    assert engine_of("large-v3-turbo") == ("whisper", "large-v3-turbo")
    assert engine_of("whisper:small") == ("whisper", "small")


def test_empty_audio_returns_empty_without_network():
    def boom(_req):
        raise AssertionError("호출되면 안 됨")

    http = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    assert run(E.ClovaSpeechRecognizer("https://x", SECRET, http=http).recognize(b"")) == ""
    assert run(E.GeminiAudioRecognizer("k", "m", http=http).recognize(b"")) == ""
    assert run(E.RtzrRecognizer("id", SECRET, http=http).recognize(b"")) == ""
    assert run(E.QwenASRRecognizer().recognize(b"")) == ""


def test_wav_bytes():
    rate, ch, n = wav_info(E.wav_bytes(b"\x00\x00" * 800, 8000))
    assert (rate, ch, n) == (8000, 1, 800)


# ---------------------------------------------------------------------------
# 리턴제로
# ---------------------------------------------------------------------------
def test_rtzr_keyword_rules():
    assert E.rtzr_keywords("별표, 우물정자:2.5 ,상담원 연결") == "별표,우물정자:2.5,상담원 연결"
    # 숫자·영문, 20자 초과, 가중치 범위 밖은 제외 (연결 거부 방지)
    assert E.rtzr_keywords("1번,abc,가" * 1 + ",다시듣기:9," + "가" * 21) == "가"
    assert E.rtzr_keywords(",".join(["별표"] * 120)).count(",") == 99


def auth_transport(calls: list, expire_in: float = 6 * 3600, status: int = 200):
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert req.url.path == "/v1/authenticate"
        form = parse_qs(req.content.decode())
        assert form == {"client_id": ["id"], "client_secret": [SECRET]}
        import time as _t

        return httpx.Response(status, json={"access_token": f"tok{len(calls)}", "expire_at": _t.time() + expire_in})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class FakeRtzr:
    """리턴제로 스트리밍 서버 흉내. replies: EOS 수신 뒤 보낼 메시지 목록. close: 전송 뒤 연결 종료 여부."""

    def __init__(self, replies: list[dict], close: bool = True, reject: int | None = None) -> None:
        self.replies, self.close, self.reject = replies, close, reject
        self.requests: list = []
        self.audio = bytearray()
        self.chunks: list[int] = []
        self.got_eos = False

    async def handler(self, ws):
        self.requests.append(ws.request)
        async for msg in ws:
            if isinstance(msg, bytes):
                self.audio += msg
                self.chunks.append(len(msg))
            elif msg == "EOS":
                self.got_eos = True
                break
        for r in self.replies:
            await ws.send(json.dumps(r, ensure_ascii=False))
        if not self.close:
            await ws.wait_closed()  # 서버 쪽 선종료 없음 (클라이언트가 닫을 때까지 대기)

    def process_request(self, conn, request):
        if self.reject:
            return conn.respond(self.reject, "rejected\n")
        return None


@contextlib.asynccontextmanager
async def rtzr_server(fake: FakeRtzr):
    from websockets.asyncio.server import serve

    async with serve(fake.handler, "127.0.0.1", 0, process_request=fake.process_request) as srv:
        port = srv.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/v1/transcribe:streaming"


def final(text, is_final=True):
    return {"seq": 0, "start_at": 0, "duration": 900, "final": is_final, "alternatives": [{"text": text}]}


def test_rtzr_stream_request_and_result():
    async def go():
        fake = FakeRtzr([final("잔액", False), final("잔액조회는 1번"), final("", True), final("샷 버튼을 누르세요")])
        calls: list = []
        async with rtzr_server(fake) as url:
            rec = E.RtzrRecognizer("id", SECRET, keywords="별표,상담원 연결:2", ws_url=url, timeout_s=5,
                                   http=auth_transport(calls))
            text = await rec.recognize(AUDIO)
            text2 = await rec.recognize(AUDIO)
        return fake, calls, text, text2

    fake, calls, text, text2 = run(go())
    assert text == "잔액조회는 1번 샵 버튼을 누르세요"  # 확정 문장만, 빈 문장 제외, 공통 키 이름 교정(샷→샵)
    assert text2 == text
    assert len(calls) == 1  # 토큰 재사용
    req = fake.requests[0]
    q = parse_qs(urlsplit(req.path).query)
    assert q["sample_rate"] == ["8000"] and q["encoding"] == ["LINEAR16"] and q["domain"] == ["CALL"]
    assert q["model_name"] == ["sommers_ko"] and q["keywords"] == ["별표,상담원 연결:2"]
    assert req.headers["Authorization"] == "bearer tok1"
    assert fake.got_eos
    assert len(fake.audio) == 2 * 2 * len(AUDIO)  # 두 번 인식, μ-law 1바이트 → PCM 2바이트 (업샘플링 없음)
    assert max(fake.chunks) <= E.RtzrRecognizer.CHUNK


def test_rtzr_refreshes_expiring_token():
    async def go():
        fake = FakeRtzr([final("안녕하십니까")])
        calls: list = []
        async with rtzr_server(fake) as url:
            rec = E.RtzrRecognizer("id", SECRET, ws_url=url, timeout_s=5, http=auth_transport(calls, expire_in=60))
            await rec.recognize(AUDIO)
            await rec.recognize(AUDIO)
        return fake, calls

    fake, calls = run(go())
    assert len(calls) == 2  # 만료 10분 이내 토큰은 재발급
    assert fake.requests[1].headers["Authorization"] == "bearer tok2"


def test_rtzr_server_error_returns_empty_and_hides_secret(caplog):
    async def go():
        fake = FakeRtzr([{"error": "invalid sample_rate"}])
        async with rtzr_server(fake) as url:
            rec = E.RtzrRecognizer("id", SECRET, ws_url=url, timeout_s=5, http=auth_transport([]))
            return await rec.recognize(AUDIO)

    with caplog.at_level(logging.WARNING, logger="ars.stt"):
        assert run(go()) == ""
    assert "invalid sample_rate" in caplog.text and SECRET not in caplog.text


def test_rtzr_unauthorized_clears_token():
    async def go():
        fake = FakeRtzr([], reject=401)
        async with rtzr_server(fake) as url:
            rec = E.RtzrRecognizer("id", SECRET, ws_url=url, timeout_s=5, http=auth_transport([]))
            text = await rec.recognize(AUDIO)
            return text, rec._token

    text, token = run(go())
    assert text == "" and token is None


def test_rtzr_keeps_finals_when_server_does_not_close():
    async def go():
        fake = FakeRtzr([final("상담원 연결은 0번")], close=False)
        async with rtzr_server(fake) as url:
            rec = E.RtzrRecognizer("id", SECRET, ws_url=url, timeout_s=1.0, http=auth_transport([]))
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            text = await rec.recognize(AUDIO)
            return text, loop.time() - t0

    text, dt = run(go())
    assert text == "상담원 연결은 0번" and dt < 3


def test_rtzr_auth_failure_returns_empty():
    rec = E.RtzrRecognizer("id", SECRET, ws_url="ws://127.0.0.1:9/x", timeout_s=2, http=auth_transport([], status=401))
    assert run(rec.recognize(AUDIO)) == ""
    run(rec.warmup())  # 시작 시 토큰 실패는 경고만 기록 (예외 없음)


# ---------------------------------------------------------------------------
# CLOVA Speech
# ---------------------------------------------------------------------------
def clova_client(seen: list, reply: dict | None = None, status: int = 200):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(status, json=reply if reply is not None else {"result": "COMPLETED", "message": "Succeeded",
                                                                              "text": "본 뱅킹 서비스는 1번"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def multipart_fields(req: httpx.Request) -> dict:
    from email.parser import BytesParser
    from email.policy import default

    msg = BytesParser(policy=default).parsebytes(
        f"Content-Type: {req.headers['content-type']}\r\n\r\n".encode() + req.content)
    return {p.get_param("name", header="content-disposition"): p.get_payload(decode=True) for p in msg.iter_parts()}


def test_clova_request_and_result():
    seen: list = []
    rec = E.ClovaSpeechRecognizer("https://clova.example/ext/v1/9/k/", SECRET, boostings="별표, 상담원",
                                  http=clova_client(seen))
    assert run(rec.recognize(AUDIO)) == "본 뱅킹 서비스는 1번"  # stt._clean 적용 (낱말 교정 없음)
    req = seen[0]
    assert str(req.url) == "https://clova.example/ext/v1/9/k/recognizer/upload"
    assert req.headers["X-CLOVASPEECH-API-KEY"] == SECRET
    f = multipart_fields(req)
    params = json.loads(f["params"])
    assert params["language"] == "ko-KR" and params["completion"] == "sync"
    assert params["diarization"] == {"enable": False}
    assert params["boostings"] == [{"words": "별표,상담원", "weight": 2.0}]
    assert f["type"] == b"application/json"
    assert wav_info(f["media"]) == (8000, 1, len(AUDIO))


def test_clova_16k_option():
    seen: list = []
    rec = E.ClovaSpeechRecognizer("https://c", SECRET, sample_rate=16000, http=clova_client(seen))
    run(rec.recognize(AUDIO))
    assert wav_info(multipart_fields(seen[0])["media"])[0] == 16000
    with pytest.raises(ValueError):
        E.ClovaSpeechRecognizer("https://c", SECRET, sample_rate=44100)


@pytest.mark.parametrize("reply,status", [({"result": "FAILED", "message": "invalid media"}, 200),
                                          ({"message": "server"}, 500)])
def test_clova_failure_returns_empty(reply, status, caplog):
    rec = E.ClovaSpeechRecognizer("https://c", SECRET, http=clova_client([], reply, status))
    with caplog.at_level(logging.WARNING, logger="ars.stt"):
        assert run(rec.recognize(AUDIO)) == ""
    assert "CLOVA STT 실패" in caplog.text and SECRET not in caplog.text


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def gemini_client(seen: list, text: str | None = '{"text": "잔액조회는 1번입니다"}', status: int = 200, extra=None):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if text is None:
            return httpx.Response(status, json={"candidates": [{"finishReason": "SAFETY"}]})
        parts = (extra or []) + [{"text": text}]
        return httpx.Response(status, json={"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"x-goog-api-key": "g-key"})


def test_gemini_request_and_result():
    seen: list = []
    rec = E.GeminiAudioRecognizer("g-key", "gemini-test", http=gemini_client(seen))
    assert run(rec.recognize(AUDIO)) == "잔액조회는 1번입니다"
    req = seen[0]
    assert req.url.path == "/v1beta/models/gemini-test:generateContent"
    body = json.loads(req.content)
    audio = body["contents"][0]["parts"][1]["inlineData"]
    assert audio["mimeType"] == "audio/wav"
    assert wav_info(base64.b64decode(audio["data"])) == (16000, 1, 2 * len(AUDIO))
    gen = body["generationConfig"]
    assert gen["temperature"] == 0 and gen["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}
    assert gen["responseJsonSchema"]["required"] == ["text"]
    assert "빈 문자열" in body["systemInstruction"]["parts"][0]["text"]


def test_gemini_without_thinking_and_skips_thought_parts():
    seen: list = []
    rec = E.GeminiAudioRecognizer("g-key", "gemini-2.5-flash", thinking="",
                                  http=gemini_client(seen, extra=[{"text": "생각 중...", "thought": True}]))
    assert run(rec.recognize(AUDIO)) == "잔액조회는 1번입니다"
    assert "thinkingConfig" not in json.loads(seen[0].content)["generationConfig"]


def test_gemini_silence_and_hallucination_filtered():
    assert run(E.GeminiAudioRecognizer("k", "m", http=gemini_client([], '{"text": ""}')).recognize(AUDIO)) == ""
    assert run(E.GeminiAudioRecognizer("k", "m", http=gemini_client([], '{"text": "시청해 주셔서 감사합니다"}'))
               .recognize(AUDIO)) == ""


@pytest.mark.parametrize("text,status", [(None, 200), ('{"text": "x"}', 429), ("not json", 200)])
def test_gemini_failure_returns_empty(text, status, caplog):
    rec = E.GeminiAudioRecognizer("g-key", "m", http=gemini_client([], text, status))
    with caplog.at_level(logging.WARNING, logger="ars.stt"):
        assert run(rec.recognize(AUDIO)) == ""
    assert "Gemini STT 실패" in caplog.text and "g-key" not in caplog.text


def test_gemini_timeout_returns_empty():
    async def slow(_req):
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    rec = E.GeminiAudioRecognizer("k", "m", timeout_s=0.2, http=httpx.AsyncClient(transport=httpx.MockTransport(slow)))
    assert run(asyncio.wait_for(rec.recognize(AUDIO), 3)) == ""


# ---------------------------------------------------------------------------
# Qwen3-ASR (가짜 모델)
# ---------------------------------------------------------------------------
def test_qwen_backend_selection():
    assert E.QwenASRRecognizer().backend == "qwen-asr"
    assert E.QwenASRRecognizer("Qwen/Qwen3-ASR-1.7B-hf").backend == "hf"
    assert E.QwenASRRecognizer("Qwen/Qwen3-ASR-1.7B-hf", adapter="Playedwell03/qwen3-asr-1.7b-119ko-v5").backend == "hf"
    with pytest.raises(ValueError):
        E.QwenASRRecognizer("TeamUNIVA/qwen3_asr_1.7b_ko", adapter="x", backend="qwen-asr")
    with pytest.raises(ValueError):
        E.QwenASRRecognizer(backend="vllm")


def test_qwen_asr_backend_call():
    calls = []

    class FakeQwenModel:
        def transcribe(self, audio, context="", language=None):
            calls.append((audio, context, language))
            return [types.SimpleNamespace(language="Korean", text="상담사 연결은 0번")]

    rec = E.QwenASRRecognizer(context="별표, 우물정자")
    rec._model = FakeQwenModel()
    assert run(rec.recognize(AUDIO)) == "상담사 연결은 0번"
    audio, context, language = calls[0]
    arr, sr = audio
    assert sr == 16000 and arr.dtype == np.float32 and len(arr) == 2 * len(AUDIO) and np.abs(arr).max() <= 1.0
    assert context == "별표, 우물정자" and language == "Korean"


def test_qwen_hf_backend_call(monkeypatch):
    """transformers -hf 방식 흐름 확인: apply_transcription_request → generate → 입력 길이 뒤 토큰만 decode."""
    fake_torch = types.ModuleType("torch")
    fake_torch.inference_mode = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    seen = {}

    class Inputs(dict):
        def to(self, device, dtype):
            seen["to"] = (device, dtype)
            return self

    class Proc:
        def apply_transcription_request(self, audio, language=None, prompt=None):
            seen["req"] = (audio, language, prompt)
            return Inputs(input_ids=np.zeros((1, 5), dtype=np.int64))

        def decode(self, ids, return_format="raw"):
            seen["decode"] = (ids.tolist(), return_format)
            return ["이전 단계는 별표"]

    class Model:
        def generate(self, **kw):
            seen["max_new_tokens"] = kw["max_new_tokens"]
            return np.array([[0, 0, 0, 0, 0, 7, 8, 9]])

    rec = E.QwenASRRecognizer("Qwen/Qwen3-ASR-1.7B-hf", adapter="a/b", max_new_tokens=128)
    rec._model, rec._processor, rec._device, rec._dtype = Model(), Proc(), "cuda:0", "bf16"
    assert run(rec.recognize(AUDIO)) == "이전 단계는 별표"
    audio, language, prompt = seen["req"]
    assert len(audio) == 2 * len(AUDIO) and language == "Korean" and prompt is None  # 힌트 없음 (119 LoRA 권고)
    assert seen["to"] == ("cuda:0", "bf16") and seen["max_new_tokens"] == 128
    assert seen["decode"] == ([[7, 8, 9]], "transcription_only")


def test_qwen_failure_returns_empty():
    class Broken:
        def transcribe(self, **kw):
            raise RuntimeError("CUDA out of memory")

    rec = E.QwenASRRecognizer()
    rec._model = Broken()
    assert run(rec.recognize(AUDIO)) == ""


# ---------------------------------------------------------------------------
# 서버 흐름 연결 (Segmenter → 엔진 → 해석 → DTMF). test_flows 의 수동 ARS 재사용
# ---------------------------------------------------------------------------
def sequence_client(texts: list[str], seen: list, kind: str):
    """요청마다 texts 를 차례로 돌려주는 가짜 Gemini·CLOVA 서버."""
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        t = texts.pop(0) if texts else ""
        if kind == "gemini":
            body = {"candidates": [{"content": {"parts": [{"text": json.dumps({"text": t}, ensure_ascii=False)}]}}]}
        else:
            body = {"result": "COMPLETED", "message": "Succeeded", "text": t}
        return httpx.Response(200, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("kind", ["gemini", "clova"])
def test_http_engine_drives_session(kind):
    from test_flows import BAL, ROOT, manual, speak, wait_for

    seen: list = []
    http = sequence_client([ROOT, BAL], seen, kind)
    rec = (E.GeminiAudioRecognizer("k", "m", http=http) if kind == "gemini"
           else E.ClovaSpeechRecognizer("https://c", SECRET, http=http))

    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=rec, no_response_s=5, paragraph_silence_ms=200,
                               clarify_expire_s=5, clarify_drop_s=10)
        await speak(s, 0.3)
        await wait_for(s, lambda s: call.keys == ["3"])
        await speak(s, 0.3)
        await wait_for(s, lambda s: call.keys == ["3", "2"])
        # 단락마다 자막 하나. 메뉴 안내 자막은 원문 대신 선택지 목록으로 축약 (session.caption_text)
        texts = [c["text"] for c in s.captions]
        assert len(texts) == 2
        assert texts[0].startswith("잔액조회 3번") and "0번" in texts[0] and "눌러 주십시오" not in texts[0]
        assert "1번" in texts[1] and "2번" in texts[1] and "눌러 주십시오" not in texts[1]
        await s.action({"type": "cancel"})

    run(go())
    assert len(seen) == 2


def test_rtzr_engine_drives_session():
    from test_flows import BAL, ROOT, manual, speak, wait_for

    class Seq(FakeRtzr):
        def __init__(self, texts):
            super().__init__([])
            self.texts = texts

        async def handler(self, ws):
            self.replies = [final(self.texts.pop(0))] if self.texts else []
            await super().handler(ws)

    async def go():
        fake = Seq([ROOT, BAL])
        async with rtzr_server(fake) as url:
            rec = E.RtzrRecognizer("id", SECRET, ws_url=url, timeout_s=5, http=auth_transport([]))
            s, call = await manual("통장 잔액 알고 싶어", stt=rec, no_response_s=5, paragraph_silence_ms=200,
                                   clarify_expire_s=5, clarify_drop_s=10)
            await speak(s, 0.3)
            await wait_for(s, lambda s: call.keys == ["3"])
            await speak(s, 0.3)
            await wait_for(s, lambda s: call.keys == ["3", "2"])
            await s.action({"type": "cancel"})
        return fake

    fake = run(go())
    assert len(fake.requests) == 2


def test_external_engine_skips_secure_readback():
    """민감정보 입력 뒤 ARS 가 입력값을 다시 읽어 주는 구간의 외부 엔진 미전송 확인 (session.py 규칙)."""
    from test_flows import manual, speak

    seen: list = []
    rec = E.GeminiAudioRecognizer("k", "m", http=sequence_client(["주민등록번호 앞 여섯 자리는 구 영 영 일 일 일"], seen,
                                                                 "gemini"))

    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=rec, paragraph_silence_ms=200, clarify_expire_s=5,
                               clarify_drop_s=10)
        s.secure_window = True
        s.secure_sent_at = time.monotonic()
        await speak(s, 0.3)
        await asyncio.sleep(0.2)
        await s.action({"type": "cancel"})

    run(go())
    assert seen == []


def test_external_engine_resumes_after_secure_skip_time():
    """외부 엔진에는 되읽기 구간을 닫을 계기가 없으므로, 입력 전송 뒤 secure_skip_s 경과 시 구간 종료·인식 재개 확인
    (그렇지 않으면 민감정보 입력 뒤 통화 종료까지 인식 불가)."""
    from test_flows import manual, speak, wait_for

    seen: list = []
    rec = E.GeminiAudioRecognizer("k", "m", http=sequence_client(["잠시만 기다려 주십시오."], seen, "gemini"))

    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=rec, paragraph_silence_ms=200, clarify_expire_s=5,
                               clarify_drop_s=10, secure_skip_s=0.5)
        s.secure_window = True
        s.secure_sent_at = time.monotonic() - 1.0          # 건너뛰기 시간 이미 경과
        await speak(s, 0.3)
        await wait_for(s, lambda s: len(seen) == 1)
        assert not s.secure_window
        await s.action({"type": "cancel"})

    run(go())


# ---------------------------------------------------------------------------
# 공통 규칙 추가 시험
# ---------------------------------------------------------------------------
def test_external_engines_have_no_prefetch():
    """외부 엔진에 prefetch 가 없음을 확인.
    미리 인식(prefetch)은 app.build_deps 가 Segmenter 에 연결하며 session 의 민감정보 구간 검사를 거치지 않음.
    외부 엔진에 prefetch 가 있으면 ARS 가 입력값을 다시 읽어 주는 음성이 외부로 나가는 문제 발생."""
    import stt

    for cls in (E.RtzrRecognizer, E.ClovaSpeechRecognizer, E.GeminiAudioRecognizer, stt.OpenAIRecognizer):
        assert cls.is_local is False
        assert not hasattr(cls, "prefetch"), cls.__name__


def test_qwen_language_option(clean_env):
    assert make_recognizer("qwen").language == "Korean"  # 기본: 모델 카드 평가와 같은 조건
    clean_env.setenv("QWEN_ASR_LANGUAGE", "")
    assert make_recognizer("qwen").language is None  # 비우면 자동 판별 (무음이면 'language None' → 빈 결과)
    calls = []

    class FakeQwenModel:
        def transcribe(self, audio, context="", language=None):
            calls.append(language)
            return [types.SimpleNamespace(language="", text="")]

    rec = E.QwenASRRecognizer(language="")
    rec._model = FakeQwenModel()
    assert run(rec.recognize(AUDIO)) == "" and calls == [None]


def test_rtzr_hard_limit_when_auth_hangs():
    """토큰 발급 응답이 없어도 제한 시간 + 2초 안에 빈 결과로 종료됨을 확인 (단락 처리 정지 방지)."""
    async def hang(_req):
        await asyncio.sleep(30)
        return httpx.Response(200, json={})

    async def go():
        rec = E.RtzrRecognizer("id", SECRET, ws_url="ws://127.0.0.1:9/x", timeout_s=0.3,
                               http=httpx.AsyncClient(transport=httpx.MockTransport(hang)))
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        text = await rec.recognize(AUDIO)
        return text, loop.time() - t0

    text, dt = run(go())
    assert text == "" and dt < 3


# ---------------------------------------------------------------------------
# OpenAI (stt.OpenAIRecognizer, 제한 시간 포함)
# ---------------------------------------------------------------------------
def _openai_rec(monkeypatch, create, timeout_s=10.0):
    pytest.importorskip("openai")
    import stt

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    rec = stt.OpenAIRecognizer("gpt-4o-transcribe", timeout_s=timeout_s)
    monkeypatch.setattr(rec.client.audio.transcriptions, "create", create)
    return rec


def test_openai_request_and_result(monkeypatch):
    seen = {}

    async def create(**kw):
        seen.update(kw)
        return types.SimpleNamespace(text="본 뱅킹은 7번")

    rec = _openai_rec(monkeypatch, create)
    assert run(rec.recognize(AUDIO)) == "본 뱅킹은 7번"  # 낱말 교정 없음
    assert seen["model"] == "gpt-4o-transcribe" and seen["language"] == "ko"
    assert wav_info(seen["file"].getvalue()) == (16000, 1, 2 * len(AUDIO))
    assert run(rec.recognize(b"")) == ""


def test_openai_timeout_returns_empty(monkeypatch, caplog):
    async def slow(**kw):
        await asyncio.sleep(5)

    rec = _openai_rec(monkeypatch, slow, timeout_s=0.2)
    with caplog.at_level(logging.WARNING, logger="ars.stt"):
        assert run(asyncio.wait_for(rec.recognize(AUDIO), 3)) == ""
    assert "TimeoutError" in caplog.text


def test_openai_timeout_setting(clean_env, monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("STT_TIMEOUT_S", "4")
    clean_env.setenv("OPENAI_STT_MODEL", "")
    rec = make_recognizer("openai")
    assert rec.timeout_s == 4.0 and rec.model == "gpt-4o-transcribe"


# ---------------------------------------------------------------------------
# 비교 도구(stt_bench)로 엔진 실행: 저장소 시험 음성 bench_clips 사용
# ---------------------------------------------------------------------------
def test_bench_runs_engine_over_repo_clips(monkeypatch, capsys):
    """stt_bench 의 엔진 이름별 엔진 생성, warmup 없는 엔진의 시험 음성 19개 전체 인식과 요약 표 출력 확인."""
    import stt_bench

    meta = json.loads((stt_bench.CLIPS / "clips.json").read_text(encoding="utf-8"))["clips"]
    answer = {stt_bench.pcm16_to_ulaw(stt_bench.read_pcm(stt_bench.CLIPS / m["file"])): m["text"] for m in meta}
    made = []

    class Echo:  # 시험 음성마다 정답 문장을 돌려주는 가짜 외부 엔진 (warmup 없음)
        is_local = False

        async def recognize(self, ulaw8k):
            return answer.get(ulaw8k, "")

    def fake_make(kind, model="small", device="cpu", quiet_rms=500):
        made.append((kind, model, device))
        return Echo()

    monkeypatch.setattr(stt_bench, "make_recognizer", fake_make)
    monkeypatch.setattr(sys, "argv", ["stt_bench.py", "--models", "gemini", "whisper:large-v3-turbo", "--quiet",
                                      "--device", "cpu"])
    run(stt_bench.main())
    out = capsys.readouterr().out
    assert made == [("gemini", "", "cpu"), ("whisper", "large-v3-turbo", "cpu")]
    assert "시험 음성 19개" in out
    assert "| gemini | 0.0% |" in out and "| whisper:large-v3-turbo | 0.0% |" in out


def test_qwen_missing_packages_fail_at_startup(clean_env):
    clean_env.setattr(E.QwenASRRecognizer, "missing_packages", staticmethod(lambda backend, adapter="": ["torch"]))
    with pytest.raises(RuntimeError, match="pip install qwen-asr torch"):
        make_recognizer("qwen")
    clean_env.setenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B-hf")
    clean_env.setenv("QWEN_ASR_ADAPTER", "Playedwell03/qwen3-asr-1.7b-119ko-v5")
    with pytest.raises(RuntimeError, match=r'transformers>=5.13" torch peft'):
        make_recognizer("qwen")


def test_qwen_missing_packages_check(monkeypatch):
    """모듈을 불러오지 않는 설치 여부·transformers 버전 판정 확인."""
    import importlib.metadata
    import importlib.util

    installed = {"torch", "transformers"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name in installed else None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "4.57.6")
    check = E.QwenASRRecognizer.missing_packages
    assert check("qwen-asr") == ["qwen_asr"]
    assert check("hf") == ["transformers>=5.13"]
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "5.21.0")
    assert check("hf") == [] and check("hf", "a/b") == ["peft"]


# ---------------------------------------------------------------------------
# 음성 직접 판단: 받아쓰기 + 분류·점수를 Gemini 호출 한 번으로 처리 (GEMINI_STT_COMBINED=1)
# ---------------------------------------------------------------------------
def test_gemini_combined_call_transcribes_and_primes_interpreter():
    """해석기 연결 시 동작 확인: 녹음 요청에 분류 지시문·스키마 동봉, 응답의 transcript 를 인식 문장으로 사용,
    분류·점수는 캐시에 저장되어 이어지는 classify 의 Gemini 재호출 없음. 미연결 시 받아쓰기만 수행."""
    from gemini_interpreter import GeminiInterpreter
    from test_gemini_interpreter import FakeClient

    reply = {"transcript": "잔액조회는 3번, 거래내역조회는 4번을 눌러 주십시오.", "kind": "menu",
             "options": [{"key": "3", "label": "잔액조회", "category": "잔액 조회"},
                         {"key": "4", "label": "거래내역조회", "category": "거래내역 조회"}],
             "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None, "agent_key": None,
             "label": None, "terminator": None, "critical": False,
             "scores": [{"key": "3", "confidence": 0.9}, {"key": "4", "confidence": 0.1}]}
    seen: list = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(reply, ensure_ascii=False)}]}}]})

    fake = FakeClient()                       # 해석기 쪽 Gemini. 호출 금지
    interp = GeminiInterpreter(fake)
    interp.goal = "통장 잔액 알고 싶어"
    rec = E.GeminiAudioRecognizer("k", "m", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    rec.attach_interpreter(interp)

    async def go():
        text = await rec.recognize(b"\xff" * 8000)
        assert text == "잔액조회는 3번, 거래내역조회는 4번을 눌러 주십시오."
        p = await interp.classify(text)
        assert p.kind == "menu" and [o.key for o in p.options] == ["3", "4"]
        s = await interp.score("통장 잔액 알고 싶어", p.options)
        assert s["3"] == 0.9
    asyncio.run(go())
    assert fake.calls == []                                           # 해석기의 별도 Gemini 호출 없음
    body = seen[0]
    assert "transcript" in body["generationConfig"]["responseJsonSchema"]["properties"]
    assert "<goal>" in body["contents"][0]["parts"][0]["text"]
    assert body["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "audio/wav"

    # 미연결 시 일반 받아쓰기 요청 (text 스키마)
    seen.clear()
    plain = E.GeminiAudioRecognizer("k", "m", http=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: (seen.append(json.loads(req.content)) or httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": json.dumps({"text": "안녕하십니까"}, ensure_ascii=False)}]}}]})))))
    assert asyncio.run(plain.recognize(b"\xff" * 8000)) == "안녕하십니까"
    assert "transcript" not in seen[0]["generationConfig"]["responseJsonSchema"]["properties"]


# 음성 입력 LLM (STT=omni, llama-server OpenAI 호환) -----------------------------
def omni_client(seen: list, content, status: int = 200):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        if status >= 400:
            return httpx.Response(status, text="server error")
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_omni_combined_call_sends_audio_and_primes_interpreter():
    """목적을 아는 해석기 연결 시 녹음(input_audio)과 분류 지시문·스키마의 일괄 전송, 스키마 맨 앞 transcript 배치 확인.
    응답의 분류·점수는 해석기 캐시에 저장되어 이어지는 classify 의 모델 재호출 없음."""
    from gemini_interpreter import GeminiInterpreter
    from test_gemini_interpreter import FakeClient

    reply = {"transcript": "잔액조회는 3번, 거래내역조회는 4번을 눌러 주십시오.", "kind": "menu",
             "options": [{"key": "3", "label": "잔액조회", "category": "잔액 조회"},
                         {"key": "4", "label": "거래내역조회", "category": "거래내역 조회"}],
             "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None, "agent_key": None,
             "label": None, "terminator": None, "critical": False,
             "scores": [{"key": "3", "confidence": 0.9}, {"key": "4", "confidence": 0.1}]}
    seen: list = []
    fake = FakeClient()                       # 해석기 쪽 모델. 호출 금지
    interp = GeminiInterpreter(fake)
    interp.goal = "통장 잔액 알고 싶어"
    rec = E.OmniRecognizer(http=omni_client(seen, json.dumps(reply, ensure_ascii=False)))
    rec.attach_interpreter(interp)
    assert rec.is_local is True

    async def go():
        text = await rec.recognize(AUDIO)
        assert text == reply["transcript"]
        p = await interp.classify(text)
        assert p.kind == "menu" and [o.key for o in p.options] == ["3", "4"]
        assert (await interp.score("통장 잔액 알고 싶어", p.options))["3"] == 0.9
    asyncio.run(go())
    assert fake.calls == []
    body = seen[0]
    schema = body["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"])[0] == "transcript" and schema["required"][0] == "transcript"
    user = body["messages"][1]["content"]
    assert "<goal>" in user[0]["text"] and user[1]["type"] == "input_audio" and user[1]["input_audio"]["format"] == "wav"
    assert wav_info(base64.b64decode(user[1]["input_audio"]["data"])) == (16000, 1, 2 * len(AUDIO))
    assert body["temperature"] == 0 and "녹음" in body["messages"][0]["content"]


def test_omni_plain_transcription_without_goal_and_failures():
    seen: list = []
    rec = E.OmniRecognizer(http=omni_client(seen, '{"text": "안녕하십니까"}'))
    assert run(rec.recognize(AUDIO)) == "안녕하십니까"
    assert seen[0]["response_format"]["json_schema"]["schema"]["required"] == ["text"]
    assert run(rec.recognize(QUIET)) == "" and len(seen) == 1            # 무음은 전송하지 않음
    for content, status in (("", 200), ("not json", 200), ('{"text": "x"}', 500)):
        assert run(E.OmniRecognizer(http=omni_client([], content, status)).recognize(AUDIO)) == ""
    assert E.OmniRecognizer("http://10.0.0.5:8080/v1").is_local is False


def test_omni_engine_is_selected_and_interpreter_attached(clean_env):
    from app import build_deps
    from config import Settings

    clean_env.setenv("OMNI_URL", "http://127.0.0.1:8099/v1")
    rec = make_recognizer("omni")
    assert isinstance(rec, E.OmniRecognizer) and rec.url == "http://127.0.0.1:8099/v1"
    st = Settings()
    st.backend, st.sim_audio, st.stt, st.interpreter = "clawops", False, "omni", "gemini_interpreter:GeminiInterpreter"
    clean_env.setenv("GEMINI_API_KEY", "k")
    deps = build_deps(st)
    assert deps.recognizer.interp is deps.interp


# Qwen3-ASR 미리 인식 (가짜 모델) ---------------------------------------------
class _PieceModel:
    """조각마다 입력 길이를 기록하고 정해 둔 문장을 차례로 반환. fail=True 면 다음 호출에서 예외."""

    def __init__(self, texts: list[str]) -> None:
        self.texts, self.lens, self.fail = list(texts), [], False

    def transcribe(self, audio, context="", language=None):
        if self.fail:
            raise RuntimeError("CUDA error")
        self.lens.append(len(audio[0]))
        return [types.SimpleNamespace(language="Korean", text=self.texts.pop(0))]


SPEECH = hum_ulaw(1000)          # 1초 말소리
QUIET = b"\xff" * 8000           # 1초 무음 (μ-law 0)


def test_qwen_prefetch_recognizes_only_tail():
    """앞 조각을 미리 인식해 둔 경우 단락 종료 뒤 꼬리만 인식하는지 확인."""
    m = _PieceModel(["잔액조회는 3번", "상담원 연결은 0번"])
    rec = E.QwenASRRecognizer()
    rec._model = m
    first = SPEECH * 2
    run(rec.prefetch(first, 0))
    whole = first + SPEECH
    assert run(rec.recognize(whole)) == "잔액조회는 3번 상담원 연결은 0번"
    assert m.lens == [2 * len(first), 2 * len(SPEECH)]  # 16kHz 변환으로 길이 2배
    assert rec._prefix == []  # 단락 종료 시 미리 인식 결과 초기화


def test_qwen_prefetch_quiet_tail_skips_model():
    m = _PieceModel(["잔액조회는 3번"])
    rec = E.QwenASRRecognizer()
    rec._model = m
    run(rec.prefetch(SPEECH, 0))
    assert run(rec.recognize(SPEECH + QUIET)) == "잔액조회는 3번"
    assert len(m.lens) == 1  # 단락 끝 무음은 모델 입력 제외


def test_qwen_quiet_piece_is_not_transcribed():
    """잡음뿐인 조각의 모델 입력 제외 확인 (무음에서 '응' 같은 글자를 지어내는 현상 방지)."""
    m = _PieceModel([])
    rec = E.QwenASRRecognizer()
    rec._model = m
    run(rec.prefetch(QUIET * 2, 0))
    assert m.lens == [] and rec._prefix[-1][2] == ""


def test_qwen_tail_failure_keeps_prefetched_text():
    m = _PieceModel(["잔액조회는 3번"])
    rec = E.QwenASRRecognizer()
    rec._model = m
    run(rec.prefetch(SPEECH, 0))
    m.fail = True
    assert run(rec.recognize(SPEECH * 2)) == "잔액조회는 3번"


def test_qwen_without_prefetch_recognizes_whole_paragraph():
    m = _PieceModel(["전체 문장"])
    rec = E.QwenASRRecognizer()
    rec._model = m
    assert run(rec.recognize(SPEECH + QUIET)) == "전체 문장"
    assert m.lens == [2 * len(SPEECH + QUIET)]  # 미리 인식이 없으면 단락 전체 인식


def test_qwen_prefetch_mismatched_prefix_restarts():
    """다른 단락의 조각(앞부분 불일치)은 쓰지 않고 처음부터 인식하는지 확인."""
    m = _PieceModel(["앞 단락", "새 단락 전체"])
    rec = E.QwenASRRecognizer()
    rec._model = m
    run(rec.prefetch(SPEECH, 0))
    other = hum_ulaw(500) + QUIET[:4000] + SPEECH
    assert run(rec.recognize(other)) == "새 단락 전체"


def test_build_deps_wires_qwen_prefetch(clean_env):
    """미리 인식을 지원하는 STT=qwen 에서 Segmenter 의 쉼마다 조각 전달 연결 확인."""
    from app import build_deps
    from audio import Segmenter
    from config import Settings

    st = Settings()
    st.backend, st.sim_audio, st.stt, st.interpreter, st.stt_prefetch = "clawops", False, "qwen", "interpreter:MockInterpreter", True
    try:
        build_deps(st)
        assert Segmenter.on_chunk is not None
    finally:
        Segmenter.on_chunk = None


# Qwen3-ASR 서버 방식 (vLLM OpenAI 음성 인식 API, 가짜 전송) -----------------------
def _server_rec(handler, url: str = "http://127.0.0.1:8001/v1", **kw) -> tuple[E.QwenASRRecognizer, list]:
    seen: list = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    rec = E.QwenASRRecognizer("TeamUNIVA/qwen3_asr_1.7b_ko", backend="server", url=url, **kw)
    rec._model = httpx.Client(transport=httpx.MockTransport(h))
    return rec, seen


def test_qwen_server_request_and_reply():
    rec, seen = _server_rec(lambda r: httpx.Response(200, json={"text": "잔액조회는 3번"}))
    assert rec.is_local is True and rec.missing_packages("server") == []
    assert run(rec.recognize(SPEECH)) == "잔액조회는 3번"
    req = seen[0]
    assert str(req.url) == "http://127.0.0.1:8001/v1/audio/transcriptions"
    body = req.content
    assert b'name="model"' in body and b"TeamUNIVA/qwen3_asr_1.7b_ko" in body
    assert b'name="language"\r\n\r\nko\r\n' in body  # 언어 값 ko 확인 (모델 이름의 _ko 와 구분)
    assert b"RIFF" in body and b"WAVE" in body  # WAV 파일로 전송


def test_qwen_server_strips_language_prefix():
    rec, _ = _server_rec(lambda r: httpx.Response(200, json={"text": "language Korean<asr_text>상담원 연결은 0번"}), language=None)
    assert run(rec.recognize(SPEECH)) == "상담원 연결은 0번"


def test_qwen_server_error_returns_empty():
    rec, _ = _server_rec(lambda r: httpx.Response(503, text="loading"))
    assert run(rec.recognize(SPEECH)) == ""


def test_qwen_server_remote_url_is_not_local():
    """다른 컴퓨터의 서버 주소면 외부 STT 로 간주(is_local=False, 민감정보 재확인 구간 미전송)하는지 확인."""
    rec = E.QwenASRRecognizer(backend="server", url="http://192.168.0.10:8001/v1")
    assert rec.is_local is False
    assert E.QwenASRRecognizer(backend="server", url="http://localhost:8001/v1").is_local is True


def test_qwen_server_prefetch_sends_pieces():
    texts = iter(["잔액조회는 3번", "상담원 연결은 0번"])
    rec, seen = _server_rec(lambda r: httpx.Response(200, json={"text": next(texts)}))
    run(rec.prefetch(SPEECH * 2, 0))
    assert run(rec.recognize(SPEECH * 3)) == "잔액조회는 3번 상담원 연결은 0번"
    assert len(seen) == 2


def test_make_recognizer_qwen_server(clean_env):
    clean_env.setenv("QWEN_ASR_BACKEND", "server")
    clean_env.setenv("QWEN_ASR_URL", "http://127.0.0.1:9000/v1")
    q = make_recognizer("qwen")
    assert q.backend == "server" and q.url == "http://127.0.0.1:9000/v1" and q.is_local is True


# 원격 서버·미리 인식 경계 조건 ---------------------------------------------
def test_remote_server_backend_gets_no_prefetch(clean_env):
    """다른 컴퓨터로 보내는 Qwen 서버 방식의 미리 인식 미연결 확인 (되읽기 조각이 쉼마다 외부로 나가는 상황 방지)."""
    from app import build_deps
    from audio import Segmenter
    from config import Settings

    clean_env.setenv("QWEN_ASR_BACKEND", "server")
    clean_env.setenv("QWEN_ASR_URL", "http://192.168.0.10:8001/v1")
    st = Settings()
    st.backend, st.sim_audio, st.stt, st.interpreter, st.stt_prefetch = "clawops", False, "qwen", "interpreter:MockInterpreter", True
    try:
        deps = build_deps(st)
        assert deps.recognizer.is_local is False and Segmenter.on_chunk is None
    finally:
        Segmenter.on_chunk = None


def test_remote_server_prefetch_sends_nothing():
    rec, seen = _server_rec(lambda r: httpx.Response(200, json={"text": "x"}), url="http://192.168.0.10:8001/v1")
    run(rec.prefetch(SPEECH * 2, 0))
    assert seen == [] and rec._prefix == []


def test_qwen_short_last_item_after_prefetch_is_kept():
    """마지막 미리 인식 뒤의 짧은 항목(0.8초)이 단락 끝 무음(1.2초)과 평균되어 빠지지 않음을 확인."""
    m = _PieceModel(["잔액조회는 3번", "상담원 연결은 0번"])
    rec = E.QwenASRRecognizer()
    rec._model = m
    first = SPEECH * 2
    run(rec.prefetch(first, 0))
    tail = hum_ulaw(800) + QUIET + QUIET[:1600]  # 0.8초 말 + 1.2초 무음
    assert run(rec.recognize(first + tail)) == "잔액조회는 3번 상담원 연결은 0번"


def test_qwen_prefetch_failure_skips_rest_of_paragraph():
    calls = {"n": 0}

    def h(r):
        calls["n"] += 1
        return httpx.Response(503, text="stalled")

    rec, _ = _server_rec(h)
    run(rec.prefetch(SPEECH, 0))
    run(rec.prefetch(SPEECH * 2, 0))
    run(rec.prefetch(SPEECH * 3, 0))
    assert calls["n"] == 1  # 첫 실패 뒤 같은 단락의 조각은 전송하지 않음
    run(rec.recognize(SPEECH * 3))
    assert rec._prefetch_failed is False  # 다음 단락에서는 미리 인식 재개


def test_qwen_loopback_server_ignores_proxy_env(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    rec = E.QwenASRRecognizer(backend="server", url="http://127.0.0.1:8001/v1")
    client = rec._load()
    assert client._trust_env is False


def test_whisper_closing_greeting_in_tail_piece_is_kept():
    """미리 인식 뒤 쉼 다음 꼬리에서 따로 인식된 끝맺음 '감사합니다.' 의 단락 내 유지 확인.
    조각 단위로 제거하면 쉼 위치에 따라 같은 말이 결과·일반 안내로 다르게 분류되는 문제 방지.
    단락 전체가 '감사합니다.' 뿐인 경우만 제외."""
    import types
    from stt import WhisperRecognizer

    def seg(text):
        return types.SimpleNamespace(text=text, no_speech_prob=0.0, compression_ratio=1.0, avg_logprob=-0.1)

    outs = iter(["고객님의 등급은 골드 등급입니다.", "감사합니다."])
    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec.quiet_rms, rec._prefix, rec._lock, rec._beam, rec.vad = 500, [], asyncio.Lock(), 1, False
    rec._model = types.SimpleNamespace(transcribe=lambda pcm, **kw: ([seg(next(outs))], None))
    first = SPEECH * 2
    run(rec.prefetch(first, 0))
    assert run(rec.recognize(first + SPEECH)) == "고객님의 등급은 골드 등급입니다. 감사합니다."
    rec._model = types.SimpleNamespace(transcribe=lambda pcm, **kw: ([seg("감사합니다.")], None))
    assert run(rec.recognize(SPEECH)) == ""


def test_whisper_short_last_item_after_prefetch_is_kept():
    from stt import WhisperRecognizer

    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec.quiet_rms, rec._prefix, rec._lock = 500, [], asyncio.Lock()
    pieces = iter(["잔액조회는 3번", "상담원 연결은 0번"])
    rec._run = lambda audio, prev="": next(pieces)
    first = SPEECH * 2
    run(rec.prefetch(first, 0))
    assert run(rec.recognize(first + hum_ulaw(800) + QUIET + QUIET[:1600])) == "잔액조회는 3번 상담원 연결은 0번"

