"""
STT 후보 엔진 시험 (실제 API 호출 없음)

  python -m pytest -q test_stt_engines.py

- 리턴제로: 127.0.0.1 에 띄운 가짜 WebSocket 서버와 가짜 인증 응답으로 요청 형식·응답 처리·실패 처리를 확인한다.
- CLOVA·Gemini: 가짜 HTTP 응답(httpx.MockTransport)으로 요청 형식·응답 처리·실패 처리를 확인한다.
- Qwen3-ASR: 모델 대신 가짜 객체를 넣어 입력 변환·호출 인자를 확인한다 (모델 가중치 없이 시험).
"""

from __future__ import annotations

import asyncio
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
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
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
    """.env.example 를 그대로 복사해 'KEY=' 로 비어 있어도 기본값으로 동작한다."""
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
    """후보 엔진은 미리 인식(prefetch)이 없으므로 Segmenter 가 조각을 넘기지 않는다."""
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
    """리턴제로 스트리밍 서버 흉내. replies: EOS 를 받은 뒤 보낼 메시지 목록. close: 보낸 뒤 연결을 닫을지."""

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
            await ws.wait_closed()  # 서버가 먼저 닫지 않는다 (클라이언트가 닫을 때까지 대기)

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
    assert text == "잔액조회는 1번 샵 버튼을 누르세요"  # 확정 문장만, 빈 문장 제외, stt._clean 적용
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
    assert len(calls) == 2  # 만료 10분 이내 토큰은 다시 받는다
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
    run(rec.warmup())  # 시작 시 토큰 실패는 경고만 남긴다 (예외 없음)


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
    assert run(rec.recognize(AUDIO)) == "폰뱅킹 서비스는 1번"  # stt._clean 적용
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
    """transformers -hf 방식: apply_transcription_request → generate → 입력 길이 뒤 토큰만 decode."""
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
# 서버 흐름에 연결 (Segmenter → 엔진 → 해석 → DTMF). 기존 test_flows 의 수동 ARS 를 그대로 쓴다
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
        assert [c["text"] for c in s.captions] == [ROOT, BAL]
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
    """민감정보 입력 뒤 ARS 가 입력값을 다시 읽어 주는 구간은 외부 엔진으로 보내지 않는다 (session.py 규칙)."""
    from test_flows import manual, speak

    seen: list = []
    rec = E.GeminiAudioRecognizer("k", "m", http=sequence_client(["주민등록번호 앞 여섯 자리는 구 영 영 일 일 일"], seen,
                                                                 "gemini"))

    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=rec, paragraph_silence_ms=200, clarify_expire_s=5,
                               clarify_drop_s=10)
        s.secure_window = True
        await speak(s, 0.3)
        await asyncio.sleep(0.2)
        await s.action({"type": "cancel"})

    run(go())
    assert seen == []
