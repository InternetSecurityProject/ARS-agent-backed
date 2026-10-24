"""
ARS 자동화 API 서버 (API 명세서 v0.3)

실행
  python app.py                      # .env 의 CALL_BACKEND 사용 (clawops / sim)
  python app.py --sim                # 가상 ARS 로 실행 (실통화 없음)
브라우저에서 http://localhost:8000 을 열면 시험용 앱 화면이 나온다.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from audio import Segmenter
from config import Settings, settings as default_settings
from interpreter import load_interpreter
from navigator import Navigator
from session import ApiError, Deps, SessionManager, now_iso
from stt import make_recognizer
from telephony import ClawOpsBackend, SimBackend

log = logging.getLogger("ars.app")
_PREFETCH_TASKS: set[asyncio.Task] = set()
HERE = Path(__file__).parent


def build_deps(st: Settings) -> Deps:
    backend = (SimBackend(st.sim_tree, st.sim_speed, st.sim_audio, st.sim_voice) if st.backend == "sim"
               else ClawOpsBackend(st.clawops_from, st.dtmf_mode))
    recognizer = make_recognizer(stt_kind(st), st.whisper_model, st.whisper_device, st.vad_rms)
    prefetch = getattr(recognizer, "prefetch", None)
    if prefetch and st.stt_prefetch:
        # ARS 가 말하는 동안 단락 앞부분을 미리 인식한다 (Segmenter 가 쉼마다 넘김)
        def on_chunk(prefix: bytes, start: int) -> None:
            t = asyncio.get_running_loop().create_task(prefetch(prefix, start))
            _PREFETCH_TASKS.add(t)  # 작업이 끝나기 전에 사라지지 않게 참조를 둔다
            t.add_done_callback(_PREFETCH_TASKS.discard)

        Segmenter.on_chunk = staticmethod(on_chunk)
    else:
        Segmenter.on_chunk = None
    interp = load_interpreter(st.interpreter)
    nav = Navigator(interp, st.match_threshold, st.margin, st.max_revisit, st.max_presses, st.low_threshold)
    return Deps(st, backend, recognizer, interp, nav)


def stt_kind(st: Settings) -> str:
    """가상 ARS 는 문장을 직접 주므로 인식하지 않는다. 음성 가상 ARS 는 실통화와 같이 인식한다."""
    return "none" if st.backend == "sim" and not st.sim_voice else st.stt


def create_app(st: Settings | None = None) -> FastAPI:
    st = st or default_settings
    deps = build_deps(st)
    manager = SessionManager(deps)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await deps.backend.start()
        warm = getattr(deps.recognizer, "warmup", None)
        if warm:
            asyncio.create_task(warm())  # STT 모델은 백그라운드로 미리 로드
        yield
        for s in manager.sessions.values():
            if not s.ended:
                await s.finish("user_canceled")
        await deps.backend.stop()

    app = FastAPI(title="ARS 자동화 API", version="0.3", lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, e: ApiError):
        return JSONResponse({"error": {"code": e.code, "message": e.message}}, status_code=e.status)

    def bearer(auth: str | None) -> str | None:
        return auth[7:] if auth and auth.lower().startswith("bearer ") else None

    # 4.1 세션 생성 ------------------------------------------------------------
    @app.post("/v1/sessions", status_code=201)
    async def create_session(request: Request, x_app_key: str | None = Header(None)):
        if x_app_key != st.app_key:
            raise ApiError(401, "unauthorized", "앱 키가 올바르지 않습니다.")
        try:
            body = await request.json()
        except Exception:
            raise ApiError(400, "invalid_request", "JSON 본문이 필요합니다.")
        phone = str(body.get("phone_number") or "")
        if not phone:
            raise ApiError(400, "invalid_request", "phone_number 가 필요합니다.")
        out = await manager.create(phone, body.get("goal"))
        host = request.headers.get("host", "localhost")
        scheme = "wss" if request.url.scheme == "https" else "ws"
        sid = out["session_id"]
        out["events_url"] = f"{scheme}://{host}/v1/sessions/{sid}/events"
        out["audio_url"] = f"{scheme}://{host}/v1/sessions/{sid}/audio"
        return out

    # 4.2 이벤트 수신 ------------------------------------------------------------
    @app.websocket("/v1/sessions/{sid}/events")
    async def events(ws: WebSocket, sid: str, token: str | None = None):
        try:
            s = manager.get(sid, token)
        except ApiError as e:
            await ws.close(code=4401 if e.status == 401 else 4404, reason=e.code)
            return
        await ws.accept()
        q = s.hub.subscribe()
        try:
            await ws.send_json({"event": "snapshot", "at": now_iso(), "data": s.snapshot()})
            while True:
                msg = await q.get()
                await ws.send_json(msg)
                if msg["event"] == "ended":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            s.hub.unsubscribe(q)
        try:
            await ws.close()
        except Exception:
            pass

    # 4.3 사용자 조작 ------------------------------------------------------------
    @app.post("/v1/sessions/{sid}/actions")
    async def actions(sid: str, request: Request, authorization: str | None = Header(None)):
        s = manager.get(sid, bearer(authorization))
        try:
            body = await request.json()
        except Exception:
            raise ApiError(400, "invalid_request", "JSON 본문이 필요합니다.")
        return await s.action(body)

    # 4.4 결과 조회 ------------------------------------------------------------
    @app.get("/v1/sessions/{sid}/result")
    async def result(sid: str, authorization: str | None = Header(None)):
        s = manager.get(sid, bearer(authorization))
        code, body = s.result()
        return JSONResponse(body, status_code=code)

    # 4.5 오디오 채널 ------------------------------------------------------------
    @app.websocket("/v1/sessions/{sid}/audio")
    async def audio(ws: WebSocket, sid: str, token: str | None = None):
        try:
            s = manager.get(sid, token)
        except ApiError as e:
            await ws.close(code=4401 if e.status == 401 else 4404, reason=e.code)
            return
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        s.audio_out.add(q)

        async def downlink():
            while True:
                frame = await q.get()
                if frame is None:
                    break
                await ws.send_bytes(frame)

        down = asyncio.create_task(downlink())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    for i in range(0, len(data) - len(data) % 160, 160):
                        await s.mic_audio(data[i:i + 160])
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            s.audio_out.discard(q)
            down.cancel()

    # 시험용 앱 화면 ------------------------------------------------------------
    @app.get("/")
    async def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/v1/health")
    async def health():
        a = manager.active()
        ex = getattr(deps.interp, "goal_examples", None)
        return {"backend": st.backend, "stt": stt_kind(st), "sim_voice": st.sim_voice, "interpreter": st.interpreter,
                "mock_goals": ex() if ex else [], "active_session": a.id if a else None}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="ARS 자동화 API 서버")
    ap.add_argument("--sim", action="store_true", help="가상 ARS 로 실행")
    ap.add_argument("--sim-voice", action="store_true", help="음성 가상 ARS (TTS 안내 → STT, DTMF 톤 인식)")
    ap.add_argument("--sim-tree", help="가상 ARS 메뉴 트리 (예: sim_nhis.json)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.debug else logging.INFO,
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("clawops").setLevel(logging.INFO if a.debug else logging.WARNING)
    st = Settings()
    if a.sim or a.sim_voice:
        st.backend = "sim"
    if a.sim_voice:
        st.sim_voice = True
    if a.sim_tree:
        st.sim_tree = a.sim_tree
    import uvicorn

    log.info("백엔드=%s%s STT=%s 해석기=%s", st.backend, "(음성)" if st.sim_voice else "", stt_kind(st), st.interpreter)
    uvicorn.run(create_app(st), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
