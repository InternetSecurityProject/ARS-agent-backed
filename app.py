"""
ARS 자동화 API 서버 (API 명세서 v0.3)

실행
  python app.py                      # .env 의 CALL_BACKEND 사용 (clawops / sim)
  python app.py --sim                # 가상 ARS 로 실행 (실제 발신 없음)
  python app.py --no-reload          # 코드가 바뀌어도 자동 재시작하지 않음 (시연·실제 발신 중 권장)
브라우저에서 http://localhost:8000 접속 시 시험용 앱 화면 표시.
기본값: .py 파일이 바뀌면 서버 자동 재시작. 이때 진행 중인 통화는 끊김.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from audio import FrameChunker, Segmenter
from combos import ComboFactory
from config import Settings, settings as default_settings
from interpreter import MockInterpreter, load_interpreter
from navigator import Navigator
from session import ApiError, Deps, SessionManager, now_iso
from stt import make_recognizer
from telephony import ClawOpsBackend, SimBackend, mask_number_logs, parse_numbers

log = logging.getLogger("ars.app")
_PREFETCH_TASKS: set[asyncio.Task] = set()
HERE = Path(__file__).parent


def build_deps(st: Settings) -> Deps:
    backend = (SimBackend(st.sim_tree, st.sim_speed, st.sim_audio, st.sim_voice, lines=st.sim_lines) if st.backend == "sim"
               else ClawOpsBackend(line_spec(st), st.dtmf_mode))
    recognizer = make_recognizer(stt_kind(st), st.whisper_model, st.whisper_device, st.vad_rms)
    prefetch = getattr(recognizer, "prefetch", None)
    # 미리 인식: 로컬 인식기 전용. 외부로 보내는 인식기에 연결하면 민감정보 되읽기 구간(session 의 is_local 확인)을 거치지 않고
    # 쉼마다 조각이 외부로 전송되는 문제 방지
    if prefetch and st.stt_prefetch and getattr(recognizer, "is_local", False):
        # ARS 발화 중 단락 앞부분 미리 인식 (Segmenter 가 쉼마다 전달)
        def on_chunk(prefix: bytes, start: int) -> None:
            t = asyncio.get_running_loop().create_task(prefetch(prefix, start))
            _PREFETCH_TASKS.add(t)  # 작업 완료 전 가비지 수집 방지용 참조 보관
            t.add_done_callback(_PREFETCH_TASKS.discard)

        Segmenter.on_chunk = staticmethod(on_chunk)
    else:
        Segmenter.on_chunk = None
    interp = load_interpreter(st.interpreter)
    multi = getattr(backend, "capacity", 1) > 1
    # 판단 조합을 통화마다 고를 수 있는 해석기(hybrid·local): 통화마다 해석기 새로 생성 (combos.py)
    combos = ComboFactory.from_interp(interp)
    combined = ((st.gemini_stt_combined or st.stt == "omni") and hasattr(recognizer, "attach_interpreter")
                and hasattr(interp, "audio_request"))
    if combined and combos is not None:
        # 받아쓰기+분류 한 번 호출은 서버의 해석기 하나에 묶임. 통화마다 해석기를 만들면 목적·캐시 불일치
        if multi:
            log.warning("%s STT 의 받아쓰기·분류 한 번 호출은 동시 통화(통화마다 해석기)와 함께 쓸 수 없어 끈다 "
                        "(받아쓰기 뒤 따로 판단)", st.stt)
            combined = False
        else:
            # 번호 하나: 한 번 호출 유지, 조합 선택 끔 (서버 설정 조합 하나로 동작)
            log.info("%s STT 의 받아쓰기·분류 한 번 호출을 쓰므로 판단 조합 선택은 끈다", st.stt)
            combos = None
    if combined:
        # STT=gemini(GEMINI_STT_COMBINED=1) 또는 STT=omni(음성 입력 LLM): 받아쓰기+분류를 한 번의 호출로 처리
        recognizer.attach_interpreter(interp)
        log.info("%s STT: 받아쓰기와 분류를 한 번의 호출로 처리한다", st.stt)
    if multi and combos is None and not isinstance(interp, MockInterpreter):
        # 해석기 하나를 동시 통화가 함께 쓰면 한 통화의 목적 요약이 다른 통화의 목적·최근 메뉴를 덮어쓰는 문제 방지
        raise SystemExit(f"번호가 여럿(동시 통화)일 때는 통화마다 해석기를 만들 수 있는 해석기만 쓴다 "
                         f"(hybrid_interpreter:HybridJevInterpreter, local_llm:LocalLlmInterpreter). 지금: {st.interpreter}. "
                         f"CLAWOPS_LINES·SIM_LINES 를 비우거나 INTERPRETER 를 바꾸십시오.")
    nav = Navigator(interp, st.match_threshold, st.margin, st.max_revisit, st.max_presses, st.low_threshold)
    return Deps(st, backend, recognizer, interp, nav, combos)


def line_spec(st: Settings) -> str:
    """ClawOps 발신 번호 목록: CLAWOPS_LINES(쉼표 목록)가 있으면 그것, 없으면 CLAWOPS_FROM 하나."""
    return st.clawops_lines.strip() or st.clawops_from


def stt_kind(st: Settings) -> str:
    """받아쓰기 종류 결정. 가상 ARS 는 문장을 직접 주므로 인식 없음(none). 음성 가상 ARS 는 ClawOps 백엔드와 같이 인식."""
    return "none" if st.backend == "sim" and not st.sim_voice else st.stt


class TokenMaskFilter(logging.Filter):
    """uvicorn 접속 로그에 찍히는 session_token(WebSocket 주소의 ?token=…)을 앞 6자만 남기고 가림.
    session_token 전체 로그 기록 금지 (uvicorn 은 요청 주소를 쿼리까지 그대로 기록)."""
    _RE = re.compile(r"(token=)([^&\s\"']{0,6})[^&\s\"']*")

    def _mask(self, v):
        return self._RE.sub(r"\1\2…", v) if isinstance(v, str) else v

    def filter(self, record: logging.LogRecord) -> bool:
        # 인자 묶음의 모양 유지: uvicorn 접속 로그 형식기가 (주소, 메서드, 경로, 버전, 상태) 5개를 기대함
        if isinstance(record.args, tuple):
            record.args = tuple(self._mask(a) for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: self._mask(v) for k, v in record.args.items()}
        record.msg = self._mask(record.msg)
        return True


class HealthAccessFilter(logging.Filter):
    """성공한 GET /v1/health 접속 로그 제외. 웹 /app 은 열린 창마다 10초 주기로 health 를 읽어 오른쪽 위에
    '접속 중 N명' 을 표시하므로, 남기면 접속 로그가 health 줄로 덮임. 실패한 응답은 기록."""

    def filter(self, record: logging.LogRecord) -> bool:
        a = record.args
        return not (isinstance(a, tuple) and len(a) >= 5 and a[1] == "GET" and str(a[2]).startswith("/v1/health")
                    and a[4] == 200)


def mask_access_logs() -> None:
    """접속 로그(uvicorn.access)와 WebSocket 접속 로그(uvicorn.error)에 토큰 가림 필터 부착 (uvicorn.access 에는 health 필터도 부착).
    uvicorn 이 서버 시작 때 로깅 설정을 다시 하므로 lifespan(설정이 끝난 뒤)에서 호출. 여러 번 호출해도 하나만 부착."""
    for name in ("uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, TokenMaskFilter) for f in lg.filters):
            lg.addFilter(TokenMaskFilter())
    acc = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, HealthAccessFilter) for f in acc.filters):
        acc.addFilter(HealthAccessFilter())


def create_app(st: Settings | None = None) -> FastAPI:
    st = st or default_settings
    deps = build_deps(st)
    manager = SessionManager(deps)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mask_access_logs()
        if st.backend != "sim":
            mask_number_logs(parse_numbers(line_spec(st)))  # SDK 로그에 찍히는 발신 번호를 '번호k' 로 가림
        await deps.backend.start()
        warm = getattr(deps.recognizer, "warmup", None)
        if warm:
            asyncio.create_task(warm())  # STT 모델은 백그라운드로 미리 로드
        warm_ai = getattr(deps.interp, "warmup", None)
        if warm_ai:
            asyncio.create_task(warm_ai())  # 로컬 LLM 해석기의 모델 미리 로드
        yield
        for s in list(manager.sessions.values()):
            if not s.ended:
                await s.finish("user_canceled")
        # 통화 기록 디스크 저장 완료 전 서버 종료 방지용 대기 (자동 재시작·Ctrl+C)
        saving = [t for s in manager.sessions.values() if (t := getattr(s, "_save_task", None)) and not t.done()]
        if saving:
            await asyncio.wait(saving, timeout=st.hangup_flush_s + 5)
        await deps.backend.stop()

    app = FastAPI(title="ARS 자동화 API", version="0.3", lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, e: ApiError):
        err = {"code": e.code, "message": e.message}
        if getattr(e, "reason", None):
            err["reason"] = e.reason
        return JSONResponse({"error": err}, status_code=e.status)

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
        combo = body.get("combo") if isinstance(body.get("combo"), dict) else {}
        out = await manager.create(phone, body.get("goal"), record=body.get("record") is True,
                                   interp=combo.get("interp"), fix=combo.get("fix"))
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
            # 이미 끝난 통화: snapshot(ended) 와 남은 이벤트만 보내고 닫음. 발행 대기열(받아쓰기 고침을 기다리는 자막)에
            # 이벤트가 남아 있으면 상태가 ended 여도 대기 (마지막 자막·ended 가 대기열 뒤에 있음)
            while not (s.events_done() and q.empty()):
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

    # 시험용 통화 기록 (record=true 로 만든 통화, 해당 session_token 보유자만) ---------------------------
    @app.get("/v1/sessions/{sid}/recording")
    async def recording(sid: str, authorization: str | None = Header(None)):
        s = manager.get(sid, bearer(authorization))
        if s.recorder is None:
            raise ApiError(404, "no_recording", "통화 기록이 없거나 이미 지워졌습니다.")
        if not s.ended:
            raise ApiError(409, "invalid_state", "통화가 끝난 뒤 받을 수 있습니다.")
        data = await asyncio.to_thread(s.recorder.export_zip)
        return Response(data, media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="ars-call-{sid}.zip"', "Cache-Control": "no-store"})

    @app.delete("/v1/sessions/{sid}/recording")
    async def delete_recording(sid: str, authorization: str | None = Header(None)):
        s = manager.get(sid, bearer(authorization))
        s.drop_recording()
        # 삭제 대상: 서버 메모리의 기록. RECORDING_SAVE=1 이면 서버 디스크의 recordings/ 사본은 유지 (수동 삭제 필요)
        return {"deleted": True, "disk_copy_kept": getattr(s, "_save_task", None) is not None}

    # 4.5 오디오 채널 ------------------------------------------------------------
    @app.websocket("/v1/sessions/{sid}/audio")
    async def audio(ws: WebSocket, sid: str, token: str | None = None):
        try:
            s = manager.get(sid, token)
        except ApiError as e:
            await ws.close(code=4401 if e.status == 401 else 4404, reason=e.code)
            return
        await ws.accept()
        if s.ended:
            await ws.close()
            return
        q: asyncio.Queue = asyncio.Queue()
        s.audio_out.add(q)

        async def downlink():
            while True:
                frame = await q.get()
                if frame is None:
                    break
                await ws.send_bytes(frame)

        down = asyncio.create_task(downlink())
        chunker = FrameChunker()  # 연결별로 남은 바이트 이어 붙임
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    for frame in chunker.feed(data):
                        await s.mic_audio(frame)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            s.audio_out.discard(q)
            down.cancel()

    # 시험용 앱 화면 ------------------------------------------------------------
    @app.get("/")
    async def index():
        return FileResponse(HERE / "static" / "index.html")

    # Android 앱과 같은 동작의 웹 화면 (Android 기기 없이 확인용)
    @app.get("/app")
    async def web_app():
        return FileResponse(HERE / "static" / "app.html", headers={"Cache-Control": "no-store"})

    @app.get("/v1/health")
    async def health(x_app_key: str | None = Header(None)):
        a = manager.active()
        ex = getattr(deps.interp, "goal_examples", None)
        cap = manager.capacity
        live = len(manager.active_sessions())
        if cap == 1:
            busy = a.id if a else None   # 번호 하나: 진행 중 session id (하위 호환 값)
        else:
            # 번호 여러 개: 다른 사용자의 session id 비공개. 빈 번호가 없을 때만 채우는 차단 값 (구버전 클라이언트의 발신 차단 기준)
            busy = "lines_full" if manager._pick_line() is None else None
        out = {"backend": st.backend, "stt": stt_kind(st), "sim_voice": st.sim_voice, "interpreter": st.interpreter,
               "mock_goals": ex() if ex else [], "active_session": busy,
               # 동시 통화 표시용 (인증 없이 공개되므로 개수만 포함): 진행 중인 통화 수(마지막 안내 처리 중 포함), 번호 수·사용 중
               "active_sessions": live, "concurrent": live >= 2,
               "lines": {"total": cap, "in_use": manager.lines_in_use()}}
        if x_app_key == st.app_key:
            u = getattr(deps.backend, "usable_count", None)
            out["lines"]["usable"] = u() if u else cap
        if manager.combos is not None:
            out["combos"] = manager.combos.options()
        return out

    return app


def setup_logging(debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("clawops").setLevel(logging.INFO if debug else logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)  # 파일 변경 감지 로그 반복 출력 방지


def reload_app() -> FastAPI:
    """자동 재시작 때 uvicorn 이 새 프로세스에서 호출하는 앱 생성 함수. 설정은 main() 이 남긴 환경변수에서 다시 읽음."""
    setup_logging(os.environ.get("ARS_DEBUG") == "1")
    return create_app(Settings())


def reload_excludes() -> list[str]:
    """변경돼도 서버 재시작 대상이 아닌 파일·폴더 목록. 폴더는 절대 경로로 넘겨야 uvicorn 이 폴더로 인식."""
    dirs = [HERE / d for d in (".venv", ".venv-qwen", ".git", "bench", "bench_real", "bench_clips", "_to_delete", "docs", "recordings")]
    return ["test_*.py", *(str(d) for d in dirs if d.is_dir())]


def main() -> None:
    ap = argparse.ArgumentParser(description="ARS 자동화 API 서버")
    ap.add_argument("--sim", action="store_true", help="가상 ARS 로 실행")
    ap.add_argument("--sim-voice", action="store_true", help="음성 가상 ARS (TTS 안내 → STT, DTMF 톤 인식)")
    ap.add_argument("--sim-tree", help="가상 ARS 메뉴 트리 (예: sim_nhis.json)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-reload", action="store_true", help="코드가 바뀌어도 자동 재시작하지 않음 (시연·실통화 시험 중 권장)")
    a = ap.parse_args()
    setup_logging(a.debug)
    # 명령행 옵션의 환경변수 전달: 자동 재시작으로 뜨는 새 프로세스도 같은 설정을 읽게 하기 위함
    if a.sim or a.sim_voice:
        os.environ["CALL_BACKEND"] = "sim"
    if a.sim_voice:
        os.environ["SIM_VOICE"] = "1"
    if a.sim_tree:
        os.environ["SIM_TREE"] = a.sim_tree
    os.environ["ARS_DEBUG"] = "1" if a.debug else "0"
    st = Settings()
    import uvicorn

    log.info("백엔드=%s%s STT=%s 해석기=%s", st.backend, "(음성)" if st.sim_voice else "", stt_kind(st), st.interpreter)
    if a.no_reload:
        uvicorn.run(create_app(st), host=a.host, port=a.port, log_level="info")
        return
    log.info("자동 재시작 켜짐: .py 파일이 바뀌면 서버를 다시 시작한다 (진행 중 통화는 끊김). 끄려면 --no-reload")
    uvicorn.run("app:reload_app", factory=True, reload=True, reload_dirs=[str(HERE)], reload_excludes=reload_excludes(),
                host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
