"""
API 호출 확인 (HTTP·WebSocket 요청만 사용)

실행 중인 서버에 명세서 4장의 API 를 앱과 같은 방식으로 직접 호출하고,
요청(→)·응답(←)·이벤트(⇐)를 시간순으로 출력한다. 시험 화면을 거치지 않는다.

  1) 서버 실행 (별도 터미널)
       python app.py --sim        가상 ARS (실통화·과금 없음)
       python app.py              ClawOps 실통화
  2) 확인 실행
       python api_check.py check      전체 API 점검 (가상 ARS 서버 전용). 엔드포인트·오류 코드·이벤트·오디오 채널을
                                      차례로 호출하고 통과/실패를 표로 출력한다. 실패가 있으면 종료 코드 1
       python api_check.py goal       목적과 함께 발신 → 탐색 → 민감정보 직접 입력 → 결과(핵심 값) → 종료
       python api_check.py lowconf    신뢰도 낮음 → low_confidence 재질문 → 답변 → 탐색 → 결과 → 종료
       python api_check.py revise     첫 메뉴 재질문 중 다시 말하기(revise_goal) → 결과 → 첫 번호 뒤 다시 말하기는 409 → 종료
       python api_check.py real --phone 15889999 --goal "상담원 연결해줘" --keys "12:3" --sec 45
                                      실통화: 발신 → (지정 시각에 직접 선택 키) → 자막·경로 확인 → 종료
  오디오 채널 확인: --audio 를 붙이면 오디오 채널(4.5)에 연결하여 받은 음성을 WAV 파일로 저장하고
                   프레임 형식·수신 비율·끊김·음량을 출력한다. check 는 항상 확인한다. 저장 파일은 --wav 로 바꾼다.
옵션: --base http://127.0.0.1:8000  --app-key dev-app-key
      (Windows 에서 localhost 는 IPv6 연결을 먼저 시도하여 WebSocket 연결이 2초가량 늦어지므로 127.0.0.1 을 기본으로 둔다)
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import math
import os
import sys
import time
import wave
from collections import Counter

import httpx
import websockets

T0 = time.monotonic()
RATE = 8000
VOICE_RMS = 500  # 소리 구간 판정 기준. 서버의 발화 판정 기본값(VAD_RMS)과 같다
STALL_MS = 300   # 이 시간 이상 프레임이 오지 않으면 재생이 끊긴 것으로 본다
# 주소가 IPv6·IPv4 둘 다로 풀리면 0.25초 뒤 다음 주소도 함께 시도한다 (Windows 의 localhost → ::1 연결 지연 방지)
WS_OPTS = {"happy_eyeballs_delay": 0.25}
SIM_GOAL = "통장 잔액 알고 싶어"  # check: 목 데이터에서 민감정보 직접 입력 3회 → 결과까지 가는 목적


def ts() -> str:
    return f"[{time.monotonic() - T0:6.2f}s]"


def show(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def err_code(d) -> str | None:
    return (d.get("error") or {}).get("code") if isinstance(d, dict) else None


# ---------------------------------------------------------------------------
# μ-law 복호 (G.711). 서버 코드에 의존하지 않도록 직접 구현한다
# ---------------------------------------------------------------------------
def _ulaw_sample(b: int) -> int:
    u = ~b & 0xFF
    t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
    return (0x84 - t) if u & 0x80 else (t - 0x84)


ULAW_TABLE = [_ulaw_sample(b) for b in range(256)]
_ULAW_SQ = [x * x for x in ULAW_TABLE]


def ulaw_to_pcm16(data: bytes) -> bytes:
    a = array.array("h", (ULAW_TABLE[b] for b in data))
    if sys.byteorder == "big":
        a.byteswap()  # WAV 는 리틀 엔디언
    return a.tobytes()


def frame_rms(frame: bytes) -> float:
    return math.sqrt(sum(_ULAW_SQ[b] for b in frame) / max(1, len(frame)))


def dbfs(rms: float) -> float:
    return 20 * math.log10(max(rms, 1e-9) / 32768)


# ---------------------------------------------------------------------------
# 점검 결과
# ---------------------------------------------------------------------------
class Checks:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []

    def add(self, name: str, ok: bool | None, detail: str = "") -> bool:
        mark = "생략" if ok is None else "통과" if ok else "실패"
        self.items.append((mark, name, detail))
        print(f"{ts()} [{mark}] {name}" + (f" - {detail}" if detail else ""))
        return bool(ok)

    @property
    def failed(self) -> int:
        return sum(1 for m, _, _ in self.items if m == "실패")

    def report(self, title: str = "API 점검 결과") -> int:
        cnt = Counter(m for m, _, _ in self.items)
        print(f"\n===== {title}: {len(self.items)}개 중 통과 {cnt['통과']}, 실패 {cnt['실패']}, 생략 {cnt['생략']} =====")
        for mark, name, detail in self.items:
            print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))
        return 1 if self.failed else 0


# ---------------------------------------------------------------------------
# 오디오 채널 확인 (4.5)
# ---------------------------------------------------------------------------
class AudioProbe:
    """오디오 채널에 연결하여(소리 켜기) 받은 프레임을 모은다. 끝나면 WAV 로 저장하고 수신 상태를 계산한다."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.frames: list[bytes] = []
        self.arrivals: list[float] = []
        self.error: str | None = None
        self.task: asyncio.Task | None = None

    def start(self, url: str) -> None:
        self.task = asyncio.create_task(self._run(url))

    @property
    def seconds(self) -> float:
        return sum(len(f) for f in self.frames) / RATE

    async def _run(self, url: str) -> None:
        try:
            async with websockets.connect(url, **WS_OPTS) as ws:
                print(f"{ts()} ⇐ audio           오디오 채널 연결 (소리 켜기)")
                async for msg in ws:
                    if isinstance(msg, str):
                        continue
                    self.arrivals.append(time.monotonic())
                    self.frames.append(msg)
                    n = len(self.frames)
                    if n == 1:
                        print(f"{ts()} ⇐ audio           첫 프레임 수신 ({len(msg)}바이트)")
                    elif n % 500 == 0:
                        print(f"{ts()} ⇐ audio           누적 {n}프레임 ({self.seconds:.1f}초)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"{ts()} ⇐ audio           연결 오류 {self.error}")

    async def stop(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass

    def analyze(self) -> dict:
        n = len(self.frames)
        r = {"frames": n, "seconds": self.seconds, "sizes": Counter(len(f) for f in self.frames)}
        if n == 0:
            return r
        span = self.arrivals[-1] - self.arrivals[0] + len(self.frames[-1]) / RATE
        gaps = [b - a for a, b in zip(self.arrivals, self.arrivals[1:])]
        levels = [frame_rms(f) for f in self.frames]
        voiced = [x for x in levels if x >= VOICE_RMS]
        r.update(
            coverage=min(1.0, self.seconds / span) if span > 0 else 1.0,  # 받은 음성 길이 ÷ 받는 데 걸린 시간
            gaps=sum(1 for g in gaps if g >= 0.1),  # 참고: 네트워크·서버 지연으로 늦게 도착한 횟수
            max_gap_ms=round(max(gaps) * 1000) if gaps else 0,
            voiced=len(voiced) / n,
            voiced_dbfs=round(dbfs(math.sqrt(sum(x * x for x in voiced) / len(voiced))), 1) if voiced else None,
        )
        return r

    def save_wav(self) -> None:
        with wave.open(self.path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(ulaw_to_pcm16(b"".join(self.frames)))


REAL_AUDIO_CMD = 'python api_check.py real --phone 15889999 --goal "상담원 연결해줘" --sec 30 --audio'


def audio_checks(ck: Checks, p: AudioProbe, sim: bool | None = None) -> None:
    r = p.analyze()
    if r["frames"] == 0:
        ck.add("WS audio 연결·프레임 수신", False,
               p.error or "프레임 없음 (통화가 연결되지 않았거나 가상 ARS 를 SIM_AUDIO=0 으로 실행함)")
        return
    p.save_wav()
    sizes = ", ".join(f"{k}바이트×{v}" for k, v in sorted(r["sizes"].items()))
    ck.add("WS audio 연결·프레임 수신", True, f"{r['frames']}프레임, {r['seconds']:.1f}초")
    ck.add("WS audio 프레임 형식 (μ-law 8kHz, 160바이트=20ms)", set(r["sizes"]) == {160}, sizes)
    # 빠진 프레임(수신 비율)과 긴 멈춤만 실패로 본다. 100ms 안팎의 지연은 앱의 재생 버퍼가 흡수하므로 참고 수치로 출력한다
    ck.add("WS audio 끊김 (수신 비율 95% 이상, 300ms 이상 멈춤 없음)",
           r["coverage"] >= 0.95 and r["max_gap_ms"] < STALL_MS,
           f"수신 비율 {r['coverage']:.1%}, 최대 간격 {r['max_gap_ms']}ms, 100ms 이상 간격 {r['gaps']}회")
    ck.add("WS audio 소리 포함", r["voiced"] > 0,
           f"소리 구간 {r['voiced']:.0%}" + (f", 평균 {r['voiced_dbfs']} dBFS" if r["voiced_dbfs"] is not None else ""))
    print(f"\n오디오 저장: {os.path.abspath(p.path)} (8kHz 16bit mono, {r['seconds']:.1f}초)")
    if sim is not False:
        print("  가상 ARS 는 음성이 없다. 안내 문장을 글자로 서버에 직접 전달하고, 오디오 채널에는 안내가 나오는 동안\n"
              "  400Hz '삐' 시험음만 보낸다. 끊김 없이 이어지면 오디오 채널은 정상이다.")
        print(f"  실제 ARS 음성 확인: 서버를 python app.py(실통화)로 실행한 뒤 {REAL_AUDIO_CMD}")
    if sim is not True:
        print("  실통화 음성이다. 재생하여 ARS 안내가 끊기지 않고 또렷하게 들리는지 확인한다.")


# ---------------------------------------------------------------------------
# 클라이언트
# ---------------------------------------------------------------------------
class Client:
    def __init__(self, base: str, app_key: str) -> None:
        self.base = base.rstrip("/")
        self.app_key = app_key
        self.http = httpx.AsyncClient(timeout=30)
        self.sid = self.token = None
        self.urls: dict[str, str] = {}
        self.events: list[dict] = []
        self.status = None
        self.new_event = asyncio.Event()

    async def request(self, method: str, path: str, body: dict | None = None, secret: bool = False,
                      headers: dict | None = None) -> tuple[int, dict]:
        if headers is None:  # 세션 생성은 앱 키, 그 외는 세션 토큰 (2.2)
            headers = ({"X-App-Key": self.app_key} if (method == "POST" and path == "/v1/sessions")
                       else {"Authorization": f"Bearer {self.token}"} if self.token else {})
        shown = {**body, "digits": "*" * len(body["digits"])} if (body and secret) else body
        print(f"{ts()} → {method} {path} {show(shown) if shown else ''}")
        r = await self.http.request(method, self.base + path, json=body, headers=headers)
        try:
            data = r.json()
        except ValueError:
            data = {"_text": r.text[:200]}
        shown = {**data, "session_token": data["session_token"][:7] + "…"} if isinstance(data, dict) and "session_token" in data else data
        print(f"{ts()} ← {r.status_code} {show(shown)}")  # 세션 토큰은 앞부분만 출력 (로그 공유 대비)
        return r.status_code, data

    async def call(self, method: str, path: str, body: dict | None = None, secret: bool = False) -> dict:
        return (await self.request(method, path, body, secret))[1]

    async def listen(self) -> None:
        async with websockets.connect(f"{self.urls['events']}?token={self.token}", **WS_OPTS) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                self.events.append(msg)
                d = msg.get("data", {})
                if msg["event"] == "status_changed":
                    self.status = d["status"]
                elif msg["event"] == "snapshot":
                    self.status = d["status"]
                    d = {k: d[k] for k in ("status", "goal_summary") if k in d}
                print(f"{ts()} ⇐ {msg['event']:<15} {show(d)}")
                self.new_event.set()
                if msg["event"] == "ended":
                    return

    async def wait(self, cond, timeout: float = 60, fail: bool = True) -> dict | None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for e in self.events:
                if cond(e):
                    return e
            self.new_event.clear()
            try:
                await asyncio.wait_for(self.new_event.wait(), end - time.monotonic())
            except asyncio.TimeoutError:
                break
        if fail:
            raise SystemExit(f"\n기다리던 이벤트가 오지 않았습니다 (현재 상태: {self.status})")
        return None

    def seen(self, cond) -> bool:
        return any(cond(e) for e in self.events)


def is_status(st):
    return lambda e: (e["event"] == "status_changed" and e["data"]["status"] == st) or \
                     (e["event"] == "snapshot" and e["data"]["status"] == st)


def is_event(name):
    return lambda e: e["event"] == name


def request_of(reason):
    return lambda e: e["event"] == "request" and e["data"].get("reason") == reason


def begin(c: Client, out: dict, probe: AudioProbe | None) -> asyncio.Task:
    """세션 생성 응답의 events_url·audio_url 로 이벤트 채널과 (선택) 오디오 채널에 연결한다."""
    c.sid, c.token = out["session_id"], out["session_token"]
    ws_base = c.base.replace("http", "ws", 1)
    c.urls = {"events": out.get("events_url") or f"{ws_base}/v1/sessions/{c.sid}/events",
              "audio": out.get("audio_url") or f"{ws_base}/v1/sessions/{c.sid}/audio"}
    if probe:
        probe.start(f"{c.urls['audio']}?token={c.token}")
    return asyncio.create_task(c.listen())


async def open_session(c: Client, phone: str, goal: str, probe: AudioProbe | None = None) -> asyncio.Task:
    st, out = await c.request("POST", "/v1/sessions", {"phone_number": phone, "goal": goal})
    if st != 201 or "session_id" not in out:
        raise SystemExit("세션 생성 실패")
    return begin(c, out, probe)


async def finish(c: Client, listener: asyncio.Task) -> None:
    if not c.seen(is_event("ended")):
        await c.call("POST", f"/v1/sessions/{c.sid}/actions", {"type": "cancel"})
    await asyncio.wait_for(listener, 15)
    await c.call("GET", f"/v1/sessions/{c.sid}/result")


# ---------------------------------------------------------------------------
# 전체 API 점검 (가상 ARS 서버 전용)
# ---------------------------------------------------------------------------
async def scenario_check(c: Client, phone: str, wav: str) -> int:
    ck = Checks()
    actions = lambda: f"/v1/sessions/{c.sid}/actions"  # noqa: E731
    result = lambda: f"/v1/sessions/{c.sid}/result"  # noqa: E731

    st, h = await c.request("GET", "/v1/health")
    ck.add("GET /v1/health → 200", st == 200, f"backend={h.get('backend')}, interpreter={h.get('interpreter')}")
    if h.get("backend") != "sim":
        print("\ncheck 는 실제 발신과 과금을 막기 위해 가상 ARS 서버에서만 실행한다. 서버를 python app.py --sim 으로 실행하십시오.")
        print("실통화 확인은 python api_check.py real --phone ... --goal ... --audio 를 사용한다.")
        return 2
    if h.get("active_session"):
        print("\n서버에 진행 중인 통화가 있습니다. 서버를 다시 실행한 뒤 점검하십시오.")
        return 2

    # 세션 생성 (4.1)
    st, d = await c.request("POST", "/v1/sessions", {"phone_number": phone, "goal": SIM_GOAL},
                            headers={"X-App-Key": "wrong-app-key"})
    ck.add("POST /v1/sessions 앱 키 오류 → 401 unauthorized", st == 401 and err_code(d) == "unauthorized")
    st, d = await c.request("POST", "/v1/sessions", {"phone_number": phone})
    ck.add("POST /v1/sessions 목적(goal) 누락 → 400 invalid_request", st == 400 and err_code(d) == "invalid_request")

    probe = AudioProbe(wav)
    st, out = await c.request("POST", "/v1/sessions", {"phone_number": phone, "goal": SIM_GOAL})
    need = ["session_id", "session_token", "events_url", "audio_url", "phone_number", "status", "goal_summary"]
    missing = [k for k in need if k not in out]
    if not ck.add("POST /v1/sessions → 201 (생성 즉시 발신)", st == 201 and not missing and out.get("status") == "dialing",
                  f"누락 필드 {missing}" if missing else f"status={out.get('status')}"):
        return ck.report()
    listener = begin(c, out, probe)
    st, d = await c.request("POST", "/v1/sessions", {"phone_number": phone, "goal": SIM_GOAL})
    ck.add("POST /v1/sessions 통화 중 재요청 → 409 invalid_state (동시 통화 1건)",
           st == 409 and err_code(d) == "invalid_state")

    # 이벤트 채널 (4.2)
    snap = await c.wait(is_event("snapshot"), 10, fail=False)
    if not ck.add("WS events 연결 → snapshot 수신", snap is not None):
        await c.request("POST", actions(), {"type": "cancel"})
        await probe.stop()
        return ck.report()

    # 오류 응답 (2.3)
    st, d = await c.request("POST", actions(), {"type": "no_such_type"})
    ck.add("POST actions 알 수 없는 type → 400 invalid_request", st == 400 and err_code(d) == "invalid_request")
    st, d = await c.request("GET", result())
    ck.add("GET result 통화 중 → 202", st == 202)
    st, d = await c.request("GET", result(), headers={"Authorization": "Bearer wrong-token"})
    ck.add("GET result 토큰 오류 → 401 unauthorized", st == 401 and err_code(d) == "unauthorized")
    st, d = await c.request("GET", "/v1/sessions/ses_00000000/result")
    ck.add("GET result 없는 세션 → 404 not_found", st == 404 and err_code(d) == "not_found")

    # 민감정보 직접 입력 (6.3)
    answered: set[str] = set()
    inputs: list[bool] = []
    while True:
        e = await c.wait(lambda e: (e["event"] == "request" and e["data"]["kind"] == "secure_input"
                                    and e["data"]["request_id"] not in answered)
                         or is_status("agent_done")(e) or e["event"] == "ended", 60, fail=False)
        if e is None or e["event"] != "request":
            break
        rid = e["data"]["request_id"]
        answered.add(rid)
        if not inputs:
            st, d = await c.request("POST", actions(), {"type": "key", "key": "1"})
            ck.add("POST actions key (민감정보 입력 요청 중) → 409 sensitive_input_blocked",
                   st == 409 and err_code(d) == "sensitive_input_blocked")
        st, d = await c.request("POST", actions(), {"type": "secure_input", "request_id": rid, "digits": "1234"},
                                secret=True)
        inputs.append(st == 200 and d.get("accepted") is True)
    ck.add("POST actions secure_input → 200", len(inputs) == 3 and all(inputs), f"{sum(inputs)}/{len(inputs)}회 (기대 3회)")

    # 이벤트 종류·형식 (5장·6장)
    names = {e["event"] for e in c.events}
    need_ev = ["status_changed", "caption", "path_updated", "request", "summary"]
    ck.add("WS events status_changed·caption·path_updated·request·summary 수신", all(n in names for n in need_ev),
           f"받지 못함 {[n for n in need_ev if n not in names]}" if not all(n in names for n in need_ev) else "")
    paths = [e["data"] for e in c.events if e["event"] == "path_updated"]
    ck.add("PathStep 형식 (step·text·options·selected·actor, AI 선택 신뢰도)",
           bool(paths) and all({"step", "text", "options", "selected", "actor"} <= set(p) for p in paths)
           and paths[0]["selected"].get("confidence") is not None, f"경로 {len(paths)}건")
    summ = next((e["data"] for e in c.events if e["event"] == "summary"), None)
    ck.add("summary(result) 형식 (summary·key_info)",
           bool(summ) and summ.get("kind") == "result" and bool(summ.get("summary")) and bool(summ.get("key_info")))
    ck.add("status_changed(agent_done) 수신 (결과 뒤 통화 유지)", c.seen(is_status("agent_done")))

    # 업무 완료 뒤 조작 (4.3)
    st, d = await c.request("POST", actions(), {"type": "revise_goal", "text": "카드 결제일 바꾸고 싶어"})
    ck.add("POST actions revise_goal (첫 번호 이후) → 409 invalid_state", st == 409 and err_code(d) == "invalid_state")
    st, d = await c.request("POST", actions(), {"type": "cancel"})
    if st == 409 and c.seen(is_event("ended")):
        ck.add("POST actions cancel → 200", None, "ARS 가 먼저 통화를 종료")
    else:
        ck.add("POST actions cancel → 200", st == 200 and d.get("accepted") is True)
    end = await c.wait(is_event("ended"), 15, fail=False)
    reason = end["data"].get("end_reason") if end else None
    ck.add("WS events ended(completed) 수신", reason == "completed", f"end_reason={reason}")
    try:
        await asyncio.wait_for(listener, 15)
    except Exception:
        pass

    # 결과 조회 (4.4)
    st, d = await c.request("GET", result())
    keys = ["outcome", "end_reason", "summary", "key_info", "handoff_script", "path", "duration_sec"]
    ck.add("GET result 종료 후 → 200 (outcome·summary·key_info·path·duration_sec)",
           st == 200 and all(k in d for k in keys) and d.get("outcome") == "success", f"outcome={d.get('outcome')}")
    st, d = await c.request("POST", actions(), {"type": "cancel"})
    ck.add("POST actions 종료 후 → 409 invalid_state", st == 409 and err_code(d) == "invalid_state")

    # 오디오 채널 (4.5)
    await probe.stop()
    audio_checks(ck, probe, sim=True)
    return ck.report()


# ---------------------------------------------------------------------------
# 흐름별 시나리오
# ---------------------------------------------------------------------------
async def scenario_goal(c: Client, phone: str, probe: AudioProbe | None) -> None:
    listener = await open_session(c, phone, SIM_GOAL, probe)
    answered: set[str] = set()
    while not c.seen(is_status("agent_done")) and not c.seen(is_event("ended")):
        e = await c.wait(lambda e: (e["event"] == "request" and e["data"]["kind"] == "secure_input"
                                    and e["data"]["request_id"] not in answered)
                         or e["event"] == "ended" or is_status("agent_done")(e))
        if e["event"] == "request":
            answered.add(e["data"]["request_id"])
            await c.call("POST", f"/v1/sessions/{c.sid}/actions",
                         {"type": "secure_input", "request_id": e["data"]["request_id"], "digits": "1234"}, secret=True)
    await finish(c, listener)


async def scenario_lowconf(c: Client, phone: str, probe: AudioProbe | None) -> None:
    listener = await open_session(c, phone, "카드 관련 문의", probe)
    req = (await c.wait(request_of("low_confidence")))["data"]
    await c.call("POST", f"/v1/sessions/{c.sid}/actions",
                 {"type": "answer", "request_id": req["request_id"], "text": "카드 결제일 바꾸고 싶어"})
    await c.wait(lambda e: is_status("agent_done")(e) or e["event"] == "ended")
    await finish(c, listener)


async def scenario_revise(c: Client, phone: str, probe: AudioProbe | None) -> None:
    listener = await open_session(c, phone, "카드 관련 문의", probe)
    await c.wait(request_of("low_confidence"))  # 첫 메뉴에서 불확실. 아직 번호를 누르기 전(목적 설정 단계)
    await c.call("POST", f"/v1/sessions/{c.sid}/actions", {"type": "revise_goal", "text": "이번 달 카드 대금 알고 싶어"})
    await c.wait(lambda e: is_status("agent_done")(e) or e["event"] == "ended")
    # 첫 번호를 누른 뒤에는 다시 말하기를 받지 않는다 (409)
    await c.call("POST", f"/v1/sessions/{c.sid}/actions", {"type": "revise_goal", "text": "카드 결제일 바꾸고 싶어"})
    await finish(c, listener)


async def scenario_real(c: Client, phone: str, goal: str, keys: str, sec: float, probe: AudioProbe | None) -> None:
    listener = await open_session(c, phone, goal, probe)
    plan = sorted((float(t), k.strip()) for t, k in (p.split(":", 1) for p in keys.split(",") if p.strip()))
    start = time.monotonic()
    for at, k in plan:
        await asyncio.sleep(max(0.0, start + at - time.monotonic()))
        if listener.done():
            break
        await c.call("POST", f"/v1/sessions/{c.sid}/actions", {"type": "key", "key": k})
    await asyncio.sleep(max(0.0, start + sec - time.monotonic()))
    await finish(c, listener)


async def main() -> int:
    try:  # Windows 에서 출력을 파일로 돌려도 글자가 깨지지 않도록 UTF-8 로 출력한다
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="API 호출 확인 (요청만 사용)")
    ap.add_argument("scenario", choices=["check", "goal", "lowconf", "revise", "real"])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--app-key", default="dev-app-key")
    ap.add_argument("--phone", default="15889999")
    ap.add_argument("--goal", default="상담원 연결해줘", help="real: 목적 (발신 시 필수)")
    ap.add_argument("--keys", default="", help="real: 직접 선택할 '초:키' 목록. 예: 12:3,25:2")
    ap.add_argument("--sec", type=float, default=40, help="real: 통화 유지 시간(초)")
    ap.add_argument("--audio", action="store_true", help="오디오 채널에 연결하여 받은 음성을 WAV 로 저장하고 끊김·음량을 확인")
    ap.add_argument("--wav", default="audio_check.wav", help="오디오 저장 파일 (기본 audio_check.wav)")
    a = ap.parse_args()
    c = Client(a.base, a.app_key)
    probe = AudioProbe(a.wav) if a.audio else None
    try:
        if a.scenario == "check":
            return await scenario_check(c, a.phone, a.wav)
        _, h = await c.request("GET", "/v1/health")
        if a.scenario == "goal":
            await scenario_goal(c, a.phone, probe)
        elif a.scenario == "lowconf":
            await scenario_lowconf(c, a.phone, probe)
        elif a.scenario == "revise":
            await scenario_revise(c, a.phone, probe)
        else:
            await scenario_real(c, a.phone, a.goal, a.keys, a.sec, probe)
        if probe is None:
            return 0
        await probe.stop()
        ck = Checks()
        audio_checks(ck, probe, sim=h.get("backend") == "sim")
        return ck.report("오디오 채널 확인 결과")
    except httpx.ConnectError:
        print(f"\n서버에 연결할 수 없습니다 ({a.base}). 다른 터미널에서 python app.py --sim 으로 서버를 먼저 실행하십시오.")
        return 2
    finally:
        if probe:
            await probe.stop()
        await c.http.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
