"""
통화 백엔드

  ClawOpsBackend : ClawOps 발신 번호로 실제 발신 (Voice Agent SDK). 번호 하나당 통화 1건.
                   발신 번호 여러 개(CLAWOPS_LINES, 쉼표 구분): 번호마다 ClawOpsAgent 하나씩 연결, 통화마다 빈 번호로 발신.
                   ClawOps 의 연결 자리는 번호당 하나: 같은 번호로 두 통화를 하거나 다른 프로세스가 같은 번호를 쓰면
                   먼저 쓰던 쪽의 소리 끊김.
  SimBackend     : 가상 ARS. 메뉴 트리(JSON)를 따라 안내 문장 직접 전달. 실제 발신 없이 전체 흐름 시험용.

서버(session)는 Call 프로토콜만 사용.
  send_audio(ulaw)   : 사용자 마이크 음성 송출 (20ms 프레임)
  send_keys(keys)    : DTMF 전송 (in-band 톤). 반환 뒤에도 남은 송출 시간(초) 반환
  hangup()
콜백(CallHandlers)
  on_answer()        : 상대 응답
  on_audio(ulaw)     : 상대(ARS·상담원) 음성 프레임
  on_transcript(txt, started_at) : 백엔드가 문장을 직접 주는 경우(sim). started_at: 안내 발화 시작 시각(monotonic). ClawOps 백엔드는 미사용
  on_end(reason)     : no_answer / busy / failed / remote_hangup / local_hangup
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from audio import FRAME, SILENCE_FRAME, dtmf_ulaw, hum_ulaw

log = logging.getLogger("ars.call")


@dataclass
class CallHandlers:
    on_answer: Callable[[], Awaitable[None]]
    on_audio: Callable[[bytes], Awaitable[None]]
    on_transcript: Callable[..., Awaitable[None]]
    on_end: Callable[[str], Awaitable[None]]


class Call(Protocol):
    provides_transcripts: bool

    async def send_audio(self, ulaw: bytes) -> None: ...
    async def send_keys(self, keys: str) -> float:
        """키 전송. 반환값: 반환 뒤에도 톤이 더 나갈 수 있는 시간(초). session 쪽은 그동안(+메아리 여유) 수신음 무시."""
    async def hangup(self) -> None: ...


class Backend(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def dial(self, to: str, handlers: CallHandlers) -> Call: ...
    # 번호 여러 개를 쓰는 백엔드: capacity(동시 발신 가능 통화 수)·line_rank(i)·line_name(i)·usable_count() 제공,
    # dial(to, handlers, line=i) 로 번호 지정. 없으면 통화 1건 (SessionManager 에서 getattr 로 확인)


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def parse_numbers(spec: str) -> list[str]:
    """쉼표로 나눈 발신 번호 목록 → 숫자만 남긴 번호 목록 (적힌 순서, 같은 번호는 한 번만).
    번호 하나(쉼표 없음)의 결과: 숫자만 이어 붙인 값과 동일. 국제 표기(82…)와 국내 표기(0…)는 같은 번호로 취급:
    한 프로세스의 두 ClawOpsAgent 가 같은 번호 자리를 다투면 한쪽의 번호 상실. 번호 값은 로그 기록 금지."""
    out: list[str] = []
    seen: set[str] = set()
    for i, part in enumerate((spec or "").split(",")):
        d = _digits(part)
        if not d:
            continue
        k = "0" + d[2:] if d.startswith("82") else d
        if k in seen:
            log.warning("발신 번호 목록 %d번째 항목이 앞의 번호와 같아 한 번만 쓴다", i + 1)
            continue
        seen.add(k)
        out.append(d)
    return out


class NumberMaskFilter(logging.Filter):
    """로그 문장의 발신 번호 값을 '번호k' 로 치환. SDK(clawops) 로그의 연결 주소(…/agent/listen?number=…)와
    오류 문구에 번호가 그대로 남는 문제 대응. 로거 필터는 자식 로거의 기록에 적용되지 않으므로 처리기(handler)에 부착."""

    def __init__(self, numbers: list[str]) -> None:
        super().__init__()
        sep = r"[\s\-]?"
        self.subs: list[tuple[re.Pattern, str]] = []
        for i, n in enumerate(numbers):
            if not n:
                continue
            self.subs.append((re.compile(sep.join(n)), f"번호{i + 1}"))
            if n.startswith("0") and len(n) > 1:  # 국제 표기(+82 70…)로 찍히는 경우
                self.subs.append((re.compile(r"\+?82" + sep + sep.join(n[1:])), f"번호{i + 1}"))
        self._num = re.compile(r"(number=)[^&\s\"']+")

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.subs:
            return True
        try:
            msg = record.getMessage()
        except Exception:
            return True
        out = self._num.sub(r"\1…", msg)
        for pat, name in self.subs:
            out = pat.sub(name, out)
        if out != msg:
            record.msg, record.args = out, None
        return True


def mask_number_logs(numbers: list[str]) -> None:
    """루트 처리기에 번호 가림 필터 부착 (여러 번 호출해도 처리기마다 하나). 번호가 없으면 아무 동작 없음."""
    if not numbers:
        return
    for h in logging.getLogger().handlers:
        if not any(isinstance(f, NumberMaskFilter) for f in h.filters):
            h.addFilter(NumberMaskFilter(numbers))


def mask_dest(to: str) -> str:
    """로그용 착신 번호: 끝 4자리만 표시 (4자리 이하는 그대로)."""
    d = _digits(to)
    return d if len(d) <= 4 else "*" * (len(d) - 4) + d[-4:]


# ---------------------------------------------------------------------------
# ClawOps
# ---------------------------------------------------------------------------
class _Leg:
    """ClawOps Session Protocol 구현. SDK 가 통화마다 생성, 수신 오디오 전달."""

    def __init__(self) -> None:
        self.call = None

    async def prewarm(self) -> None:
        pass

    async def start(self, call) -> None:
        self.call = call
        ClawOpsBackend.LEGS[call.call_id] = self

    async def attach(self, call) -> None:
        await self.start(call)

    async def feed_audio(self, audio: bytes, timestamp: int) -> None:
        c = ClawOpsBackend.CALLS.get(self.call.call_id) if self.call else None
        if c is not None:
            await c.handlers.on_audio(audio)

    async def feed_dtmf(self, digits: str) -> None:
        pass

    async def stop(self) -> None:
        pass

    def get_telemetry(self):
        return None


class ClawOpsCall:
    provides_transcripts = False

    def __init__(self, session, handlers: CallHandlers, dtmf_mode: str = "inband", backend=None, line: int = 0) -> None:
        self.session = session
        self.handlers = handlers
        self.dtmf_mode = dtmf_mode
        self.backend = backend
        self.line = line            # 이 통화의 발신 번호 (ClawOpsBackend.lines 의 순번)
        self._tone_until = 0.0
        self._hung_up = False       # 이쪽에서 끊기로 함 (call_end 를 local_hangup 으로 처리)
        self._media_closed = False  # 응답 뒤 미디어 채널 종료 전송함
        self._canceled = False      # 울리는 중 REST 로 통화 종료 전송함

    @property
    def leg(self) -> _Leg | None:
        return ClawOpsBackend.LEGS.get(self.session.call_id)

    async def send_audio(self, ulaw: bytes) -> None:
        leg = self.leg
        if leg and leg.call and time.monotonic() >= self._tone_until:
            await leg.call.send_audio(ulaw)

    async def send_keys(self, keys: str) -> float:
        leg = self.leg
        if not (leg and leg.call):
            raise RuntimeError("통화가 연결되지 않았습니다")
        if self.dtmf_mode == "sdk":
            # 미디어 채널의 DTMF 이벤트로 전송 (톤을 음성에 섞지 않음). 송출 시간은 톤 방식과 같은 값으로 계산
            await self.session.send_dtmf_sequence(keys)
            return len(dtmf_ulaw(keys)) / 8000
        tone = dtmf_ulaw(keys)
        dur = len(tone) / 8000
        self._tone_until = time.monotonic() + dur + 0.05  # 톤 송출 중 마이크 음성 차단
        # 실시간 속도로 전송 (20ms 프레임 단위). 한꺼번에 밀어 넣으면 전송 쪽 버퍼에서 톤이 겹치거나 잘려
        # 긴 번호(전화번호 등)의 일부를 ARS 가 인식하지 못할 가능성 있음
        t0 = time.monotonic()
        for n, i in enumerate(range(0, len(tone), FRAME)):
            await leg.call.send_audio(tone[i:i + FRAME])
            wait = t0 + (n + 1) * FRAME / 8000 - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        # 실시간 속도로 모두 보낸 뒤 반환하므로 남은 톤 없음 (반환값 = 반환 뒤에도 톤이 더 나갈 수 있는 시간).
        # 톤 길이를 반환하면 session 쪽이 전송 뒤 그만큼 더 수신음을 무시하여, 번호 입력 직후 ARS 가 바로 시작한 되읽기의
        # 앞부분('입력하신 번호는')과 첫 숫자를 놓치는 문제 방지
        return 0.0

    async def hangup(self) -> None:
        """통화 종료. 응답 뒤: 미디어 채널 닫기(SDK hangup). 울리는 중: SDK hangup 이 아무 동작도 하지 않으므로
        (미디어가 붙어야 종료 함수가 생김) REST 로 통화 종료 전송. 울리는 중 끊은 뒤 상대가 늦게 받으면 재호출로 미디어 닫기."""
        if self.session.ended_status is not None:
            return
        self._hung_up = True
        if self.leg is not None:
            if not self._media_closed:
                self._media_closed = True
                await self.session.hangup()
        elif not self._canceled and self.backend is not None:
            self._canceled = True
            await self.backend.cancel(self.session.call_id)


@dataclass
class _Line:
    """발신 번호 하나와 그 번호의 ClawOpsAgent. 번호 값은 repr·로그에 비노출 (이름 '번호k' 만 사용)."""
    idx: int
    number: str = field(repr=False)
    agent: object = field(default=None, repr=False)
    lost: bool = False              # 다른 프로세스가 번호를 가져갔거나(4409) 계정에서 쓸 수 없다고 통지됨(4403)
    lost_code: int | None = None

    @property
    def name(self) -> str:
        return f"번호{self.idx + 1}"

    def usable(self) -> bool:
        """이 번호로 지금 발신 가능한지 여부. 번호를 잃었거나 연결이 끊긴(재연결 대기 포함) 번호는 선택 제외:
        SDK 의 call() 은 연결 객체가 있으면 재연결 없이 바로 발신하므로, 끊긴 번호로 걸면 소리 없는 통화가 됨.
        연결 상태 판단 기준: clawops 0.70.0 의 비공개 속성(_control_ws._connected·_ws). 인계(4409) 판단 근거: taken_over."""
        a = self.agent
        if a is None or self.lost or getattr(a, "taken_over", False):
            return False
        if not hasattr(a, "_control_ws"):   # 시험용 가짜 ClawOpsAgent
            return True
        ws = a._control_ws
        if ws is None:
            return False
        ev = getattr(ws, "_connected", None)
        if ev is not None and not ev.is_set():
            return False
        sock = getattr(ws, "_ws", None)
        return sock is not None and not getattr(sock, "closed", False)


class ClawOpsBackend:
    LEGS: dict[str, _Leg] = {}
    CALLS: dict[str, ClawOpsCall] = {}

    # 서버가 알리는 미연결 종료 상태 → 종료 사유 end_reason (API 명세 3.2)
    FAIL_REASON = {"busy": "busy", "no-answer": "no_answer", "no_answer": "no_answer", "rejected": "no_answer",
                   "timeout": "no_answer", "canceled": "local_hangup", "failed": "failed"}

    def __init__(self, from_number: str, dtmf_mode: str = "inband") -> None:
        self.lines = [_Line(i, n) for i, n in enumerate(parse_numbers(from_number))]
        self.from_number = self.lines[0].number if self.lines else ""
        self.dtmf_mode = dtmf_mode

    # 번호 상태 (SessionManager 사용) ------------------------------------------------------
    @property
    def capacity(self) -> int:
        """동시 발신 가능 통화 수 = 번호 수 (번호가 없으면 1: start() 에서 차단)."""
        return max(1, len(self.lines))

    @property
    def agent(self):
        """첫 번호의 ClawOpsAgent (번호 하나 구성과의 호환, cancel() 의 계정 정보 출처)."""
        return self.lines[0].agent if self.lines else None

    def line_name(self, i: int) -> str:
        return self.lines[i].name if 0 <= i < len(self.lines) else f"번호{i + 1}"

    def line_rank(self, i: int) -> int:
        """번호 고르는 순서: 0 = 연결됨·통화 없음, 1 = 연결됨·이전 통화 정리 중(ClawOps 종료 알림 전), 2 = 쓸 수 없음."""
        if not 0 <= i < len(self.lines) or not self.lines[i].usable():
            return 2
        return 1 if any(c.line == i and c.backend is self for c in list(self.CALLS.values())) else 0

    def usable_count(self) -> int:
        return sum(1 for ln in self.lines if ln.usable())

    # 연결 -------------------------------------------------------------------------------------
    def _wire(self, agent) -> None:
        @agent.on("call_start")
        async def _start(call):
            c = self.CALLS.get(call.call_id)
            if c:
                # call_start 처리가 끝나야 SDK 의 Session 시작이 진행되므로 별도 태스크로 처리.
                asyncio.create_task(self._answered(c))

        @agent.on("call_failed")
        async def _failed(call, reason: str):
            c = self.CALLS.pop(call.call_id, None)
            if c:
                r = str(reason).lower()
                code = self.FAIL_REASON.get(r) or ("busy" if "busy" in r else "no_answer" if "answer" in r else "failed")
                log.info("발신 실패 (%s → %s)", r, code)
                await c.handlers.on_end(code)

        @agent.on("call_end")
        async def _end(call):
            c = self.CALLS.pop(call.call_id, None)
            self.LEGS.pop(call.call_id, None)
            if c:
                await c.handlers.on_end("local_hangup" if c._hung_up else "remote_hangup")

    def _on_taken_over(self, idx: int, code: int, reason: str) -> None:
        """SDK 가 await 없이 호출하는 콜백 (agent.retired 통지 뒤 4409 로 닫히면 두 번 호출됨). 이후 그 번호는 선택 제외."""
        ln = self.lines[idx]
        if ln.lost:
            return
        ln.lost, ln.lost_code = True, code
        if code == 4403:
            log.warning("%s: 이 계정에서 쓸 수 없는 번호라는 통지를 받았다 (4403). 서버를 다시 띄울 때까지 이 번호로 걸지 않는다",
                        ln.name)
        else:
            log.warning("%s: 다른 프로세스가 이 번호의 연결을 가져갔다 (%s). 서버를 다시 띄울 때까지 이 번호로 걸지 않는다",
                        ln.name, code)

    async def start(self) -> None:
        from clawops.agent import BuiltinTool, ClawOpsAgent

        if not self.lines:
            raise SystemExit(".env 에 CLAWOPS_FROM(070 번호)을 지정하십시오.")
        multi = len(self.lines) > 1
        try:
            takes_cb = "on_taken_over" in inspect.signature(ClawOpsAgent).parameters
        except (TypeError, ValueError):
            takes_cb = False
        for ln in self.lines:
            kw = {"on_taken_over": functools.partial(self._on_taken_over, ln.idx)} if multi and takes_cb else {}
            ln.agent = ClawOpsAgent(from_=ln.number, session_factory=_Leg,
                                    builtin_tools=BuiltinTool.NONE, prewarm_enabled=False, **kw)
            self._wire(ln.agent)
        if not multi:
            await self.lines[0].agent.connect()
            log.info("ClawOps 연결 완료 (번호1)")
            return
        res = await asyncio.gather(*(ln.agent.connect() for ln in self.lines), return_exceptions=True)
        bad = [(ln, r) for ln, r in zip(self.lines, res) if isinstance(r, BaseException)]
        for ln, r in bad:
            ln.lost = True
            log.warning("%s 연결 실패: %s. 서버를 다시 띄울 때까지 이 번호로 걸지 않는다", ln.name, type(r).__name__)
            # SDK 는 연결 대기 시간(10초) 초과 시 예외를 내지만 연결 시도는 뒤에서 계속함. 늦게 붙으면 이 프로세스가 쓰지도 않는
            # 번호의 자리를 잡아, 그 사이 그 번호를 쓰기 시작한 다른 서버의 연결을 빼앗음(4409). 이를 막기 위한 연결 해제
            try:
                await ln.agent.disconnect()
            except Exception as e:
                log.warning("%s 연결 시도 정리 실패: %s", ln.name, type(e).__name__)
        if len(bad) == len(self.lines):
            raise bad[0][1]
        log.info("ClawOps 연결 완료 (번호 %d개 중 %d개). 통화마다 빈 번호로 건다", len(self.lines), len(self.lines) - len(bad))

    async def _answered(self, c: ClawOpsCall) -> None:
        end = time.monotonic() + 5
        while c.leg is None and time.monotonic() < end:
            await asyncio.sleep(0.02)
        await c.handlers.on_answer()

    async def stop(self) -> None:
        agents = [ln.agent for ln in self.lines if ln.agent is not None]
        if len(agents) == 1:
            await agents[0].disconnect()
        elif agents:
            for r in await asyncio.gather(*(a.disconnect() for a in agents), return_exceptions=True):
                if isinstance(r, BaseException):
                    log.warning("ClawOps 연결 종료 실패: %s", type(r).__name__)

    async def cancel(self, call_id: str) -> None:
        """아직 응답 전(울리는 중)인 발신을 REST 로 종료 (SDK calls.update, Status=completed). 실패 시 경고만 기록.
        계정 정보는 모든 번호의 ClawOpsAgent 에서 동일."""
        try:
            from clawops import AsyncClawOps

            a = self.agent
            async with AsyncClawOps(api_key=a._api_key, account_id=a._account_id, base_url=a._base_url) as cl:
                await cl.calls.update(call_id, status="completed")
            log.info("울리는 발신 종료 요청 (call_id=%s…)", call_id[:8])
        except Exception as e:
            log.warning("울리는 발신 종료 요청 실패: %s", type(e).__name__)

    async def dial(self, to: str, handlers: CallHandlers, line: int | None = None) -> ClawOpsCall:
        i = 0 if line is None else line
        if not 0 <= i < len(self.lines) or self.lines[i].agent is None:
            raise RuntimeError(f"{self.line_name(i)} 이 연결되지 않았습니다")
        if len(self.lines) > 1 and not self.lines[i].usable():
            # 번호 선택 뒤 발신 전(목적 요약 대기 중)에 번호를 잃거나 연결이 끊긴 경우. SDK call() 은 연결 객체가 남아
            # 있으면 재연결 없이 발신하므로 그대로 걸면 소리 없는 통화가 됨 (session 쪽에서 다른 번호를 먼저 다시 선택)
            raise RuntimeError(f"{self.line_name(i)} 을 지금 쓸 수 없습니다")
        session = await self.lines[i].agent.call(to, timeout=40)
        c = ClawOpsCall(session, handlers, self.dtmf_mode, backend=self, line=i)
        self.CALLS[session.call_id] = c
        log.info("발신 %s → %s (call_id=%s…)", self.line_name(i), mask_dest(to), session.call_id[:8])
        return c


# ---------------------------------------------------------------------------
# 가상 ARS
# ---------------------------------------------------------------------------
class SimCall:
    """메뉴 트리 노드
      {"say": "...", "keys": {"1": "노드", "*": "@back"}, "type": "menu|input|result|end|agent",
       "length": 6, "terminator": "#", "next": "노드"}
    """

    provides_transcripts = True

    def __init__(self, tree: dict, handlers: CallHandlers, speed: float, audio: bool) -> None:
        self.tree, self.h, self.speed, self.audio = tree, handlers, speed, audio
        self.stack: list[str] = []
        self.node_id = tree.get("start", "root")
        self.digits = ""
        self.repeats = 0
        self.speaking = False
        self.ended = False
        self._input_gen = 0
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    @property
    def node(self) -> dict:
        return self.tree["nodes"][self.node_id]

    async def _say(self, text: str) -> None:
        started = time.monotonic()
        self.speaking = True
        await asyncio.sleep(max(0.01, len(text) * 0.07 * self.speed))
        self.speaking = False
        if not self.ended:
            await self.h.on_transcript(text, started)

    async def _enter(self, node_id: str, push: bool = True) -> None:
        if push and node_id != self.node_id:
            self.stack.append(self.node_id)
        self.node_id, self.digits, self.repeats = node_id, "", 0
        await self._play_node()

    async def _play_node(self) -> None:
        self._input_gen += 1
        gen = self._input_gen
        n = self.node
        await self._say(n["say"])
        t = n.get("type", "menu")
        if t in ("result", "end"):
            await asyncio.sleep(4 * self.speed)  # 결과 안내 뒤 잠시 후 ARS 쪽 종료
            await self._remote_end()
        elif t == "agent":
            await asyncio.sleep(3 * self.speed)
            if gen == self._input_gen and not self.ended:
                await self._say(n.get("human", "상담원 김민지입니다. 무엇을 도와드릴까요?"))
        else:
            self._spawn(self._timeout(gen))

    async def _timeout(self, gen: int) -> None:
        await asyncio.sleep(self.tree.get("timeout_s", 12) * self.speed)
        if gen != self._input_gen or self.ended:
            return
        self.repeats += 1
        if self.repeats > 2:
            await self._say("입력이 없어 통화를 종료합니다.")
            await self._remote_end()
            return
        log.info("가상 ARS 입력 대기 초과 %d회 (%s)", self.repeats, self.node_id)
        await self._say(self.tree.get("timeout_say", "입력 시간이 초과되었습니다."))
        if gen != self._input_gen or self.ended:
            return  # 안내 도중 키 입력으로 다른 메뉴로 넘어간 경우: 그 메뉴의 중복 재생 방지
        await self._play_node()

    async def _remote_end(self) -> None:
        if not self.ended:
            self.ended = True
            await self.h.on_end("remote_hangup")

    async def _key(self, ch: str) -> None:
        if self.ended:
            return
        n = self.node
        if n.get("type") == "input":
            if ch == n.get("terminator") or (ch.isdigit() and len(self.digits) + 1 >= n.get("length", 99)
                                             and not n.get("terminator")):
                if ch.isdigit():
                    self.digits += ch
                self._input_gen += 1
                await self._enter(n["next"])
            elif ch.isdigit():
                self.digits += ch  # 값은 저장·출력하지 않음
            return
        target = n.get("keys", {}).get(ch)
        self._input_gen += 1
        if target is None:
            await self._say("잘못 누르셨습니다.")
            await self._play_node()
        elif target == "@back":
            prev = self.stack.pop() if self.stack else self.node_id
            await self._enter(prev, push=False)
        elif target == "@home":
            self.stack.clear()
            await self._enter(self.tree.get("start", "root"), push=False)
        elif target == "@repeat":
            await self._play_node()
        else:
            await self._enter(target)

    # Call 프로토콜 ---------------------------------------------------------
    async def answer(self) -> None:
        await asyncio.sleep(0.2 * self.speed)
        await self.h.on_answer()
        if self.audio:
            self._spawn(self._audio_loop())
        self._spawn(self._play_node())

    async def _audio_loop(self) -> None:
        hum, t0, i = hum_ulaw(), time.monotonic(), 0
        while not self.ended:
            await self.h.on_audio(hum if self.speaking else SILENCE_FRAME)
            i += 1
            await asyncio.sleep(max(0.0, t0 + i * 0.02 - time.monotonic()))

    async def send_audio(self, ulaw: bytes) -> None:
        pass  # 가상 ARS 는 음성 인식 없음

    async def send_keys(self, keys: str) -> float:
        dur = len(keys) * 0.25
        for ch in keys:
            self._spawn(self._key_later(ch))
            await asyncio.sleep(0.01)
        return dur

    async def _key_later(self, ch: str) -> None:
        async with self._lock:  # 키 순서 보장
            await self._key(ch)

    async def hangup(self) -> None:
        if not self.ended:
            self.ended = True
            for t in list(self._tasks):
                t.cancel()
            await self.h.on_end("local_hangup")


class SimBackend:
    def __init__(self, tree_path: str, speed: float = 1.0, audio: bool = True, voice: bool = False, lines: int = 1) -> None:
        self.tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))
        self.speed, self.audio, self.voice = speed, audio, voice
        # 가상 번호 수 (SIM_LINES). 2 이상이면 ClawOps 번호 여러 개와 같은 배정 규칙을 가상 ARS 로 시험 가능
        self.capacity = max(1, int(lines))

    def line_name(self, i: int) -> str:
        return f"번호{i + 1}"

    def line_rank(self, i: int) -> int:
        return 0 if 0 <= i < self.capacity else 2

    def usable_count(self) -> int:
        return self.capacity

    async def start(self) -> None:
        log.info("가상 ARS 사용: %s%s", self.tree.get("name"), " (음성)" if self.voice else "")
        if self.voice:
            from sim_voice import prepare

            await asyncio.to_thread(prepare, self.tree)

    async def stop(self) -> None:
        pass

    async def dial(self, to: str, handlers: CallHandlers, line: int | None = None) -> SimCall:
        if self.voice:
            from sim_voice import VoiceSimCall

            c: SimCall = VoiceSimCall(self.tree, handlers, self.speed)
        else:
            c = SimCall(self.tree, handlers, self.speed, self.audio)
        c._spawn(c.answer())
        return c
