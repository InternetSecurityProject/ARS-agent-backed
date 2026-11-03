"""
통화 백엔드

  ClawOpsBackend : ClawOps 070 번호로 실제 발신 (Voice Agent SDK). 동시 통화 1건(Trial·Individual).
  SimBackend     : 가상 ARS. 메뉴 트리(JSON)를 따라 안내 문장을 직접 전달한다. 실통화 없이 전체 흐름을 시험한다.

서버(session)는 Call 프로토콜만 사용한다.
  send_audio(ulaw)   : 사용자 마이크 음성 송출 (20ms 프레임)
  send_keys(keys)    : DTMF 전송 (in-band 톤). 송출 시간(초) 반환
  hangup()
콜백(CallHandlers)
  on_answer()        : 상대 응답
  on_audio(ulaw)     : 상대(ARS·상담원) 음성 프레임
  on_transcript(txt, started_at) : 백엔드가 문장을 직접 주는 경우(sim). started_at 은 안내를 말하기 시작한 시각(monotonic). 실통화는 사용하지 않음
  on_end(reason)     : no_answer / busy / failed / remote_hangup / local_hangup
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
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
    async def send_keys(self, keys: str) -> float: ...
    async def hangup(self) -> None: ...


class Backend(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def dial(self, to: str, handlers: CallHandlers) -> Call: ...


# ---------------------------------------------------------------------------
# ClawOps
# ---------------------------------------------------------------------------
class _Leg:
    """ClawOps Session Protocol 구현. SDK 가 통화마다 생성하여 수신 오디오를 넘긴다."""

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

    def __init__(self, session, handlers: CallHandlers, dtmf_mode: str = "inband") -> None:
        self.session = session
        self.handlers = handlers
        self.dtmf_mode = dtmf_mode
        self._tone_until = 0.0
        self._hung_up = False

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
            # 미디어 채널의 DTMF 이벤트로 전송 (톤을 음성에 섞지 않음). 송출 시간은 톤 방식과 같게 잡는다
            await self.session.send_dtmf_sequence(keys)
            return len(dtmf_ulaw(keys)) / 8000
        tone = dtmf_ulaw(keys)
        dur = len(tone) / 8000
        self._tone_until = time.monotonic() + dur + 0.05  # 톤 송출 중 마이크 음성 차단
        # 실시간 속도로 보낸다 (20ms 프레임마다). 한꺼번에 밀어 넣으면 전송 쪽 버퍼에서 톤이 겹치거나 잘려
        # 긴 번호(전화번호 등)의 일부가 ARS 에 인식되지 않을 수 있다
        t0 = time.monotonic()
        for n, i in enumerate(range(0, len(tone), FRAME)):
            await leg.call.send_audio(tone[i:i + FRAME])
            wait = t0 + (n + 1) * FRAME / 8000 - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        return dur

    async def hangup(self) -> None:
        if not self._hung_up and self.session.ended_status is None:
            self._hung_up = True
            await self.session.hangup()


class ClawOpsBackend:
    LEGS: dict[str, _Leg] = {}
    CALLS: dict[str, ClawOpsCall] = {}

    # 서버가 알리는 미연결 종료 상태 → 세션 종료 사유 (명세 3.2)
    FAIL_REASON = {"busy": "busy", "no-answer": "no_answer", "no_answer": "no_answer", "rejected": "no_answer",
                   "timeout": "no_answer", "canceled": "local_hangup", "failed": "failed"}

    def __init__(self, from_number: str, dtmf_mode: str = "inband") -> None:
        self.from_number = "".join(ch for ch in from_number if ch.isdigit())
        self.dtmf_mode = dtmf_mode
        self.agent = None

    async def start(self) -> None:
        from clawops.agent import BuiltinTool, ClawOpsAgent

        if not self.from_number:
            raise SystemExit(".env 에 CLAWOPS_FROM(070 번호)을 지정하십시오.")
        self.agent = ClawOpsAgent(from_=self.from_number, session_factory=_Leg,
                                  builtin_tools=BuiltinTool.NONE, prewarm_enabled=False)

        @self.agent.on("call_start")
        async def _start(call):
            c = self.CALLS.get(call.call_id)
            if c:
                # call_start 처리가 끝나야 SDK 가 세션을 시작하므로 별도 태스크로 넘긴다.
                asyncio.create_task(self._answered(c))

        @self.agent.on("call_failed")
        async def _failed(call, reason: str):
            c = self.CALLS.pop(call.call_id, None)
            if c:
                r = str(reason).lower()
                code = self.FAIL_REASON.get(r) or ("busy" if "busy" in r else "no_answer" if "answer" in r else "failed")
                log.info("발신 실패 (%s → %s)", r, code)
                await c.handlers.on_end(code)

        @self.agent.on("call_end")
        async def _end(call):
            c = self.CALLS.pop(call.call_id, None)
            self.LEGS.pop(call.call_id, None)
            if c:
                await c.handlers.on_end("local_hangup" if c._hung_up else "remote_hangup")

        await self.agent.connect()
        log.info("ClawOps 연결 완료 (발신 번호 %s)", self.from_number)

    async def _answered(self, c: ClawOpsCall) -> None:
        end = time.monotonic() + 5
        while c.leg is None and time.monotonic() < end:
            await asyncio.sleep(0.02)
        await c.handlers.on_answer()

    async def stop(self) -> None:
        if self.agent:
            await self.agent.disconnect()

    async def dial(self, to: str, handlers: CallHandlers) -> ClawOpsCall:
        session = await self.agent.call(to, timeout=40)
        c = ClawOpsCall(session, handlers, self.dtmf_mode)
        self.CALLS[session.call_id] = c
        log.info("발신 %s → %s (call_id=%s)", self.from_number, to, session.call_id)
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
            await asyncio.sleep(4 * self.speed)  # 결과 안내 뒤 잠시 후 ARS 가 끊는다
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
            return  # 안내 도중 키를 받아 다른 메뉴로 넘어갔다. 그 메뉴를 다시 재생하지 않는다
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
                self.digits += ch  # 값은 저장·출력하지 않는다
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
        pass  # 가상 ARS 는 음성을 인식하지 않는다

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
    def __init__(self, tree_path: str, speed: float = 1.0, audio: bool = True, voice: bool = False) -> None:
        self.tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))
        self.speed, self.audio, self.voice = speed, audio, voice

    async def start(self) -> None:
        log.info("가상 ARS 사용: %s%s", self.tree.get("name"), " (음성)" if self.voice else "")
        if self.voice:
            from sim_voice import prepare

            await asyncio.to_thread(prepare, self.tree)

    async def stop(self) -> None:
        pass

    async def dial(self, to: str, handlers: CallHandlers) -> SimCall:
        if self.voice:
            from sim_voice import VoiceSimCall

            c: SimCall = VoiceSimCall(self.tree, handlers, self.speed)
        else:
            c = SimCall(self.tree, handlers, self.speed, self.audio)
        c._spawn(c.answer())
        return c
