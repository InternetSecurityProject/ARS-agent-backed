"""
통화 세션 (상태 머신)

통화 1건의 상태, 자막, 경로, 사용자 요청을 관리하고 이벤트를 전송한다.
전화 제어(키 전송·종료)는 이 모듈만 수행한다. 해석기(AI)는 판단 결과만 돌려준다.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from audio import VALID_KEYS, Segmenter
from config import Settings
from interpreter import Interpreter, Option, Prompt
from navigator import Layer, NavState, Navigator
from stt import Recognizer
from telephony import Backend, Call, CallHandlers

log = logging.getLogger("ars.session")
KST = timezone(timedelta(hours=9))


def now_iso(offset_s: float = 0) -> str:
    return (datetime.now(KST) + timedelta(seconds=offset_s)).isoformat(timespec="seconds")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


STATUS_TEXT = {
    "dialing": "전화를 걸고 있습니다.",
    "listening": "안내를 듣고 있습니다.",
    "selecting": "누를 번호를 고르고 있습니다.",
    "awaiting_user": "확인이 필요합니다.",
    "listening_result": "조회 결과를 듣고 있습니다.",
    "agent_done": "업무를 마쳤습니다. 결과를 확인한 뒤 통화를 종료하세요.",
    "waiting_agent": "상담원 연결을 기다리고 있습니다.",
    "connected_to_agent": "상담원이 연결되었습니다. 직접 말씀하세요.",
    "ended": "통화가 종료되었습니다.",
}

END_TEXT = {
    "completed": "업무를 마쳐 통화를 종료했습니다.",
    "agent_call_ended": "상담원과의 통화가 종료되었습니다.",
    "user_input_timeout": "입력이 없어 ARS 가 통화를 종료했습니다.",
    "message_sent": "ARS 가 문자로 안내를 보내고 통화를 종료했습니다.",
    "out_of_hours": "상담 가능 시간이 아니어서 통화를 종료했습니다.",
    "ars_rejected": "ARS 가 요청을 처리하지 못해 통화를 종료했습니다.",
    "loop_detected": "같은 메뉴가 반복되어 통화를 종료했습니다.",
    "not_found": "원하시는 메뉴를 찾지 못해 통화를 종료했습니다.",
    "user_canceled": "통화를 종료했습니다.",
    "no_answer": "상대방이 전화를 받지 않습니다.",
    "busy": "상대방이 통화 중입니다.",
    "failed": "통화가 끊겼습니다.",
    "max_duration": "최대 통화 시간이 지나 통화를 종료했습니다.",
}

_READBACK_RE = re.compile(r"입력하신|누르신|맞으면|맞으시면|맞습니까")
_DIGITS_RE = re.compile(r"[0-9][0-9\s,\-]{2,}[0-9]|(?:[공영일이삼사오육칠팔구]\s*){4,}")


def _obj(word: str) -> str:
    """목적격 조사 (을/를)."""
    ch = word.strip()[-1:] or "가"
    return "을" if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 else "를"


def mask_digits(text: str) -> tuple[str, bool]:
    """ARS 가 입력값을 다시 읽어 주는 문장에서 숫자열을 가린다."""
    masked, n = _DIGITS_RE.subn(lambda m: "*" * len(re.sub(r"\D", "", m.group()) or "****"), text)
    return masked, n > 0


# ---------------------------------------------------------------------------
# 이벤트 전송
# ---------------------------------------------------------------------------
class EventHub:
    def __init__(self) -> None:
        self.queues: set[asyncio.Queue] = set()
        self.history: list[dict] = []  # 시험·디버깅용

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.queues.discard(q)

    def publish(self, event: str, data: dict) -> None:
        msg = {"event": event, "at": now_iso(), "data": data}
        self.history.append(msg)
        for q in list(self.queues):
            q.put_nowait(msg)


@dataclass
class Deps:
    settings: Settings
    backend: Backend
    recognizer: Recognizer
    interp: Interpreter
    navigator: Navigator


# ---------------------------------------------------------------------------
# 세션
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, phone: str, deps: Deps) -> None:
        self.id = "ses_" + secrets.token_hex(4)
        self.token = "st_" + secrets.token_urlsafe(24)
        self.phone = phone
        self.d = deps
        self.s = deps.settings
        self.hub = EventHub()
        self.audio_out: set[asyncio.Queue] = set()   # 앱 오디오 채널 (서버 → 앱)

        self.status = "created"
        self.status_message = ""
        self.agent_active = False
        self.goal: str | None = None
        self.goal_summary: str | None = None

        self.call: Call | None = None
        self.answered_at: float | None = None
        self.ended_at: float | None = None
        self.end_reason: str | None = None
        self.created_mono = time.monotonic()

        self.captions: list[dict] = []
        self.path: list[dict] = []
        self.step = 0
        self.layers: list[Layer] = []        # 키를 눌러 지나온 메뉴 (사용자·AI 공통)
        self.current: Prompt | None = None   # 가장 최근에 들은 메뉴·확인 질문
        self.current_fresh = False           # 마지막 키 전송 이후에 들은 안내인지
        self.last_key_actor = "ai"
        self.nav: NavState | None = None
        self.key_queue: list[str] = []       # 이전 계층 복귀 시 차례로 누를 키 (처음으로 + 지나온 키)
        self.redo_same_menu = False          # 입력 오류 뒤 같은 메뉴를 다시 판단
        self.after_backtrack = False         # 이전 계층으로 돌아가는 중
        self.input_errors = 0

        self.pending: dict | None = None
        self.req_seq = 0
        self.secure_window = False
        self.ai_hold_until = 0.0
        self.summary: dict | None = None
        self.agent_goal_at_caption = 0       # AI 시작 시점의 자막 위치 (결과 요약 범위)

        self.segmenter = Segmenter(self.s.vad_rms, self.s.paragraph_silence_ms)
        self.stt_q: asyncio.Queue = asyncio.Queue()
        self._timers: dict[str, asyncio.Task] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    # 공통 ------------------------------------------------------------------
    def spawn(self, coro: Awaitable) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        t.add_done_callback(self._log_exc)
        return t

    @staticmethod
    def _log_exc(t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception():
            log.error("세션 작업 오류", exc_info=t.exception())

    def timer(self, name: str, delay: float, fn: Callable[[], Awaitable[None]]) -> None:
        self.cancel_timer(name)

        async def run():
            await asyncio.sleep(delay)
            self._timers.pop(name, None)
            await fn()

        self._timers[name] = self.spawn(run())

    def cancel_timer(self, name: str) -> None:
        t = self._timers.pop(name, None)
        if t and t is not asyncio.current_task():
            t.cancel()

    @property
    def ended(self) -> bool:
        return self.status == "ended"

    def set_status(self, status: str, message: str | None = None) -> None:
        msg = message or STATUS_TEXT.get(status, "")
        if status == self.status and msg == self.status_message:
            return
        self.status, self.status_message = status, msg
        self.hub.publish("status_changed", {"status": status, "message": msg})

    def idle_status(self) -> str:
        """사용자 입력이 끝난 뒤 돌아갈 상태. AI 탐색 중이면 listening, 업무를 마친 뒤면 agent_done."""
        return "listening" if self.agent_active else "agent_done"

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "message": self.status_message,
            "goal": self.goal,
            "goal_summary": self.goal_summary,
            "path": self.path,
            "captions": self.captions,
            "pending_request": self.pending,
            "summary": self.summary,
            "end_reason": self.end_reason,
        }

    # 생성·발신 ---------------------------------------------------------------
    async def create(self, goal: str | None) -> dict:
        # AI 도우미 사용과 목적은 통화 시작 시 정한다. 승인 없이 바로 발신하고 첫 메뉴부터 탐색한다.
        out = {"session_id": self.id, "session_token": self.token, "phone_number": self.phone}
        self.goal = (goal or "").strip()
        self.goal_summary = await self.d.interp.summarize_goal(self.goal)
        self.agent_active = True
        self.nav = NavState(self.goal)
        out["goal_summary"] = self.goal_summary
        await self.dial()
        out["status"] = self.status
        return out

    async def dial(self) -> None:
        self.set_status("dialing")
        handlers = CallHandlers(self._on_answer, self._on_audio, self._on_transcript, self._on_end)
        try:
            self.call = await self.d.backend.dial(self.phone, handlers)
        except Exception as e:
            log.error("발신 실패: %s", e)
            await self.finish("failed")
            return
        self.spawn(self._stt_worker())

    async def _on_answer(self) -> None:
        self.answered_at = time.monotonic()
        self.timer("max_duration", self.s.max_call_s, lambda: self.finish("max_duration"))
        self.set_status("listening", self.goal_summary)

    async def _on_end(self, reason: str) -> None:
        if self.end_reason is not None:
            return
        if reason == "remote_hangup":
            if self.pending is not None:  # 민감정보 직접 입력·재질문 응답 대기 중
                reason = "user_input_timeout"
            elif self.status == "connected_to_agent":
                reason = "agent_call_ended"
            elif self.summary and self.summary["kind"] == "result":
                reason = "completed"
            else:
                reason = "failed"
        elif reason == "local_hangup":
            reason = "user_canceled"
        await self.finish(reason, hangup=False)

    async def finish(self, reason: str, hangup: bool = True, note: str = "") -> None:
        if self.end_reason is not None:
            return
        self.end_reason = reason
        self.ended_at = time.monotonic()
        self.pending = None
        self.agent_active = False
        for name in list(self._timers):
            self.cancel_timer(name)
        if hangup and self.call is not None:
            try:
                await self.call.hangup()
            except Exception as e:
                log.warning("통화 종료 요청 실패: %s", e)
        self.set_status("ended", END_TEXT.get(reason, ""))
        data = {"end_reason": reason, "message": END_TEXT.get(reason, "")}
        if note:
            data["detail"] = note
        self.hub.publish("ended", data)
        for q in list(self.audio_out):
            q.put_nowait(None)
        self.stt_q.put_nowait(None)  # 음성 인식 작업 종료

    # 오디오 ----------------------------------------------------------------
    async def _on_audio(self, frame: bytes) -> None:
        for q in list(self.audio_out):
            if q.qsize() < 50:  # 1초 이상 밀리면 버린다
                q.put_nowait(frame)
        if self.call is None or self.call.provides_transcripts or self.status == "connected_to_agent":
            return
        seg = self.segmenter.feed(frame)
        if seg is not None:
            self.stt_q.put_nowait(seg)

    async def _stt_worker(self) -> None:
        while not self.ended:
            seg = await self.stt_q.get()
            if seg is None:
                break  # 통화 종료
            if self.secure_window and not self.d.recognizer.is_local:
                continue  # 외부 STT 에는 입력값을 다시 읽어 주는 구간을 보내지 않는다
            text = await self.d.recognizer.recognize(seg)
            del seg
            if text:
                await self.on_paragraph(text)

    async def mic_audio(self, frame: bytes) -> None:
        """앱 마이크 음성. 상담원 연결 중에만 통화로 보낸다 (그 외에는 사용자 음성을 중계하지 않는다)."""
        if self.call is None or self.ended or self.status != "connected_to_agent":
            return
        await self.call.send_audio(frame)

    async def _on_transcript(self, text: str) -> None:
        await self.on_paragraph(text)

    # 안내 처리 ---------------------------------------------------------------
    async def on_paragraph(self, text: str) -> None:
        if self.ended or self.status == "connected_to_agent":
            return
        async with self._lock:
            await self._paragraph(text)

    async def _paragraph(self, text: str) -> None:
        interp = self.d.interp
        self.cancel_timer("no_response")

        if self.secure_window:
            # ARS 가 입력값을 다시 읽어 주는 안내("입력하신 번호는 … 맞으면 1번")는 숫자를 가린 뒤 판단만 하고 자막으로 보내지 않는다
            if _READBACK_RE.search(text):
                masked, _ = mask_digits(text)
                p = await interp.classify(masked)
                del text
                if p.kind == "sensitive":
                    await self._ask_secure(p)
                elif p.kind in ("menu", "confirm"):
                    self.current, self.current_fresh = p, True
                    if self.agent_active:
                        await self._ask(p, "unknown_prompt", p.options,
                                        "ARS 가 입력하신 번호가 맞는지 묻고 있습니다. 맞으면 해당 번호를 골라 주세요.")
                return
            self.secure_window = False

        p = await interp.classify(text)
        if p.kind == "input_error" and (self.current_fresh or not self.path):
            p.kind = "info"  # 키를 누르지 않은 상태의 '입력 시간 초과' 안내는 ARS 의 대기 시간 경과일 뿐이다
        if p.kind != "input_error" and not self.redo_same_menu:
            self.input_errors = 0  # 직전 키가 받아들여졌다. 입력 오류 '2회 연속'을 다시 센다
        self.current_fresh = True if p.kind in ("menu", "confirm") else self.current_fresh
        if p.kind not in ("sensitive", "input_error"):
            cap = {"step": self.step + 1, "speaker": "ars", "text": text, "at": now_iso()}
            self.captions.append(cap)
            self.hub.publish("caption", cap)

        # AI 사용 여부와 관계없이 처리하는 안내
        if p.kind == "sensitive":
            if not (self.pending and self.pending["kind"] == "secure_input"):
                await self._ask_secure(p)
            return
        if p.kind == "input_error":
            self.hub.publish("input_rejected", {
                "keys": self.path[-1]["selected"]["key"] if self.path else None,
                "actor": self.last_key_actor, "ars_prompt": text, "retry_count": self.input_errors + 1,
                "message": "번호가 인식되지 않아 다시 시도합니다." if self.agent_active else "번호가 인식되지 않았습니다.",
            })
            if self.agent_active:
                self.input_errors += 1
                self.redo_same_menu = True
                if self.input_errors >= 2 and self.current:
                    await self._ask(self.current, "input_rejected", self.current.options)
            return
        if p.kind == "agent_human":
            if self.agent_active or self.status == "waiting_agent":
                await self._handoff()
            return
        if p.kind in ("menu", "confirm"):
            self.current = p

        if not self.agent_active:
            return  # 업무 완료(agent_done) 뒤의 안내는 자막만 전송한다

        # 이하 AI 탐색 중
        if self.pending is not None:
            return  # 사용자 응답 대기 중 ARS 가 안내를 다시 읽는 경우. 계속 듣는다.
        if self.key_queue and p.kind == "menu":
            await self._press(self.key_queue.pop(0), "ai", p)
            return
        if p.kind in ("menu", "confirm"):
            if time.monotonic() < self.ai_hold_until:
                return  # 사용자가 직접 누른 직후 보류 시간. 보류가 끝나면 가장 최근 메뉴로 판단한다
            await self._decide_heard(p)
        elif p.kind == "result":
            # 결과를 받으면 AI 만 마치고 통화는 유지한다. 사용자가 통화를 종료하면 end_reason=completed
            self.set_status("listening_result")
            caps = [c["text"] for c in self.captions[self.agent_goal_at_caption:]]
            res = await interp.summarize_result(self.goal or "", caps)
            self.summary = {"kind": "result", "summary": res.get("summary", ""),
                            "key_info": res.get("key_info", []), "handoff_script": None}
            self.hub.publish("summary", self.summary)
            self.agent_active = False
            self.key_queue.clear()
            self.set_status("agent_done")
        elif p.kind in ("message_sent", "out_of_hours", "rejected"):
            await self.finish({"rejected": "ars_rejected"}.get(p.kind, p.kind))
        elif p.kind == "agent_wait":
            self.set_status("waiting_agent")
        elif p.kind == "voice_request":
            # [검증 필요] 전환 문구 음성 송출(TTS)은 아직 구현하지 않았다. 사용자에게 알리고 응답을 받는다.
            await self._ask(p, "unknown_prompt", p.options,
                            "ARS 가 말로 대답하라고 합니다. 음성으로 답하는 기능은 아직 지원하지 않습니다. "
                            "번호로 고를 수 있으면 골라 주세요.")

    async def _handoff(self) -> None:
        texts = [p["text"] for p in self.path]
        script = await self.d.interp.handoff_script(self.goal or "", texts) if self.goal else None
        self.summary = {"kind": "handoff", "summary": "상담원 연결까지 진행했습니다.", "handoff_script": script}
        self.hub.publish("summary", self.summary)
        self.agent_active = False
        self.pending = None
        self.set_status("connected_to_agent")

    # 판단 ------------------------------------------------------------------
    async def _decide_heard(self, p: Prompt) -> None:
        """새로 들은 메뉴·확인 질문을 판단한다.
        입력 오류 뒤 같은 메뉴, 이전 계층 복귀로 도착한 메뉴는 반복으로 세지 않는다."""
        arrived = not (self.redo_same_menu or self.after_backtrack)
        self.redo_same_menu = self.after_backtrack = False
        await self._decide(p, arrived)

    async def _decide(self, p: Prompt, arrived: bool) -> None:
        if not self.agent_active or self.pending is not None or self.ended or self.nav is None:
            return
        # 메뉴·확인 질문을 판단하기 시작하면 selecting (명세 3.1).
        # 결과에 따라 listening(키 전송)·awaiting_user(재질문)·ended(not_found) 로 넘어간다
        self.set_status("selecting")
        if p.kind == "confirm":
            if p.critical:  # 결제·계약·해지 최종 승인 등은 확인 질문이어도 키를 보내지 않는다 (1.1 원칙 5)
                await self._ask(p, "critical_action", p.options)
                return
            key = await self.d.interp.answer_confirm(self.goal or "", p)
            if key:
                await self._press(key, "ai", p)
            else:
                await self._ask(p, "unknown_prompt", p.options)
            return
        dec = await self.d.navigator.decide(self.nav, p, self.layers, arrived)
        log.info("판단 %s %s %s", dec.action, dec.key or dec.keys or dec.reason, dec.note)
        if dec.action == "press":
            self.nav.presses += 1
            self.nav.exclude.clear()
            self.nav.exclude_at = None
            self.set_status("selecting", f"'{p.label_of(dec.key)}' 메뉴를 선택합니다.")
            await self._press(dec.key, "ai", p, confidence=dec.scores.get(dec.key))
        elif dec.action == "ask":
            await self._ask(p, dec.reason, dec.options, scores=dec.scores)
        elif dec.action == "backtrack":
            # '이전 메뉴' 키를 누르면 _press 가 마지막 계층을 지운다.
            # '처음으로' 키를 쓰는 경우 지나온 키를 key_queue 로 다시 누르며 계층을 다시 쌓는다.
            self.nav.backtracked = True
            self.nav.exclude = {self.layers[-1].key}
            self.nav.exclude_at = self.layers[-1].prompt.signature()
            self.after_backtrack = True
            keys = dec.keys
            if keys[0] == p.home_key and len(keys) > 1:
                self.key_queue = list(keys[1:])
                keys = keys[0]
            self.set_status("selecting", "맞는 메뉴가 없어 이전 메뉴를 확인합니다.")
            await self._press(keys, "ai", p)
        elif dec.action == "end":
            await self.finish(dec.reason, note=dec.note)

    async def _press(self, keys: str, actor: str, p: Prompt | None, confidence: float | None = None) -> None:
        if self.call is None or self.ended:
            raise ApiError(409, "invalid_state", "통화가 연결되지 않았습니다.")
        dur = await self.call.send_keys(keys)
        self.segmenter.mute(dur)
        self.current_fresh = False
        self.last_key_actor = actor
        if p is not None:
            if keys == p.back_key:
                if self.layers:
                    self.layers.pop()
            elif keys == p.home_key:
                self.layers.clear()
            elif keys != p.repeat_key and p.kind == "menu":
                self.layers.append(Layer(p, keys, actor))
        self.step += 1
        label = p.label_of(keys) if p else keys
        names = [o.label for o in (p.options if p else [])]
        if label == keys:
            text = f"{keys} 입력"  # 안내에서 듣지 못한 키 (직접 선택)
        elif len(names) >= 2 and any(o.key == keys for o in p.options):
            # 명세 6.2: 들은 선택지 중 고른 선택지 (예: "잔액조회·거래내역조회·상담직원 연결 중 잔액조회(3) 선택")
            text = f"{'·'.join(names[:5])}{' 등' if len(names) > 5 else ''} 중 {label}({keys}) 선택"
        else:
            text = f"{label}({keys}) 선택"  # 이전 메뉴·처음으로·다시 듣기, 선택지가 하나인 안내
        sel = next((o for o in p.all_options() if o.key == keys), None) if p else None
        step = {
            "step": self.step, "text": text,
            "options": [o.as_dict() for o in (p.all_options() if p else [])],
            # 이동 키(이전 메뉴·처음으로·다시 듣기)와 안내에 없던 키는 category=null (options 와 같게)
            "selected": {"key": keys, "label": label, "category": sel.category if sel else None,
                         "confidence": confidence},
            "actor": actor,
        }
        self.path.append(step)
        self.hub.publish("path_updated", step)
        if self.agent_active and self.pending is None:
            self.set_status("listening")  # 재질문에 select·key 로 답한 경우도 awaiting_user 를 해제한다
        repeat = p.repeat_key if p else None
        self.timer("no_response", self.s.no_response_s, lambda: self._no_response(repeat))

    async def _no_response(self, repeat_key: str | None) -> None:
        """키 전송 후 새 안내가 없으면 같은 키를 다시 누르지 않고 다시 듣기 키를 누른다."""
        if self.ended or not repeat_key or self.call is None:
            return
        dur = await self.call.send_keys(repeat_key)
        self.segmenter.mute(dur)
        self.redo_same_menu = True  # 다시 듣기 키로 다시 나온 메뉴는 반복으로 세지 않는다 (명세 7장 5항)

    # 사용자 요청 -------------------------------------------------------------
    def _new_request(self, kind: str, expire_s: float, **fields) -> dict:
        self.req_seq += 1
        req = {"request_id": f"req_{self.req_seq}", "kind": kind, **fields, "expires_at": now_iso(expire_s)}
        self.pending = req
        return req

    async def _ask(self, p: Prompt, reason: str, options: list[Option], message: str | None = None,
                   scores: dict[str, float] | None = None) -> None:
        labels = [o.label for o in options]
        scores = scores or {}
        default = {
            "ambiguous": f"{' / '.join(labels)} 중 어느 쪽인가요? 원하시는 업무를 말씀해 주세요.",
            "low_confidence": (f"'{labels[0]}' 메뉴가 맞는지 확실하지 않습니다. " if labels else "")
                              + "원하시는 업무를 조금 더 자세히 말씀해 주세요.",
            "critical_action": "결제·계약·해지처럼 중요한 단계입니다. 진행하려면 화면에서 번호를 직접 골라 주세요.",
            "loop": "같은 메뉴가 반복되고 있습니다. 원하시는 업무를 조금 더 자세히 말씀해 주세요.",
            "unknown_prompt": f"ARS 가 이렇게 묻습니다. \"{p.text[:60]}\" 어떻게 답할까요?",
            "input_rejected": "번호가 계속 인식되지 않습니다. 원하시는 메뉴를 말씀하거나 직접 골라 주세요.",
        }
        req = self._new_request("clarification", self.s.clarify_expire_s, reason=reason,
                                message=message or default.get(reason, "확인이 필요합니다."),
                                options=[{**o.as_dict(), "confidence": scores.get(o.key)} for o in options],
                                ars_prompt=p.text)
        self.set_status("awaiting_user", req["message"])
        self.hub.publish("request", req)
        rid = req["request_id"]
        self.timer("clarify_expire", self.s.clarify_expire_s, lambda: self._clarify_expire(rid))
        self.timer("clarify_drop", self.s.clarify_drop_s, lambda: self._clarify_drop(rid))

    async def _clarify_expire(self, rid: str) -> None:
        """30초: ARS 가 통화를 끊지 않도록 다시 듣기 키를 1회 누른다. 요청은 유지한다."""
        if self.pending and self.pending["request_id"] == rid and self.current and self.current.repeat_key:
            dur = await self.call.send_keys(self.current.repeat_key)
            self.segmenter.mute(dur)

    async def _clarify_drop(self, rid: str) -> None:
        """60초: 요청을 무효로 하고 상담원 연결 메뉴로 진행한다.
        상담원 연결 선택지가 없으면 요청을 유지한 채 계속 듣는다 (ARS 가 통화를 끊으면 user_input_timeout)."""
        async with self._lock:
            if not (self.pending and self.pending["request_id"] == rid) or self.ended:
                return
            cur = self.current
            if cur and cur.agent_key:
                self._close_request()
                self.set_status("selecting", "응답이 없어 상담원 연결 메뉴로 진행합니다.")
                await self._press(cur.agent_key, "ai", cur)

    async def _ask_secure(self, p: Prompt) -> None:
        self.secure_window = True
        label = p.label or "요청 정보"
        req = self._new_request("secure_input", self.s.secure_expire_s, label=label, ars_prompt=p.text,
                                terminator=p.terminator,
                                message=f"{label}{_obj(label)} 직접 눌러 주세요. 입력하신 번호는 저장되지 않습니다.")
        self.set_status("awaiting_user", "직접 입력이 필요합니다.")
        self.hub.publish("request", req)

    # 사용자 조작 (4.3) ---------------------------------------------------------
    def _check_request(self, body: dict, kind: str) -> dict:
        req = self.pending
        if not req or req["kind"] != kind or body.get("request_id") != req["request_id"]:
            raise ApiError(409, "invalid_state", "이미 응답했거나 만료된 요청입니다.")
        return req

    async def action(self, body: dict) -> dict:
        t = body.get("type")
        if self.ended:
            raise ApiError(409, "invalid_state", "이미 종료된 통화입니다.")
        async with self._lock:
            if t == "cancel":
                if self.status == "connected_to_agent":
                    reason = "agent_call_ended"
                elif self.summary and self.summary["kind"] == "result":
                    reason = "completed"
                else:
                    reason = "user_canceled"
                await self.finish(reason)
                return {"accepted": True}
            if t == "answer":
                return await self._act_answer(body)
            if t == "select":
                req = self._check_request(body, "clarification")
                key = str(body.get("key", ""))
                if key not in {o["key"] for o in req["options"]}:
                    raise ApiError(400, "invalid_request", "요청에 포함된 후보가 아닙니다.")
                self._close_request()
                if req["reason"] == "loop" and self.nav:
                    self.nav.reset_loop()
                self.input_errors = 0
                await self._press(key, "user", self.current)
                return {"accepted": True, "key_sent": key}
            if t == "key":
                return await self._act_key(body)
            if t == "secure_input":
                return await self._act_secure(body)
            if t == "revise_goal":
                return await self._act_revise_goal(body)
            raise ApiError(400, "invalid_request", f"알 수 없는 type: {t}")

    def _close_request(self, keep_secure: bool = False) -> None:
        if keep_secure and self.pending and self.pending["kind"] == "secure_input":
            return
        self.pending = None
        self.cancel_timer("clarify_expire")
        self.cancel_timer("clarify_drop")

    async def _act_answer(self, body: dict) -> dict:
        text = str(body.get("text", "")).strip()
        if not text:
            raise ApiError(400, "invalid_request", "text 가 비어 있습니다.")
        req = self._check_request(body, "clarification")
        if req["reason"] == "critical_action":
            raise ApiError(409, "invalid_state", "중요 선택은 번호를 직접 골라야 합니다 (select).")
        self._close_request()
        self.goal = await self.d.interp.merge_answer(self.goal or "", req["message"], text)
        self.nav.goal = self.goal
        if req["reason"] == "loop":
            self.nav.reset_loop()
        self.input_errors = 0
        if self.current is not None and self.current_fresh:
            await self._decide(self.current, arrived=False)
        else:
            # 음성 응답 요구처럼 지금 안내가 메뉴가 아니면 지난 메뉴로 판단하지 않고 다음 안내를 기다린다
            self.set_status(self.idle_status())
        return {"accepted": True}

    async def _act_key(self, body: dict) -> dict:
        key = str(body.get("key", ""))
        if not key or len(key) > 4 or any(ch not in VALID_KEYS for ch in key):
            raise ApiError(400, "invalid_request", "key 는 0-9, *, # 1~4자입니다.")
        if self.pending and self.pending["kind"] == "secure_input":
            raise ApiError(409, "sensitive_input_blocked", "민감정보는 secure_input 으로 입력하십시오.")
        if self.call is None or self.answered_at is None:
            raise ApiError(409, "invalid_state", "통화가 연결되지 않았습니다.")
        self._close_request()
        self.key_queue.clear()          # 이전 계층 복귀 중이었으면 복귀를 그만둔다
        self.after_backtrack = False
        if self.agent_active:
            self.ai_hold_until = time.monotonic() + self.s.user_key_hold_s
            self.timer("ai_hold", self.s.user_key_hold_s, self._resume_after_hold)
        await self._press(key, "user", self.current)
        return {"accepted": True, "key_sent": key}

    async def _resume_after_hold(self) -> None:
        async with self._lock:
            if self.agent_active and self.current_fresh and self.current and self.pending is None:
                await self._decide_heard(self.current)

    async def _act_secure(self, body: dict) -> dict:
        req = self._check_request(body, "secure_input")
        digits = str(body.get("digits", ""))
        if not re.fullmatch(r"[0-9*#]{1,20}", digits):
            raise ApiError(400, "invalid_request", "digits 는 0-9, *, # 1~20자입니다.")
        keys = digits + (req.get("terminator") or "")
        del digits
        self.pending = None
        dur = await self.call.send_keys(keys)  # 메모리에서 톤으로 변환하여 전송
        del keys
        self.segmenter.mute(dur)
        self.current_fresh = False
        self.last_key_actor = "user"
        self.set_status(self.idle_status())
        return {"accepted": True}

    async def _act_revise_goal(self, body: dict) -> dict:
        """다시 말하기 (피그마 03): 목적 설정 단계, 즉 첫 번호를 누르기 전까지만 받는다.
        새 목적으로 바꾸고 지금 머물러 있는 메뉴(첫 메뉴)에서 다시 판단한다."""
        text = str(body.get("text", "")).strip()
        if not text:
            raise ApiError(400, "invalid_request", "text(목적)가 비어 있습니다.")
        if self.path or self.status == "connected_to_agent":
            raise ApiError(409, "invalid_state",
                           "목적은 첫 번호를 누르기 전까지만 바꿀 수 있습니다. 이후에는 재질문에 답하거나 직접 선택하십시오.")
        self.goal = text
        self.goal_summary = await self.d.interp.summarize_goal(text)
        self.summary = None
        self._close_request(keep_secure=True)
        self.key_queue.clear()
        await self._activate_agent()
        return {"accepted": True, "status": self.status, "goal_summary": self.goal_summary}

    async def _activate_agent(self) -> None:
        """새 목적으로 탐색을 (다시) 시작한다. 지금 머물러 있는 계층(가장 최근 메뉴)에서 판단한다."""
        self.agent_active = True
        self.nav = NavState(self.goal or "")
        self.agent_goal_at_caption = len(self.captions)
        self.redo_same_menu = self.after_backtrack = False
        self.input_errors = 0
        if self.pending and self.pending["kind"] == "secure_input":
            return  # 직접 입력이 끝나면 다음 안내부터 탐색한다
        if self.answered_at is None:
            return  # 상대가 받으면 listening 으로 전환된다
        self.set_status("listening", self.goal_summary)
        if self.current is not None and self.current_fresh:
            await self._decide(self.current, arrived=True)
        # 마지막 키 전송 이후 아직 메뉴를 듣지 못했으면 다음 안내를 기다린다 (ARS 가 다시 읽거나 다음 메뉴를 안내)

    # 결과 (4.4) ---------------------------------------------------------------
    def result(self) -> tuple[int, dict]:
        if not self.ended:
            return 202, {"status": self.status}
        outcome = {"completed": "success", "agent_call_ended": "partial", "message_sent": "partial"}.get(
            self.end_reason or "", "failed")
        dur = round((self.ended_at or 0) - self.answered_at) if self.answered_at else 0
        return 200, {
            "outcome": outcome,
            "end_reason": self.end_reason,
            "summary": (self.summary or {}).get("summary"),
            "key_info": (self.summary or {}).get("key_info", []),
            "handoff_script": (self.summary or {}).get("handoff_script"),
            "path": self.path,
            "duration_sec": dur,
        }


class SessionManager:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.sessions: dict[str, Session] = {}

    def active(self) -> Session | None:
        return next((s for s in self.sessions.values() if not s.ended), None)

    def get(self, sid: str, token: str | None) -> Session:
        s = self.sessions.get(sid)
        if s is None:
            raise ApiError(404, "not_found", "세션이 없습니다.")
        if not token or not secrets.compare_digest(token, s.token):
            raise ApiError(401, "unauthorized", "세션 토큰이 올바르지 않습니다.")
        if s.ended and time.monotonic() - (s.ended_at or 0) > self.deps.settings.token_ttl_s:
            raise ApiError(401, "unauthorized", "만료된 세션 토큰입니다.")
        return s

    async def create(self, phone: str, goal: str | None) -> dict:
        if not goal or not str(goal).strip():
            raise ApiError(400, "invalid_request", "goal(목적)이 필요합니다. AI 도우미 사용과 목적은 발신 시 정합니다.")
        cur = self.active()
        if cur is not None:
            raise ApiError(409, "invalid_state", "다른 통화가 진행 중입니다. (동시 통화 1건)")
        digits = "".join(ch for ch in phone if ch.isdigit())
        if not 3 <= len(digits) <= 13:
            raise ApiError(400, "invalid_request", "전화번호 형식이 올바르지 않습니다.")
        s = Session(digits, self.deps)
        self.sessions[s.id] = s
        return await s.create(goal)
