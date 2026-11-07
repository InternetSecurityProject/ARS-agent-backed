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

from audio import FRAME, VALID_KEYS, Segmenter, rms
from config import Settings
from interpreter import Interpreter, Option, Prompt
from navigator import Layer, NavState, Navigator
from recording import CallRecorder
from recovery_budget import MenuRecoveryBudgets
from dtmf_echo import DtmfEchoGuard
from readback_display import (DIGIT_RUN, SPOKEN_LONG_RUN, readback_message, readback_options,
                              phone_confirmation_source)
from phone_readback import PhoneReadbackPermit
from phone_comparison import PhoneComparison, comparison_message
from choice_display import choice_display_options
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

# 입력값 되읽기: "입력하신 번호는 123456, 맞으면 1번" 처럼 입력 표현과 확인 표현이 함께 있는 안내
_READBACK_RE = re.compile(r"(입력하신|누르신|입력한).*(맞으|맞습니까|확인|정확)|(맞으|맞습니까|확인|정확).*(입력하신|누르신|입력한)")
_DIGITS_RE = DIGIT_RUN
_NO_ANSWER_RE = re.compile(r"틀|아니|다시|재입력|취소|정정|수정")  # 되읽기 확인에서 '아니요' 쪽 선택지 이름
_SPOKEN_RUN_RE = SPOKEN_LONG_RUN  # 문장부호로 끊긴 숫자 되읽기도 포함


def _obj(word: str) -> str:
    """목적격 조사 (을/를)."""
    ch = word.strip()[-1:] or "가"
    return "을" if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 else "를"


def _digit_count(s: str) -> int:
    return len(re.sub(r"[^0-9공영일이삼사오육칠팔구]", "", s))


def key_name(key: str) -> str:
    """키를 읽는 말: 3 → '3번', * → '별표', # → '샵'."""
    return {"*": "별표", "#": "샵"}.get(key, f"{key}번")


def caption_text(p: Prompt, text: str) -> str:
    """ARS 자막은 받아쓴 문장을 보존한다. AI 메뉴/미확정 라벨로 재구성하지 않는다.

    민감정보 마스킹과 되읽기·입력 안내 제외는 호출 전 _paragraph에서 처리한다.
    메뉴 옵션과 이동 경로는 기존 구조화 이벤트를 사용한다.
    """
    return text


def mask_digits(text: str, length: int | None = None) -> tuple[str, bool]:
    """ARS 가 입력값을 다시 읽어 주는 문장에서 숫자열을 가린다.
    length 를 주면 사용자가 입력한 자릿수와 같은 숫자열만 가린다 (금액·날짜 같은 다른 숫자는 남긴다)."""
    def rep(m: re.Match) -> str:
        n = _digit_count(m.group())
        if length is not None and n != length:
            return m.group()
        return "*" * (n or 4) + (" " if m.group().endswith(" ") else "")

    masked, n = _DIGITS_RE.subn(rep, text)
    return masked, masked != text


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

    def publish(self, event: str, data: dict, *, history_data: dict | None = None) -> None:
        msg = {"event": event, "at": now_iso(), "data": data}
        self.history.append(msg if history_data is None else {**msg, "data": history_data})
        for q in list(self.queues):
            q.put_nowait(msg)


@dataclass
class Deps:
    settings: Settings
    backend: Backend
    recognizer: Recognizer
    interp: Interpreter
    navigator: Navigator
    private_recognizer: Recognizer | None = None


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
        fork = getattr(deps.interp, 'new_session', None)
        self.interp = fork() if fork else deps.interp
        self.navigator = (Navigator(self.interp, deps.navigator.threshold, deps.navigator.margin,
                                   deps.navigator.max_revisit, deps.navigator.max_presses, deps.navigator.low)
                          if fork else deps.navigator)
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
        self.recorder: CallRecorder | None = None   # 시험용 통화 기록 (record=true 로 만든 세션만, 메모리에만 보관)

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
        # Recovery budgets reset only on a real menu transition / goal change,
        # never merely because ASR wording changed during the same menu replay.
        self._system_recovery_budget = MenuRecoveryBudgets()

        self.pending: dict | None = None
        self.req_seq = 0
        self.secure_window = False
        self.ai_hold_until = 0.0
        self.summary: dict | None = None
        self.result_heard = False            # 결과 안내를 들음 (요약을 마치기 전에 끊겨도 completed)
        self.agent_goal_at_caption = 0       # AI 시작 시점의 자막 위치 (결과 요약 범위)
        self.hangup_seen = False             # 상대가 끊음. 남은 안내를 처리한 뒤 종료한다
        self.heard_since_press = False       # 마지막 키 전송 뒤 ARS 가 소리를 냈는지
        self.key_outstanding = False         # 키를 보냈고 아직 ARS 의 응답 안내를 듣지 못함 (입력 오류 안내의 귀속 판단)
        self.last_key_at: float | None = None  # 마지막 키 전송 시각. 그 전에 시작된 안내는 키 이전 상태의 안내다
        self.seg_started_at: float | None = None  # 지금 듣고 있는 단락의 소리가 시작된 시각
        self.last_send_secure = False        # 마지막으로 보낸 키가 민감정보 입력값인지
        self.secure_queued = False           # 민감정보 입력값을 받아 ARS 안내가 끝나기를 기다리는 중 (값은 전송 작업 안에만 있다)
        self.secure_abort = False            # 기다리는 동안 ARS 가 메뉴·결과 등 다른 안내로 넘어감 → 입력값을 보내지 않는다
        self.secure_len: int | None = None   # 마지막 민감정보 입력의 자릿수 (값은 저장하지 않는다)
        self.secure_label: str = "입력 번호"   # 마지막 민감정보 요청의 항목명 (되읽기 확인 단계의 경로 표시용)
        self.secure_sent_at = 0.0
        self.secure_errors = 0               # 민감정보 입력 뒤 연속 입력 오류 횟수
        self.queued_key: str | None = None   # 이전 계층 복귀 중 마지막으로 누른 대기열 키 (입력 오류 시 재전송)
        self.closed = False                  # ended 이벤트를 보냄. 이후에는 이벤트를 보내지 않는다

        self.segmenter = Segmenter(self.s.vad_rms, self.s.paragraph_silence_ms)
        self._prefetch = None
        self._prefetch_segments = set()
        self._segment_private = False
        self._segment_phone_ticket = None
        self._private_generation = 0
        self._segment_private_generation = 0
        self._private_tail_frames = 0
        self._phone_readback = PhoneReadbackPermit(
            self.s.allow_openai_phone_readback and self.s.stt == 'openai'
            and not self.d.recognizer.is_local)
        self._phone_confirmation_pending = False
        self._phone_comparison = PhoneComparison()
        self._private_display_rid = None
        self._private_display_message = None
        if self.s.stt_prefetch and callable(getattr(self.d.recognizer, 'new_segment', None)):
            self.segmenter.on_chunk = self._prefetch_chunk
        self.stt_q: asyncio.Queue = asyncio.Queue()
        self.stt_inflight = 0                # 인식을 기다리거나 처리 중인 단락 수
        self._processing_stt_task = None
        self._key_echo = DtmfEchoGuard()
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

    async def _ai(self, name: str, coro: Awaitable, fallback):
        """AI 모듈 호출. 실패(시간 초과·API 오류)해도 통화가 멈추지 않도록 대체값을 쓴다."""
        try:
            return await coro
        except Exception:
            log.exception("AI 모듈 %s 실패. 대체값으로 진행", name)
            return fallback

    async def _classify(self, text: str) -> Prompt:
        historical = getattr(self.interp, 'classify_history', None)
        if historical:
            return await historical(text, self.goal or '', self.layers)
        contextual = getattr(self.interp, 'classify_context', None)
        path = [layer.prompt.category_of(layer.key) for layer in self.layers]
        return (await contextual(text, self.goal or '', path) if contextual
                else await self.interp.classify(text))

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

    def publish(self, event: str, data: dict, *, record_data: dict | None = None) -> None:
        """이벤트 전송. ended 가 마지막 이벤트이며, 종료 뒤 늦게 끝난 처리의 이벤트는 보내지 않는다."""
        if not self.closed:
            if self.recorder is not None:
                self.recorder.event(event, data if record_data is None else record_data)
            if record_data is None:
                self.hub.publish(event, data)
            else:
                self.hub.publish(event, data, history_data=record_data)

    def set_status(self, status: str, message: str | None = None) -> None:
        if self.ended:
            return  # 종료 뒤 늦게 끝난 처리가 상태를 되돌리지 않는다
        msg = message or STATUS_TEXT.get(status, "")
        if status == self.status and msg == self.status_message:
            return
        self.status, self.status_message = status, msg
        self.publish("status_changed", {"status": status, "message": msg})

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
    async def create(self, goal: str | None, record: bool = False) -> dict:
        # AI 도우미 사용과 목적은 통화 시작 시 정한다. 승인 없이 바로 발신하고 첫 메뉴부터 탐색한다.
        out = {"session_id": self.id, "session_token": self.token, "phone_number": self.phone}
        self.goal = (goal or "").strip()
        if record:
            self.recorder = CallRecorder({"session_id": self.id, "phone": self.phone, "goal": self.goal,
                                          "backend": self.s.backend, "stt": self.s.stt, "interpreter": self.s.interpreter})
        self.goal_summary = await self._ai("summarize_goal", self.interp.summarize_goal(self.goal),
                                           f"'{self.goal}' 업무를 찾아 드릴게요.")
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
        if self.end_reason is not None or self.hangup_seen:
            return
        if reason == "remote_hangup":
            # 상대가 끊었다. 끊기 직전까지 들은 안내와 결과 요약을 마친 뒤 종료 사유를 정한다 (명세 7장 7항).
            # 백엔드의 종료 알림은 바로 돌려준다.
            self.hangup_seen = True
            self.spawn(self._after_hangup())
            return
        await self.finish("user_canceled" if reason == "local_hangup" else reason, hangup=False)

    async def _after_hangup(self) -> None:
        # 단락 끝(무음)을 기다리지 못한 마지막 안내. 결과 안내 직후 바로 끊는 ARS 가 많다
        if self.segmenter.speaking:
            self._queue_segment(bytes(self.segmenter.buf), self.seg_started_at)
            self.segmenter.reset()
        deadline = time.monotonic() + self.s.hangup_flush_s
        while (self.stt_inflight or self._lock.locked()) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if self.summary is None and (self.result_heard or self.status == "connected_to_agent"):
            # 결과(또는 상담원 인계) 요약이 제한 시간 안에 끝나지 않았다 (AI 응답 지연). 요약 없이 끝내지 않고
            # 안내 원문·기본 문장으로 summary 를 보낸 뒤 ended 를 보낸다. 늦게 끝난 AI 요약은 버린다 (명세 7장 7항, 9.3)
            log.warning("끊긴 뒤 %.0f초 안에 요약이 끝나지 않아 안내 원문으로 대신한다", self.s.hangup_flush_s)
            if self.result_heard:
                res = self._result_fallback()
                self.summary = {"kind": "result", "summary": res["summary"], "key_info": [], "handoff_script": None}
            else:
                self.summary = {"kind": "handoff", "summary": "상담원 연결까지 진행했습니다.",
                                "handoff_script": self._handoff_fallback()}
            self.publish("summary", self.summary)
        if self.status == "connected_to_agent":
            reason = "agent_call_ended"
        elif self.result_heard:
            reason = "completed"
        elif self.pending is not None:  # 민감정보 직접 입력·재질문 응답 대기 중
            reason = "user_input_timeout"
        else:
            reason = "failed"
        await self.finish(reason, hangup=False)

    async def finish(self, reason: str, hangup: bool = True, note: str = "") -> None:
        if self.end_reason is not None:
            return
        self.end_reason = reason
        self._private_generation += 1
        self._phone_readback.revoke()
        self._clear_phone_comparison()
        self._redact_private_display()
        self._phone_confirmation_pending = False
        self._discard_prefetch()
        self._key_echo.clear()
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
        self.publish("ended", data)
        self.closed = True
        if self.recorder is not None:
            asyncio.get_running_loop().call_later(self.s.recording_keep_s, self.drop_recording)
        for q in list(self.audio_out):
            q.put_nowait(None)
        self.stt_q.put_nowait(None)  # 음성 인식 작업 종료

    # 오디오 ----------------------------------------------------------------
    def _private_stt_audio(self) -> bool:
        if self.d.recognizer.is_local:
            return False
        if self.pending and self.pending.get('kind') == 'secure_input':
            return True
        if self._phone_confirmation_pending:
            return True  # The one recognized readback is already shown; wait for the user's reply.
        if self._phone_readback.active():
            return False
        if self.d.private_recognizer is not None and self.secure_window:
            return True  # Fail closed: unknown readbacks cannot become public by timer expiry.
        return bool(self.secure_queued or (self.secure_window and
                    (not self.secure_sent_at or time.monotonic()-self.secure_sent_at<self.s.secure_skip_s)))

    def _discard_prefetch(self):
        for context in self._prefetch_segments: context.discard()
        self._prefetch_segments.clear()
        self._prefetch = None

    def _prefetch_chunk(self, prefix: bytes, start: int):
        if self.closed or self.end_reason is not None or self.hangup_seen: return
        if self._private_stt_audio():
            self._segment_private = True
        if self._segment_private or self._phone_readback.active(): return
        if self._prefetch is None:
            if len(self._prefetch_segments)>=3: return
            self._prefetch = self.d.recognizer.new_segment()
            self._prefetch_segments.add(self._prefetch)
        self._prefetch.offer(prefix)

    async def _on_audio(self, frame: bytes) -> None:
        if self.recorder is not None and not self.closed:
            self.recorder.audio(frame)
        for q in list(self.audio_out):
            if q.qsize() < 50:  # 1초 이상 밀리면 버린다
                q.put_nowait(frame)
        if (self.call is None or self.call.provides_transcripts or self.status == "connected_to_agent"
                or self.hangup_seen or self.end_reason is not None):
            return
        frame=self._key_echo.filter(frame)
        private = self._private_stt_audio()
        self._private_tail_frames = 10 if private else max(0,self._private_tail_frames-1)
        was_speaking = self.segmenter.speaking
        if not was_speaking:
            self._segment_private_generation = self._private_generation
            self._segment_private = private or bool(self._private_tail_frames)
            self._segment_phone_ticket = (self._phone_readback.generation
                                          if self._phone_readback.active() else None)
        elif private:
            self._segment_private = True
            if self._prefetch is not None: self._prefetch.discard()
        seg = self.segmenter.feed(frame)
        if self.segmenter.speaking and not was_speaking:
            self.seg_started_at = time.monotonic()  # 이 단락의 소리가 시작됐다고 판정한 때 (키 이전 안내 판별용)
        if seg is not None or self.segmenter.speaking:
            self.heard_since_press = True
        if seg is not None:
            self._queue_segment(seg, self.seg_started_at)
        elif was_speaking and not self.segmenter.speaking:
            # The detector can discard a music/noise paragraph without
            # returning audio. Do not carry that paragraph's context forward.
            if self._prefetch is not None:
                self._prefetch.discard()
                self._prefetch_segments.discard(self._prefetch)
                self._prefetch = None
            self._segment_private = False
            self._segment_phone_ticket = None

    def _queue_segment(self, seg: bytes, started_at: float | None) -> None:
        self.stt_inflight += 1
        self.stt_q.put_nowait((seg, started_at, self.recorder.pos if self.recorder else 0,
                              self._prefetch, self._segment_private, self._segment_phone_ticket,
                              self._segment_private_generation))
        self._prefetch = None
        self._segment_private = False
        self._segment_phone_ticket = None

    def _ars_active(self) -> bool:
        """ARS 가 안내하는 중이거나, 끝난 안내를 아직 인식·처리하는 중인지 (가상 ARS 는 call.speaking 으로 알린다)."""
        return bool(getattr(self.call, "speaking", False) or self.segmenter.speaking or self.stt_inflight)

    def _ars_active_for_recovery(self) -> bool:
        active=self._ars_active()
        if (active and self._processing_stt_task is not None
                and self._processing_stt_task is asyncio.current_task()):
            # The paragraph being decided is done. Other queued recognition or
            # incoming speech must still prevent an interrupting replay.
            return bool(getattr(self.call,'speaking',False) or self.segmenter.speaking or self.stt_inflight>1)
        return active

    async def _stt_worker(self) -> None:
        while not self.ended:
            item = await self.stt_q.get()
            if item is None:
                break  # 통화 종료
            seg, started_at, end_sample = item[:3] if len(item)>=3 else (*item, 0)
            prefetched = item[3] if len(item)>3 else None
            private_segment = item[4] if len(item)>4 else False
            phone_ticket = item[5] if len(item)>5 else None
            private_generation = item[6] if len(item)>6 else self._private_generation
            del item
            rec = self.recorder
            try:
                if phone_ticket is not None and not self._phone_readback.valid(phone_ticket):
                    if prefetched is not None: prefetched.discard()
                    continue
                if private_segment or self._private_stt_audio():
                    if prefetched is not None: prefetched.discard()
                    if rec: rec.note('stt_skipped', reason='개인정보 보호 구간과 겹친 단락', seg_end_t=round(end_sample/8000,2))
                    if self.d.private_recognizer is not None:
                        await self._protected_segment(seg, started_at, private_generation)
                    continue
                if self.secure_window and not self.d.recognizer.is_local and phone_ticket is None:
                    # 외부 STT 에는 입력값을 다시 읽어 주는 구간을 보내지 않는다. 문장을 받지 못하면 구간을 닫을 계기가 없으므로
                    # 입력 전송 뒤 secure_skip_s 가 지나면 구간을 닫고 다시 인식한다 (그 사이의 되읽기 확인은 사용자가 소리를 듣고 직접 누른다)
                    if time.monotonic() - self.secure_sent_at < self.s.secure_skip_s:
                        if rec:
                            rec.note("stt_skipped", reason="민감정보 입력 뒤 외부 STT 로 보내지 않는 구간",
                                     seg_end_t=round(end_sample / 8000, 2))
                        continue
                    log.info("외부 STT: 되읽기 건너뛰기 %.0f초가 지나 인식을 재개한다", self.s.secure_skip_s)
                    self.secure_window = False
                voiced = sum(1 for i in range(0, len(seg) - FRAME + 1, FRAME) if rms(seg[i:i + FRAME]) > self.s.vad_rms) * FRAME / 8000
                if voiced < self.s.min_voiced_s:
                    # 키 전송(DTMF)에 끊긴 조각·딸깍음·잔향. 인식하면 "콩" 같은 글자가 자막으로 남는다 (설계안 2.4)
                    log.info("소리 %.2f초뿐인 단락은 인식하지 않는다", voiced)
                    if rec:
                        rec.note("stt_skipped", reason=f"소리 {voiced:.2f}초뿐인 단락", seg_end_t=round(end_sample / 8000, 2))
                    continue
                t_stt = time.monotonic()
                text = (await prefetched.finish(seg) if prefetched is not None
                        else await self.d.recognizer.recognize(seg))
                if rec:
                    record_text = phone_confirmation_source(text) if phone_ticket is not None else text
                    rec.stt(record_text, end_sample, len(seg), time.monotonic() - t_stt, self.s.stt)
                    if prefetched is not None: rec.note('stt_prefetch', **prefetched.metrics)
                del seg
                if phone_ticket is not None and not self._phone_readback.valid(phone_ticket):
                    continue  # A new request/close must not receive this older result.
                if text and len(re.sub(r"[^가-힣0-9A-Za-z]", "", text)) < 3:
                    log.info("두 글자 이하 인식 결과는 버린다: %s", text)
                    continue
                if text:
                    self._processing_stt_task=asyncio.current_task()
                    try:
                        if phone_ticket is not None:
                            await self.on_paragraph(text, started_at, phone_readback_ticket=phone_ticket)
                        else:
                            await self.on_paragraph(text, started_at)
                    finally:
                        self._processing_stt_task=None
            except Exception:
                log.exception("안내 처리 오류")  # 한 단락의 오류로 인식 작업이 멈추지 않게 한다
            finally:
                if prefetched is not None:
                    prefetched.discard()
                    self._prefetch_segments.discard(prefetched)
                self.stt_inflight -= 1

    async def _protected_segment(self, audio: bytes, started_at: float | None, generation: int) -> None:
        """Protected audio/text never goes through the external STT/interpreter."""
        if generation != self._private_generation or self.ended:
            return
        started = time.monotonic()
        try:
            text = await self.d.private_recognizer.recognize(audio)
        except Exception:
            text = ''  # Recognizer errors can contain private text. Do not log them.
        from protected_input import protected_prompt
        p = protected_prompt(text)
        async with self._lock:
            if (generation != self._private_generation or self.ended
                    or self.status == 'connected_to_agent'
                    or (started_at is not None and self.last_key_at is not None and started_at < self.last_key_at)):
                return
            if self.hangup_seen and (p is None or p.kind != 'result'):
                return  # Final results can flush; new requests/controls cannot.
            if self.recorder:
                self.recorder.note('protected_stt', engine='local_whisper',
                    seconds=round(time.monotonic()-started,3), kind=p.kind if p else 'unresolved')
            if p is None:
                if self.pending is None:
                    self.set_status(self.idle_status(), '개인정보 안내를 로컬에서 확인 중이에요. 인식이 불확실하면 ARS 소리로 확인해 주세요.')
                return
            self.cancel_timer('no_response')
            self.key_outstanding = False
            same_input = (p.kind == 'sensitive' and p.label and
                          re.sub(r'\s*(?:앞|뒤)?\s*\d+자리$', '', p.label).strip() == self.secure_label)
            if self.secure_queued and same_input:
                return  # Still the same input prompt; let the queued value send after recognition.
            if self.secure_queued:
                self.secure_abort = True
                task = getattr(self, '_secure_task', None)
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if self.pending and self.pending['kind'] == 'secure_input' and same_input:
                self.pending['expires_at'] = now_iso(self.s.secure_expire_s)
                self.publish('request', self.pending)
                return
            if p.kind == 'sensitive':
                self._close_request()
                await self._ask_secure(p)  # Canonical field metadata, never the raw paragraph.
            elif p.kind in ('confirm', 'menu'):
                # Consume the phone comparison before closing the old request,
                # whose privacy cleanup deliberately clears retained values.
                display_message=self._readback_message(text) if p.kind=='confirm' else text
                self._close_request()
                if p.kind == 'confirm':
                    p.readback_of = self.secure_label
                self.current, self.current_fresh = p, True
                display_options=choice_display_options(text,p.options)
                if p.kind=='confirm':
                    p._readback_ui_options=display_options
                # Button-only: spoken replies must not forward a private UI
                # question to an external merge_answer call. History is redacted.
                await self._ask(p, 'critical_action', display_options,
                    display_message,
                    record_message=p.text)
            elif p.kind == 'input_error':
                self._close_request()
                self.set_status(self.idle_status(), p.text + ' 다음 입력 안내를 기다리고 있어요.')
            elif p.kind == 'result':
                self._close_request()
                self.result_heard = True
                self.agent_active = False
                self.key_queue.clear()
                # Safe generic spoken summary; private result stays in UI-only
                # key_info, not cloud summarization, caption/history or logs.
                self.summary = {'kind': 'result', 'summary': p.text,
                                'key_info': [{'label': 'ARS 결과 안내', 'value': text}], 'handoff_script': None}
                self.publish('summary', self.summary, record_data={**self.summary, 'key_info': []})
                if not self.hangup_seen:
                    self.set_status('agent_done')

    async def mic_audio(self, frame: bytes) -> None:
        """앱 마이크 음성. 상담원 연결 중에만 통화로 보낸다 (그 외에는 사용자 음성을 중계하지 않는다)."""
        if self.call is None or self.ended or self.status != "connected_to_agent":
            return
        await self.call.send_audio(frame)

    async def _on_transcript(self, text: str, started_at: float | None = None) -> None:
        await self.on_paragraph(text, started_at)

    # 안내 처리 ---------------------------------------------------------------
    async def on_paragraph(self, text: str, started_at: float | None = None,
                           *, phone_readback_ticket: int | None = None) -> None:
        """ARS 안내 한 단락. started_at 은 그 안내의 소리가 시작된 시각(monotonic). 모르면 None."""
        if self.ended or self.status == "connected_to_agent":
            return
        async with self._lock:
            if phone_readback_ticket is not None and not self._phone_readback.valid(phone_readback_ticket):
                return
            await self._paragraph(text, started_at, phone_readback_source=phone_readback_ticket is not None)

    async def _paragraph(self, text: str, started_at: float | None = None,
                         *, phone_readback_source=False) -> None:
        interp = self.interp
        self.cancel_timer("no_response")
        self.heard_since_press = True  # ARS 가 키에 응답했다
        # 우리가 키를 보내기 전에 시작된 안내는 키 이전 상태를 말한 것이다 (예: AI 판단이 ARS 입력 대기보다 늦어
        # ARS 가 같은 메뉴를 다시 읽기 시작한 뒤 키를 누른 경우). 메뉴·확인 질문·입력 오류로는 다루지 않고 자막만 남긴다.
        stale = started_at is not None and self.last_key_at is not None and started_at < self.last_key_at
        replying = self.key_outstanding and not stale  # 이 안내가 우리가 보낸 키에 대한 ARS 의 첫 반응인지
        if not stale:
            self.key_outstanding = False

        # 되읽기 구간이 닫혔더라도 "입력하신 … 맞으면" 문장에 한국어 숫자를 길게 읽는 부분이 있으면 입력값 되읽기로 본다
        phone_choices = (readback_options(text, Prompt('confirm', '')) if phone_readback_source else [])
        readback = (bool(_READBACK_RE.search(text)) and (self.secure_window or bool(_SPOKEN_RUN_RE.search(text))))
        readback |= bool(phone_choices)
        if phone_readback_source and not readback:
            # Consent is for OpenAI recognition only. Even a wrong-length,
            # Arabic-only or partial readback must never reach Jev/captions.
            text = phone_confirmation_source(text)
        if self.secure_window or readback:
            # ARS 가 입력값을 다시 읽어 주는 안내("입력하신 번호는 … 맞으면 1번")는 숫자를 가린 뒤 판단만 하고 자막으로 보내지 않는다
            if readback:
                consented_phone = self._phone_readback.active()
                self._phone_readback.revoke()
                masked = phone_confirmation_source(text) if phone_readback_source else mask_digits(text)[0]
                important = bool(phone_readback_source and re.search(r'결제|송금|이체|해지|해약|계약|가입', text))
                # 판단(Gemini)·자막·로그에는 숫자를 가린 문장을 쓴다. 사용자에게 묻는 문구만 원문을 그대로 쓴다:
                # 사용자가 입력한 값이 맞게 전달됐는지 화면에서 확인하려면 되읽힌 숫자를 봐야 한다 (사용자 결정 2026-10-27).
                # 서버는 이 문장을 저장하지 않으며, 요청이 닫히면 사라진다
                shown = text
                del text
                p = await self._ai("classify", self._classify(masked), Prompt(kind="info", text=masked))
                p.critical |= important
                if phone_choices and p.kind not in ('menu', 'confirm'):
                    # Explicit source-proven confirmation asks the user; it
                    # never authorizes an automatic business-menu selection.
                    p = Prompt('confirm', masked, phone_choices, critical=p.critical)
                if p.kind in ("menu", "confirm"):
                    p.readback_of = self.secure_label  # 경로 표시용: 어떤 입력을 확인한 단계인지
                if p.kind == "sensitive":
                    await self._ask_secure(p)
                elif p.kind in ("menu", "confirm"):
                    self.current, self.current_fresh = p, True
                    if self.agent_active:
                        self._phone_confirmation_pending |= consented_phone
                        p._readback_ui_options = readback_options(shown, p)
                        # The live UI contains private values. Keep replies
                        # button-only so a spoken number cannot enter intent AI.
                        await self._ask(p, 'critical_action', p._readback_ui_options,
                                        self._readback_message(shown), record_message=p.text)
                return
            # 되읽기 문장이 아니어도, 입력한 자릿수와 같은 숫자열이 들리면 입력값으로 보고 가린다 (금액·날짜는 자릿수가 달라 남는다).
            # 인식이 한두 글자 틀려 자릿수가 달라져도, 한국어 숫자를 여섯 자 이상 이어 읽는 것은 입력값 되읽기이므로 가린다
            text, hit = mask_digits(text, self.secure_len)
            # Mixed Hangul/Arabic readback can have an STT length error too.
            # Do not turn a different-length ordinary Arabic balance/date into
            # private digits merely because a secure window is still closing.
            guarded = _SPOKEN_RUN_RE.sub(
                lambda m: '*' * _digit_count(m.group())
                if re.search('[공영일이삼사오육칠팔구]', m.group()) else m.group(), text)
            hit2 = guarded != text
            text = guarded
            if hit or hit2:
                log.info("민감정보 입력 뒤 숫자열을 가렸다")
            closing_secure = True
        else:
            closing_secure = False

        p = await self._ai("classify", self._classify(text), Prompt(kind="info", text=text))
        if p.kind != 'info':
            self._phone_readback.revoke()
            self._phone_confirmation_pending = False
        if self.recorder is not None:
            self.recorder.note("classified", kind=p.kind, text=text,
                               options=[f"{o.key}:{o.label}" for o in (p.options or [])])
        if self.secure_queued and p.kind not in ("sensitive", "input_error", "info"):
            log.info("입력값 전송 대기 중 ARS 가 다른 안내(%s)로 넘어감. 입력값을 보내지 않는다", p.kind)
            self.secure_abort = True
        if closing_secure and (p.kind != "info" or hit or hit2):
            # 되읽기 구간은 메뉴·결과·입력 요청처럼 다음 단계가 분명한 안내가 와야 닫는다.
            # "잠시만 기다려 주십시오" 같은 짧은 안내로 닫으면, 그 뒤에 오는 실제 되읽기 문장의 숫자가 가려지지 않는다
            self.secure_window = False
        if p.kind == "input_error" and (not replying or stale):
            # 우리가 보낸 키에 대한 반응이 아닌 '입력 시간이 초과되었습니다'·'잘못 누르셨습니다'(사용자 응답 대기 중
            # ARS 의 대기 시간 경과 등)는 입력 오류가 아니라 ARS 의 안내일 뿐이다 (명세 5.2)
            p.kind = "info"
        if stale and p.kind in ("menu", "confirm"):
            p.kind = "info"
        if p.kind != "input_error" and not self.redo_same_menu:
            self.input_errors = 0  # 직전 키가 받아들여졌다. 입력 오류 '2회 연속'을 다시 센다
        if p.kind != "input_error":
            self.queued_key = None
            if p.kind != "sensitive":
                self.secure_errors = 0
        self.current_fresh = True if p.kind in ("menu", "confirm") else self.current_fresh
        # 키 전송에 잘린 안내(키보다 먼저 시작된 단락, 또는 키 전송 뒤 post_key_caption_s 안에 끝난 일반 안내)는 자막으로 보내지 않는다.
        # "이번에 입력해 주십시오" 같은 조각이 남는 것을 막는다. 메뉴·결과·확인처럼 다음 단계가 분명한 안내는 그대로 보낸다
        cut = p.kind == "info" and (stale or (self.last_key_at is not None
                                             and time.monotonic() - self.last_key_at < self.s.post_key_caption_s))
        if cut:
            log.info("키 전송에 잘린 안내는 자막으로 보내지 않는다: %s", text[:40])
        if p.kind not in ("sensitive", "input_error") and not cut:
            cap = {"step": self.step + 1, "speaker": "ars", "text": caption_text(p, text), "at": now_iso()}
            self.captions.append(cap)
            self.publish("caption", cap)

        # AI 사용 여부와 관계없이 처리하는 안내
        if p.kind == "sensitive":
            if stale and self.last_send_secure:
                # 사용자가 번호를 누르는 동안 ARS 가 같은 입력 안내를 되풀이한 것이 뒤늦게 인식된 경우.
                # 이미 입력을 보냈으므로 새 입력 요청을 만들지 않는다 (만들면 사용자가 같은 번호를 또 입력해 '잘못된 입력'이 반복된다)
                log.info("입력 전송 전에 시작된 입력 안내가 뒤늦게 도착. 무시한다")
                return
            if self.secure_queued:
                # 입력값을 받아 ARS 안내가 끝나기를 기다리는 중에 같은 입력 안내가 다시 들렸다. 안내가 끝나면 그 값을 보낸다
                log.info("입력값 전송 대기 중 입력 안내가 다시 들림. 새 입력 요청을 만들지 않는다")
                return
            if self.pending and self.pending["kind"] == "secure_input":
                # 입력하는 동안 ARS 가 안내를 되풀이함. 요청은 그대로 두고 만료 시각만 늘려 같은 요청을 다시 알린다
                self.pending["expires_at"] = now_iso(self.s.secure_expire_s)
                self.publish("request", self.pending)
                return
            await self._ask_secure(p)
            return
        if p.kind == "input_error":
            if self.last_send_secure:
                # 민감정보 입력값에 대한 오류. 입력값은 어디에도 남기지 않으며(keys=null), 메뉴 키 오류로 세지 않는다.
                # ARS 가 입력 안내를 다시 하면 새 secure_input 요청이 나간다 (명세 6.3 처리 규칙 6)
                self.secure_errors += 1
                self.publish("input_rejected", {
                    "keys": None, "actor": "user", "ars_prompt": text, "retry_count": self.secure_errors,
                    "message": "입력하신 번호가 인식되지 않았습니다. 안내에 따라 다시 입력해 주세요.",
                })
                return
            self.publish("input_rejected", {
                "keys": self.path[-1]["selected"]["key"] if self.path else None,
                "actor": self.last_key_actor, "ars_prompt": text, "retry_count": self.input_errors + 1,
                "message": "번호가 인식되지 않아 다시 시도합니다." if self.agent_active else "번호가 인식되지 않았습니다.",
            })
            if self.agent_active:
                self.input_errors += 1
                self.redo_same_menu = True
                if self.queued_key:
                    self.key_queue.insert(0, self.queued_key)  # 이전 계층 복귀 중 거부된 키는 메뉴가 다시 나오면 다시 누른다
                    self.queued_key = None
                if self.input_errors >= 2 and self.current:
                    await self._ask(self.current, "input_rejected", self.current.options)
            return
        if p.kind == "agent_human":
            if self.agent_active or self.status == "waiting_agent":
                await self._handoff()
            return
        if p.kind in ("menu", "confirm"):
            self.current = p
            if self.pending and self.pending["kind"] == "secure_input":
                # 입력을 기다리던 ARS 가 입력 없이 메뉴로 돌아갔다. 민감정보 입력 요청을 닫고 메뉴를 판단한다
                log.info("민감정보 입력 대기 중 ARS 가 메뉴로 넘어감. 입력 요청을 닫는다")
                self._close_request()
                self.secure_window = False
                self.set_status(self.idle_status())

        if not self.agent_active:
            return  # 업무 완료(agent_done) 뒤의 안내는 자막만 전송한다

        # 이하 AI 탐색 중
        if self.pending is not None:
            return  # 사용자 응답 대기 중 ARS 가 안내를 다시 읽는 경우. 계속 듣는다.
        if self.key_queue and p.kind == "menu":
            if not self.hangup_seen:
                self.queued_key = self.key_queue.pop(0)
                try:
                    await self._press(self.queued_key, "ai", p)
                except ApiError as e:
                    log.warning("이전 계층 복귀 키 전송 실패: %s", e.message)
            return
        if p.kind in ("menu", "confirm"):
            if self.hangup_seen or time.monotonic() < self.ai_hold_until:
                return  # 끊긴 뒤에 인식한 메뉴이거나, 사용자가 직접 누른 직후 보류 시간 (보류가 끝나면 가장 최근 메뉴로 판단)
            await self._decide_heard(p)
        elif p.kind == "result":
            # 결과를 받으면 AI 만 마치고 통화는 유지한다. 사용자가 통화를 종료하면 end_reason=completed.
            # 요약을 마치기 전에 ARS 가 끊어도 completed 이며, 요약을 마친 뒤 summary → ended 순서로 보낸다 (_after_hangup)
            self.result_heard = True
            self.set_status("listening_result")
            try:
                res = await interp.summarize_result(self.goal or "", self._result_captions())
            except Exception:
                log.exception("결과 요약 실패")
                res = self._result_fallback()  # 결과 안내 원문으로 대신한다
            if self.ended or self.summary is not None:
                return  # 요약하는 동안 통화가 끊기고 제한 시간이 지나 _after_hangup 이 원문 요약을 이미 보냈다
            self.summary = {"kind": "result", "summary": res.get("summary", ""),
                            "key_info": res.get("key_info", []), "handoff_script": None}
            self.publish("summary", self.summary)
            self.agent_active = False
            self.key_queue.clear()
            if not self.hangup_seen:
                self.set_status("agent_done")  # 통화가 이미 끊겼으면 바로 ended 로 넘어간다
        elif p.kind in ("message_sent", "out_of_hours", "rejected"):
            await self.finish({"rejected": "ars_rejected"}.get(p.kind, p.kind), hangup=not self.hangup_seen)
        elif p.kind == "agent_wait":
            self.set_status("waiting_agent")
        elif p.kind == "voice_request" and not self.hangup_seen:
            # [검증 필요] 전환 문구 음성 송출(TTS)은 아직 구현하지 않았다. 사용자에게 알리고 응답을 받는다.
            await self._ask(p, "unknown_prompt", p.options,
                            "ARS 가 말로 대답하라고 합니다. 음성으로 답하는 기능은 아직 지원하지 않습니다. "
                            "번호로 고를 수 있으면 골라 주세요.")

    def _result_captions(self) -> list[str]:
        return [c["text"] for c in self.captions[self.agent_goal_at_caption:]]

    def _result_fallback(self) -> dict:
        """AI 요약을 쓸 수 없을 때의 결과 요약: 마지막 결과 안내 원문 (명세 9.3)."""
        caps = self._result_captions()
        return {"summary": caps[-1] if caps else "", "key_info": []}

    async def _handoff(self) -> None:
        # 상담원이 받았음을 먼저 알린다 (앱 마이크 중계가 바로 시작된다). 상황 설명 문장은 AI 응답(최대 12초)을 기다린 뒤
        # summary 로 보낸다. 설명 문장을 만드는 동안 끊기면 _after_hangup 이 기본 문장으로 summary 를 보낸다
        self.agent_active = False
        self.pending = None
        self.set_status("connected_to_agent")
        texts = [p["text"] for p in self.path]
        script = (await self._ai("handoff_script", self.interp.handoff_script(self.goal, texts),
                                 self._handoff_fallback()) if self.goal else None)
        if self.ended or self.summary is not None:
            return
        self.summary = {"kind": "handoff", "summary": "상담원 연결까지 진행했습니다.", "handoff_script": script}
        self.publish("summary", self.summary)

    def _handoff_fallback(self) -> str | None:
        return f"{self.goal} 건으로 연락드렸습니다." if self.goal else None

    # 판단 ------------------------------------------------------------------
    async def _decide_heard(self, p: Prompt) -> None:
        """새로 들은 메뉴·확인 질문을 판단한다.
        입력 오류 뒤 같은 메뉴, 이전 계층 복귀로 도착한 메뉴는 반복으로 세지 않는다."""
        arrived = not (self.redo_same_menu or self.after_backtrack)
        self.redo_same_menu = self.after_backtrack = False
        await self._decide(p, arrived)

    async def _decide(self, p: Prompt, arrived: bool) -> None:
        if not self.agent_active or self.pending is not None or self.ended or self.hangup_seen or self.nav is None:
            return
        # 메뉴·확인 질문을 판단하기 시작하면 selecting (명세 3.1).
        # 결과에 따라 listening(키 전송)·awaiting_user(재질문)·ended(not_found) 로 넘어간다
        if p.kind != "confirm":
            self.set_status("selecting")  # 확인 질문은 답을 정한 뒤에만 selecting (대부분 사용자에게 묻게 되어 상태가 깜빡인다)
        try:
            await self._decide_inner(p, arrived)
        except ApiError as e:
            log.warning("키 전송 실패: %s", e.message)  # 통화가 끊기는 중. 종료 처리는 _on_end 가 한다

    async def _decide_inner(self, p: Prompt, arrived: bool) -> None:
        if p.kind == "confirm":
            if p.critical:  # 결제·계약·해지 최종 승인 등은 확인 질문이어도 키를 보내지 않는다 (1.1 원칙 5)
                await self._ask(p, "critical_action", p.options)
                return
            key = await self._ai("answer_confirm", self.interp.answer_confirm(self.goal or "", p), None)
            if self.ended or self.hangup_seen:
                return  # 판단하는 동안 통화가 끊겼다
            if key:
                self.set_status("selecting")
                await self._press(key, "ai", p)
            else:
                if getattr(self.interp, 'system_recovery', False):
                    # A missing fact in an ordinary confirmation (e.g. whether
                    # the announced appointment is correct) still belongs to
                    # the user. Require the same verified alternatives as menus.
                    route = await self._ai('confirm_route', self.interp.assess_route(
                        self.goal or '', p, p.options,
                        [l.prompt.category_of(l.key) for l in self.layers]), None)
                    if self.ended or self.hangup_seen or self.pending is not None or not self.agent_active:
                        return
                    relevant = [o for o in p.options if o.key in getattr(route,'clarification_keys',())]
                    if getattr(route,'status',None)=='needs_info' and len(relevant)>=2:
                        await self._ask(p,'ambiguous',relevant,
                            message=f"{' / '.join(o.label for o in relevant)} 중 어떤 쪽인가요?")
                    else:
                        await self._recover_system(p)
                else:
                    await self._ask(p, "unknown_prompt", p.options)
            return
        try:
            dec = await self.navigator.decide(self.nav, p, self.layers, arrived)
        except Exception:
            log.exception("AI 모듈 score 실패")
            if not (self.ended or self.hangup_seen):
                if getattr(self.interp, 'system_recovery', False):
                    await self._recover_system(p)
                else:
                    await self._ask(p, "unknown_prompt", p.options,
                                    "안내를 해석하지 못했습니다. 원하시는 메뉴를 말씀하거나 직접 골라 주세요.")
            return
        if self.ended or self.hangup_seen:
            return  # 판단하는 동안 통화가 끊겼다
        log.info("판단 %s %s %s", dec.action, dec.key or dec.keys or dec.reason, dec.note)
        if dec.action == "press":
            self.nav.presses += 1
            self.nav.exclude.clear()
            self.nav.exclude_at = None
            self.set_status("selecting", f"'{p.label_of(dec.key)}' 메뉴를 선택합니다.")
            await self._press(dec.key, "ai", p, confidence=dec.scores.get(dec.key))
        elif dec.action == "ask":
            await self._ask(p, dec.reason, dec.options, message=dec.message, scores=dec.scores)
        elif dec.action == 'recover':
            await self._recover_system(p, reanalyze=dec.reason != 'navigation_limit')
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
            if dec.reason == "not_found" and callable(getattr(self.interp, 'assess_route', None)):
                self.set_status("selecting", "말씀하신 업무로 진행할 수 있는 메뉴를 찾지 못했어요. 통화를 종료할게요.")
            await self.finish(dec.reason, note=dec.note)

    async def _recover_system(self, p: Prompt, *, reanalyze=True) -> None:
        """No user-purpose question for an AI/STT failure. No guessed keys.

        One bounded internal reanalysis and one verified current-menu replay
        per distinct observed menu, with a shared per-layer cap. A persistent
        failure remains visibly listening.
        """
        if (self.ended or self.hangup_seen or not self.agent_active or self.pending is not None
                or p is not self.current or not self.current_fresh):
            return
        key = (self.goal, tuple((l.prompt.text,l.key) for l in self.layers))
        budget = self._system_recovery_budget.for_prompt(key,p)
        if budget is None:
            self.set_status('listening', '다음 안내를 기다리고 있어요.')
            return
        if reanalyze and not budget.analyzed:
            budget.analyzed = True
            self.set_status('selecting', '안내와 이전 경로를 다시 확인하고 있어요.')
            start = time.monotonic()
            try:
                fixed = await self.interp.recover_context(p, self.goal or '',
                    [l.prompt.category_of(l.key) for l in self.layers])
            except Exception as exc:
                log.warning('시스템 재판단 실패: %s', type(exc).__name__)
                fixed = p
            if self.ended or self.hangup_seen or self.pending is not None or not self.agent_active:
                return
            log.info('시스템 재판단 %.0fms', (time.monotonic()-start)*1000)
            self.current = fixed
            if self.recorder is not None:
                self.recorder.note('system_reanalysis', duration_s=round(time.monotonic()-start,3),
                    kind=fixed.kind, options=[f'{o.key}:{o.label}' for o in fixed.options])
            if fixed.kind in ('menu','confirm'):
                await self._decide_inner(fixed, arrived=False)
                return
            # Recovery cannot directly execute newly inferred secure/result or
            # non-menu states; only the ordinary paragraph pipeline can do so.
            p = fixed
        if self._ars_active_for_recovery():
            self.set_status('listening', '새 안내를 듣고 다시 확인하고 있어요.')
            return
        if p.repeat_key and not p.critical and not budget.replayed:
            budget.replayed = True
            self.redo_same_menu = True
            self.set_status('listening', '안내를 다시 듣고 확인할게요.')
            await self._press(p.repeat_key, 'ai', p)
            retry = getattr(self.interp, 'retry_uncertain_after_replay', None)
            if retry:
                retry(p)
            return
        self.set_status('listening', '안내를 정확히 확인하지 못해 자동 선택을 보류했어요. 새 안내를 기다리고 있어요.')

    async def _send_keys(self, keys: str, actor: str = "ai", secure: bool = False) -> float:
        """키(DTMF 톤) 전송. 키를 누르면 ARS 는 하던 안내를 끊거나 다음 안내로 넘어가므로,
        말하던 중인 단락은 버리고(앞뒤 안내가 한 자막으로 섞이지 않게) 보낸 톤의 반향은 듣지 않는다."""
        if self.call is None or self.ended or self.hangup_seen:
            raise ApiError(409, "invalid_state", "통화가 연결되지 않았습니다.")
        self._private_generation += 1
        self._discard_prefetch()
        self.segmenter.reset_for_key()
        self._segment_private = False
        self._segment_phone_ticket = None
        opened = self._phone_readback.begin(keys) if secure else False
        if secure:
            if self._phone_comparison.begin(keys):
                generation=self._phone_comparison.generation
                self.timer('phone_comparison',self._phone_comparison.seconds,
                           lambda:self._expire_phone_comparison(generation))
        else:
            self._clear_phone_comparison()
        if not secure:
            self._phone_readback.revoke()
            if self._phone_confirmation_pending:
                # The explicit reply leaves this already recognized phone
                # confirmation. The following public result must not be lost
                # to the old private-input countdown. A new secure request
                # arms its own normal guard independently.
                self.secure_window = False
                self._private_tail_frames = 0
                self._phone_confirmation_pending = False
        if opened:
            # Only after the send boundary has discarded older buffered audio.
            self._private_tail_frames = 0
            log.info('전화번호 재확인: 승인된 OpenAI STT 구간 시작')
        self.seg_started_at=None
        self.last_key_at = time.monotonic()  # 전송 시작 시각. 이보다 먼저 시작된 안내는 키 이전 상태의 안내다
        self.heard_since_press = False
        self.key_outstanding = True
        self.last_send_secure = secure
        if getattr(self.call,'dtmf_mode','inband')!='sdk':
            self._key_echo.start(keys)
        if self.recorder is not None:
            self.recorder.keys(keys, actor, secure)  # 민감정보는 값 대신 자릿수·대체 신호음만 남긴다
        try:
            dur = await self.call.send_keys(keys)
        except Exception as e:
            self._phone_readback.revoke()
            self._clear_phone_comparison()
            self._discard_prefetch()
            self._key_echo.clear()
            log.warning("키 전송 실패: %s", type(e).__name__)
            raise ApiError(409, "invalid_state", "키를 전송하지 못했습니다. 통화가 끊겼을 수 있습니다.")
        self._key_echo.finish()
        if secure:
            self._system_recovery_budget.clear()
        return dur

    async def _press(self, keys: str, actor: str, p: Prompt | None, confidence: float | None = None) -> None:
        await self._send_keys(keys, actor)
        if p is None or keys != p.repeat_key:
            # Confirmations/manual selections can transition without adding a
            # Layer. A real non-replay input starts a new recovery budget too.
            self._system_recovery_budget.clear()
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
        readback_ui = getattr(p, '_readback_ui_options', p.options if p else [])
        if p is not None and getattr(p, "readback_of", None) and any(o.key == keys for o in readback_ui):
            # 입력값 되읽기 확인 단계: 경로에는 "확인" 같은 답이 아니라 무엇을 확인했는지 남긴다
            chosen_label = next(o.label for o in readback_ui if o.key == keys)
            label = f"{p.readback_of} {'재입력' if _NO_ANSWER_RE.search(chosen_label) else '확인'}"
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
        self.publish("path_updated", step)
        if self.agent_active and self.pending is None:
            self.set_status("listening")  # 재질문에 select·key 로 답한 경우도 awaiting_user 를 해제한다
        if self.agent_active:
            # 무응답 복구(다시 듣기 키 재전송)는 AI 탐색 중에만 한다. 업무 완료·상담원 연결 뒤
            # 사용자가 직접 누른 키에는 자동 재전송을 걸지 않는다 (상담원 통화에 DTMF 가 끼어들지 않게).
            self.timer("no_response", self.s.no_response_s, lambda: self._no_response(p))

    async def _no_response(self, p: Prompt | None) -> None:
        """키 전송 후 일정 시간(기본 8초) 새 안내가 없을 때 (명세 7장 3항).
        그동안 ARS 가 아무 소리도 내지 않았으면 키가 인식되지 않아 같은 메뉴에서 기다리는 것으로 보고,
        같은 키를 다시 보내지 않고 키를 누른 그 메뉴의 다시 듣기 키를 누른다.
        소리를 냈으면(다음 안내를 하는 중이거나 인식 중) 누르지 않고 안내가 끝나기를 기다린다.
        이전 메뉴의 다시 듣기 키는 다음 메뉴에서 다른 뜻일 수 있다."""
        async with self._lock:
            if (self.ended or self.hangup_seen or not self.agent_active or p is None
                    or not p.repeat_key or self.current_fresh or self.pending is not None):
                return
            if self.heard_since_press or self._ars_active():
                return
            recovery_key = (self.goal, tuple((l.prompt.text,l.key) for l in self.layers))
            budget = self._system_recovery_budget.for_prompt(recovery_key,p,create=False)
            if budget is not None and budget.replayed:
                self.set_status('listening', '다시 듣기 응답을 확인하지 못해 자동 선택을 보류했어요. 새 안내를 기다리고 있어요.')
                return
            try:
                await self._send_keys(p.repeat_key)
            except ApiError:
                return
            self.redo_same_menu = True  # 다시 듣기 키로 다시 나온 메뉴는 반복으로 세지 않는다 (명세 7장 5항)

    # 사용자 요청 -------------------------------------------------------------
    def _clear_phone_comparison(self):
        self._phone_comparison.clear()
        self.cancel_timer('phone_comparison')

    async def _expire_phone_comparison(self,generation):
        if generation==self._phone_comparison.generation:
            self._clear_phone_comparison()

    def _readback_message(self,text):
        entered=self._phone_comparison.take()
        self.cancel_timer('phone_comparison')
        return (comparison_message(text,entered) if entered is not None
                else readback_message(text,self.secure_label,self.secure_len))

    def _redact_private_display(self):
        redacted=False
        if self.pending and self.pending.get('request_id')==self._private_display_rid:
            # Mutate the live request too: undelivered queue aliases must not
            # retain the old private message after its request has closed.
            self.pending['message']=self._private_display_message or '개인정보 확인 안내입니다.'
            redacted=True
        self._private_display_rid=None
        self._private_display_message=None
        return redacted

    def _new_request(self, kind: str, expire_s: float, **fields) -> dict:
        self._redact_private_display()
        if kind!='secure_input':self._clear_phone_comparison()
        self.req_seq += 1
        req = {"request_id": f"req_{self.req_seq}", "kind": kind, **fields, "expires_at": now_iso(expire_s)}
        self.pending = req
        return req

    async def _ask(self, p: Prompt, reason: str, options: list[Option], message: str | None = None,
                   scores: dict[str, float] | None = None, *, record_message: str | None = None) -> None:
        options=choice_display_options(p.text,options)
        labels = [o.label for o in options]
        scores = scores or {}
        default = {
            "ambiguous": f"{' / '.join(labels)} 중 어느 쪽인가요? 원하시는 업무를 말씀해 주세요.",
            "low_confidence": (f"'{labels[0]}' 메뉴가 맞는지 확실하지 않습니다. " if labels else "")
                              + "원하시는 업무를 조금 더 자세히 말씀해 주세요.",
            "critical_action": "결제·계약·해지처럼 중요한 단계입니다. 진행하려면 화면에서 번호를 직접 골라 주세요.",
            "loop": "같은 메뉴가 반복되고 있습니다. 원하시는 업무를 조금 더 자세히 말씀해 주세요.",
            "unknown_prompt": p.text,  # 안내문 그대로 보여 준다 (앞뒤 설명을 붙이거나 자르지 않는다)
            "input_rejected": "번호가 계속 인식되지 않습니다. 원하시는 메뉴를 말씀하거나 직접 골라 주세요.",
        }
        req = self._new_request("clarification", self.s.clarify_expire_s, reason=reason,
                                message=message or default.get(reason, "확인이 필요합니다."),
                                options=[{**o.as_dict(), "confidence": scores.get(o.key)} for o in options],
                                ars_prompt=p.text)
        if record_message is not None:
            self._private_display_rid=req['request_id']
            self._private_display_message=record_message
        self.set_status("awaiting_user", record_message if record_message is not None else req["message"])
        self.publish("request", req, record_data={**req, "message": record_message} if record_message is not None else None)
        rid = req["request_id"]
        self.timer("clarify_expire", self.s.clarify_expire_s, lambda: self._clarify_expire(rid))
        self.timer("clarify_drop", self.s.clarify_drop_s, lambda: self._clarify_drop(rid))

    async def _clarify_expire(self, rid: str) -> None:
        """30초: ARS 가 통화를 끊지 않도록 지금 메뉴의 다시 듣기 키를 1회 누른다. 요청은 유지한다.
        ARS 가 안내하는 중이면 안내가 끝난 뒤 누른다 (안내 중에 누르면 안내가 끊기고 자막이 섞인다)."""
        async with self._lock:
            if not (self.pending and self.pending["request_id"] == rid) or self.ended or self.hangup_seen:
                return
            if self._redact_private_display():
                self._clear_phone_comparison()
                self.pending['message']='확인 시간이 지나 번호를 가렸어요. ARS 안내를 확인하거나 다시 입력해 주세요.'
                self.publish('request',self.pending)
            if self._ars_active():
                self.timer("clarify_expire", 0.5, lambda: self._clarify_expire(rid))
                return
            if self.current and self.current.repeat_key:
                try:
                    await self._send_keys(self.current.repeat_key)
                except ApiError:
                    pass

    async def _clarify_drop(self, rid: str) -> None:
        """60초: 요청을 무효로 하고 상담원 연결 메뉴로 진행한다.
        상담원 연결 선택지가 없으면 요청을 유지한 채 계속 듣는다 (ARS 가 통화를 끊으면 user_input_timeout)."""
        async with self._lock:
            if not (self.pending and self.pending["request_id"] == rid) or self.ended:
                return
            cur = self.current
            if cur and cur.agent_key and self.s.drop_to_agent:
                self._close_request()
                self.set_status("selecting", "응답이 없어 상담원 연결 메뉴로 진행합니다.")
                try:
                    await self._press(cur.agent_key, "ai", cur)
                except ApiError:
                    pass

    async def _ask_secure(self, p: Prompt) -> None:
        self._private_generation += 1
        self._discard_prefetch()
        self.secure_window = True
        label = p.label or "요청 정보"
        # "주민등록번호 앞 6자리"·"계좌 비밀번호 네 자리" → "주민등록번호"·"계좌 비밀번호" (경로에 보일 항목명)
        self.secure_label = re.sub(r"\s*(?:앞|뒤)?\s*(?:[0-9]+|[한두세네]|다섯|여섯|일곱|여덟|아홉|열\s?[한두세네]?)\s*자리\s*$", "", label).strip() or label
        self._phone_readback.arm(self.secure_label, p.text)
        self._redact_private_display()
        self._clear_phone_comparison()
        self._phone_comparison.arm(self.secure_label,p.text,p.terminator)
        self._phone_confirmation_pending = False
        req = self._new_request("secure_input", self.s.secure_expire_s, label=label, ars_prompt=p.text,
                                terminator=p.terminator,
                                message=f"{label}{_obj(label)} 직접 눌러 주세요. "+
                                ('확인이 끝나면 입력 번호를 지워요.' if self._phone_comparison.eligible
                                 else '입력하신 번호는 저장되지 않습니다.'))
        self.set_status("awaiting_user", "직접 입력이 필요합니다.")
        self.publish("request", req)

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
        if self.recorder is not None and t!='answer':
            self.recorder.user_action(body)
        async with self._lock:
            if t == "cancel":
                if self.status == "connected_to_agent":
                    reason = "agent_call_ended"
                elif self.result_heard:
                    reason = "completed"
                else:
                    reason = "user_canceled"
                await self.finish(reason, hangup=not self.hangup_seen)
                return {"accepted": True}
            if self.hangup_seen:
                raise ApiError(409, "invalid_state", "상대가 통화를 끊었습니다.")
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
        self._redact_private_display()
        self._clear_phone_comparison()
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
        # Validate the request and private/button-only gate BEFORE recording
        # answer text. A rejected private reply must not enter any text log.
        if self.recorder is not None:
            self.recorder.user_action(body)
        self._close_request()
        merge_for_prompt = getattr(self.interp, 'merge_answer_for_prompt', None)
        merged = (merge_for_prompt(self.goal or '', req.get('ars_prompt', ''), text,
                                  allow_context=req['reason'] == 'ambiguous')
                  if callable(merge_for_prompt)
                  else self.interp.merge_answer(self.goal or '', req['message'], text))
        self.goal = await self._ai('merge_answer', merged, text)
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
        # 마지막 키 전송 뒤 새 메뉴를 듣지 못한 상태(민감정보 안내 중, 안내가 끝나기 전 등)에서 누른 키는
        # 지난 메뉴의 선택지로 해석하지 않고 "N 입력" 으로만 기록한다 (6.2)
        await self._press(key, "user", self.current if self.current_fresh else None)
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
        term = req.get("terminator") or ""
        keys = digits if (term and digits.endswith(term)) else digits + term  # 사용자가 종료 키까지 누른 경우 중복하지 않는다
        self.secure_len = _digit_count(digits)  # 기본은 자릿수만; 단일 전화번호 대조는 전송 때 한시 보관
        self.secure_sent_at = time.monotonic()  # 외부 STT 는 이 뒤 secure_skip_s 동안 음성을 보내지 않는다
        del digits
        self.pending = None
        self.current_fresh = False
        self.last_key_actor = "user"
        if self._secure_send_blocked() and (self.s.secure_wait_max_s > 0
                or (self.d.private_recognizer is not None and self.stt_inflight)):
            # ARS 가 안내 중이다. '입력 시간이 초과되었습니다' 처럼 키를 받지 않는 안내 중에 보내면 입력이 무시되므로
            # 안내가 끝난 뒤 보낸다. 기다리는 동안 다른 처리(안내 인식·종료 요청)가 막히지 않도록 잠금 밖에서 기다린다
            self.secure_queued, self.secure_abort = True, False
            self.set_status(self.idle_status())
            self._spawn_secure_send(keys)
            del keys
            return {"accepted": True}
        await self._send_keys(keys, "user", secure=True)  # 메모리에서 톤으로 변환하여 전송
        del keys
        self.last_send_secure = True
        self.set_status(self.idle_status())
        return {"accepted": True}

    def drop_recording(self) -> None:
        """통화 기록을 메모리에서 지운다 (주인의 삭제 요청 또는 보관 시간 경과)."""
        if self.recorder is not None:
            self.recorder.wipe()
            self.recorder = None
            log.info("통화 기록을 지웠다 (%s)", self.id)

    def _ars_speaking(self) -> bool:
        """ARS 소리가 지금 나고 있는지 (단락 종료 무음 paragraph_silence_ms 전까지는 말하는 중으로 본다)."""
        return bool(getattr(self.call, "speaking", False) or self.segmenter.speaking)

    def _secure_send_blocked(self) -> bool:
        return self._ars_speaking() or bool(self.d.private_recognizer is not None and self.stt_inflight)

    def _spawn_secure_send(self, keys: str) -> None:
        task = asyncio.create_task(self._send_secure_when_quiet(keys))
        self._secure_task = task  # 작업 참조를 유지한다 (값은 작업이 끝나면 사라진다)

    async def _send_secure_when_quiet(self, keys: str) -> None:
        t0 = time.monotonic()
        try:
            while (self._secure_send_blocked() and not self.ended and not self.hangup_seen and not self.secure_abort
                   and time.monotonic() - t0 < self.s.secure_wait_max_s):
                await asyncio.sleep(0.05)
            if self.ended or self.hangup_seen:
                return
            if self.secure_abort or self._secure_send_blocked():
                # 다른 안내로 넘어갔거나, 입력 안내 되풀이로 보기에는 너무 길게 말한다(첫 메뉴로 돌아간 경우 등).
                # 입력값을 메뉴 키로 잘못 보내지 않도록 버리고, 입력 안내가 다시 나오면 새로 입력받는다
                log.info("ARS 가 다른 안내로 넘어가 입력값을 보내지 않았다 (%.1f초 기다림)", time.monotonic() - t0)
                self.secure_errors += 1
                self.publish("input_rejected", {
                    "keys": None, "actor": "user", "ars_prompt": "", "retry_count": self.secure_errors,
                    "message": "ARS가 다른 안내로 넘어가 입력하신 번호를 보내지 않았습니다. 입력 안내가 다시 나오면 다시 입력해 주세요.",
                })
                return
            log.info("ARS 안내가 끝나 입력값을 보낸다 (%.1f초 기다림)", time.monotonic() - t0)
            # 잠금 없이 보낸다: 안내 인식·AI 판단이 잠금을 잡고 있으면 ARS 의 입력 대기 시간이 지나 버린다
            self.last_send_secure = True       # 전송 중에 처리되는 같은 입력 안내는 무시된다 (stale)
            self.secure_sent_at = time.monotonic()
            await self._send_keys(keys, "user", secure=True)
            self.last_send_secure = True
        except ApiError:
            pass
        except Exception:
            log.exception("입력값 전송 실패")
        finally:
            del keys
            self.secure_queued = False

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
        self.goal_summary = await self._ai("summarize_goal", self.interp.summarize_goal(text),
                                           f"'{text}' 업무를 찾아 드릴게요.")
        self.summary = None
        self.result_heard = False
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
        # 상대가 끊은 뒤 마지막 안내를 처리하는 중(hangup_seen)인 세션은 회선을 쓰지 않으므로 새 통화를 막지 않는다
        return next((s for s in self.sessions.values() if not s.ended and not s.hangup_seen), None)

    def get(self, sid: str, token: str | None) -> Session:
        s = self.sessions.get(sid)
        if s is None:
            raise ApiError(404, "not_found", "세션이 없습니다.")
        if not token or not secrets.compare_digest(token, s.token):
            raise ApiError(401, "unauthorized", "세션 토큰이 올바르지 않습니다.")
        if s.ended and time.monotonic() - (s.ended_at or 0) > self.deps.settings.token_ttl_s:
            raise ApiError(401, "unauthorized", "만료된 세션 토큰입니다.")
        return s

    async def create(self, phone: str, goal: str | None, record: bool = False) -> dict:
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
        return await s.create(goal, record)
