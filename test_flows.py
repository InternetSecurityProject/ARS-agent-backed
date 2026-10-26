"""
가상 ARS 로 전체 흐름을 시험한다 (실통화 없음).

  python -m pytest -q test_flows.py
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app import build_deps, create_app
from config import Settings
from session import ApiError, SessionManager, mask_digits

SPEED = 0.01  # 안내 길이를 1/100 로 줄인다


def make_settings(**kw) -> Settings:
    st = Settings()
    st.backend, st.sim_speed, st.sim_audio = "sim", SPEED, False
    # .env 의 INTERPRETER/STT 설정과 무관하게 항상 모의 해석기로 시험한다 (실제 AI 호출 없음)
    st.interpreter, st.stt = "interpreter:MockInterpreter", "none"
    st.no_response_s, st.clarify_expire_s, st.clarify_drop_s, st.user_key_hold_s = 5, 0.3, 0.6, 0.2
    for k, v in kw.items():
        setattr(st, k, v)
    return st


async def new_manager(**kw) -> SessionManager:
    deps = build_deps(make_settings(**kw))
    deps.backend.tree["timeout_s"] = 300  # 가상 ARS 입력 대기 3초 (재질문 시간 기준보다 길게)
    await deps.backend.start()
    return SessionManager(deps)


async def wait_for(s, cond, timeout: float = 5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = cond(s)
        if r:
            return r
        await asyncio.sleep(0.01)
    raise AssertionError(f"시간 초과: status={s.status} events={[e['event'] for e in s.hub.history]}")


def ev(s, name):
    return [e["data"] for e in s.hub.history if e["event"] == name]


def status_seq(s):
    return [d["status"] for d in ev(s, "status_changed")]


def caption_has(word):
    return lambda s: any(word in c["text"] for c in s.captions)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
async def finish_after_result(s):
    """결과를 받으면 AI 만 끝나고(agent_done) 통화는 유지된다. 사용자가 통화를 종료한다."""
    await wait_for(s, lambda s: "agent_done" in status_seq(s))
    assert not s.agent_active
    if not s.ended:  # 가상 ARS 는 결과 안내 뒤 스스로 끊기도 한다 (그 경우도 completed)
        await s.action({"type": "cancel"})
    assert s.end_reason == "completed"


async def start(m, goal):
    out = await m.create("15889999", goal)
    return out, m.sessions[out["session_id"]]


def test_goal_required_at_start():
    """AI 도우미 사용과 목적은 통화 시작 시 정한다. 목적 없이 세션을 만들 수 없다."""
    async def go():
        m = await new_manager()
        for goal in (None, "", "   "):
            with pytest.raises(ApiError) as e:
                await m.create("15889999", goal)
            assert e.value.code == "invalid_request"
        assert m.active() is None
    run(go())


def test_goal_first_secure_input_completed():
    """시나리오 A: 목적과 함께 생성 → 바로 발신·탐색 → 민감정보 직접 입력 3회 → 결과(핵심 값) → 사용자 종료"""
    async def go():
        m = await new_manager()
        out, s = await start(m, "통장 잔액 알고 싶어")
        assert out["status"] == "dialing" and out["goal_summary"] == "계좌 잔액을 확인해 드릴게요."
        for expect in ("주민등록번호", "계좌번호", "비밀번호"):
            req = await wait_for(s, lambda s: s.pending if s.pending and s.pending["kind"] == "secure_input"
                                 and expect in s.pending["label"] else None)
            with pytest.raises(ApiError) as e:  # 민감정보 구간에서 key 입력 차단
                await s.action({"type": "key", "key": "1"})
            assert e.value.code == "sensitive_input_blocked"
            r = await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "1234"})
            assert r == {"accepted": True}
        await finish_after_result(s)
        sm = ev(s, "summary")[0]
        assert sm["kind"] == "result" and "1,234,000" in sm["summary"]
        assert sm["key_info"] == [{"label": "출금 가능 금액", "value": "1,234,000원"}]
        assert [p["selected"]["key"] for p in s.path] == ["3", "2"]
        assert all(p["actor"] == "ai" for p in s.path)
        # 신뢰도 출력: AI 가 고른 선택지의 신뢰도
        assert [p["selected"]["confidence"] for p in s.path] == [0.9, 0.8]
        assert s.path[0]["selected"]["category"] == "잔액 조회"
        # 경로 문장: 들은 선택지 중 고른 선택지 (명세 6.2)
        assert s.path[0]["text"] == "잔액조회·거래내역조회·카드 업무·상담직원 연결 중 잔액조회(3) 선택"
        assert s.path[1]["text"] == "알림톡으로 받기·ARS로 계속 중 ARS로 계속(2) 선택"
        # 입력값·민감정보 안내는 자막·경로에 남지 않는다
        assert not any("주민등록번호" in c["text"] for c in s.captions)
        assert "1234" not in str(s.hub.history)
        code, res = s.result()
        assert code == 200 and res["outcome"] == "success" and res["key_info"]
    run(go())


def test_low_confidence_then_answer():
    """신뢰도 0.30~0.50 → low_confidence 재질문 → 답변 반영 → 같은 메뉴에서 다시 판단"""
    async def go():
        m = await new_manager(clarify_expire_s=5, clarify_drop_s=10)
        _, s = await start(m, "카드 관련 문의")
        req = await wait_for(s, lambda s: s.pending)
        assert req["reason"] == "low_confidence"
        assert req["options"] == [{"key": "5", "label": "카드 업무", "category": "카드 업무", "confidence": 0.4}]
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "카드 결제일 바꾸고 싶어"})
        await finish_after_result(s)
        assert [(p["selected"]["key"], p["selected"]["confidence"]) for p in s.path] == [("5", 0.8), ("3", 0.95)]
    run(go())


def test_low_confidence_no_answer_goes_to_agent_menu():
    """재질문 무응답 60초(시험 0.6초) → 현재 메뉴의 상담원 연결 선택지로 진행"""
    async def go():
        m = await new_manager()
        _, s = await start(m, "카드 관련 문의")
        await wait_for(s, lambda s: s.pending)
        await wait_for(s, lambda s: s.status == "connected_to_agent", timeout=6)
        assert s.path[0]["selected"]["key"] == "0" and s.path[0]["actor"] == "ai"
        await s.action({"type": "cancel"})
        assert s.end_reason == "agent_call_ended"
    run(go())


def test_ambiguous_no_answer_keeps_waiting():
    """상담원 연결 선택지가 없는 메뉴에서 재질문 무응답 → 요청을 유지한 채 계속 듣는다"""
    async def go():
        m = await new_manager()
        _, s = await start(m, "결제 관련")
        req = await wait_for(s, lambda s: s.pending)
        assert req["reason"] == "ambiguous"
        await asyncio.sleep(0.9)  # clarify_drop(0.6초) 경과
        assert s.pending and s.pending["request_id"] == req["request_id"] and s.status == "awaiting_user"
        await s.action({"type": "cancel"})
        assert s.end_reason == "user_canceled"
    run(go())


def test_ambiguous_then_answer():
    async def go():
        m = await new_manager(clarify_expire_s=5, clarify_drop_s=10)
        _, s = await start(m, "결제 관련")
        req = await wait_for(s, lambda s: s.pending)
        assert req["kind"] == "clarification" and req["reason"] == "ambiguous"
        assert {(o["key"], o["confidence"]) for o in req["options"]} == {("2", 0.7), ("3", 0.7)}
        assert s.status == "awaiting_user"
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "결제일을 바꾸고 싶어"})
        await finish_after_result(s)
        assert [p["selected"]["key"] for p in s.path] == ["5", "3"]
    run(go())


def test_revise_goal_before_first_key():
    """다시 말하기는 목적 설정 단계(첫 번호 전)에만 받는다. 첫 메뉴 재질문 중 새 목적 → 재질문을 닫고 첫 메뉴에서 다시 판단"""
    async def go():
        m = await new_manager(clarify_expire_s=5, clarify_drop_s=10)
        _, s = await start(m, "카드 관련 문의")
        req = await wait_for(s, lambda s: s.pending)     # 첫 메뉴에서 low_confidence, 아직 누른 번호 없음
        assert req["reason"] == "low_confidence" and s.path == []
        r = await s.action({"type": "revise_goal", "text": "이번 달 카드 대금 알고 싶어"})
        assert r["goal_summary"].startswith("이번 달")
        await wait_for(s, lambda s: "agent_done" in status_seq(s))
        assert [p["selected"]["key"] for p in s.path] == ["5", "2"]
        if not s.ended:  # 첫 번호를 누른 뒤에는 409
            with pytest.raises(ApiError) as e:
                await s.action({"type": "revise_goal", "text": "카드 결제일 바꾸고 싶어"})
            assert e.value.code == "invalid_state"
            await s.action({"type": "cancel"})
        assert s.end_reason == "completed"
    run(go())


def test_backtrack_to_previous_layer():
    """AI 가 잔액조회로 들어감 → 하위 메뉴에 맞는 선택지 없음 → 이전 계층 확인 → 별표로 복귀 → 카드 업무 → 결제일 변경"""
    async def go():
        m = await new_manager()
        _, s = await start(m, "결제일 확인하고 바꾸고 싶어")
        await finish_after_result(s)
        assert [(p["selected"]["key"], p["actor"]) for p in s.path] == [("3", "ai"), ("*", "ai"), ("5", "ai"), ("3", "ai")]
        assert any("이전 메뉴를 확인" in d["message"] for d in ev(s, "status_changed"))
        back = s.path[1]["selected"]
        assert s.path[1]["text"] == "이전 메뉴(*) 선택" and back["confidence"] is None and back["category"] is None
    run(go())


def test_not_found_at_first_layer():
    async def go():
        m = await new_manager()
        _, s = await start(m, "외화 환전하고 싶어")
        await wait_for(s, lambda s: s.ended)
        assert s.end_reason == "not_found" and s.path == []
    run(go())


def test_not_found_after_checking_previous_layer():
    async def go():
        m = await new_manager()
        _, s = await start(m, "외화 잔액 환전하고 싶어")
        await wait_for(s, lambda s: s.ended)
        assert s.end_reason == "not_found"
        assert [p["selected"]["key"] for p in s.path] == ["3"]  # 돌아가지 않고 종료
    run(go())


def test_agent_connection_handoff():
    async def go():
        m = await new_manager()
        _, s = await start(m, "상담원 연결해줘")
        await wait_for(s, lambda s: s.status == "connected_to_agent")
        sm = ev(s, "summary")[0]
        assert sm["kind"] == "handoff" and "상담원 연결해줘" in sm["handoff_script"]
        assert "waiting_agent" in status_seq(s)
        await s.action({"type": "cancel"})
        assert s.end_reason == "agent_call_ended"
    run(go())


def test_user_direct_key_and_input_rejected():
    """재질문 대기 중 사용자가 직접 없는 번호를 누름 → 재질문 닫힘 → ARS 입력 오류 → input_rejected(actor=user)"""
    async def go():
        m = await new_manager(clarify_expire_s=5, clarify_drop_s=10)
        _, s = await start(m, "결제 관련")
        await wait_for(s, lambda s: s.pending)
        await s.action({"type": "key", "key": "9"})
        assert s.pending is None
        rej = await wait_for(s, lambda s: ev(s, "input_rejected"))
        assert rej[0]["actor"] == "user" and rej[0]["keys"] == "9"
        assert s.path[-1]["actor"] == "user" and s.path[-1]["selected"]["confidence"] is None
        assert s.path[-1]["text"] == "9 입력"
        await s.action({"type": "cancel"})
    run(go())


def test_one_call_at_a_time():
    async def go():
        m = await new_manager()
        await m.create("15889999", "결제 관련")
        with pytest.raises(ApiError) as e:
            await m.create("15771000", "결제 관련")
        assert e.value.code == "invalid_state"
    run(go())


def test_mask_digits():
    t, hit = mask_digits("입력하신 주민등록번호는 950101 입니다. 맞으면 1번")
    assert hit and "950101" not in t and "1번" in t
    assert mask_digits("잔액조회는 3번")[1] is False


# ---------------------------------------------------------------------------
def test_rest_and_websocket():
    from fastapi.testclient import TestClient

    st = make_settings(sim_speed=0.05)
    with TestClient(create_app(st)) as c:
        r = c.post("/v1/sessions", json={"phone_number": "1588-9999"}, headers={"X-App-Key": "bad"})
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
        r = c.post("/v1/sessions", json={"phone_number": "1588-9999"}, headers={"X-App-Key": st.app_key})
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"   # 목적 필수
        r = c.post("/v1/sessions", json={"phone_number": "1588-9999", "goal": "카드 결제일 바꾸고 싶어"},
                   headers={"X-App-Key": st.app_key})
        assert r.status_code == 201, r.text
        out = r.json()
        assert out["status"] == "dialing" and out["goal_summary"]
        sid, tok = out["session_id"], out["session_token"]
        h = {"Authorization": f"Bearer {tok}"}
        with c.websocket_connect(f"/v1/sessions/{sid}/events?token={tok}") as ws:
            snap = ws.receive_json()
            assert snap["event"] == "snapshot" and snap["at"]
            assert c.get(f"/v1/sessions/{sid}/result", headers=h).status_code == 202
            names, status = [], snap["data"]["status"]
            while status != "agent_done":
                msg = ws.receive_json()
                names.append(msg["event"])
                if msg["event"] == "status_changed":
                    status = msg["data"]["status"]
            r = c.post(f"/v1/sessions/{sid}/actions", json={"type": "cancel"}, headers=h)
            assert r.status_code == 200
            while True:
                msg = ws.receive_json()
                names.append(msg["event"])
                if msg["event"] == "ended":
                    break
        assert "summary" in names and names[-2:] == ["status_changed", "ended"]
        r = c.get(f"/v1/sessions/{sid}/result", headers=h)
        body = r.json()
        assert r.status_code == 200 and body["outcome"] == "success" and body["end_reason"] == "completed"
        assert [p["selected"]["key"] for p in body["path"]] == ["5", "3"] and body["key_info"]
        r = c.post(f"/v1/sessions/{sid}/actions", json={"type": "cancel"}, headers=h)
        assert r.status_code == 409 and r.json()["error"]["code"] == "invalid_state"
        r = c.post(f"/v1/sessions/{sid}/actions", json={"type": "revise_goal", "text": "x"}, headers=h)
        assert r.status_code == 409   # 종료된 통화
        r = c.get(f"/v1/sessions/{sid}/result", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        # 종료된 세션에 이벤트 채널로 다시 붙으면 snapshot(ended) 만 받고 서버가 닫는다 (영원히 매달리지 않는다)
        with c.websocket_connect(f"/v1/sessions/{sid}/events?token={tok}") as ws:
            snap = ws.receive_json()
            assert snap["event"] == "snapshot" and snap["data"]["status"] == "ended"
            with pytest.raises(Exception):
                ws.receive_json()
        with c.websocket_connect(f"/v1/sessions/{sid}/audio?token={tok}") as ws:
            with pytest.raises(Exception):
                ws.receive_bytes()


# ---------------------------------------------------------------------------
# ClawOps 백엔드 연결부: SDK 를 흉내 낸 가짜 에이전트로 발신 → 오디오 수신 → 발화 구간 → STT → 키 전송 → 종료
# ---------------------------------------------------------------------------
def test_clawops_backend_wiring(monkeypatch):
    import clawops.agent as ca

    import telephony
    from audio import FRAME, SILENCE_FRAME, hum_ulaw, Segmenter
    from session import Deps
    from interpreter import MockInterpreter
    from navigator import Navigator

    class FakeCall:
        def __init__(self, cid):
            self.call_id, self.ended_status, self.sent = cid, None, bytearray()

        async def send_audio(self, b):
            self.sent += b

    class FakeSession:
        def __init__(self, agent, cid):
            self.agent, self.call_id, self.ended_status = agent, cid, None

        async def hangup(self):
            self.ended_status = "completed"
            for h in self.agent.h["call_end"]:
                await h(self.agent.calls[self.call_id])

    class FakeAgent:
        last = None

        def __init__(self, from_, session_factory, **kw):
            self.factory, self.h, self.calls = session_factory, {}, {}
            FakeAgent.last = self

        def on(self, ev):
            def deco(fn):
                self.h.setdefault(ev, []).append(fn)
                return fn
            return deco

        async def connect(self):
            pass

        async def disconnect(self):
            pass

        async def call(self, to, timeout=40):
            return FakeSession(self, "c1")

        async def answer(self):  # SDK 동작 흉내: call_start → 세션 시작
            call = FakeCall("c1")
            self.calls["c1"] = call
            for h in self.h["call_start"]:
                await h(call)
            self.leg = self.factory()
            await self.leg.start(call)
            return call

    monkeypatch.setattr(ca, "ClawOpsAgent", FakeAgent)

    class ScriptSTT:
        is_local = True
        lines = ["KB국민은행입니다. 잔액조회는 3번, 거래내역조회는 4번, 카드 업무는 5번, 상담직원 연결은 0번을 눌러 주십시오. 다시 들으시려면 우물정자를 눌러 주십시오.",
                 "고객님의 출금 가능 금액은 1,234,000원입니다. 감사합니다."]

        async def recognize(self, seg):
            return self.lines.pop(0) if self.lines else ""

    async def go():
        st = make_settings(backend="clawops", clawops_from="070-0000-0000")
        backend = telephony.ClawOpsBackend(st.clawops_from)
        interp = MockInterpreter()
        deps = Deps(st, backend, ScriptSTT(), interp, Navigator(interp))
        await backend.start()
        m = SessionManager(deps)
        out = await m.create("15889999", "통장 잔액 알고 싶어")
        s = m.sessions[out["session_id"]]
        call = await FakeAgent.last.answer()
        await wait_for(s, lambda s: s.status == "listening")
        leg = FakeAgent.last.leg

        async def speak(sec):
            for _ in range(int(sec * 50)):
                await leg.feed_audio(hum_ulaw() if True else b"", 0)
            for _ in range(70):  # 1.4초 무음 → 단락 종료
                await leg.feed_audio(SILENCE_FRAME, 0)
            await asyncio.sleep(0.05)

        # 허밍 소리는 RMS 가 낮으므로 임계값을 낮춘다
        s.segmenter = Segmenter(threshold=50, silence_ms=1200)
        await speak(1.0)
        await wait_for(s, lambda s: s.path)
        assert s.path[0]["selected"]["key"] == "3" and len(call.sent) > 0  # DTMF 톤 송출
        await asyncio.sleep(0.1)
        s.segmenter._mute_frames = 0
        await speak(1.0)
        await finish_after_result(s)
    run(go())


# ---------------------------------------------------------------------------
# 수동 ARS: 안내 문장을 시험 코드가 직접 넣어 단계별 동작을 확인한다 (가상 ARS 의 시간 흐름 없이)
# ---------------------------------------------------------------------------
ROOT = ("KB국민은행입니다. 잔액조회는 3번, 거래내역조회는 4번, 카드 업무는 5번, 상담직원 연결은 0번을 눌러 주십시오. "
        "다시 들으시려면 우물정자를 눌러 주십시오.")
CARD = "카드 분실 신고는 1번, 결제 대금 조회는 2번, 결제일 변경은 3번을 눌러 주십시오. 이전 메뉴는 별표를 눌러 주십시오."
BILL = "이번 달 결제 예정 금액은 352,000원입니다. 감사합니다."
ERR = "잘못 누르셨습니다."
CONFIRM = "결제일을 25일로 변경하시겠습니까? 변경하시려면 1번, 취소는 2번을 눌러 주십시오."
BAL = "알림톡으로 받으시려면 1번, ARS로 계속하시려면 2번을 눌러 주십시오. 이전 메뉴는 별표를 눌러 주십시오."


class ManualCall:
    """transcripts=True 이면 시험 코드가 안내 문장을 직접 넣고(on_paragraph),
    False 이면 실통화처럼 음성 프레임을 넣어 Segmenter → STT 경로로 처리한다(speak)."""

    def __init__(self, transcripts: bool = True) -> None:
        self.keys: list[str] = []
        self.provides_transcripts = transcripts

    async def send_keys(self, keys: str) -> float:
        self.keys.append(keys)
        return 0.0

    async def send_audio(self, b: bytes) -> None:
        pass

    async def hangup(self) -> None:
        pass


class ManualBackend:
    def __init__(self, transcripts: bool = True) -> None:
        self.call: ManualCall | None = None
        self.transcripts = transcripts

    async def start(self) -> None:
        pass

    async def dial(self, to, handlers) -> ManualCall:
        self.call = ManualCall(self.transcripts)
        return self.call


class ScriptSTT:
    """시험용 STT: 단락이 끝날 때마다 정해 둔 문장을 차례로 돌려준다."""

    is_local = True

    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)

    async def recognize(self, seg: bytes) -> str:
        return self.lines.pop(0) if self.lines else ""


async def speak(s, sec: float, silence: float = 0.3) -> None:
    """ARS 음성 흉내: sec 초 동안 소리 프레임, 이어서 silence 초 동안 무음 프레임을 넣는다 (20ms 단위)."""
    from audio import SILENCE_FRAME, hum_ulaw

    for _ in range(int(sec * 50)):
        await s._on_audio(hum_ulaw())
    for _ in range(int(silence * 50)):
        await s._on_audio(SILENCE_FRAME)
    await asyncio.sleep(0.02)


async def manual(goal: str, stt: ScriptSTT | None = None, **kw):
    """stt 를 주면 음성 경로(Segmenter → STT)로 시험한다."""
    from interpreter import MockInterpreter, _norm
    from navigator import Navigator
    from session import Deps
    from stt import NullRecognizer

    st = make_settings(**kw)
    interp = MockInterpreter()
    interp.prompts[_norm(CONFIRM)] = (CONFIRM, {
        "kind": "confirm", "critical": True,
        "options": [{"key": "1", "label": "변경 확정"}, {"key": "2", "label": "취소"}]})
    deps = Deps(st, ManualBackend(transcripts=stt is None), stt or NullRecognizer(), interp,
                Navigator(interp, st.match_threshold, st.margin, st.max_revisit, st.max_presses, st.low_threshold))
    m = SessionManager(deps)
    out = await m.create("15889999", goal)
    s = m.sessions[out["session_id"]]
    await s._on_answer()
    return s, deps.backend.call


def test_result_during_user_key_hold_is_not_lost():
    """직접 선택 뒤 10초 보류 중이어도 결과 안내는 처리한다 (보류는 메뉴 판단만 미룬다)"""
    async def go():
        s, call = await manual("결제 관련", user_key_hold_s=5, clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        assert s.pending and s.pending["reason"] == "ambiguous"
        await s.action({"type": "key", "key": "2"})
        assert s.pending is None and s.status == "listening"   # 재질문이 닫히면 awaiting_user 도 해제
        await s.on_paragraph(BILL)                               # 보류 시간 안에 결과 안내
        assert s.status == "agent_done" and s.summary["kind"] == "result"
        await s.action({"type": "cancel"})
        assert s.end_reason == "completed"
    run(go())


def test_select_releases_awaiting_user():
    async def go():
        s, call = await manual("결제 관련", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        req = s.pending
        await s.action({"type": "select", "request_id": req["request_id"], "key": "3"})
        assert call.keys == ["5", "3"] and s.status == "listening"
        assert status_seq(s)[-1] == "listening"
    run(go())


def test_critical_confirm_is_never_pressed_by_ai():
    """결제·계약·해지 최종 승인 확인 질문은 critical_action 재질문, select 로만 진행"""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        await s.on_paragraph(CONFIRM)
        req = s.pending
        assert req["reason"] == "critical_action" and call.keys == ["5", "3"]
        with pytest.raises(ApiError) as e:
            await s.action({"type": "answer", "request_id": req["request_id"], "text": "네"})
        assert e.value.code == "invalid_state"
        await s.action({"type": "select", "request_id": req["request_id"], "key": "1"})
        assert call.keys[-1] == "1" and s.path[-1]["actor"] == "user"
    run(go())


def test_input_rejected_needs_two_consecutive_errors():
    """입력 오류 → 재시도 성공 → 다른 단계에서 입력 오류 1회는 재질문하지 않는다. 같은 단계 2회 연속이면 재질문"""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)      # 5
        await s.on_paragraph(ERR)       # 1회
        await s.on_paragraph(ROOT)      # 같은 메뉴 재생 → 5 재전송
        await s.on_paragraph(CARD)      # 받아들여짐 → 3
        await s.on_paragraph(ERR)       # 다른 단계 1회
        assert s.pending is None
        await s.on_paragraph(CARD)      # 3 재전송
        await s.on_paragraph(ERR)       # 같은 단계 2회 연속
        assert s.pending and s.pending["reason"] == "input_rejected"
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1, 1, 2]
        assert call.keys == ["5", "5", "3", "3"]
    run(go())


def test_loop_after_two_revisits():
    """키 전송 결과로 지나온 메뉴에 다시 도착하는 상황이 2회 발생하면 loop 재질문"""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)      # 첫 도착 → 5
        await s.on_paragraph(ROOT)      # 다시 도착 1회 → 아직 반복 아님 → 5
        assert s.pending is None and call.keys == ["5", "5"]
        await s.on_paragraph(ROOT)      # 다시 도착 2회 → loop
        assert s.pending and s.pending["reason"] == "loop"
    run(go())


def test_backtrack_exclusion_applies_only_to_that_layer():
    from interpreter import MockInterpreter
    from navigator import Layer, NavState, Navigator

    async def go():
        interp = MockInterpreter()
        nav = Navigator(interp)
        root, card = await interp.classify(ROOT), await interp.classify(CARD)
        st = NavState("카드 결제일 바꾸고 싶어", exclude={"3"}, exclude_at=root.signature())
        dec = await nav.decide(st, card, [Layer(root, "5", "ai")])
        assert dec.action == "press" and dec.key == "3"   # 다른 메뉴의 3번은 제외하지 않는다
    run(go())


def test_selecting_from_decision_start():
    """메뉴를 판단하기 시작하면 selecting. 키를 누르면 listening, 판단을 보류하면 awaiting_user (명세 3.1)"""
    async def go():
        s, call = await manual("결제 관련", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        assert status_seq(s)[-3:] == ["selecting", "selecting", "listening"]   # 판단 시작 → '카드 업무' 선택 → 키 전송
        await s.on_paragraph(CARD)
        assert status_seq(s)[-2:] == ["selecting", "awaiting_user"] and s.pending["reason"] == "ambiguous"
    run(go())


def test_no_response_replay_is_not_counted_as_revisit():
    """키 전송 뒤 무응답 → 다시 듣기 키 → 다시 나온 메뉴는 반복으로 세지 않는다 (명세 7장 5항)"""
    async def go():
        s, call = await manual("결제 관련", no_response_s=0.1, clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)                                   # 5
        for n in (2, 4):                                             # 무응답 → '#' → 같은 메뉴 다시 수신 (2회)
            await wait_for(s, lambda s: len(call.keys) == n, timeout=2)
            assert call.keys[-1] == "#"
            await s.on_paragraph(ROOT)
        assert s.pending is None and call.keys == ["5", "#", "5", "#", "5"]
    run(go())


def test_answer_to_voice_request_waits_for_next_menu():
    """음성 응답 요구 재질문에 answer 하면 지난 메뉴로 판단하지 않고 다음 안내를 기다린다"""
    async def go():
        from interpreter import Prompt
        s, call = await manual("결제 관련", clarify_expire_s=5, clarify_drop_s=10)
        orig = s.d.interp.classify

        async def classify(text):
            return Prompt(kind="voice_request", text=text) if text.startswith("VOICE") else await orig(text)

        s.d.interp.classify = classify
        await s.on_paragraph(ROOT)                                   # 5
        await s.on_paragraph("VOICE 원하시는 업무를 말씀해 주세요.")
        req = s.pending
        assert req["reason"] == "unknown_prompt" and req["options"] == []
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "이번 달 카드 대금 알고 싶어"})
        assert call.keys == ["5"] and not s.ended and s.status == "listening"
        await s.on_paragraph(CARD)                                   # 다음 메뉴에서 새 목적으로 판단
        assert call.keys == ["5", "2"]
    run(go())



# ---------------------------------------------------------------------------
# 음성 경로 시간 처리 (음성 가상 ARS 시험에서 발견된 문제)
# ---------------------------------------------------------------------------
def test_no_repeat_key_while_next_menu_is_long():
    """다음 메뉴 안내가 무응답 시간보다 길어도 ARS 가 말하는 동안에는 다시 듣기 키를 누르지 않는다.
    누르면 ARS 가 '잘못 누르셨습니다' 후 메뉴를 다시 읽고 두 안내가 한 자막으로 섞인다."""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=ScriptSTT([ROOT, BAL]), no_response_s=0.2,
                               paragraph_silence_ms=200, clarify_expire_s=5, clarify_drop_s=10)
        await speak(s, 0.3)                                          # 첫 메뉴 → 잔액조회(3)
        await wait_for(s, lambda s: call.keys == ["3"])
        await speak(s, 0.6, silence=0)                               # 다음 메뉴를 말하는 중
        await asyncio.sleep(0.4)                                     # 무응답 시간(0.2초)이 지나도 계속 말하는 중
        assert call.keys == ["3"]
        await speak(s, 0)                                            # 안내 끝 → 인식 → ARS로 계속(2)
        await wait_for(s, lambda s: call.keys == ["3", "2"])
        assert [c["text"] for c in s.captions] == [ROOT, BAL]
    run(go())


def test_no_previous_menu_repeat_key_after_ars_spoke():
    """키 전송 뒤 ARS 가 소리를 냈으면(인식된 문장이 없어도) 다음 안내로 넘어간 것이다.
    이전 메뉴의 다시 듣기 키(#)는 다음 메뉴에서 다른 뜻일 수 있으므로 누르지 않는다."""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=ScriptSTT([ROOT, ""]), no_response_s=0.2,
                               paragraph_silence_ms=200, clarify_expire_s=5, clarify_drop_s=10)
        await speak(s, 0.3)
        await wait_for(s, lambda s: call.keys == ["3"])
        await speak(s, 0.2)                                          # 인식되지 않은 안내
        await asyncio.sleep(0.4)
        assert call.keys == ["3"]
    run(go())


def test_no_response_after_silence_presses_that_menus_repeat_key():
    """키 전송 뒤 ARS 가 아무 소리도 내지 않으면(키가 인식되지 않음) 키를 누른 그 메뉴의 다시 듣기 키를 누른다"""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=ScriptSTT([ROOT]), no_response_s=0.2,
                               paragraph_silence_ms=200, clarify_expire_s=5, clarify_drop_s=10)
        await speak(s, 0.3)
        await wait_for(s, lambda s: call.keys == ["3"])
        await speak(s, 0, silence=0.4)                               # 회선 무음만 들어옴
        await wait_for(s, lambda s: call.keys == ["3", "#"], timeout=2)
    run(go())


def test_clarify_repeat_key_waits_until_ars_is_quiet():
    """재질문 30초(시험 0.2초) 다시 듣기 키는 ARS 가 안내하는 중이면 안내가 끝난 뒤 누른다"""
    async def go():
        s, call = await manual("카드 관련 문의", clarify_expire_s=0.2, clarify_drop_s=5)
        await s.on_paragraph(ROOT)                                   # low_confidence 재질문 (누른 키 없음)
        assert s.pending and s.pending["reason"] == "low_confidence"
        call.speaking = True                                         # ARS 가 입력 대기 시간이 지나 메뉴를 다시 읽는 중
        await asyncio.sleep(0.4)
        assert call.keys == []
        call.speaking = False
        await wait_for(s, lambda s: call.keys == ["#"], timeout=2)
        assert s.pending is not None                                 # 요청은 유지
    run(go())


def test_hangup_during_result_summary_is_completed():
    """결과 안내 직후 ARS 가 끊어도(요약 생성 중) completed 이며, 요약을 마친 뒤 summary → ended 순서로 전송한다.
    종료 뒤 늦게 끝난 처리가 상태를 되돌리지 않는다 (새 통화가 409 로 막히지 않는다)."""
    async def go():
        s, call = await manual("이번 달 카드 대금 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        orig = s.d.interp.summarize_result

        async def slow(goal, caps):
            await asyncio.sleep(0.3)                                 # LLM 요약 시간
            return await orig(goal, caps)

        s.d.interp.summarize_result = slow
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        t = asyncio.create_task(s.on_paragraph(BILL))
        await wait_for(s, lambda s: s.status == "listening_result")
        await s._on_end("remote_hangup")                             # 요약 중에 ARS 가 끊음
        await t
        await wait_for(s, lambda s: s.ended)
        await asyncio.sleep(0.1)
        names = [e["event"] for e in s.hub.history]
        assert s.end_reason == "completed" and s.status == "ended" and names[-1] == "ended"
        assert names.index("summary") < names.index("ended")
        code, res = s.result()
        assert res["outcome"] == "success" and "352,000" in res["summary"]
    run(go())


def test_hangup_flush_timeout_sends_caption_summary_before_ended():
    """결과 안내 뒤 끊겼는데 AI 요약이 제한 시간(hangup_flush_s) 안에 끝나지 않으면, 요약 없이 끝내지 않고
    결과 안내 원문으로 summary 를 보낸 뒤 ended 를 보낸다. 늦게 끝난 AI 요약은 버린다 (명세 7장 7항)."""
    async def go():
        s, call = await manual("이번 달 카드 대금 알고 싶어", hangup_flush_s=0.3, clarify_expire_s=5, clarify_drop_s=10)

        async def very_slow(goal, caps):
            await asyncio.sleep(1.0)                                 # 제한 시간보다 긴 AI 응답
            return {"summary": "늦은 AI 요약", "key_info": []}

        s.d.interp.summarize_result = very_slow
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        t = asyncio.create_task(s.on_paragraph(BILL))
        await wait_for(s, lambda s: s.status == "listening_result")
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended, timeout=2)
        names = [e["event"] for e in s.hub.history]
        assert s.end_reason == "completed" and names[-1] == "ended" and names.count("summary") == 1
        assert s.summary["summary"] == BILL                          # 결과 안내 원문
        await t
        await asyncio.sleep(0.05)
        assert s.summary["summary"] == BILL and [e["event"] for e in s.hub.history].count("summary") == 1
    run(go())


def test_agent_connected_status_comes_before_handoff_summary():
    """상담원 음성을 감지하면 connected_to_agent 를 먼저 보내고(마이크 중계 시작), 상담사 전달 문장(summary)은
    AI 응답을 기다린 뒤 보낸다. 표시가 AI 응답 시간만큼 늦지 않는다."""
    async def go():
        s, call = await manual("상담원 연결해줘", clarify_expire_s=5, clarify_drop_s=10)
        orig = s.d.interp.handoff_script

        async def slow(goal, texts):
            await asyncio.sleep(0.3)
            return await orig(goal, texts)

        s.d.interp.handoff_script = slow
        await s.on_paragraph(ROOT)
        await s.on_paragraph(AGENT_WAIT)
        t = asyncio.create_task(s.on_paragraph(AGENT_HUMAN))
        await wait_for(s, lambda s: s.status == "connected_to_agent", timeout=0.2)
        assert s.summary is None                                     # 연결 표시가 먼저
        await t
        names = [e["event"] for e in s.hub.history]
        statuses = [e["data"]["status"] for e in s.hub.history if e["event"] == "status_changed"]
        assert names.index("summary") > names.index("status_changed") and statuses[-1] == "connected_to_agent"
        assert s.summary["kind"] == "handoff" and s.summary["handoff_script"]
    run(go())


def test_hangup_right_after_result_speech_is_recognized():
    """결과를 말하고 바로 끊어 단락 끝 무음이 오지 않아도, 끊기 전까지 들은 안내를 인식해 completed 로 끝낸다"""
    async def go():
        s, call = await manual("이번 달 카드 대금 알고 싶어", stt=ScriptSTT([ROOT, CARD, BILL]),
                               paragraph_silence_ms=200, clarify_expire_s=5, clarify_drop_s=10)
        await speak(s, 0.3)
        await wait_for(s, lambda s: call.keys == ["5"])
        await speak(s, 0.3)
        await wait_for(s, lambda s: call.keys == ["5", "2"])
        await speak(s, 0.3, silence=0)                               # 결과 안내 직후 끊김
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended)
        assert s.end_reason == "completed" and "352,000" in s.summary["summary"]
        assert s.captions[-1]["text"] == BILL
    run(go())


# ---------------------------------------------------------------------------
# 사용자 입력 엣지 케이스 (ARS자동화-엣지케이스-점검표.md 참조)
#   입력 종류: DTMF 키(key), 통화 종료(cancel), 요청 응답(answer·select·secure_input·revise_goal)
# ---------------------------------------------------------------------------
AGENT_WAIT = "상담원과 연결 중입니다. 잠시만 기다려 주십시오."
AGENT_HUMAN = "상담원 김민지입니다. 무엇을 도와드릴까요?"


def test_no_auto_repeat_after_agent_done():
    """업무 완료(agent_done) 뒤 사용자가 직접 누른 키에는 무응답 재전송(다시 듣기 키)을 걸지 않는다.
    AI 탐색이 끝났으므로 서버가 임의로 키를 더 보내면 안 된다."""
    async def go():
        s, call = await manual("이번 달 카드 대금 알고 싶어", no_response_s=0.1, clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        await s.on_paragraph(BILL)
        assert s.status == "agent_done"
        await s.action({"type": "key", "key": "9"})       # 완료 뒤 직접 키
        await asyncio.sleep(0.3)                            # 무응답 시간이 지나도
        assert call.keys == ["5", "2", "9"]                 # 다시 듣기 키(#)가 더해지지 않는다
    run(go())


def test_no_auto_repeat_when_connected_to_agent():
    """상담원 연결 중 사용자가 직접 누른 키에는 무응답 재전송을 걸지 않는다 (상담원 통화에 DTMF 가 끼어들지 않게)."""
    async def go():
        s, call = await manual("상담원 연결해줘", no_response_s=0.1, clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)                          # 상담직원 연결(0)
        await s.on_paragraph(AGENT_WAIT)
        await s.on_paragraph(AGENT_HUMAN)
        assert s.status == "connected_to_agent"
        await s.action({"type": "key", "key": "1"})         # 상담원 안내에 따라 직접 누름
        await asyncio.sleep(0.3)
        assert call.keys == ["0", "1"]                       # 자동 재전송 없음
    run(go())


def test_cancel_reason_by_state():
    """cancel 의 end_reason 은 상태에 따라 정해진다: 탐색 중 user_canceled, 결과 뒤 completed."""
    async def go():
        s, _ = await manual("이번 달 카드 대금 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        await s.action({"type": "cancel"})                  # 탐색 도중 종료
        assert s.end_reason == "user_canceled"

        s2, _ = await manual("이번 달 카드 대금 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s2.on_paragraph(ROOT)
        await s2.on_paragraph(CARD)
        await s2.on_paragraph(BILL)
        assert s2.status == "agent_done"
        await s2.action({"type": "cancel"})                 # 결과 확인 뒤 종료
        assert s2.end_reason == "completed"
    run(go())


def test_select_rejects_non_candidate_and_answer_rejects_stale():
    """select 는 요청 후보에 없는 키를 400, answer 는 이미 닫힌(만료된) 요청을 409 로 거절한다."""
    async def go():
        s, _ = await manual("결제 관련", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        req = s.pending
        assert req and req["reason"] == "ambiguous"
        with pytest.raises(ApiError) as e:
            await s.action({"type": "select", "request_id": req["request_id"], "key": "7"})   # 후보 아님
        assert e.value.code == "invalid_request"
        await s.action({"type": "select", "request_id": req["request_id"], "key": "2"})        # 정상 응답으로 닫음
        with pytest.raises(ApiError) as e:                                                     # 닫힌 요청에 다시 답
            await s.action({"type": "answer", "request_id": req["request_id"], "text": "결제 대금"})
        assert e.value.code == "invalid_state"
    run(go())


def test_key_rejected_during_secure_input_request():
    """민감정보 입력 요청이 열려 있으면 일반 key 입력은 409 sensitive_input_blocked (경로에 숫자가 남지 않게)."""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(BAL)
        await s.on_paragraph("주민등록번호 앞 여섯 자리와 우물정자를 눌러 주십시오.")
        assert s.pending and s.pending["kind"] == "secure_input"
        with pytest.raises(ApiError) as e:
            await s.action({"type": "key", "key": "1"})
        assert e.value.code == "sensitive_input_blocked"
        assert "1" not in [p["selected"]["key"] for p in s.path]
    run(go())


RRN = "주민등록번호 앞 여섯 자리와 우물정자를 눌러 주십시오."
TIMEOUT_MSG = "입력 시간이 초과되었습니다."
GIVEUP_MSG = "입력이 없어 통화를 종료합니다."


async def secure_pending(**kw):
    """잔액조회 → ARS 계속 → 주민등록번호 입력 요청까지 진행한 세션"""
    s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10, **kw)
    await s.on_paragraph(ROOT)
    await s.on_paragraph(BAL)
    await s.on_paragraph(RRN)
    assert s.pending and s.pending["kind"] == "secure_input" and call.keys == ["3", "2"]
    return s, call


def test_timeout_notice_while_secure_pending_is_caption_only():
    """민감정보 입력 대기 중(아직 아무 키도 보내지 않음) ARS 의 '입력 시간이 초과되었습니다'는 키 오류가 아니다.
    input_rejected 를 보내지 않고 자막만 남기며, 입력 요청은 그대로 유지한다."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.on_paragraph(TIMEOUT_MSG)
        await s.on_paragraph(RRN)                                    # ARS 가 입력 안내를 다시 읽음
        assert ev(s, "input_rejected") == [] and s.pending == req   # 같은 요청 유지, 새 요청 없음
        assert s.captions[-1]["text"] == TIMEOUT_MSG and call.keys == ["3", "2"]
    run(go())


def test_timeout_notice_while_clarification_pending_is_caption_only():
    """재질문 대기 중 ARS 가 '입력 시간이 초과되었습니다' 하고 메뉴를 다시 읽어도,
    키 오류로 세지 않고 메뉴를 다시 판단하지도 않는다 (재질문은 그대로 사용자 응답을 기다린다)."""
    async def go():
        s, call = await manual("카드 관련 문의", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)                                   # low_confidence 재질문
        req = s.pending
        assert req and req["reason"] == "low_confidence" and call.keys == []
        await s.on_paragraph(TIMEOUT_MSG)
        await s.on_paragraph(ROOT)
        assert ev(s, "input_rejected") == [] and s.pending == req and call.keys == []
        assert s.input_errors == 0
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "카드 결제일 바꾸고 싶어"})
        assert call.keys == ["5"]                                    # 답변 뒤 다시 읽어 둔 메뉴로 판단
    run(go())


def test_ars_gives_up_and_hangs_up_while_waiting_for_user():
    """사용자 응답 대기 중 ARS 가 '입력이 없어 통화를 종료합니다' 하고 끊으면 user_input_timeout (재질문·민감정보 모두)."""
    async def go():
        for build in (secure_pending, None):
            if build:
                s, call = await build()
            else:
                s, call = await manual("카드 관련 문의", clarify_expire_s=5, clarify_drop_s=10)
                await s.on_paragraph(ROOT)
            await s.on_paragraph(GIVEUP_MSG)
            await s._on_end("remote_hangup")
            await wait_for(s, lambda s: s.ended)
            assert s.end_reason == "user_input_timeout" and s.captions[-1]["text"] == GIVEUP_MSG
            assert s.result()[1]["outcome"] == "failed"
    run(go())


def test_secure_input_error_is_not_a_menu_error_and_digits_stay_out():
    """민감정보 입력값이 거부되면 input_rejected 는 keys=null 로 보내고, 메뉴 키 오류로 세지 않는다 (재질문 없음).
    ARS 가 입력 안내를 다시 하면 새 secure_input 요청이 나간다 (6.3 처리 규칙 6)."""
    async def go():
        s, call = await secure_pending()
        for n in (1, 2):
            req = s.pending
            await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "9999"})
            await s.on_paragraph(ERR)                                # 입력값 거부
            rej = ev(s, "input_rejected")[-1]
            assert rej["keys"] is None and rej["actor"] == "user" and rej["retry_count"] == n
            assert s.pending is None                                 # 메뉴 재질문(input_rejected)이 열리지 않는다
            await s.on_paragraph(RRN)                                # 입력 안내 반복 → 새 요청
            assert s.pending and s.pending["kind"] == "secure_input" and s.pending["request_id"] != req["request_id"]
        assert "9999" not in str(s.hub.history) and "9999" not in str(s.path)
        assert s.input_errors == 0
    run(go())


def test_secure_input_terminator_is_not_duplicated():
    """사용자가 종료 키(#)까지 눌러 보낸 경우 서버가 #을 한 번 더 붙이지 않는다 (두 번째 #은 ARS 에서 다른 뜻일 수 있다)."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "123456#"})
        assert call.keys[-1] == "123456#"
        await s.on_paragraph("계좌번호와 우물정자를 눌러 주십시오.")
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "1234567890"})
        assert call.keys[-1] == "1234567890#"                        # 종료 키를 안 눌렀으면 붙인다
    run(go())


def test_digits_matching_secure_input_length_are_masked_but_amounts_are_not():
    """민감정보 입력 뒤 ARS 가 입력값을 다시 읽으면(확인 표현 없이도) 입력 자릿수와 같은 숫자열만 가리고,
    금액처럼 자릿수가 다른 숫자는 그대로 둔다 (결과 요약에 쓰인다). 입력값 자체는 저장하지 않는다."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "950101"})
        assert s.secure_len == 6 and "950101" not in str(vars(s))
        await s.on_paragraph("입력하신 계좌 950101 의 출금 가능 금액은 1,234,000원입니다. 감사합니다.")
        cap = s.captions[-1]["text"]
        assert "950101" not in cap and "1,234,000원" in cap and "******" in cap
        assert "950101" not in str(s.hub.history)
    run(go())


def test_secure_request_closes_when_ars_returns_to_menu():
    """민감정보 입력을 기다리던 ARS 가 입력 없이 메뉴로 돌아가면, 입력 요청을 닫고 그 메뉴를 판단한다 (요청에 매달려 멈추지 않는다)."""
    async def go():
        s, call = await secure_pending()
        await s.on_paragraph(TIMEOUT_MSG)
        await s.on_paragraph(ROOT)                                   # ARS 가 처음 메뉴로 돌아감
        assert s.pending is None and not s.secure_window
        assert call.keys == ["3", "2", "3"]                          # 첫 메뉴를 다시 판단하여 잔액조회(3)
    run(go())


def test_stale_menu_heard_before_key_is_not_decided_again():
    """AI 판단이 ARS 입력 대기보다 늦어 ARS 가 같은 메뉴를 다시 읽기 시작한 뒤 키를 누른 경우,
    키보다 먼저 시작된 그 안내는 키 이전 상태의 안내이므로 다시 판단하지 않는다 (같은 키 두 번 전송 방지)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        t0 = time.monotonic()
        await s.on_paragraph(ROOT, started_at=t0)                    # 판단 → 카드 업무(5)
        assert call.keys == ["5"] and s.last_key_at >= t0            # Windows 는 monotonic 이 16ms 단위라 같을 수 있다
        replay_started = s.last_key_at - 0.5                         # 키보다 먼저 시작된 같은 메뉴(재생)
        await s.on_paragraph(ROOT, started_at=replay_started)
        assert call.keys == ["5"]                                    # 두 번 누르지 않는다
        assert s.captions[-1]["text"] == ROOT                        # 자막은 남긴다
        await s.on_paragraph(CARD, started_at=time.monotonic())      # 키 이후 시작된 다음 메뉴 → 판단
        assert call.keys == ["5", "3"]
    run(go())


def test_stale_error_heard_before_key_is_not_a_rejection():
    """키보다 먼저 시작된 '잘못 누르셨습니다'는 지금 보낸 키에 대한 거부가 아니다. 거부로 세지 않고,
    그 뒤 키에 대한 진짜 반응은 정상적으로 처리한다."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT, started_at=time.monotonic())      # 5
        await s.on_paragraph(ERR, started_at=s.last_key_at - 0.5)    # 키 이전에 시작된 오류 안내
        assert ev(s, "input_rejected") == [] and s.input_errors == 0
        await s.on_paragraph(ERR, started_at=time.monotonic())       # 키에 대한 진짜 거부
        assert len(ev(s, "input_rejected")) == 1
    run(go())


def test_ai_classify_failure_keeps_call_alive():
    """AI classify 가 예외를 내도 통화는 계속된다. 그 안내는 자막만 남기고, 다음 안내부터 정상 처리한다."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        orig = s.d.interp.classify
        calls = {"n": 0}

        async def flaky(text):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("AI 시간 초과")
            return await orig(text)

        s.d.interp.classify = flaky
        await s.on_paragraph(ROOT)                                   # 실패 → 자막만
        assert call.keys == [] and s.captions[-1]["text"] == ROOT and s.status == "listening"
        await s.on_paragraph(ROOT)                                   # ARS 가 다시 읽음 → 정상 판단
        assert call.keys == ["5"]
    run(go())


def test_ai_score_failure_asks_user_instead_of_stalling():
    """AI score 가 예외를 내면 메뉴 판단을 멈추지 않고 사용자에게 묻는다 (unknown_prompt). 사용자는 select 로 진행할 수 있다."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)

        async def broken(goal, options, path=None):
            raise RuntimeError("AI 오류")

        s.d.interp.score = broken
        await s.on_paragraph(ROOT)
        req = s.pending
        assert req and req["reason"] == "unknown_prompt" and "해석하지 못했습니다" in req["message"]
        assert [o["key"] for o in req["options"]] == ["3", "4", "5", "0"]
        await s.action({"type": "select", "request_id": req["request_id"], "key": "5"})
        assert call.keys == ["5"] and s.status == "listening"
    run(go())


def test_ai_goal_summary_and_merge_failure_fallbacks():
    """summarize_goal·merge_answer 가 실패해도 세션 생성과 답변 처리는 진행된다."""
    async def go():
        from interpreter import MockInterpreter
        from navigator import Navigator
        from session import Deps
        from stt import NullRecognizer

        class Broken(MockInterpreter):
            async def summarize_goal(self, goal):
                raise RuntimeError("AI 오류")

            async def merge_answer(self, goal, question, answer):
                raise RuntimeError("AI 오류")

        st = make_settings(clarify_expire_s=5, clarify_drop_s=10)
        interp = Broken()
        m = SessionManager(Deps(st, ManualBackend(), NullRecognizer(), interp, Navigator(interp)))
        out = await m.create("15889999", "카드 관련 문의")
        assert out["goal_summary"] == "'카드 관련 문의' 업무를 찾아 드릴게요."
        s = m.sessions[out["session_id"]]
        await s._on_answer()
        await s.on_paragraph(ROOT)                                   # low_confidence 재질문
        req = s.pending
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "카드 결제일 바꾸고 싶어"})
        assert s.goal == "카드 결제일 바꾸고 싶어" and m.deps.backend.call.keys == ["5"]
    run(go())


def test_key_send_failure_is_409_for_user_and_harmless_for_ai():
    """통화가 끊기는 중이라 키 전송이 실패하면 사용자 조작은 409 로 답하고, AI 탐색은 예외 없이 멈춘다."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)

        async def failing(keys):
            raise RuntimeError("leg closed")

        call.send_keys = failing
        await s.on_paragraph(ROOT)                                   # AI 가 5를 누르려다 실패 → 예외 없이 계속
        assert not s.ended and s.path == []
        with pytest.raises(ApiError) as e:
            await s.action({"type": "key", "key": "5"})
        assert e.value.status == 409
    run(go())


def test_backtrack_queue_retries_key_rejected_by_ars():
    """이전 계층 복귀('처음으로' 키 + 지나온 키) 중 ARS 가 키를 거부하면, 메뉴가 다시 나올 때 같은 키를 다시 누른다."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        s.key_queue = ["5"]                                          # 복귀 중: 다음에 누를 키
        s.after_backtrack = True
        await s.on_paragraph(ROOT)                                   # 대기열 키 5 전송
        assert call.keys == ["5"] and s.key_queue == []
        await s.on_paragraph(ERR)                                    # 거부
        assert s.key_queue == ["5"]
        await s.on_paragraph(ROOT)                                   # 메뉴 재생 → 다시 5
        assert call.keys == ["5", "5"] and s.key_queue == []
    run(go())


def test_remote_hangup_while_waiting_for_user_is_user_input_timeout():
    """재질문·민감정보 입력 대기 중 상대가 끊으면 user_input_timeout. 상담원 대기 중 끊기면 failed."""
    async def go():
        s, _ = await manual("카드 관련 문의", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        assert s.pending
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended)
        assert s.end_reason == "user_input_timeout"

        s, _ = await secure_pending()
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended)
        assert s.end_reason == "user_input_timeout"

        s, _ = await manual("상담원 연결해줘", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(AGENT_WAIT)
        assert s.status == "waiting_agent"
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended)
        assert s.end_reason == "failed"
    run(go())


def test_ars_ending_notices_end_the_call_with_matching_reason():
    """문자 발송·업무시간 외·거절 안내는 해당 end_reason 으로 통화를 끝낸다 (outcome: partial / failed / failed)."""
    async def go():
        from interpreter import _norm
        cases = [("최근 거래내역을 문자로 보내 드렸습니다. 감사합니다.", None, "message_sent", "partial"),
                 ("지금은 상담 가능 시간이 아닙니다. 평일 9시부터 18시까지 이용해 주십시오.", "out_of_hours", "out_of_hours", "failed"),
                 ("본인 확인에 실패하여 서비스를 이용하실 수 없습니다.", "rejected", "ars_rejected", "failed")]
        for text, kind, reason, outcome in cases:
            s, call = await manual("최근 거래내역 보고 싶어", clarify_expire_s=5, clarify_drop_s=10)
            if kind:
                s.d.interp.prompts[_norm(text)] = (text, {"kind": kind})
            await s.on_paragraph(ROOT)
            await s.on_paragraph(text)
            assert s.ended and s.end_reason == reason and s.result()[1]["outcome"] == outcome
            assert s.captions[-1]["text"] == text
    run(go())


def test_max_call_duration_ends_call():
    """상대가 받은 뒤 최대 통화 시간이 지나면 max_duration 으로 종료한다."""
    async def go():
        s, call = await manual("카드 관련 문의", max_call_s=0.2, clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)                                   # 재질문 대기 중
        await wait_for(s, lambda s: s.ended, timeout=2)
        assert s.end_reason == "max_duration" and s.result()[1]["outcome"] == "failed"
    run(go())


def test_dial_failures_map_to_end_reasons():
    """상대가 받지 않음·통화 중·발신 오류는 그대로 end_reason 이 되고, 목적 요약은 유지된다."""
    async def go():
        for reason in ("no_answer", "busy", "failed"):
            s, _ = await manual("카드 관련 문의", clarify_expire_s=5, clarify_drop_s=10)
            await s._on_end(reason)
            assert s.ended and s.end_reason == reason and s.result()[1]["outcome"] == "failed"
            assert s.goal_summary
    run(go())


def test_confirm_question_without_answer_asks_user():
    """중요 선택이 아닌 확인 질문에 AI 가 누를 키를 정하지 못하거나(None) 실패하면 unknown_prompt 로 사용자에게 묻는다."""
    async def go():
        from interpreter import _norm
        text = "조회 결과를 문자로 받으시겠습니까? 받으시려면 1번, 아니면 2번을 눌러 주십시오."
        for broken in (False, True):
            s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
            s.d.interp.prompts[_norm(text)] = (text, {"kind": "confirm", "options": [
                {"key": "1", "label": "문자 수신"}, {"key": "2", "label": "수신 안 함"}]})
            if broken:
                async def boom(goal, prompt):
                    raise RuntimeError("AI 오류")
                s.d.interp.answer_confirm = boom
            await s.on_paragraph(ROOT)
            await s.on_paragraph(text)
            req = s.pending
            assert req and req["reason"] == "unknown_prompt" and [o["key"] for o in req["options"]] == ["1", "2"]
            assert call.keys == ["5"]
    run(go())


def test_key_pressed_before_next_menu_is_not_labeled_as_old_menu_choice():
    """마지막 키 전송 뒤 새 메뉴를 듣기 전에(민감정보 안내 중 등) 사용자가 누른 키는 지난 메뉴의 선택지로
    해석하지 않는다. "5 입력" 으로만 기록하고 지나온 계층에도 넣지 않는다."""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", user_key_hold_s=0.1, clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)                                   # AI 가 잔액조회(3)
        assert call.keys == ["3"] and len(s.layers) == 1
        await s.action({"type": "key", "key": "5"})                  # 다음 안내를 듣기 전에 누름
        await s.action({"type": "key", "key": "6"})
        assert [p["text"] for p in s.path[1:]] == ["5 입력", "6 입력"]
        assert all(p["options"] == [] and p["selected"]["category"] is None for p in s.path[1:])
        assert len(s.layers) == 1                                    # 첫 메뉴의 '카드 업무(5)' 로 쌓이지 않는다
        await s.on_paragraph(ROOT)                                   # 메뉴를 새로 들은 뒤(AI 보류 중) 누른 키는 선택지로 기록
        await s.action({"type": "key", "key": "5"})
        assert s.path[-1]["text"].endswith("카드 업무(5) 선택")
    run(go())


def test_new_call_allowed_while_previous_finishes_after_hangup():
    """상대가 끊은 뒤 마지막 안내·요약을 처리하는 동안(최대 10초)에도 새 통화를 만들 수 있다 (회선은 비어 있다)."""
    async def go():
        m = await new_manager()
        m.deps.backend.tree["timeout_s"] = 300
        out, s = await start(m, "이번 달 카드 대금 알고 싶어")
        await wait_for(s, lambda s: s.status == "listening")
        s.hangup_seen = True                                         # 끊긴 뒤 마무리 처리 중인 상태
        assert m.active() is None
        out2 = await m.create("15889999", "통장 잔액 알고 싶어")   # 409 가 아니다
        assert out2["session_id"] != out["session_id"]
        for x in list(m.sessions.values()):
            if not x.ended:
                await x.finish("user_canceled")
    run(go())


# ---------------------------------------------------------------------------
# API 호출 확인 도구 (api_check.py)
# ---------------------------------------------------------------------------
def test_api_check_ulaw_decoder_matches_server():
    """api_check 의 μ-law 복호는 서버(ClawOps SDK)와 같은 값을 낸다 (WAV 저장·음량 계산의 기준)"""
    from clawops.agent._audio import ulaw_to_pcm16 as server_decode

    import api_check

    every = bytes(range(256))
    assert api_check.ulaw_to_pcm16(every) == server_decode(every)


from contextlib import contextmanager


@contextmanager
def live_server(**kw):
    """실제 HTTP·WebSocket 서버(가상 ARS, 오디오 포함)를 별도 스레드에서 띄우고 주소를 돌려준다."""
    import socket
    import threading

    import uvicorn

    st = make_settings(sim_speed=0.25, sim_audio=True, **kw)
    app = create_app(st)
    app.state.manager.deps.backend.tree["timeout_s"] = 300  # 가상 ARS 입력 대기를 길게 둔다
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    end = time.monotonic() + 10
    while not server.started and time.monotonic() < end:
        time.sleep(0.05)
    assert server.started
    try:
        yield f"http://127.0.0.1:{port}", st
    finally:
        server.should_exit = True
        th.join(10)


def test_api_check_tool_against_live_server(tmp_path):
    """api_check.py check 를 실제 HTTP·WebSocket 서버(가상 ARS, 오디오 포함)에 실행하면 모든 항목이 통과한다"""
    import wave

    import api_check

    wav = tmp_path / "audio_check.wav"
    with live_server() as (base, st):
        async def go():
            c = api_check.Client(base, st.app_key)
            try:
                return await api_check.scenario_check(c, "15889999", str(wav))
            finally:
                await c.http.aclose()

        assert run(go()) == 0
    with wave.open(str(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 8000)
        assert w.getnframes() > 8000  # 1초 이상


def test_api_check_audio_judgement(tmp_path):
    """오디오 판정: 정상은 통과, 긴 멈춤·프레임 손실·무음·형식 이상·프레임 없음은 실패로 잡는다"""
    import api_check
    from audio import SILENCE_FRAME, hum_ulaw

    def judge(frames, arrivals):
        p = api_check.AudioProbe(str(tmp_path / "a.wav"))
        p.frames, p.arrivals = frames, arrivals
        ck = api_check.Checks()
        api_check.audio_checks(ck, p)
        return {name.split(" (")[0]: mark for mark, name, _ in ck.items}

    tone = [hum_ulaw()] * 100
    every20 = [i * 0.02 for i in range(100)]

    def delayed(sec):  # 49번 프레임 뒤 sec 만큼 멈췄다가 밀린 프레임이 한꺼번에 도착 (실제 스트림의 지연 모양)
        return every20[:50] + [max(t, every20[49] + sec) for t in every20[50:]]

    assert set(judge(tone, every20).values()) == {"통과"}
    assert set(judge(tone, delayed(0.12)).values()) == {"통과"}     # 120ms 지연: 참고 수치만
    assert judge(tone, delayed(0.4))["WS audio 끊김"] == "실패"     # 400ms 멈춤
    kept = [i for i in range(100) if i % 10 != 9]                   # 10프레임마다 1개 손실 (수신 비율 약 90%)
    assert judge([tone[i] for i in kept], [every20[i] for i in kept])["WS audio 끊김"] == "실패"
    assert judge([SILENCE_FRAME] * 100, every20)["WS audio 소리 포함"] == "실패"
    assert judge([hum_ulaw(40)] * 50, [i * 0.04 for i in range(50)])["WS audio 프레임 형식"] == "실패"
    assert judge([], []) == {"WS audio 연결·프레임 수신": "실패"}


# ---------------------------------------------------------------------------
# 실시간 듣기·디버깅 콘솔 (live.py)
# ---------------------------------------------------------------------------
def _band_energy(pcm, hz, rate=8000):
    import numpy as np

    x = np.asarray(pcm, dtype=float)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / rate)
    return spec[(f > hz - 15) & (f < hz + 15)].sum() / max(spec.sum(), 1e-9)


def test_live_dtmf_tone_is_mixed_into_playback():
    """번호가 눌리면 그 번호의 DTMF 톤(3: 697+1477Hz)이 재생음에 섞이고, 톤이 끝나면 원래 소리만 남는다"""
    import numpy as np

    import live
    from audio import SILENCE_FRAME

    sp = live.Speaker(None, None)
    try:
        sp.press("3")
        out = np.concatenate([sp.process(SILENCE_FRAME) for _ in range(15)])  # 300ms
        tone = out[:960]                                                     # 앞 120ms
        assert _band_energy(tone, 697) > 0.2 and _band_energy(tone, 1477) > 0.2
        assert _band_energy(tone, 1336) < 0.05                               # 다른 번호(2·5·8·0) 성분은 없음
        assert not out[1600:].any()                                           # 톤(120ms)+간격(80ms) 뒤는 무음
        sp.press("5")
        sp.clear_tone()                                                       # 소리를 끄면 남은 톤을 버린다
        assert not sp.process(SILENCE_FRAME).any()
    finally:
        sp.close()


def test_live_describe_shows_keys_and_what_to_type():
    import live

    lines = live.describe({"event": "path_updated", "data": {
        "step": 1, "text": "", "actor": "ai",
        "options": [{"key": "3", "label": "잔액조회"}, {"key": "4", "label": "거래내역조회"}],
        "selected": {"key": "3", "label": "잔액조회", "confidence": 0.9}}})
    assert lines[0] == "▶ AI: [3] 누름 - 잔액조회 (신뢰도 0.90)" and "[4] 거래내역조회" in lines[1]
    amb = live.describe({"event": "request", "data": {"kind": "clarification", "reason": "ambiguous", "message": "어느 쪽인가요?",
                                                      "options": [{"key": "2", "label": "결제 대금 조회", "confidence": 0.7}]}})
    assert amb[-1].endswith("a <답변> 또는 s <번호>") and "[2] 결제 대금 조회 0.70" in amb[1]
    crit = live.describe({"event": "request", "data": {"kind": "clarification", "reason": "critical_action", "message": "m", "options": []}})
    assert "음성 답변 불가" in crit[-1]
    sec = live.describe({"event": "request", "data": {"kind": "secure_input", "label": "주민등록번호 앞 여섯 자리", "terminator": "#"}})
    assert "p 를 입력" in sec[-1] and "#" in sec[0]


def test_live_sounddevice_sink_upsamples_to_48k(monkeypatch):
    """sounddevice 출력: 시작 시 100ms 여유를 쓰고, 20ms 프레임(160표본)을 48kHz(960표본)로 올려 쓴다"""
    import sys
    import types

    import numpy as np

    import live

    writes = []

    class FakeStream:
        def __init__(self, **kw):
            assert kw["samplerate"] == 48000 and kw["channels"] == 1 and kw["dtype"] == "int16"

        def start(self):
            pass

        def write(self, data):
            writes.append(len(data) // 2)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(RawOutputStream=FakeStream))
    sink = live.SoundDeviceSink()
    sink.write(np.full(160, 1000, dtype=np.int16))
    sink.close()
    assert writes == [4800, 960]


def test_live_ffplay_input_options_with_ffmpeg(tmp_path):
    """ffplay 에 넘기는 입력 옵션(16bit PCM 8kHz, 표준 입력)을 실제 FFmpeg 로 확인한다 (FFmpeg 가 없으면 생략)"""
    import shutil
    import wave

    import numpy as np

    import live

    exe = shutil.which("ffmpeg")
    if not exe:
        pytest.skip("FFmpeg 없음")
    out = tmp_path / "ffmpeg_out.wav"
    sink = live.FFplaySink([exe, "-hide_banner", "-loglevel", "error", *live.FFPLAY_INPUT, "-y", str(out)])
    t = np.arange(8000) / 8000
    pcm = (8000 * np.sin(2 * np.pi * 400 * t)).astype(np.int16)
    for i in range(0, 8000, 160):
        sink.write(pcm[i:i + 160])
    sink.close()
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels()) == (8000, 1) and abs(w.getnframes() - 8000) < 200


def test_live_console_against_live_server(tmp_path, capsys):
    """콘솔로 실제 서버에 발신 → 재질문에 텍스트로 답변 → 소리 끄기·켜기 → 종료. 번호 선택 표시와 DTMF 톤 녹음을 확인한다"""
    import wave

    import numpy as np

    import live

    wav = tmp_path / "live.wav"
    with live_server() as (base, st):
        async def go():
            sp = live.Speaker(None, str(wav))
            con = live.Console(base, st.app_key, sp)
            try:
                assert await con.create("15889999", "결제 관련")
                con.start(sound=True)

                async def until(cond, sec=10):
                    end = time.monotonic() + sec
                    while not cond():
                        assert time.monotonic() < end, f"시간 초과 status={con.status}"
                        await asyncio.sleep(0.02)

                await until(lambda: con.pending and con.pending.get("reason") == "ambiguous")
                await con.command("a 결제일을 바꾸고 싶어")
                await until(lambda: con.status in ("agent_done", "ended"))
                await con.command("m")                       # 소리 끄기
                assert not con.sound_is_on
                f0 = con.frames
                await con.command("m")                       # 소리 켜기
                await until(lambda: con.frames > f0)         # 오디오 채널이 다시 연결되어 소리를 받은 뒤 종료한다
                await con.command("q")
                await until(lambda: con.ended.is_set())
                await con.close(hang_up=False)
            finally:
                sp.close()

        run(go())
    out = capsys.readouterr().out
    assert "▶ AI: [5] 누름 - 카드 업무" in out and "▶ AI: [3] 누름 - 결제일 변경" in out
    assert "★ 재질문 (ambiguous)" in out and "★ 결과:" in out and "■ 통화 종료: completed" in out
    assert "소리 꺼짐" in out and out.count("소리 켜짐") == 2
    with wave.open(str(wav)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    # 녹음 어딘가에 5번 DTMF(770+1336Hz)가 섞여 있다
    best = max(_band_energy(pcm[i:i + 960], 770) + _band_energy(pcm[i:i + 960], 1336) for i in range(0, len(pcm) - 960, 160))
    assert best > 0.4


def test_live_asks_goal_before_dialing(monkeypatch, capsys):
    """목적 없이 실행하면 받는 번호와 원하는 업무를 먼저 묻는다. 목 데이터 예시는 번호로 고를 수 있다"""
    import live

    class R:
        def json(self):
            return {"backend": "sim", "interpreter": "interpreter:MockInterpreter",
                    "mock_goals": ["통장 잔액 알고 싶어", "카드 결제일 바꾸고 싶어", "결제 관련"]}

    monkeypatch.setattr(live.httpx, "get", lambda url, timeout=5: R())

    def answers(*lines):
        it = iter(lines)

        def read(prompt):
            try:
                return next(it)
            except StopIteration:
                raise EOFError
        return read

    assert live.ask_goal("http://x", "15889999", answers("", "3")) == ("15889999", "결제 관련")
    assert live.ask_goal("http://x", "15889999", answers("1577-1000", "", "카드 잃어버렸어")) == ("1577-1000", "카드 잃어버렸어")
    assert live.ask_goal("http://x", "15889999", answers("")) == ("15889999", None)      # 입력 없이 끝나면 발신하지 않음
    out = capsys.readouterr().out
    assert " 3) 결제 관련" in out and "목적을 입력해야 전화를 걸 수 있습니다." in out and '"결제 관련" 목적의 전화를 겁니다' in out
