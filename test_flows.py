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


class ManualCall:
    provides_transcripts = True

    def __init__(self) -> None:
        self.keys: list[str] = []

    async def send_keys(self, keys: str) -> float:
        self.keys.append(keys)
        return 0.0

    async def send_audio(self, b: bytes) -> None:
        pass

    async def hangup(self) -> None:
        pass


class ManualBackend:
    def __init__(self) -> None:
        self.call: ManualCall | None = None

    async def start(self) -> None:
        pass

    async def dial(self, to, handlers) -> ManualCall:
        self.call = ManualCall()
        return self.call


async def manual(goal: str, **kw):
    from interpreter import MockInterpreter, _norm
    from navigator import Navigator
    from session import Deps
    from stt import NullRecognizer

    st = make_settings(**kw)
    interp = MockInterpreter()
    interp.prompts[_norm(CONFIRM)] = (CONFIRM, {
        "kind": "confirm", "critical": True,
        "options": [{"key": "1", "label": "변경 확정"}, {"key": "2", "label": "취소"}]})
    deps = Deps(st, ManualBackend(), NullRecognizer(), interp,
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
                await con.command("m")                       # 소리 켜기
                await until(lambda: con.sound_is_on)
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
