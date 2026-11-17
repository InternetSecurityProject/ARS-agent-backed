"""
가상 ARS 기반 전체 흐름 시험 (실제 발신 없음).

  python -m pytest -q test_flows.py
"""

from __future__ import annotations

import asyncio
import re
import time

import pytest

from app import build_deps, create_app
from config import Settings
from session import ApiError, SessionManager, mask_digits

SPEED = 0.01  # 안내 길이 1/100 로 단축


def make_settings(**kw) -> Settings:
    st = Settings()
    st.backend, st.sim_speed, st.sim_audio = "sim", SPEED, False
    # .env 의 INTERPRETER/STT 설정과 무관하게 항상 모의 해석기 사용 (실제 AI 호출 없음)
    st.interpreter, st.stt = "interpreter:MockInterpreter", "none"
    st.no_response_s, st.clarify_expire_s, st.clarify_drop_s, st.user_key_hold_s = 5, 0.3, 0.6, 0.2
    st.min_voiced_s = 0.0  # 0.3초 길이의 시험 음성이 많아 짧은 단락 제거는 전용 시험에서만 사용
    st.post_key_caption_s = 0.0  # 키 전송 직후 자막 생략도 전용 시험에서만 사용
    st.recording_save, st.record_always = False, False  # 통화 기록 디스크 저장은 전용 시험에서만 사용 (저장소 폴더 쓰기 방지)
    for k, v in kw.items():
        setattr(st, k, v)
    return st


async def new_manager(**kw) -> SessionManager:
    deps = build_deps(make_settings(**kw))
    deps.backend.tree["timeout_s"] = 300  # 가상 ARS 입력 대기 3초 (재질문 시간 기준보다 긴 값)
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
    """결과 수신 뒤 AI 만 종료(agent_done)되고 통화는 유지됨. 통화 종료는 사용자 조작으로 처리."""
    await wait_for(s, lambda s: "agent_done" in status_seq(s))
    assert not s.agent_active
    if not s.ended:  # 가상 ARS 가 결과 안내 뒤 스스로 끊는 경우도 있음 (그 경우도 completed)
        await s.action({"type": "cancel"})
    assert s.end_reason == "completed"


async def start(m, goal):
    out = await m.create("15889999", goal)
    return out, m.sessions[out["session_id"]]


def test_goal_required_at_start():
    """AI 도우미 사용과 목적은 통화 시작 시 결정. 목적 없는 생성 요청은 invalid_request 로 거절됨을 확인."""
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
        # 입력값·민감정보 안내는 자막·경로에 남지 않음
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
        # 0.3 이상이 하나뿐이면 고를 수 없는 질문이 되므로 점수 순으로 다른 선택지도 표시 (1순위가 맨 앞)
        assert req["options"][0] == {"key": "5", "label": "카드 업무", "category": "카드 업무", "confidence": 0.4}
        assert len(req["options"]) >= 2
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "카드 결제일 바꾸고 싶어"})
        await finish_after_result(s)
        assert [(p["selected"]["key"], p["selected"]["confidence"]) for p in s.path] == [("5", 0.8), ("3", 0.95)]
    run(go())


def test_low_confidence_no_answer_goes_to_agent_menu():
    """재질문 무응답 60초(시험 0.6초) → CLARIFY_DROP_TO_AGENT=1 이면 현재 메뉴의 상담원 연결 선택지로 진행 (기본은 꺼짐: 요청 유지)"""
    async def go():
        m = await new_manager(drop_to_agent=True)
        _, s = await start(m, "카드 관련 문의")
        await wait_for(s, lambda s: s.pending)
        await wait_for(s, lambda s: s.status == "connected_to_agent", timeout=6)
        assert s.path[0]["selected"]["key"] == "0" and s.path[0]["actor"] == "ai"
        await s.action({"type": "cancel"})
        assert s.end_reason == "agent_call_ended"
    run(go())


def test_ambiguous_no_answer_keeps_waiting():
    """상담원 연결 선택지가 없는 메뉴에서 재질문 무응답 → 요청을 유지한 채 계속 청취"""
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
        # 업무 선택지 2개 (한 단계 들어온 메뉴라 '이전 메뉴'·'처음으로' 같은 이동 키도 선택지로 추가됨)
        assert {(o["key"], o["confidence"]) for o in req["options"] if o["confidence"] is not None} == {("2", 0.7), ("3", 0.7)}
        assert s.status == "awaiting_user"
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "결제일을 바꾸고 싶어"})
        await finish_after_result(s)
        assert [p["selected"]["key"] for p in s.path] == ["5", "3"]
    run(go())


def test_revise_goal_before_first_key():
    """다시 말하기는 목적 설정 단계(첫 번호 전)에만 허용. 첫 메뉴 재질문 중 새 목적 → 재질문 종료 후 첫 메뉴에서 다시 판단"""
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
    """맞는 선택지가 없으면 종료 전에 이 계층에서 한 번 사용자 재질문 (첫 메뉴에서 묻지 않고 바로 종료되는 경우 방지, 선택지는 점수 순).
    다시 말한 목적으로도 없으면 not_found 로 종료."""
    from navigator import NOT_FOUND_ASK

    async def go():
        m = await new_manager(clarify_expire_s=5, clarify_drop_s=10)
        _, s = await start(m, "외화 환전하고 싶어")
        req = await wait_for(s, lambda s: s.pending)
        assert req["reason"] == "low_confidence" and req["message"] == NOT_FOUND_ASK and len(req["options"]) >= 2
        assert not any(o["label"] in ("이전 메뉴", "처음으로") for o in req["options"])   # 첫 계층에는 이동 키 미추가
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "외화 바꾸는 거요"})
        await wait_for(s, lambda s: s.ended)
        assert s.end_reason == "not_found" and s.path == []
    run(go())


def test_not_found_after_checking_previous_layer():
    async def go():
        m = await new_manager(clarify_expire_s=5, clarify_drop_s=10)
        _, s = await start(m, "외화 잔액 환전하고 싶어")
        req = await wait_for(s, lambda s: s.pending)          # 종료 전 재질문
        assert req["message"].startswith("원하시는 업무와 맞는 메뉴를 찾지 못했습니다")
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "외화 환전"})
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
        # 종료된 통화의 이벤트 채널 재연결 시 snapshot(ended) 만 받고 서버가 연결 종료 (무한 대기 방지)
        with c.websocket_connect(f"/v1/sessions/{sid}/events?token={tok}") as ws:
            snap = ws.receive_json()
            assert snap["event"] == "snapshot" and snap["data"]["status"] == "ended"
            with pytest.raises(Exception):
                ws.receive_json()
        with c.websocket_connect(f"/v1/sessions/{sid}/audio?token={tok}") as ws:
            with pytest.raises(Exception):
                ws.receive_bytes()


# ---------------------------------------------------------------------------
# ClawOps 백엔드 연결부: SDK 대역(가짜 ClawOpsAgent)으로 발신 → 오디오 수신 → 발화 구간 → STT → 키 전송 → 종료
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

        async def answer(self):  # SDK 동작 모사: call_start 핸들러 → session_factory 객체 시작
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

        # 허밍 소리의 낮은 RMS 를 고려한 낮은 임계값
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
# 수동 ARS: 시험 코드가 안내 문장을 직접 넣어 단계별 동작 확인 (가상 ARS 의 시간 흐름 없음)
# ---------------------------------------------------------------------------
ROOT = ("KB국민은행입니다. 잔액조회는 3번, 거래내역조회는 4번, 카드 업무는 5번, 상담직원 연결은 0번을 눌러 주십시오. "
        "다시 들으시려면 우물정자를 눌러 주십시오.")
CARD = "카드 분실 신고는 1번, 결제 대금 조회는 2번, 결제일 변경은 3번을 눌러 주십시오. 이전 메뉴는 별표를 눌러 주십시오."
BILL = "이번 달 결제 예정 금액은 352,000원입니다. 감사합니다."
ERR = "잘못 누르셨습니다."
CONFIRM = "결제일을 25일로 변경하시겠습니까? 변경하시려면 1번, 취소는 2번을 눌러 주십시오."
BAL = "알림톡으로 받으시려면 1번, ARS로 계속하시려면 2번을 눌러 주십시오. 이전 메뉴는 별표를 눌러 주십시오."


class ManualCall:
    """transcripts=True: 시험 코드가 안내 문장을 직접 입력(on_paragraph).
    False: 실제 회선처럼 음성 프레임을 넣어 Segmenter → STT 경로로 처리(speak)."""

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
    """시험용 STT: 단락이 끝날 때마다 정해 둔 문장을 차례로 반환."""

    is_local = True

    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)

    async def recognize(self, seg: bytes) -> str:
        return self.lines.pop(0) if self.lines else ""


async def speak(s, sec: float, silence: float = 0.3) -> None:
    """ARS 음성 모사: sec 초 동안 소리 프레임, 이어서 silence 초 동안 무음 프레임 입력 (20ms 단위)."""
    from audio import SILENCE_FRAME, hum_ulaw

    for _ in range(int(sec * 50)):
        await s._on_audio(hum_ulaw())
    for _ in range(int(silence * 50)):
        await s._on_audio(SILENCE_FRAME)
    await asyncio.sleep(0.02)


async def manual(goal: str, stt: ScriptSTT | None = None, record: bool = False, agent: bool = False, **kw):
    """stt 지정 시 음성 경로(Segmenter → STT)로 시험."""
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
    out = await m.create("15889999", goal, record=record, agent=agent)
    s = m.sessions[out["session_id"]]
    await s._on_answer()
    return s, deps.backend.call


def test_result_during_user_key_hold_is_not_lost():
    """직접 선택 뒤 보류 시간(기본 10초) 중에도 결과 안내 처리 (보류 대상은 메뉴 판단뿐)"""
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
    """입력 오류 → 재시도 성공 → 다른 단계의 입력 오류 1회는 재질문 없음. 같은 단계 2회 연속이면 재질문"""
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
    """키 전송 결과로 지나온 메뉴에 다시 도착하는 상황이 max_revisit(여기선 2)회 발생하면 loop 재질문 (기본값은 3, 0 이면 끔)"""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10, max_revisit=2)
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
        assert dec.action == "press" and dec.key == "3"   # 다른 메뉴의 3번은 제외 대상 아님
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
    """키 전송 뒤 무응답 → 다시 듣기 키 → 다시 나온 메뉴는 반복 횟수에서 제외 (명세 7장 5항)"""
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
    """음성 응답 요구 재질문에 answer 시 지난 메뉴로 판단하지 않고 다음 안내 대기"""
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
# 음성 경로 시간 처리
# ---------------------------------------------------------------------------
def test_no_repeat_key_while_next_menu_is_long():
    """다음 메뉴 안내가 무응답 시간보다 길어도 ARS 가 말하는 동안에는 다시 듣기 키 전송 없음.
    누르면 ARS 의 '잘못 누르셨습니다' 뒤 메뉴 재낭독으로 두 안내가 한 자막에 섞임."""
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
        assert len(s.captions) == 2 and "잔액조회 3번" in s.captions[0]["text"] and "ARS" in s.captions[1]["text"]  # 두 안내가 섞이지 않음
    run(go())


def test_no_previous_menu_repeat_key_after_ars_spoke():
    """키 전송 뒤 ARS 가 소리를 냈으면(인식된 문장이 없어도) 다음 안내로 넘어간 것으로 판단.
    이전 메뉴의 다시 듣기 키(#)는 다음 메뉴에서 다른 뜻일 수 있으므로 전송 안 함."""
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
    """키 전송 뒤 ARS 가 아무 소리도 내지 않으면(키가 인식되지 않음) 키를 누른 그 메뉴의 다시 듣기 키 전송"""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=ScriptSTT([ROOT]), no_response_s=0.2,
                               paragraph_silence_ms=200, clarify_expire_s=5, clarify_drop_s=10)
        await speak(s, 0.3)
        await wait_for(s, lambda s: call.keys == ["3"])
        await speak(s, 0, silence=0.4)                               # 회선 무음만 들어옴
        await wait_for(s, lambda s: call.keys == ["3", "#"], timeout=2)
    run(go())


def test_clarify_repeat_key_waits_until_ars_is_quiet():
    """재질문 30초(시험 0.2초) 무응답 시의 다시 듣기 키: ARS 안내 중이면 안내가 끝난 뒤 전송"""
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
    """결과 안내 직후 ARS 가 끊어도(요약 생성 중) completed. 요약 완료 뒤 summary → ended 순서로 전송.
    종료 뒤 늦게 끝난 처리의 상태 되돌림 없음 (새 통화가 409 로 막히지 않음)."""
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
    """결과 안내 뒤 끊겼는데 AI 요약이 제한 시간(hangup_flush_s) 안에 끝나지 않는 경우: 요약 없이 끝내지 않고
    결과 안내 원문으로 summary 전송 뒤 ended 전송. 늦게 끝난 AI 요약은 폐기 (명세 7장 7항)."""
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
    """상담원 음성 감지 시 connected_to_agent 를 먼저 전송(마이크 중계 시작), 상담사 전달 문장(summary)은
    AI 응답 대기 뒤 전송. 연결 표시가 AI 응답 시간만큼 늦어지지 않음."""
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


def test_menu_caption_is_condensed_to_options_and_other_captions_keep_text():
    """자막(caption.text)은 앱에서 해석 없이 그대로 표시. 선택지가 있는 안내는 STT 원문 대신
    'AI 가 뽑은 선택지 이름 + 키' 목록으로 전송, 결과·일반 안내는 원문 전송 (결과 요약 입력 유지)."""
    async def go():
        s, call = await manual("이번 달 카드 대금 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        menu = s.captions[-1]["text"]
        assert menu != ROOT and " · " in menu and "카드 업무 5번" in menu and len(menu) < len(ROOT)
        assert menu == s.hub.history[-1]["data"]["text"] if s.hub.history[-1]["event"] == "caption" else True
        await s.on_paragraph(CARD)
        await s.on_paragraph(BILL)
        assert s.captions[-1]["text"] == BILL                        # 결과 안내는 원문 (요약 입력)
        assert "352,000" in s.summary["summary"]
    run(go())


def test_caption_uses_fixed_transcript_but_records_original():
    """해석기의 받아쓰기 고침 사용 시(LLM_FIX_TRANSCRIPT=1) 자막·결과 요약에는 고친 문장, 판단 기록에는 원문 저장"""
    heard = BILL.replace("결제 예정", "결재 예정")
    async def go():
        s, call = await manual("이번 달 카드 대금 알고 싶어", record=True, clarify_expire_s=5, clarify_drop_s=10)
        orig = s.d.interp.classify

        async def classify(text):
            p = await orig(text)
            if text == heard:
                p.shown, p.fixes = BILL, [("결재 예정", "결제 예정")]
            return p
        s.d.interp.classify = classify
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        await s.on_paragraph(heard)
        assert s.captions[-1]["text"] == BILL
        notes = [e for e in s.recorder.events if e.get("type") == "classified"]
        assert notes[-1]["text"] == heard and notes[-1]["fixes"] == ["결재 예정→결제 예정"]
    run(go())


def test_result_caption_keeps_text_even_with_options():
    """결과·일반 안내에 선택지('다시 들으시려면 1번')가 붙어도 자막은 원문 (결과 요약 입력의 본문 누락 방지)."""
    from interpreter import Option, Prompt
    from session import caption_text
    text = "이번 달 결제 예정 금액은 352,000원입니다. 다시 들으시려면 1번을 눌러 주십시오."
    for kind in ("result", "info", "message_sent", "agent_wait"):
        assert caption_text(Prompt(kind=kind, text=text, options=[Option("1", "다시 듣기")]), text) == text
    menu = Prompt(kind="menu", text=text, options=[Option("1", "다시 듣기")])
    assert caption_text(menu, text) == "다시 듣기 1번"


def test_hangup_right_after_result_speech_is_recognized():
    """결과 안내 직후 끊겨 단락 끝 무음이 없어도 끊기 전까지 들은 안내를 인식해 completed 로 종료"""
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
# 사용자 입력 엣지 케이스
#   입력 종류: DTMF 키(key), 통화 종료(cancel), 요청 응답(answer·select·secure_input·revise_goal)
# ---------------------------------------------------------------------------
AGENT_WAIT = "상담원과 연결 중입니다. 잠시만 기다려 주십시오."
AGENT_HUMAN = "상담원 김민지입니다. 무엇을 도와드릴까요?"


def test_no_auto_repeat_after_agent_done():
    """업무 완료(agent_done) 뒤 사용자가 직접 누른 키에는 무응답 재전송(다시 듣기 키) 없음.
    AI 탐색 종료 뒤 서버의 임의 키 전송 금지."""
    async def go():
        s, call = await manual("이번 달 카드 대금 알고 싶어", no_response_s=0.1, clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        await s.on_paragraph(BILL)
        assert s.status == "agent_done"
        await s.action({"type": "key", "key": "9"})       # 완료 뒤 직접 키
        await asyncio.sleep(0.3)                            # 무응답 시간이 지나도
        assert call.keys == ["5", "2", "9"]                 # 다시 듣기 키(#) 추가 없음
    run(go())


def test_no_auto_repeat_when_connected_to_agent():
    """상담원 연결 중 사용자가 직접 누른 키에는 무응답 재전송 없음 (상담원 통화에 DTMF 끼어듦 방지)."""
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
    """상태별 cancel 의 end_reason: 탐색 중 user_canceled, 결과 뒤 completed."""
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
    """select 의 후보 외 키는 400, answer 의 닫힌(만료된) 요청은 409 로 거절."""
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
    """민감정보 입력 요청이 열려 있으면 일반 key 입력은 409 sensitive_input_blocked (경로의 숫자 기록 방지)."""
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


async def secure_pending(record: bool = False, **kw):
    """잔액조회 → ARS 계속 → 주민등록번호 입력 요청까지 진행한 상태의 (s, call) 반환"""
    s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10, record=record, **kw)
    await s.on_paragraph(ROOT)
    await s.on_paragraph(BAL)
    await s.on_paragraph(RRN)
    assert s.pending and s.pending["kind"] == "secure_input" and call.keys == ["3", "2"]
    return s, call


def test_timeout_notice_while_secure_pending_is_caption_only():
    """민감정보 입력 대기 중(아직 아무 키도 보내지 않음) ARS 의 '입력 시간이 초과되었습니다'는 키 오류 아님.
    input_rejected 없이 자막만 남기고 입력 요청은 그대로 유지."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.on_paragraph(TIMEOUT_MSG)
        await s.on_paragraph(RRN)                                    # ARS 가 입력 안내를 다시 읽음
        assert ev(s, "input_rejected") == [] and s.pending == req   # 같은 요청 유지, 새 요청 없음
        assert s.captions[-1]["text"] == TIMEOUT_MSG and call.keys == ["3", "2"]
    run(go())


def test_timeout_notice_while_clarification_pending_is_caption_only():
    """재질문 대기 중 ARS 가 '입력 시간이 초과되었습니다' 뒤 메뉴를 다시 읽어도
    키 오류 집계·메뉴 재판단 없음 (재질문은 그대로 사용자 응답 대기)."""
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
    """민감정보 입력값 거부 시 input_rejected 는 keys=null 로 전송, 메뉴 키 오류로 집계 안 함 (메뉴 재질문 없음).
    거부 직후 같은 항목의 새 secure_input 요청 생성 (6.3 처리 규칙 6). 이어서 ARS 가 입력 안내를 되풀이하면
    그 요청을 그대로 다시 알림 (요청 추가 생성 없음)."""
    async def go():
        s, call = await secure_pending()
        for n in (1, 2):
            req = s.pending
            await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "9999"})
            await s.on_paragraph(ERR)                                # 입력값 거부
            rej = ev(s, "input_rejected")[-1]
            assert rej["keys"] is None and rej["actor"] == "user" and rej["retry_count"] == n
            again = s.pending                                        # 메뉴 재질문이 아니라 같은 항목의 입력 요청
            assert again and again["kind"] == "secure_input" and again["request_id"] != req["request_id"]
            assert again["label"] == req["label"] and again.get("terminator") == req.get("terminator")
            await s.on_paragraph(RRN)                                # 입력 안내 반복 → 같은 요청을 다시 알림
            assert s.pending["request_id"] == again["request_id"]
        assert "9999" not in str(s.hub.history) and "9999" not in str(s.path)
        assert s.input_errors == 0
    run(go())


def test_secure_input_waits_until_ars_stops_speaking():
    """'입력 시간이 초과되었습니다' 처럼 ARS 가 말하는 중에 받은 입력값은 안내가 끝난 뒤 전송 (말하는 중 전송은 무시됨).
    대기 중 다시 들린 같은 입력 안내로 새 요청 생성 안 함."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        call.speaking = True                                         # ARS 가 시간 초과 안내 중
        r = await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "991332"})
        assert r == {"accepted": True} and call.keys == ["3", "2"]   # 아직 보내지 않음
        await s.on_paragraph(TIMEOUT_MSG)
        await s.on_paragraph(RRN)                                    # 기다리는 동안 입력 안내가 다시 들림
        assert s.pending is None                                     # 새 입력 요청 없음
        await asyncio.sleep(0.2)
        assert call.keys == ["3", "2"]
        call.speaking = False                                        # 안내가 끝남
        await wait_for(s, lambda s: len(call.keys) == 3, timeout=2)
        assert call.keys[-1] == "991332#" and s.last_send_secure and not s.secure_queued
        assert "991332" not in str(s.hub.history)
    run(go())


def test_secure_input_dropped_if_ars_moves_on():
    """대기 중 ARS 가 메뉴로 넘어가거나 너무 길게 말하면 입력값을 메뉴 키로 보내지 않고 폐기."""
    async def go():
        s, call = await secure_pending()
        call.speaking = True
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "991332"})
        await s.on_paragraph(ROOT)                                   # 첫 메뉴로 돌아감 (AI 의 메뉴 재판단)
        await wait_for(s, lambda s: not s.secure_queued, timeout=2)
        assert "991332#" not in call.keys and ev(s, "input_rejected")[-1]["keys"] is None

        s, call = await secure_pending(secure_wait_max_s=0.3)
        call.speaking = True
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "991332"})
        await wait_for(s, lambda s: not s.secure_queued, timeout=2)  # 0.3초가 지나도 계속 말함
        assert call.keys == ["3", "2"] and ev(s, "input_rejected")
    run(go())


def test_secure_input_terminator_is_not_duplicated():
    """사용자가 종료 키(#)까지 눌러 보낸 경우 서버의 # 중복 추가 없음 (두 번째 # 은 ARS 에서 다른 뜻일 수 있음)."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "123456#"})
        assert call.keys[-1] == "123456#"
        await s.on_paragraph("계좌번호와 우물정자를 눌러 주십시오.")
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "1234567890"})
        assert call.keys[-1] == "1234567890#"                        # 종료 키가 없으면 추가
    run(go())


def test_digits_matching_secure_input_length_are_masked_but_amounts_are_not():
    """민감정보 입력 뒤 ARS 가 입력값을 다시 읽으면(확인 표현 없이도) 입력 자릿수와 같은 숫자열만 가림.
    금액처럼 자릿수가 다른 숫자는 유지 (결과 요약에 사용). 입력값 자체는 저장 안 함."""
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


def test_amount_with_unit_is_not_masked_even_if_length_matches_input():
    """비밀번호 네 자리 입력 뒤 결과 금액의 네 자리 묶음('○○만 ○○○○원')은 단위가 붙어 있으므로 가리지 않음
    (잔액·출금 가능액의 끝 네 자리가 별표로 가려지는 오동작 방지). 단위 없는 같은 자릿수 숫자는 계속 가림."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "4321"})
        assert s.secure_len == 4
        await s.on_paragraph("고객님의 현재 잔액은 36만 2000원이고 출금 가능액은 35만 1500원입니다. 다시 들으시려면 1번, 그만 들으시려면 2번을 눌러주십시오.")
        cap = s.captions[-1]["text"]
        assert "36만 2000원" in cap and "35만 1500원" in cap and "*" not in cap   # 자막은 결과 요약의 입력
    run(go())

    from session import mask_digits
    assert mask_digits("2026년 11월 거래 15건, 4321 확인", 4) == ("2026년 11월 거래 15건, **** 확인", True)
    assert mask_digits("비밀번호 4321 원하시는 서비스", 4)[0] == "비밀번호 **** 원하시는 서비스"   # 띄어 쓴 '원' 은 단위가 아님
    assert mask_digits("1,234,000원", 7)[1] is False


def test_secure_request_closes_when_ars_returns_to_menu():
    """민감정보 입력을 기다리던 ARS 가 입력 없이 메뉴로 돌아가면 입력 요청을 닫고 그 메뉴를 판단 (요청에 묶여 멈추는 상황 방지)."""
    async def go():
        s, call = await secure_pending()
        await s.on_paragraph(TIMEOUT_MSG)
        await s.on_paragraph(ROOT)                                   # ARS 가 처음 메뉴로 돌아감
        assert s.pending is None and not s.secure_window
        assert call.keys == ["3", "2", "3"]                          # 첫 메뉴를 다시 판단하여 잔액조회(3)
    run(go())


def test_stale_menu_heard_before_key_is_not_decided_again():
    """AI 판단이 ARS 입력 대기보다 늦어 ARS 가 같은 메뉴를 다시 읽기 시작한 뒤 키를 누른 경우,
    키보다 먼저 시작된 그 안내는 키 이전 상태의 안내이므로 재판단 없음 (같은 키 두 번 전송 방지)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        t0 = time.monotonic()
        await s.on_paragraph(ROOT, started_at=t0)                    # 판단 → 카드 업무(5)
        assert call.keys == ["5"] and s.last_key_at >= t0            # Windows 의 monotonic 은 16ms 단위라 같은 값 가능
        replay_started = s.last_key_at - 0.5                         # 키보다 먼저 시작된 같은 메뉴(재생)
        await s.on_paragraph(ROOT, started_at=replay_started)
        assert call.keys == ["5"]                                    # 중복 전송 없음
        assert s.captions[-1]["text"] == s.captions[0]["text"] != ROOT  # 자막은 유지 (메뉴는 선택지 요약으로)
        await s.on_paragraph(CARD, started_at=time.monotonic())      # 키 이후 시작된 다음 메뉴 → 판단
        assert call.keys == ["5", "3"]
    run(go())


def test_tail_of_speech_cut_by_key_is_not_a_result():
    """ARS 가 말하는 도중에 키를 보내면 그 안내의 뒷부분이 키 뒤 새 단락으로 들림. 그 조각은 결과·종료로 보지 않음
    (잘린 뒷부분의 오인식이 결과로 처리되어 탐색이 끝나는 오동작 방지).
    ARS 가 조용할 때 보낸 키 뒤의 결과 안내는 그대로 결과."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        call.speaking = True                                         # ARS 가 말하는 도중의 키 전송
        await s.on_paragraph(ROOT, started_at=time.monotonic())      # 5
        call.speaking = False
        assert call.keys == ["5"] and s.key_cut_speech
        await s.on_paragraph("변경이 완료되었습니다.", started_at=s.last_key_at + 0.4)   # 잘린 안내의 짧은 뒷부분
        assert s.agent_active and s.summary is None and s.status != "agent_done"
        await s.on_paragraph(CARD, started_at=time.monotonic())      # 이어지는 메뉴는 그대로 판단
        assert call.keys == ["5", "3"] and not s.key_cut_speech
        await s.on_paragraph(BILL, started_at=s.last_key_at + 0.4)   # 조용할 때 보낸 키 뒤의 결과
        assert not s.agent_active and s.summary is not None
    run(go())


def test_long_result_right_after_key_cut_speech_is_still_a_result():
    """ARS 가 다시 읽는 도중에 사용자가 답하면 키 뒤 남은 소리 때문에 다음 안내 전체가 키 직후 시작한 단락으로 잡힘. 짧은 조각이 아닌
    결과 안내는 잘린 뒷부분으로 버리지 않고 결과로 처리 (2026-11-09·10 실통화: 결과 안내 2건을 놓치고 문자 발송 종료 2건을 failed 로 기록)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        call.speaking = True
        await s.on_paragraph(ROOT, started_at=time.monotonic())      # 5 (ARS 가 말하는 도중)
        call.speaking = False
        assert call.keys == ["5"] and s.key_cut_speech
        await s.on_paragraph(BILL, started_at=s.last_key_at + 0.36)  # 키 직후 시작했지만 결과 안내 전체
        assert not s.agent_active and s.summary is not None
    run(go())


def test_stale_error_heard_before_key_is_not_a_rejection():
    """키보다 먼저 시작된 '잘못 누르셨습니다'는 지금 보낸 키에 대한 거부 아님. 거부로 집계하지 않고
    그 뒤 키에 대한 실제 반응은 정상 처리."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT, started_at=time.monotonic())      # 5
        await s.on_paragraph(ERR, started_at=s.last_key_at - 0.5)    # 키 이전에 시작된 오류 안내
        assert ev(s, "input_rejected") == [] and s.input_errors == 0
        await s.on_paragraph(ERR, started_at=time.monotonic())       # 키에 대한 실제 거부
        assert len(ev(s, "input_rejected")) == 1
    run(go())


def test_ai_classify_failure_keeps_call_alive():
    """AI classify 예외에도 통화 유지. 그 안내는 자막만 남기고 다음 안내부터 정상 처리."""
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
    """AI score 예외 시 메뉴 판단을 멈추지 않고 사용자 재질문 (unknown_prompt). 사용자는 select 로 진행 가능."""
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
    """summarize_goal·merge_answer 실패에도 생성 요청과 답변 처리는 정상 진행."""
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
    """통화 종료 중 키 전송 실패 시 사용자 조작은 409 응답, AI 탐색은 예외 없이 중단."""
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
    """이전 계층 복귀('처음으로' 키 + 지나온 키) 중 ARS 가 키를 거부하면 메뉴가 다시 나올 때 같은 키 재전송."""
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
    """문자 발송·업무시간 외·거절 안내 시 해당 end_reason 으로 통화 종료 (outcome: partial / failed / failed)."""
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
    """상대가 받은 뒤 최대 통화 시간이 지나면 max_duration 으로 종료."""
    async def go():
        s, call = await manual("카드 관련 문의", max_call_s=0.2, clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)                                   # 재질문 대기 중
        await wait_for(s, lambda s: s.ended, timeout=2)
        assert s.end_reason == "max_duration" and s.result()[1]["outcome"] == "failed"
    run(go())


def test_dial_failures_map_to_end_reasons():
    """상대 미응답·통화 중·발신 오류는 그대로 end_reason 으로 사용, 목적 요약은 유지."""
    async def go():
        for reason in ("no_answer", "busy", "failed"):
            s, _ = await manual("카드 관련 문의", clarify_expire_s=5, clarify_drop_s=10)
            await s._on_end(reason)
            assert s.ended and s.end_reason == reason and s.result()[1]["outcome"] == "failed"
            assert s.goal_summary
    run(go())


def test_confirm_question_without_answer_asks_user():
    """중요 선택이 아닌 확인 질문에 AI 가 누를 키를 정하지 못하거나(None) 실패하면 unknown_prompt 로 사용자 재질문."""
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
    해석하지 않음. "5 입력" 으로만 기록하고 지나온 계층에도 미포함."""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", user_key_hold_s=0.1, clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(ROOT)                                   # AI 가 잔액조회(3)
        assert call.keys == ["3"] and len(s.layers) == 1
        await s.action({"type": "key", "key": "5"})                  # 다음 안내를 듣기 전에 누름
        await s.action({"type": "key", "key": "6"})
        assert [p["text"] for p in s.path[1:]] == ["5 입력", "6 입력"]
        assert all(p["options"] == [] and p["selected"]["category"] is None for p in s.path[1:])
        assert len(s.layers) == 1                                    # 첫 메뉴의 '카드 업무(5)' 로 쌓이지 않음
        await s.on_paragraph(ROOT)                                   # 메뉴를 새로 들은 뒤(AI 보류 중) 누른 키는 선택지로 기록
        await s.action({"type": "key", "key": "5"})
        assert s.path[-1]["text"].endswith("카드 업무(5) 선택")
    run(go())


def test_new_call_allowed_while_previous_finishes_after_hangup():
    """상대가 끊은 뒤 마지막 안내·요약을 처리하는 동안에도 새 통화 생성 가능 (회선은 비어 있음)."""
    async def go():
        m = await new_manager()
        m.deps.backend.tree["timeout_s"] = 300
        out, s = await start(m, "이번 달 카드 대금 알고 싶어")
        await wait_for(s, lambda s: s.status == "listening")
        s.hangup_seen = True                                         # 끊긴 뒤 마무리 처리 중인 상태
        assert m.active() is None
        out2 = await m.create("15889999", "통장 잔액 알고 싶어")   # 409 아님
        assert out2["session_id"] != out["session_id"]
        for x in list(m.sessions.values()):
            if not x.ended:
                await x.finish("user_canceled")
    run(go())


# ---------------------------------------------------------------------------
# API 호출 확인 도구 (api_check.py)
# ---------------------------------------------------------------------------
def test_api_check_ulaw_decoder_matches_server():
    """api_check 의 μ-law 복호 값이 서버(ClawOps SDK)와 같음을 확인 (WAV 저장·음량 계산의 기준)"""
    from clawops.agent._audio import ulaw_to_pcm16 as server_decode

    import api_check

    every = bytes(range(256))
    assert api_check.ulaw_to_pcm16(every) == server_decode(every)


from contextlib import contextmanager


@contextmanager
def live_server(**kw):
    """실제 HTTP·WebSocket 서버(가상 ARS, 오디오 포함)를 별도 스레드에서 실행하고 주소 반환."""
    import socket
    import threading

    import uvicorn

    st = make_settings(sim_speed=0.25, sim_audio=True, **kw)
    app = create_app(st)
    app.state.manager.deps.backend.tree["timeout_s"] = 300  # 가상 ARS 입력 대기 연장
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
    """실제 HTTP·WebSocket 서버(가상 ARS, 오디오 포함) 대상 api_check.py check 의 전 항목 통과 확인"""
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
    """오디오 판정: 정상은 통과, 긴 멈춤·프레임 손실·무음·형식 이상·프레임 없음은 실패 판정"""
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
    kept = [i for i in range(100) if i % 10 != 9]                   # 10프레임마다 1개 손실 (10개 중 9개 수신)
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
    """번호 입력 시 그 번호의 DTMF 톤(3: 697+1477Hz)이 재생음에 섞이고 톤이 끝나면 원래 소리만 남음"""
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
        sp.clear_tone()                                                       # 소리 끄기 시 남은 톤 폐기
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
    """sounddevice 출력: 시작 시 100ms 여유분 출력, 20ms 프레임(160표본)은 48kHz(960표본)로 올려 출력"""
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
    """ffplay 에 넘기는 입력 옵션(16bit PCM 8kHz, 표준 입력)을 실제 FFmpeg 로 확인 (FFmpeg 가 없으면 생략)"""
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
    """콘솔로 실제 서버에 발신 → 재질문에 텍스트로 답변 → 소리 끄기·켜기 → 종료. 번호 선택 표시와 DTMF 톤 녹음 확인"""
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
                await until(lambda: con.frames > f0)         # 오디오 채널 재연결로 소리를 받은 뒤 종료
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
    # 녹음 안의 5번 DTMF(770+1336Hz) 포함 확인
    best = max(_band_energy(pcm[i:i + 960], 770) + _band_energy(pcm[i:i + 960], 1336) for i in range(0, len(pcm) - 960, 160))
    assert best > 0.4


def test_live_asks_goal_before_dialing(monkeypatch, capsys):
    """목적 없이 실행하면 받는 번호와 원하는 업무를 먼저 질문. 목 데이터 예시는 번호로 선택 가능"""
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


def test_secure_window_survives_short_notice_and_masks_late_readback():
    """민감정보 전송 뒤 '잠시만 기다려 주십시오' 같은 짧은 안내가 먼저 오고 되읽기가 그다음에 와도 되읽기 숫자 가림
    (짧은 안내 뒤 되읽기 숫자 노출 방지). 자막·저장 값에 숫자 없음 (묻는 문구만 되읽힌 원문, 판단 입력은 가린 문장)."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "9501011234567"})
        await s.on_paragraph("잠시만 기다려 주십시오.")
        assert s.secure_window                                        # 짧은 안내로는 구간 유지
        readback = "입력하신 주민등록번호는 구 오 공 일 공 일 일 이 삼 사 오 육 칠 입니다. 맞으면 1번, 틀리면 2번을 눌러 주십시오."
        from interpreter import _norm
        masked = "입력하신 주민등록번호는 ************* 입니다. 맞으면 1번, 틀리면 2번을 눌러 주십시오."
        s.d.interp.prompts[_norm(masked)] = (masked, {"kind": "confirm",
            "options": [{"key": "1", "label": "맞으면"}, {"key": "2", "label": "틀리면"}]})
        await s.on_paragraph(readback)
        assert "구 오 공 일" not in str(s.captions) and "9501011234567" not in str(vars(s))   # 자막·저장 값에 숫자 없음
        assert s.pending is not None and s.pending["kind"] == "clarification"   # 사용자에게 확인 질문
        assert s.pending["message"] == readback                                 # 묻는 문구만 되읽힌 숫자 그대로 (사용자가 확인)
        assert s.pending["ars_prompt"] == masked                                # 그 밖의 필드·판단 입력은 가린 문장
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "1"})
        assert s.path[-1]["selected"]["label"] == "주민등록번호 확인"           # 경로에는 답('맞음')이 아니라 확인한 항목
    run(go())


def test_readback_is_masked_even_after_secure_window_closed():
    """되읽기 구간이 닫힌 뒤에 '입력하신 … 맞으면' 문장이 와도 한국어 숫자를 길게 읽으면 입력값 되읽기로 보고 가림."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "9501011234567"})
        s.secure_window = False
        await s.on_paragraph("입력하신 주민등록번호는 구 오 공 일 공 일 일 이 삼 사 오 육 칠 입니다. 맞으면 1번, 틀리면 2번을 눌러 주십시오.")
        assert "구 오 공" not in str(s.captions)                                 # 자막 전송 없음
    run(go())


def test_spoken_digit_run_is_masked_even_if_length_differs():
    """인식이 틀려 자릿수가 달라도 한국어 숫자를 여섯 자 이상 이어 읽는 문장은 가림."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "950101"})
        await s.on_paragraph("확인된 번호 구 오 공 일 공 일 일 로 조회합니다. 잠시만 기다려 주십시오.")
        assert "구 오 공" not in str(s.captions) and "*******" in s.captions[-1]["text"]
    run(go())


# "입력하신" 없이 번호만 읽고 확인을 묻는 되읽기 (전화 음질로 숫자 인식이 깨진 형태)
BROKEN_READBACK = "1. 영 1. 영 구해 3. 영 번입니다. 맞으면 1번, 다시 입력하시려면 2번을 눌러 주십시오."


def test_mask_readback_hides_broken_digits_but_keeps_options():
    from session import mask_readback, readback_options

    masked = mask_readback(BROKEN_READBACK)
    head, tail = masked.split("맞으면", 1)
    assert not re.search(r"[0-9공영일이삼사오육칠팔구]", head)            # 확인 표현 앞에는 숫자가 하나도 남지 않음
    assert tail.startswith(" 1번") and "2번" in tail
    assert [(o.key, o.label) for o in readback_options(masked)] == [("1", "맞으면"), ("2", "다시 입력하시려면")]


def test_readback_without_input_word_is_masked_and_asked_even_if_ai_says_info():
    """'입력하신' 없는 되읽기도 되읽기 구간의 확인 질문이면 숫자를 가리고(자막 없음) 사용자에게 질문.
    AI 가 일반 안내(info)로 분류해도 규칙으로 선택지를 찾아 질문 (묻지 않으면 ARS 가 입력 시간 경과로 넘어감)."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "01055554444"})
        n_captions = len(s.captions)
        await s.on_paragraph(BROKEN_READBACK)                                   # 목 해석기는 모르는 문장을 info 로 반환
        assert len(s.captions) == n_captions                                    # 되읽기는 자막 전송 없음
        assert s.pending is not None and s.pending["kind"] == "clarification"
        assert [o["key"] for o in s.pending["options"]] == ["1", "2"]
        head = s.pending["ars_prompt"].split("맞으면", 1)[0]
        assert not re.search(r"[0-9공영일이삼사오육칠팔구]", head)            # 요청의 원문 필드·판단 입력은 가린 문장
        # 입력값은 저장 안 함. 되읽힌 원문은 묻는 문구(message)에만 사용, 자막에는 없음
        assert "01055554444" not in str(vars(s)) and "구해 3" not in str(s.captions)
        assert s.pending["message"] == BROKEN_READBACK
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "1"})
        assert call.keys[-1] == "1" and s.path[-1]["selected"]["label"] == "주민등록번호 확인"
    run(go())


def test_readback_question_joins_spaced_digit_groups():
    """되읽기 확인 질문에 보여 주는 원문은 숫자 묶음 사이 공백 제거 (STT 가 ARS 의 쉼마다 공백을 넣어
    통화마다 끊기는 위치가 다름). 확인 표현 뒤('맞으면 1번 다시 입력하시려면 2번')는 그대로 유지. 시험 값은 가짜 번호."""
    from session import join_readback_digits

    assert join_readback_digits("입력하신 번호는 0101 2341234 번입니다. 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오.") == \
        "입력하신 번호는 01012341234 번입니다. 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오."
    assert join_readback_digits("입력하신 번호는 0 1 0 1 2 3 4 번입니다. 맞으시면 1 번") == "입력하신 번호는 0101234 번입니다. 맞으시면 1 번"
    assert join_readback_digits("010-1234 -5678 번입니다. 맞으면 1번") == "010-1234 -5678 번입니다. 맞으면 1번"   # 기호는 유지
    assert join_readback_digits("확인 표현 없는 문장 12 34") == "확인 표현 없는 문장 1234"

    async def go():
        s, call = await secure_pending()
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "01012341234"})
        text = "입력하신 번호는 0101 2341234 번입니다. 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오."
        await s.on_paragraph(text)
        assert s.pending["message"].startswith("입력하신 번호는 01012341234 번입니다.")
        assert "2341234" not in s.pending["ars_prompt"]                       # 판단·요청 원문 필드는 여전히 가린 문장
    run(go())


def test_readback_split_handles_early_confirm_word_and_other_confirm_phrases():
    """확인 표현이 번호보다 먼저 나오는 경우('…맞는지 확인합니다' → 번호 → '맞으면 1번')나 목록에 없는 확인 표현('정확하면 1번')에도
    번호는 가리고 선택 키는 유지. 숫자 잇기도 같은 경계 사용. 시험 값은 가짜 번호."""
    from session import join_readback_digits, mask_readback, readback_options

    early = "입력하신 번호가 맞는지 확인합니다. 0101 2341234 번. 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오."
    m = mask_readback(early)
    assert "2341234" not in m and "0101" not in m and m.endswith("맞으면 1번 다시 입력하시려면 2번을 눌러주십시오.")
    assert join_readback_digits(early).startswith("입력하신 번호가 맞는지 확인합니다. 01012341234 번.")
    other = "입력하신 번호는 0101 2341234 번입니다. 정확하면 1번, 다시 입력하시려면 2번을 눌러주십시오."
    m = mask_readback(other)
    assert "2341234" not in m and [o.key for o in readback_options(m)] == ["1", "2"]
    # 선택지가 번호보다 먼저 나오거나 되읽기가 되풀이돼도 깨진 형태의 숫자까지 모두 가리고 선택 키만 유지
    assert "2341234" not in mask_readback("확인하시려면 1번, 입력하신 번호는 0101 2341234 번입니다.")
    first = "정확하면 1번, 다시 입력하시려면 2번. 입력하신 번호는 7. 영 5. 영 구해 8. 영 번입니다."
    m = mask_readback(first)
    assert not re.search(r"[0-9공영일이삼사오육칠팔구](?!번)", m.replace("1번", "").replace("2번", ""))
    assert [o.key for o in readback_options(m)] == ["1", "2"]
    twice = ("입력하신 번호는 7. 영 5. 영 구해 8. 영 입니다 정확하시면 1번 다시 입력하시려면 2번 "
             "입력하신 번호는 7. 영 5. 영 구해 8. 영 입니다 정확하시면 1번")
    m = mask_readback(twice)
    assert "7." not in m and "영 5" not in m and "구해" not in m and m.count("1번") == 2


def test_free_text_answer_to_readback_question_gives_ai_masked_question():
    """되읽기 확인 질문에 사용자가 버튼 대신 글로 답하면 AI(merge_answer)에는 숫자를 가린 문장 전달.
    사용자 화면의 문구(message)에는 숫자 포함. 시험 값은 가짜 번호."""
    async def go():
        s, call = await secure_pending()
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "01012341234"})
        await s.on_paragraph("입력하신 번호는 0101 2341234 번입니다. 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오.")
        assert "01012341234" in s.pending["message"]
        seen = []
        orig = s.d.interp.merge_answer

        async def spy(goal, question, answer):
            seen.append(question)
            return await orig(goal, question, answer)

        s.d.interp.merge_answer = spy
        await s.action({"type": "answer", "request_id": s.pending["request_id"], "text": "번호를 다시 입력할게요"})
        assert seen and not re.search(r"[0-9]{4,}", seen[0]) and "맞으면 1번" in seen[0]
    run(go())


def test_plain_confirm_question_outside_secure_window_is_not_readback():
    """민감정보를 입력하지 않은 상태의 '맞으면 1번, 다시 입력하시려면 2번'은 되읽기로 보지 않음 (자막·판단 그대로)."""
    async def go():
        s, call = await secure_pending()
        s.secure_window = False
        text = "고객님 휴대폰으로 받으시겠습니까. 맞으면 1번, 다시 입력하시려면 2번을 눌러 주십시오."
        await s.on_paragraph(text)
        assert s.captions and s.captions[-1]["text"] == text
    run(go())



def test_short_segments_and_tiny_results_are_not_captioned():
    """키 전송(DTMF)에 끊긴 짧은 조각은 인식 생략, 인식돼도 두 글자 이하면 폐기 ("콩" 같은 잡음 자막 방지)."""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=ScriptSTT(["콩", "KB국민은행입니다."]), paragraph_silence_ms=200,
                               clarify_expire_s=5, clarify_drop_s=10, min_voiced_s=0.4)
        await speak(s, 0.2)                                                      # 0.2초 조각: 인식 생략
        await asyncio.sleep(0.3)
        assert s.captions == []
        await speak(s, 1.0)                                                      # 1초 단락: 인식하되 결과 "콩"은 폐기
        await asyncio.sleep(0.3)
        assert s.captions == []
        await speak(s, 1.0)
        await wait_for(s, lambda s: len(s.captions) == 1)
        assert s.captions[0]["text"] == "KB국민은행입니다."
        await s.action({"type": "cancel"})
    run(go())



def test_info_cut_by_key_press_is_not_captioned():
    """키 전송 직후(post_key_caption_s 안)에 끝난 일반 안내는 키에 잘린 조각으로 보고 자막 전송 없음.
    메뉴처럼 다음 단계가 분명한 안내는 그대로 자막 전송."""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10, post_key_caption_s=3.0)
        await s.on_paragraph(ROOT)                                   # 잔액조회(3) 전송
        n = len(s.captions)
        await s.on_paragraph("이번에 입력해 주십시오")                  # 키에 잘린 조각 → 자막 없음
        assert len(s.captions) == n
        await s.on_paragraph(BAL)                                    # 다음 메뉴는 자막
        assert len(s.captions) == n + 1
    run(go())



def test_confirm_prompts_do_not_count_as_revisits_and_loop_can_be_disabled():
    """확인 질문이 여러 번 같은 모양으로 나와도 반복으로 집계 안 함. MAX_REVISIT=0 이면 반복 판정 끔."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10, max_revisit=0, max_presses=0)
        for _ in range(6):
            await s.on_paragraph(ROOT)      # 같은 메뉴에 여섯 번 도착해도 재질문 없음
        assert s.pending is None and call.keys == ["5"] * 6
    run(go())


def test_sibling_choices_are_asked_even_when_model_is_confident():
    """'신용·체크카드 1번 / 현금카드 2번' 처럼 같은 대상의 종류를 고르는 안내에서 목적에 구분이 없으면
    모델 점수가 높아도 사용자 재질문 (구분 없는 '카드' 목적에 한쪽 종류를 높은 점수로 바로 누르는 오동작 방지)."""
    from navigator import sibling_choices
    from interpreter import Option, _norm

    a, b = Option("1", "신용 체크카드"), Option("2", "현금카드")
    assert [o.key for o in sibling_choices("카드 분실 신고", a, [b])] == ["2"]
    assert sibling_choices("신용카드 분실 신고", a, [b]) == []
    assert sibling_choices("통장 잔액 알고 싶어", Option("3", "잔액조회"), [Option("4", "거래내역조회")]) == []
    # 선택지 이름의 조사('…으로')는 떼고 비교 (목적이 '다른 은행' 을 말했는데도 묻는 경우 방지)
    other, own = Option("2", "다른은행으로 송금"), Option("1", "하나은행으로 송금")
    assert sibling_choices("다른 은행 계좌로 돈 보내고 싶어", other, [own]) == []
    assert [o.key for o in sibling_choices("송금하고 싶어", other, [own])] == ["1"]

    menu = "신용 체크카드는 1번, 현금카드는 2번, 다른 서비스는 샵 버튼을 눌러 주십시오."

    async def go():
        s, call = await manual("카드 분실 신고하고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        s.d.interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": [
            {"key": "1", "label": "신용 체크카드"}, {"key": "2", "label": "현금카드"}, {"key": "#", "label": "다른 서비스"}]})
        s.d.interp.goals["카드 분실 신고하고 싶어"] = {"match": {"신용 체크카드": 0.9, "현금카드": 0.1, "다른 서비스": 0.0}}
        await s.on_paragraph(menu)
        assert call.keys == [] and s.pending and s.pending["reason"] == "ambiguous"
        # 같은 종류(신용 체크카드 / 현금카드) 뒤에 같은 메뉴의 나머지 업무 선택지도 선택지로 추가 (문구는 같은 종류끼리 질문)
        assert [o["key"] for o in s.pending["options"]] == ["1", "2", "#"]
        assert s.pending["message"].startswith("신용 체크카드 / 현금카드 중 어느 쪽인가요?")
        assert "다른 서비스도 고를 수 있습니다" in s.pending["message"]
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "2"})
        assert call.keys == ["2"]
    run(go())


def test_sibling_ask_keeps_other_business_options_of_the_menu():
    """같은 종류 재질문(서식 6개)에서 머리말이 다른 같은 메뉴의 업무 선택지('재증명 신청 7번')도 버튼으로 선택 가능
    ('… 서식 중 어느 쪽인가요?' 질문에서 나머지 업무 선택지가 빠지는 문제 방지). 이동 키는 따로 추가."""
    from interpreter import Option, Prompt, _norm
    from navigator import Layer, sibling_message

    menu = ("서식 팩스 전송 서비스 신청입니다. 근로자 자격관리 서식은 1번, 보험가입지원 서식은 2번, 보험료 부과고지 서식은 3번, "
            "요양보상 서식은 4번. 재활요양비 서식은 5번, 임금 채권 서식은 6번, 재증명 신청은 7번을 눌러주십시오. "
            "다시 듣기를 원하시면 별표, 처음으로 돌아가시려면 우물정자를 눌러주십시오.")
    names = ["근로자 자격관리 서식", "보험가입지원 서식", "보험료 부과고지 서식", "요양보상 서식", "재활요양비 서식", "임금 채권 서식",
             "재증명 신청"]
    goal = "대리인 선임 해임 신고서 팩스로 받을래"

    async def go():
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        s.d.interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": [
            {"key": str(i + 1), "label": n} for i, n in enumerate(names)], "repeat_key": "*", "home_key": "#"})
        s.d.interp.goals[goal] = {"match": {names[0]: 0.8, names[1]: 0.6, names[6]: 0.2}}
        s.layers.append(Layer(Prompt(kind="menu", text="첫 메뉴", options=[Option("2", "서식 팩스")]), "2", "ai"))
        await s.on_paragraph(menu)
        assert call.keys == [] and s.pending and s.pending["reason"] == "ambiguous"
        keys = [o["key"] for o in s.pending["options"]]
        assert keys[:6] == ["1", "2", "3", "4", "5", "6"] or set(keys[:6]) == {"1", "2", "3", "4", "5", "6"}
        assert "7" in keys and keys.index("7") == 6     # 나머지 업무 선택지는 같은 종류 뒤에
        assert keys[-1] == "#"                          # 한 단계 들어온 메뉴의 재질문: 처음으로는 따로 한 번
        msg = s.pending["message"]
        assert "재증명 신청" not in msg.split("중 어느 쪽인가요?")[0]
        assert "재증명 신청도 고를 수 있습니다" in msg
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "7"})
        assert call.keys == ["7"]
    run(go())

    # 나머지가 많으면 이름을 모두 나열하지 않음
    twins = [Option("1", "가 서식"), Option("2", "나 서식")]
    rest = [Option(str(i), f"업무{i}") for i in range(3, 8)]
    assert sibling_message(twins, rest) == "가 서식 / 나 서식 중 어느 쪽인가요? 그 밖의 선택지도 아래에서 고를 수 있습니다. 원하시는 업무를 말씀해 주세요."
    assert sibling_message(twins, []) == "가 서식 / 나 서식 중 어느 쪽인가요? 원하시는 업무를 말씀해 주세요."


def test_no_answer_after_dial_timeout_hangs_up():
    """상대가 받지 않고 통신망이 아무 신호도 주지 않으면 DIAL_TIMEOUT_S 뒤에 끊고 no_answer 로 종료
    ('울림' 뒤 '전화를 걸고 있습니다' 상태로 오래 머무는 경우 방지). 받으면 제한 시간 취소."""
    from interpreter import MockInterpreter
    from navigator import Navigator
    from session import Deps
    from stt import NullRecognizer

    async def go(answer: bool):
        st = make_settings(dial_timeout_s=0.2)
        interp = MockInterpreter()
        backend = ManualBackend()
        hung = []
        deps = Deps(st, backend, NullRecognizer(), interp, Navigator(interp))
        m = SessionManager(deps)
        out = await m.create("15881300", "업무시간 알고 싶어")
        s = m.sessions[out["session_id"]]
        backend.call.hangup = lambda: hung.append(1) or asyncio.sleep(0)
        if answer:
            await s._on_answer()
        await asyncio.sleep(0.4)
        return s, hung

    s, hung = run(go(False))
    assert s.status == "ended" and s.end_reason == "no_answer" and hung
    s, hung = run(go(True))
    assert s.status == "listening" and s.end_reason is None and not hung


def test_late_answer_after_dial_timeout_is_hung_up():
    """발신 제한 시간으로 끝난 뒤 상대가 늦게 받으면 통화를 되살리지 않고 바로 끊음 (max_duration 타이머·음수 통화 시간 없음)."""
    from interpreter import MockInterpreter
    from navigator import Navigator
    from session import Deps
    from stt import NullRecognizer

    async def go():
        st = make_settings(dial_timeout_s=0.1)
        interp = MockInterpreter()
        backend = ManualBackend()
        deps = Deps(st, backend, NullRecognizer(), interp, Navigator(interp))
        m = SessionManager(deps)
        out = await m.create("15881300", "업무시간 알고 싶어")
        s = m.sessions[out["session_id"]]
        hung = []
        backend.call.hangup = lambda: hung.append(1) or asyncio.sleep(0)
        await asyncio.sleep(0.3)
        assert s.end_reason == "no_answer" and len(hung) == 1
        await s._on_answer()                                   # 늦은 응답
        assert s.status == "ended" and s.answered_at is None and "max_duration" not in s._timers and len(hung) == 2
        await s._on_audio(bytes([0xFF]) * 160)                 # 종료된 통화는 소리를 쌓지 않음
    run(go())


def test_clawops_hangup_cancels_ringing_call_and_closes_media_after_answer():
    """울리는 중에는 SDK hangup 이 동작하지 않으므로 REST 로 통화 종료 요청. 그 뒤 늦게 받으면 미디어 종료."""
    import types
    from telephony import ClawOpsBackend, ClawOpsCall

    sent = []

    class FakeBackend:
        async def cancel(self, call_id):
            sent.append(("cancel", call_id))

    async def sdk_hangup():
        sent.append(("media",))

    session = types.SimpleNamespace(call_id="CA_ring", ended_status=None, hangup=sdk_hangup)
    c = ClawOpsCall(session, handlers=None, backend=FakeBackend())
    run(c.hangup())
    run(c.hangup())                                            # 두 번 호출해도 한 번만 전송
    assert sent == [("cancel", "CA_ring")] and c._hung_up
    ClawOpsBackend.LEGS["CA_ring"] = types.SimpleNamespace(call=object())   # 늦게 받아 미디어가 붙음
    try:
        run(c.hangup())
        run(c.hangup())
        assert sent == [("cancel", "CA_ring"), ("media",)]
    finally:
        ClawOpsBackend.LEGS.pop("CA_ring", None)


def test_result_after_secure_input_with_input_word_is_not_dropped():
    """민감정보 입력 직후 '입력하신 번호로 … 발송했습니다' 처럼 되읽기 표현이 섞인 결과 안내도 결과로 처리
    (되읽기로만 보고 결과를 버려 요약 없이 failed 로 끝나는 경우 방지). 숫자는 가린 문장으로 처리. 시험 값은 가짜 번호."""
    from interpreter import _norm
    from session import mask_readback

    async def go():
        s, call = await secure_pending()
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "01012341234"})
        text = "입력하신 번호 01012341234 로 안내 문자를 발송했습니다. 확인 부탁드립니다."
        masked = mask_readback(text)
        s.d.interp.prompts[_norm(masked)] = (masked, {"kind": "result", "options": []})
        await s.on_paragraph(text)
        assert s.summary is not None and s.summary["kind"] == "result"
        assert "01012341234" not in str(s.captions) and "2341234" not in str(s.summary)
    run(go())


def test_user_can_go_back_from_submenu_question():
    """한 단계 이상 들어온 메뉴의 재질문에는 '이전 메뉴'·'처음으로' 를 선택지로 추가. 사용자가 '이전 메뉴'를 고르면
    돌아간 메뉴에서 방금 나온 갈래는 다시 누르지 않음 (잘못 들어간 하위 메뉴에서 돌아갈 방법이 없는 문제 방지)."""
    from interpreter import _norm

    top = "갑 업무는 1번, 을 업무는 2번을 눌러 주십시오."
    sub = "가 조회는 1번, 나 조회는 2번, 이전 메뉴는 샵 버튼을 눌러 주십시오."

    async def go():
        s, call = await manual("을 업무 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        s.d.interp.prompts[_norm(top)] = (top, {"kind": "menu", "options": [{"key": "1", "label": "갑 업무"}, {"key": "2", "label": "을 업무"}]})
        s.d.interp.prompts[_norm(sub)] = (sub, {"kind": "menu", "back_key": "#",
                                               "options": [{"key": "1", "label": "가 조회"}, {"key": "2", "label": "나 조회"}]})
        s.d.interp.goals["을 업무 알고 싶어"] = {"match": {"갑 업무": 0.9, "을 업무": 0.6, "가 조회": 0.4, "나 조회": 0.35}}
        await s.on_paragraph(top)
        assert call.keys == ["1"]                                   # AI 가 갑(0.9)을 누름 (잘못된 갈래)
        await asyncio.sleep(0.05)
        await s.on_paragraph(sub)
        req = s.pending
        assert req and req["reason"] == "low_confidence"
        assert [(o["key"], o["label"]) for o in req["options"]][-1] == ("#", "이전 메뉴")
        await s.action({"type": "select", "request_id": req["request_id"], "key": "#"})
        assert call.keys == ["1", "#"] and s.layers == [] and s.nav.exclude == {"1"}
        await asyncio.sleep(0.05)
        await s.on_paragraph(top)                                   # 돌아간 메뉴: 방금 나온 1 은 빼고 판단
        assert call.keys == ["1", "#", "2"]
    run(go())


def test_user_back_to_already_asked_menu_asks_again_and_keeps_rejections():
    """이미 물어본 메뉴로 사용자가 '이전 메뉴'·'처음으로'로 돌아오면 끝내지 않고 재질문 (거절한 갈래 제외).
    거절한 갈래는 AI 가 키를 눌러도 유지되어 자동 복귀·두 번째 거절 뒤 재진입 없음."""
    from interpreter import _norm

    top = "갑 업무는 1번, 을 업무는 2번, 병 업무는 3번을 눌러 주십시오."
    sub = "가 조회는 1번, 나 조회는 2번, 이전 메뉴는 샵 버튼, 처음으로는 별표를 눌러 주십시오."
    same = "다 조회는 1번, 라 조회는 2번, 이전 메뉴나 처음으로 가시려면 별표를 눌러 주십시오."

    async def setup(goal):
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        s.d.interp.prompts[_norm(top)] = (top, {"kind": "menu", "options": [
            {"key": "1", "label": "갑 업무"}, {"key": "2", "label": "을 업무"}, {"key": "3", "label": "병 업무"}]})
        s.d.interp.prompts[_norm(sub)] = (sub, {"kind": "menu", "back_key": "#", "home_key": "*",
                                               "options": [{"key": "1", "label": "가 조회"}, {"key": "2", "label": "나 조회"}]})
        s.d.interp.prompts[_norm(same)] = (same, {"kind": "menu", "back_key": "*", "home_key": "*",
                                                 "options": [{"key": "1", "label": "다 조회"}, {"key": "2", "label": "라 조회"}]})
        s.d.interp.goals[goal] = {"match": {"갑 업무": 0.1, "을 업무": 0.0, "병 업무": 0.2, "가 조회": 0.1, "나 조회": 0.0}}
        return s, call

    async def go(nav_key):
        s, call = await setup("정 업무 알고 싶어")
        await s.on_paragraph(top)                               # 모두 0.3 미만 → 종료 전 재질문
        req = s.pending
        assert req and req["reason"] == "low_confidence" and not s.ended
        await s.action({"type": "select", "request_id": req["request_id"], "key": "3"})
        await asyncio.sleep(0.05)
        await s.on_paragraph(sub)
        req = s.pending
        assert req and {o["label"] for o in req["options"]} >= {"이전 메뉴", "처음으로"}
        await s.action({"type": "select", "request_id": req["request_id"], "key": nav_key})
        await asyncio.sleep(0.05)
        await s.on_paragraph(top)                               # 돌아온 첫 메뉴: 종료 대신 재질문
        req = s.pending
        assert not s.ended and req and req["reason"] == "low_confidence"
        assert "3" not in [o["key"] for o in req["options"]] and s.nav.rejected
        return s, call

    for k in ("#", "*"):
        run(go(k))

    async def dedupe():
        from interpreter import Prompt
        from navigator import Layer

        s, call = await setup("정 업무 알고 싶어")
        s.layers = [Layer(Prompt(kind="menu", text=top), "1", "ai")]   # 한 단계 들어온 상태
        s.d.interp.goals["정 업무 알고 싶어"]["match"].update({"다 조회": 0.4, "라 조회": 0.35})
        await s.on_paragraph(same)
        keys = [o["key"] for o in s.pending["options"]]
        assert keys.count("*") == 1                             # 이전 메뉴·처음으로가 같은 키면 한 번만
    run(dedupe())


def test_clarification_never_offers_a_single_option():
    """선택지 하나뿐인 재질문 방지 (예: '업무시간 및 각종 안내 중 어느 쪽인가요?' 처럼 고를 수 없는 질문).
    1·2순위 차이는 반올림해 비교 (0.6−0.5 가 0.0999… 로 계산되는 소수 오차 대응), 후보가 둘 미만이면 점수 순으로 추가 표시."""
    from interpreter import Option, Prompt
    from navigator import Navigator, NavState

    class Fixed:
        def __init__(self, scores):
            self.scores = scores

        async def score(self, goal, options, path=None):
            return {o.key: self.scores.get(o.key, 0.0) for o in options}

    opts = [Option("1", "반품 접수"), Option("2", "국내 등기 조회"), Option("3", "국제 등기 조회"),
            Option("5", "업무시간 및 각종 안내"), Option("7", "보이는 ARS 안내")]
    menu = Prompt(kind="menu", text="메뉴", options=opts)

    def decide(scores):
        return run(Navigator(Fixed(scores)).decide(NavState("목적"), menu, [], True))

    d = decide({"5": 0.6, "2": 0.5})                       # 차이 0.1 = 기준과 같음 → 누름 (소수 오차로 묻지 않음)
    assert d.action == "press" and d.key == "5"
    d = decide({"5": 0.55, "2": 0.5})                      # 차이 0.05 → 재질문, 2위가 0.5 미만이어도 함께 표시
    assert d.action == "ask" and d.reason == "ambiguous" and [o.key for o in d.options] == ["5", "2"]
    d = decide({"5": 0.6, "2": 0.45})
    assert d.action == "press"
    d = decide({"2": 0.4, "5": 0.2, "1": 0.1})             # 확신 없음: 0.3 이상이 하나뿐 → 점수 순으로 5개까지
    assert d.action == "ask" and d.reason == "low_confidence" and [o.key for o in d.options][:3] == ["2", "5", "1"]
    assert len(d.options) == 5


# 동점 묶음 시험용 첫 메뉴 키 구성 (선택지 이름은 시험용): 선택지 8개,
# '*'·'#' 도 업무 선택지, 상담원 연결이 목록 중간. 목록 앞 5개만 자르면 '*·8·0·#·3' 만 남는 구성
TIE_OPTS = [("*", "사기 피해 신고", None), ("8", "청년 적금", None), ("0", "상담직원 연결", "상담원 연결"),
            ("#", "빠른 안내 서비스", None), ("3", "잔액 조회", None), ("4", "거래내역 조회", None),
            ("5", "카드 및 기타 서비스", None), ("7", "외화 업무", None)]


def test_clarification_shows_whole_tie_group():
    """확신 없음 재질문을 점수 순 5개로 자를 때 5번째와 점수 차이가 1·2위 차이 기준(0.1) 미만인 선택지는 모두 표시
    (모든 선택지가 같은 점수일 때 목록 앞 5개만 보여 정답 선택지가 빠지는 문제 방지). 0점 묶음도 같음.
    같은 점수면 업무 선택지를 상담원 연결보다 앞에 배치."""
    from interpreter import Option, Prompt
    from navigator import NOT_FOUND_ASK, Navigator, NavState, shown_options

    class Fixed:
        def __init__(self, scores):
            self.scores = scores

        async def score(self, goal, options, path=None):
            return {o.key: self.scores.get(o.key, 0.0) for o in options}

    opts = [Option(k, label, cat) for k, label, cat in TIE_OPTS]
    menu = Prompt(kind="menu", text="메뉴", options=opts)

    def decide(scores, st=None):
        return run(Navigator(Fixed(scores)).decide(st or NavState("목적"), menu, [], True))

    def keys(d):
        return [o.key for o in d.options]

    # 모두 0.4: 8개 모두 표시. 순서는 선택지 목록 순서, 단 상담원 연결은 같은 점수의 업무 선택지 뒤
    d = decide({o.key: 0.4 for o in opts})
    assert d.action == "ask" and d.reason == "low_confidence"
    assert keys(d) == ["*", "8", "#", "3", "4", "5", "7", "0"]

    # 5·6번째가 같은 점수, 7번째부터 낮음 → 6개 (뒤의 0.1·0 은 미표시)
    d = decide({"*": 0.45, "8": 0.4, "#": 0.3, "3": 0.3, "4": 0.3, "5": 0.3, "7": 0.1})
    assert d.reason == "low_confidence" and keys(d) == ["*", "8", "#", "3", "4", "5"]

    # 5번째 아래가 모두 0.1 이상 낮으면 기본대로 5개
    d = decide({"*": 0.45, "8": 0.4, "#": 0.4, "3": 0.35, "4": 0.3, "5": 0.2})
    assert keys(d) == ["*", "8", "#", "3", "4"]

    # 0 이 아닌 점수가 5개 미만이면 5번째가 0점이라 0점 선택지 전체가 한 묶음 (흔한 모양). 점수 받은 선택지가 앞,
    # 0점끼리는 목록 순서, 상담원 연결은 맨 뒤. 해석기가 점수를 받지 못한 선택지에도 0 을 주므로 0점이라는 이유로 제외 안 함
    d = decide({"*": 0.4, "5": 0.2, "7": 0.1})
    assert d.reason == "low_confidence" and keys(d) == ["*", "5", "7", "8", "#", "3", "4", "0"]

    # 맞는 선택지 없음(모두 0, 첫 계층): 끝내기 전에 묻는 질문도 같은 점수 묶음 전체 (상담원 연결은 맨 뒤)
    d = decide({})
    assert d.reason == "low_confidence" and d.message == NOT_FOUND_ASK
    assert keys(d) == ["*", "8", "#", "3", "4", "5", "7", "0"]

    # 사용자가 갈래를 모두 거절한 뒤: 점수가 없으므로 메뉴 전체를 목록 순서대로 표시 (앞 5개로 자르지 않음)
    st = NavState("목적")
    st.rejected[menu.signature()] = {o.key for o in opts}
    d = decide({}, st)
    assert d.reason == "low_confidence" and keys(d) == [o.key for o in opts]

    # 1위 동점(0.5 이상)은 차이 재질문, 업무 선택지가 상담원 연결보다 앞
    d = decide({"0": 0.6, "5": 0.6})
    assert d.reason == "ambiguous" and keys(d) == ["5", "0"]
    # 순서 규칙은 누를 키에 영향 없음: 차이가 기준 이상이면 상담원 연결도 그대로 누름(목적에 상담원 요청이 있을 때)
    # 목적에 상담원 요청이 없으면 상담원 연결이 1위여도 한 번 질문 (원치 않는 상담원 연결 방지)
    d = decide({"0": 0.7, "5": 0.6}, NavState("상담원 연결해줘"))
    assert d.action == "press" and d.key == "0"
    d = decide({"0": 0.7, "5": 0.6})
    assert d.action == "ask" and d.reason == "ambiguous" and keys(d)[0] == "0"

    # 이어지는 작은 차이는 5번째 점수와만 비교 (0.44 기준: 차이 0.1 미만인 0.35 까지 포함, 0.34 부터 제외)
    ranked = [(Option(str(i), f"선택지 {i}"), s)
              for i, s in enumerate([0.48, 0.47, 0.46, 0.45, 0.44, 0.40, 0.35, 0.34, 0.33])]
    assert [o.key for o in shown_options(ranked, 0.1)] == ["0", "1", "2", "3", "4", "5", "6"]
    flat = [(Option(str(i), f"선택지 {i}"), 0.4) for i in range(3)]
    assert len(shown_options(flat, 0.1)) == 3              # 5개 이하면 그대로


def test_low_confidence_message_names_an_option_only_when_it_leads():
    """확신 없음 문구("'…' 메뉴가 맞는지 확실하지 않습니다")는 1위가 홀로 가장 높을 때만 그 이름 표시. 1·2위가 같은 점수면
    목록 맨 앞은 목록 순서일 뿐이므로 이름 미표시 (모두 같은 점수인 메뉴에서 목록 맨 앞 이름을 대는 문제 방지).
    선택지가 하나면 그 이름 표시."""
    from interpreter import Option, Prompt, _norm

    menu = "첫 메뉴입니다. " + ", ".join(f"{label}는 {k}번" for k, label, _ in TIE_OPTS) + "을 눌러 주십시오."
    options = [{"key": k, "label": label, **({"category": cat} if cat else {})} for k, label, cat in TIE_OPTS]

    async def ask(match: dict[str, float]):
        s, call = await manual("카드 발급 방법 알려줘", clarify_expire_s=5, clarify_drop_s=10)
        s.d.interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": options})
        s.d.interp.goals["카드 발급 방법 알려줘"] = {"match": match}
        await s.on_paragraph(menu)
        assert call.keys == [] and s.pending and s.pending["reason"] == "low_confidence"
        return s.pending

    async def go():
        tie = {(cat or label): 0.4 for _, label, cat in TIE_OPTS}
        req = await ask(tie)
        keys = [o["key"] for o in req["options"]]
        assert keys == ["*", "8", "#", "3", "4", "5", "7", "0"]
        assert "메뉴가 맞는지" not in req["message"] and "사기 피해 신고" not in req["message"]
        req = await ask({**tie, "카드 및 기타 서비스": 0.45})
        assert req["options"][0]["key"] == "5" and "'카드 및 기타 서비스' 메뉴가 맞는지" in req["message"]

        # 선택지 하나짜리 확신 없음 질문 (다른 갈래를 모두 거절한 메뉴에서 발생): 두 번째 선택지가 없어도 이름 표시
        s, _ = await manual("카드 발급 방법 알려줘", clarify_expire_s=5, clarify_drop_s=10)
        only = Option("5", "카드 및 기타 서비스")
        await s._ask(Prompt(kind="menu", text=menu, options=[only]), "low_confidence", [only], scores={"5": 0.4})
        assert "'카드 및 기타 서비스' 메뉴가 맞는지" in s.pending["message"]
    run(go())


def test_only_other_choice_is_agent_is_asked():
    """선택지가 '하나 / 상담원 연결' 뿐이고 목적에 그 하나를 가리키는 말이 없으면 모델 점수가 높아도 사용자 재질문
    (목적과 무관한 단일 업무 선택지를 높은 점수로 바로 누르는 오동작 방지)."""
    from navigator import agent_alternative
    from interpreter import Option, _norm

    sms, agent = Option("1", "모바일 홈페이지 문자 수신"), Option("0", "상담원 연결", "상담원 연결")
    assert agent_alternative("카드 발급 방법 알려줘", sms, [sms, agent]) is agent
    assert agent_alternative("문자로 받고 싶어", sms, [sms, agent]) is None             # 목적이 문자 수신을 가리킴
    assert agent_alternative("ARS로 바로 주문", Option("1", "ARS 주문"), [Option("1", "ARS 주문"), Option("2", "상담원 주문")]) is None
    assert agent_alternative("카드 발급 방법 알려줘", sms, [sms, agent, Option("#", "다른 서비스")]) is None  # 선택지가 셋
    assert agent_alternative("카드 발급 방법 알려줘", sms, [sms, Option("2", "카드 발급 신청 문의")]) is None  # 상담원 아님
    assert agent_alternative("상담원 연결", agent, [sms, agent]) is None                 # 상담원을 고른 경우는 그대로
    # '서비스·안내·문의' 같은 일반 낱말만 겹치면 가리키는 말로 보지 않음
    assert agent_alternative("발급 문의", Option("1", "발급 안내 문자 서비스"), [Option("1", "발급 안내 문자 서비스"), agent]) is agent
    # 두 선택지가 함께 가진 낱말('카드', '주문')은 best 를 가리키는 말 아님
    card_sms, card_agent = Option("1", "카드 발급 문자 안내"), Option("0", "카드 발급 상담원 연결")
    assert agent_alternative("카드 발급 방법 알려줘", card_sms, [card_sms, card_agent]) is card_agent
    ars, ars_agent = Option("1", "ARS 주문 접수"), Option("2", "상담원 주문 접수")
    assert agent_alternative("피자 주문하고 싶어", ars, [ars, ars_agent]) is ars_agent
    assert agent_alternative("ars로 할게요", ars, [ars, ars_agent]) is None             # 대소문자 무시
    assert agent_alternative("카드 발급 문자로 받을래", card_sms, [card_sms, card_agent]) is None

    menu = "카드 발급 문의 서비스입니다. 모바일 홈페이지 문자 수신을 원하시면 1번, 상담원 연결을 원하시면 0번을 눌러주십시오."

    async def go():
        s, call = await manual("카드 발급 방법 알려줘", clarify_expire_s=5, clarify_drop_s=10)
        s.d.interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": [
            {"key": "1", "label": "모바일 홈페이지 문자 수신"}, {"key": "0", "label": "상담원 연결", "category": "상담원 연결"}]})
        s.d.interp.goals["카드 발급 방법 알려줘"] = {"match": {"모바일 홈페이지 문자 수신": 0.8, "상담원 연결": 0.5}}
        await s.on_paragraph(menu)
        assert call.keys == [] and s.pending and s.pending["reason"] == "ambiguous"
        assert [o["key"] for o in s.pending["options"]] == ["1", "0"]
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "1"})
        assert call.keys == ["1"]
    run(go())


def test_repeated_input_prompt_while_typing_keeps_request_and_late_repeat_after_send_is_ignored():
    """전화번호처럼 긴 입력을 누르는 동안 ARS 가 입력 안내를 되풀이하면 요청 유지(같은 request_id 로 만료만 연장).
    입력 전송 뒤 뒤늦게 인식된 되풀이 안내로는 새 입력 요청 생성 안 함 (같은 번호 재입력으로 오류가 반복되는 문제 방지)."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.on_paragraph(RRN)                                    # 입력 중 ARS 가 같은 안내를 되풀이
        assert s.pending["request_id"] == req["request_id"]          # 요청 유지
        t_send = time.monotonic()
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "950101"})
        assert call.keys[-1].startswith("950101")
        await s.on_paragraph(RRN, started_at=t_send - 1.0)           # 입력 전에 시작된 되풀이 안내가 뒤늦게 도착
        assert s.pending is None                                     # 새 입력 요청 없음
        await s.on_paragraph(RRN, started_at=t_send + 1.0)           # 입력 뒤에 ARS 가 실제로 다시 요구하면 새 요청
        assert s.pending and s.pending["kind"] == "secure_input"
    run(go())



def test_no_answer_keeps_request_by_default_instead_of_calling_agent():
    """기본값: 재질문에 답이 없어도 상담원 연결 키를 누르지 않고 요청을 유지한 채 계속 청취."""
    async def go():
        m = await new_manager()
        _, s = await start(m, "카드 관련 문의")
        await wait_for(s, lambda s: s.pending)
        rid = s.pending["request_id"]
        await asyncio.sleep(0.9)                                      # clarify_drop(0.6초) 경과
        assert s.pending and s.pending["request_id"] == rid and s.path == [] and s.status == "awaiting_user"
        await s.action({"type": "cancel"})
    run(go())


def test_plain_wait_notice_is_not_agent_wait():
    """"잠시만 기다려 주십시오" 만으로는 상담원 대기 아님 (조회 대기). 상담원 언급이 있어야 agent_wait 분류."""
    from gemini_interpreter import rule_classify
    assert rule_classify("잠시만 기다려 주십시오.").kind == "info"
    assert rule_classify("확인 중입니다. 잠시만 기다려 주십시오.").kind == "info"
    assert rule_classify("상담사 연결 중입니다. 잠시만 기다려 주십시오.").kind == "agent_wait"
    assert rule_classify("카드 ARS로 연결해 드리겠습니다.").kind == "info"


# ---------------------------------------------------------------------------
# 시험용 통화 기록 (record=true)
# ---------------------------------------------------------------------------
def _goertzel(samples, hz: float, rate: int = 8000) -> float:
    import math
    k = 2 * math.cos(2 * math.pi * hz / rate)
    a = b = 0.0
    for x in samples:
        a, b = x + k * a - b, a
    return a * a + b * b - k * a * b


def test_recording_keeps_no_secure_digits():
    """통화 기록에 민감정보 입력값 없음: 로그에는 자릿수만, 소리에는 DTMF 대신 1kHz 대체 신호음."""
    import io, json, zipfile
    from audio import SILENCE_FRAME

    async def go():
        s, call = await secure_pending(record=True)
        for _ in range(50):
            await s._on_audio(SILENCE_FRAME)                         # 통화 소리 1초
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "991332"})
        await s.finish("user_canceled")
        z = zipfile.ZipFile(io.BytesIO(s.recorder.export_zip()))
        assert set(z.namelist()) == {"call.wav", "log.json", "log.txt"}
        blob = z.read("log.json").decode() + z.read("log.txt").decode()
        assert "991332" not in blob
        log_ = json.loads(z.read("log.json"))
        acts = log_["actions"]
        assert [a.get("keys") for a in acts if a["type"] == "key"] == ["3", "2"]
        sec = [a for a in acts if a["type"] == "input"]
        assert len(sec) == 1 and sec[0]["digits"] is None and sec[0]["digit_count"] == 6 and sec[0]["actor"] == "user"
        tone = s.recorder.tones[-1][1][:1200]                        # 민감정보 입력 자리의 소리
        assert _goertzel(tone, 1000) > 100 * max(_goertzel(tone, f) for f in (697, 770, 852, 941, 1209, 1336, 1477))
        assert any(e["type"] == "event:request" and e["data"]["kind"] == "secure_input" for e in log_["events"])
        assert log_["wav"] == "call.wav" and log_["seconds"] >= 1.0
    run(go())


def test_recording_rest_owner_only_and_deletable():
    """기록 내려받기는 record=true 통화의 토큰 소유자만, 통화 종료 뒤에만 가능. 삭제 뒤에는 내려받기 불가."""
    import io, zipfile
    from fastapi.testclient import TestClient

    st = make_settings(sim_speed=0.05)
    with TestClient(create_app(st)) as c:
        def make(record):
            r = c.post("/v1/sessions", json={"phone_number": "15889999", "goal": "카드 결제일 바꾸고 싶어", "record": record},
                       headers={"X-App-Key": st.app_key})
            assert r.status_code == 201, r.text
            return r.json()["session_id"], {"Authorization": f"Bearer {r.json()['session_token']}"}

        sid, h = make(True)
        assert c.get(f"/v1/sessions/{sid}/recording", headers=h).status_code == 409        # 통화 중
        assert c.get(f"/v1/sessions/{sid}/recording", headers={"Authorization": "Bearer x"}).status_code == 401
        c.post(f"/v1/sessions/{sid}/actions", json={"type": "cancel"}, headers=h)
        r = c.get(f"/v1/sessions/{sid}/recording", headers=h)
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
        assert "call.wav" in zipfile.ZipFile(io.BytesIO(r.content)).namelist()
        # 시험 설정에서 디스크 저장 꺼짐 (make_settings) → 디스크 사본 없음
        assert c.delete(f"/v1/sessions/{sid}/recording", headers=h).json() == {"deleted": True, "disk_copy_kept": False}
        assert c.get(f"/v1/sessions/{sid}/recording", headers=h).status_code == 404

        sid, h = make(False)                                         # 기록을 켜지 않은 통화
        c.post(f"/v1/sessions/{sid}/actions", json={"type": "cancel"}, headers=h)
        assert c.get(f"/v1/sessions/{sid}/recording", headers=h).status_code == 404


def test_prefetch_min_chunk_comes_from_settings():
    """미리 인식 최소 조각 길이는 설정(STT_PREFETCH_MIN_S, 기본 5초) 기준 (짧은 조각은 낱말 경계 오인식이 많음)."""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt_prefetch_min_s=2.5)
        assert s.segmenter.min_chunk_ms == 2500
        from config import Settings
        assert Settings().stt_prefetch_min_s == 5.0 or "STT_PREFETCH_MIN_S" in __import__("os").environ
    run(go())


# 통화 기록 디스크 저장 (RECORDING_SAVE, RECORDING_DIR, RECORD_ALWAYS) -------------------
def test_recording_folder_name_shows_goal_number_org_time():
    from datetime import datetime
    from recording import KST, folder_name

    t = datetime(2026, 11, 4, 16, 37, 5, tzinfo=KST)
    assert folder_name("카드 발급/방법?", "1588-9999", t) == "카드발급방법_15889999_KB국민은행_20261104-163705"
    assert folder_name("잔액 조회", "0212345678", t) == "잔액조회_0212345678_20261104-163705"   # 모르는 번호는 기관 생략
    assert folder_name("", "", t) == "목적없음_번호없음_20261104-163705"


def test_recording_is_saved_to_disk_when_call_ends(tmp_path):
    """기록을 켠 통화 종료 시 RECORDING_DIR/<목적>_<번호>_<기관>_<시각>/ 에 call.wav·log.json·log.txt 저장.
    민감정보 입력값은 파일에도 없음 (자릿수만)."""
    async def go():
        s, call = await secure_pending(record=True, recording_save=True, recording_dir=str(tmp_path))
        req = s.pending
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "950101"})
        await s.finish("user_canceled")
        await s._save_task
        folders = list(tmp_path.iterdir())
        assert len(folders) == 1 and folders[0].name.startswith("통장잔액알고싶어_15889999_KB국민은행_")
        assert s.recording_path == str(folders[0])
        names = sorted(p.name for p in folders[0].iterdir())
        assert names == ["call.wav", "log.json", "log.txt"]
        text = (folders[0] / "log.txt").read_text(encoding="utf-8")
        assert "AI → 키 3" in text and "민감정보 6자리 (값 기록 안 함)" in text
        assert "950101" not in text and "950101" not in (folders[0] / "log.json").read_text(encoding="utf-8")
        s.drop_recording()                                              # 메모리 기록을 지워도 디스크 파일은 유지
        assert (folders[0] / "call.wav").exists()
    run(go())


def test_recording_not_saved_when_disabled_or_not_requested(tmp_path):
    async def go():
        s, _ = await manual("잔액 조회", record=True, recording_save=False, recording_dir=str(tmp_path))
        await s.finish("user_canceled")
        assert s._save_task is None and list(tmp_path.iterdir()) == []
        s2, _ = await manual("잔액 조회", record=False, recording_save=True, recording_dir=str(tmp_path / "b"))
        await s2.finish("user_canceled")
        assert s2.recorder is None and not (tmp_path / "b").exists()
    run(go())


def test_record_always_records_without_request(tmp_path):
    """RECORD_ALWAYS=1 이면 앱이 record 를 보내지 않아도 기록 후 디스크에 저장."""
    async def go():
        s, _ = await manual("잔액 조회", record=False, record_always=True, recording_save=True, recording_dir=str(tmp_path))
        assert s.recorder is not None
        await s.finish("user_canceled")
        await s._save_task
        assert len(list(tmp_path.iterdir())) == 1
    run(go())


def test_same_folder_name_gets_suffix(tmp_path):
    from recording import save_files

    a = save_files(tmp_path, "x", {"log.txt": b"1"})
    b = save_files(tmp_path, "x", {"log.txt": b"2"})
    assert a.name == "x" and b.name == "x-2" and (b / "log.txt").read_bytes() == b"2"


def test_recording_save_failure_does_not_break_call_end(tmp_path):
    """저장 위치에 쓸 수 없어도(같은 이름의 파일 존재 등) 통화 종료는 정상 처리 (경고 로그만)."""
    blocker = tmp_path / "file"
    blocker.write_text("x")
    async def go():
        s, _ = await manual("잔액 조회", record=True, recording_save=True, recording_dir=str(blocker))
        await s.finish("user_canceled")
        await s._save_task
        assert s.status == "ended" and s.recording_path is None
    run(go())


def test_keys_typed_during_secure_request_are_not_recorded(tmp_path):
    """민감정보 입력 요청 중 키패드로 누른 숫자(409 로 거절)·형식이 틀린 키는 기록에 값 대신 자릿수만 남음."""
    async def go():
        s, call = await secure_pending(record=True, recording_save=True, recording_dir=str(tmp_path))
        req = s.pending
        with pytest.raises(ApiError):
            await s.action({"type": "key", "key": "9"})                  # 요청 중 key → 409 sensitive_input_blocked
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "950101"})
        with pytest.raises(ApiError):
            await s.action({"type": "key", "key": "95010112"})          # 형식이 틀린 키 (5자 이상)
        await s.finish("user_canceled")
        await s._save_task
        folder = next(tmp_path.iterdir())
        text = (folder / "log.txt").read_text(encoding="utf-8") + (folder / "log.json").read_text(encoding="utf-8")
        assert "사용자 응답 key 9" not in text and "95010112" not in text and "950101" not in text
        assert "사용자 응답 key 1자리" in text
    run(go())


def test_reprompt_judged_after_secure_send_keeps_readback_window():
    """시험 순서: 입력하는 동안 ARS 의 '입력시간이 경과하였습니다…' 재안내 → 안내가 끝나 서버가 번호 전송 →
    그 뒤 재안내의 판단(input_error) 완료. 전송 전에 시작된 안내로는 되읽기 구간을 닫지 않으므로 이어지는
    '입력하신' 없는 되읽기도 가려지고 사용자에게 확인 질문."""
    from interpreter import _norm

    async def go():
        s, call = await secure_pending()
        req = s.pending
        reprompt = "입력시간이 경과하였습니다. 주민등록번호 앞 여섯 자리와 우물정자를 눌러 주십시오."
        s.d.interp.prompts[_norm(reprompt)] = (reprompt, {"kind": "input_error"})
        call.speaking = True
        t_reprompt = time.monotonic() - 1.0                                   # 재안내 시작 (입력 1초 전. Windows 시계 단위 15ms)
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "01055554444"})
        call.speaking = False                                                 # 재안내가 끝남 → 번호 전송
        await wait_for(s, lambda s: len(call.keys) == 3, timeout=2)
        await s.on_paragraph(reprompt, started_at=t_reprompt)                 # 전송 전에 시작된 안내의 판단이 늦게 끝남
        assert s.secure_window                                                # 되읽기 구간 유지
        await s.on_paragraph(BROKEN_READBACK, started_at=time.monotonic())
        assert s.pending and s.pending["kind"] == "clarification" and "구해 3" not in str(s.captions)
    run(go())


def test_secure_window_reopens_when_digits_are_sent():
    """되읽기 구간이 어떤 이유로 닫혀 있어도 번호를 실제로 보내는 순간 다시 열림."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        s.secure_window = False
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "950101"})
        assert s.secure_window and s.secure_sent_at > 0
    run(go())


def test_no_prefetch_chunks_inside_readback_window():
    """되읽기 구간(민감정보 입력 직후)에서는 미리 인식 조각 전달 없음 → 단락 전체 인식
    (한 자씩 끊어 읽는 번호를 조각마다 인식하면 숫자 누락). 구간이 닫히면 미리 인식 재개."""
    from audio import Segmenter

    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=ScriptSTT(["안내"] * 5))
        chunks: list = []
        old = Segmenter.on_chunk
        Segmenter.on_chunk = staticmethod(lambda prefix, start: chunks.append(len(prefix)))
        try:
            s.segmenter = Segmenter(threshold=50, silence_ms=1200)   # 시험음(허밍)의 낮은 음량 고려
            s.segmenter.min_chunk_ms = 1500
            await speak(s, 2.0, silence=0.4)                          # 쉼 0.3초 이상 → 미리 인식 조각
            assert len(chunks) == 1
            s.secure_window = True
            await speak(s, 2.0, silence=0.4)
            assert len(chunks) == 1                                   # 되읽기 구간: 조각을 넘기지 않음
            s.secure_window = False
            await speak(s, 2.0, silence=0.4)
            assert len(chunks) == 2                                   # 구간이 닫히면 다시 미리 인식
        finally:
            Segmenter.on_chunk = old
    run(go())


TIMEOUT_PAST = "입력 시간이 지났습니다. "
FAX_REPROMPT = TIMEOUT_PAST + "지역번호를 포함한 팩스 번호를 누르신 후 우물정자를 눌러주십시오."


def _as_input_error(s, text: str, options_from: str | None = None) -> None:
    """모의 해석기가 text 를 입력 오류로 분류하게 함. options_from 을 주면 그 안내의 선택지를 함께 반환(JEV 해석기와 같은 형태),
    주지 않으면 선택지 없이 반환(로컬 모델 해석기와 같은 형태)."""
    from interpreter import _norm

    d = dict(s.d.interp.prompts[_norm(options_from)][1]) if options_from else {}
    d["kind"] = "input_error"
    s.d.interp.prompts[_norm(text)] = (text, d)


def test_input_error_without_options_menu_reread_is_retried_at_once():
    """키 직후 '입력 시간이 지났습니다' 와 메뉴 재낭독이 한 단락이고 해석기가 선택지 없이 입력 오류로만 답해도(로컬 모델 해석기 형태),
    규칙 선택지로 같은 메뉴를 바로 다시 판단해 키를 다시 보냄. 같은 단계에서 한 번 더 거부되면 사용자에게 재질문."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_PAST + ROOT
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        assert call.keys == ["5"]
        await s.on_paragraph(err)
        assert call.keys == ["5", "5"] and s.pending is None
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1]
        await s.on_paragraph(err)                                    # 같은 단계 2회 연속
        assert call.keys == ["5", "5"]
        assert s.pending and s.pending["reason"] == "input_rejected"
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1, 2]
    run(go())


def test_input_error_with_options_menu_reread_is_retried_at_once():
    """해석기가 선택지를 함께 준 입력 오류(JEV 해석기 형태)도 첫 번째에 바로 다시 누름 (다음 단락의 메뉴를 기다리지 않음)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_PAST + ROOT
        _as_input_error(s, err, options_from=ROOT)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(err)
        assert call.keys == ["5", "5"] and s.pending is None
        assert len(ev(s, "input_rejected")) == 1
    run(go())


def test_input_error_with_secure_reprompt_after_menu_key_opens_secure_request():
    """메뉴 키 뒤 '입력 시간이 지났습니다. 지역번호를 포함한 팩스 번호를 누르신 후 우물정자…' 를 해석기가 입력 오류로 분류해도
    민감정보 입력 요청을 엶 (메뉴 키 거부 알림·재시도 문구는 보내지 않음)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        _as_input_error(s, FAX_REPROMPT)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(FAX_REPROMPT)
        assert s.pending and s.pending["kind"] == "secure_input" and s.pending["terminator"] == "#"
        assert ev(s, "input_rejected") == [] and call.keys == ["5"]
    run(go())


def test_rejected_secure_input_with_reprompt_keeps_reject_path():
    """민감정보 입력값을 보낸 뒤의 '잘못 누르셨습니다. … 우물정자를 눌러 주십시오' 입력 오류는 입력 거부 알림(keys=null)과
    같은 항목 재요청으로 처리 (입력 요청으로 바꾸지 않음)."""
    async def go():
        s, call = await secure_pending()
        first = s.pending
        err = ERR + " " + RRN
        _as_input_error(s, err)
        await s.action({"type": "secure_input", "request_id": first["request_id"], "digits": "9999"})
        await s.on_paragraph(err)
        rej = ev(s, "input_rejected")
        assert len(rej) == 1 and rej[0]["keys"] is None
        assert s.pending and s.pending["kind"] == "secure_input" and s.pending["request_id"] != first["request_id"]
        assert "9999" not in str(s.hub.history)
    run(go())


def test_timeout_with_menu_reread_while_clarification_pending_same_paragraph():
    """재질문 대기 중 '입력 시간이 초과되었습니다' + 메뉴 재낭독이 한 단락이어도 키 오류로 세지 않고 다시 판단하지 않음."""
    async def go():
        s, call = await manual("카드 관련 문의", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_MSG + " " + ROOT
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        req = s.pending
        assert req and req["reason"] == "low_confidence"
        await s.on_paragraph(err)
        assert ev(s, "input_rejected") == [] and s.pending == req and call.keys == []
    run(go())


def test_timeout_with_reprompt_while_secure_pending_keeps_request():
    """민감정보 입력 대기 중 '입력 시간이 초과되었습니다' + 같은 입력 안내가 한 단락이면 새 요청 없이 같은 요청 유지."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        err = TIMEOUT_MSG + " " + RRN
        _as_input_error(s, err)
        await s.on_paragraph(err)
        assert ev(s, "input_rejected") == [] and s.pending["request_id"] == req["request_id"]
        assert call.keys == ["3", "2"]
    run(go())


def test_retry_after_input_error_does_not_stack_layers():
    """거부된 키의 계층을 되돌린 뒤 다시 눌러, 같은 메뉴 계층이 두 번 쌓이지 않음 ('처음으로' 복귀 시 실패한 갈래로 다시 들어가는 것 방지)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_PAST + ROOT
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(err)
        assert call.keys == ["5", "5"] and [(l.key, l.actor) for l in s.layers] == [("5", "ai")]
    run(go())


def test_reread_of_critical_confirm_after_user_cancel_is_not_auto_pressed():
    """중요 선택 확인 질문에서 사용자가 '취소'(2)를 고른 뒤 시간 초과 + 같은 질문 재낭독이 선택지 없는 입력 오류로 와도
    AI 가 키를 누르지 않고 사용자에게 다시 물음."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_PAST + CONFIRM
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(CARD)
        await s.on_paragraph(CONFIRM)
        req = s.pending
        assert req and req["reason"] == "critical_action" and call.keys == ["5", "3"]
        await s.action({"type": "select", "request_id": req["request_id"], "key": "2"})
        await s.on_paragraph(err)
        assert call.keys == ["5", "3", "2"]
        assert s.pending and s.pending["reason"] in ("critical_action", "input_rejected")
    run(go())


def test_rule_menu_after_input_error_with_approval_words_is_critical():
    """메뉴 키 뒤 '입력 시간이 지났습니다. 결제를 승인하시려면 1번, 취소하시려면 2번 …' 을 선택지 없는 입력 오류로 받아도
    규칙으로 붙인 메뉴를 중요 선택으로 보아 AI 가 누르지 않고 critical_action 으로 물음."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_PAST + "결제를 승인하시려면 1번, 취소하시려면 2번을 눌러 주십시오."
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(err)
        assert call.keys == ["5"]
        assert s.pending and s.pending["reason"] == "critical_action"
    run(go())


def test_user_key_rejected_then_menu_reread_asks_user_again():
    """사용자가 재질문에서 고른 키가 거부되고 같은 메뉴를 다시 읽으면 AI 가 다시 고르지 않고 사용자에게 다시 물음
    (not_found 로 끝나거나 다른 갈래를 누르지 않음)."""
    async def go():
        s, call = await manual("카드 관련 문의", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_PAST + ROOT
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        req = s.pending
        assert req and req["reason"] == "low_confidence"
        await s.action({"type": "select", "request_id": req["request_id"], "key": "5"})
        await s.on_paragraph(err)
        assert not s.ended and call.keys == ["5"]
        assert s.pending and s.pending["reason"] == "input_rejected"
    run(go())


def test_single_option_menu_with_sensitive_word_after_error_is_not_input_request():
    """메뉴 키 뒤 '입력 시간이 초과되었습니다. 비밀번호 변경을 원하시면 1번을 눌러 주십시오.' (입력 오류, 선택지 없음)는
    민감정보 입력 요청을 열지 않음."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_MSG + " 비밀번호 변경을 원하시면 1번을 눌러 주십시오."
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(err)
        assert not (s.pending and s.pending["kind"] == "secure_input") and call.keys == ["5"]
    run(go())


def test_secure_pending_timeout_with_single_option_menu_closes_request():
    """민감정보 입력 대기 중 '…입력 시간이 초과되었습니다. 인증번호를 다시 받으시려면 1번' 을 해석기가 선택지와 함께 입력 오류로
    주면 그 메뉴로 판단 (지난 입력 요청을 붙잡지 않음)."""
    async def go():
        s, call = await secure_pending()
        err = "인증번호 입력 시간이 초과되었습니다. 인증번호를 다시 받으시려면 1번을 눌러주세요."
        from interpreter import _norm
        s.d.interp.prompts[_norm(err)] = (err, {"kind": "input_error", "options": [{"key": "1", "label": "인증번호 다시 받기"}]})
        await s.on_paragraph(err)
        assert not (s.pending and s.pending["kind"] == "secure_input")
    run(go())


def test_two_digit_key_menu_reread_after_error_presses_same_key():
    """1~10번 메뉴에서 10 을 누른 뒤 시간 초과 + 같은 메뉴 재낭독(선택지 없는 입력 오류)이 오면 직전 메뉴 선택지로 10 을 다시 누름."""
    from interpreter import _norm

    async def go():
        goal = "대리인 신고서 팩스로 받고 싶어"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        names = [f"{n}번 서식 신고서" for n in range(1, 10)] + ["대리인 선임 신고서"]
        menu = "보험가입지원 서식입니다. " + ", ".join(f"{name}는 {i}번" for i, name in enumerate(names, 1)) + "을 눌러주십시오."
        s.d.interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": [
            {"key": str(i), "label": name, "category": name} for i, name in enumerate(names, 1)]})
        s.d.interp.goals[goal] = {"match": {"대리인 선임 신고서": 0.95}}
        err = TIMEOUT_PAST + menu
        _as_input_error(s, err)
        await s.on_paragraph(menu)
        assert call.keys == ["10"]
        await s.on_paragraph(err)
        assert call.keys == ["10", "10"]
    run(go())


def test_menu_reread_with_trailing_notice_is_retried_but_not_with_ending():
    """메뉴 재낭독 뒤에 안내 문장이 붙어도(같은 메뉴) 다시 누름. 같은 단락에 종료 안내('다시 이용해 주시기 바랍니다')가 있으면 누르지 않음."""
    async def go():
        for tail, want in ((" 본 서비스에서 제공되지 않는 업무는 홈페이지를 이용해 주시기 바랍니다.", ["5", "5"]),
                           (" 이용 방법을 확인하신 후 다시 이용해 주시기 바랍니다.", ["5"])):
            s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
            err = TIMEOUT_PAST + ROOT + tail
            _as_input_error(s, err)
            await s.on_paragraph(ROOT)
            await s.on_paragraph(err)
            assert call.keys == want, tail
    run(go())


def test_secure_input_then_error_with_next_item_opens_new_request():
    """민감정보를 보낸 뒤 '입력 시간이 초과되었습니다. 비밀번호 네 자리를 누르신 후 우물정자…' (다른 항목)가 입력 오류로 오면
    이전 항목을 다시 묻지 않고 새 항목으로 요청 (거부 알림 없음)."""
    async def go():
        s, call = await secure_pending()
        first = s.pending
        err = TIMEOUT_MSG + " 비밀번호 네 자리를 누르신 후 우물정자를 눌러 주십시오."
        _as_input_error(s, err)
        await s.action({"type": "secure_input", "request_id": first["request_id"], "digits": "991332"})
        await s.on_paragraph(err)
        assert ev(s, "input_rejected") == []
        assert s.pending and s.pending["kind"] == "secure_input" and "비밀번호" in s.pending["label"]
    run(go())


def test_secure_input_rejected_with_menu_reread_decides_menu():
    """민감정보를 보낸 뒤 '잘못 누르셨습니다.' + 첫 메뉴 재낭독이 입력 오류로 오면 거부 알림(keys=null)은 보내고,
    입력 요청을 다시 열지 않고 그 메뉴로 판단."""
    async def go():
        s, call = await secure_pending()
        first = s.pending
        err = ERR + " " + ROOT
        _as_input_error(s, err)
        await s.action({"type": "secure_input", "request_id": first["request_id"], "digits": "991332"})
        n_req = len([r for r in ev(s, "request") if r["kind"] == "secure_input"])
        await s.on_paragraph(err)
        rej = ev(s, "input_rejected")
        assert len(rej) == 1 and rej[0]["keys"] is None
        assert not (s.pending and s.pending["kind"] == "secure_input")
        assert len([r for r in ev(s, "request") if r["kind"] == "secure_input"]) == n_req
    run(go())


def test_session_secure_request_notifies_interpreter():
    """세션이 해석기 판정 밖에서 민감정보 입력 요청을 열면 해석기에 알림 (hybrid 해석기가 입력 요청 뒤의 안내를 모두 로컬 모델로 판단하는 데 사용)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        calls = []
        s.d.interp.note_input_request = lambda: calls.append(1)
        _as_input_error(s, FAX_REPROMPT)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(FAX_REPROMPT)
        assert s.pending and s.pending["kind"] == "secure_input" and calls == [1]
        assert s.pending["label"] == "지역번호를 포함한 팩스번호"
    run(go())


READBACK_NONE = "입력하신 번호는 번입니다. 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오."
READBACK_11 = "입력하신 번호는 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8 번입니다. 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오."


class _RetryRecognizer:
    is_local = True

    def __init__(self, alt: str) -> None:
        self.alt, self.calls = alt, 0

    async def recognize_retry(self, seg: bytes) -> str:
        self.calls += 1
        return self.alt


def test_readback_retry_uses_second_recognition_only_when_closer_to_entered_length():
    """되읽기 숫자가 입력 자릿수보다 적으면 빔 1 다시 인식을 한 번 하고, 입력 자릿수에 더 가까운 결과를 씀.
    - 번호가 빠짐(0자리, 입력 11) → 다시 인식(11자리) 사용
    - 입력보다 길게 눌렀고(12) ARS 가 11자리만 읽음 → 처음 11자리면 다시 인식해도 같아 처음 결과 유지
    - 번호가 다 있음 → 다시 인식하지 않음 / 되읽기 문장이 아님 → 다시 인식하지 않음"""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어")
        for want, text, alt, expect, calls in (
                (11, READBACK_NONE, READBACK_11, READBACK_11, 1),
                (12, READBACK_NONE, READBACK_11, READBACK_11, 1),
                (12, READBACK_11, READBACK_11.replace("0, 1, 0", "0, 1, 0, 0"), READBACK_11.replace("0, 1, 0", "0, 1, 0, 0"), 1),
                (11, READBACK_11, READBACK_NONE, READBACK_11, 0),
                (11, "고객님의 요청이 접수되었습니다.", READBACK_11, "고객님의 요청이 접수되었습니다.", 0)):
            rec = _RetryRecognizer(alt)
            s.d.recognizer = rec
            s.secure_len, s.readback_retry_armed = want, True
            out = await s._readback_retry(b"\xff" * 800, text)
            assert out == expect and rec.calls == calls, (want, text, alt)
        rec = _RetryRecognizer(READBACK_NONE)
        s.d.recognizer, s.secure_len, s.readback_retry_armed = rec, 12, True
        assert await s._readback_retry(b"\xff" * 800, READBACK_11) == READBACK_11 and rec.calls == 1  # 11→0 은 멀어짐
    run(go())


class _ScriptRetrySTT(ScriptSTT):
    """시험용 STT: 정해 둔 문장 + 다시 인식(recognize_retry) 결과."""

    def __init__(self, lines: list[str], alt: str) -> None:
        super().__init__(lines)
        self.alt, self.retries = alt, 0

    async def recognize_retry(self, seg: bytes) -> str:
        self.retries += 1
        return self.alt


def test_readback_missing_digits_is_recognized_again_in_voice_path():
    """음성 경로: 민감정보 11자리 입력 뒤 되읽기 단락을 STT 가 번호 없이 받아 적으면 같은 소리를 한 번 더 인식해
    번호가 든 문장으로 되읽기 확인 질문을 보냄 (화면 문구에 숫자 표시)."""
    async def go():
        stt = _ScriptRetrySTT([ROOT, BAL, RRN, READBACK_NONE], READBACK_11)
        s, call = await manual("통장 잔액 알고 싶어", stt=stt, paragraph_silence_ms=200, clarify_expire_s=5, clarify_drop_s=10)
        await speak(s, 0.3)
        await wait_for(s, lambda s: call.keys == ["3"])
        await speak(s, 0.3)
        await wait_for(s, lambda s: call.keys == ["3", "2"])
        await speak(s, 0.3)
        await wait_for(s, lambda s: s.pending and s.pending["kind"] == "secure_input")
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "01012345678"})
        await wait_for(s, lambda s: len(call.keys) == 3)
        await speak(s, 0.3)
        await wait_for(s, lambda s: s.pending and s.pending["kind"] != "secure_input")
        assert stt.retries == 1
        assert len(re.findall(r"[0-9]", re.sub(r"맞으면\s*1번|2번을", "", s.pending["message"]))) == 11
    run(go())


def test_readback_retry_rejects_unsafe_or_unneeded_second_recognition():
    """다시 인식 결과는 되읽기 확인 문장이고 확인 질문 선택지(1·2번)를 모두 담고 번호가 더 많을 때만 씀. 번호 일부만 되읽는 문장,
    한국어 숫자로 온전히 받아 적은 문장, 민감정보 전송 뒤가 아닌 구간, '입력하신 후 확인' 안내는 다시 인식하지 않음.
    다시 인식 오류 시 처음 문장 사용."""
    alt_cases = [
        "공일공일이삼사오육칠팔",                                                       # 숫자만 (확인 질문 없음)
        "고객님께서 누르신 번호는 0101234567 번 입니다.",                                 # 되읽기 문장 아님 (가리기를 빠져나감)
        "입력하신 번호는 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8 번입니다.",                         # 확인 질문 꼬리 없음
        "입력하신 번호는 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8 번입니다. 맞으면 1번을 눌러주십시오.",  # '다시 입력 2번' 빠짐
        "고객님 번호는 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8 번. 확인하시려면 1번, 수정하시려면 2번을 눌러주십시오.",  # 선택지는 있으나 되읽기 문장 아님
    ]

    class _Boom(_RetryRecognizer):
        async def recognize_retry(self, seg):
            self.calls += 1
            raise RuntimeError("decode failed")

    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어")
        s.secure_len, s.readback_retry_armed = 11, True
        for alt in alt_cases:
            s.d.recognizer = rec = _RetryRecognizer(alt)
            assert await s._readback_retry(b"\xff" * 800, READBACK_NONE) == READBACK_NONE and rec.calls == 1, alt
        for text in ("입력하신 번호 끝 네 자리는 1, 2, 3, 4번입니다. 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오.",
                     "입력하신 번호는 공일공 일이삼사 오육칠팔 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오.",
                     "입력하신 번호는 공, 일, 공, 일, 이, 삼, 사, 오, 육, 칠, 팔 맞으면 1번 다시 입력하시려면 2번을 눌러주십시오.",
                     "계좌 비밀번호 네 자리를 입력하신 후 확인을 위해 우물정자를 눌러주십시오."):
            s.d.recognizer = rec = _RetryRecognizer(READBACK_11)
            assert await s._readback_retry(b"\xff" * 800, text) == text and rec.calls == 0, text
        s.d.recognizer = rec = _RetryRecognizer(READBACK_11)
        s.readback_retry_armed = False
        assert await s._readback_retry(b"\xff" * 800, READBACK_NONE) == READBACK_NONE and rec.calls == 0
        s.readback_retry_armed = True
        s.d.recognizer = rec = _Boom(READBACK_11)
        assert await s._readback_retry(b"\xff" * 800, READBACK_NONE) == READBACK_NONE and rec.calls == 1
    run(go())


def test_whisper_recognize_retry_skipped_when_beam_is_one():
    """설정 빔이 1 이면 같은 디코딩이므로 다시 인식하지 않음 (빈 문자열, 세션은 처음 문장 유지)."""
    import stt

    rec = stt.WhisperRecognizer("small", "cpu", 1)
    rec._run = lambda *a, **k: (_ for _ in ()).throw(AssertionError("_run 이 불림"))
    assert run(rec.recognize_retry(b"\xff" * 1600)) == ""


def test_retry_with_model_labels_differing_from_text_keeps_model_menu():
    """해석기가 선택지 이름을 원문과 다르게 적었어도, 두 단락의 규칙 추출이 같은 메뉴면 직전 메뉴(해석기 이름)로 다시 누름
    (계층의 메뉴가 바뀌지 않아 '처음으로' 복귀 때 제외가 유지됨)."""
    from interpreter import _norm

    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        d = dict(s.d.interp.prompts[_norm(ROOT)][1])
        names = ["잔액 확인", "거래 내역", "카드 관련 업무", "상담원 연결"]
        d["options"] = [{**o, "label": n} for o, n in zip(d["options"], names)]
        s.d.interp.prompts[_norm(ROOT)] = (ROOT, d)
        err = TIMEOUT_PAST + ROOT
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(err)
        assert call.keys == ["5", "5"] and len(s.layers) == 1
        assert [o.label for o in s.layers[0].prompt.options] == names
    run(go())


def test_timeout_notice_then_next_menu_keeps_layer_and_decides_new_menu():
    """키가 받아들여졌는데 다음 메뉴의 첫 낭독을 못 듣고 '입력 시간이 초과되었습니다' 뒤 다음 메뉴가 오면, 거부로 보지 않고
    계층을 유지한 채 새 메뉴로 판단. 한 단락으로 와도 같음 (거부 알림 없음, 계층 유지)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        _as_input_error(s, TIMEOUT_MSG)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(TIMEOUT_MSG)
        await s.on_paragraph(CARD)
        assert call.keys == ["5", "3"] and [l.key for l in s.layers] == ["5", "3"] and s.input_errors == 0

        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_MSG + " " + CARD
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(err)
        assert ev(s, "input_rejected") == [] and [l.key for l in s.layers] == ["5"]
    run(go())


def test_split_resident_number_next_part_after_error_opens_new_request():
    """주민등록번호 앞자리를 보낸 뒤 '입력 시간이 초과되었습니다. 주민등록번호 뒤 일곱 자리를 …' 가 입력 오류로 오면 다른 항목(뒤
    일곱 자리)으로 새 요청 (앞자리를 다시 묻지 않음, 거부 알림 없음)."""
    async def go():
        s, call = await secure_pending()
        first = s.pending
        err = TIMEOUT_MSG + " 주민등록번호 뒤 일곱 자리를 누르신 후 우물정자를 눌러 주십시오."
        _as_input_error(s, err)
        await s.action({"type": "secure_input", "request_id": first["request_id"], "digits": "991332"})
        await s.on_paragraph(err)
        assert ev(s, "input_rejected") == []
        assert s.pending and s.pending["kind"] == "secure_input" and "뒤" in s.pending["label"]
    run(go())


def test_names_only_notice_after_error_is_not_menu_reread():
    """입력 오류 뒤 직전 메뉴 이름만 나오고 번호·누르라는 말이 없는 안내는 메뉴 재낭독으로 보지 않음 (다시 누르지 않음)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_MSG + " 잔액조회와 거래내역조회, 카드 업무는 인터넷뱅킹에서도 이용하실 수 있습니다."
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(err)
        assert call.keys == ["5"]
    run(go())


def test_two_digit_only_new_menu_after_error_gets_options():
    """입력 오류 문장 뒤 두 자리 키만 있는 새 메뉴('기타 서식은 10번, 재증명 신청은 11번')도 선택지를 찾아 판단 (선택지 10·11)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_MSG + " 기타 서식은 10번, 재증명 신청은 11번을 눌러주십시오."
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(err)
        assert s.pending and {"10", "11"} <= {o["key"] for o in s.pending["options"]}
    run(go())


def test_keypad_after_ai_key_rejection_then_next_menu_is_decided_by_ai():
    """AI 키가 거부로 보인 뒤 사용자가 키패드로 누르고 ARS 가 다음 메뉴로 가면, 보류 뒤 AI 가 그 메뉴를 판단 (거부 재질문 아님)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10, user_key_hold_s=0.05)
        _as_input_error(s, TIMEOUT_MSG)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(TIMEOUT_MSG)
        await s.action({"type": "key", "key": "5"})
        await s.on_paragraph(CARD)
        await wait_for(s, lambda s: call.keys == ["5", "5", "3"])
        assert not (s.pending and s.pending["reason"] == "input_rejected")
    run(go())


def test_key_sent_outside_press_does_not_pop_parent_layer():
    """_press 를 거치지 않은 키(재질문 만료의 다시 듣기 키 등) 뒤의 입력 오류는 부모 계층을 지우지 않음."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        err = TIMEOUT_PAST + CARD
        _as_input_error(s, err)
        await s.on_paragraph(ROOT)
        s.current = await s.d.interp.classify(CARD)
        await s._send_keys("*")
        await s.on_paragraph(err)
        assert s.layers and s.layers[0].key == "5"
    run(go())


def test_secure_key_sends_each_digit_immediately_and_done_adds_terminator():
    """한 자씩 입력(secure_key): 누를 때마다 바로 ARS 로 보내고 요청은 열어 둠. done 이면 종료 키(#)를 붙이고 요청을 닫음.
    자릿수만 기억(되읽기 가리기용), 값은 이벤트·기록에 남지 않음."""
    async def go():
        s, call = await secure_pending(record=True)
        req = s.pending
        for k in "991":
            out = await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
            assert out == {"accepted": True, "done": False}
        assert call.keys[-3:] == ["9", "9", "1"] and s.pending and s.pending["request_id"] == req["request_id"]
        out = await s.action({"type": "secure_key", "request_id": req["request_id"], "done": True})
        assert out["done"] is True and call.keys[-1] == "#" and s.pending is None
        assert s.secure_len == 3 and s.last_send_secure and s.secure_window
        with pytest.raises(ApiError):
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "5"})  # 닫힌 요청
        assert "991" not in str(s.hub.history) and "991" not in str(s.recorder.events)
        assert [e.get("digit_count") for e in s.recorder.events if e.get("action") == "secure_key"][:3] == [1, 1, 1]
    run(go())


def test_secure_key_terminator_key_finishes_without_duplicate_and_rejects_bad_input():
    """요청의 종료 키(#)를 직접 누르면 그대로 보내고 끝(종료 키를 또 붙이지 않음). 키 형식이 틀리거나 다른 요청이면 거절."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        for body, code in (({"key": "12"}, 400), ({"key": "a"}, 400), ({}, 400), ({"request_id": "req_x", "key": "1"}, 409)):
            with pytest.raises(ApiError) as e:
                await s.action({"type": "secure_key", "request_id": req["request_id"], **body})
            assert e.value.status == code, body
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "7"})
        out = await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "#"})
        assert out["done"] is True and call.keys[-2:] == ["7", "#"] and s.pending is None and s.secure_len == 1
    run(go())


def test_secure_key_while_ars_speaking_is_held_then_sent_in_order():
    """ARS 가 말하는 동안 누른 키는 모았다가 안내가 끝나면 순서대로 보냄 (말하는 중에 보낸 키는 무시될 수 있음)."""
    async def go():
        s, call = await secure_pending(secure_wait_max_s=3)
        req = s.pending
        n0 = len(call.keys)
        call.speaking = True
        for k in "25":
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
        await asyncio.sleep(0.1)
        assert len(call.keys) == n0 and s.secure_queued
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "8"})
        call.speaking = False
        await wait_for(s, lambda s: len(call.keys) > n0)
        assert "".join(call.keys[n0:]) == "258" and not s.secure_queued and s.secure_buf == ""
    run(go())


def test_secure_key_buffer_dropped_when_ars_moves_on():
    """모아 둔 키를 보내기 전에 ARS 가 다른 안내로 넘어가면 보내지 않고 버림 (입력 거부 알림)."""
    async def go():
        s, call = await secure_pending(secure_wait_max_s=0.3)
        req = s.pending
        n0 = len(call.keys)
        call.speaking = True
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "4"})
        await wait_for(s, lambda s: not s.secure_queued)
        assert len(call.keys) == n0 and s.secure_buf == "" and ev(s, "input_rejected")[-1]["keys"] is None
    run(go())


ACCT_PW = "계좌 비밀번호 네 자리를 눌러 주십시오."
BAL_RESULT = "고객님의 출금 가능 금액은 1,234,000원입니다. 감사합니다."


def test_secure_key_fixed_length_field_closes_on_result_without_done():
    """종료 키 없는 고정 자릿수 항목(비밀번호 네 자리)은 마지막 자리를 받으면 ARS 가 스스로 다음 단계로 진행. 사용자가 입력 완료를
    누르기 전에 결과 안내가 오면 요청을 닫고 결과를 처리 (요청을 열어 둔 채 결과를 버리고 user_input_timeout 으로 끝나는 문제 방지)."""
    async def go():
        s, call = await secure_pending()
        await s.action({"type": "secure_key", "request_id": s.pending["request_id"], "done": True})
        await s.on_paragraph(ACCT_PW)
        req = s.pending
        assert req and req["kind"] == "secure_input" and not req.get("terminator")
        for k in "1234":
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
        await s.on_paragraph(BAL_RESULT)
        assert s.pending is None and s.status == "agent_done" and s.summary["kind"] == "result"
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended)
        assert s.end_reason == "completed"
    run(go())


def test_secure_key_new_input_prompt_after_streamed_keys_opens_new_request():
    """한 자씩 보낸 뒤 입력 안내가 새로 들리면(다음 항목이거나 다시 받기) 요청을 새로 엶: 항목 이름·자릿수가 새로 시작.
    누르는 동안 시작된 안내(마지막 키보다 먼저 시작)는 같은 요청 유지."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        t_before = time.monotonic() - 0.5                        # (Windows monotonic 해상도 약 16ms: 시각이 같은 값이 되는 것 방지)
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "9"})
        await s.on_paragraph(RRN, t_before)                      # 누르는 중에 시작된 같은 안내
        assert s.pending["request_id"] == req["request_id"]
        await s.on_paragraph(ACCT_PW)                            # 마지막 키 뒤의 새 안내
        assert s.pending and s.pending["request_id"] != req["request_id"] and "비밀번호" in s.pending["label"]
        await s.action({"type": "secure_key", "request_id": s.pending["request_id"], "key": "1"})
        assert s.secure_len == 1
    run(go())


def test_secure_key_input_error_after_streamed_keys_reopens_same_field():
    """한 자씩 보낸 번호를 ARS 가 거부하면 같은 항목으로 요청을 새로 엶 (입력 거부 알림, 자릿수가 이어지지 않음)."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        for k in "12":
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
        _as_input_error(s, ERR)
        await s.on_paragraph(ERR)
        new = s.pending
        assert ev(s, "input_rejected")[-1]["keys"] is None
        assert new and new["kind"] == "secure_input" and new["request_id"] != req["request_id"]
        assert new["label"] == req["label"] and new.get("terminator") == req.get("terminator")
        await s.action({"type": "secure_key", "request_id": new["request_id"], "key": "3"})
        assert s.secure_len == 1
    run(go())


def test_secure_key_info_paragraph_keeps_request_open():
    """입력 도중의 일반 안내로는 요청을 닫지 않음 (닫으면 다음 키가 거절됨)."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "5"})
        await s.on_paragraph("잠시만 기다려 주십시오.")
        assert s.pending and s.pending["request_id"] == req["request_id"]
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "6"})
    run(go())


def _slow_send(call, per_key_s: float = 0.05):
    """보내는 데 시간이 걸리는 키 전송 (실제 통화의 톤 전송처럼). 동시에 보내는 수의 최댓값을 call.max_conc 에 기록."""
    call.wire, call.conc, call.max_conc = [], 0, 0

    async def send(keys: str) -> float:
        call.conc += 1
        call.max_conc = max(call.max_conc, call.conc)
        for ch in keys:
            await asyncio.sleep(per_key_s)
            call.wire.append(ch)
        call.conc -= 1
        call.keys.append(keys)
        return 0.0

    call.send_keys = send


def test_secure_key_during_buffer_flush_keeps_order_and_never_overlaps():
    """모아 둔 키를 보내는 중에 들어온 키·끝(done)은 그 뒤에 붙여 한 작업이 순서대로 전송 (따로 보내면 두 전송이 겹쳐
    '1243'·'12#3' 처럼 순서가 바뀌는 문제 방지)."""
    async def go():
        s, call = await secure_pending(secure_wait_max_s=3)
        req = s.pending
        _slow_send(call)
        call.speaking = True
        for k in "123":
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
        call.speaking = False
        await wait_for(s, lambda s: len(call.wire) >= 1)                # 흘려보내기 시작
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "4"})
        await s.action({"type": "secure_key", "request_id": req["request_id"], "done": True})
        await wait_for(s, lambda s: not s.secure_queued and len(call.wire) >= 5)
        assert "".join(call.wire) == "1234#" and call.max_conc == 1 and s.secure_len == 4
    run(go())


def test_secure_key_input_error_while_typing_reopens_and_drops_keys_typed_during_error():
    """키를 보낸 뒤 시작된 입력 오류 안내는, 사용자가 그 안내 중에 더 눌러(모아 둔 키) 마지막 키보다 먼저 시작된 셈이 되어도
    오류로 처리: 입력 거부 알림, 같은 항목으로 새 요청, 오류 안내 중에 누른 키는 보내지 않음 (ARS 의 새 시도에 섞이는 것 방지)."""
    async def go():
        s, call = await secure_pending(secure_wait_max_s=3)
        req = s.pending
        for k in "12":
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
        n0 = len(call.keys)
        started = time.monotonic() + 0.001
        await asyncio.sleep(0.05)
        call.speaking = True                                            # ARS 가 오류 안내를 시작, 사용자는 계속 누름
        for k in "34":
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
        s.stt_inflight += 1                                             # 안내가 끝나고 인식·판단 중 (실제 통화와 같은 순서)
        call.speaking = False
        await asyncio.sleep(0.1)
        assert len(call.keys) == n0                                     # 그 안내의 판단이 끝날 때까지 보내지 않음
        _as_input_error(s, ERR)
        await s.on_paragraph(ERR, started)
        s.stt_inflight -= 1
        await asyncio.sleep(0.1)
        new = s.pending
        assert ev(s, "input_rejected") and new and new["request_id"] != req["request_id"]
        assert len(call.keys) == n0 and s.secure_buf == "" and not s.secure_queued   # '34' 는 보내지 않음
        await s.action({"type": "secure_key", "request_id": new["request_id"], "key": "5"})
        assert s.secure_len == 1
    run(go())


def test_secure_key_dropped_keys_are_not_counted():
    """보내지 못하고 버린 키는 자릿수(secure_len)에 넣지 않음 (되읽기 가리기·다시 인식이 틀린 자릿수를 쓰는 것 방지)."""
    async def go():
        s, call = await secure_pending(secure_wait_max_s=0.2)
        req = s.pending
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "1"})
        call.speaking = True
        for k in "234":
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
        await wait_for(s, lambda s: not s.secure_queued)
        assert s.secure_len == 1 and ev(s, "input_rejected")
    run(go())


def test_secure_key_republishes_request_when_entry_takes_long():
    """입력이 길어지면 늘린 기한을 같은 요청 id 로 다시 알림 (화면이 처음 기한에 입력 칸을 닫는 것 방지). 자주 보내지 않음."""
    async def go():
        s, call = await secure_pending()
        req = s.pending
        n0 = len(ev(s, "request"))
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "1"})
        assert len(ev(s, "request")) == n0
        s.secure_req_pub_at -= 20
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "2"})
        assert len(ev(s, "request")) == n0 + 1 and ev(s, "request")[-1]["request_id"] == req["request_id"]
    run(go())


def test_secure_key_buffered_keys_not_sent_as_readback_answer():
    """한 자씩 입력 중 모아 둔 키는 ARS 가 되읽기 확인 질문으로 넘어가면 보내지 않음 (확인 질문의 답 '1'·'2' 로 들어가는 것 방지)."""
    async def go():
        s, call = await secure_pending(secure_wait_max_s=3)
        req = s.pending
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "1"})
        n0 = len(call.keys)
        call.speaking = True
        for k in "12":
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
        s.stt_inflight += 1
        call.speaking = False
        await s.on_paragraph(READBACK_NONE)
        s.stt_inflight -= 1
        await wait_for(s, lambda s: not s.secure_queued)
        assert len(call.keys) == n0 and s.secure_buf == ""
    run(go())


def test_secure_key_input_error_with_menu_drops_buffered_keys():
    """입력 오류 안내와 메뉴 재낭독이 한 단락이면 입력 요청을 닫고 메뉴로 판단하며, 모아 둔 키는 보내지 않음."""
    async def go():
        s, call = await secure_pending(secure_wait_max_s=3)
        req = s.pending
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "1"})
        call.speaking = True
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "7"})
        s.stt_inflight += 1
        call.speaking = False
        err = TIMEOUT_MSG + " " + BAL
        _as_input_error(s, err, options_from=BAL)
        await s.on_paragraph(err)
        s.stt_inflight -= 1
        await wait_for(s, lambda s: not s.secure_queued)
        assert "7" not in call.keys                                      # 모아 둔 '7' 은 보내지 않음
        assert not (s.pending and s.pending["kind"] == "secure_input" and s.pending["request_id"] == req["request_id"])
    run(go())


def test_secure_key_done_pressed_during_error_drops_keys_and_reopens():
    """오류 안내 중에 남은 번호와 입력 완료까지 누른 경우(요청은 닫혔지만 키는 모아 둔 채): 그 오류에 대해 입력 거부를 알리고
    같은 항목으로 새 요청을 열며, 모아 둔 키·종료 키는 보내지 않음 (앞자리가 빠진 번호가 ARS 의 새 시도로 들어가는 것 방지)."""
    async def go():
        s, call = await secure_pending(secure_wait_max_s=3)
        req = s.pending
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "1"})
        n0 = len(call.keys)
        started = time.monotonic() + 0.001
        await asyncio.sleep(0.05)
        call.speaking = True
        for k in "23":
            await s.action({"type": "secure_key", "request_id": req["request_id"], "key": k})
        await s.action({"type": "secure_key", "request_id": req["request_id"], "done": True})
        assert s.pending is None
        s.stt_inflight += 1
        call.speaking = False
        _as_input_error(s, ERR)
        await s.on_paragraph(ERR, started)
        s.stt_inflight -= 1
        await wait_for(s, lambda s: not s.secure_queued)
        assert len(call.keys) == n0 and ev(s, "input_rejected")
        assert s.pending and s.pending["kind"] == "secure_input" and s.pending["request_id"] != req["request_id"]
    run(go())


def test_secure_key_dropped_while_request_open_reopens_with_new_id():
    """모아 둔 키를 기다리다 버렸는데 같은 요청이 열려 있으면 같은 항목의 새 요청으로 바꿈 (화면 입력 칸이 버린 키를 세는 것 방지)."""
    async def go():
        s, call = await secure_pending(secure_wait_max_s=0.2)
        req = s.pending
        call.speaking = True
        await s.action({"type": "secure_key", "request_id": req["request_id"], "key": "4"})
        await wait_for(s, lambda s: not s.secure_queued)
        new = s.pending
        assert new and new["kind"] == "secure_input" and new["request_id"] != req["request_id"] and new["label"] == req["label"]
    run(go())


def test_whisper_recognize_retry_uses_beam_one():
    """WhisperRecognizer.recognize_retry 는 같은 소리를 빔 1 로 인식 (미리 인식 결과 미사용)."""
    import stt

    seen = []
    rec = stt.WhisperRecognizer("small", "cpu", 3)
    rec._run = lambda ulaw, context="", beam=None: (seen.append(beam), "입력하신 번호는 일입니다")[1]
    out = run(rec.recognize_retry(b"\xff" * 1600))
    assert seen == [1] and out


def test_timeout_notice_with_menu_reread_is_decided_as_menu():
    """키를 보내지 않은 상태에서 '입력 시간이 초과되었습니다' 뒤에 메뉴를 다시 읽으면 그 메뉴로 판단
    (메뉴를 다시 읽는 동안 아무것도 누르지 않아 ARS 가 끊는 경우 방지). 선택지가 없으면 일반 안내로 처리."""
    from interpreter import _norm

    async def go():
        s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        d = dict(s.d.interp.prompts[_norm(ROOT)][1])
        d["kind"] = "input_error"
        err = "입력 시간이 초과되었습니다. " + ROOT
        s.d.interp.prompts[_norm(err)] = (err, d)
        s.d.interp.prompts[_norm("입력 시간이 초과되었습니다.")] = ("입력 시간이 초과되었습니다.", {"kind": "input_error"})
        await s.on_paragraph("입력 시간이 초과되었습니다.")
        assert call.keys == [] and s.input_errors == 0
        await s.on_paragraph(err)
        assert call.keys == ["3"]
    run(go())


def test_notice_before_bounce_back_to_visited_menu_is_judged():
    """AI 가 키를 누른 뒤 ARS 가 안내와 함께 이미 지나온 메뉴로 되돌려 보내면 그 안내가 목적의 답인지 먼저 판단
    (예: 업무 시간 종료 안내 뒤 첫 메뉴를 다시 읽을 때 같은 경로를 한 번 더 누르는 문제 방지).
    답이면 결과로 처리하고 같은 키 재전송 없음. 답이 아니면 메뉴로 판단."""
    from interpreter import _norm

    notice = "지금은 상담 업무 시간이 종료되어 상담을 해드릴 수가 없습니다. 상담 가능 시간은 평일 오전 9시부터입니다. "

    async def go(verdict):
        s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        bounced = notice + ROOT.replace("KB국민은행입니다. ", "")
        s.d.interp.prompts[_norm(bounced)] = (bounced, dict(s.d.interp.prompts[_norm(ROOT)][1]))
        asked = _fake_answers_goal(s, verdict)
        await s.on_paragraph(ROOT)
        assert call.keys == ["3"]
        await s.on_paragraph(bounced)
        return s, call, asked

    s, call, asked = run(go(True))
    assert asked and "상담 업무 시간" in asked[0][1] and asked[0][2] == "잔액조회"
    assert call.keys == ["3"] and s.summary and s.summary["kind"] == "result" and s.status == "agent_done"
    assert s.captions[-1]["text"] == notice.strip()                          # 자막·요약 입력은 안내 문장만 (메뉴 제외)
    s, call, asked = run(go(None))
    assert asked and call.keys == ["3", "3"]                                   # 판단 불가 시 메뉴로 판단
    assert menu_preface_of("안녕하세요. 잔액조회는 3번입니다.") == "안녕하세요."


def test_bounce_back_with_one_key_missing_is_same_menu_only_for_tolerant_interpreters():
    """되돌아온 메뉴의 추출에서 키 하나가 빠져도(앞 안내가 붙은 단락) 해석기 tolerant_menu_match 면 같은 메뉴로 보고 앞 안내를 판단.
    속성이 없으면(다른 조합) 키 집합이 같아야 같은 메뉴. 공통 키가 적거나 이름이 다르면 다른 메뉴."""
    from interpreter import _norm, Option, Prompt
    from session import _same_menu

    notice = "지금은 상담 업무 시간이 종료되어 상담을 해드릴 수가 없습니다. 상담 가능 시간은 평일 오전 9시부터입니다. "

    async def go(tolerant):
        s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        bounced = notice + ROOT.replace("KB국민은행입니다. ", "")
        d = dict(s.d.interp.prompts[_norm(ROOT)][1])
        d["options"] = [o for o in d["options"] if o["key"] != "0"]       # 다시 읽은 메뉴의 추출에서 키 하나 누락
        s.d.interp.prompts[_norm(bounced)] = (bounced, d)
        if tolerant:
            s.d.interp.tolerant_menu_match = True
        asked = _fake_answers_goal(s, True)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(bounced)
        return s, call, asked

    s, call, asked = run(go(True))
    assert asked and call.keys == ["3"] and s.summary and s.summary["kind"] == "result"
    s, call, asked = run(go(False))
    assert not asked and call.keys == ["3", "3"]
    a = Prompt(kind="menu", text="", options=[Option("1", "가"), Option("2", "나"), Option("3", "다"), Option("4", "라")])
    b = Prompt(kind="menu", text="", options=[Option("1", "가"), Option("2", "나"), Option("3", "다")])
    c = Prompt(kind="menu", text="", options=[Option("1", "마"), Option("2", "바"), Option("3", "사"), Option("4", "아")])
    assert _same_menu(a, b, tolerant=True) and not _same_menu(a, b)
    assert not _same_menu(a, c, tolerant=True)
    assert not _same_menu(Prompt(kind="menu", text="", options=a.options[:2]), Prompt(kind="menu", text="", options=a.options[:1]), tolerant=True)


def menu_preface_of(text):
    from session import menu_preface
    return menu_preface(text)


def test_agent_goal_without_agent_option_says_so():
    """목적이 상담원 연결인데 메뉴에 상담원 선택지가 없으면 재질문 문구로 그 사실 안내 (NO_AGENT_ASK)."""
    from navigator import NO_AGENT_ASK, NOT_FOUND_ASK

    async def go():
        s, call = await manual("상담사랑 통화하고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s.on_paragraph(CARD)                                         # 상담원 선택지 없는 메뉴
        assert s.pending and s.pending["message"] == NO_AGENT_ASK and call.keys == []
        s2, call2 = await manual("이번 주 날씨 알려줘", clarify_expire_s=5, clarify_drop_s=10)
        await s2.on_paragraph(CARD)
        assert s2.pending and s2.pending["message"] == NOT_FOUND_ASK
    run(go())


def test_status_message_does_not_double_menu_word():
    """선택지 이름이 '…메뉴' 로 끝나면 상태 문구에 '메뉴' 중복 추가 없음."""
    from interpreter import _norm

    async def go():
        s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        d = dict(s.d.interp.prompts[_norm(ROOT)][1])
        d["options"] = [dict(o, label="잔액조회 메뉴") if o["key"] == "3" else o for o in d["options"]]
        menu = ROOT.replace("잔액조회는", "잔액조회 메뉴는")
        s.d.interp.prompts[_norm(menu)] = (menu, d)
        await s.on_paragraph(menu)
        msgs = [e["data"].get("message") for e in s.hub.history if e["event"] == "status_changed"]
        assert call.keys == ["3"] and "'잔액조회 메뉴'를 선택합니다." in msgs
    run(go())


def test_paragraph_cut_at_max_length_is_joined_with_the_next():
    """무음이 아니라 최대 길이에서 잘린 단락(말하는 도중)은 바로 판단하지 않고 다음 단락과 이어 붙여 한 안내로 판단
    (긴 안내가 메뉴 중간에서 잘려 앞쪽 선택지가 판단에서 빠지는 문제 방지)."""
    from audio import Segmenter

    async def go():
        head, tail = ROOT[:30], ROOT[30:]
        s, call = await manual("통장 잔액 알고 싶어", stt=ScriptSTT([head, tail]), no_response_s=5,
                               paragraph_silence_ms=200, clarify_expire_s=5, clarify_drop_s=10)
        s.segmenter = Segmenter(threshold=50, silence_ms=200, max_s=0.5)  # 시험음(허밍)의 낮은 음량 고려
        await speak(s, 0.9, silence=0.4)                                   # 0.5초에서 잘리고 이어서 말함
        await wait_for(s, lambda s: call.keys == ["3"])
        assert s._held_head is None and [c["text"] for c in s.captions if "잔액" in c["text"]]
        assert not any(c["text"] == head for c in s.captions)              # 앞부분만 따로 자막·판단하지 않음
    run(go())


def test_cut_paragraph_without_continuation_is_handled_alone():
    """잘린 뒤 이어지는 소리가 없으면 잠시 뒤 앞부분만으로 판단. 이어진 소리의 인식 결과가 빈 글이면 바로 판단."""
    import session as session_mod
    from audio import Segmenter

    async def go():
        s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        await s._deliver(ROOT, time.monotonic(), forced=True)
        assert s._held_head is not None and call.keys == []
        await wait_for(s, lambda s: call.keys == ["3"], timeout=3)          # 이어지는 소리 없음 → 앞부분만으로
        s2, call2 = await manual("통장 잔액 알고 싶어", stt=ScriptSTT([ROOT]), no_response_s=5,
                                 paragraph_silence_ms=200, clarify_expire_s=5, clarify_drop_s=10)
        s2.segmenter = Segmenter(threshold=50, silence_ms=200, max_s=0.5)
        await speak(s2, 0.7, silence=0.5)                                   # 잘린 뒤 짧은 소리, 인식 글 없음
        await wait_for(s2, lambda s: call2.keys == ["3"], timeout=3)
    old = session_mod.HELD_HEAD_CHECK_S
    session_mod.HELD_HEAD_CHECK_S = 0.1
    try:
        run(go())
    finally:
        session_mod.HELD_HEAD_CHECK_S = old


def test_cut_paragraph_is_not_joined_across_a_key_press(monkeypatch):
    """키 전후 안내는 따로 처리하고, 키 이전에 잘린 일반 안내는 자막에서 제외한다."""
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        delivered = []
        on_paragraph = s.on_paragraph

        async def observe(text, started_at=None, **kw):
            delivered.append((text, started_at))
            await on_paragraph(text, started_at, **kw)  # 실처리를 생략하지 않고 전달 경계만 기록 (taken 등 키워드 인자는 그대로 전달)

        monkeypatch.setattr(s, "on_paragraph", observe)
        try:
            # 실제로 먼저 시작된 발화. 시계 해상도가 낮아도 키 전송 시각과 같아지지 않게 한다.
            t0 = time.monotonic() - 1.0
            await s._deliver("첫 부분 안내입니다.", t0, forced=True)
            assert s._held_head is not None and not s.captions and call.keys == []
            await s.action({"type": "key", "key": "1"})                    # 모의 통화의 실제 사용자 키 전송 경로
            assert call.keys == ["1"]
            t1 = time.monotonic()
            await s._deliver("두 번째 안내입니다.", t1)
            assert delivered == [("첫 부분 안내입니다.", t0), ("두 번째 안내입니다.", t1)]
            assert [c["text"] for c in s.captions] == ["두 번째 안내입니다."]
            assert s._held_head is None and call.keys == ["1"]            # 보류 해제·불필요한 추가 키 없음
        finally:
            if not s.ended:
                await s.action({"type": "cancel"})
    run(go())


def test_inband_keys_return_no_remaining_tone():
    """톤을 실시간 속도로 다 보낸 뒤 반환하므로 '남은 톤 시간'은 0 (톤 길이를 반환하면 Session 이 그만큼 듣기를 한 번 더 막는 문제 방지)."""
    import types
    from telephony import ClawOpsBackend, ClawOpsCall

    sent = []

    class FakeMediaCall:
        async def send_audio(self, b):
            sent.append(len(b))

    session = types.SimpleNamespace(call_id="CA_test")
    ClawOpsBackend.LEGS["CA_test"] = types.SimpleNamespace(call=FakeMediaCall())
    try:
        call = ClawOpsCall(session, handlers=None)
        t0 = time.monotonic()
        rest = asyncio.run(call.send_keys("12"))
        took = time.monotonic() - t0
        assert rest == 0.0 and sum(sent) == 2 * 2000 and took >= 0.45     # 두 키 0.5초를 실시간으로 전송
    finally:
        ClawOpsBackend.LEGS.pop("CA_test", None)


def test_readback_right_after_long_input_is_heard_from_start():
    """Session 쪽 계약: 통화가 '남은 톤 시간 0'을 반환하면 전송 뒤 0.1초만 막고 바로 시작한 되읽기를 처음부터 청취.
    실시간 송출 뒤에도 톤 길이를 반환해 듣기를 또 막는 결함(되읽기 첫 숫자 누락)은 test_inband_keys_return_no_remaining_tone 에서 확인
    (이 시험의 가짜 통화는 항상 0 을 반환하므로 그 결함은 검출 불가)."""
    from audio import Segmenter, hum_ulaw

    async def go():
        s, call = await manual("통장 잔액 알고 싶어", stt=ScriptSTT(["안내"] * 3))   # 음성 경로(Segmenter)로 듣는 통화
        s.segmenter = Segmenter(threshold=50, silence_ms=1200)               # 시험음(허밍)의 낮은 음량 고려
        await s._send_keys("01055554444#", "user", secure=True)
        assert s.segmenter._mute_frames <= 5                                    # 톤이 끝난 뒤 0.1초만 막음
        for _ in range(25):                                                     # 보낸 뒤 0.5초 무음
            await s._on_audio(bytes([0xFF]) * 160)                              # μ-law 무음
        for _ in range(10):                                                     # 0.5초 뒤 ARS 되읽기 시작
            await s._on_audio(hum_ulaw())
        assert s.segmenter.speaking                                             # 시작부터 청취
    run(go())


def test_access_log_masks_session_token():
    """uvicorn 접속 로그의 WebSocket 주소는 쿼리까지 기록되므로 session_token 은 앞 6자만 남기고 가림.
    서버 시작(lifespan) 시 접속 로그 두 곳에 필터 부착, 다시 시작해도 하나만 유지."""
    import logging

    from fastapi.testclient import TestClient

    from app import TokenMaskFilter

    rec = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1,
                            '%s - "WebSocket %s" [accepted]',
                            ("127.0.0.1:5000", "/v1/sessions/ses_1/events?token=st_ABCDEFGHIJKLMNOP&x=1"), None)
    assert TokenMaskFilter().filter(rec)
    msg = rec.getMessage()
    assert "st_ABC" in msg and "GHIJKLMNOP" not in msg and "&x=1" in msg
    plain = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "GET /v1/health", (), None)
    assert TokenMaskFilter().filter(plain) and plain.getMessage() == "GET /v1/health"
    # 접속 로그 형식기에는 인자 5개를 그대로 전달 필요 (인자를 비우면 로그 한 줄마다 형식 오류 발생)
    from uvicorn.logging import AccessFormatter
    acc = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                            ("127.0.0.1:5000", "GET", "/v1/sessions/ses_1/result?token=st_ABCDEFGHIJKLMNOP", "1.1", 200),
                            None)
    assert TokenMaskFilter().filter(acc)
    line = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False).format(acc)
    assert "st_ABC" in line and "GHIJKLMNOP" not in line and "200" in line

    st = make_settings(sim_speed=0.05)
    for _ in range(2):
        with TestClient(create_app(st)):
            pass
    for name in ("uvicorn.access", "uvicorn.error"):
        assert sum(isinstance(f, TokenMaskFilter) for f in logging.getLogger(name).filters) == 1


def test_rejected_secure_input_is_asked_again_without_separate_prompt():
    """ARS 가 '잘못 누르셨습니다'와 재입력 안내를 한 단락으로 말해 입력 오류로만 분류되는 경우:
    입력 안내가 따로 오지 않아도 거부 직후 같은 항목으로 재질문. 그 뒤 '입력 시간 초과' 안내에는 요청 추가 생성 없음.
    ARS 가 입력 없이 메뉴로 돌아가면 요청을 닫고 메뉴 판단."""
    async def go():
        s, call = await secure_pending()
        first = s.pending
        await s.action({"type": "secure_input", "request_id": first["request_id"], "digits": "9999"})
        await s.on_paragraph(ERR)
        again = s.pending
        assert again and again["kind"] == "secure_input" and again["request_id"] != first["request_id"]
        n = len(ev(s, "request"))
        await s.on_paragraph(TIMEOUT_MSG)                            # 사용자가 아직 누르지 않음
        assert s.pending["request_id"] == again["request_id"] and len(ev(s, "request")) == n
        await s.on_paragraph(ROOT)                                   # ARS 가 처음 메뉴로 돌아감
        assert s.pending is None and call.keys[-1] == "3"
        assert "9999" not in str(s.hub.history)
    run(go())


# 받아쓰기 고침만 따로 묻기 (LLM_FIX_TRANSCRIPT=separate) -------------------------------------------------------
def _separate_fix(interp, delay: float, seen: list, ref: list, fail: bool = False):
    """모의 해석기에 따로 묻는 고침 부착: '잔액조회' → '잔액 조회'. 호출 시점의 경로 길이를 seen 에 기록."""
    interp.fix_mode = "separate"

    async def fix_prompt(text, options, kind):
        s = ref[0]
        seen.append({"kind": kind, "path_len": len(s.path) if s else None, "text": text})
        await asyncio.sleep(delay)
        if fail:
            raise RuntimeError("고침 실패 시험")
        applied = [("잔액조회", "잔액 조회")] if "잔액조회" in text else []
        return {"source": "model", "ms": int(delay * 1000), "proposed": list(applied), "applied": applied,
                "rejected": [], "shown": text.replace("잔액조회", "잔액 조회") if applied else None}

    interp.fix_prompt = fix_prompt


def _idx(s, pred):
    return next(i for i, e in enumerate(s.hub.history) if pred(e))


def test_separate_fix_names_shown_after_decision_in_order():
    """판단은 원래 이름으로 먼저 수행(키 전송 뒤 고침 호출), 자막 → 선택 상태 → 경로 순서로 고친 이름 표시."""
    async def go():
        m = await new_manager()
        seen, ref = [], [None]
        _separate_fix(m.deps.interp, 0.05, seen, ref)
        out = await m.create("15889999", "통장 잔액 알고 싶어")
        s = ref[0] = m.sessions[out["session_id"]]
        await wait_for(s, lambda s: ev(s, "path_updated"))
        i_cap = _idx(s, lambda e: e["event"] == "caption" and "3번" in e["data"]["text"])
        i_sel = _idx(s, lambda e: e["event"] == "status_changed" and "메뉴를 선택합니다" in (e["data"]["message"] or ""))
        i_path = _idx(s, lambda e: e["event"] == "path_updated")
        assert i_cap < i_sel < i_path
        assert "잔액 조회 3번" in s.hub.history[i_cap]["data"]["text"]
        assert "'잔액 조회' 메뉴를 선택합니다" in s.hub.history[i_sel]["data"]["message"]
        step = s.hub.history[i_path]["data"]
        assert step["selected"]["label"] == "잔액 조회" and "잔액 조회(3) 선택" in step["text"]
        assert s.path[0]["selected"]["label"] == "잔액 조회"           # 상태에 저장된 경로도 같은 사전
        assert s.layers[0].prompt.options[0].label == "잔액조회"        # 판단에 쓰는 이름은 원문
        root = next(x for x in seen if "잔액조회" in x["text"])
        assert root["kind"] == "menu" and root["path_len"] == 1        # 키 전송 뒤 고침 호출
        await s.action({"type": "cancel"})
        await wait_for(s, lambda s: ev(s, "ended"))
        assert s.hub.history[-1]["event"] == "ended"
        assert all(x["kind"] in ("menu", "info") for x in seen)        # 민감정보·확인 질문 등은 고침 대상 아님
    run(go())


def test_separate_fix_does_not_delay_key_and_ended_comes_last():
    """고침이 느려도 키는 바로 전송. 고침 대기 중 종료해도 마지막 자막 뒤에 ended 전송."""
    async def go():
        m = await new_manager()
        seen, ref = [], [None]
        _separate_fix(m.deps.interp, 0.6, seen, ref)
        out = await m.create("15889999", "통장 잔액 알고 싶어")
        s = ref[0] = m.sessions[out["session_id"]]
        await wait_for(s, lambda s: s.path)                            # 키 전송·경로 상태는 바로
        assert not any(e["event"] == "caption" and "3번" in e["data"]["text"] for e in s.hub.history)
        assert not ev(s, "path_updated")                               # 앱 이벤트는 자막 뒤 순서 대기
        assert not any("3번" in c["text"] for c in s.snapshot()["captions"])  # 보내기 전 자막은 상태 조회에 없음
        await s.action({"type": "cancel"})
        assert s.ended and not s.events_done()                          # 상태는 바로 종료, 이벤트는 남아 있음
        await wait_for(s, lambda s: ev(s, "ended"), timeout=5)
        assert s.events_done() and any("3번" in c["text"] for c in s.snapshot()["captions"])
        i_cap = _idx(s, lambda e: e["event"] == "caption" and "3번" in e["data"]["text"])
        assert i_cap < _idx(s, lambda e: e["event"] == "path_updated") < _idx(s, lambda e: e["event"] == "ended")
        assert s.hub.history[-1]["event"] == "ended"
    run(go())


def test_separate_fix_failure_sends_original_caption():
    """고침 처리 오류 시에도 원문 자막 전송, 흐름 유지."""
    async def go():
        m = await new_manager()
        seen, ref = [], [None]
        _separate_fix(m.deps.interp, 0.01, seen, ref, fail=True)
        out = await m.create("15889999", "통장 잔액 알고 싶어")
        s = ref[0] = m.sessions[out["session_id"]]
        await wait_for(s, lambda s: ev(s, "path_updated"))
        assert any("잔액조회 3번" in e["text"] for e in ev(s, "caption"))
        assert ev(s, "path_updated")[0]["selected"]["label"] == "잔액조회"
        await s.action({"type": "cancel"})
        await wait_for(s, lambda s: ev(s, "ended"))
    run(go())


def test_fix_words_and_display_fix_touch_only_heard_names():
    """화면 고침: 이미 고친 말의 재고침 없음, 분류 항목·이동 키 이름·서버 문구는 고침 대상 아님."""
    from types import SimpleNamespace

    from interpreter import Option, Prompt
    from session import Session, fix_words
    assert fix_words("카드 재발급", [("발급", "재발급")]) == "카드 재발급"
    assert fix_words("카드 발급 · 재발급 안내", [("발급", "재발급")]) == "카드 재발급 · 재발급 안내"
    p = Prompt(kind="menu", text="상담 연결은 0번, 이력 조회는 1번", options=[Option("0", "상담 연결", "상담원 연결"),
                                                                       Option("1", "입력 조회", "입력 조회")])
    p.display_fixes = [("상담", "상담원"), ("입력", "이력")]
    stub = SimpleNamespace(status_message="")
    step = {"text": "상담 연결·입력 조회 중 상담 연결(0) 선택", "options": [o.as_dict() for o in p.options]
            + [{"key": "*", "label": "이전 메뉴", "category": None}],
            "selected": {"key": "0", "label": "상담 연결", "category": "상담원 연결"}}
    Session._display_fix(stub, "path_updated", step, p)
    assert step["text"] == "상담원 연결·이력 조회 중 상담원 연결(0) 선택"
    assert step["selected"] == {"key": "0", "label": "상담원 연결", "category": "상담원 연결"}  # 분류 항목은 그대로
    assert [o["label"] for o in step["options"]] == ["상담원 연결", "이력 조회", "이전 메뉴"]
    assert [o["category"] for o in step["options"]] == ["상담원 연결", "입력 조회", None]
    other = {"text": "5 입력", "options": [], "selected": {"key": "5", "label": "5", "category": None}}
    Session._display_fix(stub, "path_updated", other, p)
    assert other["text"] == "5 입력"                                          # 서버가 만든 문구는 그대로
    req = {"message": p.text, "ars_prompt": p.text, "options": []}
    Session._display_fix(stub, "request", req, p)
    assert req["ars_prompt"] == "상담원 연결은 0번, 이력 조회는 1번" == req["message"]


def test_separate_fix_answer_uses_original_question():
    """재질문 요청은 고친 이름으로 표시, 답 해석(merge_answer)에 넘기는 질문은 원문."""
    async def go():
        m = await new_manager()
        seen, ref = [], [None]
        _separate_fix(m.deps.interp, 0.01, seen, ref)

        async def score(goal, options, path=None):  # 잔액조회만 애매한 점수 → low_confidence 재질문 (문구에 이름이 들어감)
            return {o.key: (0.45 if o.key == "3" else 0.0) for o in options}

        m.deps.interp.score = score
        out = await m.create("15889999", "적금 해지하고 싶어")
        s = ref[0] = m.sessions[out["session_id"]]
        await wait_for(s, lambda s: ev(s, "request") and "잔액 조회" in str(ev(s, "request")[-1]))
        req = ev(s, "request")[-1]
        q = s.ai_questions.get(req["request_id"], "")
        assert "잔액조회" in q and "잔액 조회" not in q                   # 판단 입력은 원문
        assert "잔액 조회" in req["message"] and s.pending["message"] == req["message"]  # 화면·상태는 고친 이름
        await s.action({"type": "cancel"})
        await wait_for(s, lambda s: ev(s, "ended"))
    run(go())


# 막다른 안내: 메뉴 키를 누른 뒤 업무 선택지 없이 이동 키만 남은 안내 ------------------------------
DEAD = ("이 업무는 휴대전화 본인인증이 필요합니다. 휴대전화로 다시 이용하시거나 홈페이지를 이용해 주십시오. "
        "이전 단계로 돌아가시려면 별표를 눌러 주십시오.")
DEAD_HOME = ("이 업무는 휴대전화 본인인증이 필요합니다. 휴대전화로 다시 이용하시거나 홈페이지를 이용해 주십시오. "
             "처음으로 가시려면 우물정자를 눌러 주십시오.")
DEAD_WAIT = 0.05  # 시험용으로 줄인 막다른 안내 판단 대기 시간


def _fake_answers_goal(s, verdicts):
    """answers_goal 교체. verdicts: 차례로 반환할 판정(마지막 값을 계속 사용), 예외 객체면 raise."""
    asked = []
    seq = list(verdicts) if isinstance(verdicts, list) else [verdicts]

    async def answers_goal(goal, text, chosen=None, siblings=None):
        asked.append((goal, text, chosen, siblings))
        v = seq[min(len(asked), len(seq)) - 1]
        if isinstance(v, Exception):
            raise v
        return v

    s.d.interp.answers_goal = answers_goal
    return asked


async def _dead_end_session(verdict, goal="결제일 확인하고 바꾸고 싶어", **kw):
    from interpreter import _norm
    kw.setdefault("dead_end_wait_s", DEAD_WAIT)
    s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10, **kw)
    s.d.interp.prompts[_norm(DEAD)] = (DEAD, {"kind": "info", "back_key": "*"})
    s.d.interp.prompts[_norm(DEAD_HOME)] = (DEAD_HOME, {"kind": "info", "home_key": "#"})
    asked = _fake_answers_goal(s, verdict)
    await s.on_paragraph(ROOT)                      # 목적 기본값: '잔액조회'(0.7)·'카드 업무'(0.55) 중 잔액조회(3) 누름
    assert len(call.keys) == 1
    return s, call, asked


def test_dead_end_that_answers_goal_becomes_result():
    """막다른 안내가 목적에 대한 답이면 잠시 대기 뒤 결과로 처리 (요약 → agent_done)."""
    async def go():
        s, call, asked = await _dead_end_session(True)
        await s.on_paragraph(DEAD)
        assert not asked and s.status == "listening"  # 즉시 판단 없음 (ARS 가 말을 이어 갈 가능성)
        await wait_for(s, lambda s: s.status == "agent_done")
        assert asked[0][2] == "잔액조회" and "본인인증" in asked[0][1]
        assert s.summary["kind"] == "result" and call.keys == ["3"]
    run(go())


def test_result_summary_input_leaves_out_menu_captions():
    """결과 요약 입력에서 지나온 메뉴 자막(선택지 목록 '잔액조회 3번 · …') 제외. 메뉴 자막의 번호를 결과로 적는 요약 방지
    (예: '…를 위해 1번을 선택해야 함'). 요약 원문 대체(_result_fallback)도 결과 안내 사용."""
    async def go():
        s, call, asked = await _dead_end_session(True)
        seen = []
        orig = s.d.interp.summarize_result

        async def spy(goal, caps):
            seen.append(list(caps))
            return await orig(goal, caps)

        s.d.interp.summarize_result = spy
        await s.on_paragraph(DEAD)
        await wait_for(s, lambda s: s.status == "agent_done")
        assert any("잔액조회 3번" in c["text"] for c in s.captions)       # 메뉴 자막은 화면에는 그대로 전송
        assert seen and seen[0] and all("3번" not in c for c in seen[0]) and "본인인증" in seen[0][-1]
        assert "본인인증" in s._result_fallback()["summary"]
    run(go())


def test_dead_end_on_wrong_branch_goes_back_and_skips_it_once():
    """다른 갈래 판정이면 이전 단계 키를 누르고 돌아간 메뉴에서 이번에만 그 갈래를 빼고 선택 (거절 목록에는 미포함)."""
    async def go():
        s, call, asked = await _dead_end_session(False)
        await s.on_paragraph(DEAD)
        await wait_for(s, lambda s: call.keys == ["3", "*"])
        assert s.layers == [] and s.pending is None and not any(s.nav.rejected.values())
        await s.on_paragraph(ROOT)                  # 돌아간 첫 메뉴: 잔액조회(3)는 빼고 카드 업무(5)
        assert call.keys == ["3", "*", "5"]
    run(go())


def test_dead_end_unsure_or_failure_asks_with_navigation_keys():
    """판단 불가(실패 포함) 시 이동 키를 선택지로 재질문, 고른 '이전 메뉴' 는 지나온 계층에서 한 단계 제거."""
    async def go():
        for verdict in (None, RuntimeError("시간 초과")):
            s, call, asked = await _dead_end_session(verdict)
            await s.on_paragraph(DEAD)
            req = await wait_for(s, lambda s: s.pending)
            assert req["reason"] == "unknown_prompt" and [o["key"] for o in req["options"]] == ["*"]
            assert s.ai_questions[req["request_id"]] == DEAD  # 글 답변은 ARS 원문 기준으로 해석
            await s.action({"type": "select", "request_id": req["request_id"], "key": "*"})
            assert call.keys == ["3", "*"] and s.layers == [] and s.dead_end_prompt is None
            await s.action({"type": "cancel"})
    run(go())


def test_navigation_only_info_without_prior_menu_key_is_not_a_dead_end():
    """메뉴를 누르기 전이거나, 키보다 먼저 시작된 조각·키에 잘린 안내의 뒷부분(cut_tail)은 막다른 안내로 보지 않음."""
    async def go():
        from interpreter import _norm
        s, call = await manual("결제일 확인하고 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10, dead_end_wait_s=DEAD_WAIT)
        s.d.interp.prompts[_norm(DEAD)] = (DEAD, {"kind": "info", "back_key": "*"})
        asked = _fake_answers_goal(s, False)
        await s.on_paragraph(DEAD)                  # 아직 누른 키 없음
        await asyncio.sleep(DEAD_WAIT * 4)
        assert not asked and call.keys == []
        await s.on_paragraph(ROOT)
        assert call.keys == ["3"]
        await s.on_paragraph(DEAD, started_at=s.last_key_at - 1.0)   # 키보다 먼저 시작된 조각(앞 안내의 꼬리)
        s.key_cut_speech = True                     # ARS 가 말하는 도중의 키 전송
        await s.on_paragraph(DEAD, started_at=s.last_key_at + 0.4)   # 키 직후 시작된 조각: 잘린 안내의 뒷부분 가능성
        await asyncio.sleep(DEAD_WAIT * 4)
        assert not asked and call.keys == ["3"] and s.pending is None and s.since_key_texts == []
    run(go())


def test_dead_end_judgment_is_dropped_when_ars_keeps_talking():
    """막다른 안내 후보 뒤 대기 시간 안에 새 안내(메뉴)가 오면 막다른 안내 판단 없이 그 메뉴를 판단."""
    async def go():
        s, call, asked = await _dead_end_session(False, dead_end_wait_s=0.3)
        await s.on_paragraph(DEAD)
        await s.on_paragraph(CARD)                  # ARS 가 쉬었다가 하위 메뉴를 이어 말함
        await asyncio.sleep(0.5)
        assert not asked and "*" not in call.keys
    run(go())


NO_INPUT = "입력하지 않으셨습니다."


def test_dead_end_is_judged_when_ars_says_no_input_before_the_wait_ends():
    """해석기 dead_end_on_timeout: 막다른 안내 후보의 판단을 기다리는 중 키 전송 없이 무입력 안내가 오면 앞 후보를 그때 판단
    (ARS 가 그 안내에서 입력을 기다렸다는 뜻). 속성이 없으면(다른 조합) 판단 폐기. 사이에 키를 눌렀으면 판단하지 않음."""
    from interpreter import _norm

    async def go():
        s, call, asked = await _dead_end_session(True, dead_end_wait_s=5.0)
        s.d.interp.prompts[_norm(NO_INPUT)] = (NO_INPUT, {"kind": "input_error"})
        s.d.interp.dead_end_on_timeout = True
        await s.on_paragraph(DEAD)
        assert not asked
        await s.on_paragraph(NO_INPUT)
        await wait_for(s, lambda s: s.status == "agent_done")
        assert asked and "본인인증" in asked[0][1] and s.summary["kind"] == "result" and call.keys == ["3"]
        await s.finish("user_canceled")

        s, call, asked = await _dead_end_session(True, dead_end_wait_s=5.0)
        s.d.interp.prompts[_norm(NO_INPUT)] = (NO_INPUT, {"kind": "input_error"})
        await s.on_paragraph(DEAD)
        await s.on_paragraph(NO_INPUT)
        await asyncio.sleep(0.3)
        assert not asked and s.status != "agent_done"
        await s.finish("user_canceled")

        s, call, asked = await _dead_end_session(True, dead_end_wait_s=5.0)
        s.d.interp.prompts[_norm(NO_INPUT)] = (NO_INPUT, {"kind": "input_error"})
        s.d.interp.dead_end_on_timeout = True
        await s.on_paragraph(DEAD)
        s.last_key_at = (s.last_key_at or 0) + 1.0        # 후보 뒤 키 전송 (무입력 안내가 그 키에 대한 반응일 가능성 있음)
        await s.on_paragraph(NO_INPUT)
        await asyncio.sleep(0.3)
        assert not asked
        await s.finish("user_canceled")
    run(go())


def test_dead_end_is_not_judged_while_waiting_for_an_agent():
    """상담원 연결 대기 중에 들은 이동 키 안내는 막다른 안내로 보지 않음."""
    async def go():
        s, call, asked = await _dead_end_session(False)
        s.set_status("waiting_agent")
        await s.on_paragraph(DEAD)
        await asyncio.sleep(DEAD_WAIT * 4)
        assert not asked and call.keys == ["3"] and s.status == "waiting_agent"
    run(go())


def test_dead_end_on_user_chosen_branch_asks_instead_of_going_back():
    """사용자가 직접 고른 갈래의 막다른 안내는 '다른 갈래' 판정이어도 되돌아가지 않고 재질문 (보류 시간이 지난 뒤 판단)."""
    async def go():
        s, call, asked = await _dead_end_session(False, goal="결제 관련", user_key_hold_s=0.2)
        await s.on_paragraph(CARD)
        assert s.pending and s.pending["reason"] == "ambiguous"
        await s.action({"type": "key", "key": "2"})  # 사용자가 결제 대금 조회(2)를 직접 누름 → 보류 시작
        await s.on_paragraph(DEAD)                  # 보류 시간 안에 막다른 안내
        req = await wait_for(s, lambda s: s.pending)
        assert asked and asked[-1][2] == "결제 대금 조회" and req["reason"] == "unknown_prompt"
        assert "*" not in call.keys[2:]
    run(go())


def test_dead_end_ask_keypad_navigation_key_leaves_that_branch():
    """막다른 안내 재질문 중 키패드로 누른 이동 키도 그 안내 기준으로 처리 (지나온 계층에서 한 단계 제거)."""
    async def go():
        s, call, asked = await _dead_end_session(None, user_key_hold_s=5)
        await s.on_paragraph(DEAD)
        await wait_for(s, lambda s: s.pending)
        await s.action({"type": "key", "key": "*"})
        assert call.keys == ["3", "*"] and s.layers == [] and s.pending is None
    run(go())


def test_dead_end_ask_text_answer_rejudges_with_revised_goal():
    """막다른 안내 재질문에 글로 답하면 고친 목적으로 그 안내를 다시 판단 (지난 메뉴로 판단하지 않음)."""
    async def go():
        s, call, asked = await _dead_end_session([None, True])
        await s.on_paragraph(DEAD)
        req = await wait_for(s, lambda s: s.pending)
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "잔액만 알면 돼요"})
        await wait_for(s, lambda s: s.status == "agent_done")
        assert len(asked) == 2 and s.summary["kind"] == "result" and call.keys == ["3"]
    run(go())


def test_dead_end_ask_home_rejects_only_the_dead_end_branch():
    """막다른 안내 재질문에서 '처음으로' 를 고르면 막다른 갈래만 거절로 기록 (첫 메뉴의 갈래 전체 거절 아님)."""
    async def go():
        s, call, asked = await _dead_end_session(None, goal="결제 관련")
        await s.on_paragraph(CARD)
        req = s.pending
        await s.action({"type": "select", "request_id": req["request_id"], "key": "2"})
        await s.on_paragraph(DEAD_HOME)
        req = await wait_for(s, lambda s: s.pending)
        assert [o["key"] for o in req["options"]] == ["#"]
        await s.action({"type": "select", "request_id": req["request_id"], "key": "#"})
        rejected = {k: v for k, v in s.nav.rejected.items() if v}
        assert list(rejected.values()) == [{"2"}] and s.layers == []
    run(go())


def test_separate_fix_skips_a_paragraph_that_became_a_result():
    """받아쓰기 고침(따로): 고침 호출 전에 안내가 결과로 바뀌었으면 고침 생략."""
    async def go():
        from interpreter import Prompt
        s, call = await manual("결제 관련")
        called = []

        async def fix_prompt(text, options=None, kind="info"):
            called.append(kind)
            return {"source": "llm", "applied": [("본인", "보닌")]}

        s.d.interp.fix_prompt = fix_prompt
        p = Prompt(kind="result", text=DEAD)
        cap = {"step": 1, "speaker": "ars", "text": DEAD, "at": ""}
        await s._fix_caption(p, cap, DEAD, None)
        assert not called and cap["text"] == DEAD and ev(s, "caption")
    run(go())


def test_key_pressed_during_dead_end_wait_cancels_the_judgment():
    """막다른 안내 판단 대기 중 사용자가 키를 누르면 그 판단 취소 (ARS 가 그 안내를 떠난 상태)."""
    async def go():
        s, call, asked = await _dead_end_session(False, dead_end_wait_s=0.2, user_key_hold_s=0.1)
        await s.on_paragraph(DEAD)
        await s.action({"type": "key", "key": "0"})
        await asyncio.sleep(0.6)                    # 대기·보류 시간이 모두 지나도
        assert not asked and "*" not in call.keys and s.pending is None
    run(go())


NHIS_ROOT = ["증명서 발급", "건강보험 자격 문의", "가상계좌 등 보험료 문의", "건강검진, 증진 문의", "노인장기요양보험 문의",
             "외국어 상담", "각종 서식 팩스 발송", "기타 문의", "상담사 연결", "말로하는 AI 상담"]
NHIS_DEAD = ("2024년 1월 22일부터 증명서 발급 시 간편인증을 통한 휴대전화 본인인증이 필요합니다. 휴대전화로 다시 이용하시거나 "
             "홈페이지 또는 모바일 앱을 통해 간편하게 발급받을 수 있습니다. 전 단계로 돌아가시려면 우물정자를 눌러주세요.")


def test_goal_words_keep_target_names_only():
    from navigator import goal_words
    assert goal_words("건강보험증 발급받고 싶어") == ["건강보험증"]
    assert goal_words("건강보험증을 새로 받고 싶어") == ["건강보험증"]
    assert goal_words("건강보험증 다시 발급받을래") == ["건강보험증"]       # 부사·'-ㄹ래' 는 대상 낱말 아님
    assert goal_words("결제일 확인하고 바꾸고 싶어") == ["결제일"]
    assert goal_words("우체국 업무시간이 궁금해") == ["우체국", "업무시간"]
    assert goal_words("이사 가는데 우편물을 새 주소로 받는 방법 알고 싶어") == ["이사", "우편물", "주소"]
    assert goal_words("카드 한도 올리고 싶어") == ["카드", "한도"]          # '도' 를 떼면 한 글자: 조사 아님
    assert goal_words("조회하고 싶어") == []


def test_better_sibling_on_real_dead_ends():
    """막다른 안내 메뉴와 그 변형: 목적 낱말이 지나온 메뉴·안내에 없고 다른 선택지 하나에만 있을 때 그 선택지 반환."""
    from navigator import better_sibling

    def nhis(goal):
        return better_sibling(goal, NHIS_ROOT[0], NHIS_DEAD, NHIS_ROOT[1:])

    for goal in ("건강보험증 발급", "건강보험증을 새로 받고 싶어", "건강보험증 다시 발급받고 싶어", "건강보험증 발급받을래"):
        assert nhis(goal) == "건강보험 자격 문의", goal       # 같은 목적의 표현 변형
    assert nhis("증명서 발급받고 싶어") is None                     # 고른 메뉴가 목적의 대상 → 모델 판단을 따름
    assert nhis("건강보험 자격득실 확인서 같은 증명서를 발급받고 싶어") is None
    assert nhis("보험료 납부 확인서 발급") is None
    # 고른 메뉴 이름의 조각이 목적 낱말 안에 포함되면 언급한 것으로 판단
    assert better_sibling("건강보험료 얼마인지 알고 싶어", NHIS_ROOT[2], "보험료 고지 내역은 홈페이지에서 확인할 수 있습니다.",
                          [x for x in NHIS_ROOT if x != NHIS_ROOT[2]]) is None
    post = ["국내 우편요금", "우편번호 안내", "접수불가품목", "우체국 4층 개인정보 수집 피해 예방 안내",
            "우체국 및 우편 고객센터 업무 시간", "손해배상제도", "주거이전 서비스", "국내 우편 반환 및 수추인 정보 변경 신청"]
    assert better_sibling("우체국 업무시간이 궁금해", post[4], "업무 시간은 평일 오전 9시부터 오후 6시까지", post[:4] + post[5:]) is None
    assert better_sibling("소포로 보낼 수 없는 물건이 뭔지 알고 싶어", "국내 우편",
                          "국내 우편 접수불가품목 안내입니다. 폭발할 수 있는 물질은 접수가 어렵습니다.", ["국제우편"]) is None
    assert better_sibling("하나카드 고객센터로 연결하고 싶어", "카드 분실", "본인인증이 필요합니다", ["상담원 연결"]) is None


def test_better_sibling_stays_quiet_on_related_branches():
    """맞는 갈래를 다른 갈래로 보지 않도록: 붙여 쓴 서술어의 줄기('해지하고'→'해지'), 메뉴 이름에 그대로 있는 '신고',
    지나온 윗단계 메뉴 이름, 일반 낱말로 시작하는 낱말, 앞 세 글자만 같은 다른 대상, 여러 선택지에 있는 중심 낱말."""
    from navigator import better_sibling
    assert better_sibling("카드 해지하고 싶어", "회원 탈퇴 및 해지", "해지 신청은 앱에서 가능합니다",
                          ["분실 도난 신고", "결제 대금", "카드 발급", "상담원 연결"]) is None
    assert better_sibling("보이스피싱 피해 신고", "금융사기 신고 및 지급정지", "지급정지는 앱에서 신청하실 수 있습니다",
                          ["보이스피싱 예방 안내", "상담원 연결"]) is None
    assert better_sibling("체크카드 발급", "체크카드 발급 신청 안내", "발급 신청은 앱에서 하실 수 있습니다",
                          ["체크카드 분실 신고", "상담원 연결"]) is None
    assert better_sibling("서비스센터 어디 있어", "매장 찾기", "가까운 매장은 앱에서 찾으실 수 있습니다",
                          ["부가서비스", "요금 조회", "상담원 연결"]) is None
    assert better_sibling("주민등록등본 떼고 싶어", "각종 증명서 발급", "각종 증명서는 정부24에서 발급받으실 수 있습니다",
                          ["주민등록증 재발급", "전입 신고", "상담원 연결"]) is None
    assert better_sibling("보험 해지하고 싶어", "계약 변경", "계약 변경은 홈페이지에서 하실 수 있습니다",
                          ["보험금 청구", "보험계약 대출", "보험료 납입", "상담원 연결"]) is None
    # 같은 뜻의 다른 말은 인식 불가 (한계): 이런 경우에도 결과 대신 재질문까지만 처리
    assert better_sibling("택배 언제 와", "배송 조회", "배송 조회는 운송장 번호로 앱에서 하실 수 있습니다",
                          ["택배 예약", "반품 접수", "상담원 연결"]) == "택배 예약"


async def _guard_session(verdicts, **kw):
    """목적 낱말이 고른 '잔액조회' 갈래와 맞지 않고 다른 선택지 '카드 업무' 에 있는 막다른 안내 상태의 (s, call, asked) 반환."""
    s, call, asked = await _dead_end_session(verdicts, **kw)
    s.goal = s.nav.goal = "카드 업무 확인하고 싶어"
    return s, call, asked


def test_dead_end_answer_verdict_is_asked_when_goal_names_another_option():
    """모델이 목적의 답으로 판정해도 목적 낱말이 지나온 메뉴·안내에 없고 다른 선택지 이름에 있으면 결과 대신 재질문.
    낱말 규칙은 같은 뜻의 다른 말을 모르므로 자동 되돌아가기 없음. 묻는 문구에 그 선택지 이름(메뉴가 말한 이름) 포함."""
    async def go():
        s, call, asked = await _guard_session(True)
        await s.on_paragraph(DEAD)
        req = await wait_for(s, lambda s: s.pending)
        assert req["reason"] == "unknown_prompt" and "찾는 업무가 '카드 업무' 쪽이면 이전 메뉴를 골라 주세요" in req["message"]
        assert "'카드 업무'" in s.ai_questions[req["request_id"]]   # 글 답 해석 시 그 선택지 이름도 참고
        assert call.keys == ["3"] and s.summary is None
        await s.action({"type": "select", "request_id": req["request_id"], "key": "*"})
        assert call.keys == ["3", "*"] and s.layers == []
    run(go())


def test_dead_end_word_check_does_not_turn_unsure_or_other_into_backtrack():
    """낱말 확인은 '답' 판정만 재질문으로 전환. 모델 판단 불가(실패 포함)면 그대로 재질문, '다른 갈래' 는 그대로 되돌아감."""
    async def go():
        for verdict in (None, RuntimeError("시간 초과")):
            s, call, asked = await _guard_session(verdict)
            await s.on_paragraph(DEAD)
            req = await wait_for(s, lambda s: s.pending)
            assert req["reason"] == "unknown_prompt" and call.keys == ["3"]
            await s.action({"type": "cancel"})
        s, call, asked = await _guard_session(False)
        await s.on_paragraph(DEAD)
        await wait_for(s, lambda s: call.keys == ["3", "*"])
    run(go())


def test_dead_end_word_check_skips_user_choices_and_text_answers():
    """사용자가 고른 갈래이거나 막다른 안내 재질문에 글로 답한 뒤에는 낱말 확인 미사용 (사용자 판단이 낱말 겹침보다 강한 근거)."""
    async def go():
        s, call, asked = await _dead_end_session(True, goal="결제 관련", user_key_hold_s=0.1)
        await s.on_paragraph(CARD)
        await s.action({"type": "key", "key": "2"})  # 사용자가 결제 대금 조회(2)를 직접 누름
        s.goal = s.nav.goal = "분실 신고하고 싶어"   # 고른 갈래와 맞지 않고 '카드 분실 신고' 에만 있는 낱말
        await s.on_paragraph(DEAD)
        await wait_for(s, lambda s: s.status == "agent_done")
        assert s.summary["kind"] == "result"
        s, call, asked = await _guard_session(True)
        await s.on_paragraph(DEAD)
        req = await wait_for(s, lambda s: s.pending)   # 낱말 확인으로 재질문
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "카드 업무 안내면 돼요"})
        await wait_for(s, lambda s: s.status == "agent_done")   # 글로 답한 뒤에는 모델 판단(답) 따름
        assert s.summary["kind"] == "result" and call.keys == ["3"]
    run(go())


def test_dead_end_auto_backtrack_once_per_search():
    """같은 탐색에서 이미 되돌아간 적이 있으면 막다른 안내가 '다른 갈래' 여도 다시 되돌아가지 않고 재질문 (두 갈래 왕복 방지)."""
    async def go():
        s, call, asked = await _dead_end_session(False)
        s.nav.backtracked = True
        await s.on_paragraph(DEAD)
        req = await wait_for(s, lambda s: s.pending)
        assert req["reason"] == "unknown_prompt" and call.keys == ["3"]
        assert "'잔액조회' 은 찾는 업무와 다른 것 같습니다" in req["message"]
    run(go())


def test_dead_end_after_unrecorded_user_key_asks_without_judging():
    """새 메뉴를 듣기 전에 사용자가 키패드로 누른 키(지난 메뉴의 선택지로 기록하지 않음) 뒤의 막다른 안내는
    지나온 계층이 지금 위치와 맞지 않으므로 마지막 계층의 메뉴 기준 판단 없이 재질문 (되돌아가기 없음)."""
    async def go():
        s, call, asked = await _dead_end_session(False, user_key_hold_s=0.1)
        await s.action({"type": "key", "key": "2"})
        await s.on_paragraph(DEAD)
        req = await wait_for(s, lambda s: s.pending)
        assert not asked and req["reason"] == "unknown_prompt" and call.keys == ["3", "2"]
    run(go())


def test_unchanged_goal_answer_to_word_check_question_keeps_the_check():
    """낱말 확인으로 물은 안내에 목적이 그대로인 답('네')을 하면 낱말 확인 유지 (같은 모델 판단으로 결과 처리되는 것 방지)."""
    async def go():
        s, call, asked = await _guard_session(True)

        async def merge_answer(goal, question, answer):
            return goal if answer.strip() in ("네", "예") else answer.strip()

        s.d.interp.merge_answer = merge_answer
        await s.on_paragraph(DEAD)
        req = await wait_for(s, lambda s: s.pending)
        await s.action({"type": "answer", "request_id": req["request_id"], "text": "네"})
        req2 = await wait_for(s, lambda s: s.pending)
        assert req2["request_id"] != req["request_id"] and "'카드 업무'" in req2["message"]
        assert s.summary is None and call.keys == ["3"]
    run(go())


def test_keep_alive_during_dead_end_question_uses_that_prompts_repeat_key_only():
    """막다른 안내 재질문 중 30초 무응답 처리는 그 안내의 다시 듣기 키만 사용. 지난 메뉴의 다시 듣기 키는 막다른 안내에서
    다른 뜻일 수 있으므로('#' 이 처음으로) 그 안내에 다시 듣기 키가 없으면 전송 안 함."""
    async def go():
        s, call, asked = await _dead_end_session(None)
        s.s.clarify_expire_s = 0.1
        await s.on_paragraph(DEAD_HOME)
        await wait_for(s, lambda s: s.pending)
        await asyncio.sleep(0.4)
        assert call.keys == ["3"] and s.pending is not None
    run(go())


def test_dead_end_other_verdict_is_asked_when_chosen_menu_names_the_goal():
    """모델이 다른 갈래로 판정해도 고른 메뉴 이름이 목적의 중심 낱말을 담으면 자동 되돌아가기 없이 재질문 (낱말 근거와 엇갈림)."""
    async def go():
        from navigator import chosen_names_goal
        assert chosen_names_goal("청약통장 해지하면 다시 만들수있나요", "청약통장 안내")
        assert not chosen_names_goal("건강보험증 발급", "증명서 발급")
        s, call, asked = await _dead_end_session(False)
        s.goal = s.nav.goal = "잔액조회 하고 싶어"
        await s.on_paragraph(DEAD)
        req = await wait_for(s, lambda s: s.pending)
        assert req["reason"] == "unknown_prompt" and call.keys == ["3"]
    run(go())


# --- 입력 오류 문장으로 시작하는 메뉴 재낭독: 해석기 분류와 관계없이 거부로 집계 ---------------------------------------------
KCW_GOAL = "대리인 선임 해임 신고서 팩스로 받을래"
KCW_MENU = ("서류 접수 및 처리 현황 조회는 1번, 서식 팩스 전송 서비스 신청은 2번, 팩스 수신 여부 확인은 3번, "
            "상담사 연결은 0번을 눌러주십시오. 다시 듣기를 원하시면 별표를 눌러주십시오.")
KCW_ERR = "입력시간이 지났습니다. " + KCW_MENU
KCW_END = "입력 시간이 지났습니다. 이용 방법을 확인하신 후 다시 이용해 주시기 바랍니다."
KCW_LABELS = [("1", "서류 접수 및 처리 현황 조회"), ("2", "서식 팩스 전송 서비스 신청"), ("3", "팩스 수신 여부 확인"), ("0", "상담사 연결")]
KCW_MENU_D = {"kind": "menu", "options": [{"key": k, "label": lb, "category": lb} for k, lb in KCW_LABELS],
              "repeat_key": "*", "agent_key": "0"}


async def _kcw_session(err_kind: str, err_options: bool):
    from interpreter import _norm
    s, call = await manual(KCW_GOAL, clarify_expire_s=5, clarify_drop_s=10)
    s.d.interp.prompts[_norm(KCW_MENU)] = (KCW_MENU, KCW_MENU_D)
    d = {k: (list(v) if isinstance(v, list) else v) for k, v in KCW_MENU_D.items()} if err_options else {}
    d["kind"] = err_kind
    s.d.interp.prompts[_norm(KCW_ERR)] = (KCW_ERR, d)
    s.d.interp.prompts[_norm(KCW_END)] = (KCW_END, {"kind": "input_error"})
    s.d.interp.goals[KCW_GOAL] = {"match": {"서식 팩스 전송 서비스 신청": 1.0}}
    return s, call


@pytest.mark.parametrize("err_kind,err_options", [("menu", True), ("confirm", True), ("input_error", True), ("input_error", False)])
def test_timeout_with_same_menu_counts_as_rejection_whatever_the_model_says(err_kind, err_options):
    """'입력시간이 지났습니다. (같은 메뉴)' 한 단락을 해석기가 메뉴로 분류해도 입력 오류로 분류할 때와 같게 처리: 첫 번째에서 같은 키를 바로
    다시 누르고(거부 1회, 계층은 하나), 다시 누른 키 뒤 두 번째에서 거부 2회 → 사용자에게 재질문
    (메뉴로 분류되면 거부로 세지 않아 묻지 않고 같은 메뉴 계층이 쌓이는 문제 방지)."""
    async def go():
        s, call = await _kcw_session(err_kind, err_options)
        await s.on_paragraph(KCW_MENU)
        assert call.keys == ["2"]
        await s.on_paragraph(KCW_ERR)
        assert call.keys == ["2", "2"]
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1]
        assert [layer.key for layer in s.layers] == ["2"] and s.pending is None
        await s.on_paragraph(KCW_ERR)
        assert call.keys == ["2", "2"]
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1, 2]
        assert s.pending and s.pending["reason"] == "input_rejected"
        assert [o["key"] for o in s.pending["options"]] == ["1", "2", "3", "0"]
        await s.on_paragraph(KCW_END)
        assert call.keys == ["2", "2"]
        await s.finish("user_canceled")
    run(go())


@pytest.mark.parametrize("err_kind,err_options", [("menu", True), ("input_error", False)])
def test_timeout_after_other_notice_on_pressed_menu_counts_as_rejection(err_kind, err_options):
    """키를 보낸 뒤 ARS 의 다른 안내가 먼저 와서(키에 대한 첫 반응이 끝남) 시간 초과 + 같은 메뉴가 와도, 방금 키를 누른 메뉴이면
    거부로 집계: 계층이 쌓이지 않고 두 번째 시간 초과에서 재질문 (메뉴로 다시 눌러 계층이 [2, 2] 로 쌓이고 세 번째에야 묻는 문제 방지)."""
    async def go():
        s, call = await _kcw_session(err_kind, err_options)
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph("잠시만 기다려 주십시오.")
        await s.on_paragraph(KCW_ERR)
        assert call.keys == ["2", "2"]
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1]
        assert [layer.key for layer in s.layers] == ["2"]
        await s.on_paragraph(KCW_ERR)
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1, 2]
        assert s.pending and s.pending["reason"] == "input_rejected"
        await s.finish("user_canceled")
    run(go())


def test_timeout_with_old_menu_after_new_menu_is_not_a_rejection_of_old_key():
    """키가 받아들여져 다음 메뉴를 들은 뒤 ARS 가 시간 초과로 앞 메뉴를 다시 읽으면 오래전 키의 거부로 세지 않음
    (거부는 키를 누른 메뉴가 아직 현재 메뉴일 때만)."""
    from interpreter import _norm
    sub = "서식 팩스 전송 서비스 신청입니다. 근로자 자격관리 서식은 1번, 보험가입지원 서식은 2번을 눌러주십시오."

    async def go():
        s, call = await _kcw_session("menu", True)
        s.d.interp.prompts[_norm(sub)] = (sub, {"kind": "menu", "options": [
            {"key": "1", "label": "근로자 자격관리 서식"}, {"key": "2", "label": "보험가입지원 서식"}]})
        s.d.interp.goals[KCW_GOAL]["match"].update({"근로자 자격관리 서식": 0.5, "보험가입지원 서식": 0.5})
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph(sub)            # 2 가 받아들여짐: 하위 메뉴 (1·2위 같은 점수 → 사용자에게 물음)
        assert call.keys == ["2"] and s.pending and s.pending["reason"] == "ambiguous"
        await s.on_paragraph(KCW_ERR)        # 답을 기다리는 사이 ARS 가 시간 초과로 첫 메뉴를 다시 읽음
        assert ev(s, "input_rejected") == []
        await s.finish("user_canceled")
    run(go())


def test_timeout_with_different_menu_after_key_means_key_was_accepted():
    """키 뒤 '입력시간이 지났습니다' 와 함께 다른 메뉴(첫 낭독을 못 들은 다음 메뉴)가 오면, 해석기가 메뉴로 분류해도 키는 받아들여진
    것으로 보고 계층을 유지한 채 새 메뉴로 판단 (입력 오류로 분류할 때의 처리와 같음)."""
    from interpreter import _norm
    sub = "서식 목록 듣기는 1번, 팩스 발송 신청은 2번을 눌러주십시오."
    err_sub = "입력시간이 지났습니다. " + sub

    async def go():
        s, call = await _kcw_session("menu", True)
        s.d.interp.prompts[_norm(err_sub)] = (err_sub, {"kind": "menu", "options": [
            {"key": "1", "label": "서식 목록 듣기"}, {"key": "2", "label": "팩스 발송 신청"}]})
        s.d.interp.goals[KCW_GOAL]["match"].update({"팩스 발송 신청": 0.9})
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph(err_sub)
        assert ev(s, "input_rejected") == []
        assert call.keys == ["2", "2"] and [layer.key for layer in s.layers] == ["2", "2"]
        await s.finish("user_canceled")
    run(go())


# --- 두 자리 메뉴 키 뒤 고른 선택지 안내 없이 입력 안내: 입력 요청에 경고 -----------------------------------------------------
KCW_INS = ("보험가입지원서식입니다. 보험관계 소멸신고서는 9번, 대리인 선임, 해임신고서는 10번을 눌러주십시오. "
           "다시 듣기를 원하시면 별표, 처음으로 돌아가시려면 우물정자를 눌러주십시오.")
KCW_FAX = "입력 시간이 지났습니다. 지역번호를 포함한 팩스 번호를 누르신 후 우물정자를 눌러주십시오."
KCW_FORM_OK = ("대리인 선임, 해임신고서 서식입니다. 서식 수량은 총 1매입니다. 지역번호를 포함한 팩스 번호를 누르신 후 "
               "우물정자를 눌러주십시오.")


async def _kcw_ins_session(key: str = "10"):
    from interpreter import _norm
    goal = "대리인 선임 해임신고서 팩스로 받을래"
    s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
    s.d.interp.prompts[_norm(KCW_INS)] = (KCW_INS, {"kind": "menu", "options": [
        {"key": "9", "label": "보험관계 소멸신고서"}, {"key": "10", "label": "대리인 선임, 해임신고서"}],
        "repeat_key": "*", "home_key": "#"})
    s.d.interp.prompts[_norm(KCW_FAX)] = (KCW_FAX, {"kind": "input_error"})
    s.d.interp.prompts[_norm(KCW_FORM_OK)] = (KCW_FORM_OK, {"kind": "sensitive", "label": "지역번호를 포함한 팩스번호",
                                                            "terminator": "#"})
    target = "대리인 선임, 해임신고서" if key == "10" else "보험관계 소멸신고서"
    s.d.interp.goals[goal] = {"match": {target: 1.0}}
    await s.on_paragraph(KCW_INS)
    assert call.keys == [key]
    return s, call


def test_two_digit_key_then_input_prompt_without_chosen_name_warns():
    """10 을 누른 뒤 10번 서식 안내 없이 '입력 시간이 지났습니다. … 팩스 번호를 누르신 후 우물정자를' 이 오면 입력 요청은 열되
    앞자리만 들어가 다른 메뉴로 갔을 가능성을 경고. 같은 안내가 다시 와도 경고는 그 요청에만 (새 요청을 만들지 않음)."""
    async def go():
        s, call = await _kcw_ins_session("10")
        await s.on_paragraph(KCW_FAX)
        req = s.pending
        assert req and req["kind"] == "secure_input" and req["terminator"] == "#"
        assert req["message"].startswith("10번을 누른 뒤 '대리인 선임, 해임신고서' 안내 없이 바로 입력 안내가 나왔습니다.")
        assert req["message"].endswith("직접 눌러 주세요. 입력하신 번호는 저장되지 않습니다.")
        assert s.unconfirmed_multi is None
        await s.on_paragraph(KCW_FAX)
        assert s.pending["request_id"] == req["request_id"]
        await s.finish("user_canceled")
    run(go())


def test_two_digit_key_then_chosen_name_then_input_prompt_has_no_warning():
    """고른 선택지 안내가 나온 뒤의 입력 안내(같은 단락이든 앞 단락이든)와 한 자리 키 뒤의 입력 안내에는 경고 없음."""
    from interpreter import _norm

    async def same_paragraph():
        s, call = await _kcw_ins_session("10")
        await s.on_paragraph(KCW_FORM_OK)
        assert s.pending and s.pending["kind"] == "secure_input"
        assert s.pending["message"].startswith("지역번호를 포함한 팩스번호를 직접 눌러 주세요.")
        await s.finish("user_canceled")

    async def earlier_paragraph():
        s, call = await _kcw_ins_session("10")
        intro = "대리인 선임, 해임신고서 서식입니다. 서식 수량은 총 1매입니다."
        s.d.interp.prompts[_norm(intro)] = (intro, {"kind": "info"})
        await s.on_paragraph(intro)
        await s.on_paragraph(KCW_FAX)
        assert s.pending and s.pending["kind"] == "secure_input"
        assert "안내 없이" not in s.pending["message"]
        await s.finish("user_canceled")

    async def one_digit():
        s, call = await _kcw_ins_session("9")
        await s.on_paragraph(KCW_FAX)
        assert s.pending and s.pending["kind"] == "secure_input"
        assert "안내 없이" not in s.pending["message"]
        await s.finish("user_canceled")

    run(same_paragraph())
    run(earlier_paragraph())
    run(one_digit())


# --- 시간 초과 재안내('입력시간이 지났습니다' + 메뉴)의 경계 상황 ---------------------------------------------------------------
@pytest.mark.parametrize("err_kind,err_options", [("menu", True), ("input_error", True), ("input_error", False)])
def test_timeout_after_notice_repeating_is_asked_on_second_rejection(err_kind, err_options):
    """'잠시만 기다려 주십시오' 뒤 시간 초과 + 같은 메뉴가 다시 누를 때마다 되풀이돼도 두 번째에 재질문. 일반 안내로는 거부 횟수를
    초기화하지 않음 (초기화하면 매번 1회로 세어 키 전송 한도(10회)에서야 묻게 됨)."""
    async def go():
        s, call = await _kcw_session(err_kind, err_options)
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph("잠시만 기다려 주십시오.")
        await s.on_paragraph(KCW_ERR)
        assert call.keys == ["2", "2"]
        await s.on_paragraph("잠시만 기다려 주십시오.")
        await s.on_paragraph(KCW_ERR)
        assert call.keys == ["2", "2"]
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1, 2]
        assert s.pending and s.pending["reason"] == "input_rejected"
        await s.finish("user_canceled")
    run(go())


@pytest.mark.parametrize("err_kind", ["menu", "input_error"])
def test_timeout_menu_after_result_is_not_a_rejection(err_kind):
    """결과 안내로 업무가 끝난 뒤 ARS 가 시간 초과로 같은 메뉴를 다시 읽으면, 받아들여진 키의 거부로 세지 않고 메뉴 자막 표시."""
    from interpreter import _norm
    res = "서식 팩스 전송이 완료되었습니다. 이용해 주셔서 감사합니다."

    async def go():
        s, call = await _kcw_session(err_kind, True)
        s.d.interp.prompts[_norm(res)] = (res, {"kind": "result"})
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph(res)
        assert not s.agent_active
        n = len(s.captions)
        await s.on_paragraph(KCW_ERR)
        assert ev(s, "input_rejected") == [] and call.keys == ["2"]
        assert len(s.captions) == n + 1 and s.current is not None and s.current.text.startswith("입력시간")
        await s.finish("user_canceled")
    run(go())


@pytest.mark.parametrize("err_kind", ["menu", "input_error"])
def test_queued_secure_value_is_dropped_when_ars_returns_to_menu_with_error(err_kind):
    """입력값을 받아 ARS 안내가 끝나기를 기다리는 중에 ARS 가 '입력시간이 지났습니다. (메뉴)' 로 돌아가면 입력값 폐기
    (메뉴 키로 나가는 것 방지). 해석기가 메뉴로 분류하든 선택지 있는 입력 오류로 분류하든 같음."""
    from interpreter import _norm
    fax = "지역번호를 포함한 팩스 번호를 누르신 후 우물정자를 눌러주십시오."

    async def go():
        s, call = await _kcw_session(err_kind, True)
        s.d.interp.prompts[_norm(fax)] = (fax, {"kind": "sensitive", "label": "팩스 번호", "terminator": "#"})
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph(fax)
        assert s.pending and s.pending["kind"] == "secure_input"
        call.speaking = True
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "0212345678"})
        assert s.secure_queued
        await s.on_paragraph(KCW_ERR)
        assert s.secure_abort
        call.speaking = False
        for _ in range(100):
            if not s.secure_queued:
                break
            await asyncio.sleep(0.02)
        assert all("0212345678" not in k for k in call.keys)
        await s.finish("user_canceled")
    run(go())


@pytest.mark.parametrize("err_kind", ["confirm", "menu", "input_error"])
def test_timeout_confirm_reread_stays_a_confirm_question(err_kind):
    """확인 질문에 답한 뒤 '입력시간이 지났습니다. (같은 확인 질문)' 이 오면 확인 질문으로 다시 답하고(answer_confirm), 확인 질문을
    메뉴 계층에 쌓지 않음."""
    from interpreter import _norm
    conf = "팩스 서식 신청을 진행하시겠습니까? 진행하시려면 1번, 취소하시려면 2번을 눌러주십시오."
    econf = "입력시간이 지났습니다. " + conf
    cd = {"kind": "confirm", "options": [{"key": "1", "label": "진행"}, {"key": "2", "label": "취소"}]}

    async def go():
        s, call = await _kcw_session("menu", True)
        s.d.interp.prompts[_norm(conf)] = (conf, cd)
        s.d.interp.prompts[_norm(econf)] = (econf, dict(cd, kind=err_kind))
        calls = []

        async def answer_confirm(goal, prompt):
            calls.append(prompt.text)
            return "1"
        s.d.interp.answer_confirm = answer_confirm
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph(conf)
        await s.on_paragraph(econf)
        assert call.keys == ["2", "1", "1"] and len(calls) == 2
        assert s.current.kind == "confirm" and [layer.key for layer in s.layers] == ["2"]
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1]
        await s.finish("user_canceled")
    run(go())


def test_timeout_with_different_menu_keeps_menu_caption():
    """키 뒤 '입력시간이 지났습니다' 와 함께 다른 메뉴가 오면(해석기는 메뉴로 분류) 입력 오류로 바꾸지 않아 새 메뉴 자막 전송."""
    from interpreter import _norm
    sub = "서식 목록 듣기는 1번, 팩스 발송 신청은 2번을 눌러주십시오."
    err_sub = "입력시간이 지났습니다. " + sub

    async def go():
        s, call = await _kcw_session("menu", True)
        s.d.interp.prompts[_norm(err_sub)] = (err_sub, {"kind": "menu", "options": [
            {"key": "1", "label": "서식 목록 듣기"}, {"key": "2", "label": "팩스 발송 신청"}]})
        s.d.interp.goals[KCW_GOAL]["match"].update({"팩스 발송 신청": 0.9})
        await s.on_paragraph(KCW_MENU)
        n = len(s.captions)
        await s.on_paragraph(err_sub)
        assert len(s.captions) == n + 1 and "팩스 발송 신청" in s.captions[-1]["text"]
        assert call.keys == ["2", "2"]
        await s.finish("user_canceled")
    run(go())


def test_backtrack_queue_key_is_cleared_when_error_paragraph_shows_next_menu():
    """'처음으로' 복귀 중 누른 대기 키 뒤 '입력시간이 지났습니다. (다음 메뉴)' 가 오면 그 키는 받아들여진 것으로 처리. 다음 거부 때 그
    대기 키를 다른 메뉴에서 다시 누르지 않음."""
    from interpreter import _norm
    sub = "서식 목록 듣기는 1번, 팩스 발송 신청은 2번을 눌러주십시오."
    err_sub = "입력시간이 지났습니다. " + sub
    wrong = "잘못 누르셨습니다. " + sub

    async def go():
        s, call = await _kcw_session("menu", True)
        sub_d = {"kind": "input_error", "options": [{"key": "1", "label": "서식 목록 듣기"}, {"key": "2", "label": "팩스 발송 신청"}]}
        s.d.interp.prompts[_norm(err_sub)] = (err_sub, dict(sub_d))
        s.d.interp.prompts[_norm(wrong)] = (wrong, dict(sub_d))
        s.d.interp.goals[KCW_GOAL]["match"].update({"팩스 발송 신청": 0.9})
        s.key_queue = ["2"]
        await s.on_paragraph(KCW_MENU)     # 대기 키 2 를 누름
        await s.on_paragraph(err_sub)      # 다른 메뉴: 2 는 받아들여짐 → 새 메뉴에서 2
        await s.on_paragraph(wrong)        # 새 메뉴의 2 거부 → 같은 메뉴에서 2 다시
        assert call.keys == ["2", "2", "2"] and s.queued_key is None
        await s.finish("user_canceled")
    run(go())


def test_timeout_menu_while_clarification_open_is_not_a_rejection():
    """재질문이 열린 채 30초 다시 듣기 키를 보낸 뒤 '입력시간이 지났습니다. (같은 메뉴)' 가 와도(해석기는 메뉴로 분류) 오래전 키의 거부로
    세지 않음 (열린 재질문을 새 요청으로 바꾸지 않음)."""
    from interpreter import _norm
    sub = "근로자 자격관리 서식은 1번, 보험가입지원 서식은 2번을 눌러주십시오. 다시 듣기를 원하시면 별표를 눌러주십시오."
    err_sub = "입력시간이 지났습니다. " + sub

    async def go():
        s, call = await _kcw_session("menu", True)
        d = {"kind": "menu", "options": [{"key": "1", "label": "근로자 자격관리 서식"}, {"key": "2", "label": "보험가입지원 서식"}],
             "repeat_key": "*"}
        s.d.interp.prompts[_norm(sub)] = (sub, d)
        s.d.interp.prompts[_norm(err_sub)] = (err_sub, dict(d))
        s.d.interp.goals[KCW_GOAL]["match"].update({"근로자 자격관리 서식": 0.5, "보험가입지원 서식": 0.5})
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph(sub)
        rid = s.pending["request_id"]
        await s._clarify_expire(rid)
        await s.on_paragraph(err_sub)
        assert ev(s, "input_rejected") == [] and s.pending["request_id"] == rid
        await s.finish("user_canceled")
    run(go())


# 두 자리 키 경고: 증거 낱말 고르기 · 수명 · 재알림 · 화면 고침 ------------------------------------------------------------------
_KCW_FORM_MENU = ("근로자 자격관리 서식은 1번, 보험가입지원 서식은 2번, 보험료 부과고지 서식은 3번, 요양보상 서식은 4번, "
                  "고용보험 서식은 10번을 눌러주십시오. 다시 듣기를 원하시면 별표, 처음으로 돌아가시려면 우물정자를 눌러주십시오.")
_KCW_FAX_TAIL = "지역번호를 포함한 팩스 번호를 누르신 후 우물정자를 눌러주십시오."


async def _two_digit_menu(goal, menu, opts, target, **kw):
    from interpreter import _norm
    s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10, **kw)
    s.d.interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": [{"key": k, "label": lb} for k, lb in opts],
                                               "repeat_key": "*", "home_key": "#"})
    s.d.interp.goals[goal] = {"match": {target: 1.0}}
    await s.on_paragraph(menu)
    return s, call


def _sensitive(s, text, label="지역번호를 포함한 팩스번호", term="#"):
    from interpreter import _norm
    s.d.interp.prompts[_norm(text)] = (text, {"kind": "sensitive", "label": label, "terminator": term})


def test_two_digit_warning_ignores_words_shared_with_sibling_options():
    """형제 선택지와 같이 쓰는 낱말('서식')은 고른 선택지를 들은 증거로 보지 않음: 10 을 눌렀는데 1번 서식 안내가 오면 경고."""
    opts = [("1", "근로자 자격관리 서식"), ("2", "보험가입지원 서식"), ("3", "보험료 부과고지 서식"), ("4", "요양보상 서식"),
            ("10", "고용보험 서식")]

    async def go(next_text, warned):
        s, call = await _two_digit_menu("고용보험 서식 팩스로 받을래", _KCW_FORM_MENU, opts, "고용보험 서식")
        assert call.keys == ["10"]
        _sensitive(s, next_text)
        await s.on_paragraph(next_text)
        assert s.pending and s.pending["kind"] == "secure_input"
        assert ("안내 없이" in s.pending["message"]) is warned
        await s.finish("user_canceled")

    run(go("근로자 자격관리 서식입니다. 서식 수량은 총 1매입니다. " + _KCW_FAX_TAIL, True))
    run(go("고용보험 서식입니다. 서식 수량은 총 1매입니다. " + _KCW_FAX_TAIL, False))


def test_two_digit_warning_ignores_words_of_the_input_field():
    """입력 항목 이름에 든 낱말('번호')은 증거로 보지 않음: '휴대전화 번호 변경 10번' 뒤 '주민등록번호 …' 입력 안내에 경고."""
    menu = "서류 접수 현황 조회는 1번, 팩스 수신 확인은 2번, 휴대전화 번호 변경은 10번을 눌러주십시오."
    opts = [("1", "서류 접수 현황 조회"), ("2", "팩스 수신 확인"), ("10", "휴대전화 번호 변경")]

    async def go():
        s, call = await _two_digit_menu("휴대전화 번호 바꾸고 싶어", menu, opts, "휴대전화 번호 변경")
        text = "주민등록번호 13자리를 누르신 후 우물정자를 눌러주십시오."
        _sensitive(s, text, label="주민등록번호 13자리")
        await s.on_paragraph(text)
        assert "안내 없이" in s.pending["message"]
        await s.finish("user_canceled")
    run(go())


@pytest.mark.parametrize("label,said", [("기타 문의", "기타 문의입니다. "), ("결과 확인", "결과 확인입니다. ")])
def test_two_digit_warning_uses_whole_name_when_words_are_generic(label, said):
    """낱말이 모두 일반 낱말이거나 잘려 쓸 낱말이 없으면 붙여 쓴 이름 전체로 판단: 그 이름을 말했으면 경고 없음."""
    menu = f"서류 접수 현황 조회는 1번, 팩스 수신 확인은 2번, {label}는 10번을 눌러주십시오."
    opts = [("1", "서류 접수 현황 조회"), ("2", "팩스 수신 확인"), ("10", label)]

    async def go():
        s, call = await _two_digit_menu("그 업무 하고 싶어", menu, opts, label)
        text = said + "주민등록번호 13자리를 누르신 후 우물정자를 눌러주십시오."
        _sensitive(s, text, label="주민등록번호 13자리")
        await s.on_paragraph(text)
        assert "안내 없이" not in s.pending["message"]
        await s.finish("user_canceled")
    run(go())


def test_two_digit_warning_survives_repeat_key_sent_by_no_response():
    """키 뒤 ARS 가 조용해 무응답 복구가 다시 듣기 키를 보내도 두 자리 키 확인 유지 (이어서 온 입력 안내에 경고)."""
    from interpreter import _norm

    async def go():
        goal = "대리인 선임 해임신고서 팩스로 받을래"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10, no_response_s=0.1)
        s.d.interp.prompts[_norm(KCW_INS)] = (KCW_INS, {"kind": "menu", "options": [
            {"key": "9", "label": "보험관계 소멸신고서"}, {"key": "10", "label": "대리인 선임, 해임신고서"}],
            "repeat_key": "*", "home_key": "#"})
        s.d.interp.prompts[_norm(KCW_FAX)] = (KCW_FAX, {"kind": "input_error"})
        s.d.interp.goals[goal] = {"match": {"대리인 선임, 해임신고서": 1.0}}
        await s.on_paragraph(KCW_INS)
        await asyncio.sleep(0.4)
        assert call.keys == ["10", "*"] and s.unconfirmed_multi is not None
        await s.on_paragraph(KCW_FAX)
        assert "안내 없이" in s.pending["message"]
        await s.finish("user_canceled")
    run(go())


def test_two_digit_warning_is_removed_when_renotified_prompt_names_the_choice():
    """경고를 붙인 입력 요청을 ARS 가 고른 선택지 이름과 함께 다시 안내하면(키가 제대로 들어감) 재알림에서 경고 제거."""
    async def go():
        s, call = await _kcw_ins_session("10")
        await s.on_paragraph(KCW_FAX)
        rid = s.pending["request_id"]
        assert "안내 없이" in s.pending["message"]
        await s.on_paragraph(KCW_FORM_OK)
        last = ev(s, "request")[-1]
        assert last["request_id"] == rid and "안내 없이" not in last["message"]
        assert last["message"].startswith("지역번호를 포함한 팩스번호를 직접 눌러 주세요.")
        await s.finish("user_canceled")
    run(go())


def test_two_digit_warning_name_follows_display_fix_of_the_menu():
    """화면용 받아쓰기 고침(따로 묻기)이 있으면 경고 속 선택지 이름도 경로 표시처럼 고친 이름으로 표시."""
    from interpreter import _norm
    menu = ("근로자 자격관리 서식은 1번, 보험가입지원 서식은 2번, 임금 채권 서식은 3번, 재증명 신청은 10번을 눌러주십시오. "
            "다시 듣기를 원하시면 별표, 처음으로 돌아가시려면 우물정자를 눌러주십시오.")
    opts = [("1", "근로자 자격관리 서식"), ("2", "보험가입지원 서식"), ("3", "임금 채권 서식"), ("10", "재증명 신청")]

    async def go():
        goal = "재증명 팩스로 받을래"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        interp = s.d.interp
        interp.fix_mode = "separate"

        async def fix_prompt(text, options=None, kind="info"):
            applied = [("재증명", "제증명")] if "재증명" in text else []
            return {"source": "model", "ms": 1, "proposed": list(applied), "applied": applied, "rejected": [],
                    "shown": text.replace("재증명", "제증명") if applied else None}

        interp.fix_prompt = fix_prompt
        interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": [{"key": k, "label": lb} for k, lb in opts],
                                               "repeat_key": "*", "home_key": "#"})
        interp.goals[goal] = {"match": {"재증명 신청": 1.0}}
        await s.on_paragraph(menu)
        await wait_for(s, lambda s: ev(s, "path_updated"))
        await wait_for(s, lambda s: not s._outbox_busy())
        assert ev(s, "path_updated")[-1]["selected"]["label"] == "제증명 신청"
        _sensitive(s, _KCW_FAX_TAIL)
        await s.on_paragraph(_KCW_FAX_TAIL)
        await wait_for(s, lambda s: not s._outbox_busy())
        msg = ev(s, "request")[-1]["message"]
        assert "'제증명 신청' 안내 없이" in msg and "재증명" not in msg
        await s.finish("user_canceled")
    run(go())


# 같은 종류 재질문에서 사용자가 고른 갈래 ------------------------------------------------------------------------------------
def test_user_chosen_rest_option_branch_asks_instead_of_backtracking():
    """같은 종류 재질문에서 사용자가 나머지 업무 선택지(7)를 골랐는데 하위 메뉴에 맞는 선택지가 없으면, 이전 메뉴로 스스로 돌아가
    1위 서식을 누르지 않고 하위 선택지와 이동 키로 재질문."""
    from interpreter import Option, Prompt, _norm
    from navigator import Layer

    forms = ("근로자 자격관리 서식은 1번, 보험가입지원 서식은 2번, 보험료 부과고지 서식은 3번, 재증명 신청은 7번을 눌러주십시오. "
             "다시 듣기를 원하시면 별표, 처음으로 돌아가시려면 우물정자를 눌러주십시오.")
    sub7 = "제증명 신청입니다. 가입 증명원은 1번, 보험료 납부 확인서는 2번을 눌러주십시오. 이전 메뉴는 별표를 눌러주십시오."
    goal = "대리인 선임 해임 신고서 팩스로 받을래"

    async def go():
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        names = ["근로자 자격관리 서식", "보험가입지원 서식", "보험료 부과고지 서식"]
        s.d.interp.prompts[_norm(forms)] = (forms, {"kind": "menu", "options": [
            {"key": str(i + 1), "label": n} for i, n in enumerate(names)] + [{"key": "7", "label": "재증명 신청"}],
            "repeat_key": "*", "home_key": "#"})
        s.d.interp.prompts[_norm(sub7)] = (sub7, {"kind": "menu", "options": [
            {"key": "1", "label": "가입 증명원"}, {"key": "2", "label": "보험료 납부 확인서"}], "back_key": "*"})
        s.d.interp.goals[goal] = {"match": {names[0]: 0.8, names[1]: 0.6, "재증명 신청": 0.2}}
        s.layers.append(Layer(Prompt(kind="menu", text="첫 메뉴", options=[Option("2", "서식 팩스")]), "2", "ai"))
        await s.on_paragraph(forms)
        assert s.pending and s.pending["reason"] == "ambiguous"
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "7"})
        await s.on_paragraph(sub7)
        assert call.keys == ["7"]
        assert s.pending and s.pending["reason"] == "low_confidence"
        keys = [o["key"] for o in s.pending["options"]]
        assert "1" in keys and "2" in keys and "*" in keys
        await s.finish("user_canceled")
    run(go())


# --- 단계별 거부 횟수·입력값 전송 대기·두 자리 키 경고의 증거 낱말·재알림 ------------------------------------------------------
def test_rejection_count_does_not_carry_into_next_step_after_info():
    """첫 메뉴 거부 1회 → 다시 누른 키가 받아들여져 일반 안내('잠시만 기다려 주십시오') → 다음 단계에서 누른 키의 첫 오류는
    1회로 집계 (앞 단계 거부 횟수가 남아 첫 오류에서 지난 메뉴 선택지로 재질문하지 않음)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10, user_key_hold_s=0.05,
                               max_revisit=3, max_presses=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(ERR)
        await s.on_paragraph(ROOT)
        await s.on_paragraph("잠시만 기다려 주십시오.")
        await s.action({"type": "key", "key": "3"})
        await s.on_paragraph(ERR)
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1, 1]
        assert s.pending is None
        await s.finish("user_canceled")
    run(go())


def test_queued_secure_value_is_sent_when_reprompt_of_same_field_has_option():
    """입력값 전송 대기 중 같은 입력 안내의 시간 초과 재안내에 번호 선택지('모르시면 1번')가 있어도(해석기가 선택지 있는 입력 오류로
    분류) 입력 단계에 머문 것이므로 입력값을 버리지 않음."""
    from interpreter import _norm
    sens = "카드번호 16자리를 누르신 후 우물정자를 눌러 주시고, 카드번호를 모르시면 1번을 눌러 주십시오."
    resens = "입력 시간이 초과되었습니다. " + sens

    async def go():
        s, call = await _kcw_session("menu", True)
        s.d.interp.prompts[_norm(sens)] = (sens, {"kind": "sensitive", "label": "카드번호 16자리", "terminator": "#"})
        s.d.interp.prompts[_norm(resens)] = (resens, {"kind": "input_error", "options": [{"key": "1", "label": "카드번호를 모르시면"}]})
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph(sens)
        call.speaking = True
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "1234567812345678"})
        await s.on_paragraph(resens)
        assert not s.secure_abort
        call.speaking = False
        for _ in range(100):
            if not s.secure_queued:
                break
            await asyncio.sleep(0.02)
        assert "1234567812345678#" in call.keys
        await s.finish("user_canceled")
    run(go())


@pytest.mark.parametrize("label,field,text,warned", [
    ("휴대전화 번호 변경", "휴대전화 번호", "변경하실 휴대전화 번호를 누르신 후 우물정자를 눌러주십시오.", False),
    ("수신 팩스번호 변경", "새로 등록할 팩스 번호", "새로 등록하실 팩스 번호를 누르신 후 우물정자를 눌러주십시오.", False),
    ("휴대전화 번호 변경", "주민등록번호 13자리", "주민등록번호 13자리를 누르신 후 우물정자를 눌러주십시오.", True),
])
def test_two_digit_warning_field_words_count_when_they_head_the_field(label, field, text, warned):
    """입력 항목 이름의 덩어리 첫머리로 나오는 낱말('휴대전화 번호' 의 '휴대전화')은 고른 항목의 증거로 사용 (바른 갈래에 경고 없음).
    다른 덩어리 안 뒷부분으로만 나오는 낱말('주민등록번호' 의 '번호')은 증거로 쓰지 않음 (틀린 갈래에 경고)."""
    menu = f"서류 접수 현황 조회는 1번, 팩스 수신 확인은 2번, {label}은 10번을 눌러주십시오."
    opts = [("1", "서류 접수 현황 조회"), ("2", "팩스 수신 확인"), ("10", label)]

    async def go():
        s, call = await _two_digit_menu("그 업무 하고 싶어", menu, opts, label)
        _sensitive(s, text, label=field)
        await s.on_paragraph(text)
        assert ("안내 없이" in s.pending["message"]) is warned
        await s.finish("user_canceled")
    run(go())


def test_two_digit_warning_removal_keeps_sent_event_in_record():
    """재알림에서 경고를 빼도 이미 보낸 첫 요청 이벤트(이벤트 기록·통화 기록)의 문구는 그대로 유지."""
    async def go():
        s, call = await _kcw_ins_session("10")
        await s.on_paragraph(KCW_FAX)
        await s.on_paragraph(KCW_FORM_OK)
        reqs = ev(s, "request")
        assert len(reqs) == 2 and reqs[0] is not reqs[1]
        assert "안내 없이" in reqs[0]["message"] and "안내 없이" not in reqs[1]["message"]
        await s.finish("user_canceled")
    run(go())


def test_two_digit_warning_removed_after_name_heard_in_separate_info():
    """경고를 단 요청이 열린 뒤 고른 선택지 이름을 일반 안내 단락으로 듣고, 입력 안내가 따로 다시 오면 그 재알림에서 경고 제거."""
    from interpreter import _norm
    intro = "대리인 선임, 해임신고서 서식입니다. 서식 수량은 총 1매입니다."
    tail = "지역번호를 포함한 팩스 번호를 누르신 후 우물정자를 눌러주십시오."

    async def go():
        s, call = await _kcw_ins_session("10")
        s.d.interp.prompts[_norm(intro)] = (intro, {"kind": "info"})
        _sensitive(s, tail)
        await s.on_paragraph(KCW_FAX)
        assert "안내 없이" in s.pending["message"]
        await s.on_paragraph(intro)
        await s.on_paragraph(tail)
        assert "안내 없이" not in ev(s, "request")[-1]["message"]
        await s.finish("user_canceled")
    run(go())


# 사용자 갈래: 글 답·앞선 되돌아가기·키패드 키 -------------------------------------------------------------------------------
_NAV_ROOT = "카드 업무는 1번, 대출 업무는 2번을 눌러주세요."
_NAV_CARD = "카드 업무입니다. 분실 신고는 1번, 결제 대금 조회는 2번을 눌러주세요. 이전 메뉴는 별표를 눌러주세요."
_NAV_BILL = "결제 대금 조회입니다. 이번 달 대금은 1번, 지난 달 대금은 2번을 눌러주세요. 이전 메뉴는 별표를 눌러주세요."
_NAV_LOAN = "대출 업무입니다. 신용대출은 1번, 주택담보대출은 2번을 눌러주세요. 이전 메뉴는 별표를 눌러주세요."
_NAV_MORT = "주택담보대출입니다. 금리 안내는 1번, 한도 조회는 2번을 눌러주세요. 이전 메뉴는 별표를 눌러주세요."


def _nav_setup(s, goals):
    from interpreter import _norm

    def menu(opts, back=None):
        return {"kind": "menu", "options": [{"key": k, "label": v} for k, v in opts], "back_key": back}
    P = s.d.interp.prompts
    P[_norm(_NAV_ROOT)] = (_NAV_ROOT, menu([("1", "카드 업무"), ("2", "대출 업무")]))
    P[_norm(_NAV_CARD)] = (_NAV_CARD, menu([("1", "분실 신고"), ("2", "결제 대금 조회")], back="*"))
    P[_norm(_NAV_BILL)] = (_NAV_BILL, menu([("1", "이번 달 대금"), ("2", "지난 달 대금")], back="*"))
    P[_norm(_NAV_LOAN)] = (_NAV_LOAN, menu([("1", "신용대출"), ("2", "주택담보대출")], back="*"))
    P[_norm(_NAV_MORT)] = (_NAV_MORT, menu([("1", "금리 안내"), ("2", "한도 조회")], back="*"))
    for g, m in goals.items():
        s.d.interp.goals[g] = {"match": m}


def _nav_root_layer():
    from interpreter import Option, Prompt
    from navigator import Layer
    return Layer(Prompt(kind="menu", text=_NAV_ROOT, options=[Option("1", "카드 업무"), Option("2", "대출 업무")]), "2", "ai")


def test_user_branch_text_answer_naming_previous_option_goes_back():
    """사용자가 고른 갈래(주택담보대출)에서 묻고, 사용자가 글로 이전 메뉴의 선택지('신용대출로 바꿀게요')를 말하면 이전 메뉴로
    돌아가 그 선택지를 누름 (같은 질문 반복 없음)."""
    async def go():
        g, want = "대출 관련 문의", "신용대출로 바꿀게요"
        s, call = await manual(g, clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {g: {"신용대출": 0.35, "주택담보대출": 0.32}, want: {"신용대출": 0.9}})
        s.layers.append(_nav_root_layer())
        await s.on_paragraph(_NAV_LOAN)
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "2"})
        await s.on_paragraph(_NAV_MORT)
        assert s.pending and s.pending["reason"] == "low_confidence" and call.keys == ["2"]
        await s.action({"type": "answer", "request_id": s.pending["request_id"], "text": want})
        assert call.keys == ["2", "*"] and s.pending is None
        await s.on_paragraph(_NAV_LOAN)
        assert call.keys == ["2", "*", "1"]
        await s.finish("user_canceled")
    run(go())


def test_user_branch_after_earlier_ai_backtrack_asks_again_instead_of_ending():
    """같은 탐색에서 AI 가 한 번 되돌아간 뒤 사용자가 고른 갈래에 맞는 선택지가 없고, 사용자가 글로 답해도 맞는 것이 없으면
    통화를 끝내지 않고 재질문."""
    async def go():
        g = "담보 대출 상품 문의"
        s, call = await manual(g, clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {g: {"카드 업무": 0.9, "대출 업무": 0.6, "신용대출": 0.55, "주택담보대출": 0.5}})
        await s.on_paragraph(_NAV_ROOT)
        await s.on_paragraph(_NAV_CARD)
        assert call.keys == ["1", "*"] and s.nav.backtracked
        await s.on_paragraph(_NAV_ROOT)
        await s.on_paragraph(_NAV_LOAN)
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "2"})
        await s.on_paragraph(_NAV_MORT)
        assert s.pending
        await s.action({"type": "answer", "request_id": s.pending["request_id"], "text": "잘 모르겠어요 다른 거요"})
        assert not s.ended and s.pending and s.pending["reason"] == "low_confidence"
        await s.finish("user_canceled")
    run(go())


def test_blind_keypad_key_branch_asks_instead_of_backtracking():
    """새 메뉴를 듣기 전에 사용자가 키패드로 누른 키(계층에 기록되지 않음) 뒤의 메뉴에 맞는 선택지가 없으면, 지난 계층을 기준으로
    스스로 되돌아가지 않고 재질문 (막다른 안내 판단과 같은 기준)."""
    async def go():
        g = "담보 대출 상품 문의"
        s, call = await manual(g, clarify_expire_s=5, clarify_drop_s=10, user_key_hold_s=0.2)
        _nav_setup(s, {g: {"카드 업무": 0.9, "대출 업무": 0.6}})
        await s.on_paragraph(_NAV_ROOT)
        await s.action({"type": "key", "key": "2"})
        await s.on_paragraph(_NAV_BILL)
        await asyncio.sleep(0.5)
        assert call.keys == ["1", "2"] and s.pending and s.pending["reason"] == "low_confidence"
        await s.finish("user_canceled")
    run(go())


def test_answer_revised_flag_is_used_by_one_decision_only():
    """글 답으로 목적이 바뀐 표시는 그 답 뒤의 판단 한 번에만 사용. 판단이 일찍 끝나도(요청 열림 등) 표시를 지워, 나중에 사용자가 고른
    갈래에서 맞는 선택지가 없을 때 스스로 되돌아가지 않고 재질문."""
    async def go():
        g = "대출 관련 문의"
        s, call = await manual(g, clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {g: {"신용대출": 0.35, "주택담보대출": 0.32}})
        s.layers.append(_nav_root_layer())
        await s.on_paragraph(_NAV_LOAN)
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "2"})
        s.answer_revised = True
        s.pending = {"request_id": "req_x", "kind": "clarification"}
        await s._decide(s.current, arrived=False)
        s.pending = None
        assert not s.answer_revised
        await s.on_paragraph(_NAV_MORT)
        assert call.keys == ["2"] and s.pending and s.pending["reason"] == "low_confidence"
        await s.finish("user_canceled")
    run(go())


def test_user_branch_text_answer_without_any_match_asks_again():
    """사용자가 고른 갈래에서 글로 답해 목적이 바뀌었지만 이전 메뉴에도 맞는 선택지가 없으면 통화를 끝내지 않고 재질문."""
    async def go():
        g = "담보 대출 상품 문의"
        s, call = await manual(g, clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {g: {"신용대출": 0.55, "주택담보대출": 0.5}})
        s.layers.append(_nav_root_layer())
        await s.on_paragraph(_NAV_LOAN)
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "2"})
        await s.on_paragraph(_NAV_MORT)
        assert s.pending
        await s.action({"type": "answer", "request_id": s.pending["request_id"], "text": "잘 모르겠어요 다른 거요"})
        assert not s.ended and s.pending and s.pending["reason"] == "low_confidence" and call.keys == ["2"]
        await s.finish("user_canceled")
    run(go())


# --- 거부 사이 안내·다른 항목 입력 안내·항목 이름 띄어쓰기·사용자 갈래 글 답 ---------------------------------------------------
def test_notice_between_rejection_and_menu_reread_keeps_rejection_count():
    """'잘못 누르셨습니다' → '메뉴를 다시 안내해 드리겠습니다' → 같은 메뉴 재낭독 → 다시 누른 키도 거부되면 두 번째 거부로 세어 재질문
    (거부 뒤 재낭독 전의 안내로 횟수를 초기화하지 않음)."""
    async def go():
        s, call = await manual("카드 결제일 바꾸고 싶어", clarify_expire_s=5, clarify_drop_s=10, max_revisit=3, max_presses=10)
        await s.on_paragraph(ROOT)
        await s.on_paragraph(ERR)
        await s.on_paragraph("메뉴를 다시 안내해 드리겠습니다.")
        await s.on_paragraph(ROOT)
        await s.on_paragraph(ERR)
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1, 2]
        assert s.pending and s.pending["reason"] == "input_rejected" and call.keys == ["5", "5"]
        await s.finish("user_canceled")
    run(go())


def test_queued_secure_value_is_dropped_when_reprompt_is_for_another_field():
    """입력값 전송 대기 중 다른 항목의 입력 안내(번호 선택지 붙음)로 넘어가면 앞 항목 값을 보내지 않음."""
    from interpreter import _norm
    sens = "카드번호 16자리를 누르신 후 우물정자를 눌러 주십시오."
    other = "입력 시간이 초과되었습니다. 주민등록번호 13자리를 누르신 후 우물정자를 눌러 주시고, 상담원 연결은 0번을 눌러 주십시오."

    async def go():
        s, call = await _kcw_session("menu", True)
        s.d.interp.prompts[_norm(sens)] = (sens, {"kind": "sensitive", "label": "카드번호 16자리", "terminator": "#"})
        s.d.interp.prompts[_norm(other)] = (other, {"kind": "input_error", "options": [{"key": "0", "label": "상담원 연결"}]})
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph(sens)
        call.speaking = True
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "1234567812345678"})
        await s.on_paragraph(other)
        assert s.secure_abort
        call.speaking = False
        for _ in range(100):
            if not s.secure_queued:
                break
            await asyncio.sleep(0.02)
        assert all("1234567812345678" not in k for k in call.keys)
        await s.finish("user_canceled")
    run(go())


@pytest.mark.parametrize("label,field,text,warned", [
    ("휴대전화 번호 변경", "사업장 관리 번호", "사업장 관리 번호를 누르신 후 우물정자를 눌러주십시오.", True),
    ("휴대전화 번호 변경", "사업장관리번호", "사업장 관리 번호를 누르신 후 우물정자를 눌러주십시오.", True),
    ("수신 팩스번호 변경", "새로등록할팩스번호", "새로 등록하실 팩스 번호를 누르신 후 우물정자를 눌러주십시오.", False),
    ("수신 팩스번호 변경", "접수 번호", "접수 번호를 누르신 후 우물정자를 눌러주십시오.", True),
])
def test_two_digit_warning_does_not_depend_on_field_spacing(label, field, text, warned):
    """항목 이름의 띄어쓰기('사업장 관리 번호'·'사업장관리번호')와 관계없이 같은 판정: 틀린 갈래는 경고, 바른 갈래는 경고 없음."""
    menu = f"서류 접수 현황 조회는 1번, 팩스 수신 확인은 2번, {label}은 10번을 눌러주십시오."
    opts = [("1", "서류 접수 현황 조회"), ("2", "팩스 수신 확인"), ("10", label)]

    async def go():
        s, call = await _two_digit_menu("그 업무 하고 싶어", menu, opts, label)
        _sensitive(s, text, label=field)
        await s.on_paragraph(text)
        assert ("안내 없이" in s.pending["message"]) is warned
        await s.finish("user_canceled")
    run(go())


def test_user_branch_text_answer_with_mid_score_or_no_back_key_asks_again():
    """사용자 갈래에서 글 답으로 목적이 바뀌었을 때, 지금 메뉴에 중간 점수 선택지만 있고 이전 메뉴 대안이 기준 미만이거나, 이전 메뉴
    대안은 높지만 돌아가는 키가 없으면 통화를 끝내지 않고 재질문."""
    from interpreter import _norm
    mort_noback = "주택담보대출입니다. 금리 안내는 1번, 한도 조회는 2번을 눌러주세요."

    async def mid():
        g, ans = "대출 관련 문의", "금리 말고 한도 쪽인데 잘 모르겠어요"
        s, call = await manual(g, clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {g: {"신용대출": 0.45, "주택담보대출": 0.4}, ans: {"한도 조회": 0.45, "신용대출": 0.2}})
        s.layers.append(_nav_root_layer())
        await s.on_paragraph(_NAV_LOAN)
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "2"})
        await s.on_paragraph(_NAV_MORT)
        await s.action({"type": "answer", "request_id": s.pending["request_id"], "text": ans})
        assert not s.ended and s.pending and call.keys == ["2"]
        await s.finish("user_canceled")

    async def no_back():
        g, ans = "대출 관련 문의", "신용대출로 바꿀게요"
        s, call = await manual(g, clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {g: {"신용대출": 0.45, "주택담보대출": 0.4}, ans: {"신용대출": 0.9}})
        s.d.interp.prompts[_norm(mort_noback)] = (mort_noback, {"kind": "menu", "options": [
            {"key": "1", "label": "금리 안내"}, {"key": "2", "label": "한도 조회"}]})
        s.layers.append(_nav_root_layer())
        await s.on_paragraph(_NAV_LOAN)
        await s.action({"type": "select", "request_id": s.pending["request_id"], "key": "2"})
        await s.on_paragraph(mort_noback)
        await s.action({"type": "answer", "request_id": s.pending["request_id"], "text": ans})
        assert not s.ended and s.pending and call.keys == ["2"]
        await s.finish("user_canceled")

    run(mid())
    run(no_back())


# --- 상담원 목적: 상담원 선택지가 없는 메뉴에서는 목적의 업무 부분으로 메뉴를 골라 들어가 그 안에서 상담원 연결 탐색 -----------------
_AGENT_CARD = "카드 업무입니다. 분실 신고는 1번, 결제 대금 조회는 2번, 상담원 연결은 0번을 눌러주세요. 이전 메뉴는 별표를 눌러주세요."


def _agent_setup(s, goal, topic_scores, full_scores):
    from interpreter import _norm
    from navigator import agent_topic
    _nav_setup(s, {})
    s.d.interp.prompts[_norm(_AGENT_CARD)] = (_AGENT_CARD, {"kind": "menu", "options": [
        {"key": "1", "label": "분실 신고"}, {"key": "2", "label": "결제 대금 조회"},
        {"key": "0", "label": "상담원 연결", "category": "상담원 연결"}], "back_key": "*", "agent_key": "0"})
    topic = agent_topic(goal)
    assert topic
    s.d.interp.goals[topic] = {"match": topic_scores}
    s.d.interp.goals[goal] = {"match": full_scores}


def test_agent_goal_goes_to_topic_menu_and_presses_agent_there():
    """목적 '카드 분실 관련 상담원 연결해줘': 첫 메뉴에 상담원 선택지가 없으면 업무 부분('카드 분실 관련')으로 카드 업무를 눌러 들어가고,
    그 안의 상담원 연결을 누름 (목적 전체로 매긴 업무 선택지 점수가 모두 낮아 매번 사용자에게 묻는 문제 방지)."""
    async def go():
        goal = "카드 분실 관련 상담원 연결해줘"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        _agent_setup(s, goal, {"카드 업무": 0.9, "대출 업무": 0.1}, {"상담원 연결": 0.9})
        await s.on_paragraph(_NAV_ROOT)
        assert call.keys == ["1"] and s.pending is None
        await s.on_paragraph(_AGENT_CARD)
        assert call.keys == ["1", "0"]
        await s.finish("user_canceled")
    run(go())


def test_agent_goal_topic_unclear_asks_with_no_agent_message():
    """업무 부분으로도 메뉴를 정하지 못하면(점수 낮음) 상담원 선택지가 없다고 알리며 재질문. 업무 부분이 없는 목적('상담사랑 통화하고
    싶어')도 같음 (test_agent_goal_without_agent_option_says_so)."""
    from navigator import NO_AGENT_ASK

    async def go():
        goal = "카드 분실 관련 상담원 연결해줘"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        _agent_setup(s, goal, {"카드 업무": 0.35, "대출 업무": 0.3}, {})
        await s.on_paragraph(_NAV_ROOT)
        assert call.keys == [] and s.pending and s.pending["message"] == NO_AGENT_ASK
        await s.finish("user_canceled")
    run(go())


# --- 상담원 연결 요청(앱의 상담원 연결 버튼, 세션 생성 agent=true): 함께 입력한 업무로 목적 구성, 첫 메뉴에 없으면 업무 메뉴로 진행 ------------
def test_agent_request_goal_from_business():
    """상담원 연결 버튼으로 건 통화의 목적: 업무 뒤에 '상담원 연결' 을 붙임. 이미 상담원 말이 있으면 그대로, 업무가 없으면 '상담원 연결'."""
    from navigator import agent_goal, agent_topic
    assert agent_goal("카드 분실") == "카드 분실 상담원 연결" and agent_topic(agent_goal("카드 분실")) == "카드 분실"
    assert agent_goal(" 카드 분실 관련 상담원 연결해줘 ") == "카드 분실 관련 상담원 연결해줘"
    assert agent_goal("") == agent_goal(None) == agent_goal("   ") == "상담원 연결"
    assert agent_topic(agent_goal("")) == ""


def test_agent_request_session_goal_and_empty_business():
    """agent=true 면 업무를 입력하지 않아도 발신 가능(목적 '상담원 연결'), 목적은 '<업무> 상담원 연결'. agent 가 아니면 목적 필수.
    첫 번호를 누르기 전 목적을 다시 말해도 상담원 연결 요청 유지."""
    async def go():
        m = await new_manager()
        out = await m.create("15889999", "카드 분실", agent=True)
        s = m.sessions[out["session_id"]]
        assert s.goal == "카드 분실 상담원 연결" and s.nav.goal == s.goal and s.agent_request
        r = await s.action({"type": "revise_goal", "text": "결제 대금 조회"})
        assert r["accepted"] and s.goal == "결제 대금 조회 상담원 연결"
        await s.finish("user_canceled")
        out = await m.create("15889999", "", agent=True)
        s2 = m.sessions[out["session_id"]]
        assert s2.goal == "상담원 연결"
        await s2.finish("user_canceled")
        with pytest.raises(ApiError):
            await m.create("15889999", "")
    run(go())


def test_agent_request_follows_business_menu_when_first_menu_has_no_agent():
    """상담원 연결 버튼 + 업무 '카드 분실 관련': 첫 메뉴에 상담원 선택지가 없으면 그 업무 메뉴(카드 업무)로 들어가 상담원 연결을 누름."""
    async def go():
        s, call = await manual("카드 분실 관련", agent=True, clarify_expire_s=5, clarify_drop_s=10)
        assert s.goal == "카드 분실 관련 상담원 연결"
        _agent_setup(s, s.goal, {"카드 업무": 0.9, "대출 업무": 0.1}, {"상담원 연결": 0.9})
        await s.on_paragraph(_NAV_ROOT)
        assert call.keys == ["1"] and s.pending is None
        await s.on_paragraph(_AGENT_CARD)
        assert call.keys == ["1", "0"]
        await s.finish("user_canceled")
    run(go())


def test_agent_request_api_field():
    """REST: agent=true 는 goal 없이도 201, agent 가 없으면 goal 필수(400). 기록 메타에 상담원 연결 요청 기록."""
    from fastapi.testclient import TestClient

    st = make_settings(sim_speed=0.05)
    with TestClient(create_app(st)) as c:
        h = {"X-App-Key": st.app_key}
        r = c.post("/v1/sessions", json={"phone_number": "1588-9999", "agent": True}, headers=h)
        assert r.status_code == 201, r.text
        sid, tok = r.json()["session_id"], r.json()["session_token"]
        assert c.post(f"/v1/sessions/{sid}/actions", json={"type": "cancel"},
                      headers={"Authorization": f"Bearer {tok}"}).status_code == 200
        r = c.post("/v1/sessions", json={"phone_number": "1588-9999", "agent": "yes"}, headers=h)
        assert r.status_code == 400  # agent 는 true 일 때만 (목적 없음)


# --- '처음으로' 복귀 대기열: 중간 안내 뒤 거부된 대기 키는 다음 대기 키 대신 다시 누름 ----------------------------------------------
def test_backtrack_queue_key_rejected_after_notice_is_pressed_again():
    """복귀 대기열 [2, 1] 에서 첫 메뉴에 2 를 누른 뒤 '잠시만 기다려 주십시오' 와 '입력시간이 지났습니다. (같은 메뉴)' 가 오면, 2 를
    다시 누름 (중간 안내에서 대기 키가 지워져 다음 대기 키 1 을 첫 메뉴에서 누르는 문제 방지)."""
    async def go():
        s, call = await _kcw_session("input_error", True)
        s.key_queue = ["2", "1"]
        await s.on_paragraph(KCW_MENU)
        assert call.keys == ["2"] and s.queued_key == "2"
        await s.on_paragraph("잠시만 기다려 주십시오.")
        await s.on_paragraph(KCW_ERR)
        assert call.keys == ["2", "2"] and s.key_queue == ["1"]
        assert [d["retry_count"] for d in ev(s, "input_rejected")] == [1]
        await s.finish("user_canceled")
    run(go())


def test_backtrack_queue_key_is_not_reinserted_after_user_key():
    """복귀 중 대기 키를 누른 뒤 사용자가 키패드로 다른 키를 눌렀다가 그 키가 거부되면, 앞의 대기 키를 다시 누를 목록에 넣지 않음."""
    async def go():
        s, call = await _kcw_session("input_error", True)
        s.key_queue = ["2", "1"]
        await s.on_paragraph(KCW_MENU)
        await s.action({"type": "key", "key": "3"})
        assert s.queued_key is None
        await s.on_paragraph(KCW_ERR)
        assert "2" not in s.key_queue and call.keys[:2] == ["2", "3"] and call.keys.count("2") == 1
        await s.finish("user_canceled")
    run(go())


# --- 상담원 목적 경로: 업무 메뉴 안의 결과 안내·상담원 선택지 판정·탐색 한도 ----------------------------------------------------
def test_agent_seek_result_is_not_task_done_and_asks_after_wait(monkeypatch):
    """업무 부분으로 들어간 갈래에서 결과 안내를 들어도 업무 완료로 끝내지 않음. 잠시 새 안내가 없으면 상담원 연결을 찾지 못했다고 재질문."""
    import session as S
    from interpreter import _norm
    monkeypatch.setattr(S, "AGENT_SEEK_RESULT_WAIT_S", 0.05)
    bill = "결제 대금 조회입니다. 이번 달 결제 예정 금액 안내는 1번을 눌러주세요."
    res = "이번 달 결제 예정 금액은 삼십오만 이천 원입니다."

    async def go():
        goal = "카드 결제 대금 관련 상담원 연결해줘"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        _agent_setup(s, goal, {"카드 업무": 0.9, "대출 업무": 0.1, "이번 달 결제 예정 금액 안내": 0.9}, {})
        s.d.interp.prompts[_norm(bill)] = (bill, {"kind": "menu", "options": [{"key": "1", "label": "이번 달 결제 예정 금액 안내"}]})
        s.d.interp.prompts[_norm(res)] = (res, {"kind": "result"})
        await s.on_paragraph(_NAV_ROOT)
        await s.on_paragraph(bill)
        assert call.keys == ["1", "1"]
        await s.on_paragraph(res)
        assert s.agent_active and s.status != "agent_done" and s.summary is None
        await wait_for(s, lambda s: s.pending)
        assert s.pending["message"] == S.AGENT_SEEK_RESULT_ASK
        await s.finish("user_canceled")
    run(go())


def test_agent_seek_result_followed_by_agent_menu_presses_agent(monkeypatch):
    """업무 부분으로 들어간 갈래의 결과 안내 뒤에 상담원 연결 선택지가 있는 메뉴가 오면 그 선택지를 누름."""
    import session as S
    from interpreter import _norm
    monkeypatch.setattr(S, "AGENT_SEEK_RESULT_WAIT_S", 2.0)
    res = "분실 신고가 접수되었습니다."
    after = "다시 들으시려면 1번, 상담원 연결은 0번을 눌러주세요."

    async def go():
        goal = "카드 분실 관련 상담원 연결해줘"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        _agent_setup(s, goal, {"카드 업무": 0.9, "대출 업무": 0.1}, {"상담원 연결": 0.9})
        s.d.interp.prompts[_norm(res)] = (res, {"kind": "result"})
        s.d.interp.prompts[_norm(after)] = (after, {"kind": "menu", "options": [
            {"key": "1", "label": "다시 듣기"}, {"key": "0", "label": "상담원 연결", "category": "상담원 연결"}], "agent_key": "0"})
        await s.on_paragraph(_NAV_ROOT)
        await s.on_paragraph(res)
        await s.on_paragraph(after)
        assert call.keys == ["1", "0"] and s.summary is None
        await s.finish("user_canceled")
    run(go())


def test_agent_goal_follows_original_score_for_unnamed_agent_option():
    """'상담을 원하시면 0번' 처럼 이름 정규식이 상담원 선택지로 보지 못한 선택지를 목적 전체로 매긴 점수가 맞게 고르면 그대로 누름
    (업무 부분으로 점수를 다시 매겨 업무를 누르지 않음)."""
    from interpreter import _norm
    menu = "카드 분실 접수는 1번, 포인트 조회는 2번, 상담을 원하시면 0번을 눌러주세요."

    async def go():
        goal = "카드 분실 관련 상담원 연결해줘"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        s.d.interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": [
            {"key": "1", "label": "카드 분실 접수"}, {"key": "2", "label": "포인트 조회"}, {"key": "0", "label": "상담"}]})
        s.d.interp.goals[goal] = {"match": {"상담": 0.9, "카드 분실 접수": 0.3}}
        s.d.interp.goals["카드 분실 관련"] = {"match": {"카드 분실 접수": 0.9}}
        await s.on_paragraph(menu)
        assert call.keys == ["0"]
        await s.finish("user_canceled")
    run(go())


def test_agent_goal_menu_with_agent_unavailable_notice_asks():
    """메뉴 단락에 상담원 말이 있는데 상담원 선택지가 없으면('상담사 연결은 평일 업무시간에 …') 업무 부분으로 들어가지 않고 알리며 재질문."""
    from interpreter import _norm
    from navigator import NO_AGENT_ASK
    menu = ("지금은 셀프 발급 및 조회 업무만 가능합니다. 상담사 연결은 평일 업무시간에 다시 걸어 주십시오. "
            "보험료 조회는 1번, 자격 확인은 2번을 눌러주세요.")

    async def go():
        goal = "건강보험료 관련 상담사 연결해줘"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        s.d.interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": [
            {"key": "1", "label": "보험료 조회"}, {"key": "2", "label": "자격 확인"}]})
        s.d.interp.goals[goal] = {"match": {"보험료 조회": 0.4}}
        s.d.interp.goals["건강보험료 관련"] = {"match": {"보험료 조회": 0.9}}
        await s.on_paragraph(menu)
        assert call.keys == [] and s.pending and s.pending["message"] == NO_AGENT_ASK
        await s.finish("user_canceled")
    run(go())


def test_agent_topic_keeps_business_words():
    from navigator import agent_topic
    assert agent_topic("교통사고 접수 상담원 연결해줘") == "교통사고 접수"
    assert agent_topic("카드 정지 해제 상담원 연결해줘") == "카드 정지 해제"
    assert agent_topic("대출 연장하고 싶은데 상담원 연결해줘").startswith("대출 연장하고")
    assert agent_topic("상담원 연결해줘") == "" and agent_topic("상담사랑 통화하고 싶어") == ""


def test_no_agent_question_answer_keeps_agent_goal():
    """'상담원 연결 선택지가 없다' 질문에 업무로 답하면(목적 합치기가 선택지 이름으로 바꿔도) 상담원 연결 목적 유지."""
    async def go():
        goal = "상담원 연결해줘"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {})
        await s.on_paragraph(_NAV_ROOT)
        assert s.pending
        await s.action({"type": "answer", "request_id": s.pending["request_id"], "text": "카드 업무"})
        assert "상담원" in s.goal and "카드 업무" in s.goal
        await s.finish("user_canceled")
    run(go())


def test_agent_topic_presses_are_limited_per_search():
    """상담원 선택지 없는 메뉴가 이어지면 업무 부분으로는 탐색당 두 번까지만 들어가고, 그 뒤에는 상담원을 찾지 못했다고 재질문."""
    from interpreter import _norm
    from navigator import NO_AGENT_ASK
    menus = [f"{i}단계 메뉴입니다. 카드 관련 안내 {i}은 1번, 기타 안내 {i}는 2번을 눌러주세요." for i in range(1, 4)]

    async def go():
        goal = "카드 관련 상담원 연결해줘"
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=10)
        match = {}
        for i, m in enumerate(menus, 1):
            s.d.interp.prompts[_norm(m)] = (m, {"kind": "menu", "options": [
                {"key": "1", "label": f"카드 관련 안내 {i}"}, {"key": "2", "label": f"기타 안내 {i}"}]})
            match[f"카드 관련 안내 {i}"] = 0.9
        s.d.interp.goals["카드 관련"] = {"match": match}
        for m in menus:
            await s.on_paragraph(m)
        assert call.keys == ["1", "1"] and s.pending and s.pending["message"] == NO_AGENT_ASK
        await s.finish("user_canceled")
    run(go())


# --- 복귀 대기열: 재질문 시 중단·다시 듣기 뒤 재낭독·따로 온 오류 단락 -----------------------------------------------------------
def test_backtrack_queue_stops_when_rejection_question_opens():
    """복귀 중 대기 키가 두 번 거부되어 재질문을 열면 복귀 중단 (답 뒤에 거부된 대기 키를 또 누르는 것 방지)."""
    async def go():
        s, call = await _kcw_session("input_error", True)
        s.key_queue = ["2", "1"]
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph("잠시만 기다려 주십시오.")
        await s.on_paragraph(KCW_ERR)
        await s.on_paragraph("잠시만 기다려 주십시오.")
        await s.on_paragraph(KCW_ERR)
        assert s.pending and s.pending["reason"] == "input_rejected"
        assert s.key_queue == [] and s.queued_key is None
        await s.finish("user_canceled")
    run(go())


def test_backtrack_queue_key_repressed_after_repeat_key_reread():
    """복귀 대기 키 뒤 ARS 가 조용해 다시 듣기 키를 보냈고 같은 메뉴를 다시 읽으면, 다음 대기 키 대신 그 키를 다시 누름."""
    async def go():
        s, call = await _kcw_session("input_error", True)
        s.s.no_response_s = 0.1
        s.key_queue = ["2", "1"]
        await s.on_paragraph(KCW_MENU)
        await asyncio.sleep(0.4)
        assert call.keys == ["2", "*"]
        await s.on_paragraph(KCW_MENU)
        assert call.keys == ["2", "*", "2"] and s.key_queue == ["1"]
        await s.finish("user_canceled")
    run(go())


def test_backtrack_queue_key_repressed_after_separate_error_paragraph():
    """복귀 대기 키 뒤 중간 안내와 따로 온 '입력시간이 지났습니다.' 단락 다음에 같은 메뉴를 다시 읽으면, 그 키를 다시 누름."""
    from interpreter import _norm
    err = "입력시간이 지났습니다."

    async def go():
        s, call = await _kcw_session("input_error", True)
        s.d.interp.prompts[_norm(err)] = (err, {"kind": "input_error"})
        s.key_queue = ["2", "1"]
        await s.on_paragraph(KCW_MENU)
        await s.on_paragraph("잠시만 기다려 주십시오.")
        await s.on_paragraph(err)
        await s.on_paragraph(KCW_MENU)
        assert call.keys == ["2", "2"] and s.key_queue == ["1"]
        await s.finish("user_canceled")
    run(go())


# --- 첫 메뉴 전 결과 안내 보류 (인사말 뒤 이용 방법 안내를 결과로 보고 바로 업무 완료하는 문제 방지) ----------------
EARLY_RES = "진료 예약 및 변경은 홈페이지 또는 모바일 앱으로도 하실 수 있습니다."
EARLY_MENU = "진료 예약은 1번, 예약 확인은 2번을 눌러주세요."
EARLY_GOAL = "진료 예약 확인하고 싶어"


async def _early_session(**kw):
    from interpreter import _norm
    s, call = await manual(EARLY_GOAL, clarify_expire_s=5, clarify_drop_s=10, **kw)
    s.d.interp.prompts[_norm(EARLY_RES)] = (EARLY_RES, {"kind": "result"})
    s.d.interp.prompts[_norm(EARLY_MENU)] = (EARLY_MENU, {"kind": "menu", "options": [
        {"key": "1", "label": "진료 예약"}, {"key": "2", "label": "예약 확인"}]})
    s.d.interp.goals[EARLY_GOAL] = {"match": {"예약 확인": 0.9, "진료 예약": 0.2}}
    return s, call


def test_result_before_first_menu_is_held_and_following_menu_is_decided(monkeypatch):
    """첫 메뉴를 듣기 전의 결과 안내는 바로 업무 완료로 보지 않음. 이어서 메뉴가 오면 그 메뉴로 판단하고 결과는 폐기."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.2)

    async def go():
        s, call = await _early_session()
        await s.on_paragraph(EARLY_RES)
        assert s.summary is None and not s.result_heard and s.agent_active
        await s.on_paragraph(EARLY_MENU)
        assert call.keys == ["2"]
        await asyncio.sleep(0.4)
        assert s.summary is None and s.status != "agent_done" and "summary" not in [e["event"] for e in s.hub.history]
        await s.finish("user_canceled")
    run(go())


def test_result_before_first_menu_without_following_menu_becomes_result(monkeypatch):
    """첫 메뉴 전의 결과 안내 뒤 메뉴 없이 조용하면 결과로 확정. 뒤에 일반 안내(감사 인사)가 이어져도 같음."""
    import session as S
    from interpreter import _norm
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.1)
    thanks = "이용해 주셔서 감사합니다."

    async def go():
        for tail in (None, thanks):
            s, call = await _early_session()
            s.d.interp.prompts[_norm(thanks)] = (thanks, {"kind": "info"})
            await s.on_paragraph(EARLY_RES)
            if tail:
                await s.on_paragraph(tail)
                assert s.summary is None
            await wait_for(s, lambda s: s.summary is not None)
            assert s.status == "agent_done" and s.summary["kind"] == "result" and call.keys == []
            await s.finish("user_canceled")
    run(go())


def test_result_before_first_menu_then_hangup_completes_with_summary(monkeypatch):
    """첫 메뉴 전의 결과 안내를 확정하려고 기다리는 중에 ARS 가 끊으면 결과로 확정하고 요약을 보낸 뒤 completed 로 종료."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 5.0)

    async def go():
        s, call = await _early_session(hangup_flush_s=1.0)
        await s.on_paragraph(EARLY_RES)
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended, timeout=3)
        names = [e["event"] for e in s.hub.history]
        assert s.end_reason == "completed" and s.summary["kind"] == "result"
        assert names.index("summary") < names.index("ended")
    run(go())


def test_result_before_first_menu_then_agent_wait_is_not_result(monkeypatch):
    """첫 메뉴 전의 결과 안내 뒤 상담원 대기 안내가 오면 결과로 확정하지 않음."""
    import session as S
    from interpreter import _norm
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.1)
    wait = "상담사에게 연결해 드리겠습니다. 잠시만 기다려 주십시오."

    async def go():
        s, call = await _early_session()
        s.d.interp.prompts[_norm(wait)] = (wait, {"kind": "agent_wait"})
        await s.on_paragraph(EARLY_RES)
        await s.on_paragraph(wait)
        await asyncio.sleep(0.4)
        assert s.summary is None and not s.result_heard and s.status == "waiting_agent"
        await s.finish("user_canceled")
    run(go())


def test_result_before_first_menu_waits_while_ars_is_speaking(monkeypatch):
    """첫 메뉴 전 결과 안내 뒤 ARS 가 아직 말하는 중이면(다음 단락을 듣는 중) 확정 보류. 말이 끝나고 조용하면 확정."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.1)

    async def go():
        s, call = await _early_session()
        call.speaking = True
        await s.on_paragraph(EARLY_RES)
        await asyncio.sleep(0.4)
        assert s.summary is None and not s.result_heard
        call.speaking = False
        await wait_for(s, lambda s: s.summary is not None)
        assert s.status == "agent_done"
        await s.finish("user_canceled")
    run(go())


EARLY_ASK = "예약 번호 6자리를 누르신 후 우물정자를 눌러주십시오."
EARLY_ASK_RES = "고객님의 예약일은 다음 달 3일 오전 10시입니다."


def _early_secure(s):
    from interpreter import _norm
    s.d.interp.prompts[_norm(EARLY_ASK)] = (EARLY_ASK, {"kind": "sensitive", "label": "예약 번호", "terminator": "#"})
    s.d.interp.prompts[_norm(EARLY_ASK_RES)] = (EARLY_ASK_RES, {"kind": "result"})


def test_result_after_secure_input_is_immediate(monkeypatch):
    """메뉴 없이 입력 요청부터 오는 ARS 에서 민감정보를 보낸 뒤의 결과는 기다리지 않고 바로 확정 (보낸 키에 대한 결과).
    이어서 메뉴가 와도 결과 유지."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.2)

    async def go():
        s, call = await _early_session()
        _early_secure(s)
        await s.on_paragraph(EARLY_ASK)
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "123456"})
        await wait_for(s, lambda s: s.last_key_at is not None)
        await s.on_paragraph(EARLY_ASK_RES)
        assert s.result_heard and s.summary is not None and s.status == "agent_done" and not s.early_result_wait
        await s.on_paragraph(EARLY_MENU)
        await asyncio.sleep(0.3)
        assert s.summary["kind"] == "result" and s.status == "agent_done"
        assert "123456" not in str(s.hub.history)
        await s.finish("user_canceled")
    run(go())


def test_secure_input_during_early_hold_ends_hold(monkeypatch):
    """첫 메뉴 전 결과를 기다리는 중 입력 요청이 오고 사용자가 값을 보내면, 앞의 결과 안내로 업무를 끝내지 않음."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.3)

    async def go():
        s, call = await _early_session()
        _early_secure(s)
        await s.on_paragraph(EARLY_RES)
        await s.on_paragraph(EARLY_ASK)
        await s.action({"type": "secure_input", "request_id": s.pending["request_id"], "digits": "123456"})
        await wait_for(s, lambda s: s.last_key_at is not None)
        await asyncio.sleep(0.6)
        assert s.summary is None and not s.result_heard and s.agent_active and not s.early_result_wait
        await s.finish("user_canceled")
    run(go())


def test_user_key_during_early_hold_ends_hold(monkeypatch):
    """첫 메뉴 전 결과를 기다리는 중 사용자가 키패드로 누르면 그 결과로 확정하지 않음."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.2)

    async def go():
        s, call = await _early_session()
        await s.on_paragraph(EARLY_RES)
        await s.action({"type": "key", "key": "1"})
        await asyncio.sleep(0.5)
        assert call.keys == ["1"] and s.summary is None and not s.result_heard and not s.early_result_wait
        await s.finish("user_canceled")
    run(go())


def test_revise_goal_during_early_hold_keeps_exploring(monkeypatch):
    """첫 메뉴 전 결과를 기다리는 중 목적을 다시 말하면 앞 목적의 보류 종료. 새 목적으로는 아무것도 듣지 않았으므로
    업무 완료로 끝내지 않고 이어지는 첫 메뉴를 새 목적으로 판단. 바로 끊겨도 결과로 끝내지 않음."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.2)
    new_goal = "진료 예약하고 싶어"

    async def go():
        s, call = await _early_session()
        s.d.interp.goals[new_goal] = {"match": {"진료 예약": 0.9, "예약 확인": 0.1}}
        await s.on_paragraph(EARLY_RES)
        r = await s.action({"type": "revise_goal", "text": new_goal})
        assert r["accepted"] and not s.early_result_wait
        await asyncio.sleep(0.5)
        assert s.summary is None and s.agent_active and s.status != "agent_done"
        await s.on_paragraph(EARLY_MENU)
        assert call.keys == ["1"]
        await s.finish("user_canceled")

        s, call = await _early_session(hangup_flush_s=1.0)
        await s.on_paragraph(EARLY_RES)
        await s.action({"type": "revise_goal", "text": new_goal})
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended, timeout=3)
        assert s.end_reason == "failed" and s.summary is None
    run(go())


def test_hangup_after_early_timer_fired_still_completes(monkeypatch):
    """보류 타이머가 이미 울려(타이머 목록에서 빠짐) 확인이 아직 실행되지 않은 상태에서 ARS 가 끊어도, 보류 표시로 알아보고
    결과로 확정 (타이머 목록만 보면 이 상태를 놓쳐 failed 로 끝나는 경쟁 조건 방지)."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 5.0)

    async def go():
        s, call = await _early_session(hangup_flush_s=1.0)
        await s.on_paragraph(EARLY_RES)
        s.cancel_timer("early_result")  # 울려서 목록에서 빠진 상태 구성 (확인은 아직 실행 전)
        assert s.early_result_wait
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended, timeout=3)
        names = [e["event"] for e in s.hub.history]
        assert s.end_reason == "completed" and s.summary["kind"] == "result" and names.index("summary") < names.index("ended")

        s, call = await _early_session(hangup_flush_s=1.0)
        await s.on_paragraph(EARLY_RES)
        async with s._lock:  # 다음 단락 판단이 오래 걸리는 동안 타이머가 울리고 확인은 잠금 대기
            await asyncio.sleep(0.05)
            S.EARLY_RESULT_WAIT_S = 0.05
            s.timer("early_result", 0.05, s._early_result_check)
            await asyncio.sleep(0.2)
            assert "early_result" not in s._timers and s.early_result_wait
            await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended, timeout=3)
        assert s.end_reason == "completed" and s.summary["kind"] == "result"
    run(go())


def test_hangup_while_next_paragraph_is_recognized_waits_for_it(monkeypatch):
    """보류 중 ARS 가 메뉴를 말하고 끊었는데 그 단락을 아직 인식하는 중이면, 타이머가 울려도 결과로 확정하지 않고 인식
    대기. 메뉴였으면 결과 폐기 (인식 속도에 따라 completed·failed 가 갈리는 것 방지)."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.05)

    async def go():
        s, call = await _early_session(hangup_flush_s=2.0)
        await s.on_paragraph(EARLY_RES)
        s.stt_inflight += 1  # 메뉴 단락을 인식하는 중
        await s._on_end("remote_hangup")
        await asyncio.sleep(0.3)  # 타이머가 여러 번 울림
        assert not s.result_heard and s.summary is None and not s.ended
        await s.on_paragraph(EARLY_MENU)
        s.stt_inflight -= 1
        await wait_for(s, lambda s: s.ended, timeout=3)
        assert s.end_reason == "failed" and s.summary is None and call.keys == []
    run(go())


def test_new_paragraph_during_early_hold_restarts_wait(monkeypatch):
    """보류 중 새 일반 안내가 오면 그 안내 뒤부터 다시 대기: 처음 보류 시간이 지난 뒤에 메뉴가 와도, 일반 안내 뒤 보류
    시간 안이면 메뉴로 판단."""
    import session as S
    from interpreter import _norm
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.6)
    notice = "잠시 후 메뉴를 안내해 드리겠습니다."

    async def go():
        s, call = await _early_session()
        s.d.interp.prompts[_norm(notice)] = (notice, {"kind": "info"})
        await s.on_paragraph(EARLY_RES)
        await asyncio.sleep(0.3)
        await s.on_paragraph(notice)       # 여기서 다시 0.6초
        await asyncio.sleep(0.45)          # 처음 보류(0.6초)는 지났고, 다시 건 보류는 남음
        assert s.summary is None
        await s.on_paragraph(EARLY_MENU)
        assert call.keys == ["2"] and s.summary is None
        await asyncio.sleep(0.7)
        assert s.summary is None and s.status != "agent_done"
        await s.finish("user_canceled")
    run(go())


def test_result_while_waiting_agent_before_first_menu_is_immediate(monkeypatch):
    """첫 메뉴 전 상담원 대기 중(waiting_agent)에 들은 결과는 보류하지 않고 바로 확정 (보류를 막는 상태가 이미 있어
    보류하면 확정도 일반 처리도 되지 않고 결과가 사라지는 문제 방지)."""
    import session as S
    from interpreter import _norm
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.2)
    wait = "상담원을 연결해 드리겠습니다. 잠시만 기다려 주십시오."

    async def go():
        s, call = await _early_session()
        s.d.interp.prompts[_norm(wait)] = (wait, {"kind": "agent_wait"})
        await s.on_paragraph(wait)
        assert s.status == "waiting_agent"
        await s.on_paragraph(EARLY_RES)
        assert s.result_heard and s.summary is not None and not s.early_result_wait
        await s.finish("user_canceled")
    run(go())


def test_hangup_with_recognition_past_flush_limit_confirms_result(monkeypatch):
    """보류 중 ARS 가 끊었는데 마지막 단락 인식이 끊긴 뒤 제한 시간(hangup_flush_s)을 넘기면, 더 기다리지 않고 결과로
    확정해 안내 원문 요약과 completed 로 종료 (결과를 버리고 failed 로 끝나는 문제 방지)."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 5.0)

    async def go():
        s, call = await _early_session(hangup_flush_s=0.3)
        calls = []

        async def summ(goal, caps):
            calls.append(list(caps))
            return {"summary": "AI 요약", "key_info": []}

        s.d.interp.summarize_result = summ
        await s.on_paragraph(EARLY_RES)
        s.stt_inflight += 1  # 끝나지 않는 인식
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended, timeout=3)
        # 잠금은 비어 있으므로 인식을 더 기다리지 않고(force) 확인해 AI 요약 사용
        assert s.end_reason == "completed" and s.summary["summary"] == "AI 요약" and calls == [[EARLY_RES]]

        s, call = await _early_session(hangup_flush_s=0.3)
        await s.on_paragraph(EARLY_RES)
        async with s._lock:  # 마지막 단락 판단이 제한 시간과 최소 1초를 넘겨 잠금 점유
            await s._on_end("remote_hangup")
            await asyncio.sleep(1.6)
        await wait_for(s, lambda s: s.ended, timeout=3)
        assert s.end_reason == "completed" and s.summary["summary"] == EARLY_RES
        assert [e["event"] for e in s.hub.history].count("summary") == 1
    run(go())


def test_early_result_summary_uses_result_paragraph_only(monkeypatch):
    """보류 뒤 일반 안내가 이어지고 조용해져 결과로 확정하면, 요약 입력과 원문 대체는 결과 단락까지만 사용
    (뒤의 '이용해 주셔서 감사합니다' 가 결과 요약이 되는 문제 방지)."""
    import session as S
    from interpreter import _norm
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.1)
    thanks = "이용해 주셔서 감사합니다."

    async def go():
        seen = []
        for fail in (False, True):
            s, call = await _early_session()
            s.d.interp.prompts[_norm(thanks)] = (thanks, {"kind": "info"})
            orig = s.d.interp.summarize_result

            async def summ(goal, caps, orig=orig, fail=fail):
                seen.append(list(caps))
                if fail:
                    raise RuntimeError("요약 실패")
                return await orig(goal, caps)

            s.d.interp.summarize_result = summ
            await s.on_paragraph(EARLY_RES)
            await s.on_paragraph(thanks)
            await wait_for(s, lambda s: s.summary is not None)
            if fail:
                assert s.summary["summary"] == EARLY_RES
            await s.finish("user_canceled")
        assert seen == [[EARLY_RES], [EARLY_RES]]
    run(go())


def test_agent_wait_during_early_hold_then_result_summarizes_all(monkeypatch):
    """보류 중 상담원 대기 안내 뒤 바로 온 결과는 보류와 상관없이 확정하고, 요약 입력·원문 대체에 그 결과 포함
    (보류한 첫 단락까지로 잘리는 문제 방지)."""
    import session as S
    from interpreter import _norm
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 3.0)
    wait = "상담원을 연결해 드리겠습니다. 잠시만 기다려 주십시오."
    res2 = "고객님의 예약일은 다음 달 3일 오전 10시입니다."

    async def go():
        seen = []
        for fail in (False, True):
            s, call = await _early_session()
            s.d.interp.prompts[_norm(wait)] = (wait, {"kind": "agent_wait"})
            s.d.interp.prompts[_norm(res2)] = (res2, {"kind": "result"})
            orig = s.d.interp.summarize_result

            async def summ(goal, caps, orig=orig, fail=fail):
                seen.append(list(caps))
                if fail:
                    raise RuntimeError("요약 실패")
                return await orig(goal, caps)

            s.d.interp.summarize_result = summ
            await s.on_paragraph(EARLY_RES)
            await s.on_paragraph(wait)
            await s.on_paragraph(res2)
            assert s.result_heard and s.summary is not None and s.result_upto is None
            if fail:
                assert s.summary["summary"] == res2
            await s.finish("user_canceled")
        assert all(caps[-1] == res2 for caps in seen) and len(seen) == 2
    run(go())


def test_result_after_first_key_is_still_immediate():
    """키를 누른 뒤의 결과 안내는 기다리지 않고 바로 결과로 처리 (첫 메뉴 전 결과만 보류)."""
    from interpreter import _norm
    res = "예약일은 다음 달 3일 오전 10시입니다."

    async def go():
        s, call = await _early_session()
        s.d.interp.prompts[_norm(res)] = (res, {"kind": "result"})
        await s.on_paragraph(EARLY_MENU)
        await s.on_paragraph(res)
        assert call.keys == ["2"] and s.result_heard and s.summary is not None and s.status == "agent_done"
        await s.finish("user_canceled")
    run(go())


# 상담원 선택지 확인 질문·동점 재질문 문구·업무시간 외 상담 가능 시간 안내 ---------------------------------------------------
def test_agent_option_is_not_auto_pressed_when_goal_has_no_agent_request():
    """목적에 상담원 요청이 없는데 상담원 연결이 1위면 한 번 질문 (2026-11-10 우체국 실통화: 첫 메뉴 1~5번이 받아쓰기에서 빠져
    '운영시간' 목적에 상담사 연결 0.50 을 바로 누름). 같은 계층에서 다시 판단하면 원래대로 진행."""
    from interpreter import Option, Prompt
    from navigator import AGENT_UNASKED_ASK, Navigator, NavState

    class Fixed:
        async def score(self, goal, options, path=None):
            return {"8": 0.1, "6": 0.2, "7": 0.3, "0": 0.5}

    opts = [Option("8", "외국어 서비스"), Option("6", "생활정보 홍보 우편 서비스"), Option("7", "인터넷 우체국"), Option("0", "상담사 연결")]
    menu = Prompt(kind="menu", text="메뉴", options=opts)
    st = NavState("우체국 운영시간 알려줘")
    d = run(Navigator(Fixed()).decide(st, menu, [], True))
    assert d.action == "ask" and d.reason == "ambiguous" and d.message == AGENT_UNASKED_ASK
    assert [o.key for o in d.options][0] == "0" and len(d.options) == 4
    d = run(Navigator(Fixed()).decide(st, menu, [], False))
    assert d.action == "press" and d.key == "0"
    d = run(Navigator(Fixed()).decide(NavState("상담사랑 통화하고 싶어"), menu, [], True))
    assert d.action == "press" and d.key == "0"


def test_low_confidence_question_says_undecided_when_scores_tie():
    """1·2위 점수가 같은 확신 없음 재질문은 어느 선택지가 맞는지 정하지 못했다고 안내. 1위가 홀로 높으면 그 이름을 대는 문구 그대로."""
    async def go():
        from interpreter import Option, Prompt
        s, call = await manual("오늘 달러 환율 알고 싶어", clarify_expire_s=5, clarify_drop_s=10)
        opts = [Option("3", "잔액 조회"), Option("4", "거래 내역 조회"), Option("5", "카드 및 기타 서비스")]
        p = Prompt(kind="menu", text="메뉴", options=opts)
        await s._ask(p, "low_confidence", opts, scores={"3": 0.4, "4": 0.4, "5": 0.4})
        assert s.pending["message"].startswith("어느 선택지가 맞는지 정하지 못했습니다.")
        await s._ask(p, "low_confidence", opts, scores={"3": 0.45, "4": 0.4, "5": 0.4})
        assert s.pending["message"].startswith("'잔액 조회' 메뉴가 맞는지 확실하지 않습니다.")
        await s.finish("user_canceled")
    run(go())


def test_out_of_hours_end_message_carries_ars_hours():
    """업무시간 외 종료 문구에 ARS 가 말한 상담 가능 시간 문장을 덧붙임 (공통 표현 'N시부터·까지' 만 봄)."""
    from session import hours_sentence
    assert hours_sentence("지금은 상담 가능 시간이 아닙니다. 평일 9시부터 18시까지 이용해 주십시오.") == "평일 9시부터 18시까지 이용해 주십시오."
    assert hours_sentence("상담 업무 시간이 종료되었습니다. 상담 가능 시간은 은행 영업일 오전 9시부터 오후 6시까지이며 토요일과 휴일에는 "
                          "운영되지 않습니다.").startswith("상담 가능 시간은 은행 영업일 오전 9시부터 오후 6시까지")
    assert hours_sentence("지금은 상담 가능 시간이 아닙니다.") is None

    async def go():
        from interpreter import _norm
        text = "지금은 상담 가능 시간이 아닙니다. 평일 9시부터 18시까지 이용해 주십시오."
        s, call = await manual("상담원 연결해줘", clarify_expire_s=5, clarify_drop_s=10)
        s.d.interp.prompts[_norm(text)] = (text, {"kind": "out_of_hours"})
        await s.on_paragraph(ROOT)
        await s.on_paragraph(text)
        assert s.ended and s.end_reason == "out_of_hours"
        assert s.status_message == "상담 가능 시간이 아니어서 통화를 종료했습니다. (ARS 안내: 평일 9시부터 18시까지 이용해 주십시오.)"
    run(go())


# --- 첫 메뉴 전 끝맺음 판단 보류 (해석기 hold_early_end, jevllm 조합. 인사말의 혼잡 안내를 업무시간 외로 오판해 첫 메뉴 전에 통화를 끊는 문제 방지) ----------
EARLY_END = "지금은 상담 가능 시간이 아닙니다. 평일 9시부터 18시까지 이용해 주십시오."


async def _early_end_session(kind="out_of_hours", hold=True, **kw):
    from interpreter import _norm
    s, call = await _early_session(**kw)
    s.d.interp.prompts[_norm(EARLY_END)] = (EARLY_END, {"kind": kind})
    if hold:
        s.d.interp.hold_early_end = True
    return s, call


def test_early_end_is_held_and_following_menu_is_decided(monkeypatch):
    """hold_early_end 해석기: 첫 메뉴 전·키 전 끝맺음 판단(업무시간 외)은 바로 종료하지 않고 보류. 이어서 메뉴가 오면 그 메뉴로 판단."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.2)

    async def go():
        s, call = await _early_end_session()
        await s.on_paragraph(EARLY_END)
        assert not s.ended and s.early_result_wait and s.early_end_kind == "out_of_hours" and s.hours_note
        await s.on_paragraph(EARLY_MENU)
        assert call.keys == ["2"]
        await asyncio.sleep(0.5)
        assert not s.ended and not s.early_result_wait and s.early_end_kind is None
        await s.finish("user_canceled")
    run(go())


def test_early_end_then_silence_ends_with_that_reason(monkeypatch):
    """보류한 끝맺음 뒤 메뉴 없이 조용하면 그 사유로 종료 (업무시간 외 문구 포함, 거절은 ars_rejected). 감사 인사가 이어져도 같음."""
    import session as S
    from interpreter import _norm
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.1)
    thanks = "이용해 주셔서 감사합니다."

    async def go():
        for kind, reason in (("out_of_hours", "out_of_hours"), ("message_sent", "message_sent"), ("rejected", "ars_rejected")):
            for tail in (None, thanks):
                s, call = await _early_end_session(kind)
                s.d.interp.prompts[_norm(thanks)] = (thanks, {"kind": "info"})
                await s.on_paragraph(EARLY_END)
                if tail:
                    await s.on_paragraph(tail)
                    assert not s.ended
                await wait_for(s, lambda s: s.ended)
                assert s.end_reason == reason and call.keys == []
                if kind == "out_of_hours":
                    assert s.status_message.endswith("(ARS 안내: 평일 9시부터 18시까지 이용해 주십시오.)")
    run(go())


def test_early_end_hold_waits_for_transcript_handed_over_but_not_taken(monkeypatch):
    """받아쓰기를 직접 주는 백엔드가 단락을 넘긴 뒤 세션이 그 단락을 잠금 아래로 가져가기 전(넘김 중)에 보류 확인이 돌아도 종료하지
    않음. 넘김 중인 단락은 ARS 활동으로 취급 (실제 서버의 STT 경로는 단락이 닫힌 순간부터 처리를 마칠 때까지 stt_inflight 로 셈).
    다른 처리가 잠금을 쥔 동안 확인이 먼저 잠금을 기다리고 넘어온 단락이 그 뒤에 줄을 서도 같음."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 5.0)

    def handover(call):
        call.transcripts_inflight = getattr(call, "transcripts_inflight", 0) + 1
        done = []

        def taken():
            if not done:
                done.append(1)
                call.transcripts_inflight -= 1
        return taken

    async def go():
        # 넘김 작업 실행 전에 보류 확인이 먼저 실행
        s, call = await _early_end_session()
        await s.on_paragraph(EARLY_END)
        assert s.early_result_wait
        taken = handover(call)
        pending = s._on_transcript(EARLY_MENU, None, forced=False, taken=taken)
        await s._early_result_check()
        assert not s.ended and s.early_result_wait
        await pending
        assert call.transcripts_inflight == 0 and call.keys == ["2"] and not s.ended
        await s.finish("user_canceled")

        # 잠금을 쥔 처리 뒤에 확인이 먼저 대기하고, 넘어온 단락이 그 뒤에 대기
        s, call = await _early_end_session()
        await s.on_paragraph(EARLY_END)
        await s._lock.acquire()
        check = asyncio.create_task(s._early_result_check())
        await asyncio.sleep(0)
        taken = handover(call)
        para = asyncio.create_task(s._on_transcript(EARLY_MENU, None, forced=False, taken=taken))
        await asyncio.sleep(0)
        s._lock.release()
        await check
        assert not s.ended
        await para
        assert call.transcripts_inflight == 0 and call.keys == ["2"] and not s.ended
        await s.finish("user_canceled")
    run(go())


def test_transcript_handover_is_released_when_session_takes_paragraph(monkeypatch):
    """넘김 중 표시는 세션이 그 단락을 잠금 아래로 가져가는 순간 해제: 단락 자신의 판단 중에는 ARS 활동으로 세지 않음
    (판단 중의 키 전송·민감정보 전송 대기가 넘김 표시 때문에 바뀌지 않도록). 통화가 끝난 뒤 넘어온 단락은 가져가지 않음."""
    async def go():
        s, call = await _early_session()
        call.transcripts_inflight = 1
        seen = []
        orig = s._paragraph

        async def spy(text, started_at=None):
            seen.append((call.transcripts_inflight, s._ars_active()))
            await orig(text, started_at)
        s._paragraph = spy
        released = []

        def taken():
            released.append(1)
            call.transcripts_inflight -= 1
        await s._on_transcript(EARLY_MENU, None, forced=False, taken=taken)
        assert released == [1] and seen == [(0, False)] and call.keys == ["2"]
        await s.finish("user_canceled")
        late = []
        await s._on_transcript(EARLY_MENU, None, forced=False, taken=lambda: late.append(1))
        assert late == []
    run(go())


def test_early_end_then_hangup_ends_with_that_reason(monkeypatch):
    """보류 중 ARS 가 끊으면 기다리지 않고 그 사유로 종료 (결과 completed 로 바꾸지 않음)."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 5.0)

    async def go():
        s, call = await _early_end_session(hangup_flush_s=1.0)
        await s.on_paragraph(EARLY_END)
        await s._on_end("remote_hangup")
        await wait_for(s, lambda s: s.ended, timeout=3)
        assert s.end_reason == "out_of_hours" and s.summary is None
    run(go())


def test_early_end_is_immediate_without_attribute_or_after_first_menu(monkeypatch):
    """속성이 없는 해석기(다른 조합)는 첫 메뉴 전이어도 바로 종료. 속성이 있어도 첫 메뉴를 들은 뒤의 끝맺음은 바로 종료."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 5.0)

    async def go():
        s, call = await _early_end_session(hold=False)
        await s.on_paragraph(EARLY_END)
        assert s.ended and s.end_reason == "out_of_hours"
        s, call = await _early_end_session()
        await s.on_paragraph(EARLY_MENU)
        await s.on_paragraph(EARLY_END)
        assert s.ended and s.end_reason == "out_of_hours" and call.keys == ["2"]
    run(go())


def test_early_end_then_result_is_held_as_result(monkeypatch):
    """보류한 끝맺음 뒤 첫 메뉴 전 결과 안내가 오면 가장 최근 안내(결과)의 보류로 전환. 조용하면 결과로 확정."""
    import session as S
    monkeypatch.setattr(S, "EARLY_RESULT_WAIT_S", 0.1)

    async def go():
        s, call = await _early_end_session()
        await s.on_paragraph(EARLY_END)
        await s.on_paragraph(EARLY_RES)
        assert s.early_end_kind is None and s.early_result_wait
        await wait_for(s, lambda s: s.summary is not None)
        assert s.summary["kind"] == "result" and not s.ended
        await s.finish("user_canceled")
    run(go())


def test_agent_guard_skips_human_request_and_words_single_option():
    """사람·직원 요청 표현('사람이랑 얘기할래')은 상담원 요청으로 보아 질문 생략. 선택지가 상담원 하나뿐이면 그에 맞는 문구로 질문."""
    from interpreter import Option, Prompt
    from navigator import AGENT_UNASKED_ONE_ASK, Navigator, NavState, wants_agent

    class Fixed:
        def __init__(self, scores):
            self.scores = scores

        async def score(self, goal, options, path=None):
            return {o.key: self.scores.get(o.key, 0.0) for o in options}

    assert wants_agent("카드 발급 관련해서 사람이랑 얘기할래") and wants_agent("상담 받고 싶어") and not wants_agent("우체국 운영시간")
    opts = [Option("5", "카드 및 기타 서비스"), Option("0", "상담 직원 연결")]
    d = run(Navigator(Fixed({"0": 0.96, "5": 0.3})).decide(NavState("카드 발급 관련해서 사람이랑 얘기할래"),
                                                           Prompt(kind="menu", text="메뉴", options=opts), [], True))
    assert d.action == "press" and d.key == "0"
    one = Prompt(kind="menu", text="업무 시간 안내입니다. 상담사 연결은 0번", options=[Option("0", "상담사 연결")])
    d = run(Navigator(Fixed({"0": 0.5})).decide(NavState("우체국 운영시간 알려줘"), one, [], True))
    assert d.action == "ask" and d.message == AGENT_UNASKED_ONE_ASK


def test_text_answer_to_agent_question_excludes_agent_option_on_that_layer():
    """'상담원 연결이 가장 가까운 선택지' 질문에 상담원이 아닌 업무로 답하면, 같은 메뉴를 다시 판단할 때 상담원 선택지를 빼고 판단
    (빼지 않으면 같은 점수로 상담원을 다시 누르게 됨)."""
    async def go():
        from interpreter import _norm
        from navigator import AGENT_UNASKED_ASK
        menu = "For foreign language services, press 8. 생활정보 홍보 우편 서비스 6번, 인터넷 우체국 7번, 상담사 연결 0번을 눌러주십시오."
        s, call = await manual("우체국 운영시간 알려줘", clarify_expire_s=5, clarify_drop_s=10)
        match = {"상담사 연결": 0.5, "인터넷 우체국": 0.3}
        s.d.interp.goals["우체국 운영시간 알려줘"] = {"match": match}
        s.d.interp.goals["운영시간만 알고 싶어"] = {"match": match}
        s.d.interp.prompts[_norm(menu)] = (menu, {"kind": "menu", "options": [
            {"key": "8", "label": "외국어 서비스"}, {"key": "6", "label": "생활정보 홍보 우편 서비스"},
            {"key": "7", "label": "인터넷 우체국"}, {"key": "0", "label": "상담사 연결"}]})
        await s.on_paragraph(menu)
        assert call.keys == [] and s.pending and s.pending["message"] == AGENT_UNASKED_ASK
        await s.action({"type": "answer", "request_id": s.pending["request_id"], "text": "운영시간만 알고 싶어"})
        assert call.keys == [] and s.pending and s.pending["reason"] == "low_confidence"
        assert "0" not in {o["key"] for o in s.pending["options"]}
        await s.finish("user_canceled")
    run(go())


def test_agent_request_answer_keeps_agent_goal_in_interpreter():
    """상담원 연결 버튼 통화에서 재질문에 글로 답하면 세션 목적에 '상담원 연결' 을 되살리고, set_goal 이 있는 해석기의
    목적도 같이 맞춤 (해석기가 상담원 목적인지 보고 상담원 선택지만 남은 메뉴를 다시 판단할지 정함)."""
    async def go():
        from interpreter import Option, Prompt
        s, call = await manual("카드", agent=True, clarify_expire_s=5, clarify_drop_s=10)
        synced: list[str] = []
        s.d.interp.set_goal = synced.append
        opts = [Option("1", "카드 발급"), Option("2", "카드 분실")]
        await s._ask(Prompt(kind="menu", text="메뉴", options=opts), "low_confidence", opts, scores={"1": 0.4, "2": 0.4})
        await s.action({"type": "answer", "request_id": s.pending["request_id"], "text": "카드 발급"})
        assert s.goal == "카드 발급 상담원 연결" and synced == ["카드 발급 상담원 연결"]
        await s.finish("user_canceled")
    run(go())
