"""통화 중 상담원 연결 (request_handoff → answer 또는 cancel_handoff) 시험. 모의 해석기·모의 통화만 사용."""
from __future__ import annotations

import asyncio
import time

import pytest

from app import create_app
from session import ApiError
from test_flows import AGENT_WAIT, BAL_RESULT, RRN, _AGENT_CARD, _NAV_ROOT, _agent_setup, _nav_setup, ev, make_settings, manual, run


async def _request(s) -> str:
    r = await s.action({"type": "request_handoff"})
    assert r["accepted"] and r["status"] == "awaiting_user"
    return r["request_id"]


def test_request_handoff_pauses_auto_select_and_answer_finds_agent():
    """요청 중에는 메뉴를 들어도 키를 누르지 않음. 업무 목적 답을 받으면 그 업무 메뉴로 들어가 상담원 연결 선택."""
    async def go():
        s, call = await manual("결제 대금 조회", clarify_expire_s=5, clarify_drop_s=10)
        _agent_setup(s, "카드 분실 관련 상담원 연결", {"카드 업무": 0.9, "대출 업무": 0.1}, {"상담원 연결": 0.9})
        assert s.snapshot()["routing_mode"] == "task" and s.snapshot()["goal"] == "결제 대금 조회"
        rid = await _request(s)
        req = s.pending
        assert req["kind"] == "handoff_goal" and req["options"] == [] and req["message"] and req["expires_at"]
        assert [d["kind"] for d in ev(s, "request")] == ["handoff_goal"]
        await s.on_paragraph(_NAV_ROOT)
        assert call.keys == [] and s.pending is req  # 자동 선택 멈춤
        r = await s.action({"type": "answer", "request_id": rid, "text": "카드 분실 관련"})
        assert r["accepted"] and r["routing_mode"] == "human_agent"
        assert call.keys == ["1"] and s.pending is None
        snap = s.snapshot()
        assert snap["routing_mode"] == "human_agent" and snap["goal"] == "카드 분실 관련"
        assert s.goal == "카드 분실 관련 상담원 연결"
        await s.on_paragraph(_AGENT_CARD)
        assert call.keys == ["1", "0"]
        with pytest.raises(ApiError) as e:  # 같은 답을 다시 보내도 키를 다시 누르지 않음
            await s.action({"type": "answer", "request_id": rid, "text": "카드 분실 관련"})
        assert e.value.status == 409 and call.keys == ["1", "0"]
        await s.finish("user_canceled")
    run(go())


def test_duplicate_request_handoff_returns_same_request():
    async def go():
        s, call = await manual("결제 대금 조회", clarify_expire_s=5, clarify_drop_s=10)
        rid = await _request(s)
        r = await s.action({"type": "request_handoff"})
        assert r["request_id"] == rid and r["duplicate"] is True
        assert len(ev(s, "request")) == 1 and call.keys == []
        await s.finish("user_canceled")
    run(go())


def test_cancel_handoff_decides_menu_heard_during_request():
    """취소하면 앞 목적(task)으로 돌아가 요청 중에 들은 메뉴를 판단. 통화는 끝나지 않음."""
    async def go():
        s, call = await manual("카드 업무", clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {"카드 업무": {"카드 업무": 0.9, "대출 업무": 0.1}})
        rid = await _request(s)
        await s.on_paragraph(_NAV_ROOT)
        assert call.keys == []
        r = await s.action({"type": "cancel_handoff", "request_id": rid})
        assert r["accepted"] and r["routing_mode"] == "task" and s.goal == "카드 업무"
        assert call.keys == ["1"] and not s.ended and s.pending is None
        await s.finish("user_canceled")
    run(go())


def test_cancel_handoff_reopens_closed_clarification():
    """재질문이 열려 있을 때 요청했다가 취소하면 그 재질문이 새 요청 번호로 다시 열림. 앞 요청 번호로는 선택 불가."""
    async def go():
        s, call = await manual("카드 업무", clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {"카드 업무": {"카드 업무": 0.3, "대출 업무": 0.2}})
        await s.on_paragraph(_NAV_ROOT)
        first = s.pending
        assert first and first["kind"] == "clarification" and call.keys == []
        rid = await _request(s)
        await s.action({"type": "cancel_handoff", "request_id": rid})
        again = s.pending
        assert again["kind"] == "clarification" and again["request_id"] not in (first["request_id"], rid)
        assert again["message"] == first["message"] and again["options"] == first["options"]
        assert s.status == "awaiting_user" and ev(s, "request")[-1]["request_id"] == again["request_id"]
        with pytest.raises(ApiError):
            await s.action({"type": "select", "request_id": first["request_id"], "key": "1"})
        r = await s.action({"type": "select", "request_id": again["request_id"], "key": "1"})
        assert r["key_sent"] == "1" and call.keys == ["1"]
        await s.finish("user_canceled")
    run(go())


def test_request_handoff_blocked_states():
    """민감정보 입력·중요 선택 확인·상담원 대기/연결 중·통화 연결 전에는 409. 요청은 열리지 않음."""
    async def go():
        s, call = await manual("결제 대금 조회", clarify_expire_s=5, clarify_drop_s=10)
        s._new_request("secure_input", 5, label="주민등록번호", terminator="#", message="입력")
        with pytest.raises(ApiError) as e:
            await s.action({"type": "request_handoff"})
        assert e.value.status == 409 and e.value.code == "sensitive_input_blocked"
        s.pending = None
        s.secure_queued = True
        with pytest.raises(ApiError) as e:
            await s.action({"type": "request_handoff"})
        assert e.value.code == "sensitive_input_blocked"
        s.secure_queued = False
        s._new_request("clarification", 5, reason="critical_action", message="중요", options=[])
        with pytest.raises(ApiError) as e:
            await s.action({"type": "request_handoff"})
        assert e.value.status == 409 and s.pending["reason"] == "critical_action"
        s.pending = None
        for st in ("waiting_agent", "connected_to_agent"):
            s.set_status(st)
            with pytest.raises(ApiError) as e:
                await s.action({"type": "request_handoff"})
            assert e.value.status == 409
        s.set_status("listening")
        answered, s.answered_at = s.answered_at, None
        with pytest.raises(ApiError):
            await s.action({"type": "request_handoff"})
        s.answered_at = answered
        assert s.pending is None and ev(s, "request") == [] and call.keys == []
        await s.finish("user_canceled")
    run(go())


def test_handoff_wrong_request_id_and_cancel_is_still_call_end():
    """다른 요청 번호의 답·취소는 409. cancel 은 상담원 연결 요청 중에도 통화 종료."""
    async def go():
        s, call = await manual("결제 대금 조회", clarify_expire_s=5, clarify_drop_s=10)
        with pytest.raises(ApiError):
            await s.action({"type": "cancel_handoff", "request_id": "req_1"})
        rid = await _request(s)
        for body in ({"type": "cancel_handoff", "request_id": "req_999"},
                     {"type": "answer", "request_id": "req_999", "text": "카드"},
                     {"type": "select", "request_id": rid, "key": "1"}):
            with pytest.raises(ApiError) as e:
                await s.action(body)
            assert e.value.status == 409
        with pytest.raises(ApiError) as e:
            await s.action({"type": "answer", "request_id": rid, "text": "  "})
        assert e.value.status == 400 and s.pending["request_id"] == rid
        await s.action({"type": "cancel"})
        assert s.ended and s.end_reason == "user_canceled"
    run(go())


def test_handoff_expiry_uses_previous_goal_as_business():
    """업무 목적을 입력하지 않고 만료되면 기존 업무 목적으로 상담원 연결 탐색."""
    async def go():
        s, call = await manual("카드 분실 관련", clarify_expire_s=5, clarify_drop_s=0.2)
        _agent_setup(s, "카드 분실 관련 상담원 연결", {"카드 업무": 0.9, "대출 업무": 0.1}, {"상담원 연결": 0.9})
        await _request(s)
        await s.on_paragraph(_NAV_ROOT)
        end = time.monotonic() + 3
        while not call.keys and time.monotonic() < end:
            await asyncio.sleep(0.02)
        assert call.keys == ["1"] and s.pending is None
        assert s.routing_mode == "human_agent" and s.snapshot()["goal"] == "카드 분실 관련"
        await s.finish("user_canceled")
    run(go())


def test_secure_prompt_during_handoff_replaces_request():
    """업무 목적 입력 중 ARS 가 민감정보 입력을 요청하면 입력 요청으로 전환. 앞 요청의 취소·답은 409."""
    async def go():
        s, call = await manual("결제 대금 조회", clarify_expire_s=5, clarify_drop_s=10)
        rid = await _request(s)
        await s.on_paragraph(RRN)
        assert s.pending and s.pending["kind"] == "secure_input"
        with pytest.raises(ApiError):
            await s.action({"type": "cancel_handoff", "request_id": rid})
        with pytest.raises(ApiError):
            await s.action({"type": "request_handoff"})
        await s.finish("user_canceled")
    run(go())


def test_agent_button_session_snapshot_shows_business_goal():
    """세션 생성 agent=true 도 routing_mode=human_agent, 스냅샷 goal 은 업무 목적. 재질문 답·다시 말하기 뒤에도 업무 목적을 따름."""
    async def go():
        s, call = await manual("카드 분실", agent=True, clarify_expire_s=5, clarify_drop_s=10)
        snap = s.snapshot()
        assert snap["routing_mode"] == "human_agent" and snap["goal"] == "카드 분실" and s.goal == "카드 분실 상담원 연결"
        await s.action({"type": "revise_goal", "text": "결제 대금 조회"})
        assert s.snapshot()["goal"] == "결제 대금 조회" and s.goal == "결제 대금 조회 상담원 연결"
        await s.finish("user_canceled")
        s2, _ = await manual("", agent=True)
        assert s2.snapshot()["goal"] == "상담원 연결" and s2.routing_mode == "human_agent"
        await s2.finish("user_canceled")
    run(go())


def test_handoff_rest_and_events():
    """REST: 연결 전에는 409, 연결 뒤 request_handoff → request 이벤트(kind=handoff_goal) → cancel_handoff. 스냅샷에 routing_mode."""
    from fastapi.testclient import TestClient

    st = make_settings(sim_speed=0.05, clarify_expire_s=5, clarify_drop_s=10)
    with TestClient(create_app(st)) as c:
        r = c.post("/v1/sessions", json={"phone_number": "1588-9999", "goal": "카드 분실"},
                   headers={"X-App-Key": st.app_key})
        assert r.status_code == 201, r.text
        sid, tok = r.json()["session_id"], r.json()["session_token"]
        h = {"Authorization": f"Bearer {tok}"}
        with c.websocket_connect(f"/v1/sessions/{sid}/events?token={tok}") as ws:
            snap = ws.receive_json()
            assert snap["event"] == "snapshot" and snap["data"]["routing_mode"] == "task"
            end = time.monotonic() + 10
            while True:
                r = c.post(f"/v1/sessions/{sid}/actions", json={"type": "request_handoff"}, headers=h)
                if r.status_code == 200 or time.monotonic() > end:
                    break
                assert r.status_code == 409
                time.sleep(0.05)
            assert r.status_code == 200, r.text
            rid = r.json()["request_id"]
            while True:
                m = ws.receive_json()
                if m["event"] == "request" and m["data"]["request_id"] == rid:
                    break
            assert m["data"]["kind"] == "handoff_goal"
            assert c.post(f"/v1/sessions/{sid}/actions", json={"type": "cancel_handoff", "request_id": rid},
                          headers=h).status_code == 200
            assert c.post(f"/v1/sessions/{sid}/actions", json={"type": "cancel"}, headers=h).status_code == 200


def test_cancel_after_menu_reread_reopens_clarification():
    """요청 중 ARS 가 같은 메뉴를 다시 읽어도(새 안내 객체) 취소하면 닫아 둔 재질문이 다시 열림. 키를 누르거나 통화를 끝내지 않음."""
    async def go():
        s, call = await manual("카드 업무", clarify_expire_s=5, clarify_drop_s=10)
        _nav_setup(s, {"카드 업무": {"카드 업무": 0.3, "대출 업무": 0.2}})
        await s.on_paragraph(_NAV_ROOT)
        first = s.pending
        rid = await _request(s)
        await s.on_paragraph(_NAV_ROOT)
        await s.action({"type": "cancel_handoff", "request_id": rid})
        again = s.pending
        assert again and again["kind"] == "clarification" and again["message"] == first["message"]
        assert call.keys == [] and not s.ended and s.status == "awaiting_user"
        await s.finish("user_canceled")
    run(go())


def _menu_then_request(goal):
    async def go():
        from interpreter import _norm
        s, call = await manual(goal, clarify_expire_s=5, clarify_drop_s=0.3)
        _nav_setup(s, {goal: {"카드 업무": 0.9, "대출 업무": 0.1}})
        s.d.interp.prompts[_norm(BAL_RESULT)] = (BAL_RESULT, {"kind": "result"})
        s.d.interp.prompts[_norm(AGENT_WAIT)] = (AGENT_WAIT, {"kind": "agent_wait"})
        await s.on_paragraph(_NAV_ROOT)
        assert call.keys == ["1"]
        rid = await _request(s)
        return s, call, rid
    return go()


def test_result_during_handoff_closes_request_and_is_handled():
    """키를 누른 직후 요청했는데 결과 안내가 오면 요청을 닫고 결과로 처리 (버리지 않음). 앞 요청의 답은 409."""
    async def go():
        s, call, rid = await _menu_then_request("카드 업무")
        await s.on_paragraph(BAL_RESULT)
        assert s.pending is None and s.result_heard
        with pytest.raises(ApiError):
            await s.action({"type": "answer", "request_id": rid, "text": "카드 분실"})
        assert call.keys == ["1"]
        await s.finish("user_canceled")
    run(go())


def test_agent_wait_during_handoff_closes_request():
    """요청 중 상담원 대기 안내가 오면 요청을 닫고 waiting_agent. 만료로 탐색을 다시 시작하지 않음. 새 요청은 409."""
    async def go():
        s, call, rid = await _menu_then_request("카드 업무")
        await s.on_paragraph(AGENT_WAIT)
        assert s.pending is None and s.status == "waiting_agent"
        await asyncio.sleep(0.5)  # 만료(0.3초) 시각이 지나도 상태 유지
        assert s.status == "waiting_agent" and s.routing_mode == "task" and call.keys == ["1"]
        with pytest.raises(ApiError):
            await s.action({"type": "request_handoff"})
        await s.finish("user_canceled")
    run(go())


def test_handoff_on_som_combo_runs_without_error(caplog, monkeypatch):
    """som 조합(원본 탐색기·해석기, 가상 백엔드, JEV 자리는 정해진 답): 요청 → 업무 목적 답이 처리 오류 없이 상담원 우선 탐색으로 전환.
    실제 JEV 판단(상담원 선택지를 고르는지)은 확인하지 않음."""
    import logging

    from app import build_deps
    from navigator import Navigator
    from session import Deps, SessionManager
    from test_combo import end_all
    from test_som_mode import ScriptedJev, factory

    monkeypatch.delenv("RESULT_JUDGE", raising=False)
    monkeypatch.delenv("COMBO_DEFAULT_INTERP", raising=False)
    caplog.set_level(logging.INFO)

    async def go():
        base, f = factory(ScriptedJev())
        d = build_deps(make_settings(sim_lines=1, clarify_expire_s=5, clarify_drop_s=10))
        d.backend.tree["timeout_s"] = 300
        await d.backend.start()
        m = SessionManager(Deps(d.settings, d.backend, d.recognizer, base, Navigator(base), f))
        s = m.sessions[(await m.create("15889999", "통장 잔액 알고 싶어", interp="som"))["session_id"]]
        try:
            end = time.monotonic() + 10
            while s.answered_at is None and time.monotonic() < end:
                await asyncio.sleep(0.02)
            rid = (await s.action({"type": "request_handoff"}))["request_id"]
            out = await s.action({"type": "answer", "request_id": rid, "text": "카드 분실"})
            await asyncio.sleep(0.3)
            return s, out
        finally:
            await end_all(m)

    s, out = run(go())
    assert s.combo["interp"] == "som"
    assert out["accepted"] and out["routing_mode"] == "human_agent" and s.goal == "카드 분실 상담원 연결"
    text = " | ".join(r.getMessage() for r in caplog.records)
    assert "안내 처리 오류" not in text and "세션 작업 오류" not in text
