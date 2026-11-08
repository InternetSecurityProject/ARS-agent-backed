"""
동시 통화(발신 번호 여러 개)와 통화별 판단 조합 선택 시험. 실제 발신·실제 모델 호출 없음.

  python -m pytest -q test_combo.py

시험에 쓰는 전화번호는 모두 가짜 값. 실제 발신 동작의 확인은 이 시험의 범위 밖.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from app import build_deps, create_app, line_spec
from combos import ComboError, ComboFactory
from hybrid_interpreter import HybridJevInterpreter
from interpreter import MockInterpreter
from jev_interpreter import JevInterpreter
from local_llm import LocalLlmInterpreter, OpenAICompatClient
from navigator import Navigator
from session import ApiError, Deps, SessionManager
from telephony import ClawOpsBackend, NumberMaskFilter, SimBackend, mask_dest, parse_numbers
from test_flows import make_settings, run, wait_for

HERE = Path(__file__).parent
A, B, C = "070-0000-1111", "070-0000-2222", "070-0000-3333"   # 가짜 번호


# ---------------------------------------------------------------------------
# 번호 목록
# ---------------------------------------------------------------------------
def test_parse_numbers_single_matches_legacy():
    legacy = "".join(ch for ch in A if ch.isdigit())
    assert parse_numbers(A) == [legacy]
    b = ClawOpsBackend(A)
    assert b.capacity == 1 and len(b.lines) == 1 and b.from_number == legacy
    empty = ClawOpsBackend("")
    assert empty.capacity == 1 and empty.lines == [] and empty.agent is None
    with pytest.raises(SystemExit):
        run(empty.start())


def test_parse_numbers_dedupe_without_digits_in_log(caplog):
    with caplog.at_level(logging.WARNING, logger="ars.call"):
        got = parse_numbers(f"{A}, {B},{A}, +82 70 0000 2222,")
    assert got == ["07000001111", "07000002222"]
    warned = [r.getMessage() for r in caplog.records]
    assert len(warned) == 2 and all("0000" not in m for m in warned)


def test_number_mask_filter_and_repr():
    f = NumberMaskFilter(parse_numbers(f"{A},{B}"))
    rec = logging.LogRecord("clawops.agent._control_ws", logging.WARNING, "", 0,
                            "Control WS error: url=wss://h/v1/accounts/AC1/agent/listen?number=07000001111 %s / +82 70-0000-2222",
                            ("070 0000 1111",), None)
    f.filter(rec)
    msg = rec.getMessage()
    assert "0000" not in msg and "번호1" in msg and "번호2" in msg
    b = ClawOpsBackend(f"{A},{B}")
    assert "0000" not in repr(b.lines)
    assert mask_dest("0212345678") == "******5678" and mask_dest("1588") == "1588"


def test_line_spec_prefers_lines_setting():
    st = make_settings(backend="clawops", clawops_from=A, clawops_lines="")
    assert line_spec(st) == A
    st.clawops_lines = f"{A},{B}"
    assert line_spec(st) == f"{A},{B}"


def test_ars_env_file_is_read_after_local_env(tmp_path):
    """ARS_ENV_FILE 의 값이 실행 환경변수·프로젝트 폴더 .env 보다 뒤에 읽힘 확인 (먼저 읽은 값 우선)."""
    env = tmp_path / "shared.env"
    env.write_text("ZZ_COMBO_ONLY_SHARED=from_shared\nZZ_COMBO_BOTH=from_shared\n", encoding="utf-8")
    e = {**os.environ, "ARS_ENV_FILE": str(env), "ZZ_COMBO_BOTH": "from_process", "PYTHONUTF8": "1"}
    out = subprocess.run([sys.executable, "-c", "import os, config; print(os.environ.get('ZZ_COMBO_ONLY_SHARED'),"
                          " os.environ.get('ZZ_COMBO_BOTH'))"], cwd=HERE, env=e, capture_output=True, text=True, timeout=60)
    assert out.stdout.split() == ["from_shared", "from_process"], out.stderr


# ---------------------------------------------------------------------------
# ClawOps 백엔드 (SDK 를 흉내 낸 가짜 ClawOpsAgent)
# ---------------------------------------------------------------------------
class FakeSession:
    n = 0

    def __init__(self, agent):
        FakeSession.n += 1
        self.agent, self.call_id, self.ended_status = agent, f"CA{FakeSession.n:04d}", None


class FakeAgent:
    made: list = []

    def __init__(self, from_, session_factory, on_taken_over=None, **kw):   # clawops 0.70.0 처럼 이름이 있는 인자
        if on_taken_over is not None:
            kw["on_taken_over"] = on_taken_over
        self.from_, self.kw, self.h, self.dialed = from_, kw, {}, []
        self.fail_connect = False
        FakeAgent.made.append(self)

    def on(self, ev):
        def deco(fn):
            self.h.setdefault(ev, []).append(fn)
            return fn
        return deco

    async def connect(self):
        if self.fail_connect or FakeAgent.fail_all or self.from_ in FakeAgent.fail_numbers:
            raise ConnectionError("x")

    async def disconnect(self):
        self.disconnects = getattr(self, "disconnects", 0) + 1

    async def call(self, to, timeout=40):
        self.dialed.append(to)
        return FakeSession(self)


@pytest.fixture
def fake_agents(monkeypatch):
    import clawops.agent as ca

    FakeAgent.made, FakeAgent.fail_all, FakeAgent.fail_numbers = [], False, set()
    monkeypatch.setattr(ca, "ClawOpsAgent", FakeAgent)
    yield FakeAgent
    ClawOpsBackend.CALLS.clear()
    ClawOpsBackend.LEGS.clear()


def _handlers(ended: list):
    from telephony import CallHandlers

    async def nop(*a, **k):
        pass

    async def on_end(reason):
        ended.append(reason)

    return CallHandlers(nop, nop, nop, on_end)


def test_clawops_single_number_start_unchanged(fake_agents):
    b = ClawOpsBackend(A)
    run(b.start())
    assert len(fake_agents.made) == 1 and "on_taken_over" not in fake_agents.made[0].kw   # 번호 하나: on_taken_over 미전달
    assert b.agent is fake_agents.made[0] and set(fake_agents.made[0].h) == {"call_start", "call_failed", "call_end"}
    fake_agents.made.clear()
    fake_agents.fail_all = True
    with pytest.raises(ConnectionError):
        run(ClawOpsBackend(A).start())   # 번호 하나: 연결 예외 그대로 전파


def test_clawops_multi_agents_wiring(fake_agents):
    async def go():
        b = ClawOpsBackend(f"{A},{B}")
        await b.start()
        assert len(fake_agents.made) == 2 and all("on_taken_over" in a.kw for a in fake_agents.made)
        assert all(set(a.h) == {"call_start", "call_failed", "call_end"} for a in fake_agents.made)
        assert b.capacity == 2 and b.usable_count() == 2 and b.line_rank(1) == 0
        ended: list = []
        c = await b.dial("0212345678", _handlers(ended), line=1)
        second = fake_agents.made[1]
        assert second.dialed == ["0212345678"] and fake_agents.made[0].dialed == [] and c.line == 1
        assert b.line_rank(1) == 1 and b.line_rank(0) == 0         # 정리 중 (ClawOps 종료 알림 전)
        for h in second.h["call_end"]:
            await h(types.SimpleNamespace(call_id=c.session.call_id))
        assert ended == ["remote_hangup"] and b.line_rank(1) == 0
        c0 = await b.dial("0212345678", _handlers(ended))           # line 인자 없이 호출 시 첫 번호 사용
        assert c0.line == 0 and fake_agents.made[0].dialed == ["0212345678"]

    run(go())


def test_clawops_partial_connect_failure(fake_agents, caplog):
    fake_agents.fail_numbers = {"07000002222"}
    b = ClawOpsBackend(f"{A},{B}")
    with caplog.at_level(logging.WARNING, logger="ars.call"):
        run(b.start())
    assert b.usable_count() == 1 and b.line_rank(1) == 2 and b.line_rank(0) == 0
    assert any("번호2 연결 실패" in r.getMessage() for r in caplog.records)
    assert all("0000" not in r.getMessage() for r in caplog.records)
    fake_agents.fail_numbers = {"07000001111", "07000002222"}
    with pytest.raises(ConnectionError):
        run(ClawOpsBackend(f"{A},{B}").start())


def test_clawops_taken_over_marks_line_lost(fake_agents, caplog):
    b = ClawOpsBackend(f"{A},{B},{C}")
    run(b.start())
    with caplog.at_level(logging.WARNING, logger="ars.call"):
        fake_agents.made[0].kw["on_taken_over"](4409, "replaced")   # SDK 의 await 없는 호출 흉내
        fake_agents.made[0].kw["on_taken_over"](4409, "replaced")   # retired 뒤 close 로 인한 두 번째 호출
        fake_agents.made[2].kw["on_taken_over"](4403, "gone")
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 2 and "다른 프로세스" in msgs[0] and "4403" in msgs[1]
    assert b.line_rank(0) == 2 and b.line_rank(2) == 2 and b.line_rank(1) == 0 and b.usable_count() == 1


def test_line_usable_follows_control_ws_state():
    from telephony import _Line

    ev = asyncio.Event()
    ws = types.SimpleNamespace(_connected=ev, _ws=types.SimpleNamespace(closed=False))
    agent = types.SimpleNamespace(taken_over=False, _control_ws=ws)
    ln = _Line(0, "07000001111", agent)
    assert not ln.usable()              # 아직 연결 전 (또는 재연결 중)
    ev.set()
    assert ln.usable()
    ws._ws = None                       # 끊긴 직후 재연결 대기
    assert not ln.usable()
    ws._ws = types.SimpleNamespace(closed=False)
    agent.taken_over = True             # 다른 프로세스가 가져감
    assert not ln.usable()
    agent.taken_over, agent._control_ws = False, None   # disconnect 뒤
    assert not ln.usable()


# ---------------------------------------------------------------------------
# 번호 배정 (가상 ARS 의 가상 번호)
# ---------------------------------------------------------------------------
async def lines_manager(lines: int) -> SessionManager:
    deps = build_deps(make_settings(sim_lines=lines))
    deps.backend.tree["timeout_s"] = 300
    await deps.backend.start()
    return SessionManager(deps)


async def end_all(m: SessionManager):
    for s in list(m.sessions.values()):
        if not s.ended:
            await s.finish("user_canceled")


def test_manager_two_lines_assign_and_409():
    async def go():
        m = await lines_manager(2)
        a = m.sessions[(await m.create("15889999", "잔액 조회"))["session_id"]]
        b = m.sessions[(await m.create("15889999", "잔액 조회"))["session_id"]]
        assert {a.line, b.line} == {0, 1} and m.lines_in_use() == 2
        with pytest.raises(ApiError) as e:
            await m.create("15889999", "잔액 조회")
        assert e.value.status == 409 and e.value.code == "invalid_state" and e.value.reason == "no_free_line"
        freed = a.line
        await a.finish("user_canceled")
        c = m.sessions[(await m.create("15889999", "잔액 조회"))["session_id"]]
        assert c.line == freed
        await end_all(m)

    run(go())


def test_manager_concurrent_create_race():
    async def go(lines):
        m = await lines_manager(lines)
        orig = m.deps.interp.summarize_goal

        async def slow(goal):
            await asyncio.sleep(0.05)
            return await orig(goal)

        m.deps.interp.summarize_goal = slow
        res = await asyncio.gather(m.create("15889999", "잔액 조회"), m.create("15889999", "잔액 조회"),
                                   return_exceptions=True)
        await end_all(m)
        return m, res

    m, res = run(go(2))
    assert all(isinstance(r, dict) for r in res)
    assert sorted(s.line for s in m.sessions.values()) == [0, 1]
    m, res = run(go(1))
    errs = [r for r in res if isinstance(r, ApiError)]
    assert len(errs) == 1 and errs[0].status == 409 and errs[0].reason is None
    assert errs[0].message == "다른 통화가 진행 중입니다. (동시 통화 1건)"


class RankBackend(SimBackend):
    """번호 상태 흉내: ranks[i] (0 빈 번호, 1 정리 중, 2 쓸 수 없음)."""

    def __init__(self, tree, ranks):
        super().__init__(tree, 0.01, False, lines=len(ranks))
        self.ranks, self.dial_lines = ranks, []

    def line_rank(self, i):
        return self.ranks[i]

    async def dial(self, to, handlers, line=None):
        self.dial_lines.append(line)
        return await super().dial(to, handlers, line=line)


def test_manager_prefers_idle_and_skips_unusable_lines():
    async def go(ranks):
        st = make_settings()
        d = build_deps(st)
        d.backend = RankBackend(st.sim_tree, ranks)
        m = SessionManager(d)
        try:
            out = await m.create("15889999", "잔액 조회")
            return m.sessions[out["session_id"]].line, d.backend.dial_lines
        finally:
            await end_all(m)

    assert run(go([1, 0])) == (1, [1])        # 번호1 은 이전 통화 정리 중 → 번호2
    assert run(go([1, 1])) == (0, [0])        # 둘 다 정리 중이면 번호 순
    assert run(go([2, 1])) == (1, [1])        # 번호1 은 쓸 수 없음 (다른 프로세스가 가져감 등)
    with pytest.raises(ApiError) as e:
        run(go([2, 2]))
    assert e.value.reason == "no_free_line"


def test_session_dial_passes_line_only_when_multi():
    class ManualBackend:
        def __init__(self):
            self.calls = []

        async def dial(self, to, handlers):        # line 인자 없는 dial 형태
            self.calls.append(to)
            raise RuntimeError("발신 거절")

    async def go():
        st = make_settings()
        d = build_deps(st)
        d.backend = ManualBackend()
        m = SessionManager(d)
        out = await m.create("15889999", "잔액 조회")
        s = m.sessions[out["session_id"]]
        return d.backend.calls, s.end_reason, s.line

    calls, reason, line = run(go())
    assert calls == ["15889999"] and reason == "failed" and line is None


def test_dial_failure_log_has_no_exception_text(caplog):
    class Boom(Exception):
        status, code = 429, "concurrency_limit"

    class B:
        capacity = 2

        def line_rank(self, i):
            return 0

        async def dial(self, to, handlers, line=None):
            raise Boom("발신 번호 07000001111 한도 초과")

    async def go():
        st = make_settings()
        d = build_deps(st)
        d.backend = B()
        m = SessionManager(d)
        with caplog.at_level(logging.ERROR, logger="ars.session"):
            await m.create("15889999", "잔액 조회")

    run(go())
    msgs = [r.getMessage() for r in caplog.records if "발신 실패" in r.getMessage()]
    assert msgs and "0000" not in msgs[0] and "429" in msgs[0] and "concurrency_limit" in msgs[0] and "번호1" in msgs[0]


# ---------------------------------------------------------------------------
# REST·health
# ---------------------------------------------------------------------------
def _post(c, st, goal="잔액 조회", **extra):
    return c.post("/v1/sessions", json={"phone_number": "1588-9999", "goal": goal, **extra}, headers={"X-App-Key": st.app_key})


def test_single_line_409_body_and_health_unchanged():
    from fastapi.testclient import TestClient

    st = make_settings(sim_speed=0.05)
    with TestClient(create_app(st)) as c:
        h = c.get("/v1/health").json()
        assert h["active_session"] is None and h["lines"] == {"total": 1, "in_use": 0} and h["concurrent"] is False
        assert "combos" not in h and "usable" not in h["lines"]
        r = _post(c, st)
        assert r.status_code == 201, r.text
        sid = r.json()["session_id"]
        r2 = _post(c, st)
        assert r2.status_code == 409
        assert r2.json() == {"error": {"code": "invalid_state", "message": "다른 통화가 진행 중입니다. (동시 통화 1건)"}}
        h = c.get("/v1/health").json()
        assert h["active_session"] == sid and h["lines"] == {"total": 1, "in_use": 1} and h["active_sessions"] == 1
        assert c.get("/v1/health", headers={"X-App-Key": st.app_key}).json()["lines"]["usable"] == 1
        r3 = _post(c, st, combo={"interp": "local", "fix": "0"})   # 조합을 고를 수 없는 서버(목 해석기)
        assert r3.status_code == 400 and r3.json()["error"]["code"] == "invalid_request"
        tok = r.json()["session_token"]
        c.post(f"/v1/sessions/{sid}/actions", json={"type": "cancel"}, headers={"Authorization": f"Bearer {tok}"})


def test_health_multi_line():
    from fastapi.testclient import TestClient

    st = make_settings(sim_speed=0.05, sim_lines=2)
    with TestClient(create_app(st)) as c:
        r1 = _post(c, st)
        h = c.get("/v1/health").json()
        assert h["active_session"] is None and h["lines"] == {"total": 2, "in_use": 1} and h["concurrent"] is False
        r2 = _post(c, st)
        h = c.get("/v1/health").json()
        assert h["active_session"] == "lines_full" and h["concurrent"] is True and h["active_sessions"] == 2
        assert r1.json()["session_id"] not in str(h) and r2.json()["session_id"] not in str(h)
        r3 = _post(c, st)
        assert r3.status_code == 409 and r3.json()["error"]["reason"] == "no_free_line"
        for r in (r1, r2):
            o = r.json()
            c.post(f"/v1/sessions/{o['session_id']}/actions", json={"type": "cancel"},
                   headers={"Authorization": f"Bearer {o['session_token']}"})


def test_lifespan_finishes_all_sessions():
    from fastapi.testclient import TestClient

    st = make_settings(sim_speed=0.05, sim_lines=2)
    app = create_app(st)
    with TestClient(app) as c:
        _post(c, st)
        _post(c, st)
        m = app.state.manager
        assert len(m.active_sessions()) == 2
    assert all(s.end_reason == "user_canceled" for s in m.sessions.values())


# ---------------------------------------------------------------------------
# 판단 조합
# ---------------------------------------------------------------------------
class DownLocal:
    """로컬 LLM 대역: 항상 실패 (해석기는 규칙 판단으로 대체)."""

    async def generate(self, *a, **k):
        raise RuntimeError("no model")


class DownJev:
    async def decide(self, state, questions):
        raise TimeoutError()


def hybrid_factory(fix="separate"):
    lo = LocalLlmInterpreter(DownLocal())
    lo.fix_mode, lo.fix_transcript = fix, fix == "1"
    return ComboFactory.from_interp(HybridJevInterpreter(lo, JevInterpreter(DownJev())))


def test_combo_factory_from_interp_and_options():
    f = hybrid_factory("separate")
    assert (f.default_interp, f.default_fix) == ("hybrid", "separate")
    o = f.options()
    assert [x["id"] for x in o["interp"]] == ["hybrid", "local"] and [x["id"] for x in o["fix"]] == ["0", "1", "separate"]
    assert o["default"] == {"interp": "hybrid", "fix": "separate"}
    assert ComboFactory.from_interp(MockInterpreter()) is None
    lo_only = ComboFactory(DownLocal(), None, "local", "0")
    assert [x["id"] for x in lo_only.options()["interp"]] == ["local"]
    with pytest.raises(ComboError):
        lo_only.normalize("hybrid", "0")
    with pytest.raises(ComboError):
        f.normalize("gemini", "0")
    with pytest.raises(ComboError):
        f.normalize("local", "2")
    assert f.normalize(None, "") == ("hybrid", "separate") and f.normalize(" LOCAL ", "1") == ("local", "1")


def test_combo_build_isolates_call_state_and_shares_clients():
    f = hybrid_factory()
    a, b = f.build("hybrid", "separate"), f.build("hybrid", "separate")
    assert a is not b and a.local is not b.local and a.jev is not b.jev
    assert a.local.client is b.local.client is f.local_client and a.jev.client is b.jev.client is f.jev_client
    assert a.local.unnumbered is not b.local.unnumbered and a.local._scores is not b.local._scores
    assert a.local._cache is b.local._cache                    # 같은 조합은 분류 캐시 공유
    assert f.build("hybrid", "1").local._cache is not a.local._cache   # 고침 방식이 다르면 분리
    assert f.build("local", "separate")._cache is not a.local._cache
    a.private, a.input_heard = True, True                       # 통화 A 의 입력 요청 수신 상태
    b.start_call()
    run(b.summarize_goal("잔액 조회"))
    assert a.private and a.input_heard
    lo = f.build("local", "1")
    assert isinstance(lo, LocalLlmInterpreter) and lo.fix_mode == "1" and lo.fix_transcript
    h1 = f.build("hybrid", "1")
    assert h1.fix_mode == "1" and h1.local.fix_transcript
    h0 = f.build("hybrid", "0")
    assert h0.fix_mode == "0" and not h0.local.fix_transcript


def combo_manager(lines=1):
    d = build_deps(make_settings(sim_lines=lines))   # 가상 ARS 백엔드만 사용, 해석기는 아래에서 생성
    f = hybrid_factory()
    base = f.build("hybrid", "separate")
    nav = Navigator(base)
    deps = Deps(d.settings, d.backend, d.recognizer, base, nav, f)
    return SessionManager(deps)


def test_manager_builds_interpreter_per_call():
    async def go():
        m = combo_manager(lines=2)
        await m.deps.backend.start()
        a = m.sessions[(await m.create("15889999", "잔액 조회", interp="local", fix="0"))["session_id"]]
        b = m.sessions[(await m.create("15889999", "카드 분실", record=True))["session_id"]]
        try:
            assert a.combo == {"interp": "local", "fix": "0"} and isinstance(a.d.interp, LocalLlmInterpreter)
            assert b.combo == {"interp": "hybrid", "fix": "separate"} and isinstance(b.d.interp, HybridJevInterpreter)
            assert a.d.interp is not m.deps.interp and b.d.interp is not m.deps.interp
            assert a.d.navigator.interp is a.d.interp and b.d.navigator.interp is b.d.interp
            assert m.deps.navigator.interp is m.deps.interp          # 원래 탐색기는 그대로
            assert b.recorder.meta["combo"] == b.combo and b.recorder.meta["line"] == f"번호{b.line + 1}"
            with pytest.raises(ApiError) as e:
                await m.create("15889999", "잔액 조회", interp="gemini")
            assert e.value.status == 400
        finally:
            await end_all(m)

    run(go())


def test_combo_validation_before_line_check():
    """조합 값이 틀리면 빈 번호가 없어도 400 반환 확인 (요청 수정이 필요한 오류 우선)."""
    async def go():
        m = combo_manager(lines=1)
        await m.deps.backend.start()
        await m.create("15889999", "잔액 조회")
        try:
            with pytest.raises(ApiError) as e:
                await m.create("15889999", "잔액 조회", fix="x")
            assert e.value.status == 400
            with pytest.raises(ApiError) as e:
                await m.create("15889999", "잔액 조회", fix="1")
            assert e.value.status == 409
        finally:
            await end_all(m)

    run(go())


def test_single_line_without_factory_shares_deps():
    async def go():
        m = await lines_manager(1)
        s = m.sessions[(await m.create("15889999", "잔액 조회"))["session_id"]]
        try:
            assert s.d is m.deps and s.combo is None and s.line is None
        finally:
            await end_all(m)

    run(go())


# ---------------------------------------------------------------------------
# 로컬 LLM: 동시 요청 처리 중 다시 올리기(rewarm) 생략
# ---------------------------------------------------------------------------
def test_rewarm_skipped_while_other_requests_in_flight():
    async def go():
        c = OpenAICompatClient("http://llm.test", "m")
        started = []

        async def fake_warmup():
            started.append(1)

        c.warmup = fake_warmup
        c._inflight = 2                 # 이 요청 말고 다른 요청이 처리 중
        c._rewarm()
        await asyncio.sleep(0)
        assert started == [] and c._warm_at == 0.0
        c._inflight = 1                 # 이 요청뿐 (통화 하나일 때와 같음)
        c._rewarm()
        await asyncio.sleep(0)
        assert started == [1]

    run(go())


def test_generate_counts_in_flight():
    async def go():
        c = OpenAICompatClient("http://llm.test", "m")
        seen = []

        async def fake(*a):
            seen.append(c._inflight)
            if len(seen) == 2:
                raise RuntimeError("x")
            return {"ok": True}

        c._generate = fake
        assert await c.generate("s", "u", {}) == {"ok": True}
        with pytest.raises(RuntimeError):
            await c.generate("s", "u", {})
        assert seen == [1, 1] and c._inflight == 0

    run(go())


# ---------------------------------------------------------------------------
# 받아쓰기·분류 결합, 해석기 공유 거부, 발신 전 번호 상실, 미리 인식 결과 정리
# ---------------------------------------------------------------------------
class CombinedRecognizer:
    """받아쓰기+분류 한 번 호출 인식기 대역 (STT=omni·gemini 결합)."""
    is_local = True

    def __init__(self):
        self.interp = None

    def attach_interpreter(self, interp):
        self.interp = interp

    async def recognize(self, seg):
        return ""


def test_combined_stt_kept_with_single_line(monkeypatch):
    """번호 하나 + 받아쓰기·분류 한 번 호출: 한 번 호출 유지, 조합 선택 끔.
    번호 여러 개: 한 번 호출 끔, 통화마다 해석기 생성. 두 경우 확인."""
    import app as appmod

    for lines, attached, has_combos in ((1, True, False), (2, False, True)):
        rec, base = CombinedRecognizer(), LocalLlmInterpreter(DownLocal())
        monkeypatch.setattr(appmod, "make_recognizer", lambda *a, **k: rec)
        monkeypatch.setattr(appmod, "load_interpreter", lambda spec: base)
        st = make_settings(sim_lines=lines)
        st.gemini_stt_combined = True
        d = appmod.build_deps(st)
        assert (rec.interp is base) is attached and (d.combos is not None) is has_combos


def test_multi_line_refuses_interpreter_without_per_call_copy(monkeypatch):
    """번호 여러 개 + 통화마다 해석기를 만들 수 없는 해석기(Gemini·JEV 단독 등): 서버 기동 거부 확인
    (해석기 하나를 함께 쓰면 한 통화의 목적 요약이 다른 통화의 목적·최근 메뉴를 덮어씀). 번호 하나면 정상 기동."""
    import app as appmod

    class SharedOnly:
        async def summarize_goal(self, goal):
            return goal

    monkeypatch.setattr(appmod, "load_interpreter", lambda spec: SharedOnly())
    with pytest.raises(SystemExit):
        appmod.build_deps(make_settings(sim_lines=2))
    d = appmod.build_deps(make_settings(sim_lines=1))
    assert d.combos is None and isinstance(d.interp, SharedOnly)


def test_session_repicks_line_lost_before_dial():
    """번호 선택 뒤 발신 전(목적 요약 대기 중)에 그 번호를 잃으면 빈 번호를 다시 골라 발신, 없으면 failed 처리 확인."""
    async def go(ranks, lose):
        st = make_settings()
        d = build_deps(st)
        d.backend = RankBackend(st.sim_tree, list(ranks))
        m = SessionManager(d)
        orig = d.interp.summarize_goal

        async def slow(goal):
            for i in lose:
                d.backend.ranks[i] = 2        # 요약을 기다리는 동안 번호를 잃음
            return await orig(goal)

        d.interp.summarize_goal = slow
        try:
            out = await m.create("15889999", "잔액 조회", record=True)
            s = m.sessions[out["session_id"]]
            return s.line, d.backend.dial_lines, s.end_reason, s.recorder.meta.get("line")
        finally:
            await end_all(m)

    line, dialed, reason, meta_line = run(go([0, 0], [0]))
    assert line == 1 and dialed == [1] and reason is None and meta_line == "번호2"
    line, dialed, reason, _ = run(go([0, 0], [0, 1]))
    assert dialed == [] and reason == "failed"


def test_clawops_dial_refuses_unusable_line(fake_agents):
    async def go():
        b = ClawOpsBackend(f"{A},{B}")
        await b.start()
        b.lines[1].lost = True
        with pytest.raises(RuntimeError):
            await b.dial("0212345678", _handlers([]), line=1)
        assert fake_agents.made[1].dialed == []

    run(go())


def test_whisper_drop_prefix_keeps_other_calls_entries():
    """인식을 마친 단락의 미리 인식 결과만 삭제, 다른 통화(다른 음성)의 미리 인식 결과는 유지됨 확인."""
    import hashlib

    from stt import WhisperRecognizer

    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    a, b = bytes(range(256)) * 200, bytes(reversed(range(256))) * 200
    entry = lambda audio, end, text: (end, hashlib.sha1(audio[:end]).digest(), text)
    rec._prefix = [entry(a, 8000, "A1"), entry(b, 8000, "B1"), entry(a, 16000, "A2"), entry(b, 24000, "B2")]
    rec._drop_prefix(a + b"tail")
    assert [e[2] for e in rec._prefix] == ["B1", "B2"]
    assert rec._cached(b + b"more") == (24000, "B2")


# ---------------------------------------------------------------------------
# 모델 상태 공유, 기동 시 연결 정리, 설정 오류 내성, CLAWOPS_FROM 번호 제외, 조합별 동시 통화, 로그 필터, 조합 이름표
# ---------------------------------------------------------------------------
def test_slow_model_flag_carries_to_next_call():
    """급한 판단 시간 초과 뒤 20초 규칙 판단(_slow_until)은 모델 연결의 상태: 해석기 하나 구성처럼 다음 통화로
    이어지고, 새 통화의 해석기 생성 시에도 유지됨 확인. 그 밖의 통화 상태는 통화별 분리."""
    import time as _t

    f = hybrid_factory()
    a = f.build("local", "0")
    a._slow_until = _t.monotonic() + 20
    b = f.build("hybrid", "separate")           # 새 통화 (다른 조합이어도 같은 모델)
    assert a._slow_until == b.local._slow_until > _t.monotonic()
    assert isinstance(a, LocalLlmInterpreter) and a.unnumbered is not b.local.unnumbered

    async def go():
        t0 = _t.monotonic()
        p = await b.local.classify("잔액조회는 3번을 눌러 주십시오.")   # 모델 호출 없이 규칙으로 판단
        return p, _t.monotonic() - t0

    p, dt = run(go())
    assert p.kind == "menu" and dt < 1.0


def test_failed_number_connection_is_closed_at_startup(fake_agents):
    """기동 때 연결에 실패한 번호의 연결 해제 확인 (SDK 의 백그라운드 연결 재시도로 늦게 붙어 다른 서버의 자리를 빼앗는 문제 방지)."""
    fake_agents.fail_numbers = {"07000002222"}
    b = ClawOpsBackend(f"{A},{B}")
    run(b.start())
    first, second = fake_agents.made
    assert getattr(second, "disconnects", 0) == 1 and getattr(first, "disconnects", 0) == 0
    assert b.line_rank(1) == 2


def test_local_default_server_starts_with_bad_jev_setting(monkeypatch):
    """로컬 단독이 기본인 서버: JEV 설정값이 틀려도(빈 JEV_TIMEOUT_S 등) 예외 없이 JEV 클라이언트 None, 조합 선택지는 local 만 남음 확인."""
    from combos import _jev_client_from_env

    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-not-a-key")
    monkeypatch.setenv("JEV_TIMEOUT_S", "")
    assert _jev_client_from_env() is None
    f = ComboFactory.from_interp(LocalLlmInterpreter(DownLocal()))
    assert [x["id"] for x in f.options()["interp"]] == ["local"]


def test_run_lines_drops_live_number_in_any_form():
    from run_lines import select_lines

    acct = ["07000001111", "07000002222", "07000003333"]
    assert select_lines(acct, "070-0000-3333", False) == (["07000001111", "07000002222"], True)
    assert select_lines(acct, "+82 70 0000 3333", False) == (["07000001111", "07000002222"], True)
    assert select_lines(["827000001111", "07000003333"], "07000001111", False) == (["07000003333"], True)
    assert select_lines(acct, "070-0000-3333", True) == (acct, False)
    assert select_lines(acct, "", False) == (acct, False)
    assert select_lines(["07000003333"], "07000003333", False) == ([], True)


def test_concurrent_calls_with_different_combos_full_flow():
    """가상 ARS 번호 2개로 조합이 다른 두 통화를 동시에 진행, 첫 메뉴 선택(path_updated)까지 확인 (모델 대역은 정해진 답을 주는 가짜).
    hybrid 조합 통화는 JEV, local 조합 통화는 로컬 모델로 판단, 둘 다 같은 키(잔액조회 3번) 선택."""
    from test_flows import ev
    from test_hybrid_interpreter import FakeJev, FakeLocal

    async def go():
        local, jev = FakeLocal(), FakeJev("3")
        base = HybridJevInterpreter(LocalLlmInterpreter(local), JevInterpreter(jev))
        base.fix_mode = "separate"
        f = ComboFactory.from_interp(base)
        d = build_deps(make_settings(sim_lines=2))
        d.backend.tree["timeout_s"] = 300
        await d.backend.start()
        m = SessionManager(Deps(d.settings, d.backend, d.recognizer, base, Navigator(base), f))
        oa, ob = await asyncio.gather(m.create("15889999", "통장 잔액 알고 싶어", interp="hybrid", fix="separate"),
                                      m.create("15889999", "통장 잔액 알고 싶어", interp="local", fix="0"))
        a, b = m.sessions[oa["session_id"]], m.sessions[ob["session_id"]]
        try:
            await wait_for(a, lambda s: ev(s, "path_updated"))
            await wait_for(b, lambda s: ev(s, "path_updated"))
            return a, b, local, jev
        finally:
            await end_all(m)

    a, b, local, jev = run(go())
    assert {a.line, b.line} == {0, 1}
    assert ev(a, "path_updated")[0]["selected"]["key"] == "3" and ev(b, "path_updated")[0]["selected"]["key"] == "3"
    assert jev.states                                   # hybrid 조합 통화의 JEV 질의 발생
    assert a.d.interp.last_route[0] == "jev" and isinstance(b.d.interp, LocalLlmInterpreter)
    assert b.d.interp.fix_mode == "0" and a.d.interp.fix_mode == "separate"


def test_health_access_log_is_dropped_only_when_ok():
    """웹 /app 의 10초 주기 health 조회 대응: 성공한 health 접속 로그만 제외 확인."""
    from app import HealthAccessFilter

    f = HealthAccessFilter()
    rec = lambda path, status: logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                                                 ("1.2.3.4:5", "GET", path, "1.1", status), None)
    assert not f.filter(rec("/v1/health", 200))
    assert f.filter(rec("/v1/health", 500)) and f.filter(rec("/app", 200)) and f.filter(rec("/v1/sessions", 200))


def test_combo_labels_follow_jev_model():
    f = ComboFactory(DownLocal(), types.SimpleNamespace(model="jev-1.14.0"), "hybrid", "separate")
    o = f.options()
    assert [x["label"] for x in o["interp"]] == ["JEV 1.14 + Local STT", "all local"]
    assert [x["label"] for x in o["fix"]] == ["끔", "1단계", "2단계"]
    assert ComboFactory(DownLocal(), types.SimpleNamespace(model="jev-2.0.1"), "hybrid", "0").options()["interp"][0]["label"] == "JEV 2.0 + Local STT"
