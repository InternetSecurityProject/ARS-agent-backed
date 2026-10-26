"""
QA 재시험 (실통화 없음)

음성 가상 ARS(안내를 TTS 소리로 송출, DTMF 톤 해독)로 시나리오를 실행하고, 지금까지 찾은 문제와 확인 항목을
통과/실패 표로 출력한다. 서버·AI 코드가 바뀔 때마다 실행하여 고쳐졌는지, 새로 망가진 것은 없는지 본다.
서버는 이 프로그램 안에서 직접 만들며(API 서버를 따로 띄우지 않음) 실통화·ClawOps 를 쓰지 않는다.

  python qa_run.py --interpreter gemini_interpreter:GeminiInterpreter
  python qa_run.py --interpreter gemini_interpreter:GeminiInterpreter --only 잔액 상담원 --report qa_report.md

  - 해석기는 AI 담당 구현을 지정한다 (목 해석기는 STT 문장을 알아보지 못해 대부분 실패한다).
  - 시나리오 1건은 실제 시간으로 1~2분 걸린다. 무료 Gemini 키는 분당 호출 한도가 있어 시나리오 사이에 쉰다(--pause).
  - 민감정보 입력 요청에는 시험용 숫자를 넣는다 (실제 개인정보 아님).
  - Windows 전용 (음성 가상 ARS 가 Windows 한국어 TTS 를 쓴다).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app import build_deps
from config import Settings
from session import SessionManager

HERE = Path(__file__).parent
TEST_DIGITS = {"주민": "900101", "생년": "900101", "계좌번호": "12345678", "비밀번호": "0000"}


@dataclass
class Run:
    """시나리오 1건의 관찰 결과."""
    name: str
    goal: str
    statuses: list[str] = field(default_factory=list)
    path: list[dict] = field(default_factory=list)
    sent: list[dict] = field(default_factory=list)       # 서버가 보낸 모든 키 (경로·민감정보·서버 자체)
    secure_requests: list[dict] = field(default_factory=list)
    clarifications: list[dict] = field(default_factory=list)
    timeouts: list[tuple[float, str]] = field(default_factory=list)  # ARS 입력 대기 초과 (시각, 메뉴)
    end_reason: str | None = None
    summary: dict | None = None
    human_delay: float | None = None   # 상담원이 말하기 시작 → connected_to_agent
    elapsed: float = 0.0
    error: str | None = None

    def first_key(self) -> dict | None:
        return next((k for k in self.sent if k["kind"] == "path"), None)

    def stray(self, speaking: bool | None = None, in_input: bool | None = None) -> list[dict]:
        return [k for k in self.sent if k["kind"] == "server"
                and (speaking is None or k["speaking"] == speaking)
                and (in_input is None or k["in_input"] == in_input)]


@dataclass
class Check:
    label: str
    owner: str
    fn: Callable[[Run], tuple[bool | None, str]]  # (통과 여부(None=판단 불가), 근거)


@dataclass
class Scenario:
    name: str
    tree: str
    goal: str
    limit_s: float
    checks: list[Check]
    stop_at: tuple[str, ...] = ()      # 이 상태가 되면 통화를 끊는다
    stop_node: str | None = None       # 가상 ARS 가 이 메뉴에 들어가면 끊는다


# ---------------------------------------------------------------------------
# 확인 항목
# ---------------------------------------------------------------------------
def key_is(expect: str) -> Callable[[Run], tuple[bool | None, str]]:
    def f(r: Run):
        k = r.first_key()
        if not k:
            return False, "누른 번호 없음"
        return k["keys"] == expect, f"'{k['keys']}' 누름 (기대 '{expect}')"
    return f


def in_time(limit: float) -> Callable[[Run], tuple[bool | None, str]]:
    def f(r: Run):
        k = r.first_key()
        root_timeout = [t for t, node in r.timeouts if not k or t < k["t"]]
        if root_timeout:
            return False, f"첫 메뉴 입력 대기({limit:.1f}초) 초과 후 누름"
        if not k or k["after_end"] is None:
            return None, "측정 불가"
        return k["after_end"] <= limit, f"안내 끝 → {k['after_end']:.1f}초"
    return f


def no_stray_while_speaking(r: Run):
    s = r.stray(speaking=True)
    return not s, "없음" if not s else ", ".join(f"'{k['keys']}' ({k['node']}, {k['t']:.0f}s)" for k in s)


def no_stray_in_input(r: Run):
    s = r.stray(in_input=True)
    return not s, "없음" if not s else ", ".join(f"'{k['keys']}' → {k['node']} 입력에 섞임" for k in s)


def secure_count(n: int):
    def f(r: Run):
        return len(r.secure_requests) == n, f"키패드 요청 {len(r.secure_requests)}회 (기대 {n})"
    return f


def completed_with_summary(r: Run):
    ok = r.end_reason == "completed" and bool(r.summary and r.summary.get("kind") == "result")
    return ok, f"end_reason={r.end_reason}, 결과 요약 {'있음' if r.summary else '없음'}"


def status_seen(st: str):
    def f(r: Run):
        return st in r.statuses, f"{'표시됨' if st in r.statuses else '표시 안 됨'}"
    return f


def agent_detect_within(limit: float):
    def f(r: Run):
        if r.human_delay is None:
            return False, "상담원 연결 인식 안 됨"
        return r.human_delay <= limit, f"상담원 첫마디 → 연결 인식 {r.human_delay:.1f}초"
    return f


def path_keys(expect: list[str]):
    def f(r: Run):
        got = [p["selected"]["key"] for p in r.path]
        return got[:len(expect)] == expect, f"경로 {got} (기대 {expect})"
    return f


SCENARIOS = [
    Scenario("잔액조회 (실제 KB 조건, 민감정보 3회)", "sim_kb_real.json", "통장 잔액 알고 싶어", 150, [
        Check("첫 메뉴에서 잔액조회(3) 선택", "AI", key_is("3")),
        Check("입력 대기 4.5초 안에 누름", "AI·STT", in_time(4.5)),
        Check("민감정보 요청을 키패드로 받음", "AI·서버", secure_count(3)),
        Check("안내 도중 서버가 스스로 키를 누르지 않음 (버그 ①)", "서버", no_stray_while_speaking),
        Check("민감정보 입력에 다른 키가 섞이지 않음 (버그 ①②)", "서버", no_stray_in_input),
        Check("결과 안내 후 completed + 결과 요약 (버그 ③)", "서버", completed_with_summary),
    ]),
    Scenario("분실신고 (기호 키 업무 선택지)", "sim_kb_real.json", "카드 잃어버려서 분실 신고하려고", 80, [
        Check("첫 메뉴에서 분실신고(*) 선택", "AI", key_is("*")),
        Check("입력 대기 4.5초 안에 누름", "AI·STT", in_time(4.5)),
    ], stop_node="lost"),
    Scenario("상담원 연결 (대기 음악)", "sim_kb_real.json", "상담원 연결해줘", 110, [
        Check("첫 메뉴에서 상담직원 연결(0) 선택", "AI", key_is("0")),
        Check("입력 대기 4.5초 안에 누름", "AI·STT", in_time(4.5)),
        Check("상담원 연결 대기(waiting_agent) 표시", "STT·서버", status_seen("waiting_agent")),
        Check("상담원 첫마디 후 5초 안에 연결 인식", "STT·AI", agent_detect_within(5.0)),
        Check("안내 도중 서버가 스스로 키를 누르지 않음 (버그 ①)", "서버", no_stray_while_speaking),
    ], stop_at=("connected_to_agent",)),
    Scenario("다른 ARS: 건강보험료 조회", "sim_nhis.json", "이번 달 건강보험료 얼마인지 알고 싶어", 150, [
        Check("보험료 문의(1) → 이번 달 보험료 조회(1)", "AI", path_keys(["1", "1"])),
        Check("민감정보 요청을 키패드로 받음", "AI·서버", secure_count(1)),
        Check("결과 안내 후 completed + 결과 요약 (버그 ③)", "서버", completed_with_summary),
    ]),
]


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
class _Timeouts(logging.Handler):
    """가상 ARS 의 입력 대기 초과 로그를 모은다."""

    def __init__(self, t0: float, out: list) -> None:
        super().__init__()
        self.t0, self.out = t0, out

    def emit(self, record: logging.LogRecord) -> None:
        if "입력 대기 초과" in record.getMessage():
            self.out.append((time.monotonic() - self.t0, str(record.args[-1]) if record.args else ""))


async def run_scenario(sc: Scenario, interpreter: str) -> Run:
    st = Settings()
    st.backend, st.sim_voice, st.sim_tree, st.interpreter = "sim", True, str(HERE / sc.tree), interpreter
    deps = build_deps(st)
    await deps.backend.start()
    warm = getattr(deps.recognizer, "warmup", None)
    if warm:
        await warm()
    manager = SessionManager(deps)
    r = Run(sc.name, sc.goal)
    t0 = time.monotonic()
    handler = _Timeouts(t0, r.timeouts)
    logging.getLogger("ars.call").addHandler(handler)
    secure_digits: list[str] = []
    try:
        out = await manager.create("15889999", sc.goal)
        s = manager.sessions[out["session_id"]]
        call = s.call
        orig = call.send_keys

        async def watched(keys: str) -> float:
            # 서버가 보낸 키를 모두 기록한다: 그때 ARS 가 말하는 중이었는지, 어느 메뉴였는지
            node = call.node
            r.sent.append({"t": time.monotonic() - t0, "keys": keys, "node": call.node_id,
                           "speaking": bool(call._out), "in_input": node.get("type") == "input",
                           "after_end": (time.monotonic() - call.said_end) if (call.said_end and not call._out) else None,
                           "kind": "?"})
            return await orig(keys)

        call.send_keys = watched
        answered: set[str] = set()
        connected_at = None
        end = t0 + sc.limit_s
        while time.monotonic() < end and not s.ended:
            await asyncio.sleep(0.1)
            if s.status == "connected_to_agent" and connected_at is None:
                connected_at = time.monotonic()
            req = s.pending
            if req and req["request_id"] not in answered:
                answered.add(req["request_id"])
                if req["kind"] == "secure_input":
                    r.secure_requests.append(req)
                    digits = next((v for k, v in TEST_DIGITS.items() if k in (req.get("label") or "")), "123456")
                    secure_digits.append(digits + (req.get("terminator") or ""))
                    await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": digits})
                else:
                    r.clarifications.append(req)
            if sc.stop_at and s.status in sc.stop_at:
                break
            if sc.stop_node and call.node_id == sc.stop_node:
                await asyncio.sleep(0.5)
                break
        if connected_at and call.human_at:
            r.human_delay = max(0.0, connected_at - call.human_at)
        if not s.ended:
            await s.action({"type": "cancel"})
        r.statuses = [e["data"]["status"] for e in s.hub.history if e["event"] == "status_changed"]
        r.path, r.end_reason, r.summary = list(s.path), s.end_reason, s.summary
        # 보낸 키 분류: 경로(PathStep)에 기록된 키 / 민감정보 입력 / 그 밖(서버가 스스로 누른 키)
        pkeys = [p["selected"]["key"] for p in r.path]
        for k in r.sent:
            if pkeys and k["keys"] == pkeys[0]:
                k["kind"] = "path"
                pkeys.pop(0)
            elif k["keys"] in secure_digits:
                k["kind"] = "secure"
                secure_digits.remove(k["keys"])
            else:
                k["kind"] = "server"
    except Exception as e:  # 시나리오 하나가 실패해도 나머지는 계속한다
        r.error = f"{type(e).__name__}: {e}"
    finally:
        logging.getLogger("ars.call").removeHandler(handler)
        for s in manager.sessions.values():
            if not s.ended:
                await s.finish("user_canceled")
        await deps.backend.stop()
    r.elapsed = time.monotonic() - t0
    return r


def report(results: list[tuple[Scenario, Run]]) -> str:
    lines = ["| 시나리오 | 확인 항목 | 담당 | 결과 | 근거 |", "| --- | --- | --- | --- | --- |"]
    total = passed = 0
    for sc, r in results:
        if r.error:
            lines.append(f"| {sc.name} | 실행 | — | ❌ 오류 | {r.error} |")
            continue
        for c in sc.checks:
            ok, why = c.fn(r)
            total += 1
            passed += ok is True
            mark = "✅" if ok is True else "❌" if ok is False else "➖"
            lines.append(f"| {sc.name} | {c.label} | {c.owner} | {mark} | {why} |")
    lines.append(f"\n통과 {passed}/{total}")
    for sc, r in results:
        keys = " ".join(f"{k['keys'] if k['kind'] != 'secure' else '(민감정보)'}[{k['kind']}]" for k in r.sent)
        lines.append(f"- {sc.name}: {r.elapsed:.0f}초, 상태 {' → '.join(dict.fromkeys(r.statuses))}, 보낸 키 {keys or '없음'}"
                     f", 입력 대기 초과 {len(r.timeouts)}회, 재질문 {len(r.clarifications)}회")
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser(description="QA 재시험 (음성 가상 ARS, 실통화 없음)")
    ap.add_argument("--interpreter", required=True, help="AI 해석기 (예: gemini_interpreter:GeminiInterpreter)")
    ap.add_argument("--only", nargs="*", help="이름에 이 단어가 들어간 시나리오만 실행")
    ap.add_argument("--pause", type=float, default=20, help="시나리오 사이 쉬는 시간(초). 무료 AI 키 호출 한도 대비")
    ap.add_argument("--report", help="결과 표를 저장할 파일 (Markdown)")
    ap.add_argument("--verbose", action="store_true", help="서버·STT·AI 기록을 모두 출력")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ars.call").setLevel(logging.INFO)
    todo = [sc for sc in SCENARIOS if not a.only or any(w in sc.name for w in a.only)]
    results = []
    for i, sc in enumerate(todo):
        if i:
            await asyncio.sleep(a.pause)
        print(f"[{i + 1}/{len(todo)}] {sc.name} — 목적 \"{sc.goal}\" ({sc.tree})", flush=True)
        results.append((sc, await run_scenario(sc, a.interpreter)))
    text = report(results)
    print("\n" + text)
    if a.report:
        Path(a.report).write_text(f"# QA 재시험 결과 ({time.strftime('%Y-%m-%d %H:%M')}, 해석기 {a.interpreter})\n\n"
                                  + text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
