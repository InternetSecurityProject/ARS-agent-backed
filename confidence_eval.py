"""
ARS 메뉴 선택 평가 + 확신도 기준 찾기 (Go/No-Go)

  python confidence_eval.py collect             실제 Gemini 로 메뉴 원문×요청마다 분류·확신도 측정 → confidence_scores.json
  python confidence_eval.py report              서버 기준(0.50·차이 0.10)으로 기관별 정답률·틀린 번호
  python confidence_eval.py sweep               확신도 기준(0.5~0.95)·1·2위 차이별 결과 표

데이터: ars_menu_cases.py (Notion '02 ARS 통화·메뉴 수집' 원문, 메뉴 40개·요청 약 200개)
정답이 None 이면 '누르지 않음'(애매해서 다시 묻기, 맞는 메뉴 없음). 이미 측정한 사례는 건너뛰므로 중단 후 이어서 실행할 수 있다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import os  # noqa: E402
OUT = HERE / os.environ.get("CONF_OUT", "confidence_scores.json")

from ars_menu_cases import CASES, MENUS  # noqa: E402


def pick(prefix: str | None, failed_from: str | None) -> list:
    """다시 잴 사례만 고른다. prefix: 메뉴 id 접두어(예: nh_), failed_from: 이전 결과 파일에서 틀렸거나 못 잰 사례."""
    cases = [c for c in CASES if not prefix or c[0].startswith(prefix)]
    if failed_from:
        old = {(r["menu"], r["goal"]): r for r in json.loads(Path(failed_from).read_text(encoding="utf-8"))}
        keep = []
        for c in cases:
            r = old.get((c[0], c[1]))
            if r is None or r["timeout"] or r["error"] or decide(r, 0.5, 0.1)[1] not in r["expected"]:
                keep.append(c)
        cases = keep
    return cases


async def collect(runs: int, sleep: float, timeout: float, cases: list | None = None) -> None:
    import config  # noqa: F401  (.env 로드)
    import gemini_interpreter as g

    ai = g.GeminiInterpreter()
    ai.decide_timeout_s = timeout
    rows = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {(r["menu"], r["goal"], r["run"]) for r in rows}
    for run in range(1, runs + 1):
        for menu, goal, exp in (cases or CASES):
            if (menu, goal, run) in done:
                continue
            ai.goal, ai._scores, ai._cache, ai._heard, ai._menus, ai.last_path, ai._slow_until = goal, {}, {}, [], {}, [], 0.0
            ai.last_answer = None
            t0 = time.perf_counter()
            kind, s, confirm, err = None, {}, None, None
            try:
                p = await ai.classify(MENUS[menu][1])
                kind = p.kind
                if p.kind in ("menu", "confirm") and p.options:
                    s = await ai.score(goal, p.options, [])
                if p.kind == "confirm" and not p.critical:
                    confirm = await ai.answer_confirm(goal, p)
            except Exception as e:  # noqa: BLE001
                err = str(e)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"org": MENUS[menu][0], "menu": menu, "goal": goal, "expected": exp, "run": run, "kind": kind,
                         "scores": s, "confirm": confirm, "ms": round(ms), "timeout": ai._slow_until > 0, "error": err})
            OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            act, key = decide(rows[-1], 0.5, 0.1)
            ok = key in exp
            print(f"{'PASS' if ok else 'FAIL'}{' 위험' if key is not None and key not in exp else ''} {menu} '{goal}' → {key} "
                  f"(정답 {exp}) kind={kind} {ms:.0f}ms", flush=True)
            await asyncio.sleep(sleep)


def decide(row: dict, go: float, margin: float, low: float = 0.30):
    """서버(navigator)와 같은 판단. (동작, 누를 키 또는 None)"""
    if row.get("kind") == "confirm" and row.get("confirm"):
        return "press", row["confirm"]
    if row.get("kind") not in ("menu", "confirm"):
        return "none", None
    r = sorted(row["scores"].items(), key=lambda x: -x[1])
    if not r:
        return "ask", None
    s1, s2 = r[0][1], (r[1][1] if len(r) > 1 else 0.0)
    if s1 >= go and s1 - s2 >= margin:
        return "press", r[0][0]
    return ("ask" if s1 >= low else "none"), None


def _rows() -> list[dict]:
    return [r for r in json.loads(OUT.read_text(encoding="utf-8")) if not r["timeout"] and not r["error"]]


def report() -> None:
    rows = _rows()
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["org"], []).append(r)
    print("| 기관 | 요청 | 정답 | 틀린 번호 누름 | 눌러야 했는데 안 누름 |\n| --- | --- | --- | --- | --- |")
    tot = [0, 0, 0, 0]
    for org, rs in by.items():
        ok = wrong = missed = 0
        for r in rs:
            _, key = decide(r, 0.5, 0.1)
            if key in r["expected"]:
                ok += 1
            elif key is not None:
                wrong += 1
            else:
                missed += 1
        tot = [tot[0] + len(rs), tot[1] + ok, tot[2] + wrong, tot[3] + missed]
        print(f"| {org} | {len(rs)} | {ok} ({ok / len(rs) * 100:.0f}%) | {wrong} | {missed} |")
    print(f"| **전체** | {tot[0]} | {tot[1]} ({tot[1] / tot[0] * 100:.0f}%) | {tot[2]} | {tot[3]} |")
    ms = sorted(r["ms"] for r in rows)
    print(f"\n응답 시간 p50 {ms[len(ms) // 2] / 1000:.1f}초, p95 {ms[int(len(ms) * 0.95)] / 1000:.1f}초 (분류+확신도)")
    print("\n틀리거나 놓친 사례")
    for r in rows:
        _, key = decide(r, 0.5, 0.1)
        if key not in r["expected"]:
            top = sorted(r["scores"].items(), key=lambda x: -x[1])[:3]
            print(f"- {r['menu']} '{r['goal']}': {key} (정답 {r['expected']}) kind={r['kind']} 상위 {top}")


def sweep() -> None:
    rows = [r for r in _rows() if r["kind"] in ("menu", "confirm") and r["scores"]]
    pos = [r for r in rows if any(k is not None for k in r["expected"])]
    neg = [r for r in rows if r["expected"] == [None]]
    print(f"측정 {len(rows)}건 (정답 있는 목적 {len(pos)}, 누르면 안 되는 목적 {len(neg)})\n")

    # 1) 1위 확신도 구간별 정답률 (정답 있는 목적만)
    print("| 1위 확신도 구간 | 건수 | 1위가 정답 | 정답률 |\n| --- | --- | --- | --- |")
    bands = [(0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.4, 0.5), (0.3, 0.4), (0.0, 0.3)]
    for lo, hi in bands:
        sub = [r for r in pos if r["scores"] and lo <= max(r["scores"].values()) < hi]
        ok = sum(max(r["scores"], key=r["scores"].get) in r["expected"] for r in sub)
        print(f"| {lo:.1f}~{min(hi, 1):.1f} | {len(sub)} | {ok} | {ok / len(sub) * 100:.0f}% |" if sub else f"| {lo:.1f}~{min(hi, 1):.1f} | 0 | - | - |")
    neg_top = [max(r["scores"].values()) for r in neg if r["scores"]]
    if neg_top:
        print(f"\n누르면 안 되는 목적의 1위 확신도: 최고 {max(neg_top):.2f}, 0.5 이상 {sum(v >= 0.5 for v in neg_top)}/{len(neg_top)}건")

    # 2) 기준값·차이별 결과
    print("\n| Go 기준 | 1·2위 차이 | 바로 누름 | 맞게 누름 | 틀리게 누름 | 묻기·누르지 않음 | 정답 있는 목적 중 바로 해결 |\n| --- | --- | --- | --- | --- | --- | --- |")
    for go in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        for margin in (0.1, 0.2, 0.3):
            pressed = right = wrong = 0
            for r in rows:
                act, key = decide({**r, "confirm": None, "kind": "menu"}, go, margin)
                if act == "press":
                    pressed += 1
                    if key in r["expected"]:
                        right += 1
                    else:
                        wrong += 1
            print(f"| {go:.2f} | {margin:.1f} | {pressed} | {right} | {wrong} | {len(rows) - pressed} | {right / len(pos) * 100:.0f}% |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "report", "sweep"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=6.0)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--prefix", help="메뉴 id 접두어만 (예: nh_)")
    ap.add_argument("--failed-from", help="이 결과 파일에서 틀렸거나 못 잰 사례만 다시 잰다")
    a = ap.parse_args()
    if a.cmd == "collect":
        asyncio.run(collect(a.runs, a.sleep, a.timeout, pick(a.prefix, a.failed_from)))
    elif a.cmd == "report":
        report()
    else:
        sweep()


if __name__ == "__main__":
    main()
