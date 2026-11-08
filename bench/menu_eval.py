"""메뉴 선택 평가: ARS 안내 원문 × 목적 → 누를 번호 (해석기·모델 비교)

  python -m bench.menu_eval --cases <ars_menu_cases.py> --models local:qwen3:8b gemini:gemini-3.5-flash-lite
  python -m bench.menu_eval --cases <ars_menu_cases.py> --models local:exaone3.5:7.8b --sample 20     # 고르게 20건만
  python -m bench.menu_eval --cases <ars_menu_cases.py> --models gemini:gemini-3.5-flash-lite --sleep 4   # 무료 키 사용량 조절
  python -m bench.menu_eval --cases <ars_menu_cases.py> --models local_llm:LocalLlmInterpreter          # .env 설정 그대로

모델 표기
  local:<모델>    local_llm.LocalLlmInterpreter, LOCAL_LLM_* 설정에 모델만 바꿈 (예: local:qwen3:8b)
  gemini:<모델>   gemini_interpreter.GeminiInterpreter, GEMINI_API_KEY 사용 (예: gemini:gemini-3.5-flash-lite)
  모듈:클래스     그 해석기를 그대로 생성

데이터
  ars_menu_cases.py (MENUS: 메뉴 id → (기관, ARS 안내 원문), CASES: (메뉴 id, 목적, 허용 키 목록) 목록).
  저장소 미포함, 경로로 지정. 정답은 허용 키 목록이며 None 은 누르지 않음(묻기·이전 단계·종료).

판단 (서버와 같은 경로)
  classify → menu: navigator.decide / confirm: 중요 선택이면 묻기, 아니면 answer_confirm / sensitive: 민감정보 입력
  판단 대기는 서버와 같은 값(기본 4초, --timeout 으로 변경). 초과 시 해석기가 규칙 분류·재질문 값으로 대체하며,
  그 결과를 그대로 채점 (서버의 실제 동작과 동일). 사례마다 해석기 상태 초기화.

지표
  정답률, 틀린 번호 누름(위험), 눌러야 했는데 안 누름, 대체 발생 사례 수(LLM 호출 실패 또는 응답 모양 오류로 규칙·재질문 사용),
  (HTTP 429 사용량 초과 시 그 모델 측정을 멈추고 그때까지만 채점. 같은 무료 키를 쓰는 다른 서버의 사용량 보호용)
  판단 시간(분류+판단) 중앙값·95%·최대, 판단 대기 초과 사례 수(그 해석기의 실제 판단 대기 기준), LLM 호출 1회 시간 중앙값·95%
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import re
import statistics
import time
from pathlib import Path


def load_cases(path: str) -> tuple[dict, list]:
    spec = importlib.util.spec_from_file_location("ars_menu_cases", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MENUS, mod.CASES


def make_interp(name: str):
    """모델 표기 → 해석기 인스턴스."""
    import config  # noqa: F401  (.env 로드)

    prefix, _, rest = name.partition(":")
    if prefix == "local":
        from local_llm import LocalLlmInterpreter, OpenAICompatClient

        return LocalLlmInterpreter(OpenAICompatClient.from_env(rest))
    if prefix == "gemini":
        from gemini_interpreter import GeminiClient, GeminiInterpreter

        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise SystemExit("GEMINI_API_KEY 가 없습니다")
        return GeminiInterpreter(GeminiClient(key, rest, float(os.environ.get("GEMINI_TIMEOUT_S", "6")),
                                              thinking=os.environ.get("GEMINI_THINKING", "MINIMAL")))
    from interpreter import load_interpreter

    return load_interpreter(name)


def reset(ai, goal: str) -> None:
    """사례별 해석기 상태 초기화 (앞 사례의 캐시·지연 판정 혼입 방지)."""
    if hasattr(ai, "bench_reset"):  # 로컬·JEV 분담 해석기 (hybrid_interpreter)
        ai.bench_reset(goal)
        return
    ai.goal, ai._scores, ai._cache, ai.last_path, ai._slow_until = goal, {}, {}, [], 0.0
    ai.last_answer, ai.last_menu, ai.unnumbered = None, None, {}


# 해석기가 모델 응답을 쓰지 못하고 규칙·재질문으로 대체할 때 남기는 경고 (gemini_interpreter.py)
_FALLBACK_LOGS = ("classify 규칙 기반 대체", "score 실패", "answer_confirm 실패",
                  "Jev 분류 실패", "Jev 선택 실패", "Jev 확인 질문 실패")  # 뒤 셋: jev_interpreter·hybrid_interpreter
_QUOTA = re.compile(r"\bHTTP 429\b")


def instrument(ai) -> tuple[list, list]:
    """LLM 호출별 (성공 여부, ms, 오류 문장, 태그) 기록과 해석기의 대체(fallback) 경고 별도 수집.
    대체는 호출 실패뿐 아니라 응답이 JSON 이지만 모양이 틀릴 때(제한 없는 출력 방식)도 발생."""
    calls: list = []
    fallbacks: list = []

    class _Fallbacks(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if msg.startswith(_FALLBACK_LOGS):
                fallbacks.append(msg[:200])

    for name in ("gemini", "ars.jev", "ars.hybrid"):
        logging.getLogger(name).addHandler(_Fallbacks(logging.WARNING))

    def wrap(client, attr: str, tag: str) -> None:
        orig = getattr(client, attr)

        async def call(*a, **k):
            if getattr(client, "_warming", False):
                return await orig(*a, **k)  # 실패 뒤 백그라운드에서 도는 재적재(local_llm._rewarm)는 사례의 판단 호출 집계에서 제외
            t0 = time.perf_counter()
            try:
                out = await orig(*a, **k)
                calls.append((True, (time.perf_counter() - t0) * 1000, None, tag))
                return out
            except Exception as e:
                calls.append((False, (time.perf_counter() - t0) * 1000, str(e)[:200] or type(e).__name__, tag))
                raise

        setattr(client, attr, call)

    if hasattr(ai, "jev") and hasattr(ai, "local"):  # 로컬·JEV 분담 해석기: 두 쪽 호출을 따로 집계
        wrap(ai.local.client, "generate", "local")
        wrap(ai.jev.client, "decide", "jev")
    else:
        wrap(ai.client, "generate", "llm")
    return calls, fallbacks


async def decide_one(ai, nav, goal: str, text: str) -> tuple[str, object]:
    from navigator import NavState

    p = await ai.classify(text)
    if p.kind == "sensitive":
        return "secure", p
    if p.kind == "confirm":
        if p.critical:
            return "ask:critical_action", p
        key = await ai.answer_confirm(goal, p)
        return (f"press:{key}" if key else "ask:unknown_prompt"), p
    if p.kind == "menu":
        d = await nav.decide(NavState(goal), p, [], True)
        if d.action == "press":
            return f"press:{d.key}", p
        return f"{d.action}:{d.reason}", p
    return f"none:{p.kind}", p


class QuotaExceeded(Exception):
    """무료 API 사용량 초과(HTTP 429). 같은 키를 쓰는 다른 서버 보호를 위한 측정 중단용."""


async def eval_model(name: str, menus: dict, cases: list, timeout: float | None, sleep: float,
                     stop_on_quota: bool = True) -> list[dict]:
    from config import Settings
    from navigator import Navigator

    st = Settings()
    ai = make_interp(name)
    if timeout is not None:
        ai.decide_timeout_s = timeout
    warm = getattr(ai, "warmup", None)
    if warm:
        await warm()  # 모델 로드 시간은 채점에서 제외 (서버도 시작 시 미리 적재)
    calls, fallbacks = instrument(ai)
    nav = Navigator(ai, st.match_threshold, st.margin, st.max_revisit, st.max_presses, st.low_threshold)
    rows = []
    for i, (menu, goal, expected) in enumerate(cases):
        reset(ai, goal)
        n0, f0 = len(calls), len(fallbacks)
        t0 = time.perf_counter()
        err = None
        try:
            dec, p = await decide_one(ai, nav, goal, menus[menu][1])
            kind = p.kind
            opts = [[o.key, o.label] for o in p.options]
        except Exception as e:  # 해석기 밖으로 예외가 새면 서버는 대체값(info)으로 진행
            dec, kind, opts, err = "error", None, [], str(e)[:200]
        ms = (time.perf_counter() - t0) * 1000
        key = dec[6:] if dec.startswith("press:") else None
        mine = calls[n0:]
        row = {"model": name, "org": menus[menu][0], "menu": menu, "goal": goal, "expected": expected, "decision": dec,
               "key": key, "ok": key in expected, "kind": kind, "options": opts, "ms": round(ms),
               "llm_calls": [round(c[1]) for c in mine], "llm_failed": sum(1 for c in mine if not c[0]),
               "llm_errors": [c[2] for c in mine if c[2]], "fallbacks": fallbacks[f0:],
               "fallback": bool(fallbacks[f0:]) or any(not c[0] for c in mine), "timeout_s": ai.decide_timeout_s,
               "error": err, "call_tags": [c[3] for c in mine],
               "route": list(getattr(ai, "last_route", ("", None)))}
        rows.append(row)
        mark = "PASS" if row["ok"] else ("FAIL 위험" if key is not None else "FAIL")
        print(f"[{name}] {i + 1}/{len(cases)} {mark} {menu} '{goal}' → {dec} (정답 {expected}) "
              f"{ms:.0f}ms{' 대체' if row['fallback'] else ''}", flush=True)
        if stop_on_quota and any(_QUOTA.search(e) for e in row["llm_errors"]):
            rows.pop()  # 사용량 초과로 대체된 사례는 채점에서 제외
            print(f"[{name}] HTTP 429 (사용량 초과) — 같은 키를 쓰는 서버를 위해 여기서 멈춤 ({i}/{len(cases)}건 측정)", flush=True)
            raise QuotaExceeded(rows)
        if sleep:
            await asyncio.sleep(sleep)
    return rows


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    timeout = rows[0]["timeout_s"] if rows else 4.0  # 그 해석기가 실제로 쓴 판단 대기
    ok = sum(r["ok"] for r in rows)
    wrong = sum(1 for r in rows if not r["ok"] and r["key"] is not None)
    missed = sum(1 for r in rows if not r["ok"] and r["key"] is None)
    ms = [r["ms"] for r in rows]
    llm = [c for r in rows for c in r["llm_calls"]]
    return {"n": n, "ok": ok, "acc": ok / n if n else 0.0, "wrong_press": wrong, "missed": missed,
            "fallback_cases": sum(1 for r in rows if r["fallback"]), "errors": sum(1 for r in rows if r["error"]),
            "timeout_s": timeout,
            "p50_s": _pct(ms, 0.5) / 1000, "p95_s": _pct(ms, 0.95) / 1000, "max_s": max(ms) / 1000 if ms else 0.0,
            "over_timeout": sum(1 for x in ms if x > timeout * 1000),
            "llm_p50_s": (statistics.median(llm) / 1000) if llm else 0.0, "llm_p95_s": _pct(llm, 0.95) / 1000,
            "calls": len(llm), "calls_by_tag": {t: sum(1 for r in rows for x in r.get("call_tags", []) if x == t)
                                                for t in sorted({x for r in rows for x in r.get("call_tags", [])})},
            "routes": {t: sum(1 for r in rows if (r.get("route") or [""])[0] == t)
                       for t in sorted({(r.get("route") or [""])[0] for r in rows})}}


def table(results: dict[str, dict]) -> str:
    out = ["| 모델 | 사례 | 정답 | 틀린 번호 누름 | 안 누름(놓침) | 대체 발생 사례 | 판단 p50 | p95 | 최대 | 판단 대기 초과 | LLM 1회 p50 | p95 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in results.items():
        out.append(f"| {name} | {s['n']} | {s['ok']} ({s['acc'] * 100:.1f}%) | {s['wrong_press']} | {s['missed']} | "
                   f"{s['fallback_cases']} | {s['p50_s']:.2f}s | {s['p95_s']:.2f}s | {s['max_s']:.2f}s | {s['over_timeout']} (>{s['timeout_s']:g}s) | "
                   f"{s['llm_p50_s']:.2f}s | {s['llm_p95_s']:.2f}s |")
    return "\n".join(out)


def by_org(rows: list[dict]) -> str:
    orgs: dict[str, list] = {}
    for r in rows:
        orgs.setdefault(r["org"], []).append(r)
    out = ["| 기관 | 사례 | 정답 | 틀린 번호 누름 | 안 누름 |", "| --- | --- | --- | --- | --- |"]
    for org, rs in orgs.items():
        ok = sum(r["ok"] for r in rs)
        wrong = sum(1 for r in rs if not r["ok"] and r["key"] is not None)
        out.append(f"| {org} | {len(rs)} | {ok} ({ok / len(rs) * 100:.0f}%) | {wrong} | {len(rs) - ok - wrong} |")
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser(description="메뉴 선택 평가 (해석기·모델 비교)")
    ap.add_argument("--cases", required=True, help="ars_menu_cases.py 경로")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=0, help="사례를 고르게 이 수만큼만 (0=전부)")
    ap.add_argument("--prefix", help="메뉴 id 접두어 (예: kb_)")
    ap.add_argument("--timeout", type=float, default=None, help="판단 대기(초). 기본은 해석기 설정값(4초)")
    ap.add_argument("--sleep", type=float, default=0.0, help="사례 사이 대기(초). 무료 API 사용량 조절용")
    ap.add_argument("--json", help="사례별 결과 저장 경로")
    ap.add_argument("--no-stop-on-quota", action="store_true", help="HTTP 429(사용량 초과)가 나도 계속 잰다 (기본은 멈춤)")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.debug else logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    menus, cases = load_cases(a.cases)
    if a.prefix:
        cases = [c for c in cases if c[0].startswith(a.prefix)]
    if a.sample and a.sample < len(cases):
        step = len(cases) / a.sample
        cases = [cases[int(i * step)] for i in range(a.sample)]
    all_rows: list[dict] = []
    results: dict[str, dict] = {}
    for name in a.models:
        try:
            rows = await eval_model(name, menus, cases, a.timeout, a.sleep, not a.no_stop_on_quota)
        except QuotaExceeded as q:
            rows = q.args[0]
        if not rows:
            continue
        all_rows += rows
        results[name] = summarize(rows)
        if a.json:  # 모델 하나가 끝날 때마다 저장 (중간 중단 대비)
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            Path(a.json).write_text(json.dumps({"summary": results, "rows": all_rows}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print()
    print(table(results))
    for name, s in results.items():
        print(f"- {name}: 호출 {s['calls']}회 {s['calls_by_tag']}, 판단 경로 {s['routes']}")
    for name in a.models:
        rows = [r for r in all_rows if r["model"] == name]
        print(f"\n## {name} 기관별\n{by_org(rows)}")
        bad = [r for r in rows if not r["ok"]]
        if bad:
            print(f"\n틀리거나 놓친 사례 ({len(bad)})")
            for r in bad:
                print(f"- {r['menu']} '{r['goal']}' → {r['decision']} (정답 {r['expected']}) kind={r['kind']}"
                      f"{' 대체' if r['fallback'] else ''}")


if __name__ == "__main__":
    asyncio.run(main())
