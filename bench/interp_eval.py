"""모델별 안내 해석·판단 평가 (되묻기 판정 포함).

  python -m bench.interp_eval --models gemini-3.5-flash-lite gemini-3.6-flash
  python -m bench.interp_eval --interpreters gemini_interpreter:GeminiInterpreter local_llm:LocalLlmInterpreter
  python -m bench.interp_eval --hyp bench/out/stt.json --hyp-model gemini     # 정답 문장 대신 STT 인식 문장으로

방식
  시나리오의 목적 변형마다 해석기를 새로 만들고, 녹음 단락을 통화 순서대로 넣는다 (서버와 같은 순서·같은 판단 규칙).
  - 단락마다 classify → 안내 종류·선택지를 정답과 비교
  - 판단 지점(시나리오 키 경로에서 키를 누르거나 번호를 입력한 단락)에서는 서버와 같은 판단을 낸다
      menu    → navigator.decide (신뢰도 기준·같은 종류 선택지 규칙·이전 계층 확인 포함)
      confirm → 중요 선택이면 ask:critical_action, 아니면 answer_confirm (키를 못 정하면 ask:unknown_prompt)
      입력값 되읽기 → 항상 ask:unknown_prompt (서버 규칙)
      sensitive → secure
  - 다음 단락으로 넘어갈 때는 모델 판단과 관계없이 녹음된 경로대로 지나온 것으로 둔다 (단계별 독립 채점)

지표 (모델마다)
  안내 종류 정확도, 선택지 추출 정밀도·재현율(키+이름), 판단 정확도,
  되묻기 판정: 정밀도(물은 것 중 물어야 했던 비율)·재현율(물어야 할 때 물은 비율),
  위험 자동입력(누르면 안 되는 곳에서 누름 + 다른 키를 누름), 오류(시간 초과·예외),
  속도: 단락 해석(classify) 중앙값·95%, 판단(decide) 중앙값·95%, 판단 지점 합계(해석+판단) 중앙값·95%
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from bench import metrics as M
from bench.data import DATA, decision_steps, load_recordings, load_scenarios


def _label_key(label: str) -> str:
    return M.norm(label)


def option_scores(pred: list, gold: list, keys_only: bool = False) -> tuple[int, int, int]:
    """(맞은 수, 예측 수, 정답 수). 키가 같고 이름이 비슷하면(0.6 이상) 맞은 것으로 본다.
    keys_only: 확인 질문은 서버가 선택지 이름을 동작 이름('확인'·'다시 입력')으로 바꾸므로 키만 비교한다."""
    from difflib import SequenceMatcher

    used = set()
    ok = 0
    for k, lab in pred:
        for i, (gk, glab) in enumerate(gold):
            if i in used or str(gk) != str(k):
                continue
            if keys_only or SequenceMatcher(None, _label_key(lab), _label_key(glab)).ratio() >= 0.6:
                used.add(i)
                ok += 1
                break
    return ok, len(pred), len(gold)


async def decide(nav, interp, st, p, layers, goal: str, readback: bool) -> str:
    """서버(session._decide_inner·_paragraph)와 같은 규칙으로 판단 표기를 만든다."""
    if p.kind == "sensitive":
        return "secure"
    if readback:
        return "ask:unknown_prompt" if p.kind in ("menu", "confirm") else f"none:{p.kind}"
    if p.kind == "confirm":
        if p.critical:
            return "ask:critical_action"
        key = await interp.answer_confirm(goal, p)
        return f"press:{key}" if key else "ask:unknown_prompt"
    if p.kind == "menu":
        dec = await nav.decide(st, p, layers, True)
        if dec.action == "press":
            return f"press:{dec.key}"
        if dec.action == "ask":
            return f"ask:{dec.reason}"
        if dec.action == "backtrack":
            return "backtrack"
        return f"end:{dec.reason}"
    return f"none:{p.kind}"


async def run_goal(make_interp, settings, scenario: dict, rec, goal: dict, hyps: dict | None) -> list[dict]:
    from navigator import Layer, NavState, Navigator
    from session import _READBACK_RE, _SPOKEN_RUN_RE, mask_digits

    interp = make_interp()
    nav = Navigator(interp, settings.match_threshold, settings.margin, settings.max_revisit, settings.max_presses,
                    settings.low_threshold)
    text_goal = goal["goal"]
    t = time.monotonic()
    await interp.summarize_goal(text_goal)
    rows = [{"type": "goal", "time": time.monotonic() - t}]
    st = NavState(text_goal)
    layers: list = []
    steps = decision_steps(scenario)
    expect = goal.get("expect", [])
    for c in rec.clips:
        text = hyps.get((rec.name, c.path.name), c.text) if hyps is not None else c.text
        if not M.norm(text):
            continue
        readback = bool(_READBACK_RE.search(text)) and (bool(c.digits) or bool(_SPOKEN_RUN_RE.search(text)))
        if readback:
            text, _ = mask_digits(text)  # 서버는 되읽기 숫자를 가린 뒤 해석기에 넘긴다
        row = {"type": "clip", "file": c.path.name, "gold_kind": c.kind}
        t = time.monotonic()
        try:
            p = await interp.classify(text)
            row["error"] = None
        except Exception as e:
            from interpreter import Prompt

            p, row["error"] = Prompt(kind="info", text=text), f"{type(e).__name__}: {e}"
        row["classify_time"] = time.monotonic() - t
        row["kind"] = p.kind
        row["options"] = [[o.key, o.label] for o in p.options]
        if c.options:
            row["opt"] = option_scores(row["options"], c.options, keys_only=c.kind == "confirm")
        if c.decision is not None and c.decision < len(steps):
            exp = expect[c.decision] if c.decision < len(expect) else None
            t = time.monotonic()
            try:
                pred = await decide(nav, interp, st, p, layers, text_goal, readback)
            except Exception as e:
                pred, row["error"] = "error", f"{type(e).__name__}: {e}"
            row.update(decide_time=time.monotonic() - t, expect=exp, pred=pred, step=c.decision)
            step = steps[c.decision]
            if "key" in step and p.kind == "menu":  # 녹음된 경로대로 지나온 것으로 둔다
                layers.append(Layer(p, step["key"], "user"))
                st.presses += 1
        rows.append(row)
    return rows


def summarize(name: str, runs: list[list[dict]]) -> dict:
    kind_ok = kind_n = 0
    o_ok = o_pred = o_gold = 0
    dec_ok = dec_n = danger = errors = 0
    ask_tp = ask_fp = ask_fn = 0
    t_cls, t_dec, t_point, t_goal = [], [], [], []
    for rows in runs:
        for r in rows:
            if r["type"] == "goal":
                t_goal.append(r["time"])
                continue
            t_cls.append(r["classify_time"])
            errors += bool(r.get("error"))
            if r.get("gold_kind"):
                kind_n += 1
                kind_ok += r["kind"] == r["gold_kind"]
            if "opt" in r:
                a, b, c = r["opt"]
                o_ok, o_pred, o_gold = o_ok + a, o_pred + b, o_gold + c
            if "pred" in r:
                t_dec.append(r["decide_time"])
                t_point.append(r["decide_time"] + r["classify_time"])
                exp = r["expect"]
                if M.scored(exp):
                    dec_n += 1
                    dec_ok += M.decision_match(exp, r["pred"])
                    danger += M.dangerous(exp, r["pred"])
                    want_ask, did_ask = M.expects_ask(exp), r["pred"].startswith("ask")
                    ask_tp += want_ask and did_ask
                    ask_fp += did_ask and not want_ask and not M.decision_match(exp, r["pred"])
                    ask_fn += want_ask and not did_ask
    return {
        "model": name, "runs": len(runs),
        "kind_acc": M.ratio(kind_ok, kind_n), "kind_n": f"{kind_ok}/{kind_n}",
        "opt_precision": M.ratio(o_ok, o_pred), "opt_recall": M.ratio(o_ok, o_gold),
        "decision_acc": M.ratio(dec_ok, dec_n), "decision_n": f"{dec_ok}/{dec_n}",
        "ask_precision": M.ratio(ask_tp, ask_tp + ask_fp), "ask_recall": M.ratio(ask_tp, ask_tp + ask_fn),
        "dangerous": danger, "errors": errors,
        "classify_p50": M.pct(t_cls, 0.5), "classify_p95": M.pct(t_cls, 0.95),
        "decide_p50": M.pct(t_dec, 0.5), "decide_p95": M.pct(t_dec, 0.95),
        "point_p50": M.pct(t_point, 0.5), "point_p95": M.pct(t_point, 0.95),
        "goal_p50": M.pct(t_goal, 0.5),
        "runs_detail": runs,
    }


def table(results: list[dict]) -> str:
    head = ("| 모델 | 안내 종류 | 선택지 정밀도 | 선택지 재현율 | 판단 정확도 | 되묻기 정밀도 | 되묻기 재현율 | 위험 자동입력 | 오류 "
            "| 해석 중앙값 | 해석 95% | 판단 지점 중앙값 | 판단 지점 95% |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines = [head]
    for r in results:
        lines.append(f"| {r['model']} | {M.fmt(r['kind_acc'])} ({r['kind_n']}) | {M.fmt(r['opt_precision'])} | {M.fmt(r['opt_recall'])} "
                     f"| {M.fmt(r['decision_acc'])} ({r['decision_n']}) | {M.fmt(r['ask_precision'])} | {M.fmt(r['ask_recall'])} "
                     f"| {r['dangerous']} | {r['errors']} | {M.fmt(r['classify_p50'], 's')} | {M.fmt(r['classify_p95'], 's')} "
                     f"| {M.fmt(r['point_p50'], 's')} | {M.fmt(r['point_p95'], 's')} |")
    return "\n".join(lines)


async def evaluate(name: str, make_interp, recs, scenarios: dict, settings, hyps=None, repeat: int = 1) -> dict:
    runs = []
    for rec in recs:
        sc = scenarios.get(rec.scenario)
        if not sc:
            continue
        for g in sc.get("goals", []):
            for _ in range(repeat):
                runs.append(await run_goal(make_interp, settings, sc, rec, g, hyps))
    return summarize(name, runs)


def load_hyps(path: str, model: str) -> dict:
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        if r["model"] == model:
            return {(row["recording"], row["file"]): row["hyp"] for row in r["rows"]}
    raise SystemExit(f"{path} 에 {model} 결과가 없습니다")


async def main() -> None:
    import config
    from interpreter import load_interpreter

    ap = argparse.ArgumentParser(description="모델별 안내 해석·판단 평가")
    ap.add_argument("--interpreters", nargs="+", default=["gemini_interpreter:GeminiInterpreter"],
                    help="모듈:클래스 (Interpreter 프로토콜 구현)")
    ap.add_argument("--models", nargs="+", help="Gemini 모델 이름들. 주면 각 모델로 GeminiInterpreter 를 평가")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--scenarios", help="쉼표로 구분한 시나리오 ID")
    ap.add_argument("--repeat", type=int, default=1, help="목적 변형마다 반복 횟수 (응답 흔들림 확인)")
    ap.add_argument("--hyp", help="stt_eval 결과 JSON. 주면 정답 문장 대신 그 인식 문장으로 평가")
    ap.add_argument("--hyp-model", help="--hyp 에서 쓸 STT 모델 이름")
    ap.add_argument("--json", help="결과 저장 파일")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    settings = config.Settings()
    scenarios = load_scenarios()
    recs = load_recordings(Path(a.data), set(a.scenarios.split(",")) if a.scenarios else None)
    if not recs:
        raise SystemExit("평가할 녹음이 없습니다 (bench/data)")
    hyps = load_hyps(a.hyp, a.hyp_model) if a.hyp else None
    targets = []
    if a.models:
        for m in a.models:
            def make(m=m):
                os.environ["GEMINI_MODEL"] = m
                return load_interpreter("gemini_interpreter:GeminiInterpreter")
            targets.append((m, make))
    else:
        for spec in a.interpreters:
            targets.append((spec, lambda spec=spec: load_interpreter(spec)))
    results = []
    for name, make in targets:
        print(f"평가 중: {name}")
        results.append(await evaluate(name, make, recs, scenarios, settings, hyps, a.repeat))
    src = f"STT {a.hyp_model} 인식 문장" if hyps else "정답 문장"
    print(f"\n## 안내 해석·판단 (모델별, 입력: {src})\n" + table(results))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"\n저장: {a.json}")


if __name__ == "__main__":
    asyncio.run(main())
