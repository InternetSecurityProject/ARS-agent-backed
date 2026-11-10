"""
녹음 기반 가상 ARS 검증: 지표 타당성 대조

  python replay_validate_controls.py            # 결과: OUT_DIR/result.json, items.json

replay_ars·replay_check 와 전체 묶음(BUNDLE)으로, 주 지표 M3(MFCC DTW 코사인 유사도)와 참고 지표 M1(로그 에너지 윤곽 DTW
상관)이 같은 안내와 다른 안내를 가르는지를 실제 녹음끼리의 비교(양성 대조)와 다른 상태 소리와의 비교(음성 대조)로 측정.
묶음·녹음·기존 파일은 변경하지 않음.

1. 표본 내 엔진 경로(검사 통화의 녹음도 조각 출처에 포함된 전체 묶음으로 실행)
   - 대상 통화와 구간은 replay_check.check_call 과 같음(oracle 켬 = 변형은 원래 통화와 가장 닮은 것으로 선택, 씨앗 = 폴더 이름
     MD5, 키 시각 = keys_sent t + 0.25i + 0.15초, 0.6초 미만 구간 제외, 가상 ARS 정지 뒤 구간 제외, 양쪽 무음 구간 제외).
     통화마다 check_call 을 그대로 호출하고, 같은 인자로 run_engine 을 재실행해 경로(path)가 같음을 확인한 뒤 그 소리·경로를
     사용. 구간별 M3·M1 을 replay_check.py --out 으로 저장한 검사 결과(REF_CHECK)와 맞춰 재현 여부를 기록.
   - 기준 구간: 구간 [a, b2) 안에서 엔진이 상태에 하나 이상 들어갔고(states_entered), 원래 녹음에 말소리 덩어리가 있는 구간.
   - 구간 표지: 시작 상태(state_at_start, 통화 시작 구간은 '^call'), 그 상태에 들어선 뒤 a+0.03초까지 받은 키열(민감정보 입력
     자리는 'IN' 하나로 묶음), 구간 첫 진입 상태와 그 변형, 상태 집합 = {시작 상태} ∪ 구간 안 진입 상태.
2. 지표: replay_check.mfcc_frames·mfcc_sim(M3), envelope·dtw_corr(M1)를 그대로 사용. 한쪽에만 말소리 덩어리(islands)가 있으면
   0점, 양쪽 모두 없으면 그 비교를 제외(check_call 과 같은 규칙). 가림은 각 녹음 자기 키 톤 구간(톤 끝 −0.17 ~ +0.05초).
   창 자르기: 표본 수 int(round(L×8000)), 윤곽 칸 수 int(L/0.05), 시작은 각 구간 시작 a(구간 전체 비교는 check_call 과 같은
   int(a×8000)~int(b2×8000), int(a/0.05)~int(b2/0.05)).
3. 양성 대조(실제-같은 상태): 기준 구간을 (기관, 시작 상태, 그 상태에서 받은 키열, 첫 진입 상태, 변형) 무리로 나누고, 무리 키 순서·
   (통화 순서, 구간 순서)대로 기준 구간마다 다른 통화의 같은 무리 구간 하나를 무작위로 선택(씨앗 SEED_PAIR). 두 원래 녹음을
   짧은 쪽 길이 L 로 앞에서부터 잘라(L ≥ 1초, 양쪽 M3 말소리 프레임 × 0.01초 ≥ 0.5초) M3·M1 으로 비교. 같은 기준 구간의
   재생 대 원래도 같은 창 L 로 계산해 짝 비교(둘 다 통과·재생만·실제만·둘 다 실패, 짝지은 차)를 산출.
4. 음성 대조
   (a) 실제-다른 상태: 기준 구간마다 같은 기관 다른 통화의 기준 구간 C 중 첫 진입 상태가 기준 구간 상태 집합에 없고, 그 상태
       문구와 기준 구간 상태 문구들의 2글자 Jaccard 최댓값 < 0.6 이며, 공통 창 L2 = min(기준 길이, C 길이) ≥ 1초에서 기준 말소리
       ≥ 0.5초, C 말소리가 기준의 ±30% 안인 것 하나를 무작위로 선택(씨앗 SEED_PAIR). 민감도용 (a') 엄격판은 C 의 창 안 진입 상태
       전부에 같은 조건 적용(씨앗 SEED_PAIR + 1). 문구 같은 실녹음 쌍은 씨앗 SEED_PAIR + 2 로 선택.
   (b) 틀린 상태 조각: 재생 소리에서 구간 첫 진입 시각부터 구간 끝까지를 0 으로 지우고 그 자리에 같은 기관 다른 상태 조각을
       넣어(구간 끝에서 자름) 원래 구간과 비교. 후보는 개인정보가 아닌 상태의 조각(변형 포함) 중 기준 구간 상태 집합의 상태·
       조각이 아니고, 문구 Jaccard(기준 상태 문구들 대비 최댓값) < 0.6, 넣는 창 W = b2 − 진입 시각 안의 조각 말소리 길이가 원래 구간
       말소리 길이의 ±30% 안인 것. 구간마다 최대 3개를 무작위로(씨앗 '<SEED_PAIR>:<폴더 이름>') 선택. 구간 전체와, 양성 대조 창 L
       (진입이 창 안일 때)에서 각각 비교. 첫 진입 상태가 개인정보 상태이거나 W < 0.5초인 구간은 넣을 소리가 없어 제외.
   (c) 다른 기관 조각: 같은 방식으로 다른 기관 조각 1개(말소리 길이 ±30% 후보가 있으면 그중, 없으면 전체에서 무작위).
   문구가 같은 다른 상태(첫 진입 상태 문구와 Jaccard ≥ 0.6, 상태 ID 다름)는 음성 대조에서 빼고 따로 집계: (b) 방식의 형제 조각
   전부, (a) 방식의 실녹음 쌍(기준 구간마다 최대 1개). 소리로는 구별할 수 없는 쌍의 목록과 M3 통과율을 산출.
   문구는 묶음의 paras 받아쓰기를 이은 것(없으면 summary)이며 메모리 안에서만 쓰고 산출물에 남기지 않음(공백·문장부호 제거 뒤
   2글자 집합 Jaccard).
5. 보고: 각 분포의 문턱 0.80·0.85·0.90·0.925·0.95·0.97 이상 비율(k/n, Wilson 95%, 통화 단위 군집 부트스트랩 95%:
   1,000회, 씨앗 7, 군집 = 기준 구간의 통화), 분포 요약, 음성 대조 최댓값, AUC(Mann–Whitney, 동률 0.5, 같은 군집 부트스트랩),
   오일치가 0 일 때 상한 3/n(n = 항목, 서로 다른 (첫 진입 상태, 틀린 조각/상태) 쌍, 구간), 0.90 에서 오불일치. 기관별은 0.90 기준
   k/n 과 구간을 내고 n < 10 은 근거 미약으로 표시.
6. 엔진 표지 민감도(보조 수치): 양성·음성 대조의 '같은 상태/다른 상태' 표지는 시험 대상인 엔진 경로에서 나옴. 엔진 경로가 실제
   ARS 와 다른 구간(예: 실제 ARS 가 늦은 키를 무시하고 시간 초과 안내를 냄)에서는 표지가 틀림. 그래서 (i) 기준 구간(실녹음 쌍은
   짝 구간도)이 첫 미검증 표시 이전인 것만(비순환), (ii) 표본 내 재생이 M3 0.90 이상인 것만(순환; 표지 오류의 영향 범위 확인용
   참고)으로 0.90 통과율·최댓값·AUC 를 다시 산출. M3 0.85 이상 음성 대조 항목과 M3 0.90 미만 양성 대조 항목은 상태 ID·점수·
   재생 점수·표시 여부로 목록 기록. M3 0.90 미만 양성 쌍에는 진단으로 DTW 띠를 ±5초로 넓힌 M3 와 두 녹음의 첫 말소리
   시각 추가(판정에는 쓰지 않음). 1~5 가 주 결과이고 6 은 해석을 돕는 보조 수치.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import replay_check as rc
from audio import ulaw_to_pcm16
from replay_ars import FRAME_S, ReplayBundle

BUNDLE = r"C:\Users\User\Documents\ars_run\replay\bundle"
REF_CHECK = r"C:\Users\User\Documents\ars_run\replay\check_v3_oracle.json"
OUT_DIR = r"C:\Users\User\Documents\ars_run\replay\validate\controls"
SEED_PAIR = 20261109
SEED_BOOT = 7
N_BOOT = 1000
THRESHOLDS = (0.80, 0.85, 0.90, 0.925, 0.95, 0.97)
MAIN_THR = rc.SAME_M3
MIN_WIN_S = 1.0
MIN_ACTIVE_S = 0.5
LEN_TOL = 0.30
TEXT_SAME = 0.6
N_WRONG = 3
MIN_INSERT_S = 0.5
ORG_MIN_N = 10
LRU_CALLS = 6

# 상태 ------------------------------------------------------------------------
_B: ReplayBundle | None = None
_CLIPS: dict | None = None      # 기관 → [{sid, variant, rel, pcm(int16), isl, text}]
_TEXT: dict | None = None       # (기관, 상태) → 문구 (메모리 안에서만)
_PERSONAL: dict | None = None   # (기관, 상태) → 개인정보 상태 여부
_LRU: "OrderedDict[str, dict]" = OrderedDict()


def _norm_text(t: str) -> str:
    return re.sub(r"[\s\W_]+", "", t or "")


def bigram_jaccard(a: str, b: str) -> float | None:
    """공백·문장부호를 지운 두 문구의 2글자 집합 Jaccard. 어느 한쪽이 2글자 미만이면 None."""
    a, b = _norm_text(a), _norm_text(b)
    if len(a) < 2 or len(b) < 2:
        return None
    x = {a[i:i + 2] for i in range(len(a) - 1)}
    y = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(x & y) / len(x | y)


def state_text(st: dict) -> str:
    t = " ".join(p.get("text", "") for p in st.get("paras") or [])
    return t if t.strip() else (st.get("summary") or "")


def _init() -> None:
    global _B, _CLIPS, _TEXT, _PERSONAL
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    _B = ReplayBundle(BUNDLE)
    _CLIPS, _TEXT, _PERSONAL = {}, {}, {}
    for org, o in _B.orgs.items():
        lst = []
        for sid, st in o["states"].items():
            _TEXT[(org, sid)] = state_text(st)
            _PERSONAL[(org, sid)] = bool(st.get("personal"))
            if st.get("personal") or not st.get("clip"):
                continue
            clips = [(st["clip"], None, st.get("paras"))]
            if st.get("variants"):
                clips = [(v["clip"], v.get("name"), v.get("paras")) for v in st["variants"]]
            for rel, vname, paras in clips:
                pcm16 = np.frombuffer(ulaw_to_pcm16(_B.clip(rel)), dtype=np.int16).copy()
                pf = pcm16.astype(np.float64)
                isl = [(s, e) for s, e, _ in rc.islands(pf, 0.0, len(pf) / 8000, [])]
                txt = " ".join(p.get("text", "") for p in (paras or []))
                lst.append({"org": org, "sid": sid, "variant": vname, "rel": rel, "pcm": pcm16, "isl": isl,
                            "text": txt if txt.strip() else _TEXT[(org, sid)]})
        _CLIPS[org] = lst


def speech_in(isl, w: float) -> float:
    return sum(max(0.0, min(e, w) - s) for s, e in isl if s < w)


# 통화 준비 (replay_check.check_call 과 같은 절차) ---------------------------------
def prepare_call(bundle: ReplayBundle, dpath: str, org: str, engine: bool = True) -> dict | None:
    d = Path(dpath)
    try:
        j, pcm = rc.load_call(d)
    except Exception:
        return None
    if j.get("backend") not in (None, "clawops") or len(pcm) < 8000:
        return None
    keys = rc.key_events(j, pcm)
    end_s = len(pcm) / 8000
    mask = rc.key_mask(keys)
    eo = rc.envelope(pcm, mask)
    ctx = {"dir": d.name, "dpath": dpath, "org": org, "pcm": pcm, "keys": keys, "mask": mask, "eo": eo, "end_s": end_s}
    if not engine:
        return ctx
    seed = int(hashlib.md5(d.name.encode("utf-8")).hexdigest()[:8], 16)
    eng, out, srcs, _ = rc.run_engine(bundle, org, rc.call_clock(d.name), keys, end_s, eo, oracle=True, seed=seed)
    stop_s = eng.frame * FRAME_S if eng.ended else end_s
    sources = rc.clip_sources(bundle, org)
    marks = [0.0] + [t for t, *_ in keys if t < end_s] + [end_s]
    path = eng.path
    segs = []
    for a, b in zip(marks, marks[1:]):
        if b - a < rc.MIN_SEG_S or a >= stop_s:
            continue
        b2 = min(b, stop_s)
        io, ir = rc.islands(pcm, a, b2, mask), rc.islands(out, a, b2, mask)
        if not io and not ir:
            continue
        cur = [p["state"] for p in path if "state" in p and p["t"] < a]
        entered = [(p["t"], p["state"], p.get("variant")) for p in path if "state" in p and a <= p["t"] < b2]
        # 시작 상태에 들어선 뒤 a+0.03초까지 받은 키열 (경로 순서; 민감정보 입력은 'IN' 하나로)
        ks, st_cur = [], None
        for p in path:
            if p["t"] > a + 0.03:
                break
            if "state" in p:
                if p["t"] < a:
                    st_cur, ks = p["state"], []
                else:
                    break
            elif "key" in p and st_cur is not None:
                k = "IN" if p["key"] == "*입력*" else p["key"]
                if not (k == "IN" and ks and ks[-1] == "IN"):
                    ks.append(k)
        start = a == 0.0
        f0, f1 = int(a / FRAME_S), int(b2 / FRAME_S)
        segs.append({
            "a": a, "b2": b2, "dur": b2 - a, "start": start,
            "io": io, "ir": ir, "orig_sound": bool(io), "replay_sound": bool(ir),
            "orig_sp": sum(e - s for s, e, _ in io),
            "state_at_start": "^call" if start else (cur[-1] if cur else None),
            "keys_in_state": "" if start else "".join(ks),
            "entered": entered, "states_entered": [x[1] for x in entered],
            "self_src": any(sources.get(s) == d.name for s in {s for s in srcs[f0:f1] if s}),
        })
    ctx.update({"eng": eng, "out": out, "srcs": srcs, "er": rc.envelope(out, mask), "stop_s": stop_s, "segs": segs,
                "sources": sources})
    return ctx


def state_set(seg: dict) -> set:
    s = set(seg["states_entered"])
    if seg["state_at_start"] not in (None, "^call"):
        s.add(seg["state_at_start"])
    return s


def win_bounds(a: float, L: float):
    i0, ia = int(a * 8000), int(a / rc.ENV_S)
    return i0, i0 + int(round(L * 8000)), ia, ia + int(L / rc.ENV_S), a + L


def full_bounds(a: float, b2: float):
    return int(a * 8000), int(b2 * 8000), int(a / rc.ENV_S), int(b2 / rc.ENV_S), b2


def feat(sig: np.ndarray, env: np.ndarray, a: float, bounds, mask, isl: bool | None = None) -> dict:
    """비교 한쪽의 특징: MFCC 단위 벡터·말소리 프레임(M3), 50ms 윤곽 조각(M1), 말소리 덩어리 유무(한쪽 0점 규칙)."""
    i0, i1, ia, ib, b_end = bounds
    mf, act = rc.mfcc_frames(sig[i0:i1], a, mask)
    if isl is None:
        isl = bool(rc.islands(sig, a, b_end, mask))
    return {"mf": mf, "act": act, "env": env[ia:ib], "isl": isl}


def score(fa: dict, fb: dict):
    """(M3, M1). 양쪽 모두 말소리 덩어리가 없으면 None, 한쪽만이면 (0, 0). 지표가 정의되지 않으면 0 (check_call 과 같음)."""
    if not fa["isl"] and not fb["isl"]:
        return None
    if fa["isl"] != fb["isl"]:
        return 0.0, 0.0
    m3 = rc.mfcc_sim(fa["mf"], fa["act"], fb["mf"], fb["act"])
    m1 = rc.dtw_corr(fa["env"], fb["env"])
    return (0.0 if m3 is None else m3), (0.0 if m1 is None else m1)


def act_cum(pcm: np.ndarray, a: float, b2: float, mask) -> np.ndarray:
    _, act = rc.mfcc_frames(pcm[int(a * 8000):int(b2 * 8000)], a, mask)
    return np.cumsum(act).astype(np.int32)


def active_s(cum: np.ndarray, L: float) -> float:
    n = int(round(L * 8000))
    nf = 0 if n < rc.SPEC_WIN else 1 + (n - rc.SPEC_WIN) // rc.SPEC_HOP
    nf = min(nf, len(cum))
    return 0.0 if nf <= 0 else float(cum[nf - 1]) * 0.01


def _r(x, k=6):
    return None if x is None else round(float(x), k)


# 1단계: 표본 내 검사 재현과 구간 표지 ---------------------------------------------
def stage1_work(arg):
    dpath, org = arg
    rec = rc.check_call(_B, Path(dpath), org, oracle=True)
    if rec is None:
        return None
    ctx = prepare_call(_B, dpath, org)
    same_path = rec["path"] == ctx["eng"].path
    aligned = len(rec["segs"]) == len(ctx["segs"]) and all(
        r["from"] == round(s["a"], 2) and r["to"] == round(s["b2"], 2) and r["orig_sound"] == s["orig_sound"]
        and r["replay_sound"] == s["replay_sound"] and r["states_entered"] == s["states_entered"]
        and r["self_src"] == s["self_src"] for r, s in zip(rec["segs"], ctx["segs"]))
    segs = []
    for i, (r, s) in enumerate(zip(rec["segs"], ctx["segs"])):
        first = s["entered"][0] if s["entered"] else None
        segs.append({
            "i": i, "a": s["a"], "b2": s["b2"], "dur": s["dur"], "start": s["start"],
            "orig_sound": s["orig_sound"], "replay_sound": s["replay_sound"], "orig_sp": s["orig_sp"],
            "state_at_start": s["state_at_start"], "keys_in_state": s["keys_in_state"],
            "first": None if first is None else {"t": first[0], "sid": first[1], "variant": first[2]},
            "entered": s["entered"], "states_entered": s["states_entered"], "set": sorted(state_set(s)),
            "self_src": s["self_src"], "personal_boundary": r["personal_boundary"],
            "m3": r["m3"], "m1": r["m1"], "same": r["same"],
            "cum": act_cum(ctx["pcm"], s["a"], s["b2"], ctx["mask"]),
        })
    rec_small = {k: rec[k] for k in ("dir", "org", "start", "end_s", "replay_end_s", "offmap", "personal", "early",
                                     "flags", "keys")}
    rec_small["segs"] = [{k: v for k, v in r.items()} for r in rec["segs"]]
    return {"dir": ctx["dir"], "dpath": dpath, "org": org, "same_path": same_path, "aligned": aligned,
            "path": ctx["eng"].path, "rec": rec_small, "segs": segs}


# 2단계: 재생·틀린 조각·실녹음 쌍 점수 ---------------------------------------------
def _partner(dpath: str, org: str) -> dict:
    if dpath in _LRU:
        _LRU.move_to_end(dpath)
        return _LRU[dpath]
    ctx = prepare_call(_B, dpath, org, engine=False)
    _LRU[dpath] = ctx
    while len(_LRU) > LRU_CALLS:
        _LRU.popitem(last=False)
    return ctx


def _with_clip(out: np.ndarray, off: int, i1: int, clip16: np.ndarray):
    """재생 소리 out 의 [off, i1) 을 지우고 조각 삽입(제자리 수정). 반환값은 되돌릴 원래 값."""
    saved = out[off:i1].copy()
    out[off:i1] = 0.0
    kk = max(0, min(len(clip16), i1 - off))
    out[off:off + kk] = clip16[:kk].astype(np.float64)
    return saved


def stage2_work(arg):
    dpath, org, s1segs, anchors, rr_tasks, path_ref = arg
    ctx = prepare_call(_B, dpath, org)
    res = {"dir": ctx["dir"], "same_path": ctx["eng"].path == path_ref, "verify": [], "replay_win": [], "b": [],
           "c": [], "st_synth": [], "rr": [], "excluded_insert": 0}
    pcm, out, mask, eo, er = ctx["pcm"], ctx["out"], ctx["mask"], ctx["eo"], ctx["er"]
    segs = ctx["segs"]
    states = _B.orgs[org]["states"]
    rng = random.Random(f"{SEED_PAIR}:{ctx['dir']}")
    others = [c for o2, lst in _CLIPS.items() if o2 != org for c in lst]
    fo_cache = {}

    def fo_win(i, L):
        key = (i, round(L, 6))
        if key not in fo_cache:
            s = segs[i]
            fo_cache[key] = feat(pcm, eo, s["a"], win_bounds(s["a"], L), mask)
        return fo_cache[key]

    for i, s in enumerate(segs):
        a, b2 = s["a"], s["b2"]
        fb_full = full_bounds(a, b2)
        fo = feat(pcm, eo, a, fb_full, mask, isl=s["orig_sound"])
        fr = feat(out, er, a, fb_full, mask, isl=s["replay_sound"])
        sc = score(fo, fr)
        res["verify"].append((i, _r(sc[0]), _r(sc[1]), s1segs[i]["m3"], s1segs[i]["m1"]))
        L = anchors.get(i)
        frw = None
        if L is not None:
            fow = fo_win(i, L)
            frw = feat(out, er, a, win_bounds(a, L), mask)
            scw = score(fow, frw)
            f0, f1 = int(a / FRAME_S), int((a + L) / FRAME_S)
            self_w = any(ctx["sources"].get(x) == ctx["dir"] for x in {x for x in ctx["srcs"][f0:f1] if x})
            res["replay_win"].append({"i": i, "L": L, "m3": None if scw is None else _r(scw[0]),
                                      "m1": None if scw is None else _r(scw[1]), "self_win": self_w})
        # 틀린 상태 조각 (b), 다른 기관 조각 (c), 문구 같은 형제 조각
        if not s["entered"] or not s["orig_sound"]:
            continue
        t_entry, sid0, _ = s["entered"][0]
        W = b2 - t_entry
        if _PERSONAL.get((org, sid0)) or W < MIN_INSERT_S:
            res["excluded_insert"] += 1
            continue
        true_sids = state_set(s)
        true_rel = set()
        for sid in true_sids:
            st = states.get(sid) or {}
            if st.get("clip"):
                true_rel.add(st["clip"])
            for v in st.get("variants") or []:
                true_rel.add(v["clip"])
        true_txt = [_TEXT[(org, x)] for x in true_sids if (org, x) in _TEXT]
        first_txt = _TEXT.get((org, sid0), "")
        orig_sp = s["orig_sp"]
        diff, same_txt = [], []
        for c in _CLIPS[org]:
            if c["sid"] in true_sids or c["rel"] in true_rel:
                continue
            sims = [x for x in (bigram_jaccard(c["text"], t) for t in true_txt) if x is not None]
            if not sims:
                continue
            tj = max(sims)
            jf = bigram_jaccard(c["text"], first_txt)
            if tj < TEXT_SAME:
                if orig_sp > 0 and abs(speech_in(c["isl"], W) - orig_sp) <= LEN_TOL * orig_sp:
                    diff.append((c, tj))
            elif jf is not None and jf >= TEXT_SAME:
                same_txt.append((c, jf))
        picks = [("b", c, tj) for c, tj in rng.sample(diff, min(N_WRONG, len(diff)))]
        xl = [c for c in others if orig_sp > 0 and abs(speech_in(c["isl"], W) - orig_sp) <= LEN_TOL * orig_sp]
        xc = rng.choice(xl) if xl else rng.choice(others)
        picks.append(("c", xc, None))
        picks += [("st_synth", c, jf) for c, jf in same_txt]
        off, i1 = int(round(t_entry * 8000)), int(b2 * 8000)
        for kind, c, tj in picks:
            saved = _with_clip(out, off, i1, c["pcm"])
            try:
                env_w = rc.envelope(out[:min(len(out), (int(b2 / rc.ENV_S) + 2) * int(rc.ENV_S * 8000))], mask)
                fw = feat(out, env_w, a, fb_full, mask)
                scf = score(fo, fw)
                scw = None
                if L is not None and t_entry < a + L:
                    scw = score(fo_win(i, L), feat(out, env_w, a, win_bounds(a, L), mask))
            finally:
                out[off:i1] = saved
            item = {"i": i, "first": sid0, "wrong_org": c["org"], "wrong_sid": c["sid"], "wrong_variant": c["variant"],
                    "wrong_rel": c["rel"], "text_j": _r(tj, 3), "speech_in_W": round(speech_in(c["isl"], W), 2),
                    "orig_sp": round(orig_sp, 2), "len_matched": kind != "c" or bool(xl),
                    "m3": None if scf is None else _r(scf[0]), "m1": None if scf is None else _r(scf[1]),
                    "m3_win": None if scw is None else _r(scw[0]), "m1_win": None if scw is None else _r(scw[1])}
            res[kind].append(item)
    # 실녹음 쌍: 양성(같은 상태), 음성 (a)·(a'), 문구 같은 다른 상태
    for t in rr_tasks:
        i, kind, pdpath, porg, pa, L = t["i"], t["kind"], t["pdpath"], org, t["pa"], t["L"]
        pc = _partner(pdpath, porg)
        fa = fo_win(i, L)
        fb = feat(pc["pcm"], pc["eo"], pa, win_bounds(pa, L), pc["mask"])
        sc = score(fa, fb)
        row = {**{k: v for k, v in t.items() if k != "pdpath"},
               "m3": None if sc is None else _r(sc[0]), "m1": None if sc is None else _r(sc[1])}
        if kind == "pos" and sc is not None and sc[0] < MAIN_THR and fa["isl"] and fb["isl"]:
            # 진단(판정에는 쓰지 않음): 오불일치 양성 쌍의 DTW 띠를 ±5초로 넓힌 M3 와 두 녹음의 첫 말소리 시각(구간 시작 기준)
            ia = rc.islands(pcm, segs[i]["a"], segs[i]["a"] + L, mask)
            ib = rc.islands(pc["pcm"], pa, pa + L, pc["mask"])
            row["diag"] = {"m3_band5s": _r(rc.mfcc_sim(fa["mf"], fa["act"], fb["mf"], fb["act"], band=500), 4),
                           "onset_a": round(ia[0][0] - segs[i]["a"], 2) if ia else None,
                           "onset_b": round(ib[0][0] - pa, 2) if ib else None}
        res["rr"].append(row)
    return res


# 짝 고르기 ----------------------------------------------------------------------
def plan(calls1: list[dict], texts: dict):
    """양성 대조 짝, 음성 대조 (a)·(a') 짝, 문구 같은 실녹음 쌍 선정. 기준 구간 = 진입 상태가 있고 원래 말소리가 있는 구간."""
    ref = []  # (통화 번호, 구간 번호)
    for ci, c in enumerate(calls1):
        for s in c["segs"]:
            if s["first"] is not None and s["orig_sound"]:
                ref.append((ci, s["i"]))
    S = {(ci, i): calls1[ci]["segs"][i] for ci, i in ref}

    def gkey(ci, i):
        s = S[(ci, i)]
        return (calls1[ci]["org"], str(s["state_at_start"]), s["keys_in_state"], s["first"]["sid"],
                str(s["first"]["variant"]))

    groups = defaultdict(list)
    for ci, i in ref:
        groups[gkey(ci, i)].append((ci, i))
    rng = random.Random(SEED_PAIR)
    pos, pos_skip = [], {"no_partner": 0, "L_short": 0, "inactive": 0}
    for g in sorted(groups):
        items = sorted(groups[g])
        for ci, i in items:
            partners = [x for x in items if x[0] != ci]
            if not partners:
                pos_skip["no_partner"] += 1
                continue
            pj, pi = rng.choice(partners)
            sa, sb = S[(ci, i)], S[(pj, pi)]
            L = min(sa["dur"], sb["dur"])
            if L < MIN_WIN_S:
                pos_skip["L_short"] += 1
                continue
            aa, ab = active_s(sa["cum"], L), active_s(sb["cum"], L)
            if aa < MIN_ACTIVE_S or ab < MIN_ACTIVE_S:
                pos_skip["inactive"] += 1
                continue
            pos.append({"ci": ci, "i": i, "pj": pj, "pi": pi, "L": L, "group": list(g), "act_a": round(aa, 2),
                        "act_b": round(ab, 2)})
    # 음성 대조 (a): 기준 구간 전부
    by_org = defaultdict(list)
    for ci, i in ref:
        by_org[calls1[ci]["org"]].append((ci, i))
    jcache = {}

    def jac(org, x, y):
        k = (org, x, y) if x <= y else (org, y, x)
        if k not in jcache:
            jcache[k] = bigram_jaccard(texts.get((org, x), ""), texts.get((org, y), ""))
        return jcache[k]

    rng_a, rng_as, rng_st = random.Random(SEED_PAIR), random.Random(SEED_PAIR + 1), random.Random(SEED_PAIR + 2)
    neg_a, neg_as, st_real = [], [], []
    cnt = {"a_no_cand": 0, "as_no_cand": 0, "st_no_cand": 0, "refs": len(ref), "ambiguous_text_candidates": 0}
    for ci, i in sorted(ref):
        org = calls1[ci]["org"]
        s = S[(ci, i)]
        tset = set(s["set"])
        first = s["first"]["sid"]
        ca, cas, cst = [], [], []
        for cj, j in by_org[org]:
            if cj == ci:
                continue
            c = S[(cj, j)]
            sid2 = c["first"]["sid"]
            if sid2 in tset:
                continue
            sims = [x for x in (jac(org, sid2, t) for t in tset) if x is not None]
            if not sims:
                continue
            L2 = min(s["dur"], c["dur"])
            if L2 < MIN_WIN_S:
                continue
            a2 = active_s(s["cum"], L2)
            if a2 < MIN_ACTIVE_S:
                continue
            c2 = active_s(c["cum"], L2)
            if abs(c2 - a2) > LEN_TOL * a2:
                continue
            row = (cj, j, sid2, L2, a2, c2)
            if max(sims) < TEXT_SAME:
                ca.append(row)
                # 엄격판: C 의 창 안 진입 상태 전부가 기준 상태 집합 밖이고 문구가 다름
                win_states = [e[1] for e in c["entered"] if e[0] < c["a"] + L2]
                ok = True
                for w in win_states:
                    if w in tset:
                        ok = False
                        break
                    ws = [x for x in (jac(org, w, t) for t in tset) if x is not None]
                    if ws and max(ws) >= TEXT_SAME:
                        ok = False
                        break
                if ok:
                    cas.append(row)
            else:
                jf = jac(org, sid2, first)
                if jf is not None and jf >= TEXT_SAME:
                    cst.append(row + (jf,))
                else:
                    cnt["ambiguous_text_candidates"] += 1
        for lst, rg, out, key in ((ca, rng_a, neg_a, "a_no_cand"), (cas, rng_as, neg_as, "as_no_cand"),
                                  (cst, rng_st, st_real, "st_no_cand")):
            if not lst:
                cnt[key] += 1
                continue
            row = rg.choice(lst)
            d = {"ci": ci, "i": i, "pj": row[0], "pi": row[1], "other_sid": row[2], "L": row[3],
                 "act_a": round(row[4], 2), "act_b": round(row[5], 2), "n_cands": len(lst)}
            if len(row) > 6:
                d["text_j"] = round(row[6], 3)
            out.append(d)
    return {"ref": ref, "pos": pos, "pos_skip": pos_skip, "neg_a": neg_a, "neg_as": neg_as, "st_real": st_real,
            "counts": cnt}


# 통계 ---------------------------------------------------------------------------
def _boot_counts(n_clusters: int):
    rng = np.random.default_rng(SEED_BOOT)
    idx = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))
    cnt = np.zeros((N_BOOT, n_clusters))
    np.add.at(cnt, (np.repeat(np.arange(N_BOOT), n_clusters), idx.ravel()), 1.0)
    return cnt


def rate(cids, passed) -> dict:
    """k/n, Wilson 95%, 통화 군집 부트스트랩 95%."""
    cids = list(cids)
    passed = np.asarray(passed, dtype=float)
    n = len(cids)
    k = int(passed.sum()) if n else 0
    out = {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson": rc.wilson(k, n), "boot": None}
    if n:
        u = sorted(set(cids))
        pos = {c: x for x, c in enumerate(u)}
        ci = np.array([pos[c] for c in cids])
        K, N = np.bincount(ci, passed, len(u)), np.bincount(ci, None, len(u)).astype(float)
        cnt = _boot_counts(len(u))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (cnt @ K) / (cnt @ N)
        out["boot"] = [round(float(x), 4) for x in np.nanpercentile(r, [2.5, 97.5])]
    return out


def auc_weighted(pv, nv, wp=None, wn=None) -> float | None:
    pv, nv = np.asarray(pv, float), np.asarray(nv, float)
    if not len(pv) or not len(nv):
        return None
    wp = np.ones(len(pv)) if wp is None else wp
    wn = np.ones(len(nv)) if wn is None else wn
    o = np.argsort(nv, kind="mergesort")
    ns, ws = nv[o], wn[o]
    cw = np.concatenate([[0.0], np.cumsum(ws)])
    lo, hi = np.searchsorted(ns, pv, "left"), np.searchsorted(ns, pv, "right")
    below, eq = cw[lo], cw[hi] - cw[lo]
    den = wp.sum() * ws.sum()
    return float((wp * (below + 0.5 * eq)).sum() / den) if den > 0 else None


def auc_boot(pos_items, neg_items, key="m3") -> dict:
    """Mann–Whitney AUC 와 통화 군집 부트스트랩 95% (군집 = 기준 구간 통화, 양성·음성 공통으로 다시 뽑음)."""
    pv = [x[key] for x in pos_items]
    nv = [x[key] for x in neg_items]
    a = auc_weighted(pv, nv)
    if a is None:
        return {"auc": None, "boot": None, "n_pos": len(pv), "n_neg": len(nv)}
    u = sorted({x["cid"] for x in pos_items} | {x["cid"] for x in neg_items})
    pos = {c: i for i, c in enumerate(u)}
    cp = np.array([pos[x["cid"]] for x in pos_items])
    cn = np.array([pos[x["cid"]] for x in neg_items])
    cnt = _boot_counts(len(u))
    pv, nv = np.asarray(pv, float), np.asarray(nv, float)
    o = np.argsort(nv, kind="mergesort")
    ns = nv[o]
    lo, hi = np.searchsorted(ns, pv, "left"), np.searchsorted(ns, pv, "right")
    vals = []
    for b in range(N_BOOT):
        wp, wn = cnt[b, cp], cnt[b, cn][o]
        cw = np.concatenate([[0.0], np.cumsum(wn)])
        den = wp.sum() * wn.sum()
        if den > 0:
            vals.append(float((wp * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum() / den))
    return {"auc": round(a, 4), "boot": [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])],
            "n_pos": len(pv), "n_neg": len(nv)}


def dist(vals) -> dict | None:
    v = np.asarray([x for x in vals if x is not None], float)
    if not len(v):
        return None
    q = np.percentile(v, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {"n": int(len(v)), "min": round(float(v.min()), 4),
            **{f"p{p}": round(float(x), 4) for p, x in zip((1, 5, 10, 25, 50, 75, 90, 95, 99), q)},
            "max": round(float(v.max()), 4)}


def thr_table(items, key) -> dict:
    v = [x for x in items if x.get(key) is not None]
    return {f"{t:g}": rate([x["cid"] for x in v], [x[key] >= t for x in v]) for t in THRESHOLDS}


def describe(items, key) -> dict:
    v = [x for x in items if x.get(key) is not None]
    return {"thresholds": thr_table(v, key), "dist": dist([x[key] for x in v])}


def false_match(items, key, pair_key, seg_key="seg") -> dict:
    """문턱 0.90 이상 오일치: 항목·서로 다른 쌍·구간 단위. 0 이면 95% 상한 3/n."""
    v = [x for x in items if x.get(key) is not None]
    out = {}
    pairs = defaultdict(list)
    segs = defaultdict(list)
    for x in v:
        pairs[x[pair_key]].append(x[key])
        segs[x[seg_key]].append(x[key])
    for name, groups in (("items", {i: [x[key]] for i, x in enumerate(v)}), ("distinct_pairs", pairs),
                         ("segments", segs)):
        n = len(groups)
        k = sum(1 for g in groups.values() if max(g) >= MAIN_THR)
        out[name] = {"k": k, "n": n, "rate": round(k / n, 5) if n else None, "wilson": rc.wilson(k, n),
                     "upper95_rule_of_three": round(3 / n, 5) if (n and k == 0) else None}
    return out


def max_item(items, key, fields) -> dict | None:
    v = [x for x in items if x.get(key) is not None]
    if not v:
        return None
    m = max(v, key=lambda x: x[key])
    return {"value": m[key], **{f: m.get(f) for f in fields}}


def by_org(items, key, orgs) -> dict:
    out = {}
    for o in orgs:
        v = [x for x in items if x["org"] == o and x.get(key) is not None]
        r = rate([x["cid"] for x in v], [x[key] >= MAIN_THR for x in v])
        r["weak_evidence"] = r["n"] < ORG_MIN_N
        r["max"] = round(max(x[key] for x in v), 4) if v else None
        out[o] = r
    return out


# 실행 ---------------------------------------------------------------------------
def _list_calls(bundle: ReplayBundle):
    return [(str(d), bundle.org_for(ph)) for d, ph in rc.calls(None, bundle)]


def run(out_dir: str = OUT_DIR, workers: int = 6, limit: int | None = None) -> dict:
    t_start = time.time()
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    os.makedirs(out_dir, exist_ok=True)
    bundle = ReplayBundle(BUNDLE)
    texts = {(org, sid): state_text(st) for org, o in bundle.orgs.items() for sid, st in o["states"].items()}
    todo = _list_calls(bundle)
    if limit:
        todo = todo[:limit]
    with Pool(workers, initializer=_init) as pool:
        calls1 = [x for x in pool.imap(stage1_work, todo, chunksize=2) if x is not None]
    t1 = time.time()
    # 재현 확인: 저장된 검사 결과(REF_CHECK)와 구간별 M3·M1 대조
    refj = json.loads(Path(REF_CHECK).read_text(encoding="utf-8"))
    refc = {c["dir"]: c for c in refj["calls"]}
    repro = {"calls": len(calls1), "calls_ref": len(refj["calls"]), "same_path_rerun": sum(c["same_path"] for c in calls1),
             "aligned_rerun": sum(c["aligned"] for c in calls1), "segments": 0, "segments_ref_matched": 0,
             "m3_max_abs_diff": 0.0, "m1_max_abs_diff": 0.0, "segments_diff_gt_1e-4": 0, "calls_missing_in_ref": 0}
    for c in calls1:
        r = refc.get(c["dir"])
        segs = c["rec"]["segs"]
        repro["segments"] += len(segs)
        if r is None:
            repro["calls_missing_in_ref"] += 1
            continue
        rs = {x["from"]: x for x in r["segs"]}
        for s in segs:
            x = rs.get(s["from"])
            if x is None:
                continue
            repro["segments_ref_matched"] += 1
            d3, d1 = abs(x["m3"] - s["m3"]), abs(x["m1"] - s["m1"])
            repro["m3_max_abs_diff"] = max(repro["m3_max_abs_diff"], d3)
            repro["m1_max_abs_diff"] = max(repro["m1_max_abs_diff"], d1)
            repro["segments_diff_gt_1e-4"] += (d3 > 1e-4 or d1 > 1e-4)
    allsegs = [s for c in calls1 for s in c["rec"]["segs"]]
    nonstart = [s for s in allsegs if not s["start"]]
    repro["insample_all_M3"] = f"{sum(s['same'] for s in allsegs)}/{len(allsegs)}"
    repro["insample_nonstart_M3"] = f"{sum(s['same'] for s in nonstart)}/{len(nonstart)}"
    repro["ref_summary_all_M3"] = f"{refj['summary']['ALL']['segments']['k']}/{refj['summary']['ALL']['segments']['n']}"
    repro["ref_summary_nonstart_M3"] = f"{refj['summary']['ALL']['nonstart']['k']}/{refj['summary']['ALL']['nonstart']['n']}"

    pl = plan(calls1, texts)
    # 2단계 작업 묶기
    anchors = defaultdict(dict)
    rr = defaultdict(list)
    for p in pl["pos"]:
        anchors[p["ci"]][p["i"]] = p["L"]
    for kind, lst in (("pos", pl["pos"]), ("neg_a", pl["neg_a"]), ("neg_as", pl["neg_as"]), ("st_real", pl["st_real"])):
        for p in lst:
            rr[p["ci"]].append({"kind": kind, "i": p["i"], "pj": p["pj"], "pi": p["pi"], "L": p["L"],
                                "pdpath": calls1[p["pj"]]["dpath"], "pa": calls1[p["pj"]]["segs"][p["pi"]]["a"],
                                "text_j": p.get("text_j")})
    args = []
    for ci, c in enumerate(calls1):
        s1 = [{"m3": s["m3"], "m1": s["m1"]} for s in c["segs"]]
        args.append((c["dpath"], c["org"], s1, anchors.get(ci, {}), rr.get(ci, []), c["path"]))
    with Pool(workers, initializer=_init) as pool:
        res2 = list(pool.imap(stage2_work, args, chunksize=1))
    t2 = time.time()

    # 결과 모으기
    orgs = sorted({c["org"] for c in calls1})
    verify = {"same_path_stage2": sum(r["same_path"] for r in res2), "segments": 0, "m3_max_abs_diff": 0.0,
              "m1_max_abs_diff": 0.0}
    for r in res2:
        for i, m3, m1, r3, r1 in r["verify"]:
            verify["segments"] += 1
            verify["m3_max_abs_diff"] = max(verify["m3_max_abs_diff"], abs((m3 or 0) - r3))
            verify["m1_max_abs_diff"] = max(verify["m1_max_abs_diff"], abs((m1 or 0) - r1))

    def seginfo(ci, i):
        s = calls1[ci]["segs"][i]
        return {"cid": ci, "org": calls1[ci]["org"], "seg": f"{ci}:{i}", "start": s["start"],
                "personal_boundary": s["personal_boundary"], "first": s["first"]["sid"] if s["first"] else None,
                "state_at_start": s["state_at_start"], "after_flag": calls1[ci]["rec"]["segs"][i]["after_flag"],
                "replay_m3": s["m3"], "replay_m1": s["m1"]}

    rr_items = defaultdict(list)
    rep_win = {}
    b_items, c_items, sts_items = [], [], []
    excluded_insert = 0
    for ci, r in enumerate(res2):
        excluded_insert += r["excluded_insert"]
        for x in r["rr"]:
            it = {**seginfo(ci, x["i"]), "kind": x["kind"], "partner": f"{x['pj']}:{x['pi']}", "L": round(x["L"], 3),
                  "partner_first": calls1[x["pj"]]["segs"][x["pi"]]["first"]["sid"], "text_j": x.get("text_j"),
                  "partner_after_flag": calls1[x["pj"]]["rec"]["segs"][x["pi"]]["after_flag"],
                  "partner_replay_m3": calls1[x["pj"]]["segs"][x["pi"]]["m3"], "m3": x["m3"], "m1": x["m1"]}
            if x.get("diag"):
                it["diag"] = x["diag"]
            rr_items[x["kind"]].append(it)
        for x in r["replay_win"]:
            rep_win[(ci, x["i"])] = {**seginfo(ci, x["i"]), "L": round(x["L"], 3), "m3": x["m3"], "m1": x["m1"],
                                     "self_win": x["self_win"]}
        for lst, src in ((b_items, r["b"]), (c_items, r["c"]), (sts_items, r["st_synth"])):
            for x in src:
                lst.append({**seginfo(ci, x["i"]), **{k: v for k, v in x.items() if k != "i"},
                            "pair": f"{calls1[ci]['org']}|{x['first']}|{x['wrong_rel']}"})
    for kind in ("neg_a", "neg_as", "st_real"):
        for x in rr_items[kind]:
            x["pair"] = f"{x['org']}|{x['first']}|{x['partner_first']}"
    pos_items = [x for x in rr_items["pos"] if x["m3"] is not None]
    rw_items = [rep_win[(x["cid"], int(x["seg"].split(":")[1]))] for x in pos_items]
    b_win = [{**x, "m3": x["m3_win"], "m1": x["m1_win"]} for x in b_items if x["m3_win"] is not None]

    sets = {
        "pos_real_same": pos_items,
        "pos_real_same_nonstart": [x for x in pos_items if not x["start"]],
        "pos_real_same_not_personal_boundary": [x for x in pos_items if not x["personal_boundary"]],
        "replay_win": rw_items,
        "replay_win_not_self": [x for x in rw_items if not x["self_win"]],
        "replay_win_nonstart": [x for x in rw_items if not x["start"]],
        "neg_a_real_diff": [x for x in rr_items["neg_a"] if x["m3"] is not None],
        "neg_a_strict": [x for x in rr_items["neg_as"] if x["m3"] is not None],
        "neg_b_wrong_full": [x for x in b_items if x["m3"] is not None],
        "neg_b_wrong_win": b_win,
        "neg_c_xorg_full": [x for x in c_items if x["m3"] is not None],
        "sametext_synth_full": [x for x in sts_items if x["m3"] is not None],
        "sametext_real": [x for x in rr_items["st_real"] if x["m3"] is not None],
    }
    metrics = {}
    for mkey, label in (("m3", "M3"), ("m1", "M1")):
        m = {"sets": {k: describe(v, mkey) for k, v in sets.items()}}
        negs = ("neg_a_real_diff", "neg_a_strict", "neg_b_wrong_full", "neg_b_wrong_win", "neg_c_xorg_full")
        m["max_negative"] = {k: max_item(sets[k], mkey, ["org", "first", "state_at_start", "partner_first", "wrong_sid",
                                                          "wrong_org", "text_j", "L"]) for k in negs}
        m["false_match_0.90"] = {k: false_match(sets[k], mkey, "pair") for k in negs}
        m["false_non_match_0.90"] = {
            k: (lambda r: {"k_below": r["n"] - r["k"], "n": r["n"],
                           "rate": None if not r["n"] else round(1 - r["k"] / r["n"], 4),
                           "wilson": rc.wilson(r["n"] - r["k"], r["n"])})(thr_table(sets[k], mkey)["0.9"])
            for k in ("pos_real_same", "pos_real_same_nonstart", "replay_win", "replay_win_not_self")}
        m["auc"] = {f"pos_real_same_vs_{k}": auc_boot(sets["pos_real_same"], sets[k], mkey) for k in negs}
        m["auc"]["replay_win_vs_neg_b_wrong_win"] = auc_boot(sets["replay_win"], sets["neg_b_wrong_win"], mkey)
        m["auc"]["pos_real_same_vs_sametext_real"] = auc_boot(sets["pos_real_same"], sets["sametext_real"], mkey)
        m["by_org_0.90"] = {k: by_org(sets[k], mkey, orgs) for k in
                            ("pos_real_same", "replay_win", "neg_a_real_diff", "neg_b_wrong_full", "neg_c_xorg_full")}
        metrics[label] = m

    # 엔진 표지 민감도: 음성·양성 대조의 '상태' 표지는 엔진 경로에서 나오므로, 엔진 경로가 실제와 다르면(예: 실제 ARS 가 늦은
    # 키를 무시함) 표지가 틀림. (i) 첫 미검증 표시 이전 구간만(비순환), (ii) 표본 내 재생이 M3 0.90 이상인 구간만(순환; 표지
    # 오류의 영향 범위 확인용)으로 재집계. 실녹음 쌍은 짝 구간에도 같은 조건 적용.
    def keep(x, mode):
        if mode == "before_flag":
            return not x["after_flag"] and not x.get("partner_after_flag", False)
        return x["replay_m3"] >= MAIN_THR and (x.get("partner_replay_m3") is None or x["partner_replay_m3"] >= MAIN_THR)

    sens_sets = ("pos_real_same", "replay_win", "neg_a_real_diff", "neg_a_strict", "neg_b_wrong_full", "neg_b_wrong_win",
                 "neg_c_xorg_full", "sametext_synth_full", "sametext_real")
    sensitivity = {}
    for mode in ("before_flag", "replay_pass"):
        blk = {}
        for k in sens_sets:
            v = [x for x in sets[k] if keep(x, mode)]
            blk[k] = {label: {**{kk: vv for kk, vv in thr_table(v, mkey)["0.9"].items()},
                              "max": round(max(x[mkey] for x in v), 4) if v else None}
                      for mkey, label in (("m3", "M3"), ("m1", "M1"))}
        blk["false_match_0.90"] = {
            label: {k: false_match([x for x in sets[k] if keep(x, mode)], mkey, "pair")
                    for k in ("neg_a_real_diff", "neg_a_strict", "neg_b_wrong_full", "neg_b_wrong_win", "neg_c_xorg_full")}
            for mkey, label in (("m3", "M3"), ("m1", "M1"))}
        for mkey, label in (("m3", "M3"), ("m1", "M1")):
            pv = [x for x in sets["pos_real_same"] if keep(x, mode)]
            blk[f"auc_{label}"] = {k: auc_boot(pv, [x for x in sets[k] if keep(x, mode)], mkey)
                                   for k in ("neg_a_real_diff", "neg_b_wrong_full", "neg_b_wrong_win", "neg_c_xorg_full")}
        sensitivity[mode] = blk

    # 진단 목록: M3 0.85 이상 음성 대조 항목과 M3 0.90 미만 양성 대조 항목 (상태 ID·점수만)
    diag_fields = ("seg", "org", "state_at_start", "first", "partner", "partner_first", "wrong_org", "wrong_sid", "text_j",
                   "L", "m3", "m1", "replay_m3", "after_flag", "partner_replay_m3", "partner_after_flag", "diag")
    diagnostics = {
        "negatives_M3_ge_0.85": {k: [{f: x.get(f) for f in diag_fields if f in x} for x in sets[k] if x["m3"] >= 0.85]
                                 for k in ("neg_a_real_diff", "neg_a_strict", "neg_b_wrong_full", "neg_b_wrong_win",
                                           "neg_c_xorg_full")},
        "positives_M3_lt_0.90": [{f: x.get(f) for f in diag_fields if f in x} for x in sets["pos_real_same"]
                                 if x["m3"] < MAIN_THR],
        "positives_lt_0.90_breakdown": (lambda v: {
            "n": len(v),
            "both_replays_pass": sum(1 for x in v if x["replay_m3"] >= MAIN_THR and x["partner_replay_m3"] >= MAIN_THR),
            "both_replays_pass_and_band5s_ge_0.90": sum(1 for x in v if x["replay_m3"] >= MAIN_THR
                                                        and x["partner_replay_m3"] >= MAIN_THR
                                                        and (x.get("diag") or {}).get("m3_band5s", 0) >= MAIN_THR),
            "a_side_replay_fails": sum(1 for x in v if min(x["replay_m3"], x["partner_replay_m3"]) < MAIN_THR),
            "any_side_after_flag": sum(1 for x in v if x["after_flag"] or x["partner_after_flag"]),
            "band5s_ge_0.90": sum(1 for x in v if (x.get("diag") or {}).get("m3_band5s", 0) >= MAIN_THR)})(
            [x for x in sets["pos_real_same"] if x["m3"] < MAIN_THR]),
        "call_flags": {f"{ci}": calls1[ci]["rec"]["flags"] for ci in
                       sorted({x["cid"] for k in ("neg_b_wrong_full", "pos_real_same") for x in sets[k]
                               if (x["m3"] >= 0.85 if k != "pos_real_same" else x["m3"] < MAIN_THR)})},
    }

    # 짝 비교 (같은 창 L): 재생 대 원래 vs 실제-같은 상태
    def paired(sub_pos):
        rows = []
        for x in sub_pos:
            rw = rep_win[(x["cid"], int(x["seg"].split(":")[1]))]
            if rw["m3"] is None:
                continue
            rows.append((x["cid"], rw, x))
        out = {}
        for mkey, label in (("m3", "M3"), ("m1", "M1")):
            rp = np.array([r[1][mkey] >= MAIN_THR for r in rows], float)
            pp = np.array([r[2][mkey] >= MAIN_THR for r in rows], float)
            cids = [r[0] for r in rows]
            o = {"n": len(rows), "both": int(((rp == 1) & (pp == 1)).sum()), "replay_only": int(((rp == 1) & (pp == 0)).sum()),
                 "real_only": int(((rp == 0) & (pp == 1)).sum()), "neither": int(((rp == 0) & (pp == 0)).sum()),
                 "replay_rate": rate(cids, rp), "real_rate": rate(cids, pp)}
            if rows:
                u = sorted(set(cids))
                pos = {c: i for i, c in enumerate(u)}
                ci = np.array([pos[c] for c in cids])
                D, N = np.bincount(ci, rp - pp, len(u)), np.bincount(ci, None, len(u)).astype(float)
                cnt = _boot_counts(len(u))
                dv = (cnt @ D) / (cnt @ N)
                o["diff_replay_minus_real"] = round(float((rp - pp).mean()), 4)
                o["diff_boot"] = [round(float(x), 4) for x in np.percentile(dv, [2.5, 97.5])]
                o["median_score_replay"] = round(float(np.median([r[1][mkey] for r in rows])), 4)
                o["median_score_real"] = round(float(np.median([r[2][mkey] for r in rows])), 4)
            out[label] = o
        return out

    paired_cmp = {"all": paired(pos_items),
                  "replay_not_self": paired([x for x in pos_items if not rep_win[(x["cid"], int(x["seg"].split(":")[1]))]["self_win"]]),
                  "nonstart": paired([x for x in pos_items if not x["start"]])}

    # 문구 같은 다른 상태 쌍 목록
    def st_list(items):
        g = defaultdict(list)
        for x in items:
            other = x.get("wrong_sid") or x.get("partner_first")
            g[(x["org"], x["first"], other)].append(x)
        rows = []
        for (o, f, w), v in sorted(g.items()):
            js = [x.get("text_j") for x in v if x.get("text_j") is not None]
            rows.append({"org": o, "state": f, "other_state": w, "text_j": max(js) if js else None, "n": len(v),
                         "m3_pass_0.90": sum(1 for x in v if x["m3"] >= MAIN_THR),
                         "m3_median": round(float(np.median([x["m3"] for x in v])), 4),
                         "m1_pass_0.90": sum(1 for x in v if x["m1"] >= MAIN_THR)})
        return rows

    counts = {
        "reference_segments": len(pl["ref"]), "positive_pairs_valid": len(pl["pos"]), "positive_skip": pl["pos_skip"],
        "positive_scored": len(pos_items), "positive_groups": len({tuple(x["group"]) for x in pl["pos"]}),
        "neg_a_chosen": len(pl["neg_a"]), "neg_a_strict_chosen": len(pl["neg_as"]), "sametext_real_chosen": len(pl["st_real"]),
        "plan_counts": pl["counts"], "neg_b_items": len(b_items), "neg_b_segments": len({x["seg"] for x in b_items}),
        "neg_b_distinct_pairs": len({x["pair"] for x in b_items}), "neg_b_wrong_clips": len({x["wrong_rel"] for x in b_items}),
        "neg_b_win_items": len(b_win), "neg_c_items": len(c_items),
        "neg_c_len_matched": sum(1 for x in c_items if x["len_matched"]), "sametext_synth_items": len(sts_items),
        "segments_excluded_from_insertion": excluded_insert,
    }
    result = {
        "module": "replay_validate_controls", "method": "docs/ARS자동화-가상ARS-측정방법.md 6.2-6.3, 6.8",
        "params": {"bundle": BUNDLE, "reference_check": REF_CHECK, "oracle": True,
                   "seeds": {"positive_pairs": SEED_PAIR, "neg_a": SEED_PAIR, "neg_a_strict": SEED_PAIR + 1,
                             "sametext_real": SEED_PAIR + 2, "neg_b_c_sametext_synth": f"'{SEED_PAIR}:<폴더 이름>'"}, "seed_bootstrap": SEED_BOOT, "n_bootstrap": N_BOOT,
                   "thresholds": THRESHOLDS, "main_threshold": MAIN_THR, "min_window_s": MIN_WIN_S,
                   "min_active_s": MIN_ACTIVE_S, "length_tolerance": LEN_TOL, "text_same_jaccard": TEXT_SAME,
                   "wrong_clips_per_segment": N_WRONG, "min_insert_window_s": MIN_INSERT_S, "org_min_n": ORG_MIN_N,
                   "limit": limit, "workers": workers},
        "reproduction": repro, "stage2_verify": verify, "counts": counts, "metrics": metrics, "paired": paired_cmp,
        "label_sensitivity": sensitivity, "diagnostics": diagnostics,
        "sametext_synth_pairs": st_list(sets["sametext_synth_full"]), "sametext_real_pairs": st_list(sets["sametext_real"]),
        "runtime_s": {"stage1": round(t1 - t_start, 1), "stage2": round(t2 - t1, 1), "total": round(time.time() - t_start, 1)},
    }
    Path(out_dir, "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    # 항목별 점수 (문장 없음; 통화는 번호, 번호 → 폴더 이름 MD5 앞 12자리)
    items = {"calls": [hashlib.md5(c["dir"].encode("utf-8")).hexdigest()[:12] for c in calls1],
             "sets": {k: [{kk: vv for kk, vv in x.items() if kk not in ("wrong_rel",)} for x in v] for k, v in sets.items()}}
    Path(out_dir, "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    M3, M1 = metrics["M3"], metrics["M1"]

    def t90(m, k):
        r = m["sets"][k]["thresholds"]["0.9"]
        return f"{r['k']}/{r['n']}"

    summary = {
        "reproduction": {k: repro[k] for k in ("calls", "segments", "segments_diff_gt_1e-4", "insample_all_M3",
                                               "insample_nonstart_M3", "same_path_rerun")},
        "n": {k: counts[k] for k in ("reference_segments", "positive_scored", "neg_a_chosen", "neg_b_items",
                                     "neg_b_distinct_pairs", "neg_b_segments", "neg_c_items", "sametext_synth_items",
                                     "sametext_real_chosen")},
        "M3_pass_0.90": {k: t90(M3, k) for k in sets},
        "M1_pass_0.90": {k: t90(M1, k) for k in sets},
        "M3_max_negative": {k: (v or {}).get("value") for k, v in M3["max_negative"].items()},
        "M1_max_negative": {k: (v or {}).get("value") for k, v in M1["max_negative"].items()},
        "M3_auc": {k: [v["auc"], v["boot"]] for k, v in M3["auc"].items()},
        "M3_label_sensitivity_0.90": {mode: {k: f"{v['M3']['k']}/{v['M3']['n']} max {v['M3']['max']}"
                                             for k, v in blk.items() if not k.startswith(("auc", "false_match"))}
                                      for mode, blk in sensitivity.items()},
        "M1_auc": {k: [v["auc"], v["boot"]] for k, v in M1["auc"].items()},
        "paired_M3": {k: {kk: v["M3"].get(kk) for kk in ("n", "both", "replay_only", "real_only", "neither",
                                                          "diff_replay_minus_real", "diff_boot")}
                      for k, v in paired_cmp.items()},
        "runtime_s": result["runtime_s"],
    }
    return summary


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(run(OUT_DIR if not lim else os.path.join(OUT_DIR, "_trial"), 6, lim), ensure_ascii=False, indent=1))
