"""
녹음 기반 가상 ARS 검증: 표본 내 검사와 조각 제외 5겹 교차 검증(검사할 통화의 녹음을 빼고 만든 묶음으로 그 통화를 검사)

  python replay_validate_kfold.py [--workers N] [--out <출력 폴더>]

측정 대상과 절차
  1. 대상 통화: replay_check.calls(전체 묶음) 가 내는 녹음 폴더 가운데 log.json 의 backend 가 clawops(또는 기록 없음)이고
     call.wav 가 8000표본(1초) 이상인 통화. replay_check.check_call 의 거르기와 같으며, 저장해 둔 표본 내 결과
     (DEV_CHECK)의 통화 집합과 같은지 확인.
  2. 표본 내: 전체 묶음(FULL_BUNDLE)으로 모든 대상 통화를 replay_check.check_call 로 검사. 변형 오라클(실제 ARS 가 스스로 고르는
     소리 변형을 원래 통화와 가장 닮은 것으로 선택)을 켠 것과 끈 것 두 조건. 오라클을 끈 조건의 변형 선택 씨앗은 check_call 과
     같이 폴더 이름의 md5 앞 8자리.
     같은 spec(SPEC_DIR)으로 묶음을 다시 만들어(제외 없음) 전체 묶음과 bundle.json·조각 바이트가 같은지 확인하고, 오라클을 켠
     결과가 DEV_CHECK 와 통화별 구간·경로·표시가 같은지 확인.
  3. 겹 나누기: 기관 번호(폴더 이름의 _번호_)를 정렬한 순서로, 기관마다 폴더 이름을 정렬한 뒤 random.Random(FOLD_SEED).shuffle
     로 섞고(기관마다 새 난수기), 0~4겹에 차례로 배정. 배정 번호는 기관을 넘어 이어 셈.
  4. 조각 제외 5겹 교차 검증: 겹 i 마다 replay_build.build(SPEC_DIR, OUT_DIR/bundle_fold{i}, replay_check.ROOTS,
     exclude=겹 i 의 폴더 이름) 로 묶음 생성. 그 묶음의 어떤 상태의 clip_from 도, 어떤 변형의 from 도 겹 i 의 폴더가 아님을
     확인(assert)한 뒤, 겹 i 의 통화만 오라클을 켠 것과 끈 것으로 검사. 묶음 생성은 조각뿐 아니라 응답 지연·인식 지연·
     조용한 기다림(silent_wait_s)·기관 시간 초과 상한도 겹 녹음 없이 다시 계산. 상태 그래프·키 연결·지연 가운뎃값·시작 규칙·
     상수는 표본 내 값 그대로.
  5. 교차 검증 구간 범주(겹 묶음 기준, 개인정보가 아니면서 조각이 없는 상태 집합 N): 구간의 활성 상태 = 구간 시작 때의 상태
     (state_at_start) ∪ 구간 안에서 들어간 상태(states_entered).
       (b) 활성 상태 중 하나 이상이 N 에 속함.
       (c) (b) 가 아니고, 구간 시작이 그 통화에서 N 의 상태에 처음 들어간 시각 이후.
       (a) 그 밖(활성 상태와 그 앞 경로의 모든 상태에 조각이 있음). (a) 는 활성 상태 중 하나 이상의 조각 출처 녹음이 전체
           묶음과 달라진 것(a_changed)과 그렇지 않은 것(a_unchanged)으로 다시 나눔.
     전체 묶음에서 조각이 있던 상태가 겹 묶음에서 조각을 모두 잃은 (겹, 상태) 수와 그 가운데 겹 통화가 지나간 수를 계산.
  6. 집계(조건마다, 전체와 기관별, 주 지표 M3 = MFCC DTW 코사인 유사도 ≥ 0.90, 참고 지표 M1 = 윤곽 DTW 상관 ≥ 0.9,
     정의는 replay_check):
       all(모든 구간), nonstart(비시작 구간), nonstart_not_self(비시작·자기 출처 아님), start(시작 구간),
       before_flag_nonstart(비시작·첫 미검증 표시 이전: after_flag 거짓),
       tester_strict_nonstart(before_flag_nonstart 에서 녹음 밖 종료·개인정보 종료로 잘린 경계 구간도 뺀 것),
       nonstart_excl_personal_boundary(비시작·개인정보 경계 구간 제외, 참고),
       conservative / conservative_nonstart(녹음 밖 종료 뒤 시작하는 구간 가운데 원래 녹음에 말소리가 있는 구간을 실패로
       분모에 더함; 개인정보 종료 통화는 더하지 않음),
       통화 결과(full = 녹음 끝 1초 전보다 먼저 끝나지 않음, offmap_early, personal_early, hangup_early = 녹음 밖 표시 없이
       일찍 끊김), 시작 시각 차(같은 소리로 판정된 구간의 onset_dt: 부호 있는 가운뎃값, 절댓값 가운뎃값·90번째 백분위
       = 정렬값[int(0.9n)], replay_check.summarize 와 같은 정의).
     비율마다 k/n, Wilson 95% 구간(replay_check.wilson), 통화 단위 군집 부트스트랩 95% 구간(그 비율에 구간을 하나 이상 낸 통화를
     복원 추출, BOOT_N 회, numpy default_rng(BOOT_SEED), 백분위 2.5·97.5)을 기록.
  7. 주 수치: 교차 검증, 오라클 끔, M3, 비시작 구간. 첫 표시·녹음 밖 종료 이전 구간만 본 값(tester_view_*)을 함께 산출.
     표본 내 수치는 상한으로 기록.
  8. 그 밖: 잘린 경계 구간의 뒷부분([정지 시각, 원래 구간 끝))에 원래 말소리가 있는 수(보수적 집계에도 들어가지 않는 부분),
     같은 구간의 표본 내 대 교차 검증 판정 전이, 겹별 주 수치.
  9. 회선 잡음 조각: 겹 묶음에 replay_build 가 회선 잡음 조각(noise)을 만든 기관이 있으면, 같은 방법으로 다시 만들어 출처가 겹 밖
     조각 출처 녹음임을 확인하고 소리 세기를 기록. 주 결과는 replay_build 가 만든 묶음 그대로의 값이며, 민감도로 그 (겹, 기관)
     통화를 잡음 조각을 뺀 사본 묶음(bundle_fold{i}_nonoise_<번호>)으로도 검사해 함께 기록.
  10. 민감도(주 수치 아님): 개인정보가 아니면서 clip 이 없는 상태 진입을 녹음 밖 종료로 처리하는 엔진(NoClipOffmapEngine)으로
      같은 겹 묶음을 다시 검사해 cvsens_noclip_offmap_oracle_{on,off} 로 기록. 교차 검증 녹음 밖 종료는 끝낸 이유 × 조각 잃은
      상태 경유 × 표본 내와 같은 이유인지 × 마지막 상태의 silent_wait_s 변화로 나눠 셈. 표시 없이 일찍 끊긴 통화(hangup_early)의
      정지 뒤 원래 말소리 구간 수도 기록(보수적 집계에는 들어가지 않음).

산출물(OUT_DIR): result.json(요약), calls_<조건>.json(통화별 기록: 구간 수치·경로·표시, 받아쓰기는 글자 수만),
folds.json, bundle_full/, bundle_fold0~4/. 받아쓰기 문장은 어디에도 기록하지 않음.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import replay_build as rb  # noqa: E402
import replay_check as rc  # noqa: E402
from audio import ulaw_to_pcm16  # noqa: E402
from replay_ars import ReplayBundle, ReplayEngine  # noqa: E402

SPEC_DIR = r"C:\Users\User\Documents\ars_run\replay\specs"
FULL_BUNDLE = r"C:\Users\User\Documents\ars_run\replay\bundle"
DEV_CHECK = r"C:\Users\User\Documents\ars_run\replay\check_v3_oracle.json"
OUT_DIR = r"C:\Users\User\Documents\ars_run\replay\validate\kfold"
FOLD_SEED, K = 20261109, 5
BOOT_N, BOOT_SEED = 1000, 7
# 통화 수가 적어 기관별 결과를 따로 내세우지 않는 기관 (결과에 claim_excluded_small_org 로 표시)
SMALL_ORGS = {"15882588", "15991004", "15776000", "15882188", "15880075"}
ADDED_KEYS = ("b_orig", "truncated", "tail_orig_sound", "tail_s")


# 대상·겹 -------------------------------------------------------------------------
def universe(bundle: ReplayBundle) -> list[dict]:
    """check_call 과 같은 거르기(backend, 1초 이상)를 통과한 통화 (calls() 순서)."""
    out = []
    for d, ph in rc.calls(None, bundle):
        try:
            j = json.loads((d / "log.json").read_text(encoding="utf-8"))
            with wave.open(str(d / "call.wav"), "rb") as w:
                n = w.getnframes() * w.getnchannels()
        except Exception:
            continue
        if j.get("backend") not in (None, "clawops") or n < 8000:
            continue
        out.append({"name": d.name, "path": str(d), "phone": ph})
    return out


def make_folds(univ: list[dict]) -> list[list[str]]:
    by: dict[str, list[str]] = {}
    for c in univ:
        by.setdefault(c["phone"], []).append(c["name"])
    folds: list[list[str]] = [[] for _ in range(K)]
    i = 0
    for ph in sorted(by):
        lst = sorted(by[ph])
        random.Random(FOLD_SEED).shuffle(lst)
        for name in lst:
            folds[i % K].append(name)
            i += 1
    return folds


# 작업자 -------------------------------------------------------------------------
_BUNDLES: dict[str, ReplayBundle] = {}


def _init_worker() -> None:
    logging.getLogger("ars.replay").setLevel(logging.ERROR)


def _bundle(path: str) -> ReplayBundle:
    if path not in _BUNDLES:
        _BUNDLES[path] = ReplayBundle(path)
    return _BUNDLES[path]


def _build_task(args):
    tag, out, exclude = args
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    t = time.time()
    rep = rb.build(SPEC_DIR, out, rc.ROOTS, set(exclude))
    return tag, rep, round(time.time() - t, 1)


class NoClipOffmapEngine(ReplayEngine):
    """민감도 전용 엔진: 개인정보가 아니면서 소리 조각(clip)이 없는 상태에 들어가면 녹음 밖 종료."""

    def _enter(self, state_id: str, via: str = "key") -> None:
        super()._enter(state_id, via)
        s = self.states.get(state_id) or {}
        if not self.ended and state_id in self.states and not s.get("personal") and not s.get("clip"):
            self._end("소리 조각이 없는 상태")


def _check_task(args):
    """check_call 결과에 잘린 경계 구간 정보(원래 구간 끝, 뒷부분 원래 말소리)를 추가."""
    cond, bundle_dir, dpath, oracle, fold = args[:5]
    engine_kind = args[5] if len(args) > 5 else None
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    b = _bundle(bundle_dir)
    d = Path(dpath)
    m = re.search(r"_(\d{3,})_", d.name)
    org = b.org_for(m.group(1))
    saved = rc.ReplayEngine
    if engine_kind == "noclip_offmap":
        rc.ReplayEngine = NoClipOffmapEngine
    try:
        r = rc.check_call(b, d, org, oracle)
    finally:
        rc.ReplayEngine = saved
    if r is None:
        return cond, d.name, None
    r["n_transcripts"] = len(r.pop("transcripts", []))
    r["fold"] = fold
    j, pcm = rc.load_call(d)
    keys = rc.key_events(j, pcm)
    end_s = len(pcm) / 8000
    marks = [0.0] + [t for t, *_ in keys if t < end_s] + [end_s]
    mask = rc.key_mask(keys)
    for s in r["segs"]:
        i = max(k for k, x in enumerate(marks) if x <= s["from"] + 0.006)
        b_orig = next((x for x in marks[i + 1:] if x > marks[i]), end_s)
        s["b_orig"] = round(b_orig, 2)
        s["truncated"] = b_orig - s["to"] > 0.011
        if s["truncated"]:
            s["tail_s"] = round(b_orig - s["to"], 2)
            s["tail_orig_sound"] = bool(rc.islands(pcm, s["to"], b_orig, mask))
    return cond, d.name, r


# 집계 -------------------------------------------------------------------------
def _boot(per_call: list[tuple[int, int]]) -> list[float] | None:
    if not per_call:
        return None
    ka = np.array([k for k, _ in per_call], dtype=np.float64)
    na = np.array([n for _, n in per_call], dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(per_call), size=(BOOT_N, len(per_call)))
    rates = ka[idx].sum(axis=1) / np.maximum(na[idx].sum(axis=1), 1e-12)
    return [round(float(np.percentile(rates, 2.5)), 4), round(float(np.percentile(rates, 97.5)), 4)]


def _rate(per_call: list[tuple[int, int]]) -> dict:
    per_call = [x for x in per_call if x[1] > 0]
    k, n = sum(x[0] for x in per_call), sum(x[1] for x in per_call)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson": rc.wilson(k, n),
            "boot": _boot(per_call), "calls": len(per_call)}


def _offmap_after(r: dict) -> int:
    if not r["offmap"] or r["personal"]:
        return 0
    return sum(1 for a in r["after_stop"] if a["orig_sound"])


def _stop_boundary(r: dict, s: dict) -> bool:
    """녹음 밖 종료·개인정보 종료로 잘린 경계 구간인지 여부."""
    return bool(r["offmap"]) and s.get("truncated", False)


SEG_FILTERS = {
    "all": lambda r, s: True,
    "nonstart": lambda r, s: not s["start"],
    "nonstart_not_self": lambda r, s: not s["start"] and not s["self_src"],
    "start": lambda r, s: s["start"],
    "before_flag_nonstart": lambda r, s: not s["start"] and not s["after_flag"],
    "tester_strict_nonstart": lambda r, s: not s["start"] and not s["after_flag"] and not _stop_boundary(r, s),
    "nonstart_excl_personal_boundary": lambda r, s: not s["start"] and not s["personal_boundary"],
}


def _onset(segs: list[dict], key: str) -> dict:
    dts = [s["onset_dt"] for s in segs if s[key] and s["onset_dt"] is not None]
    if not dts:
        return {"n": 0}
    ab = sorted(abs(x) for x in dts)
    return {"n": len(dts), "signed_median": round(statistics.median(dts), 3), "abs_median": round(statistics.median(ab), 3),
            "abs_p90": round(ab[int(len(ab) * 0.9)], 3)}


def summarize_block(recs: list[dict], extra_filters: dict | None = None) -> dict:
    filters = dict(SEG_FILTERS, **(extra_filters or {}))
    out: dict = {"calls": len(recs)}
    for metric, key in (("m3", "same"), ("m1", "same_m1")):
        blk = {}
        for name, f in filters.items():
            blk[name] = _rate([(sum(1 for s in r["segs"] if f(r, s) and s[key]), sum(1 for s in r["segs"] if f(r, s)))
                               for r in recs])
        for name, f in (("conservative", SEG_FILTERS["all"]), ("conservative_nonstart", SEG_FILTERS["nonstart"])):
            blk[name] = _rate([(sum(1 for s in r["segs"] if f(r, s) and s[key]),
                                sum(1 for s in r["segs"] if f(r, s)) + _offmap_after(r)) for r in recs])
        segs = [s for r in recs for s in r["segs"]]
        blk["onset_all"] = _onset(segs, key)
        blk["onset_nonstart"] = _onset([s for s in segs if not s["start"]], key)
        out[metric] = blk
    out["outcome"] = {
        "full": sum(1 for r in recs if not r["early"]),
        "offmap_early": sum(1 for r in recs if r["early"] and not r["personal"] and r["offmap"]),
        "personal_early": sum(1 for r in recs if r["early"] and r["personal"]),
        "hangup_early": sum(1 for r in recs if r["early"] and not r["personal"] and not r["offmap"]),
        "offmap_any": sum(1 for r in recs if r["offmap"] and not r["personal"]),
        "flagged": sum(1 for r in recs if r["flags"]),
    }
    out["outcome"]["full_rate"] = {"k": out["outcome"]["full"], "n": len(recs), "wilson": rc.wilson(out["outcome"]["full"], len(recs))}
    segs = [s for r in recs for s in r["segs"]]
    out["segments_detail"] = {
        "personal_boundary": sum(1 for s in segs if s["personal_boundary"]),
        "personal_boundary_same": sum(1 for s in segs if s["personal_boundary"] and s["same"]),
        "offmap_boundary": sum(1 for r in recs for s in r["segs"] if _stop_boundary(r, s) and not r["personal"]),
        "self_src": sum(1 for s in segs if s["self_src"]),
        "one_sided": sum(1 for s in segs if s["orig_sound"] != s["replay_sound"]),
        "offmap_after_with_orig_sound": sum(_offmap_after(r) for r in recs),
        "personal_after_with_orig_sound": sum(1 for r in recs if r["personal"] for a in r["after_stop"] if a["orig_sound"]),
        "truncated_tail_with_orig_sound": sum(1 for r in recs if r["offmap"] for s in r["segs"]
                                              if s.get("truncated") and s.get("tail_orig_sound")),
        # 녹음 밖 표시 없이 일찍 끊긴 통화(보수적 집계에 들어가지 않음)의 정지 뒤 원래 말소리 구간
        "hangup_early_after_with_orig_sound": sum(1 for r in recs if r["early"] and not r["offmap"]
                                                  for a in r["after_stop"] if a["orig_sound"]),
    }
    return out


def summarize_condition(recs: list[dict], extra_filters: dict | None = None) -> dict:
    orgs = sorted({r["org"] for r in recs})
    out = {"ALL": summarize_block(recs, extra_filters)}
    for o in orgs:
        blk = summarize_block([r for r in recs if r["org"] == o], extra_filters)
        blk["claim_excluded_small_org"] = o in SMALL_ORGS
        out[o] = blk
    return out


def _jsonable(x):
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=1), encoding="utf-8")


def _compare_bundles(a: Path, b: Path) -> dict:
    ja = json.loads((a / "bundle.json").read_text(encoding="utf-8"))
    jb = json.loads((b / "bundle.json").read_text(encoding="utf-8"))
    fa = {p.relative_to(a).as_posix() for p in a.rglob("*.ulaw")}
    fb = {p.relative_to(b).as_posix() for p in b.rglob("*.ulaw")}
    diff = sum(1 for x in fa & fb if (a / x).read_bytes() != (b / x).read_bytes())
    return {"bundle_json_identical": ja == jb, "clips": len(fa), "clips_rebuilt": len(fb), "clip_set_diff": len(fa ^ fb),
            "clip_byte_diffs": diff}


# 실행 -------------------------------------------------------------------------
def run(out_dir: str = OUT_DIR, workers: int = 6) -> dict:
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    full = ReplayBundle(FULL_BUNDLE)
    univ = universe(full)
    by_name = {c["name"]: c for c in univ}
    folds = make_folds(univ)
    fold_of = {n: i for i, f in enumerate(folds) for n in f}
    phones = sorted({c["phone"] for c in univ})
    _dump(out / "folds.json", {"seed": FOLD_SEED, "k": K, "sizes": [len(f) for f in folds],
                               "per_org": {ph: [sum(1 for n in f if by_name[n]["phone"] == ph) for f in folds] for ph in phones},
                               "folds": folds})

    # 묶음 생성: 제외 없는 재생성(전체 묶음과 같은지 확인용) + 겹 5개
    jobs = [("full", str(out / "bundle_full"), [])] + [(f"fold{i}", str(out / f"bundle_fold{i}"), folds[i]) for i in range(K)]
    with mp.Pool(min(workers, len(jobs)), initializer=_init_worker) as pool:
        built = {tag: (rep, sec) for tag, rep, sec in pool.imap_unordered(_build_task, jobs)}
    rebuild_check = _compare_bundles(Path(FULL_BUNDLE), out / "bundle_full")

    full_json = json.loads((Path(FULL_BUNDLE) / "bundle.json").read_text(encoding="utf-8"))
    fold_json = [json.loads((out / f"bundle_fold{i}" / "bundle.json").read_text(encoding="utf-8")) for i in range(K)]
    build_info = []
    for i in range(K):
        ex = set(folds[i])
        srcs = [(ph, sid, s.get("clip_from")) for ph, o in fold_json[i]["orgs"].items() for sid, s in o["states"].items()
                if s.get("clip")]
        srcs += [(ph, sid, v.get("from")) for ph, o in fold_json[i]["orgs"].items() for sid, s in o["states"].items()
                 for v in s.get("variants") or []]
        leaked = [x for x in srcs if x[2] in ex]
        assert not leaked, f"fold {i}: 겹 녹음에서 잘라 온 조각 {leaked[:3]}"
        assert all(x[2] in by_name for x in srcs), f"fold {i}: 대상 밖 출처"
        noise = []
        for ph, o in fold_json[i]["orgs"].items():
            if not o.get("noise"):
                continue
            # 회선 잡음 조각: replay_build 와 같은 방법(조각 출처 녹음 정렬 목록의 quiet_noise)으로 다시 만들어 바이트가 같으면 출처가
            # 조각 출처 녹음(겹 밖임을 위에서 확인)임. 소리 세기는 PCM16 RMS 와 10ms RMS > 100 비율
            used = sorted({rb.find_dir(v["from"], rc.ROOTS) for s in o["states"].values() if s.get("clip")
                           for v in (s.get("variants") or [{"from": s.get("clip_from")}])})
            raw = (out / f"bundle_fold{i}" / o["noise"]).read_bytes()
            assert rb.quiet_noise(used) == raw, f"fold {i} {ph}: 회선 잡음 출처 확인 실패"
            p = np.frombuffer(ulaw_to_pcm16(raw), dtype=np.int16).astype(np.float64)
            e = np.sqrt(np.mean(p[: len(p) // 80 * 80].reshape(-1, 80) ** 2, axis=1))
            noise.append({"org": ph, "source_is_clip_dirs": True, "clip_dirs": len(used),
                          "rms": round(float(np.sqrt(np.mean(p ** 2))), 1), "active_10ms_frac": round(float(np.mean(e > 100)), 3)})
        lost, changed, var_lost = [], [], []
        for ph, o in full_json["orgs"].items():
            for sid, s in o["states"].items():
                if s.get("personal") or not s.get("clip"):
                    continue
                fs = fold_json[i]["orgs"][ph]["states"][sid]
                if not fs.get("clip"):
                    lost.append([ph, sid])
                elif fs.get("clip_from") != s.get("clip_from"):
                    changed.append([ph, sid])
                if len(s.get("variants") or []) > len(fs.get("variants") or []) and fs.get("clip"):
                    var_lost.append([ph, sid])
        rep, sec = built[f"fold{i}"]
        no_clip = sorted([ph, sid] for ph, r in rep.items() if not ph.startswith("_") for sid in r["no_clip"])
        assert no_clip == sorted(lost), f"fold {i}: 보고서 no_clip 과 묶음의 조각 없는 상태가 다름"
        build_info.append({"fold": i, "build_sec": sec, "excluded_calls": len(ex), "clip_sources_checked": len(srcs),
                           "leak_assert": "통과", "lost_all_clips": lost, "clip_source_changed": changed,
                           "variants_reduced": var_lost, "noise_clips": noise, "latency_buckets": rep.get("_latency_buckets"),
                           "answer_delay_s": {ph: o.get("answer_delay_s") for ph, o in fold_json[i]["orgs"].items()},
                           "unseen_timeout_s": {ph: o.get("unseen_timeout_s") for ph, o in fold_json[i]["orgs"].items()}})

    # 검사
    tasks = []
    for c in univ:
        for oracle in (True, False):
            tag = "on" if oracle else "off"
            tasks.append((f"insample_oracle_{tag}", FULL_BUNDLE, c["path"], oracle, None))
            fi = fold_of[c["name"]]
            tasks.append((f"cv_oracle_{tag}", str(out / f"bundle_fold{fi}"), c["path"], oracle, fi))
            # 민감도: 조각 없는 상태 진입을 녹음 밖 종료로 처리하는 엔진으로 같은 겹 검사
            tasks.append((f"cvsens_noclip_offmap_oracle_{tag}", str(out / f"bundle_fold{fi}"), c["path"], oracle, fi,
                          "noclip_offmap"))
    # 민감도: 회선 잡음 조각이 생긴 (겹, 기관) 은 잡음을 뺀 사본 묶음으로도 그 겹 그 기관 통화를 검사
    noisy = [(bi["fold"], x["org"]) for bi in build_info for x in bi["noise_clips"]]
    for fi, ph in noisy:
        src, dst = out / f"bundle_fold{fi}", out / f"bundle_fold{fi}_nonoise_{ph}"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        bj = json.loads((dst / "bundle.json").read_text(encoding="utf-8"))
        (dst / bj["orgs"][ph]["noise"]).unlink()
        bj["orgs"][ph]["noise"] = None
        (dst / "bundle.json").write_text(json.dumps(bj, ensure_ascii=False, indent=1), encoding="utf-8")
        for c in univ:
            if fold_of[c["name"]] == fi and c["phone"] == ph:
                for oracle in (True, False):
                    tag = "on" if oracle else "off"
                    tasks.append((f"sens_nonoise_oracle_{tag}", str(dst), c["path"], oracle, fi))
                    tasks.append((f"sens_noise_oracle_{tag}", str(src), c["path"], oracle, fi))
    tasks.sort(key=lambda t: -os.path.getsize(Path(t[2]) / "call.wav"))
    res: dict[str, dict[str, dict]] = {}
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for cond, name, r in pool.imap_unordered(_check_task, tasks, chunksize=2):
            if r is not None:
                res.setdefault(cond, {})[name] = r
    order = [c["name"] for c in univ]
    recs = {cond: [d[n] for n in order if n in d] for cond, d in res.items()}
    sens = {cond: recs.pop(cond) for cond in list(recs) if cond.startswith("sens_")}
    for cond, lst in recs.items():
        assert len(lst) == len(univ), f"{cond}: {len(lst)} != {len(univ)}"
    noisy_set = set(noisy)
    for cond in ("cv_oracle_on", "cv_oracle_off"):
        for r in recs[cond]:
            r["fold_bundle_noise"] = (r["fold"], r["org"]) in noisy_set
    sensitivity = {"noise_fold_org": [list(x) for x in noisy], "calls": {}}
    for cond, lst in sens.items():
        sensitivity["calls"][cond] = [{"dir": r["dir"], "segs": len(r["segs"]), "same_m3": sum(s["same"] for s in r["segs"]),
                                       "same_m3_nonstart": sum(s["same"] for s in r["segs"] if not s["start"]),
                                       "nonstart": sum(1 for s in r["segs"] if not s["start"]),
                                       "one_sided": sum(1 for s in r["segs"] if s["orig_sound"] != s["replay_sound"]),
                                       "m3": [s["m3"] for s in r["segs"]], "early": r["early"], "offmap": r["offmap"]}
                                      for r in lst]

    # 교차 검증 범주
    none_states = [{ph: {sid for sid, s in o["states"].items() if not s.get("clip") and not s.get("personal")}
                    for ph, o in fj["orgs"].items()} for fj in fold_json]
    changed_states = [{(ph, sid) for ph, sid in bi["clip_source_changed"]} | {(ph, sid) for ph, sid in bi["variants_reduced"]}
                      for bi in build_info]
    visited_lost: dict[str, set] = {}
    cv_conds = ("cv_oracle_on", "cv_oracle_off", "cvsens_noclip_offmap_oracle_on", "cvsens_noclip_offmap_oracle_off")
    for cond in cv_conds:
        vl = set()
        for r in recs[cond]:
            fi, ph = r["fold"], r["org"]
            N = none_states[fi][ph]
            first_none = min((p["t"] for p in r["path"] if p.get("state") in N), default=None)
            r["first_none_t"] = first_none
            vl |= {(fi, ph, p["state"]) for p in r["path"] if p.get("state") in N}
            for s in r["segs"]:
                act = ([s["state_at_start"]] if s["state_at_start"] else []) + list(s["states_entered"])
                if any(x in N for x in act):
                    cat = "b"
                elif first_none is not None and s["from"] >= first_none:
                    cat = "c"
                else:
                    cat = "a"
                s["cv_cat"] = cat
                s["cv_changed"] = cat == "a" and any((ph, x) in changed_states[fi] for x in act)
        visited_lost[cond] = vl
    cat_filters = {}
    for base in ("all", "nonstart", "before_flag_nonstart", "tester_strict_nonstart"):
        bf = SEG_FILTERS[base]
        for cat in ("a", "b", "c"):
            cat_filters[f"{base}__cat_{cat}"] = (lambda bf, cat: lambda r, s: bf(r, s) and s["cv_cat"] == cat)(bf, cat)
        cat_filters[f"{base}__cat_a_changed"] = (lambda bf: lambda r, s: bf(r, s) and s["cv_cat"] == "a" and s["cv_changed"])(bf)
        cat_filters[f"{base}__cat_a_unchanged"] = (lambda bf: lambda r, s: bf(r, s) and s["cv_cat"] == "a"
                                                   and not s["cv_changed"])(bf)

    # 저장해 둔 표본 내 결과(DEV_CHECK) 재현 확인
    dev = {c["dir"]: c for c in json.loads(Path(DEV_CHECK).read_text(encoding="utf-8"))["calls"]}
    dev_diff = []
    for r in recs["insample_oracle_on"]:
        dv = dev.get(r["dir"])
        segs = [{k: v for k, v in s.items() if k not in ADDED_KEYS} for s in r["segs"]]
        if dv is None or dv["segs"] != segs or dv["path"] != r["path"] or dv["flags"] != r["flags"] or dv["offmap"] != r["offmap"]:
            dev_diff.append(r["dir"])
    dev_repro = {"calls_compared": len(recs["insample_oracle_on"]), "dev_calls": len(dev), "calls_differing": len(dev_diff),
                 "differing": dev_diff[:20], "universe_equals_dev_set": set(order) == set(dev)}

    # 조건별 집계
    conditions = {}
    for cond in ("insample_oracle_on", "insample_oracle_off") + cv_conds:
        conditions[cond] = summarize_condition(recs[cond], cat_filters if cond.startswith("cv") else None)

    # 교차 검증 녹음 밖 종료의 원인 분해: 끝낸 이유 × (조각 잃은 상태를 지났는지) × (표본 내와 같은 이유인지) ×
    # (마지막 상태의 silent_wait_s 가 전체 묶음과 다른지)
    full_states = {ph: o["states"] for ph, o in full_json["orgs"].items()}
    lost_set = {(bi["fold"], ph, sid) for bi in build_info for ph, sid in bi["lost_all_clips"]}
    offmap_causes = {}
    for tag in ("on", "off"):
        ins = {r["dir"]: r for r in recs[f"insample_oracle_{tag}"]}
        cnt: dict[str, int] = {}
        for r in recs[f"cv_oracle_{tag}"]:
            if not r["offmap"] or r["personal"]:
                continue
            states = [p["state"] for p in r["path"] if "state" in p]
            last = states[-1] if states else None
            sw_f = fold_json[r["fold"]]["orgs"][r["org"]]["states"].get(last, {}).get("silent_wait_s")
            sw_a = full_states[r["org"]].get(last, {}).get("silent_wait_s")
            key = "|".join([re.sub(r"'.*'", "'키'", r["offmap"]),
                            "조각 잃은 상태 지남" if any((r["fold"], r["org"], x) in lost_set for x in states) else "조각 잃은 상태 안 지남",
                            "표본 내와 같은 이유" if ins[r["dir"]]["offmap"] == r["offmap"] else "교차 검증에서 새로 생김",
                            "silent_wait 다름" if sw_f != sw_a else "silent_wait 같음"])
            cnt[key] = cnt.get(key, 0) + 1
        offmap_causes[tag] = dict(sorted(cnt.items()))

    # 겹별 주 수치, 표본 내 대 교차 검증 판정 전이
    per_fold = []
    for i in range(K):
        rr = [r for r in recs["cv_oracle_off"] if r["fold"] == i]
        blk = summarize_block(rr)
        per_fold.append({"fold": i, "calls": len(rr), "nonstart_m3": blk["m3"]["nonstart"],
                         "before_flag_nonstart_m3": blk["m3"]["before_flag_nonstart"], "outcome": blk["outcome"],
                         "lost_all_clips": len(build_info[i]["lost_all_clips"])})
    transitions = {}
    for tag in ("on", "off"):
        ins = {(r["dir"], s["from"]): s for r in recs[f"insample_oracle_{tag}"] for s in r["segs"]}
        cvs = {(r["dir"], s["from"]): s for r in recs[f"cv_oracle_{tag}"] for s in r["segs"]}
        both = set(ins) & set(cvs)
        tr = {"paired": len(both), "same_same": 0, "same_to_diff": 0, "diff_to_same": 0, "diff_diff": 0,
              "insample_only": len(set(ins) - set(cvs)), "cv_only": len(set(cvs) - set(ins))}
        for key in both:
            a, b = ins[key]["same"], cvs[key]["same"]
            tr["same_same" if a and b else "same_to_diff" if a else "diff_to_same" if b else "diff_diff"] += 1
        tr["same_to_diff_by_cat"] = {c: sum(1 for key in both if ins[key]["same"] and not cvs[key]["same"]
                                            and cvs[key]["cv_cat"] == c) for c in ("a", "b", "c")}
        transitions[tag] = tr

    lost_total = sum(len(b["lost_all_clips"]) for b in build_info)
    cv_off = conditions["cv_oracle_off"]["ALL"]
    headline = {
        "definition": "조각 제외 5겹 교차 검증, 변형 오라클 끔, M3 ≥ 0.90, 비시작 구간",
        "nonstart_m3": cv_off["m3"]["nonstart"],
        "tester_view_before_flag_nonstart_m3": cv_off["m3"]["before_flag_nonstart"],
        "tester_view_strict_nonstart_m3": cv_off["m3"]["tester_strict_nonstart"],
        "conservative_nonstart_m3": cv_off["m3"]["conservative_nonstart"],
        "nonstart_m1_reference": cv_off["m1"]["nonstart"],
        "outcome": cv_off["outcome"],
        "upper_bound_insample_oracle_off_nonstart_m3": conditions["insample_oracle_off"]["ALL"]["m3"]["nonstart"],
        "upper_bound_insample_oracle_on_nonstart_m3": conditions["insample_oracle_on"]["ALL"]["m3"]["nonstart"],
    }
    sv = conditions["cvsens_noclip_offmap_oracle_off"]["ALL"]
    headline["sensitivity_noclip_state_as_offmap"] = {
        "note": "민감도(주 수치 아님): 조각 없는 상태 진입을 녹음 밖 종료로 처리하는 엔진으로 같은 겹 묶음 검사",
        "nonstart_m3": sv["m3"]["nonstart"], "before_flag_nonstart_m3": sv["m3"]["before_flag_nonstart"],
        "tester_strict_nonstart_m3": sv["m3"]["tester_strict_nonstart"],
        "conservative_nonstart_m3": sv["m3"]["conservative_nonstart"], "outcome": sv["outcome"]}
    result = {
        "module": "replay_validate_kfold",
        "method_sections": ["6.1", "6.4 표 1~3행", "6.8"],
        "parameters": {"fold_seed": FOLD_SEED, "k": K, "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "rng": "numpy default_rng",
                                                                      "cluster": "통화"},
                       "same_m3": rc.SAME_M3, "same_m1": rc.SAME_M1, "min_seg_s": rc.MIN_SEG_S,
                       "mask_s": [rc.MASK_BEFORE_S, rc.MASK_AFTER_S], "spec_dir": SPEC_DIR, "full_bundle": FULL_BUNDLE},
        "universe": {"calls": len(univ), "candidates_from_calls()": sum(1 for _ in rc.calls(None, full)),
                     "per_org": {ph: sum(1 for c in univ if c["phone"] == ph) for ph in phones}},
        "folds": {"sizes": [len(f) for f in folds],
                  "per_org": {ph: [sum(1 for n in f if by_name[n]["phone"] == ph) for f in folds] for ph in phones}},
        "sanity": {"full_rebuild_vs_current_bundle": rebuild_check, "dev_reproduction": dev_repro,
                   "cv_self_src_segments": sum(1 for c in ("cv_oracle_on", "cv_oracle_off") for r in recs[c]
                                               for s in r["segs"] if s["self_src"])},
        "builds": build_info,
        "lost_clips": {"fold_state_pairs_lost_all_clips": lost_total,
                       "states_with_clip_in_full_bundle": sum(1 for o in full_json["orgs"].values()
                                                              for s in o["states"].values() if s.get("clip")),
                       "per_fold": [len(b["lost_all_clips"]) for b in build_info],
                       "per_org": {ph: sum(1 for b in build_info for x in b["lost_all_clips"] if x[0] == ph) for ph in phones},
                       "visited_by_fold_calls_oracle_on": len(visited_lost["cv_oracle_on"]),
                       "visited_by_fold_calls_oracle_off": len(visited_lost["cv_oracle_off"]),
                       "fold_state_pairs_clip_source_changed": sum(len(b["clip_source_changed"]) for b in build_info),
                       "fold_state_pairs_variants_reduced": sum(len(b["variants_reduced"]) for b in build_info)},
        "headline": headline,
        "per_fold_cv_oracle_off": per_fold,
        "transitions_insample_to_cv": transitions,
        "sensitivity_fold_noise_clip": sensitivity,
        "cv_offmap_causes": offmap_causes,
        "conditions": conditions,
        "elapsed_s": None,
    }
    for cond, lst in recs.items():
        _dump(out / f"calls_{cond}.json", lst)
    result["elapsed_s"] = round(time.time() - t_start, 1)
    _dump(out / "result.json", result)

    def short(x):
        return f"{x['k']}/{x['n']}={x['rate']} W{x['wilson']} B{x['boot']}"

    summary = {
        "headline_cv_off_nonstart_m3": short(headline["nonstart_m3"]),
        "tester_view_before_flag": short(headline["tester_view_before_flag_nonstart_m3"]),
        "tester_view_strict": short(headline["tester_view_strict_nonstart_m3"]),
        "conservative_nonstart": short(headline["conservative_nonstart_m3"]),
        "cv_on_nonstart_m3": short(conditions["cv_oracle_on"]["ALL"]["m3"]["nonstart"]),
        "insample_on_nonstart_m3": short(conditions["insample_oracle_on"]["ALL"]["m3"]["nonstart"]),
        "insample_off_nonstart_m3": short(conditions["insample_oracle_off"]["ALL"]["m3"]["nonstart"]),
        "insample_on_all_m3": short(conditions["insample_oracle_on"]["ALL"]["m3"]["all"]),
        "sens_noclip_offmap_nonstart": short(sv["m3"]["nonstart"]),
        "sens_noclip_offmap_tester_strict": short(sv["m3"]["tester_strict_nonstart"]),
        "sens_noclip_offmap_conservative_nonstart": short(sv["m3"]["conservative_nonstart"]),
        "lost_all_clips_pairs": lost_total,
        "dev_reproduction_calls_differing": dev_repro["calls_differing"],
        "full_rebuild_identical": rebuild_check["bundle_json_identical"] and rebuild_check["clip_byte_diffs"] == 0,
        "elapsed_s": result["elapsed_s"],
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    print(json.dumps(run(a.out, a.workers), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
