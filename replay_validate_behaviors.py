"""
녹음 기반 가상 ARS 검증 — 동작별 일치율, 시험 결과 사용 규칙 평가, 실시간 경로 동등성, 묶음 근거 정적 표

  python replay_validate_behaviors.py [<출력 폴더>] [--workers 6]
  기본 출력: DEFAULT_OUT/result.json

실제 통화·음성 인식(Whisper)·LLM(Ollama)·서버는 쓰지 않음. 결과에는 받아쓰기 문장과 녹음 폴더 이름을 옮기지 않음
(통화별 행은 같은 폴더의 calls_local.json 에만 기록).

1. 동작별 검증
  - 대상과 입력: replay_check.calls 가 고르는 녹음 통화 전부를 replay_check.check_call(전체 묶음, oracle 켬 = 변형은 원래 통화와
    가장 닮은 것으로 선택)로 검사. 키 시각(keys_sent t + 0.25i + 0.15초), 구간(0.6초 미만 제외, 키 톤 가림, 양쪽 무음 제외,
    한쪽만 말소리면 0점), M3(주 지표: MFCC DTW 코사인 유사도, 같은 소리 = M3 ≥ 0.90), M1(참고 지표: 로그 에너지 윤곽 DTW 상관,
    ≥ 0.9), 표지(시작·자기 출처·개인정보 경계·표시 뒤)는 replay_check 의 것을 그대로 사용.
  - 계측: 엔진을 동작이 같은 하위 클래스로 바꾸어, 키를 받는 순간의 상황(엔진이 말하는 중인지, 그 상태의 barge_in, 다음 상태 진입을
    기다리는 중인지, 두 자리 키를 잇는 중인지)과 두 자리 이상 키의 적용을 path 에 별도 항목(_keyinfo, _multikey)으로 추가.
    이 항목은 check_call 의 판정에 쓰이지 않음. 추가 항목을 뺀 path·구간·표시가 replay_check.py --out 으로 저장한
    검사 결과(DEV_RUN, 계측 없는 엔진)와 통화마다 같은지 대조.
  - 사건의 구간 배정: path 사건(상태 진입·키·표시·종료·계측 항목)의 시각 t 가 구간 [from, to) 에 들면 그 구간에 배정. 사건이
    0.6초 미만이라 빠진 구간에 들면 바로 다음의 남은 구간(이어진 짧은 구간은 건너뜀)에 배정. 양쪽 무음으로 빠진 구간, 정지 뒤
    구간, 마지막 짧은 구간에 든 사건은 어느 구간에도 배정하지 않고 '배정 안 됨' 으로 집계.
  - 동작 표지(구간당 있음·없음):
      start(시작 구간), entry_<via>(구간 안 상태 진입 원인: key·timeout·third_fail·input·digit_timeout·invalid),
      timeout_self(같은 상태로의 시간 초과 재진입), two_digit_key(두 자리 이상 키 적용),
      barge_<menu|input>_<barge_in>(엔진이 말하는 중에 받았고 미리 누른 키로 버려지지 않은 키; 합계 barge_any·barge_yes·
      barge_unknown), flag_<이름>(미검증 표시 종류별), end_hangup(녹음에 있는 ARS 끊음), end_offmap_any·end_offmap_nonpersonal·
      end_<사유>(녹음 밖 종료 사유별: personal·unknown_key·unseen_timeout·repeat_limit·input_next_missing),
      personal_boundary(check_call 표지), variant_entered(변형 상태 진입), after_flag·self_src(check_call 표지, 참고),
      plain_nonstart(예외 표지가 하나도 없는 비시작 구간, 기준선).
  - 보고: 표지별 M3·M1 같은 소리 k/n, Wilson 95% 구간(replay_check.wilson), 통화 단위 군집 부트스트랩 95% 구간(통화를 복원
    추출 1,000회, 씨앗 7, 2.5·97.5 백분위, 비율 = Σk/Σn). n < 10 은 근거 미약(weak). 전체 구간과 비시작 구간을 따로, 기관별로도 산출.
2. 시험 결과 사용 규칙
  - 통화마다 path 에서 가장 이른 제외 사건(첫 미검증 표시, 또는 사유가 있는 녹음 밖 종료; 같은 시각이면 path 순서)을 찾음.
    그 사건 시각보다 끝이 늦은 구간(표시는 check_call 의 after_flag 와 같음)을 제외.
  - 보고: 남은 구간 비율(전체·비시작), 남은 통화 비율(제외 사건이 없는 통화; 구간이 하나도 빠지지 않은 통화도 함께),
    남은·뺀 구간의 M3·M1 일치율(Wilson·군집 부트스트랩), 첫 제외 사유별 통화 수(개인정보 종료는 따로).
    정지 뒤 구간(after_stop) 가운데 원래 녹음에 말소리가 있는 수를 녹음 밖·개인정보별로 함께 기록.
3. 실시간 경로
  - ReplayCall 을 가짜 CallHandlers 로 실시간 구동(asyncio, 서버·전화 없음). 서버(ReplayBackend.start)처럼 detect_dtmf 를
    미리 호출. RT_CASES 의 녹음 통화 두 건(그중 한 건은 민감정보 입력 포함, 각각 최대 run_s 초) × 받아쓰기 모드 켬·끔, 변형 씨앗 7.
    키는 첫 프레임 벽시계 + 녹음의 keys_sent 시각에 call.send_keys 로 전송(민감정보는 replay_check 처럼 '0' 과 '#').
  - 기록: 프레임마다 SHA-1·벽시계(perf_counter), 엔진이 key·keys_done 을 받은 프레임·벽시계, 받아쓰기(시작 시각, 글자 수, 해시만).
  - 비교: 같은 시각·씨앗의 오프라인 ReplayEngine 에 기록된 프레임에서 같은 순서로 key·keys_done 을 넣어 프레임·path·받아쓰기
    (받아쓰기 모드에서 _meaningful 거르기 뒤, 끝난 뒤 flush 포함)의 일치 여부 확인. 키 수신 오차 = (엔진이 받은 벽시계 − 전송
    시작) − (0.25i + 0.15초), keys_done 오차 = (벽시계 − 전송 시작) − 톤 전체 길이, 키 프레임과 replay_check 규약
    (첫 프레임 f: 0.02f ≥ 전송 시각 + 0.25i + 0.15)의 차, 프레임 간격 편차 |tick_i − tick_0 − 0.02i| 와 마지막 어긋남.
    프레임·path·받아쓰기 일치는 실행마다 같아야 하는 비교이고, 시각 수치는 실행마다 다름.
4. 정적 표: 묶음 bundle.json 의 상태·연결 수를 근거 종류별로 집계(barge_in 값, after_audio, third_fail, max_plays,
   silent_wait_s, 입력 상태의 input_next·자리 시간 초과·값 의존, 키 연결의 관측 수·지연 측정, invalid_key 다음 상태).
5. 결함 정량(보고용, 주 결과에 쓰지 않음)
   (a) replay_check.key_events 가 민감정보 입력의 '#' 덧붙임을 판정할 때 1kHz 대체 신호음을 세는 창(t − 0.02 ~ t + 0.25(n+3))이
   다음 keys_sent 전송의 신호음까지 세는 결함. 이 모듈 안에서만 창 끝을 다음 전송 시작 − 0.02초로 자른 판으로 바꾸어 같은 검사를
   재실행(BUNDLE_DIR 의 현 묶음 그대로, 그리고 같은 판으로 다시 만든 묶음). 대조로 수정하지 않은 판으로 다시 만든 묶음이 현 묶음과 같은지도
   확인. 묶음은 출력 폴더 아래에만 생성.
   (b) 자기 반복 상한(DEFAULT_MAX_PLAYS)이 세 번째 무입력 규칙(third_fail_next)보다 먼저 걸리는 결함. 상한에 닿을 차례이면 그
   규칙의 상태로 가는 하위 클래스 엔진으로 재실행해 경로가 바뀌는 통화를 집계.
   (c) check_call 이 엔진 정지 구간을 [시작, 정지) 로 자르므로, 정지 뒤 [정지, 구간 끝) 의 원래 말소리는 비교에도 after_stop 집계
   에도 들어가지 않음. 그 부분에 원래 말소리(replay_check.islands)가 있는 통화를 종료 사유별로 세고, 녹음 밖(개인정보 제외) 몫을
   보수적 비율의 분모에 더한 값을 함께 산출.

모든 무작위는 씨앗으로 고정(부트스트랩 7, 실시간 변형 7, check_call 은 폴더 이름 기반 씨앗).
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import hashlib
import itertools
import json
import logging
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import replay_ars  # noqa: E402
import replay_build  # noqa: E402
import replay_check as rc  # noqa: E402
from audio import detect_dtmf, dtmf_ulaw  # noqa: E402
from replay_ars import (FRAME_S, KEY_TONE_ON_S, KEY_TONE_S, ReplayBundle, ReplayCall, ReplayEngine,  # noqa: E402
                        _meaningful)
from telephony import CallHandlers  # noqa: E402

BUNDLE_DIR = r"C:\Users\User\Documents\ars_run\replay\bundle"
SPEC_DIR = r"C:\Users\User\Documents\ars_run\replay\specs"
DEV_RUN = r"C:\Users\User\Documents\ars_run\replay\check_v3_oracle.json"
DEFAULT_OUT = r"C:\Users\User\Documents\ars_run\replay\validate\behaviors"
BOOT_N, BOOT_SEED, WEAK_N = 1000, 7, 10
RT_SEED = 7
RT_CASES = [
    {"name": "kb", "org": "15889999", "stamp": "20261106-144320", "run_s": 62.0},
    {"name": "nhis", "org": "15771000", "stamp": "20261105-101024", "run_s": 78.0},
]
FLAG_KINDS = ["late_key", "type_ahead", "barge_unverified", "unmeasured_delay", "unmeasured_digit_timeout", "value_dependent"]
END_KINDS = ["personal", "unknown_key", "unseen_timeout", "repeat_limit", "input_next_missing", "unknown_state", "other"]
REQUIRED_LABELS = (["start", "entry_key", "entry_timeout", "entry_third_fail", "entry_input", "entry_digit_timeout",
                    "entry_invalid", "timeout_self", "two_digit_key", "barge_any", "barge_yes", "barge_unknown",
                    "barge_menu_yes", "barge_menu_unknown", "barge_input_yes", "barge_input_unknown"]
                   + [f"flag_{f}" for f in FLAG_KINDS]
                   + ["end_hangup", "end_offmap_any", "end_offmap_nonpersonal"] + [f"end_{k}" for k in END_KINDS[:5]]
                   + ["personal_boundary", "variant_entered", "after_flag", "self_src", "plain_nonstart"])
EXC_PREFIX = ("entry_timeout", "entry_third_fail", "entry_input", "entry_digit_timeout", "entry_invalid", "timeout_self",
              "two_digit_key", "barge_", "flag_", "end_", "personal_boundary", "variant_entered")
EXTRA_KEYS = ("_keyinfo", "_multikey")


# 계측 엔진 ----------------------------------------------------------------------
class _InstEngine(ReplayEngine):
    """동작이 같은 계측판. 키를 받는 순간의 상황과 두 자리 이상 키의 적용을 path 에 별도 항목으로 추가
    ('state'·'flag'·'key'·'end' 칸을 쓰지 않으므로 check_call 의 판정에 영향 없음)."""

    def key(self, ch: str) -> None:
        if not self.ended and self.state_id is not None:
            s = self.states[self.state_id]
            keys = s.get("keys") or {}
            continuing = bool(self._keybuf) and any(k.startswith(self._keybuf + ch) for k in keys)
            self.path.append({"t": self.t, "_keyinfo": {
                "in_state": self.state_id, "input": bool(s.get("input")), "speaking": bool(self.speaking),
                "barge_in": s.get("barge_in"), "pending_enter": bool(self._pending_enter), "continuing": continuing}})
        super().key(ch)

    def _apply_key(self, ch: str, barged: bool = False) -> None:
        if len(ch) >= 2:
            self.path.append({"t": self.t, "_multikey": len(ch)})
        super()._apply_key(ch, barged)


class _ThirdFailPatched(_InstEngine):
    """결함 정량용(주 결과에 쓰지 않음): 자기 반복 상한에 닿을 차례인 상태에 세 번째 무입력 규칙이 있으면 녹음 밖 종료 대신 그
    규칙의 상태로 이동 (replay_ars._audio_done 과 같고 third_fail 조건 한 줄만 다름)."""

    def _audio_done(self) -> None:
        self.speaking = False
        self._audio_done_frame = self.frame
        if self._pending_key is not None:
            key, self._pending_key = self._pending_key, None
            self._apply_key(key)
            return
        self._waiting_after = True
        st = self.states[self.state_id]
        if st.get("input") and self.digits:
            return
        after = st.get("after_audio") or {}
        nxt, wait, via = after.get("next"), after.get("timeout_s"), "timeout"
        limit = int(after.get("max_plays") or replay_ars.DEFAULT_MAX_PLAYS)
        if after.get("third_fail_next") and (self._timeouts >= 2 or (nxt == self.state_id and self._plays >= limit)):
            nxt, wait, via = after["third_fail_next"], after.get("third_fail_timeout_s") or wait, "third_fail"
        if nxt:
            self._schedule(self._delay(wait, 5.0), "timeout", nxt, via=via)
        else:
            seen = st.get("silent_wait_s")
            self._schedule(max(self._unseen_s, float(seen) + replay_ars.SILENT_MARGIN_S) if seen else self._unseen_s,
                           "unseen")


def _strip(path: list[dict]) -> list[dict]:
    return [p for p in path if not any(k in p for k in EXTRA_KEYS)]


def _key_events_fixed(j: dict, pcm=None) -> list[tuple[float, str, bool, float]]:
    """replay_check.key_events 와 같되, '#' 덧붙임을 판정하는 대체 신호음 세기 창이 다음 keys_sent 전송 시작 0.02초 전에서 끝남
    (결함 정량용, 주 결과에는 쓰지 않음)."""
    evs = [e for e in j.get("events", []) if e.get("type") == "keys_sent"]
    starts = sorted(float(e["t"]) for e in evs)
    out = []
    for e in evs:
        t = float(e["t"])
        nxt = next((s for s in starts if s > t), math.inf)
        if e.get("kind") == "input" or e.get("keys") is None:
            n = int(e.get("digit_count") or 0)
            tones = e.get("tone_count")
            if tones is None and pcm is not None:
                tones = rc.tone_count(pcm, t - 0.02, min(t + KEY_TONE_S * (n + 3), nxt - 0.02))
            seq = "0" * n + ("#" if (tones == n + 1 or (n == 0 and (tones or 0) >= 1)) else "")
            seq = seq or "0"
            secure = True
        else:
            seq, secure = str(e["keys"]), False
        done = t + len(dtmf_ulaw(seq)) / 8000
        for i, ch in enumerate(seq):
            out.append((t + i * KEY_TONE_S + rc.TONE_ON_S, ch, secure, done))
    return sorted(out)


# 표지 ---------------------------------------------------------------------------
def _end_kind(note: str | None) -> str:
    if note is None:
        return "hangup"
    if "개인정보" in note:
        return "personal"
    if note.startswith("녹음에 없는 키"):
        return "unknown_key"
    if "입력이 없을 때의 동작" in note:
        return "unseen_timeout"
    if note.startswith("같은 안내가"):
        return "repeat_limit"
    if "입력 뒤 상태" in note:
        return "input_next_missing"
    if note.startswith("모르는 상태"):
        return "unknown_state"
    return "other"


def _stop_tail(r: dict, keys: list, end_s: float, pcm) -> dict | None:
    """엔진이 멈춘 구간(시작 < 정지 < 끝, 0.6초 이상)에서 정지 뒤 [정지, 구간 끝) 의 원래 말소리. check_call 은 이 부분을 비교하지도
    after_stop 으로 세지도 않음(정지 시각 = path 의 end 사건 + 1프레임)."""
    ends = [p for p in r["path"] if "end" in p]
    if not ends:
        return None
    stop = round(ends[0]["t"] + FRAME_S, 2)
    marks = [0.0] + [t for t, *_ in keys if t < end_s] + [end_s]
    mask = rc.key_mask(keys)
    for a, b in zip(marks, marks[1:]):
        if a < stop < b and b - a >= rc.MIN_SEG_S:
            isl = rc.islands(pcm, stop, b, mask)
            head_kept = any(abs(s["from"] - round(a, 2)) < 1e-6 for s in r["segs"])
            return {"kind": _end_kind(r["offmap"]), "stop": stop, "seg_to": round(b, 2), "tail_s": round(b - stop, 2),
                    "orig_sound": bool(isl), "orig_speech_s": round(sum(e - s for s, e, _ in isl), 2),
                    "head_segment_scored": head_kept, "stop_vs_replay_end_ok": abs(round(stop, 1) - r["replay_end_s"]) < 0.051}
    return {"kind": _end_kind(r["offmap"]), "stop": stop, "seg_to": None, "tail_s": 0.0, "orig_sound": False,
            "orig_speech_s": 0.0, "head_segment_scored": None,
            "stop_vs_replay_end_ok": abs(round(stop, 1) - r["replay_end_s"]) < 0.051}


def _label_call(r: dict, keys: list, end_s: float) -> dict:
    """구간마다 동작 표지, 사건 배정 수, 시험 결과 사용 규칙의 제외 판정."""
    segs = r["segs"]
    kept_from = {s["from"]: i for i, s in enumerate(segs)}
    after_from = {a["from"] for a in r["after_stop"]}
    marks = [0.0] + [t for t, *_ in keys if t < end_s] + [end_s]
    raw = []  # (a, b, 상태, 구간 번호)
    for a, b in zip(marks, marks[1:]):
        ra = round(a, 2)
        if b - a < rc.MIN_SEG_S:
            raw.append((a, b, "short", None))
        elif ra in after_from:
            raw.append((a, b, "after_stop", None))
        elif ra in kept_from:
            raw.append((a, b, "kept", kept_from[ra]))
        else:
            raw.append((a, b, "nosound", None))
    starts = [x[0] for x in raw]

    def seg_of(t: float):
        if not raw:
            return None, "none"
        i = max(0, min(len(raw) - 1, bisect.bisect_right(starts, t + 1e-6) - 1))
        while raw[i][2] == "short" and i + 1 < len(raw):
            i += 1
        return (raw[i][3], "kept") if raw[i][2] == "kept" else (None, raw[i][2])

    labels: list[set[str]] = [set() for _ in segs]
    events = defaultdict(lambda: Counter())  # 표지 → {assigned, unassigned_<사유>}

    def put(t, names):
        si, where = seg_of(t)
        for nm in names:
            if si is None:
                events[nm][f"unassigned_{where}"] += 1
            else:
                events[nm]["assigned"] += 1
                labels[si].add(nm)

    prev_state = None
    first_excl = None
    note_end_t = None
    for p in r["path"]:
        t = p["t"]
        if "state" in p:
            names = [f"entry_{p['via']}"]
            if p["via"] == "timeout" and p["state"] == prev_state:
                names.append("timeout_self")
            if "variant" in p:
                names.append("variant_entered")
            put(t, names)
            prev_state = p["state"]
        elif "flag" in p:
            put(t, [f"flag_{p['flag']}"])
            if first_excl is None:
                first_excl = {"kind": "flag", "reason": f"flag_{p['flag']}", "t": t}
        elif "end" in p:
            k = _end_kind(r["offmap"])
            names = ["end_hangup"] if k == "hangup" else ["end_offmap_any", f"end_{k}"] + (
                ["end_offmap_nonpersonal"] if k != "personal" else [])
            put(t, names)
            if k != "hangup":
                note_end_t = t
                if first_excl is None:
                    first_excl = {"kind": "offmap", "reason": f"offmap_{k}", "t": t}
        elif "_keyinfo" in p:
            ki = p["_keyinfo"]
            if ki["speaking"] and not (ki["pending_enter"] and not ki["continuing"]):
                kind = "input" if ki["input"] else "menu"
                bi = ki["barge_in"] if ki["barge_in"] in ("yes", "no", "unknown") else "none"
                put(t, ["barge_any", f"barge_{bi}", f"barge_{kind}_{bi}"])
        elif "_multikey" in p:
            put(t, ["two_digit_key"])
    for i, s in enumerate(segs):
        if s["start"]:
            labels[i].add("start")
        if s["personal_boundary"]:
            labels[i].add("personal_boundary")
        if s["after_flag"]:
            labels[i].add("after_flag")
        if s["self_src"]:
            labels[i].add("self_src")
        if not s["start"] and not any(lb.startswith(EXC_PREFIX) for lb in labels[i]):
            labels[i].add("plain_nonstart")
    dropped = [bool(s["after_flag"]) or (note_end_t is not None and s["to"] > note_end_t + 1e-9) for s in segs]
    return {"labels": [sorted(x) for x in labels], "events": {k: dict(v) for k, v in events.items()},
            "first_excl": first_excl, "dropped": dropped, "raw_status": dict(Counter(x[2] for x in raw))}


# 병렬 검사 ----------------------------------------------------------------------
_W: dict = {}


def _winit(bundle_dir: str, keyfix: bool, engine: str = "inst") -> None:
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    rc.ReplayEngine = _ThirdFailPatched if engine == "thirdfail" else _InstEngine
    if keyfix:
        rc.key_events = _key_events_fixed
    _W["bundle"] = ReplayBundle(bundle_dir)


def _wcall(arg):
    d, ph = arg
    b = _W["bundle"]
    r = rc.check_call(b, Path(d), b.org_for(ph), True)
    if r is None:
        return None
    j, pcm = rc.load_call(Path(d))
    keys = rc.key_events(j, pcm)
    end_s = len(pcm) / 8000
    lab = _label_call(r, keys, end_s)
    lab["stop_tail"] = _stop_tail(r, keys, end_s, pcm)
    r.pop("transcripts", None)
    return {"rec": r, "lab": lab}


def _check_all(bundle_dir: str, targets: list, workers: int, keyfix: bool, engine: str = "inst") -> list[dict]:
    with Pool(workers, initializer=_winit, initargs=(bundle_dir, keyfix, engine)) as pool:
        res = pool.map(_wcall, targets, chunksize=4)
    return [x for x in res if x is not None]


# 비율 ---------------------------------------------------------------------------
def _boot(groups: list[tuple[int, int]]):
    if not groups:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    k = np.array([g[0] for g in groups], float)
    n = np.array([g[1] for g in groups], float)
    idx = rng.integers(0, len(groups), size=(BOOT_N, len(groups)))
    r = k[idx].sum(1) / n[idx].sum(1)
    return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]


def _rate(items: list[tuple[int, dict]], key: str, boot: bool = True) -> dict:
    """items = [(통화 번호, 구간)]. 구간 단위 비율 + 통화 군집 부트스트랩."""
    k = sum(1 for _, s in items if s[key])
    n = len(items)
    out = {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson": rc.wilson(k, n)}
    if boot:
        per = defaultdict(lambda: [0, 0])
        for ci, s in items:
            per[ci][0] += bool(s[key])
            per[ci][1] += 1
        out["boot"] = _boot([tuple(v) for v in per.values()])
    return out


def _stats(items: list[tuple[int, dict]], org_of: dict, org_names: dict) -> dict:
    st = {"m3": _rate(items, "same"), "m1": _rate(items, "same_m1"), "calls": len({ci for ci, _ in items}),
          "weak": len(items) < WEAK_N, "by_org": {}}
    by = defaultdict(list)
    for ci, s in items:
        by[org_of[ci]].append((ci, s))
    for org in sorted(by):
        it = by[org]
        m3 = _rate(it, "same", boot=len(it) >= WEAK_N)
        m1 = _rate(it, "same_m1", boot=False)
        st["by_org"][org] = {"org": org_names.get(org), "n": len(it), "m3_k": m3["k"], "m3_rate": m3["rate"],
                             "m3_wilson": m3["wilson"], "m3_boot": m3.get("boot"), "m1_k": m1["k"],
                             "m1_rate": m1["rate"], "weak": len(it) < WEAK_N}
    return st


# 1·2. 동작별 일치율과 시험 규칙 ----------------------------------------------------
def _behaviors(res: list[dict], org_names: dict) -> dict:
    org_of = {i: x["rec"]["org"] for i, x in enumerate(res)}
    rows = [(i, s, set(x["lab"]["labels"][j])) for i, x in enumerate(res) for j, s in enumerate(x["rec"]["segs"])]
    seen = sorted({lb for _, _, ls in rows for lb in ls} | {k for x in res for k in x["lab"]["events"]})
    order = REQUIRED_LABELS + [lb for lb in seen if lb not in REQUIRED_LABELS]
    ev_tot = defaultdict(Counter)
    for x in res:
        for k, v in x["lab"]["events"].items():
            ev_tot[k].update(v)
    out = {}
    for lb in order:
        items = [(i, s) for i, s, ls in rows if lb in ls]
        ns = [(i, s) for i, s in items if not s["start"]]
        out[lb] = {"all": _stats(items, org_of, org_names), "nonstart": _stats(ns, org_of, org_names),
                   "events": dict(ev_tot.get(lb, {}))}
    return out


def _tester_rule(res: list[dict], org_names: dict) -> dict:
    org_of = {i: x["rec"]["org"] for i, x in enumerate(res)}
    kept, dropped = [], []
    reasons, reasons_org = Counter(), defaultdict(Counter)
    for i, x in enumerate(res):
        for s, d in zip(x["rec"]["segs"], x["lab"]["dropped"]):
            (dropped if d else kept).append((i, s))
        fe = x["lab"]["first_excl"]
        reasons[fe["reason"] if fe else "none"] += 1
        reasons_org[x["rec"]["org"]][fe["reason"] if fe else "none"] += 1
    n_calls = len(res)
    no_excl = sum(1 for x in res if x["lab"]["first_excl"] is None)
    all_kept = sum(1 for x in res if not any(x["lab"]["dropped"]))
    some_kept_ns = sum(1 for x in res if any(not d and not s["start"] for s, d in zip(x["rec"]["segs"], x["lab"]["dropped"])))
    n_segs = len(kept) + len(dropped)
    ns_k = [(i, s) for i, s in kept if not s["start"]]
    ns_d = [(i, s) for i, s in dropped if not s["start"]]
    after = Counter()
    for x in res:
        r = x["rec"]
        k = "personal" if r["personal"] else ("offmap" if r["offmap"] else "other")
        for a in r["after_stop"]:
            after[f"{k}_{'orig_sound' if a['orig_sound'] else 'no_sound'}"] += 1
    by_org = {}
    for org in sorted({org_of[i] for i in org_of}):
        ck = [(i, s) for i, s in kept if org_of[i] == org]
        cd = [(i, s) for i, s in dropped if org_of[i] == org]
        calls_o = [x for x in res if x["rec"]["org"] == org]
        by_org[org] = {"org": org_names.get(org), "calls": len(calls_o),
                       "calls_no_exclusion": sum(1 for x in calls_o if x["lab"]["first_excl"] is None),
                       "segments": len(ck) + len(cd), "kept": len(ck),
                       "kept_m3": _rate(ck, "same", boot=len(ck) >= WEAK_N), "dropped_m3": _rate(cd, "same", boot=False),
                       "weak": (len(ck) + len(cd)) < WEAK_N, "first_exclusion": dict(reasons_org[org])}
    return {
        "rule": "통화마다 path 의 첫 제외 사건(첫 미검증 표시 또는 사유가 있는 녹음 밖 종료, 같은 시각이면 path 순서) 시각보다 "
                "끝이 늦은 구간을 뺀다 (표시는 check_call after_flag 와 같음).",
        "calls": n_calls,
        "calls_no_exclusion_event": {"k": no_excl, "n": n_calls, "rate": round(no_excl / n_calls, 4), "wilson": rc.wilson(no_excl, n_calls)},
        "calls_all_segments_kept": {"k": all_kept, "n": n_calls, "rate": round(all_kept / n_calls, 4), "wilson": rc.wilson(all_kept, n_calls)},
        "calls_with_kept_nonstart_segment": some_kept_ns,
        "segments_kept_share": {"k": len(kept), "n": n_segs, "rate": round(len(kept) / n_segs, 4), "wilson": rc.wilson(len(kept), n_segs)},
        "nonstart_kept_share": {"k": len(ns_k), "n": len(ns_k) + len(ns_d),
                                "rate": round(len(ns_k) / max(1, len(ns_k) + len(ns_d)), 4),
                                "wilson": rc.wilson(len(ns_k), len(ns_k) + len(ns_d))},
        "kept_m3": _rate(kept, "same"), "dropped_m3": _rate(dropped, "same"),
        "kept_m1": _rate(kept, "same_m1"), "dropped_m1": _rate(dropped, "same_m1"),
        "kept_nonstart_m3": _rate(ns_k, "same"), "dropped_nonstart_m3": _rate(ns_d, "same"),
        "kept_nonstart_m1": _rate(ns_k, "same_m1"),
        "first_exclusion_reason_calls": dict(reasons.most_common()),
        "after_stop_segments": dict(after),
        "by_org": by_org,
    }


def _verify_dev(res: list[dict]) -> dict:
    """저장된 replay_check 검사 결과(DEV_RUN)와 통화별 대조: 계측 항목을 뺀 path, 구간, 표시, 녹음 밖 사유."""
    try:
        dev = json.loads(Path(DEV_RUN).read_text(encoding="utf-8"))
    except OSError:
        return {"available": False}
    by = {c["dir"]: c for c in dev["calls"]}
    diff = Counter()
    for x in res:
        r = x["rec"]
        c = by.get(r["dir"])
        if c is None:
            diff["missing_in_dev"] += 1
            continue
        if _strip(r["path"]) != c["path"]:
            diff["path"] += 1
        if r["segs"] != c["segs"]:
            diff["segs"] += 1
        if r["flags"] != c["flags"] or r["offmap"] != c["offmap"]:
            diff["flags_or_offmap"] += 1
        if r["after_stop"] != c["after_stop"]:
            diff["after_stop"] += 1
    return {"available": True, "calls_now": len(res), "calls_dev": len(dev["calls"]), "differences": dict(diff),
            "dev_summary_all": {k: dev["summary"]["ALL"][k] for k in ("segments", "nonstart", "start")}}


# 3. 실시간 경로 -----------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _find_dir(stamp: str, org: str) -> Path:
    for root in rc.ROOTS:
        for d in sorted(Path(root).glob(f"*_{org}_*")):
            if d.name.endswith(stamp) and (d / "log.json").exists():
                return d
    raise FileNotFoundError(stamp)


def _schedule(case: dict):
    """녹음된 통화의 시각과 키 전송 일정 [{send_t, keys, secure}] (run_s 3초 전까지 끝나는 전송만). 민감정보 입력은
    replay_check.key_events 처럼 '0'·'#'."""
    d = _find_dir(case["stamp"], case["org"])
    j, pcm = rc.load_call(d)
    groups = defaultdict(list)
    for t, ch, sec, done in rc.key_events(j, pcm):
        groups[done].append((t, ch, sec))
    sends = []
    for done, items in sorted(groups.items()):
        items.sort()
        t0 = round(items[0][0] - KEY_TONE_ON_S, 4)
        if done <= case["run_s"] - 3.0:
            sends.append({"send_t": t0, "keys": "".join(c for _, c, _ in items), "secure": any(s for *_, s in items)})
    return rc.call_clock(d.name), sends


async def _rt_once(org: str, clock, run_s: float, sends: list[dict], transcripts: bool, seed: int) -> dict:
    bundle = ReplayBundle(BUNDLE_DIR)
    detect_dtmf(dtmf_ulaw("1"))  # 서버(ReplayBackend.start)처럼 numpy 를 미리 로드
    rec = {"ticks": [], "audio": [], "tr": [], "ev": [], "sends": [], "end": None, "answer": None}
    holder: dict = {}
    seq = itertools.count()

    async def on_answer():
        rec["answer"] = time.perf_counter()

    async def on_audio(frame):
        rec["audio"].append(_sha(frame))

    async def on_transcript(text, started_at=None, forced=False, taken=None):
        rec["tr"].append({"wall": time.perf_counter(), "rel": started_at - holder["call"]._t0, "len": len(text),
                          "h": _sha(text.encode("utf-8")), "forced": bool(forced)})

    async def on_end(reason):
        rec["end"] = reason

    call = ReplayCall(bundle, org, CallHandlers(on_answer, on_audio, on_transcript, on_end), clock=clock,
                      transcripts=transcripts, seed=seed)
    holder["call"] = call
    eng = call.eng
    o_tick, o_key, o_done = eng.tick, eng.key, eng.keys_done

    def tick():
        rec["ticks"].append(time.perf_counter())
        return o_tick()

    def key(ch):
        rec["ev"].append({"kind": "key", "frame": eng.frame, "wall": time.perf_counter(), "ch": ch, "seq": next(seq)})
        return o_key(ch)

    def keys_done():
        rec["ev"].append({"kind": "done", "frame": eng.frame, "wall": time.perf_counter(), "seq": next(seq)})
        return o_done()

    eng.tick, eng.key, eng.keys_done = tick, key, keys_done
    rec["t_dial"] = time.perf_counter()
    call._spawn(call.answer())

    async def sender(i, s):
        while not rec["ticks"]:
            await asyncio.sleep(0.001)
        tick0 = rec["ticks"][0]
        await asyncio.sleep(max(0.0, tick0 + s["send_t"] - time.perf_counter()))
        row = {"i": i, "send_t": s["send_t"], "keys_n": len(s["keys"]), "secure": s["secure"],
               "wall_start": time.perf_counter(), "n_ticks_before": len(rec["ticks"]), "wall_ret": None, "ret": None,
               "dur_expected": len(dtmf_ulaw(s["keys"])) / 8000, "keys": s["keys"]}
        rec["sends"].append(row)
        row["ret"] = await call.send_keys(s["keys"])
        row["wall_ret"] = time.perf_counter()

    tasks = [asyncio.create_task(sender(i, s)) for i, s in enumerate(sends)]
    while not rec["ticks"]:
        await asyncio.sleep(0.005)
    tick0 = rec["ticks"][0]
    while time.perf_counter() < tick0 + run_s and not call.ended:
        await asyncio.sleep(0.02)
    rec["ended_by"] = "remote" if call.ended else "local"
    if not call.ended:
        await call.hangup()
    for t in tasks:
        t.cancel()
    await asyncio.sleep(0.2)
    rec["flush"] = [{"rel": s - call._t0, "h": _sha(x.encode("utf-8")), "forced": bool(f)}
                    for x, s, f in call.flush_transcripts()]
    rec["path"] = list(eng.path)
    rec["note"] = eng.note
    return rec


def _offline(org: str, clock, seed: int, n_ticks: int, events: list[dict]):
    eng = ReplayEngine(ReplayBundle(BUNDLE_DIR), org, clock=clock, seed=seed)
    evs = sorted(events, key=lambda e: e["seq"])
    out, trs, ei = [], [], 0

    def apply_until(f):
        nonlocal ei
        while ei < len(evs) and evs[ei]["frame"] <= f:
            e = evs[ei]
            ei += 1
            if e["kind"] == "key":
                eng.key(e["ch"])
            else:
                eng.keys_done()

    for f in range(n_ticks):
        apply_until(f)
        fr = eng.tick()
        if eng.transcripts:
            trs += [{"rel": s, "h": _sha(x.encode("utf-8")), "forced": bool(fo), "frame": eng.frame}
                    for x, s, fo in eng.transcripts if _meaningful(x)]
            eng.transcripts = []
        if eng.ended:
            break
        out.append(_sha(fr))
    else:
        apply_until(n_ticks)
    flush = [{"rel": s, "h": _sha(x.encode("utf-8")), "forced": bool(fo)} for x, s, fo in eng.flush() if _meaningful(x)]
    return eng, out, trs, flush


def _pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 5) if xs else None


def _tr_equal(a: list[dict], b: list[dict]) -> bool:
    ka = sorted((round(x["rel"], 6), x["h"], x["forced"]) for x in a)
    kb = sorted((round(x["rel"], 6), x["h"], x["forced"]) for x in b)
    return len(ka) == len(kb) and all(x[1] == y[1] and x[2] == y[2] and abs(x[0] - y[0]) < 1e-6 for x, y in zip(ka, kb))


def _rt_analyze(case: dict, clock, sends: list[dict], rec: dict, transcripts: bool) -> dict:
    ticks = rec["ticks"]
    tick0 = ticks[0]
    eng2, off, off_tr, off_flush = _offline(case["org"], clock, RT_SEED, len(ticks), rec["ev"])
    rt = rec["audio"]
    n = min(len(rt), len(off))
    mism = [i for i in range(n) if rt[i] != off[i]]
    key_rows, done_rows = [], []
    kev = [e for e in rec["ev"] if e["kind"] == "key"]
    dev = [e for e in rec["ev"] if e["kind"] == "done"]
    for s in sorted(rec["sends"], key=lambda x: x["wall_start"]):
        wr = s["wall_ret"] if s["wall_ret"] is not None else math.inf
        mine = [e for e in kev if s["wall_start"] <= e["wall"] <= wr + 1e-6]
        start_rel = s["wall_start"] - tick0
        for i, e in enumerate(mine):
            exp = KEY_TONE_ON_S + KEY_TONE_S * i
            key_rows.append({
                "send": s["i"], "digit": i, "err_s": round((e["wall"] - s["wall_start"]) - exp, 4),
                "frame_diff_vs_check_sched": e["frame"] - math.ceil(round((s["send_t"] + exp) / FRAME_S, 6)),
                "frame_diff_vs_check_actual_start": e["frame"] - math.ceil(round((start_rel + exp) / FRAME_S, 6)),
                # 녹음기가 전송 시각을 그때까지 받은 프레임 수로 적었다고 볼 때의 replay_check 키 프레임과의 차
                "frame_diff_vs_check_framecount": e["frame"] - math.ceil(round((s["n_ticks_before"] * FRAME_S + exp) / FRAME_S, 6))})
        dn = [e for e in dev if s["wall_start"] <= e["wall"] <= wr + 1e-6]
        done_rows.append({"send": s["i"], "keys_n": s["keys_n"], "secure": s["secure"],
                          "start_late_s": round(start_rel - s["send_t"], 4),
                          "keys_done_err_s": round(dn[0]["wall"] - s["wall_start"] - s["dur_expected"], 4) if dn else None,
                          "return_after_s": round(wr - s["wall_start"], 4) if s["wall_ret"] is not None else None,
                          "digits_received": len(mine), "return_value": s["ret"]})
    jit = [(w - tick0) - i * FRAME_S for i, w in enumerate(ticks)]
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    errs = [r["err_s"] for r in key_rows]
    out = {
        "case": case["name"], "org": case["org"], "source_call_stamp": case["stamp"],
        "clock": clock.isoformat(timespec="seconds"), "transcripts_mode": transcripts, "seed": RT_SEED,
        "run_s": case["run_s"], "sends": [{"send_t": s["send_t"], "keys_n": len(s["keys"]), "secure": s["secure"]} for s in sends],
        "ended_by": rec["ended_by"], "end_reason": rec["end"],
        "frames_rt": len(rt), "frames_offline": len(off), "ticks": len(ticks),
        "frame_mismatch": len(mism), "first_mismatch": mism[:5], "frames_equal": not mism and len(rt) == len(off),
        "path_equal": _strip(rec["path"]) == eng2.path, "path_len": len(rec["path"]),
        "path_states": [p["state"] for p in rec["path"] if "state" in p],
        "path_vias": [p["via"] for p in rec["path"] if "state" in p],
        "path_flags": [p["flag"] for p in rec["path"] if "flag" in p],
        "note_rt": rec["note"], "note_offline": eng2.note,
        "key_events_rt": len(kev), "keys_done_events_rt": len(dev),
        "transcripts": {
            "rt_delivered": len(rec["tr"]), "offline_emitted_meaningful": len(off_tr),
            "equal": _tr_equal(rec["tr"], off_tr) if transcripts else (len(rec["tr"]) == 0),
            "flush_rt": len(rec["flush"]), "flush_offline": len(off_flush),
            "flush_equal": _tr_equal(rec["flush"], off_flush) if transcripts else (len(rec["flush"]) == 0),
            "started_at_rel_s": [round(x["rel"], 3) for x in sorted(rec["tr"], key=lambda y: y["rel"])],
            "delivery_lag_s": ([round((a["wall"] - tick0) - b["frame"] * FRAME_S, 4)
                                for a, b in zip(sorted(rec["tr"], key=lambda y: y["wall"]), off_tr)] if transcripts else []),
            "note": "받아쓰기 모드가 꺼져 있으면 on_transcript 가 한 번도 불리지 않아야 같은 것으로 본다(엔진 내부 단락은 버려진다).",
        },
        "keys": key_rows, "sends_timing": done_rows,
        "key_err_s": {"n": len(errs), "min": min(errs) if errs else None, "max": max(errs) if errs else None,
                      "median": round(statistics.median(errs), 4) if errs else None},
        "key_frame_diff_vs_check_actual_start": dict(Counter(r["frame_diff_vs_check_actual_start"] for r in key_rows)),
        "key_frame_diff_vs_check_sched": dict(Counter(r["frame_diff_vs_check_sched"] for r in key_rows)),
        "key_frame_diff_vs_check_framecount": dict(Counter(r["frame_diff_vs_check_framecount"] for r in key_rows)),
        "pacing": {"jitter_abs_p50_s": _pct([abs(x) for x in jit], 0.5), "jitter_abs_p99_s": _pct([abs(x) for x in jit], 0.99),
                   "jitter_abs_max_s": round(max(abs(x) for x in jit), 5), "final_drift_s": round(jit[-1], 5),
                   "gap_min_s": round(min(gaps), 5), "gap_max_s": round(max(gaps), 5),
                   "gaps_gt_30ms": sum(1 for g in gaps if g > 0.03), "gaps_lt_5ms": sum(1 for g in gaps if g < 0.005),
                   "dial_to_first_tick_s": round(tick0 - rec["t_dial"], 4)},
    }
    return out


def _realtime() -> dict:
    runs = []
    for case in RT_CASES:
        clock, sends = _schedule(case)
        for tr in (True, False):
            rec = asyncio.run(_rt_once(case["org"], clock, case["run_s"], sends, tr, RT_SEED))
            runs.append(_rt_analyze(case, clock, sends, rec, tr))
    errs = [k["err_s"] for r in runs for k in r["keys"]]
    return {
        "runs": runs,
        "totals": {
            "runs": len(runs), "frames_compared": sum(min(r["frames_rt"], r["frames_offline"]) for r in runs),
            "frame_mismatch": sum(r["frame_mismatch"] for r in runs),
            "frames_equal_runs": sum(r["frames_equal"] for r in runs), "path_equal_runs": sum(r["path_equal"] for r in runs),
            "transcripts_equal_runs": sum(r["transcripts"]["equal"] for r in runs),
            "transcripts_mode_on_paragraphs": sum(r["transcripts"]["rt_delivered"] for r in runs if r["transcripts_mode"]),
            "flush_equal_runs": sum(r["transcripts"]["flush_equal"] for r in runs),
            "keys": len(errs), "key_err_s_min": min(errs) if errs else None, "key_err_s_max": max(errs) if errs else None,
            "keys_within_one_frame": sum(1 for e in errs if abs(e) <= FRAME_S),
            "key_frame_diff_vs_check_actual_start": dict(sum((Counter(r["key_frame_diff_vs_check_actual_start"]) for r in runs), Counter())),
            "key_frame_diff_vs_check_framecount": dict(sum((Counter(r["key_frame_diff_vs_check_framecount"]) for r in runs), Counter())),
            "jitter_abs_p99_s": [r["pacing"]["jitter_abs_p99_s"] for r in runs],
            "final_drift_s": [r["pacing"]["final_drift_s"] for r in runs],
            "monotonic_resolution_s": time.get_clock_info("monotonic").resolution,
            "nondeterministic": "시각 수치(키 오차·프레임 간격)는 실행마다 다르다. 프레임·path·받아쓰기 일치만 결정적 비교이다.",
        },
    }


# 4. 정적 표 ---------------------------------------------------------------------
def _measured(d) -> bool:
    return isinstance(d, dict) and d.get("median") is not None


def _static_table(bundle_dir: str) -> dict:
    meta = json.loads((Path(bundle_dir) / "bundle.json").read_text(encoding="utf-8"))
    tot, per, orgc = Counter(), {}, {}
    for org, o in meta["orgs"].items():
        c = Counter()
        sts = o["states"]
        for sid, s in sts.items():
            pers = bool(s.get("personal"))
            bi = s.get("barge_in")
            keys = s.get("keys") or {}
            inp = s.get("input")
            after = s.get("after_audio") or {}
            c["states"] += 1
            c["states_personal"] += pers
            c["states_with_clip"] += bool(s.get("clip"))
            c["states_with_variants"] += bool(s.get("variants"))
            c[f"barge_in_{bi}"] += 1
            if keys and not pers:
                c[f"menu_states_with_keys_barge_in_{bi}"] += 1
            if inp:
                c[f"input_states_barge_in_{bi}"] += 1
            if after.get("next"):
                c["after_audio_states"] += 1
                c["after_audio_states_personal"] += pers
                c["after_audio_timeout_measured"] += _measured(after.get("timeout_s"))
                if after["next"] == "hangup":
                    c["after_audio_to_hangup"] += 1
                elif after["next"] == sid:
                    c["after_audio_self_loop"] += 1
                    c["self_loop_max_plays_explicit"] += after.get("max_plays") is not None
                elif after["next"] not in sts:
                    c["after_audio_to_unknown_state"] += 1
                else:
                    c["after_audio_to_other_state"] += 1
                if after.get("third_fail_next"):
                    c["third_fail_states"] += 1
                    c["third_fail_timeout_measured"] += _measured(after.get("third_fail_timeout_s"))
            elif not pers:
                c["nonpersonal_without_after_audio"] += 1
                if s.get("silent_wait_s") is not None:
                    c["silent_wait_states"] += 1
                else:
                    c["no_after_audio_no_silent_wait"] += 1
            if inp:
                c["input_states"] += 1
                c["input_with_input_next"] += bool(s.get("input_next"))
                c["input_without_input_next"] += not s.get("input_next")
                c["input_delay_measured"] += _measured(s.get("input_delay_s"))
                c["input_digit_timeout_measured"] += inp.get("digit_timeout_s") is not None
                c["input_value_dependent"] += bool(s.get("input_value_dependent"))
                c["input_with_terminator"] += bool(inp.get("terminator"))
                c["input_fixed_len"] += isinstance(inp.get("len"), int)
                c["input_with_after_audio"] += bool(after.get("next"))
                # 일부만 입력했을 때: 자리 시간 초과 측정 + 무입력 연결 / 무입력 연결만(미측정 표시) / 둘 다 없음(녹음 밖 종료)
                if after.get("next") and inp.get("digit_timeout_s") is not None:
                    c["input_partial_measured_timeout"] += 1
                elif after.get("next"):
                    c["input_partial_unmeasured_timeout"] += 1
                else:
                    c["input_partial_unseen_end"] += 1
            for k, e in keys.items():
                n = e.get("n")
                c["key_edges"] += 1
                c["key_edges_from_personal"] += pers
                c["key_edges_n1"] += n == 1
                c["key_edges_n2_4"] += isinstance(n, int) and 2 <= n <= 4
                c["key_edges_n5plus"] += isinstance(n, int) and n >= 5
                c["key_edges_n_missing"] += not isinstance(n, int)
                c["key_edges_delay_measured"] += _measured(e.get("delay_s"))
                c["key_edges_barge_delay_measured"] += _measured(e.get("barge_delay_s"))
                c["key_edges_multi_digit"] += len(k) >= 2
                c["key_edges_to_personal"] += bool((sts.get(e["next"]) or {}).get("personal"))
                c["key_edges_to_unknown_state"] += e["next"] not in sts and e["next"] != "hangup"
            inv = s.get("invalid_key")
            if isinstance(inv, dict):
                c["invalid_key_recorded"] += 1
                c["invalid_key_with_next"] += bool(inv.get("next"))
        per[org] = dict(sorted(c.items()))
        tot.update(c)
        orgc[org] = {"org": o.get("org"), "answer_delay_s": o.get("answer_delay_s"), "unseen_timeout_s": o.get("unseen_timeout_s"),
                     "barge_stop_s": o.get("barge_stop_s"), "barge_extra_s": o.get("barge_extra_s"),
                     "first_audio_median_s": (o.get("first_audio_s") or {}).get("median"), "noise": o.get("noise"),
                     "start_default": o.get("start"), "start_rules": len(o.get("start_rules") or [])}
    doc = {"states": 198, "states_with_clip": 190, "states_personal": 8, "key_edges": 159, "key_edges_n1": 47,
           "after_audio_states": 69, "input_states": 46, "silent_wait_states": 71, "barge_in_unknown": 167}
    return {"note": "개수는 개인정보 상태를 포함한다(따로 적은 칸 제외). delay·timeout 측정은 median 값이 있는 것만 센다.",
            "total": dict(sorted(tot.items())), "by_org": per, "org_constants": orgc,
            "doc_claims_check": {k: {"doc": v, "bundle": tot.get(k, 0), "same": tot.get(k, 0) == v} for k, v in doc.items()}}


# 5. 결함 정량 -------------------------------------------------------------------
def _headline(res: list[dict]) -> dict:
    recs = [x["rec"] for x in res]
    s = rc.summarize(recs)["ALL"]
    fl = Counter(p["flag"] for r in recs for p in r["path"] if "flag" in p)
    ends = Counter(_end_kind(r["offmap"]) for r in recs if r["offmap"])
    return {"segments": s["segments"], "nonstart": s["nonstart"], "outcome": s["outcome"],
            "before_flag_nonstart": s["before_flag_nonstart"], "flags": dict(fl), "offmap_kinds": dict(ends)}


def _keyevents_bug(out: Path, targets: list, workers: int, base: list[dict], org_names: dict) -> dict:
    # (1) 창이 다음 전송과 겹쳐 '#' 판정이 바뀌는 입력 사건
    ev_total = ev_flip = 0
    calls_aff, by_org = [], Counter()
    for d, ph in targets:
        try:
            j, pcm = rc.load_call(Path(d))
        except Exception:
            continue
        if j.get("backend") not in (None, "clawops") or len(pcm) < 8000:
            continue
        a, b = rc.key_events(j, pcm), _key_events_fixed(j, pcm)
        for e in j.get("events", []):
            if e.get("type") == "keys_sent" and (e.get("kind") == "input" or e.get("keys") is None):
                ev_total += 1
        if a != b:
            calls_aff.append(Path(d).name)
            by_org[ph] += 1
            ev_flip += sum(1 for x in a if x[1] == "#") - sum(1 for x in b if x[1] == "#")
    aff = set(calls_aff)
    # (2) 현 묶음 + 고친 key_events
    fixed = _check_all(BUNDLE_DIR, targets, workers, keyfix=True)
    # (3) 묶음 다시 만들기: 대조(그대로)와 고친 판
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    ctrl_dir, fix_dir = out / "bundle_ctrl", out / "bundle_keyfix"
    replay_build.build(SPEC_DIR, str(ctrl_dir), replay_build.ROOTS, set())
    orig_ke = rc.key_events
    try:
        rc.key_events = _key_events_fixed
        replay_build.build(SPEC_DIR, str(fix_dir), replay_build.ROOTS, set())
    finally:
        rc.key_events = orig_ke
    cur = json.loads((Path(BUNDLE_DIR) / "bundle.json").read_text(encoding="utf-8"))
    ctl = json.loads((ctrl_dir / "bundle.json").read_text(encoding="utf-8"))
    fxb = json.loads((fix_dir / "bundle.json").read_text(encoding="utf-8"))
    sw_changes = []
    for org, o in cur["orgs"].items():
        for sid, s in o["states"].items():
            f = fxb["orgs"][org]["states"][sid]
            if (s.get("silent_wait_s"), s.get("silent_wait_n")) != (f.get("silent_wait_s"), f.get("silent_wait_n")):
                sw_changes.append({"org": org, "state": sid, "before": [s.get("silent_wait_s"), s.get("silent_wait_n")],
                                   "after": [f.get("silent_wait_s"), f.get("silent_wait_n")]})
    fixed_rebuilt = _check_all(str(fix_dir), targets, workers, keyfix=True)

    def aff_rows(rs):
        o = Counter()
        for x in rs:
            r = x["rec"]
            if r["dir"] in aff:
                ns = [s for s in r["segs"] if not s["start"]]
                o["calls"] += 1
                o["segments"] += len(r["segs"])
                o["same_m3"] += sum(s["same"] for s in r["segs"])
                o["nonstart"] += len(ns)
                o["nonstart_same_m3"] += sum(s["same"] for s in ns)
                o["offmap_early"] += bool(r["early"] and not r["personal"])
                o["personal_early"] += bool(r["early"] and r["personal"])
                o["calls_with_flags"] += bool(r["flags"])
                o["type_ahead_flags"] += sum(1 for p in r["path"] if p.get("flag") == "type_ahead")
                o["barge_unverified_flags"] += sum(1 for p in r["path"] if p.get("flag") == "barge_unverified")
        return dict(o)

    return {
        "where": "replay_check.py 113~116행 (key_events: tone_count 창 t − 0.02 ~ t + KEY_TONE_S×(n+3))",
        "input_events": ev_total, "spurious_hash_digits_removed_by_fix": ev_flip,
        "calls_affected": len(calls_aff), "calls_affected_by_org": {k: {"org": org_names.get(k), "calls": v} for k, v in by_org.items()},
        "affected_calls_current": aff_rows(base), "affected_calls_fixed_keyevents": aff_rows(fixed),
        "affected_calls_fixed_keyevents_rebuilt_bundle": aff_rows(fixed_rebuilt),
        "headline_current": _headline(base), "headline_fixed_keyevents": _headline(fixed),
        "headline_fixed_keyevents_rebuilt_bundle": _headline(fixed_rebuilt),
        "tester_rule_fixed_keyevents_rebuilt_bundle": {k: v for k, v in _tester_rule(fixed_rebuilt, org_names).items()
                                                       if k in ("segments_kept_share", "calls_no_exclusion_event", "kept_m3",
                                                                "dropped_m3", "kept_nonstart_m3",
                                                                "first_exclusion_reason_calls")},
        "behaviors_nonstart_current_vs_fixed_rebuilt": _behavior_shift(base, fixed_rebuilt, org_names),
        "control_rebuild_equals_current_bundle": ctl == cur,
        "rebuilt_bundle_silent_wait_changes": sw_changes,
        "rebuilt_bundle_other_fields_equal": _bundle_equal_except_silent(cur, fxb),
        "affected_call_dirs_local_only": "calls_local.json 의 keyevents_bug_affected 참고",
        "_affected_dirs": sorted(aff),
    }


def _behavior_shift(base: list[dict], other: list[dict], org_names: dict) -> dict:
    """두 실행의 표지별 M3 k/n(비시작·전체) 대조표 (값이 다른 표지만)."""
    a, b = _behaviors(base, org_names), _behaviors(other, org_names)
    out = {}
    for lb in a:
        x, y = a[lb]["nonstart"]["m3"], (b.get(lb) or {}).get("nonstart", {}).get("m3", {"k": 0, "n": 0})
        xa, ya = a[lb]["all"]["m3"], (b.get(lb) or {}).get("all", {}).get("m3", {"k": 0, "n": 0})
        if (x["k"], x["n"], xa["k"], xa["n"]) != (y["k"], y["n"], ya["k"], ya["n"]):
            out[lb] = {"nonstart": [f"{x['k']}/{x['n']}", f"{y['k']}/{y['n']}"], "all": [f"{xa['k']}/{xa['n']}", f"{ya['k']}/{ya['n']}"]}
    return out


def _thirdfail_bug(targets: list, workers: int, base: list[dict], org_names: dict) -> dict:
    """자기 반복 상한이 세 번째 무입력 규칙보다 먼저 걸리는 결함 정량: 수정 엔진(_ThirdFailPatched)으로 같은 검사 재실행."""
    pt = _check_all(BUNDLE_DIR, targets, workers, keyfix=False, engine="thirdfail")
    by = {x["rec"]["dir"]: x for x in pt}
    changed = []
    for x in base:
        r0 = x["rec"]
        r1 = by[r0["dir"]]["rec"]
        if _strip(r0["path"]) != _strip(r1["path"]):
            changed.append({"org": r0["org"], "before_end": r0["offmap"], "after_end": r1["offmap"],
                            "before_early": r0["early"], "after_early": r1["early"],
                            "before_segs": [[s["from"], s["to"], s["m3"]] for s in r0["segs"]],
                            "after_segs": [[s["from"], s["to"], s["m3"]] for s in r1["segs"]],
                            "after_vias": [p["via"] for p in r1["path"] if "state" in p][-4:]})
    return {"where": "replay_ars.py 263~265행(_audio_done 의 third_fail 조건 self._timeouts >= 2)과 377~386행(_timeout_fires 자기 반복 "
                     "상한), replay_build.py 195~199행(count_rule → third_fail_next)",
            "states_selfloop_with_third_fail": [[o, sid] for o, org in ReplayBundle(BUNDLE_DIR).orgs.items()
                                                for sid, s in org["states"].items()
                                                if (s.get("after_audio") or {}).get("third_fail_next")
                                                and (s.get("after_audio") or {}).get("next") == sid],
            "calls_changed": len(changed), "changed": changed,
            "headline_current": _headline(base), "headline_patched": _headline(pt)}


def _stop_tails(res: list[dict]) -> dict:
    """정지 구간의 정지 뒤 부분(비교도, after_stop 집계도 되지 않는 부분)에 원래 말소리가 있는 통화 수."""
    agg = defaultdict(Counter)
    speech = defaultdict(float)
    ok = 0
    for x in res:
        st = x["lab"].get("stop_tail")
        if not st:
            continue
        k = st["kind"]
        agg[k]["ended_calls"] += 1
        agg[k]["tail_present"] += st["seg_to"] is not None
        agg[k]["tail_orig_sound"] += st["orig_sound"]
        agg[k]["tail_orig_sound_head_not_scored"] += st["orig_sound"] and st["head_segment_scored"] is False
        speech[k] += st["orig_speech_s"]
        ok += st["stop_vs_replay_end_ok"]
    s = rc.summarize([x["rec"] for x in res])["ALL"]["conservative"]
    extra = sum(agg[k]["tail_orig_sound"] for k in agg if k not in ("personal", "hangup"))
    return {"by_end_kind": {k: {**dict(v), "tail_orig_speech_s": round(speech[k], 1)} for k, v in agg.items()},
            "stop_time_consistent_with_replay_end_s": ok,
            "conservative_current": {**s, "rate": round(s["k"] / s["n"], 4)},
            "conservative_with_offmap_tails": {"k": s["k"], "n": s["n"] + extra, "rate": round(s["k"] / (s["n"] + extra), 4),
                                               "wilson": rc.wilson(s["k"], s["n"] + extra)}}


def _bundle_equal_except_silent(a: dict, b: dict) -> bool:
    def strip(m):
        m = json.loads(json.dumps(m))
        for o in m["orgs"].values():
            for s in o["states"].values():
                s.pop("silent_wait_s", None)
                s.pop("silent_wait_n", None)
        return m
    return strip(a) == strip(b)


# 실행 ---------------------------------------------------------------------------
def run(out_dir: str = DEFAULT_OUT, workers: int = 6) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    t_all = time.time()
    bundle = ReplayBundle(BUNDLE_DIR)
    org_names = {k: v.get("org") for k, v in bundle.orgs.items()}
    targets = [(str(d), ph) for d, ph in rc.calls(None, bundle)]
    timing = {}

    t = time.time()
    res = _check_all(BUNDLE_DIR, targets, workers, keyfix=False)
    timing["check_s"] = round(time.time() - t, 1)
    verify = _verify_dev(res)
    summary = rc.summarize([x["rec"] for x in res])

    t = time.time()
    behaviors = _behaviors(res, org_names)
    tester = _tester_rule(res, org_names)
    timing["labels_s"] = round(time.time() - t, 1)

    t = time.time()
    static = _static_table(BUNDLE_DIR)
    bug = _keyevents_bug(out, targets, workers, res, org_names)
    tf_bug = _thirdfail_bug(targets, workers, res, org_names)
    tails = _stop_tails(res)
    timing["keyevents_bug_s"] = round(time.time() - t, 1)

    t = time.time()
    realtime = _realtime()
    timing["realtime_s"] = round(time.time() - t, 1)
    timing["total_s"] = round(time.time() - t_all, 1)

    affected_dirs = bug.pop("_affected_dirs")
    result = {
        "module": "replay_validate_behaviors", "method_sections": ["6.6", "6.7", "6.8", "8.2"],
        "conditions": {"bundle": BUNDLE_DIR, "oracle": True, "clock": "폴더 이름의 날짜·시각(replay_check.call_clock)",
                       "seed": "check_call: md5(폴더 이름); 부트스트랩 7; 실시간 변형 7", "same_m3": rc.SAME_M3,
                       "same_m1": rc.SAME_M1, "min_seg_s": rc.MIN_SEG_S, "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED},
                       "weak_n": WEAK_N},
        "verification_vs_dev_run": verify,
        "summary_check": {k: summary[k] for k in summary},
        "label_rule": ("path 사건 시각 t 가 [from, to) 인 구간에 배정. 0.6초 미만으로 빠진 구간의 사건은 다음 남은 구간으로 넘긴다. "
                       "양쪽 무음·정지 뒤·마지막 짧은 구간의 사건은 배정 안 됨(events.unassigned_*). 끼어들기 = 엔진이 말하는 중에 "
                       "받았고 미리 누른 키(진입 대기 중 + 두 자리 이음 아님)로 버려지지 않은 키."),
        "behaviors": behaviors,
        "tester_rule_8_2": tester,
        "realtime_6_7": realtime,
        "static_table": static,
        "keyevents_window_bug": bug,
        "thirdfail_vs_repeat_cap_bug": tf_bug,
        "stop_segment_tail_uncounted": tails,
        "timing_s": timing,
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    local = {"note": "로컬 전용: 녹음 폴더 이름이 들어 있다. 받아쓰기 문장은 없다.",
             "keyevents_bug_affected": affected_dirs,
             "calls": [{"dir": x["rec"]["dir"], "org": x["rec"]["org"], "offmap": x["rec"]["offmap"], "flags": x["rec"]["flags"],
                        "first_exclusion": x["lab"]["first_excl"], "raw_status": x["lab"]["raw_status"],
                        "segs": [{"from": s["from"], "to": s["to"], "m3": s["m3"], "m1": s["m1"], "same": s["same"],
                                  "labels": lb, "dropped": dr}
                                 for s, lb, dr in zip(x["rec"]["segs"], x["lab"]["labels"], x["lab"]["dropped"])]}
                       for x in res]}
    (out / "calls_local.json").write_text(json.dumps(local, ensure_ascii=False, indent=1), encoding="utf-8")
    return _compact(result)


def _compact(r: dict) -> dict:
    b = r["behaviors"]

    def m(lb, scope="nonstart"):
        x = b[lb][scope]
        return {"m3": f"{x['m3']['k']}/{x['m3']['n']}", "m3_rate": x["m3"]["rate"], "m3_wilson": x["m3"]["wilson"],
                "m3_boot": x["m3"].get("boot"), "m1": f"{x['m1']['k']}/{x['m1']['n']}", "weak": x["weak"]}

    t = r["tester_rule_8_2"]
    rt = r["realtime_6_7"]["totals"]
    bug = r["keyevents_window_bug"]
    return {
        "verification_vs_dev_run": r["verification_vs_dev_run"].get("differences"),
        "all_segments_m3": r["summary_check"]["ALL"]["segments"], "nonstart_m3": r["summary_check"]["ALL"]["nonstart"],
        "behaviors_nonstart": {lb: m(lb) for lb in REQUIRED_LABELS if lb != "start"},
        "start": m("start", "all"),
        "tester_rule": {"segments_kept": t["segments_kept_share"], "calls_no_exclusion": t["calls_no_exclusion_event"],
                        "kept_m3": t["kept_m3"], "dropped_m3": t["dropped_m3"], "kept_nonstart_m3": t["kept_nonstart_m3"],
                        "first_exclusion": t["first_exclusion_reason_calls"]},
        "realtime": {k: rt[k] for k in ("runs", "frames_compared", "frame_mismatch", "path_equal_runs", "transcripts_equal_runs",
                                        "flush_equal_runs", "keys", "key_err_s_min", "key_err_s_max",
                                        "key_frame_diff_vs_check_framecount", "jitter_abs_p99_s",
                                        "final_drift_s")},
        "static_doc_check": {k: v["same"] for k, v in r["static_table"]["doc_claims_check"].items()},
        "thirdfail_bug": {"calls_changed": r["thirdfail_vs_repeat_cap_bug"]["calls_changed"],
                          "outcome_current": r["thirdfail_vs_repeat_cap_bug"]["headline_current"]["outcome"],
                          "outcome_patched": r["thirdfail_vs_repeat_cap_bug"]["headline_patched"]["outcome"]},
        "stop_tails": r["stop_segment_tail_uncounted"],
        "keyevents_bug": {"calls_affected": bug["calls_affected"], "spurious_hash": bug["spurious_hash_digits_removed_by_fix"],
                          "headline_current_nonstart": bug["headline_current"]["nonstart"],
                          "headline_fixed_rebuilt_nonstart": bug["headline_fixed_keyevents_rebuilt_bundle"]["nonstart"]},
        "timing_s": r["timing_s"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    print(json.dumps(run(a.out_dir, a.workers), ensure_ascii=False, indent=1))
