"""
녹음 기반 가상 ARS 검증 — 받아쓰기 모드 비교

  python replay_validate_transcripts.py [--out <출력 폴더>] [--workers 6]

측정 대상
  받아쓰기 모드(REPLAY_TRANSCRIPTS=1)에서 가상 ARS 가 내보내는 단락(문장·시작 시각·나온 시각)이, 같은 키를 같은 시각에 넣었을 때
  녹음된 원래 통화에서 서버가 실제로 받아쓴 단락(log.json 의 stt 사건)과 얼마나 같은지 측정. 소리 비교(M3: MFCC DTW 유사도)는 하지 않음.

대상과 입력
  - 대상: replay_check.calls 가 내는 녹음 중 log.json 의 backend 가 clawops(또는 없음)이고 call.wav 가 8,000표본(1초) 이상인
    통화. replay_check.check_call 과 같은 거르기.
  - 엔진 실행: replay_check.run_engine 을 그대로 사용. 키 = replay_check.key_events(톤 끝 시각 = keys_sent t + 0.25i + 0.15초,
    민감정보는 자릿수만큼 '0', 덧붙인 '#'), 첫 상태 = 폴더 이름의 시각, 변형 = 원래 통화와 가장 닮은 것(oracle 켬), 씨앗 = 폴더
    이름 md5 앞 8자리(check_call 과 같음), 녹음 길이까지 또는 엔진이 끝날 때까지.
  - 관찰용 엔진: replay_check.ReplayEngine 을 ReplayEngine 의 하위 클래스로 교체해, 단락마다 출처 조각·분석 받아쓰기(approx:
    서버 받아쓰기가 없는 조각에 replay_build 가 spec 문구로 넣은 단락) 여부·운명(나옴 / 키 전송 끝에 버림 / 끊김 때 넘김 /
    나오지 않음)만 기록. 동작은 바꾸지 않음(검사: 기록한 '나옴' 단락이 run_engine 이 돌려준 받아쓰기와 같고, 전체 묶음 실행의
    경로·받아쓰기가 replay_check.py --out 으로 저장한 검사 결과(CHECK_V3)와 같은지 확인).
  - 끊김 때 넘김(flush): 주 결과는 엔진이 녹음 밖 표시 없이 실제 끊음(note 없음)으로 끝난 경우에만 엔진의 flush() 를 호출해 그
    단락을 끝난 시각(stop)에 나온 것으로 추가(세션의 끊김 처리와 같은 동작). 서버 경로(ReplayCall._finish → session._after_hangup →
    flush_transcripts)는 녹음 밖 종료에서도 넘기므로, 모든 종료에서 넘긴 결과를 민감도로 함께 산출. 엔진이 끝나기 전에 녹음이 끝난
    통화(서버가 끊음)는 넘기지 않음(서버도 local_hangup 에서는 넘기지 않음).
  - 민감도(해석 보조): 엔진이 녹음 끝까지 끝나지 않은 통화는 녹음 끝 뒤 입력 없이 TAIL_S = 1.0초 더 실행해, 녹음 끝 전에
    시작한 단락 중 그 사이 나온 것(그 사이 note 없이 끝나면 flush 포함)을 더한 재현율·정밀도를 산출. 비교 창은 그대로.
    원래 통화에서 서버가 단락을 받은 직후 통화를 끝낸 경우 가상 쪽은 같은 단락을 녹음 끝 뒤에 내므로, 녹음 길이에서 실행을
    끝내는 데 따른 잘림 효과를 가늠하기 위한 값.
  - 거르기: 가상 쪽 단락은 ReplayCall 과 같이 글자·숫자 3자 이상(replay_ars._meaningful)만 사용. 원래 통화 쪽은 문장이 빈 것을
    뺀 모든 stt 사건.

비교 (통화마다)
  - 비교 창: 원래 통화 stt 중 seg_start_t < stop. stop = 엔진이 끝났으면 끝난 프레임 시각(eng.frame × 0.02), 아니면 녹음 길이.
    실행(전체 묶음·겹 묶음)마다 따로 정함. 창 밖 단락은 '가상 정지 뒤'로 정지 원인(녹음 밖·개인정보·끊음)별로 집계.
  - 맞춤: 시작 시각 기준 순서 보존 DP 정렬. 가상 시작(단락 시작 + DETECT_LAG_S 0.2초, 엔진 값 그대로)과 원래 통화
    seg_start_t + 0.2 를 비교. 맞춤 비용 |Δ시작|(허용 tol 이내만), 빈칸 비용 tol. 주 허용 1.0초, 보조 0.5·2.0초.
  - 지표: 재현율 = 맞춘 수 / 창 안 원래 단락 수, 정밀도 = 맞춘 수 / 가상 단락 수, 단락 수 비 = 가상 / 원래. 보수적 재현율은
    녹음 밖 종료(개인정보 제외) 뒤의 원래 단락을 놓친 것으로 추가.
  - 문장: 공백·문장부호를 지운 뒤(가-힣·0-9·A-Za-z 만 남김) 완전일치, 문자 유사도 = 1 − 편집거리 / max(두 길이)(주 값).
    원래 문장 길이로 나눈 값(max(0, 1 − 편집거리 / 원래 길이))도 함께 기록. 유사도 ≥ 0.9(같은 소리 문턱과 같은 값) 비율 산출.
    완전일치·유사도는 자기 출처(그 단락의 조각을 시험 통화 자신의 녹음에서 잘라 온 경우)를 뺀 쌍으로 산출.
  - 시각: 나온 시각 차 = 가상 나온 시각 − 원래 t, 시작 차 = 가상 시작 − (seg_start_t + 0.2). 가운뎃값·p10·p90(선형 보간).
    가상 나온 시각은 run_engine 이 적는 값(tick 뒤 프레임 시각)이며 끊김 때 넘긴 단락은 stop. 이 값은 실시간 경로에서
    세션이 받는 때의 받은 소리 시계와 같음(ReplayCall 은 on_transcript 를 작업으로 띄운 뒤 on_audio 로 그 프레임을 먼저 녹음).
    단락의 예정 프레임(at)보다는 한 프레임 늦으므로 가상 쪽 지연은 측정 가운뎃값 + 0.02초(가상 쪽 지연 표로 확인).
  - 못 맞춘 원래 단락의 사유(이 순서로 첫 해당): 같은 문장 후보(가상 단락 중 시작 차 10초 안, 유사도 ≥ 0.8, 가장 닮은 것)가
    있으면 그 운명으로 — 키 전송 끝에 버림(dropped_by_key), 나오지 않음(녹음 밖 종료면 pending_at_offmap_end, 녹음이 먼저 끝났으면
    pending_at_recording_end, 끊음이면 pending_at_hangup), 분석 받아쓰기(approx), 나왔으나 허용 밖(timing_shift). 후보가 없으면 —
    단락 소리 시작 + 0.3초 시각의 가상 상태가 개인정보 상태(personal_state, 설계상 재생하지 않음), 그 상태에 조각 없음(no_clip,
    겹 묶음), 가상 단락과 시간이 겹침(boundary_split_merge), 겹치는 분석 받아쓰기(approx),
    겹치는 버린 단락(dropped_by_key), 겹치는 나오지 않은 단락(pending_*), 그 밖(no_replay_paragraph).
    못 맞춘 가상 단락의 사유: approx, 같은 문장 후보가 못 맞춘 원래 단락에 있음(timing_shift), 시간 겹침(boundary_split_merge),
    끊김 때 넘긴 단락(flushed_no_recorded), 그 밖(no_recorded_paragraph).
  - 인과 지표: AI 가 누른 키(keys_sent actor=ai)마다 전송 시작 시각 이전(t ≤ 누른 시각)에 나온 마지막 원래 단락을 판단 근거로
    보고, 그 단락과 맞춘(허용 1.0초) 가상 단락이 누른 시각 이전에 나왔는지(in_time), 늦게 나왔는지(late), 맞춘 단락이 없는지
    (missing)로 분류. 비율 = in_time / (in_time + late + missing), 가상 정지 뒤의 누름은 따로 세고 보수적 비율에서는 실패로 집계.
    맞춘 쌍의 유사도 ≥ 0.9 까지 요구한 비율도 산출. 늦음의 지연(가상 나온 시각 − 누른 시각)과, 나온 시각을 단락의 예정
    프레임(at, 한 프레임 앞)으로 볼 때의 비율을 민감도로 함께 산출.
  - 진단: 키 전송 끝(keys_done)에 버린 단락 중 같은 상태 방문 안에서 그 뒤에도 그 단락 소리가 1초 이상 실제로 나간 것의 수와,
    그 단락을 원래 통화가 받아썼는지(시작 차 3초 안, 유사도 ≥ 0.8).

일반화 설계
  - 표본 내: 전체 묶음(FULL_BUNDLE, 검사 통화의 녹음도 조각 출처에 포함). 같은 코드로 다시 만든 묶음과 같은지 확인.
  - 5겹 교차 검증(검사 통화의 녹음을 조각 출처에서 제외): 기관(전화번호)별로 폴더 이름을 정렬하고 기관마다 새
    random.Random(FOLD_SEED) 로 섞은 뒤, 기관을 번호순으로 이어 i % 5 로 겹을 나눔(replay_validate_kfold 와 같은 규칙).
    겹마다 replay_build.build(--exclude 그 겹 녹음)로 묶음을 출력 폴더 아래에 새로 만들고 그 겹 통화만 검사. 묶음의 조각
    출처에 그 겹 녹음이 없는지 확인. 주 수치는 이 값.

기준값 (인식 변동)
  - 상한: 같은 안내 단락(기관·상태·변형·단락 번호)의 서로 다른 원래 통화 받아쓰기끼리의 완전일치·유사도. 단락 이름은 표본 내
    실행에서 그 원래 단락과 시각으로 맞춘 가상 단락에서 가져오고, CHECK_V3 에서 그 단락 시작(seg_start_t + 0.2)을
    담은 구간이 같은 소리(M3 ≥ 0.90)이고 경로가 같은 통화만 신뢰(문장으로 고르지 않아 순환 없음). 단락 이름마다 서로 다른 통화
    쌍을 최대 50쌍(random.Random(7), 이름순으로 소비) 사용. 같은 집합에서 가상 대 원래(자기 출처 제외) 완전일치를 나란히 산출.
  - 하한: 같은 기관의 다른 상태 원래 받아쓰기 무작위 쌍(기관마다 2,000회 뽑아 상태가 다른 것만, 같은 난수열)의 유사도 분포.

인식 지연
  - replay_build.measure_latency 와 같은 규칙: 녹음 폴더(replay_build.ROOTS)의 call.wav 가 있고 backend 가 clawops(또는 없음)인
    모든 폴더의 stt 사건에서 단락 소리 길이 = seg_end_t − seg_start_t − 0.2 − 1.2, 지연 = t − seg_end_t(음수는 가운뎃값에서 뺌),
    구간 5초 미만·5~10·10~20·20초 이상. 분포(n·음수 수·가운뎃값·p10·p90)를 내고 measure_latency 의 값과 같은지 확인. 가상 쪽
    지연(나온 시각 − 단락 닫힘)도 같은 구간으로 산출.

보고
  - 비율마다 k/n, Wilson 95% 구간(replay_check.wilson), 통화 단위 군집 부트스트랩 95% 구간(1,000회, 지표마다 새
    numpy default_rng(7), 2.5·97.5 백분위). 기준값 쌍 비율의 부트스트랩은 단락 이름 단위 군집. 기관별로도 산출.
  - 개인정보: 받아쓰기 문장은 메모리에서만 쓰고 결과에는 수·길이·유사도만 기록. 결과의 8자리 숫자는 기관 번호와 폴더 이름의
    날짜.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
import statistics
import sys
import time
import wave
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_ars as ra  # noqa: E402
import replay_build as rb  # noqa: E402
import replay_check as rc  # noqa: E402

NAME = "transcripts"
OUT_DEFAULT = r"C:\Users\User\Documents\ars_run\replay\validate\transcripts"
SPEC_DIR = r"C:\Users\User\Documents\ars_run\replay\specs"
FULL_BUNDLE = r"C:\Users\User\Documents\ars_run\replay\bundle"
CHECK_V3 = r"C:\Users\User\Documents\ars_run\replay\check_v3_oracle.json"
FOLD_SEED, K = 20261109, 5
TOLS = (0.5, 1.0, 2.0)
MAIN_TOL = 1.0
SIM_SAME = 0.9          # 문장 유사도 문턱
SIM_CAND = 0.8          # 못 맞춘 사유를 붙일 때 같은 문장 후보로 볼 유사도
CAND_WINDOW_S = 10.0    # 같은 문장 후보를 찾는 시작 차 범위
TAIL_S = 1.0            # 민감도: 녹음이 끝난 뒤 입력 없이 더 돌리는 시간
BOOT_N, BOOT_SEED = 1000, 7
PAIR_SEED, SAME_CAP, DIFF_DRAWS = 7, 50, 2000
LAT_EDGES = (5.0, 10.0, 20.0)
LAT_NAMES = ("<5", "5-10", "10-20", ">=20")
LAG = ra.DETECT_LAG_S
FS = ra.FRAME_S
CLOSE_S = rb.PARAGRAPH_SILENCE_S

_NORM = re.compile(r"[^가-힣0-9A-Za-z]")


# 문자 비교 ----------------------------------------------------------------------
def norm(text: str) -> str:
    return _NORM.sub("", text or "")


def lev(a: str, b: str) -> int:
    """Levenshtein 거리 (Hyyrö 비트 병렬, Python 정수)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    peq: dict[str, int] = {}
    for i, ch in enumerate(b):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask = (1 << m) - 1
    hi = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for ch in a:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & hi:
            score += 1
        elif mh & hi:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | (~(xv | ph) & mask)) & mask
        mv = ph & xv
    return score


def compare(rec_text: str, rep_text: str) -> tuple[bool, float, float]:
    """(완전일치, 1 − 거리/max 길이, max(0, 1 − 거리/원래 문장 길이)). 공백·문장부호 제거 뒤."""
    a, b = norm(rec_text), norm(rep_text)
    if not a and not b:
        return True, 1.0, 1.0
    d = lev(a, b)
    return a == b, 1.0 - d / max(len(a), len(b)), (max(0.0, 1.0 - d / len(a)) if a else 0.0)


# 통계 ---------------------------------------------------------------------------
def boot_ci(ks, ns) -> list[float] | None:
    """통화(군집) 단위 부트스트랩 95% 구간: 군집을 복원 추출해 Σk/Σn. 지표마다 새 default_rng(7)."""
    ks, ns = np.asarray(ks, float), np.asarray(ns, float)
    if len(ns) == 0 or ns.sum() == 0:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(ns), size=(BOOT_N, len(ns)))
    num, den = ks[idx].sum(1), ns[idx].sum(1)
    r = num[den > 0] / den[den > 0]
    return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]


def rate(ks, ns, boot: bool = True) -> dict:
    k, n = int(sum(ks)), int(sum(ns))
    out = {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson": rc.wilson(k, n)}
    if boot:
        out["boot"] = boot_ci(ks, ns)
    return out


def ratio(nums, dens) -> dict:
    a, b = sum(nums), sum(dens)
    return {"num": int(a), "den": int(b), "ratio": round(a / b, 4) if b else None, "boot": boot_ci(nums, dens)}


def dist(xs) -> dict:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    a = np.asarray(xs)
    return {"n": len(xs), "median": round(float(np.median(a)), 3), "p10": round(float(np.percentile(a, 10)), 3),
            "p90": round(float(np.percentile(a, 90)), 3), "mean": round(float(a.mean()), 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}


def lat_bucket(speech_s: float) -> int:
    return sum(speech_s >= e for e in LAT_EDGES)


# 자료 ---------------------------------------------------------------------------
def recorded_stt(j: dict) -> list[dict]:
    """원래 통화의 stt 사건(문장이 빈 것 제외): idx(사건 순서), start = seg_start_t, end = seg_end_t, t, text. 시작·사건 순 정렬."""
    out = []
    for e in j.get("events", []):
        if e.get("type") != "stt":
            continue
        a, b, at = e.get("seg_start_t"), e.get("seg_end_t"), e.get("t")
        if e.get("text") and None not in (a, b, at):
            out.append({"idx": len(out), "start": float(a), "end": float(b), "t": float(at), "text": e["text"]})
    return sorted(out, key=lambda x: (x["start"], x["idx"]))


def universe(bundle: ra.ReplayBundle) -> list[tuple[Path, str]]:
    """replay_check.check_call 과 같은 거르기 (backend clawops·없음, call.wav 8,000표본 이상)."""
    out = []
    for d, ph in rc.calls(None, bundle):
        try:
            j = json.loads((d / "log.json").read_text(encoding="utf-8"))
            with wave.open(str(d / "call.wav"), "rb") as w:
                n = w.getnframes()
        except Exception:
            continue
        if j.get("backend") not in (None, "clawops") or n < 8000:
            continue
        out.append((d, ph))
    return out


def assign_folds(univ: list[tuple[Path, str]]) -> dict[str, int]:
    """기관별 층화: 기관(번호순)마다 이름순 정렬 → 새 random.Random(FOLD_SEED).shuffle → 기관을 이어 i % 5."""
    by = defaultdict(list)
    for d, ph in univ:
        by[ph].append(d.name)
    fold_of, i = {}, 0
    for ph in sorted(by):
        lst = sorted(by[ph])
        random.Random(FOLD_SEED).shuffle(lst)
        for name in lst:
            fold_of[name] = i % K
            i += 1
    return fold_of


# 관찰용 엔진 ----------------------------------------------------------------------
class TracedEngine(ra.ReplayEngine):
    """ReplayEngine 과 같은 동작. 단락마다 출처·운명만 기록 (sched)."""

    def __init__(self, *a, **k):
        self.sched: list[dict] = []
        self._visit = 0
        super().__init__(*a, **k)

    def _enter(self, *args, **kwargs):
        self._visit += 1
        n0 = len(self._paras)
        super()._enter(*args, **kwargs)
        new = self._paras[n0:]
        if not new:
            return
        sid = self.state_id
        s = self.states[sid]
        src, vname = s.get("paras") or [], None
        for v in s.get("variants") or []:
            if v.get("clip") == self.cur_clip:
                src, vname = v.get("paras") or [], v.get("name")
                break
        for i, p in enumerate(new):
            sp = src[i] if i < len(src) else {}
            r = {"i": len(self.sched), "text": p["text"], "start_f": p["start"], "end_f": p["end"], "at_f": p["at"],
                 "start_s": p["start"] * FS + LAG, "end_s": p["end"] * FS, "clip": self.cur_clip, "state": sid,
                 "variant": vname, "pidx": i, "approx": bool(sp.get("approx")), "visit": self._visit,
                 "speech_s": (float(sp["end"]) - float(sp["start"]) - 0.2 - CLOSE_S) if sp else None,
                 "fate": "pending", "fate_t": None, "emit_t": None}
            p["_trace"] = r
            self.sched.append(r)

    def keys_done(self):
        before = list(self._paras)
        super().keys_done()
        kept = {id(p) for p in self._paras}
        for p in before:
            if id(p) not in kept:
                r = p["_trace"]
                r["fate"], r["fate_t"] = "dropped_key", round(self.frame * FS, 2)
                r["drop_frame"] = self.frame

    def tick(self):
        before = list(self._paras) if self._paras else []
        fr = super().tick()
        if before and len(self._paras) != len(before):
            kept = {id(p) for p in self._paras}
            for p in before:
                if id(p) not in kept and p["_trace"]["fate"] == "pending":
                    r = p["_trace"]
                    r["fate"], r["emit_t"] = "emitted", round(self.frame * FS, 2)
        return fr

    def flush(self):
        pend = [p for p in self._paras if p["start"] <= self.frame]
        out = super().flush()
        for p in pend:
            r = p["_trace"]
            r["fate"], r["emit_t"] = "flushed", round(self.frame * FS, 2)
        return out


# 정렬 ---------------------------------------------------------------------------
def align(a: list[float], b: list[float], tol: float) -> list[tuple[int, int]]:
    """시작 시각 순서 보존 DP. 맞춤 비용 |Δ|(tol 이내만), 빈칸 비용 tol. 동률은 원래 쪽 빈칸 → 가상 빈칸 → 맞춤 순."""
    n, m = len(a), len(b)
    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0], bt[i][0] = i * tol, 1
    for j in range(m + 1):
        dp[0][j], bt[0][j] = j * tol, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, bb = dp[i - 1][j] + tol, 1
            if dp[i][j - 1] + tol < best:
                best, bb = dp[i][j - 1] + tol, 2
            d = abs(a[i - 1] - b[j - 1])
            if d <= tol + 1e-9 and dp[i - 1][j - 1] + d < best:
                best, bb = dp[i - 1][j - 1] + d, 0
            dp[i][j], bt[i][j] = best, bb
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        bb = bt[i][j] if (i > 0 and j > 0) else (1 if i > 0 else 2)
        if bb == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif bb == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def overlaps(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


# 통화 하나 -----------------------------------------------------------------------
_BUNDLES: dict[str, tuple] = {}


def _init_worker() -> None:
    rc.ReplayEngine = TracedEngine
    logging.getLogger("ars.replay").setLevel(logging.ERROR)


def _bundle(bdir: str):
    if bdir not in _BUNDLES:
        b = ra.ReplayBundle(bdir)
        _BUNDLES[bdir] = (b, {org: rc.clip_sources(b, org) for org in b.orgs})
    return _BUNDLES[bdir]


def _state_at(path: list[dict], t: float) -> str | None:
    cur = None
    for p in path:
        if "state" in p and p["t"] <= t:
            cur = p["state"]
    return cur


def _evaluate(rec_in, reps, tol):
    pairs = align([x["start"] + LAG for x in rec_in], [r["start_s"] for r in reps], tol)
    return pairs


def eval_call(task) -> dict:
    bdir, tag, dpath, phone = task
    bundle, srcmap = _bundle(bdir)
    d = Path(dpath)
    org = bundle.org_for(phone)
    states = bundle.orgs[org]["states"]
    j, pcm = rc.load_call(d)
    keys = rc.key_events(j, pcm)
    end_s = len(pcm) / 8000
    eo = rc.envelope(pcm, rc.key_mask(keys))
    seed = int(hashlib.md5(d.name.encode("utf-8")).hexdigest()[:8], 16)
    eng, _out, frame_src, texts = rc.run_engine(bundle, org, rc.call_clock(d.name), keys, end_s, eo, oracle=True, seed=seed)
    assert isinstance(eng, TracedEngine)
    stop_s = eng.frame * FS if eng.ended else end_s
    note_kind = None if not eng.ended else ("hangup" if not eng.note else ("personal" if "개인정보" in eng.note else "offmap"))
    v3_texts = [[round(s, 2), at, len(x)] for x, s, at in texts]  # CHECK_V3 와 비교 (flush 전)
    if eng.ended:
        eng.flush()
    sched = eng.sched
    src = srcmap[org]
    for r in sched:
        r["self"] = src.get(r["clip"]) == d.name
        r["meaningful"] = ra._meaningful(r["text"])
    traced = sorted((round(r["start_s"], 2), r["emit_t"], len(r["text"])) for r in sched if r["fate"] == "emitted")
    trace_ok = traced == sorted(tuple(x) for x in v3_texts)

    emitted = [r for r in sched if r["fate"] == "emitted" and r["meaningful"]]
    flushed = [r for r in sched if r["fate"] == "flushed" and r["meaningful"]]
    not_meaningful = sum(1 for r in sched if r["fate"] in ("emitted", "flushed") and not r["meaningful"])
    order = lambda r: (r["start_s"], r["emit_t"], r["i"])  # noqa: E731
    reps = sorted(emitted + (flushed if note_kind == "hangup" else []), key=order)
    reps_prod = sorted(emitted + flushed, key=order)
    included = {r["i"] for r in reps}

    rec_all = recorded_stt(j)
    rec_in = [x for x in rec_all if x["start"] < stop_s]
    rec_after = [x for x in rec_all if x["start"] >= stop_s]

    by_tol = {}
    for tol in TOLS:
        by_tol[str(tol)] = len(_evaluate(rec_in, reps, tol))
    prod_matched = len(_evaluate(rec_in, reps_prod, MAIN_TOL))
    pairs = _evaluate(rec_in, reps, MAIN_TOL)
    m_rec = {i: jj for i, jj in pairs}
    m_rep = {jj: i for i, jj in pairs}

    # 맞춘 쌍
    pair_out = []
    for i, jj in pairs:
        x, r = rec_in[i], reps[jj]
        ex, s_max, s_rec = compare(x["text"], r["text"])
        pair_out.append({"rec_idx": x["idx"], "onset": round(x["start"] + LAG, 2), "d_start": round(r["start_s"] - (x["start"] + LAG), 3),
                         "d_emit": round(r["emit_t"] - x["t"], 3), "exact": ex, "sim": round(s_max, 4), "sim_rec": round(s_rec, 4),
                         "self": r["self"], "approx": r["approx"], "flushed": r["fate"] == "flushed",
                         "label": [phone, r["state"], r["variant"], r["pidx"]]})

    # 못 맞춘 원래 단락의 사유
    def pend_reason():
        if note_kind in ("offmap", "personal"):
            return "pending_at_offmap_end"
        if note_kind is None:
            return "pending_at_recording_end"
        return "pending_at_hangup"

    real_end = next((str((e.get("data") or {}).get("end_reason")) for e in reversed(j.get("events", []))
                     if e.get("type") == "event:ended"), None)
    rec_reasons = Counter()
    ts_dstart, pend_end = [], []
    for i, x in enumerate(rec_in):
        if i in m_rec:
            continue
        xs = x["start"] + LAG
        best = None
        for r in sched:
            if abs(r["start_s"] - xs) > CAND_WINDOW_S:
                continue
            _, s_max, _ = compare(x["text"], r["text"])
            if s_max >= SIM_CAND and (best is None or (s_max, -abs(r["start_s"] - xs)) > best[0]):
                best = ((s_max, -abs(r["start_s"] - xs)), r)
        reason = None
        if best is not None:
            r = best[1]
            if r["fate"] == "dropped_key":
                reason = "dropped_by_key"
            elif r["fate"] == "pending" or (r["fate"] == "flushed" and r["i"] not in included):
                reason = pend_reason()
                if reason == "pending_at_recording_end":
                    pend_end.append([round(x["t"] - end_s, 2), round(r["at_f"] * FS - end_s, 2), real_end])
            elif r["approx"]:
                reason = "approx"
            else:
                reason = "timing_shift"
                ts_dstart.append(round(r["start_s"] - xs, 2))
        else:
            st = _state_at(eng.path, xs + 0.3)
            sd = states.get(st or "", {})
            span = (x["start"], x["end"])
            if sd.get("personal"):
                reason = "personal_state"
            elif st is not None and not sd.get("clip"):
                reason = "no_clip"
            elif any(overlaps(r["start_s"] - LAG, r["end_s"], *span) for r in reps):
                reason = "boundary_split_merge"
            elif any(r["approx"] and overlaps(r["start_s"] - LAG, r["end_s"], *span) for r in sched):
                reason = "approx"
            elif any(r["fate"] == "dropped_key" and overlaps(r["start_s"] - LAG, r["end_s"], *span) for r in sched):
                reason = "dropped_by_key"
            elif any((r["fate"] == "pending" or (r["fate"] == "flushed" and r["i"] not in included))
                     and overlaps(r["start_s"] - LAG, r["end_s"], *span) for r in sched):
                reason = pend_reason()
            else:
                reason = "no_replay_paragraph"
        rec_reasons[reason] += 1

    rep_reasons = Counter()
    um_rec = [x for i, x in enumerate(rec_in) if i not in m_rec]
    for jj, r in enumerate(reps):
        if jj in m_rep:
            continue
        if r["approx"]:
            reason = "approx"
        elif any(abs(r["start_s"] - (x["start"] + LAG)) <= CAND_WINDOW_S and compare(x["text"], r["text"])[1] >= SIM_CAND
                 for x in um_rec):
            reason = "timing_shift"
        elif any(overlaps(r["start_s"] - LAG, r["end_s"], x["start"], x["end"]) for x in rec_in):
            reason = "boundary_split_merge"
        elif r["fate"] == "flushed":
            reason = "flushed_no_recorded"
        else:
            reason = "no_recorded_paragraph"
        rep_reasons[reason] += 1

    # 인과 지표 (AI 키)
    causal = Counter()
    for e in j.get("events", []):
        if e.get("type") != "keys_sent" or e.get("actor") != "ai":
            continue
        tp = float(e["t"])
        prior = [x for x in rec_all if x["t"] <= tp]
        if not prior:
            causal["no_prior"] += 1
            continue
        if tp >= stop_s:
            causal["after_stop"] += 1
            continue
        last = max(prior, key=lambda x: (x["t"], x["start"]))
        pos = next((i for i, x in enumerate(rec_in) if x is last), None)
        if pos is None or pos not in m_rec:
            causal["missing"] += 1
            continue
        r = reps[m_rec[pos]]
        if r["emit_t"] <= tp:
            causal["in_time"] += 1
            if compare(last["text"], r["text"])[1] >= SIM_SAME:
                causal["in_time_sim09"] += 1
        else:
            causal["late"] += 1

    # 키 전송 끝에 버린 단락 중, 같은 상태 방문 안에서 그 뒤에도 그 단락 소리가 1초 이상 실제로 나간 것 (진단)
    entries = [round(p["t"] / FS) for p in eng.path if "state" in p]
    drop_audible = []
    for r in sched:
        if r["fate"] != "dropped_key":
            continue
        nxt = min([f for f in entries if f > r["drop_frame"]], default=len(frame_src))
        f0 = max(r["drop_frame"], r["start_f"])
        f1 = min(r["end_f"] - round(CLOSE_S / FS), len(frame_src), nxt)
        aud = sum(1 for f in range(f0, max(f0, f1)) if frame_src[f] == r["clip"]) * FS
        if aud >= 1.0:
            hit = any(abs(x["start"] + LAG - r["start_s"]) <= 3.0 and compare(x["text"], r["text"])[1] >= SIM_CAND for x in rec_all)
            drop_audible.append({"audible_s": round(aud, 2), "not_started_at_drop": r["start_f"] > r["drop_frame"],
                                 "onset_to_drop_s": round((r["drop_frame"] - r["start_f"]) * FS - LAG, 2),
                                 "recorded_has_it": hit, "state": r["state"]})

    rep_lat = [[r["speech_s"], round(r["emit_t"] - r["end_s"], 3)] for r in sched
               if r["fate"] == "emitted" and r["speech_s"] is not None and not r["approx"]]

    # 늦음(인과 지표)의 지연: 가상 나온 시각 − 누른 시각
    late_lag = []
    for e in j.get("events", []):
        if e.get("type") != "keys_sent" or e.get("actor") != "ai":
            continue
        tp = float(e["t"])
        prior = [x for x in rec_all if x["t"] <= tp]
        if not prior or tp >= stop_s:
            continue
        last = max(prior, key=lambda x: (x["t"], x["start"]))
        pos = next((i for i, x in enumerate(rec_in) if x is last), None)
        if pos is not None and pos in m_rec and reps[m_rec[pos]]["emit_t"] > tp:
            late_lag.append([round(reps[m_rec[pos]]["emit_t"] - tp, 3), round(tp - last["t"], 3)])

    path_snapshot = [dict(p) for p in eng.path]
    sched_counts = {"total": len(sched), **Counter(r["fate"] for r in sched)}

    # 민감도: 녹음이 끝난 뒤 입력 없이 1초 더 돌렸을 때 (엔진이 녹음 끝까지 끝나지 않은 통화만; 비교 창은 그대로)
    tail_ended = False
    if not eng.ended:
        while eng.frame * FS < end_s + TAIL_S and not eng.ended:
            eng.tick()
        if eng.ended:
            tail_ended = True
            if not eng.note:
                eng.flush()
    for r in sched:
        if "meaningful" not in r:  # 녹음 끝 뒤 들어간 상태의 단락
            r["self"], r["meaningful"] = src.get(r["clip"]) == d.name, ra._meaningful(r["text"])
    flush_ok_tail = note_kind == "hangup" or (tail_ended and not eng.note)
    reps_tail = sorted([r for r in sched if r["meaningful"] and r["start_s"] - LAG < end_s and
                        (r["fate"] == "emitted" or (r["fate"] == "flushed" and flush_ok_tail))], key=order)
    matched_tail = len(_evaluate(rec_in, reps_tail, MAIN_TOL))

    return {
        "tag": tag, "dir": d.name, "phone": phone, "start": eng.start_state, "end_s": round(end_s, 2), "stop_s": round(stop_s, 2),
        "ended": eng.ended, "note_kind": note_kind, "early": bool(eng.ended and stop_s < end_s - 1.0),
        "n_rec_all": len(rec_all), "n_rec_in": len(rec_in), "n_rec_after": len(rec_after),
        "n_rep": len(reps), "n_rep_prod": len(reps_prod), "n_flush_included": sum(1 for r in reps if r["fate"] == "flushed"),
        "n_flushed_total": len(flushed), "n_not_meaningful": not_meaningful,
        "matched": by_tol, "matched_prod": prod_matched, "pairs": pair_out,
        "rec_reasons": dict(rec_reasons), "rep_reasons": dict(rep_reasons), "causal": dict(causal),
        "sched": sched_counts, "self_sched": sum(1 for r in sched if r.get("self")),
        "drop_audible": drop_audible, "rep_lat": rep_lat, "trace_ok": trace_ok,
        "ts_dstart": ts_dstart, "pend_end": pend_end, "late_lag": late_lag, "real_end": real_end,
        "matched_tail": matched_tail, "n_rep_tail": len(reps_tail), "tail_ended": tail_ended,
        "path": path_snapshot, "v3_texts": v3_texts,
    }


def _build_job(args) -> dict:
    out_dir, exclude = args
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    p = Path(out_dir)
    if p.exists():
        shutil.rmtree(p)
    t = time.time()
    rep = rb.build(SPEC_DIR, out_dir, rb.ROOTS, set(exclude))
    return {"report": rep, "sec": round(time.time() - t, 1)}


# 집계 ---------------------------------------------------------------------------
def aggregate(recs: list[dict]) -> dict:
    """통화 기록 묶음의 지표 (주 결과: note 없이 끝난 경우만 flush, 허용 1.0초)."""
    if not recs:
        return {"calls": 0}
    rin = [r["n_rec_in"] for r in recs]
    rep = [r["n_rep"] for r in recs]
    out = {"calls": len(recs),
           "outcome": {"full": sum(1 for r in recs if not r["early"]),
                       "offmap_early": sum(1 for r in recs if r["early"] and r["note_kind"] == "offmap"),
                       "personal_early": sum(1 for r in recs if r["early"] and r["note_kind"] == "personal"),
                       "hangup_early": sum(1 for r in recs if r["early"] and r["note_kind"] == "hangup")},
           "end_kind": dict(Counter(str(r["note_kind"]) for r in recs)),
           "recorded_total": sum(r["n_rec_all"] for r in recs), "recorded_in_window": sum(rin),
           "recorded_after_stop": {k: sum(r["n_rec_after"] for r in recs if str(r["note_kind"]) == k)
                                   for k in ("offmap", "personal", "hangup", "None")},
           "replay_emitted": sum(rep), "replay_flush_included": sum(r["n_flush_included"] for r in recs),
           "replay_filtered_lt3": sum(r["n_not_meaningful"] for r in recs)}
    out["by_tol"] = {}
    for tol in TOLS:
        mk = [r["matched"][str(tol)] for r in recs]
        out["by_tol"][str(tol)] = {"recall": rate(mk, rin), "precision": rate(mk, rep)}
    mk = [r["matched"][str(MAIN_TOL)] for r in recs]
    out["recall"] = out["by_tol"][str(MAIN_TOL)]["recall"]
    out["precision"] = out["by_tol"][str(MAIN_TOL)]["precision"]
    out["count_ratio"] = ratio(rep, rin)
    cons_n = [r["n_rec_in"] + (r["n_rec_after"] if r["note_kind"] == "offmap" else 0) for r in recs]
    out["recall_conservative"] = rate(mk, cons_n)
    out["sensitivity_flush_all_ends"] = {"recall": rate([r["matched_prod"] for r in recs], rin),
                                         "precision": rate([r["matched_prod"] for r in recs], [r["n_rep_prod"] for r in recs]),
                                         "count_ratio": ratio([r["n_rep_prod"] for r in recs], rin)}
    out["sensitivity_tail_1s"] = {"recall": rate([r["matched_tail"] for r in recs], rin),
                                  "precision": rate([r["matched_tail"] for r in recs], [r["n_rep_tail"] for r in recs]),
                                  "engines_ended_in_tail": sum(1 for r in recs if r["tail_ended"])}
    ns = lambda r: [p for p in r["pairs"] if not p["self"]]  # noqa: E731
    out["pairs"] = {"matched": sum(len(r["pairs"]) for r in recs), "self_source": sum(len(r["pairs"]) - len(ns(r)) for r in recs),
                    "approx": sum(1 for r in recs for p in r["pairs"] if p["approx"]),
                    "flushed": sum(1 for r in recs for p in r["pairs"] if p["flushed"])}
    out["exact_not_self"] = rate([sum(p["exact"] for p in ns(r)) for r in recs], [len(ns(r)) for r in recs])
    out["sim09_not_self"] = rate([sum(p["sim"] >= SIM_SAME for p in ns(r)) for r in recs], [len(ns(r)) for r in recs])
    out["sim09_rec_denominator_not_self"] = rate([sum(p["sim_rec"] >= SIM_SAME for p in ns(r)) for r in recs],
                                                 [len(ns(r)) for r in recs])
    out["exact_all_pairs"] = rate([sum(p["exact"] for p in r["pairs"]) for r in recs], [len(r["pairs"]) for r in recs])
    out["sim_not_self"] = dist([p["sim"] for r in recs for p in ns(r)])
    out["d_emit"] = dist([p["d_emit"] for r in recs for p in r["pairs"]])
    out["d_emit_not_flushed"] = dist([p["d_emit"] for r in recs for p in r["pairs"] if not p["flushed"]])
    out["d_emit_not_self"] = dist([p["d_emit"] for r in recs for p in ns(r)])
    out["d_start"] = dist([p["d_start"] for r in recs for p in r["pairs"]])
    out["abs_d_emit_le_0.5"] = rate([sum(abs(p["d_emit"]) <= 0.5 for p in r["pairs"]) for r in recs], [len(r["pairs"]) for r in recs])
    rr, pr = Counter(), Counter()
    for r in recs:
        rr.update(r["rec_reasons"])
        pr.update(r["rep_reasons"])
    out["unmatched_recorded_reasons"] = dict(rr.most_common())
    out["unmatched_replay_reasons"] = dict(pr.most_common())
    ts = [v for r in recs for v in r["ts_dstart"]]
    out["timing_shift_d_start"] = {**dist(ts), "abs_le_2s": sum(1 for v in ts if abs(v) <= 2.0),
                                   "positive": sum(1 for v in ts if v > 0)}
    pe = [v for r in recs for v in r["pend_end"]]
    out["pending_at_recording_end_detail"] = {"n_with_text_candidate": len(pe), "recorded_t_minus_rec_end": dist([v[0] for v in pe]),
                                              "replay_at_minus_rec_end": dist([v[1] for v in pe]),
                                              "real_end_reason": dict(Counter(v[2] for v in pe).most_common())}
    c = [Counter(r["causal"]) for r in recs]
    k_in = [x["in_time"] for x in c]
    n_dec = [x["in_time"] + x["late"] + x["missing"] for x in c]
    out["causal_ai"] = {"in_time": sum(k_in), "late": sum(x["late"] for x in c), "missing": sum(x["missing"] for x in c),
                        "after_stop": sum(x["after_stop"] for x in c), "no_prior": sum(x["no_prior"] for x in c),
                        "rate": rate(k_in, n_dec),
                        "rate_sim09": rate([x["in_time_sim09"] for x in c], n_dec),
                        "rate_conservative": rate(k_in, [n + x["after_stop"] for n, x in zip(n_dec, c)]),
                        # 민감도: 나온 시각을 단락의 예정 프레임(at, run_engine 값보다 한 프레임 앞)으로 볼 때
                        "rate_emit_minus_1frame": rate([x["in_time"] + sum(1 for v in r["late_lag"] if v[0] <= FS + 1e-9)
                                                        for x, r in zip(c, recs)], n_dec),
                        "late_emit_minus_press": dist([v[0] for r in recs for v in r["late_lag"]]),
                        "late_press_minus_recorded_t": dist([v[1] for r in recs for v in r["late_lag"]])}
    return out


def aggregate_all(recs: list[dict], names: dict[str, str]) -> dict:
    out = {"ALL": aggregate(recs)}
    for ph in sorted({r["phone"] for r in recs}):
        sub = [r for r in recs if r["phone"] == ph]
        out[ph] = {"name": names.get(ph), **aggregate(sub)}
    return out


# 기준값 --------------------------------------------------------------------------
def bounds(full: list[dict], v3: dict, call_dirs: dict[str, Path]) -> dict:
    trusted, untrusted = [], Counter()
    for r in full:
        c = v3.get(r["dir"])
        if c is None or c["path"] != r["path"]:
            untrusted["path_differs"] += len(r["pairs"])
            continue
        for p in r["pairs"]:
            seg = next((s for s in c["segs"] if s["from"] <= p["onset"] < s["to"]), None)
            if seg is None:
                untrusted["no_segment"] += 1
            elif not seg["same"]:
                untrusted["segment_not_same"] += 1
            else:
                trusted.append((r["dir"], p))
    texts: dict[str, dict[int, str]] = {}
    for name in sorted({dname for dname, _ in trusted}):
        j = json.loads((call_dirs[name] / "log.json").read_text(encoding="utf-8"))
        texts[name] = {x["idx"]: x["text"] for x in recorded_stt(j)}
    groups = defaultdict(list)
    for dname, p in trusted:
        groups[tuple(p["label"])].append((dname, texts[dname][p["rec_idx"]], p))
    rng = random.Random(PAIR_SEED)
    same_by_label = {}
    for lab in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        lst = groups[lab]
        cand = [(a, b) for a in range(len(lst)) for b in range(a + 1, len(lst)) if lst[a][0] != lst[b][0]]
        if len(cand) > SAME_CAP:
            cand = rng.sample(cand, SAME_CAP)
        if cand:
            same_by_label[lab] = [compare(lst[a][1], lst[b][1]) for a, b in cand]

    def upper(labs):
        ks = [sum(x[0] for x in same_by_label[l]) for l in labs]
        ks9 = [sum(x[1] >= SIM_SAME for x in same_by_label[l]) for l in labs]
        ns = [len(same_by_label[l]) for l in labs]
        sims = [x[1] for l in labs for x in same_by_label[l]]
        # 같은 집합(믿는 단락 이름이 둘 이상 통화에 있는 것)에서 가상 대 원래 (자기 출처 제외)
        rp = [p for l in labs for _, _, p in groups[l] if not p["self"]]
        rk = [sum(p["exact"] for _, _, p in groups[l] if not p["self"]) for l in labs]
        rn = [sum(1 for _, _, p in groups[l] if not p["self"]) for l in labs]
        return {"labels": len(labs), "pairs": sum(ns), "exact": rate(ks, ns), "sim09": rate(ks9, ns), "sim": dist(sims),
                "replay_vs_recorded_same_labels": {"pairs": len(rp), "exact": rate(rk, rn),
                                                   "sim09": rate([sum(p["sim"] >= SIM_SAME for _, _, p in groups[l] if not p["self"])
                                                                  for l in labs], rn)}}

    labs_all = sorted(same_by_label, key=lambda x: tuple(str(v) for v in x))
    up = {"ALL": upper(labs_all)}
    for ph in sorted({l[0] for l in labs_all}):
        up[ph] = upper([l for l in labs_all if l[0] == ph])
    # 하한: 같은 기관 다른 상태
    by_org = defaultdict(list)
    for lab in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        for dname, text, _ in groups[lab]:
            by_org[lab[0]].append((lab[1], dname, text))
    low, low_all = {}, []
    for ph in sorted(by_org):
        lst = by_org[ph]
        vals = []
        for _ in range(DIFF_DRAWS):
            if len(lst) < 2:
                break
            a, b = rng.sample(range(len(lst)), 2)
            if lst[a][0] == lst[b][0]:
                continue
            vals.append(compare(lst[a][2], lst[b][2]))
        low_all += vals
        low[ph] = {"pairs": len(vals), "sim": dist([v[1] for v in vals]),
                   "sim09": rate([sum(v[1] >= SIM_SAME for v in vals)], [len(vals)], boot=False),
                   "exact": rate([sum(v[0] for v in vals)], [len(vals)], boot=False)}
    low["ALL"] = {"pairs": len(low_all), "sim": dist([v[1] for v in low_all]),
                  "sim09": rate([sum(v[1] >= SIM_SAME for v in low_all)], [len(low_all)], boot=False),
                  "exact": rate([sum(v[0] for v in low_all)], [len(low_all)], boot=False)}
    return {"rule": ("상한 = 같은 (기관, 상태, 변형, 단락 번호) 서로 다른 통화의 실통화 받아쓰기 쌍(이름마다 최대 50쌍, random.Random(7) 이름순). "
                     "이름은 표본 내 실행에서 시각으로 맞춘 가상 단락에서 가져오고, check_v3 경로가 같고 그 단락 시작을 담은 구간이 M3 같은 소리인 것만 믿음. "
                     "하한 = 같은 기관 다른 상태 무작위 쌍(기관마다 2,000회, 같은 난수열). 부트스트랩은 단락 이름 군집."),
            "trusted_recorded": len(trusted), "untrusted": dict(untrusted), "labels": len(groups),
            "labels_with_pairs": len(same_by_label), "upper_same_label": up, "lower_diff_state": low}


# 인식 지연 ------------------------------------------------------------------------
def latency(univ_names: set[str], fold_reports: dict) -> dict:
    allv = {i: [] for i in range(4)}
    univv = {i: [] for i in range(4)}
    dirs = 0
    for root in rb.ROOTS:
        for d in Path(root).glob("*"):
            if not (d / "call.wav").exists():
                continue
            try:
                j = json.loads((d / "log.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                j = {}
            if j.get("backend") not in (None, "clawops"):
                continue
            dirs += 1
            for e in j.get("events", []):
                a, b, at = e.get("seg_start_t"), e.get("seg_end_t"), e.get("t")
                if e.get("type") != "stt" or not e.get("text") or None in (a, b, at):
                    continue
                k = lat_bucket(float(b) - float(a) - 0.2 - CLOSE_S)
                allv[k].append(float(at) - float(b))
                if d.name in univ_names:
                    univv[k].append(float(at) - float(b))

    def table(v):
        out = {}
        for i, nm in enumerate(LAT_NAMES):
            pos = [x for x in v[i] if x >= 0]
            out[nm] = {"n": len(v[i]), "negative": len(v[i]) - len(pos),
                       "median_nonneg": round(statistics.median(pos), 2) if pos else None, **{k: dd for k, dd in dist(pos).items() if k != "n"}}
        return out

    mine = table(allv)
    ref = rb.measure_latency(rb.ROOTS, set())
    same = all(mine[nm]["median_nonneg"] == med for nm, (_, med) in zip(LAT_NAMES, ref))
    return {"rule": "measure_latency 와 같음: 소리 길이 = seg_end_t − seg_start_t − 0.2 − 1.2, 지연 = t − seg_end_t(음수는 가운뎃값에서 뺌)",
            "dirs": dirs, "all_recordings": mine, "test_universe_427": table(univv),
            "measure_latency": [[str(u), m] for u, m in ref], "medians_equal_measure_latency": same,
            "bundle_buckets": {k: [[str(u), m] for u, m in v["report"]["_latency_buckets"]] for k, v in fold_reports.items()}}


def rep_latency(recs: list[dict]) -> dict:
    v = {i: [] for i in range(4)}
    for r in recs:
        for sp, lat in r["rep_lat"]:
            v[lat_bucket(sp)].append(lat)
    return {nm: dist(v[i]) for i, nm in enumerate(LAT_NAMES)}


# 실행 ---------------------------------------------------------------------------
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run(out_dir: str = OUT_DEFAULT, workers: int = 6) -> dict:
    t_all = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("ars.replay").setLevel(logging.ERROR)
    full_b = ra.ReplayBundle(FULL_BUNDLE)
    names = {ph: o.get("org") for ph, o in full_b.orgs.items()}
    univ = universe(full_b)
    call_dirs = {d.name: d for d, _ in univ}
    assert len(call_dirs) == len(univ), "폴더 이름 중복"
    fold_of = assign_folds(univ)
    folds = {k: sorted(n for n, f in fold_of.items() if f == k) for k in range(K)}
    (out / "folds.json").write_text(json.dumps({
        "rule": "기관(번호순)마다 이름순 정렬 → 새 random.Random(20261109).shuffle → 기관을 이어 i % 5",
        "seed": FOLD_SEED, "k": K, "folds": folds, "sizes": {k: len(v) for k, v in folds.items()},
        "per_org": {ph: [sum(1 for n in folds[k] if call_dirs[n].name.split("_")[1] == ph) for k in range(K)]
                    for ph in sorted({p for _, p in univ})}}, ensure_ascii=False, indent=1), encoding="utf-8")

    # 묶음 만들기 (겹 5개 + 전체 다시 만들기)
    jobs = {f"fold{k}": (str(out / f"bundle_fold{k}"), folds[k]) for k in range(K)}
    jobs["full_rebuild"] = (str(out / "bundle_full_rebuild"), [])
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        builds = dict(zip(jobs, ex.map(_build_job, jobs.values())))
    print("builds", {k: v["sec"] for k, v in builds.items()}, flush=True)
    checks: dict = {}
    a = json.loads((Path(FULL_BUNDLE) / "bundle.json").read_text(encoding="utf-8"))
    b = json.loads((out / "bundle_full_rebuild" / "bundle.json").read_text(encoding="utf-8"))
    fa = sorted(p.relative_to(FULL_BUNDLE).as_posix() for p in Path(FULL_BUNDLE).rglob("*.ulaw"))
    fb = sorted(p.relative_to(out / "bundle_full_rebuild").as_posix() for p in (out / "bundle_full_rebuild").rglob("*.ulaw"))
    checks["full_rebuild_bundle_json_equal"] = a == b
    checks["full_rebuild_clips_equal"] = fa == fb and all(
        (Path(FULL_BUNDLE) / x).read_bytes() == (out / "bundle_full_rebuild" / x).read_bytes() for x in fa)
    fold_no_clip = {}
    for k in range(K):
        bb = json.loads((out / f"bundle_fold{k}" / "bundle.json").read_text(encoding="utf-8"))
        ex_set = set(folds[k])
        leaks = [(ph, sid) for ph, o in bb["orgs"].items() for sid, s in o["states"].items()
                 if s.get("clip_from") in ex_set or any(v.get("from") in ex_set for v in s.get("variants") or [])]
        assert not leaks, f"fold {k} 조각 누설 {leaks[:3]}"
        fold_no_clip[k] = {ph: r["no_clip"] for ph, r in builds[f"fold{k}"]["report"].items()
                           if isinstance(r, dict) and r.get("no_clip")}
    checks["fold_clip_leak"] = 0
    kf = out.parent / "kfold" / "folds.json"  # 같은 겹 규칙을 쓰는 replay_validate_kfold 의 출력(옆 kfold 폴더)이 있으면 겹 비교
    if kf.exists():
        try:
            f = json.loads(kf.read_text(encoding="utf-8")).get("folds")
            f = [f[str(i)] for i in range(K)] if isinstance(f, dict) else f
            checks["folds_equal_kfold_module"] = all(set(folds[i]) == set(f[i]) for i in range(K))
        except Exception as e:  # noqa: BLE001
            checks["folds_equal_kfold_module"] = f"비교 실패: {type(e).__name__}"

    # 엔진 실행
    tasks = [(FULL_BUNDLE, "full", str(d), ph) for d, ph in univ]
    tasks += [(str(out / f"bundle_fold{fold_of[d.name]}"), f"fold{fold_of[d.name]}", str(d), ph) for d, ph in univ]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init_worker) as ex:
        res = list(ex.map(eval_call, tasks, chunksize=4))
    print("engine runs", len(res), round(time.time() - t0, 1), "s", flush=True)
    full = [r for r in res if r["tag"] == "full"]
    held = [r for r in res if r["tag"] != "full"]

    # 검사: 관찰용 엔진 기록과 run_engine 받아쓰기, 전체 묶음 실행과 CHECK_V3 의 경로·받아쓰기 대조
    v3raw = json.loads(Path(CHECK_V3).read_text(encoding="utf-8"))
    v3 = {c["dir"]: c for c in v3raw["calls"]}
    checks["trace_equals_run_engine"] = {"k": sum(r["trace_ok"] for r in res), "n": len(res)}
    checks["v3_path_equal"] = {"k": sum(1 for r in full if r["dir"] in v3 and v3[r["dir"]]["path"] == r["path"]), "n": len(full)}
    checks["v3_transcripts_equal"] = {"k": sum(1 for r in full if r["dir"] in v3 and
                                               [[t["start"], t["at"], t["chars"]] for t in v3[r["dir"]]["transcripts"]] == r["v3_texts"]),
                                      "n": len(full)}
    checks["v3_calls"] = len(v3)
    checks["held_out_self_source_scheduled"] = sum(r["self_sched"] for r in held)
    checks["held_out_self_source_pairs"] = sum(1 for r in held for p in r["pairs"] if p["self"])

    result = {
        "module": "replay_validate_transcripts", "method_section": "6.5",
        "code_sha256_16": {f: _sha(Path(__file__).resolve().parent / f) for f in
                           ("replay_ars.py", "replay_build.py", "replay_check.py", "replay_validate_transcripts.py")},
        "params": {"tolerances_s": TOLS, "main_tol_s": MAIN_TOL, "detect_lag_s": LAG, "sim_same": SIM_SAME, "sim_candidate": SIM_CAND,
                   "candidate_window_s": CAND_WINDOW_S, "similarity": "1 − Levenshtein/max(len), 공백·문장부호 제거(가-힣0-9A-Za-z)",
                   "flush_primary": "엔진이 note 없이 끝난 경우만 flush, 나온 시각 = stop", "flush_sensitivity": "엔진이 끝난 모든 경우 flush",
                   "oracle": True, "seed": "md5(폴더 이름)[:8]", "fold_seed": FOLD_SEED, "k": K,
                   "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "unit": "통화"}, "pair_seed": PAIR_SEED, "same_cap": SAME_CAP,
                   "diff_draws": DIFF_DRAWS},
        "universe": {"calls": len(univ), "per_org": dict(sorted(Counter(ph for _, ph in univ).items())), "names": names},
        "folds": {"sizes": {k: len(v) for k, v in folds.items()}, "file": str(out / "folds.json"),
                  "no_clip_states": fold_no_clip, "build_sec": {k: v["sec"] for k, v in builds.items()}},
        "checks": checks,
        "held_out_5fold": aggregate_all(held, names),
        "in_sample_upper": aggregate_all(full, names),
    }
    per_fold = {}
    for k in range(K):
        g = aggregate([r for r in held if r["tag"] == f"fold{k}"])
        per_fold[f"fold{k}"] = {"calls": g["calls"], "recall": g["recall"], "precision": g["precision"],
                                "exact_not_self": g["exact_not_self"], "causal_ai": g["causal_ai"]["rate"]}
    result["held_out_per_fold"] = per_fold
    result["bounds"] = bounds(full, v3, call_dirs)
    result["latency_recorded"] = latency(set(call_dirs), builds)
    result["latency_replay_emit_minus_close"] = {"in_sample": rep_latency(full), "held_out": rep_latency(held)}
    da = [x for r in res for x in r["drop_audible"]]
    result["diagnostics"] = {
        "dropped_by_key_then_audible_ge_1s": {
            "in_sample": {"n": sum(len(r["drop_audible"]) for r in full),
                          "recorded_has_it": sum(x["recorded_has_it"] for r in full for x in r["drop_audible"]),
                          "not_started_at_drop": sum(x["not_started_at_drop"] for r in full for x in r["drop_audible"]),
                          "onset_to_drop_s": [x["onset_to_drop_s"] for r in full for x in r["drop_audible"]],
                          "calls": sum(1 for r in full if r["drop_audible"])},
            "held_out": {"n": sum(len(r["drop_audible"]) for r in held),
                         "recorded_has_it": sum(x["recorded_has_it"] for r in held for x in r["drop_audible"]),
                         "not_started_at_drop": sum(x["not_started_at_drop"] for r in held for x in r["drop_audible"]),
                         "calls": sum(1 for r in held if r["drop_audible"])},
            "by_org_state_in_sample": dict(Counter(f"{r['phone']}:{x['state']}" for r in full for x in r["drop_audible"]).most_common()),
            "audible_s": dist([x["audible_s"] for x in da])},
        "scheduled_fates": {"in_sample": dict(sum((Counter({k: v for k, v in r["sched"].items() if k != "total"}) for r in full), Counter())),
                            "held_out": dict(sum((Counter({k: v for k, v in r["sched"].items() if k != "total"}) for r in held), Counter()))},
    }
    result["per_call"] = [{k: r[k] for k in ("tag", "dir", "phone", "start", "end_s", "stop_s", "note_kind", "n_rec_in", "n_rec_after",
                                             "n_rep", "matched", "matched_prod", "rec_reasons", "rep_reasons", "causal")}
                          | {"exact_not_self": sum(p["exact"] for p in r["pairs"] if not p["self"]),
                             "pairs_not_self": sum(1 for p in r["pairs"] if not p["self"])} for r in res]
    result["runtime_s"] = round(time.time() - t_all, 1)
    s = json.dumps(result, ensure_ascii=False, indent=1, default=str)
    assert '"text"' not in s
    (out / "result.json").write_text(s, encoding="utf-8")

    H, I = result["held_out_5fold"]["ALL"], result["in_sample_upper"]["ALL"]
    return {
        "calls": len(univ), "checks": checks,
        "held_out": {"recall": H["recall"], "precision": H["precision"], "count_ratio": H["count_ratio"]["ratio"],
                     "exact_not_self": H["exact_not_self"], "sim09_not_self": H["sim09_not_self"],
                     "d_emit": H["d_emit"], "d_start": H["d_start"], "causal_ai": H["causal_ai"]["rate"],
                     "causal_ai_emit_minus_1frame": H["causal_ai"]["rate_emit_minus_1frame"]["rate"],
                     "recall_by_tol": {t: v["recall"]["rate"] for t, v in H["by_tol"].items()},
                     "recall_tail_1s": H["sensitivity_tail_1s"]["recall"]["rate"],
                     "recall_flush_all_ends": H["sensitivity_flush_all_ends"]["recall"]["rate"],
                     "unmatched_recorded": H["unmatched_recorded_reasons"]},
        "in_sample": {"recall": I["recall"], "precision": I["precision"], "exact_not_self": I["exact_not_self"],
                      "causal_ai": I["causal_ai"]["rate"], "unmatched_recorded": I["unmatched_recorded_reasons"]},
        "bounds": {"upper_exact": result["bounds"]["upper_same_label"]["ALL"]["exact"],
                   "replay_exact_same_labels": result["bounds"]["upper_same_label"]["ALL"]["replay_vs_recorded_same_labels"]["exact"],
                   "lower_sim": result["bounds"]["lower_diff_state"]["ALL"]["sim"]},
        "latency_medians_equal": result["latency_recorded"]["medians_equal_measure_latency"],
        "runtime_s": result["runtime_s"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    print(json.dumps(run(a.out, a.workers), ensure_ascii=False, indent=1, default=str))
