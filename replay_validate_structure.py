"""
녹음 기반 가상 ARS 검증: 구조 일반화(leave-one-out·시간 분할)와 지연 일반화

  python replay_validate_structure.py            # 결과: OUT_DIR/result.json

측정 대상과 절차
  0. 엔진 실행: 녹음 통화(replay_check.calls 와 같은 선정 규칙: 폴더 이름의 기관 번호가 묶음에 있고, log.json 의
     backend 가 clawops 또는 기록 없음이며, call.wav 가 1초 이상)마다 replay_check.run_engine 으로 전체 묶음(BUNDLE)을
     실행. 입력 규칙은 replay_check.check_call 과 같음: 첫 상태는 폴더 이름의 날짜·시각,
     키 시각은 keys_sent 의 t + 0.25×i + 0.15초, 민감정보 입력은 자릿수만큼 '0' 과 덧붙인 '#', 변형은 원래 통화와 가장 닮은
     것(oracle), 씨앗은 폴더 이름의 md5 앞 8자리. 엔진은 상태 진입 원인만 기록하는 하위 클래스(동작 변경 없음)로 교체하며,
     얻은 path 가 replay_check.py --out 으로 저장한 전체 묶음 검사 결과(CHECK_INSAMPLE)의 path 와 같은지 통화마다 대조.
  1. 요소: 엔진이 멈추기 전까지 지난 (기관, 시작 상태), (기관, 상태), (기관, 출발 상태, 원인, 도착 상태). 원인은 도착 상태로 들어간
     까닭(via)이며 키 연결은 실제로 적용된 키를 붙임(key:<키>, 두 자리 키는 key:77 처럼). 그 밖의 원인은 timeout·third_fail·
     input·digit_timeout·invalid. ARS 끊음(hangup)으로 가는 연결과 개인정보 상태로 들어가는 연결·상태도 요소로 집계. 적용된
     키는 path 의 키 기록에서 미리 누른 키(type_ahead 표시가 바로 뒤따른 기록)를 뺀 뒤 끝의 한 자리 또는 두 자리로도 다시 구해,
     하위 클래스가 기록한 키와 같은지 대조. 요소의 지지 집합은 그 요소를 지난 통화의 집합.
  2. 일점 제외(leave-one-out, LOO: 자기 통화를 뺀 나머지 통화만으로 지지 판정): 지지 집합이 자기 통화 하나뿐인 요소를 하나라도
     지나는 통화를 'LOO 녹음 밖' 으로 판정. 기관별 k/n 과 Wilson 95% 구간, 첫 단독 요소의 종류, 재현 시간 보존 비율
     = Σ LOO 정지 시각 / Σ 표본 내 정지 시각을 산출. LOO 정지 시각은 첫 단독 요소의 원인 시각(키·입력·잘못 누른 키 연결은 엔진이 그 키를 받은
     시각, 시간 초과류는 다음 상태 진입 시각, 시작 상태는 0초)과 표본 내 정지 시각 중 이른 쪽이며, 상태 진입 시각 기준 값도
     함께 기록. 민감도 분석으로, CHECK_INSAMPLE 에서 M3(replay_check 의 MFCC DTW 유사도) < 0.90 인 첫 구간의 시작 0.05초 전
     이후의 요소를 지지와 판정에서 모두 뺀 값을 함께 산출.
  3. 시간 분할: 학습 = TRAIN_DATES 의 통화, 시험 = TEST_DATES 의 통화. 학습·시험 통화가 모두 있는 기관의 시험 통화마다 모든
     요소가 학습 통화 하나 이상으로 지지되는지 판정. 기관별 덮임 k/n·Wilson 구간, 첫 미지지 요소 종류, 재현 시간 보존 비율을
     내고, 학습 통화가 없는 기관과 시험 통화가 없는 기관을 기록.
  4. 지연 일반화: 키·입력 연결(via = key·input)마다 원래 녹음에서 지연 측정. 기준점은 그 연결을 일으킨 키가 속한 전송(keys_sent
     하나)의 마지막 자리 톤 끝, 시작점은 기준점 +0.1초 이후에 시작하는 첫 말소리 덩어리(replay_check.islands: 10ms RMS 100 초과,
     0.15초 미만 쉼은 이어 붙임, 0.05초 미만은 버림, 보낸 키 톤 구간 '톤 끝 −0.17~+0.05초' 는 지움). 창은 [기준점, 다음 전송
     시작·기준점 +8초·통화 끝 중 이른 때). 기준점 +0.1초 전에 시작한 덩어리(앞 안내의 꼬리)가 기준점 0.6초 뒤까지 이어지면
     잴 수 없는 것으로 제외. 가상 ARS 출력에도 같은 방법을 쓰며 창은 엔진 정지 시각에서 더 자름. 연결 무리 = (기관, 출발 상태,
     원인, 도착 상태). 보고: (a) 실제 ARS 의 연결 내 흩어짐 = 서로 다른 통화가 2건 이상인 무리에서 |x − 무리 가운뎃값| 의
     가운뎃값·p90, (b) 일점 제외 가운뎃값 예측 오차 = x − (같은 통화의 기록을 모두 뺀 무리 가운뎃값), (c) 시험일 기록의 학습일
     가운뎃값 예측 오차 = x − (학습 통화 기록의 무리 가운뎃값), (d) 가상 ARS 오차 = 가상 시작점 − 원래 시작점(양쪽 모두 잰 기록)의
     편향(부호 있는 가운뎃값)·|오차| 가운뎃값·p90, (e) 끼어들기(원래 녹음에서 전송 첫 톤 시작 0.02초 전까지 말소리 덩어리가 이어짐)
     여부별 값, (f) 측정 정의 확인 = 끼어들기가 아닌 키 연결 무리의 측정 가운뎃값 − 묶음 delay_s 가운뎃값(spec 녹음 분석 값).
     끼어들기 여부로 나눈 무리의 값도 함께 기록.
  5. 보고 규칙: 통화 단위 비율은 k/n·Wilson 95% 구간(replay_check.wilson). 지연 통계와 보존 비율은 통화 단위 군집
     부트스트랩 95% 구간(통화를 복원 추출, 1,000회, 씨앗 7)을 함께 기록. 부트스트랩은 측정값·예측값을 고정한 채 통화만 다시
     뽑으며 무리 가운뎃값은 재계산하지 않음. 가운뎃값·p90 은 numpy 선형 보간 분위수. 통화 수가 10건 미만인 기관은
     근거 미약으로 표시.

한계: 묶음의 상태 구분·같은 상태 판정·시작 규칙·끼어들기 멈춤·입력 결과의 최다 선택·대표 조각 선택·지연 가운뎃값은 모든 녹음
통화로 정하고, 요소의 지지 집합은 실제 경로가 아니라 엔진 경로로 만들며, oracle 변형은 시험 통화 자신의 소리를 사용. 따라서
LOO·시간 분할 덮임은 처음 보는 통화에 대한 낙관적 상한(녹음 밖 비율의 하한)이고, 가상 ARS 지연 오차는 표본 내 값.

개인정보: 받아쓰기 문장은 읽지도 쓰지도 않음(엔진이 내는 받아쓰기는 바로 폐기). 출력의 통화 식별자는 기관 약칭·날짜·시각만
사용. CPU 만 쓰며 실행마다 같은 결과.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_check as rc  # noqa: E402
from replay_ars import FRAME_S, KEY_TONE_ON_S, ReplayBundle, ReplayEngine  # noqa: E402

BUNDLE = r"C:\Users\User\Documents\ars_run\replay\bundle"
CHECK_INSAMPLE = r"C:\Users\User\Documents\ars_run\replay\check_v3_oracle.json"
OUT_DIR = r"C:\Users\User\Documents\ars_run\replay\validate\structure"
TRAIN_DATES = ("2026-11-04", "2026-11-05", "2026-11-06", "2026-11-07")
TEST_DATES = ("2026-11-08", "2026-11-09")
ONSET_AFTER_S = 0.1      # 기준점 이후 이 시간 뒤에 시작하는 첫 덩어리를 다음 안내의 시작으로 간주
TAIL_MAX_S = 0.6         # 앞 안내 꼬리가 기준점 뒤 이 시간을 넘어 이어지면 잴 수 없음
WINDOW_S = 8.0           # 기준점 뒤 창 길이 상한
BOOT_N, BOOT_SEED = 1000, 7
SAME_M3 = rc.SAME_M3
ORG_SHORT = {"15889999": "KB", "15771000": "NHIS", "15991111": "HANA", "15881300": "POST", "15880075": "COMWEL",
             "15776000": "HYUNDAI", "15882188": "GOV24", "15882588": "IBK", "15991004": "SEVERANCE"}
TIMEOUT_VIAS = ("timeout", "third_fail", "digit_timeout")
KEY_VIAS = ("key", "invalid", "input")


# 원인 기록 엔진 (동작 변경 없음) ----------------------------------------------------
class TraceEngine(ReplayEngine):
    """ReplayEngine 과 같은 동작에, 상태 진입마다 원인(적용된 키·키를 받은 시각·끼어들기 판정)을 trace 에 기록."""

    def __init__(self, *a, **kw):
        self.trace: list[dict] = []
        self.cur_key = None                  # 지금 넣는 키 (replay_check.run_engine 의 on_key 가 설정)
        self.speaking_at_key: dict = {}      # 키 수신 시각 → 그 키를 받기 직전 안내 중이었는지
        self.timeout_while_pending = 0       # 키 연결 진입을 기다리는 중 시간 초과가 먼저 일어난 횟수 (진단)
        self._applied = None
        self._enter_cause: dict = {}
        self._timeout_cause: dict = {}
        super().__init__(*a, **kw)

    def _apply_key(self, ch, barged=False):
        self._applied = (ch, bool(barged))
        super()._apply_key(ch, barged)

    def _schedule(self, after_s, what, target=None, flag=None, via=None):
        super()._schedule(after_s, what, target, flag=flag, via=via)
        if what == "enter" and via in KEY_VIAS:
            ch, barged = self._applied if (via != "input" and self._applied) else (None, None)
            self._enter_cause[self._gen] = {"key": ch, "barged": barged, "key_t": self.t,
                                            "true": None if self.cur_key is None else list(self.cur_key)}
        elif what == "timeout":
            self._timeout_cause[self._gen] = {"sched_t": self.t}

    def _enter(self, state_id, via="key"):
        prev = self.state_id
        cause = {}
        if via in KEY_VIAS:
            cause = self._enter_cause.get(self._gen, {})
        elif via in TIMEOUT_VIAS:
            cause = self._timeout_cause.get(self._gen, {})
            if self._pending_enter:
                self.timeout_while_pending += 1
        rec = {"t": self.t, "from": prev, "to": state_id, "via": via}
        rec.update(cause)
        self.trace.append(rec)
        super()._enter(state_id, via)


def _feed_hook(eng, key) -> None:
    eng.cur_key = key
    eng.speaking_at_key[round(key[0], 3)] = bool(eng.speaking)


def call_id(name: str, phone: str) -> str:
    m = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})((?:-\d+)?)$", name)
    return f"{ORG_SHORT.get(phone, 'ORG')}_{m.group(2)}{m.group(3)}T{m.group(4)}{m.group(5)}{m.group(6)}{m.group(7)}"


# 지연 측정 -----------------------------------------------------------------------
def onset(pcm: np.ndarray, ref: float, lim: float, mask) -> tuple[float | None, str, float | None, bool]:
    """(지연 = 첫 덩어리 시작 − 기준점, 상태, 앞 안내 꼬리 끝 − 기준점, 짧은 고립 덩어리 여부). 상태: ok / tail(꼬리 > 0.6초) /
    no_onset / short_window. 짧은 고립 덩어리(기준점 0.6초 안에 시작, 0.25초 미만, 다음 덩어리와 0.4초 넘게 떨어짐)는 판정에
    쓰지 않고 개수만 집계(참고용 규칙)."""
    if lim - ref < 0.15:
        return None, "short_window", None, False
    isl = rc.islands(pcm, ref, lim, mask)
    tail = max((e for s, e, _ in isl if s < ref + ONSET_AFTER_S), default=None)
    tail_d = None if tail is None else round(tail - ref, 3)
    if tail_d is not None and tail_d > TAIL_MAX_S:
        return None, "tail", tail_d, False
    cand = [(s, e) for s, e, _ in isl if s >= ref + ONSET_AFTER_S]
    if cand:
        s, e = cand[0]
        short = s - ref < 0.6 and e - s < 0.25 and (len(cand) == 1 or cand[1][0] - e > 0.4)
        return round(s - ref, 3), "ok", tail_d, short
    return None, "no_onset", tail_d, False


def speech_before(pcm: np.ndarray, t0: float, mask) -> bool:
    """전송 첫 톤 시작(t0) 0.02초 전까지 말소리 덩어리가 이어졌는지 (원래 녹음의 끼어들기 판정)."""
    if t0 - 0.02 <= 0.05:
        return False
    isl = rc.islands(pcm, max(0.0, t0 - 1.5), t0 - 0.02, mask)
    return bool(isl and isl[-1][1] >= t0 - 0.02 - 0.011)


# 통화 하나 ------------------------------------------------------------------------
_BUNDLE: ReplayBundle | None = None


def _init(bundle_dir: str) -> None:
    global _BUNDLE
    _BUNDLE = ReplayBundle(bundle_dir)


def _process(job: tuple[str, str]) -> dict | None:
    d, phone = Path(job[0]), job[1]
    bundle = _BUNDLE
    try:
        j, pcm = rc.load_call(d)
    except Exception:
        return None
    if j.get("backend") not in (None, "clawops") or len(pcm) < 8000:
        return None
    org = bundle.org_for(phone)
    keys = rc.key_events(j, pcm)
    end_s = len(pcm) / 8000
    mask = rc.key_mask(keys)
    eo = rc.envelope(pcm, mask)
    seed = int(hashlib.md5(d.name.encode("utf-8")).hexdigest()[:8], 16)
    clock = rc.call_clock(d.name)
    if clock is None:
        raise RuntimeError(f"폴더 이름에서 시각을 읽지 못함: {call_id(d.name, phone)}")
    saved = rc.ReplayEngine
    rc.ReplayEngine = TraceEngine  # run_engine 의 입력 규칙을 그대로 쓰기 위해 이 과정 안에서만 교체
    try:
        eng, rep, _srcs, _texts = rc.run_engine(bundle, org, clock, keys, end_s, eo, oracle=True, seed=seed,
                                                on_key=_feed_hook)
    finally:
        rc.ReplayEngine = saved
    del _texts  # 받아쓰기 문장은 사용하지 않음
    stop_s = eng.frame * FRAME_S if eng.ended else end_s
    # 전송(keys_sent 하나) = 같은 전송 끝 시각을 가진 키 묶음
    sends: dict[float, list] = defaultdict(list)
    for k in keys:
        sends[k[3]].append(k)
    send_list = sorted(({"start": min(x[0] for x in v) - KEY_TONE_ON_S, "ref": max(x[0] for x in v), "done": dn,
                         "n": len(v)} for dn, v in sends.items()), key=lambda s: s["start"])
    by_done = {s["done"]: s for s in send_list}
    starts = [s["start"] for s in send_list]
    timing = []
    for tr in eng.trace:
        if tr["via"] not in ("key", "input") or not tr.get("true"):
            continue
        t_recv, _ch, _sec, done = tr["true"]
        s = by_done.get(done)
        if s is None:
            continue
        ref = s["ref"]
        nxt = [x for x in starts if x > s["start"] + 1e-6]
        lim = min([end_s, ref + WINDOW_S] + nxt[:1])
        d_o, st_o, tail_o, short_o = onset(pcm, ref, lim, mask)
        d_r, st_r, tail_r, short_r = onset(rep, ref, min(lim, stop_s), mask)
        timing.append({
            "from": tr["from"], "cause": f"key:{tr['key']}" if tr["via"] == "key" else "input", "to": tr["to"],
            "ref": round(ref, 3), "t_key": round(t_recv, 3), "send_digits": s["n"], "entry_t": tr["t"],
            "barged_orig": speech_before(pcm, s["start"], mask),
            "barged_eng_applied": tr.get("barged"),
            "barged_eng_first": eng.speaking_at_key.get(round(min(x[0] for x in sends[done]), 3)),
            "d_orig": d_o, "status_orig": st_o, "tail_orig": tail_o, "short_orig": short_o,
            "d_rep": d_r, "status_rep": st_r, "tail_rep": tail_r, "short_rep": short_r, "win": round(lim - ref, 2)})
    # 진단: 시간 초과 진입 뒤 1초 안에 녹음된 키가 온 경우, 원래 녹음에서 보낸 키 톤 시작 전 말소리 시작 여부 (가림 없이)
    racing = []
    for tr in eng.trace:
        if tr["via"] in TIMEOUT_VIAS:
            nk = [k for k in keys if tr["t"] - 0.02 <= k[0] <= tr["t"] + 1.0]
            if nk:
                tone_start = nk[0][0] - KEY_TONE_ON_S
                isl = rc.islands(pcm, max(0.0, tr["t"] - 1.0), tone_start, [])
                racing.append({"from": tr["from"], "via": tr["via"], "to": tr["to"], "entry_t": tr["t"],
                               "key_tone_start": round(tone_start, 3),
                               "orig_onset_before_tone": round(isl[-1][0], 2) if isl else None})
    m = re.search(r"(\d{4})(\d{2})(\d{2})-\d{6}", d.name)
    return {"name": d.name, "id": call_id(d.name, phone), "org": org, "date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
            "weekday": clock.weekday(), "hm": clock.strftime("%H:%M"), "root": d.parent.parent.name,
            "end_s": round(end_s, 2), "stop_s": round(stop_s, 2), "ended": eng.ended, "note": eng.note,
            "personal": bool(eng.note and "개인정보" in eng.note), "start": eng.start_state,
            "path": eng.path, "trace": eng.trace, "timing": timing,
            "key_times": [round(k[0], 3) for k in keys], "timeout_while_pending": eng.timeout_while_pending,
            "racing": racing}


# 요소 ---------------------------------------------------------------------------
def elements(r: dict, states: dict) -> list[dict]:
    """엔진 trace → 시간순 요소 [{el, kind, t_cause, t_entry, into_personal}]. 연결 요소가 그 연결의 상태 요소보다 앞섬."""
    org, out = r["org"], []
    for tr in r["trace"]:
        to, via = tr["to"], tr["via"]
        personal = bool((states.get(to) or {}).get("personal"))
        if via == "start":
            out.append({"el": ("start", org, to), "kind": "start", "t_cause": 0.0, "t_entry": tr["t"], "into_personal": personal})
        else:
            cause = f"key:{tr.get('key')}" if via == "key" else via
            t_cause = tr.get("key_t", tr["t"]) if via in KEY_VIAS else tr["t"]
            out.append({"el": ("edge", org, tr["from"], cause, to), "kind": via, "t_cause": t_cause, "t_entry": tr["t"],
                        "into_personal": personal})
        if to != "hangup" and to in states:
            out.append({"el": ("state", org, to), "kind": "state", "t_cause": tr["t"], "t_entry": tr["t"],
                        "into_personal": personal})
    return out


def path_elements(r: dict, states: dict) -> list[tuple]:
    """같은 요소를 path 만으로 다시 계산 (ARS 끊음 연결 제외). 키 연결의 키 = 출발 상태 진입 뒤 키 기록에서 type_ahead 를 뺀
    끝 두 자리(그 상태의 키이고 도착 상태가 맞으면) 또는 끝 한 자리."""
    org, out, cur, ks = r["org"], [], None, []
    path = r["path"]
    for i, p in enumerate(path):
        if "key" in p:
            nxt = path[i + 1] if i + 1 < len(path) else {}
            if nxt.get("flag") == "type_ahead" and nxt.get("t") == p["t"]:
                continue
            ks.append(p["key"])
        elif "state" in p:
            to, via = p["state"], p["via"]
            if via == "start":
                out.append(("start", org, to))
            else:
                cause = via
                if via == "key":
                    edges = (states.get(cur) or {}).get("keys") or {}
                    two = "".join(ks[-2:]) if len(ks) >= 2 else None
                    if two and two in edges and edges[two].get("next") == to:
                        k = two
                    elif ks and ks[-1] in edges and edges[ks[-1]].get("next") == to:
                        k = ks[-1]
                    else:
                        k = "?"
                    cause = f"key:{k}"
                out.append(("edge", org, cur, cause, to))
            out.append(("state", org, to))
            cur, ks = to, []
    return out


# 통계 ---------------------------------------------------------------------------
def q(xs, p):
    return None if len(xs) == 0 else round(float(np.quantile(np.asarray(xs, dtype=float), p)), 4)


def _boot(values: np.ndarray, calls_idx: np.ndarray, stat) -> list[float] | None:
    """통화 단위 군집 부트스트랩 95% 구간. values 와 같은 길이의 calls_idx(0..m-1)."""
    m = int(calls_idx.max()) + 1 if len(calls_idx) else 0
    if m < 2:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for _ in range(BOOT_N):
        w = np.bincount(rng.integers(0, m, m), minlength=m)[calls_idx]
        xs = np.repeat(values, w)
        if len(xs):
            out.append(stat(xs))
    return [round(float(np.quantile(out, 0.025)), 4), round(float(np.quantile(out, 0.975)), 4)]


def _index(calls: list) -> np.ndarray:
    ids = {c: i for i, c in enumerate(dict.fromkeys(calls))}
    return np.array([ids[c] for c in calls], dtype=np.int64)


def sdist(pairs: list[tuple[str, float]], boot: bool = True) -> dict:
    """부호 있는 오차 [(통화, 값)] → n, 통화 수, 편향(가운뎃값), |오차| 가운뎃값·p90 과 군집 부트스트랩 구간."""
    pairs = [(c, x) for c, x in pairs if x is not None]
    if not pairs:
        return {"n": 0}
    v = np.array([x for _, x in pairs], dtype=float)
    a = np.abs(v)
    out = {"n": len(v), "calls": len({c for c, _ in pairs}), "bias_median": q(v, 0.5), "abs_median": q(a, 0.5),
           "abs_p90": q(a, 0.9), "share_abs_le_0.1": round(float(np.mean(a <= 0.1)), 4),
           "share_abs_le_0.2": round(float(np.mean(a <= 0.2)), 4)}
    if boot:
        ci = _index([c for c, _ in pairs])
        out["boot95_bias"] = _boot(v, ci, lambda x: np.median(x))
        out["boot95_abs_median"] = _boot(a, ci, lambda x: np.median(x))
        out["boot95_abs_p90"] = _boot(a, ci, lambda x: np.quantile(x, 0.9))
    return out


def adist(pairs: list[tuple[str, float]]) -> dict:
    """크기 값 [(통화, 값)] → 가운뎃값·p90 과 군집 부트스트랩 구간."""
    pairs = [(c, x) for c, x in pairs if x is not None]
    if not pairs:
        return {"n": 0}
    v = np.array([x for _, x in pairs], dtype=float)
    ci = _index([c for c, _ in pairs])
    return {"n": len(v), "calls": len({c for c, _ in pairs}), "median": q(v, 0.5), "p90": q(v, 0.9),
            "max": round(float(v.max()), 4), "boot95_median": _boot(v, ci, lambda x: np.median(x)),
            "boot95_p90": _boot(v, ci, lambda x: np.quantile(x, 0.9))}


def ratio_boot(nums: list[float], dens: list[float]) -> list[float] | None:
    """Σnum/Σden 의 통화 단위 군집 부트스트랩 95% 구간 (한 통화 = 한 항)."""
    n = len(nums)
    if n < 2:
        return None
    a, b = np.asarray(nums, float), np.asarray(dens, float)
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        s = b[idx].sum()
        if s > 0:
            out.append(a[idx].sum() / s)
    return [round(float(np.quantile(out, 0.025)), 4), round(float(np.quantile(out, 0.975)), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": rc.wilson(k, n),
            **({"note": "근거 미약(n<10)"} if n < 10 else {})}


# 본 실행 -------------------------------------------------------------------------
def run(out_dir: str = OUT_DIR, workers: int = 6) -> dict:
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = ReplayBundle(BUNDLE)
    jobs = [(str(d), ph) for d, ph in rc.calls(None, bundle)]
    if workers and workers > 1:
        with Pool(workers, initializer=_init, initargs=(BUNDLE,)) as pool:
            recs = pool.map(_process, jobs, chunksize=4)
    else:
        _init(BUNDLE)
        recs = [_process(jb) for jb in jobs]
    recs = sorted((r for r in recs if r is not None), key=lambda r: (r["root"], r["name"]))
    names = Counter(r["name"] for r in recs)
    if any(v > 1 for v in names.values()):
        raise RuntimeError("녹음 폴더 이름이 겹침")
    states = {o: v["states"] for o, v in bundle.orgs.items()}
    orgname = {o: v["org"] for o, v in bundle.orgs.items()}
    orgs = sorted({r["org"] for r in recs}, key=lambda o: (-sum(r["org"] == o for r in recs), o))
    res: dict = {"module": "replay_validate_structure", "method_section": "6.4 (rows 4-6), 6.8",
                 "bundle": BUNDLE, "calls": len(recs), "calls_by_org": {orgname[o]: sum(r["org"] == o for r in recs) for o in orgs},
                 "calls_by_date": dict(sorted(Counter(r["date"] for r in recs).items()))}

    # 0. 저장된 검사 결과(CHECK_INSAMPLE)와 경로 대조 ------------------------------------------
    ins = {c["dir"]: c for c in json.loads(Path(CHECK_INSAMPLE).read_text(encoding="utf-8"))["calls"]}
    mism = [r["id"] for r in recs if r["name"] not in ins or ins[r["name"]]["path"] != r["path"]
            or ins[r["name"]]["offmap"] != r["note"] or ins[r["name"]]["start"] != r["start"]]
    res["path_equal_insample_check"] = {"equal": len(recs) - len(mism), "n": len(recs), "mismatch_ids": mism}
    # trace 요소와 path 로 다시 구한 요소의 대조 (ARS 끊음 연결 제외)
    els = {r["name"]: elements(r, states[r["org"]]) for r in recs}
    agree = 0
    disagree = []
    for r in recs:
        a = [e["el"] for e in els[r["name"]] if not (e["el"][0] == "edge" and e["el"][4] == "hangup")]
        b = path_elements(r, states[r["org"]])
        if a == b:
            agree += 1
        else:
            disagree.append(r["id"])
    res["trace_vs_path_elements"] = {"equal": agree, "n": len(recs), "mismatch_ids": disagree}
    res["bundle_key_edges"] = {"total": sum(len(st.get("keys") or {}) for o in states for st in states[o].values()),
                               "n_zero_or_missing": sum(1 for o in states for st in states[o].values()
                                                        for e in (st.get("keys") or {}).values() if not e.get("n")),
                               "n_one": sum(1 for o in states for st in states[o].values()
                                            for e in (st.get("keys") or {}).values() if e.get("n") == 1)}
    res["engine_outcome"] = {"full": sum(1 for r in recs if not (r["ended"] and r["stop_s"] < r["end_s"] - 1.0)),
                             "personal_end": sum(1 for r in recs if r["personal"]),
                             "offmap_end": sum(1 for r in recs if r["note"] and not r["personal"]),
                             "hangup_end": sum(1 for r in recs if r["ended"] and not r["note"])}
    res["diagnostics"] = {
        "timeout_fired_while_key_enter_pending": sum(r["timeout_while_pending"] for r in recs),
        "timeouts_total": sum(1 for r in recs for tr in r["trace"] if tr["via"] in TIMEOUT_VIAS),
        "timeouts_with_recorded_key_within_1s_after": [dict(x, id=r["id"]) for r in recs for x in r["racing"]],
        "note_racing": "orig_onset_before_tone 이 진입 시각 ±0.2초 안이면 실제 ARS 도 시간 초과 안내를 시작한 것(엔진 경로와 일치)",
    }

    # 1. 요소의 지지 집합 ------------------------------------------------------------------
    support: dict = defaultdict(set)
    for r in recs:
        for e in els[r["name"]]:
            support[e["el"]].add(r["name"])
    census = defaultdict(Counter)
    for el, s in support.items():
        kind = el[0] if el[0] != "edge" else (el[3].split(":")[0])
        census[orgname[el[1]]][f"{kind}|{'n=1' if len(s) == 1 else ('n=2-4' if len(s) <= 4 else 'n>=5')}"] += 1
    res["element_support_census"] = {o: dict(sorted(c.items())) for o, c in census.items()}
    res["elements_total"] = len(support)
    res["elements_singleton"] = sum(1 for s in support.values() if len(s) == 1)

    # 2. LOO ---------------------------------------------------------------------------
    t_bad = {}
    for r in recs:
        segs = (ins.get(r["name"]) or {}).get("segs") or []
        bad = [s["from"] for s in segs if s["m3"] < SAME_M3]
        t_bad[r["name"]] = (min(bad) - 0.05) if bad else float("inf")

    def loo_eval(cut: bool) -> tuple[dict, list]:
        sup = support
        if cut:
            sup = defaultdict(set)
            for r in recs:
                for e in els[r["name"]]:
                    if e["t_cause"] < t_bad[r["name"]]:
                        sup[e["el"]].add(r["name"])
        rows = []
        for r in recs:
            lim = min(t_bad[r["name"]], r["stop_s"]) if cut else r["stop_s"]
            cand = [e for e in els[r["name"]] if (e["t_cause"] < lim or not cut)]
            first = next((e for e in cand if sup[e["el"]] == {r["name"]}), None)
            stop_in = lim
            rows.append({"id": r["id"], "name": r["name"], "org": r["org"], "offmap": first is not None,
                         "first_kind": first["kind"] if first else None,
                         "first_el": list(first["el"][2:]) if first else None,
                         "first_into_personal": bool(first and first["into_personal"]),
                         "stop_in": round(stop_in, 2),
                         "stop_loo_cause": round(min(stop_in, first["t_cause"]), 2) if first else round(stop_in, 2),
                         "stop_loo_entry": round(min(stop_in, first["t_entry"]), 2) if first else round(stop_in, 2)})
        summ = {}
        for org in orgs + [None]:
            L = [x for x in rows if org is None or x["org"] == org]
            off = [x for x in L if x["offmap"]]
            off_np = [x for x in off if not x["first_into_personal"]]
            den = [x["stop_in"] for x in L]
            summ[orgname[org] if org else "ALL"] = {
                "loo_offmap": rate(len(off), len(L)),
                "loo_offmap_excluding_first_into_personal": rate(len(off_np), len(L)),
                "first_singleton_kind": dict(Counter(x["first_kind"] for x in off)),
                "first_into_personal": sum(x["first_into_personal"] for x in off),
                "kept_time_share_cause": round(sum(x["stop_loo_cause"] for x in L) / sum(den), 4) if sum(den) else None,
                "kept_time_share_cause_boot95": ratio_boot([x["stop_loo_cause"] for x in L], den),
                "kept_time_share_entry": round(sum(x["stop_loo_entry"] for x in L) / sum(den), 4) if sum(den) else None,
                "offmap_stop_frac_of_insample": {"median": q([x["stop_loo_cause"] / x["stop_in"] for x in off if x["stop_in"] > 0], 0.5),
                                                 "p90": q([x["stop_loo_cause"] / x["stop_in"] for x in off if x["stop_in"] > 0], 0.9)},
            }
        return summ, rows

    loo_summ, loo_rows = loo_eval(False)
    res["loo"] = loo_summ
    res["loo_validated_prefix_sensitivity"], _ = loo_eval(True)
    res["loo_validated_prefix_sensitivity"]["_calls_with_failing_segment"] = sum(1 for v in t_bad.values() if v != float("inf"))

    # 3. 시간 분할 -------------------------------------------------------------------------
    train = {r["name"] for r in recs if r["date"] in TRAIN_DATES}
    temporal: dict = {"train_dates": list(TRAIN_DATES), "test_dates": list(TEST_DATES), "by_org": {}}
    trows = []
    for org in orgs:
        tr_c = [r for r in recs if r["org"] == org and r["date"] in TRAIN_DATES]
        te_c = [r for r in recs if r["org"] == org and r["date"] in TEST_DATES]
        if not te_c or not tr_c:
            temporal["by_org"][orgname[org]] = {"train": len(tr_c), "test": len(te_c),
                                                "status": "시험 통화 없음" if not te_c else "학습 통화 없음(평가 불가)"}
            continue
        tr_starts = Counter(r["start"] for r in tr_c)
        rows = []
        for r in te_c:
            first = next((e for e in els[r["name"]] if not (support[e["el"]] & train)), None)
            rows.append({"id": r["id"], "org": org, "date": r["date"], "weekday": r["weekday"], "hm": r["hm"],
                         "start": r["start"], "start_in_train": r["start"] in tr_starts,
                         "covered": first is None, "first_kind": first["kind"] if first else None,
                         "first_el": list(first["el"][2:]) if first else None,
                         "stop_in": r["stop_s"],
                         "stop_train": round(min(r["stop_s"], first["t_cause"]), 2) if first else r["stop_s"]})
        trows += rows
        cov = sum(x["covered"] for x in rows)
        temporal["by_org"][orgname[org]] = {
            "train": len(tr_c), "test": len(rows), "covered": rate(cov, len(rows)),
            "first_uncovered_kind": dict(Counter(x["first_kind"] for x in rows if not x["covered"])),
            "start_state_seen_in_train": sum(x["start_in_train"] for x in rows),
            "kept_time_share": round(sum(x["stop_train"] for x in rows) / sum(x["stop_in"] for x in rows), 4),
            "kept_time_share_boot95": ratio_boot([x["stop_train"] for x in rows], [x["stop_in"] for x in rows]),
            "test_by_date": dict(Counter(x["date"] for x in rows))}
    cov = sum(x["covered"] for x in trows)
    temporal["ALL_evaluable"] = {
        "orgs": sorted({orgname[x["org"]] for x in trows}), "covered": rate(cov, len(trows)),
        "first_uncovered_kind": dict(Counter(x["first_kind"] for x in trows if not x["covered"])),
        "kept_time_share": round(sum(x["stop_train"] for x in trows) / sum(x["stop_in"] for x in trows), 4) if trows else None,
        "kept_time_share_boot95": ratio_boot([x["stop_train"] for x in trows], [x["stop_in"] for x in trows])}
    temporal["orgs_without_train_calls"] = {orgname[o]: sum(1 for r in recs if r["org"] == o and r["date"] in TEST_DATES)
                                            for o in orgs if not any(r["org"] == o and r["date"] in TRAIN_DATES for r in recs)}
    temporal["orgs_without_test_calls"] = {orgname[o]: sum(1 for r in recs if r["org"] == o and r["date"] in TRAIN_DATES)
                                           for o in orgs if not any(r["org"] == o and r["date"] in TEST_DATES for r in recs)}
    temporal["first_uncovered_elements"] = dict(Counter(
        f"{ORG_SHORT[x['org']]}:{'/'.join(str(v) for v in x['first_el'])}" for x in trows if not x["covered"]).most_common())
    res["temporal"] = temporal

    # 4. 지연 일반화 -----------------------------------------------------------------------
    recs_t = []
    for r in recs:
        for x in r["timing"]:
            y = dict(x)
            y.update(call=r["name"], org=r["org"], date=r["date"], edge=(r["org"], x["from"], x["cause"], x["to"]))
            recs_t.append(y)
    status = {"orig": dict(Counter(x["status_orig"] for x in recs_t)), "rep": dict(Counter(x["status_rep"] for x in recs_t)),
              "rep_not_ok_with_orig_ok": dict(Counter(
                  "into_personal_state" if (states[x["org"]].get(x["to"]) or {}).get("personal") else "other"
                  for x in recs_t if x["status_orig"] == "ok" and x["status_rep"] != "ok")),
              "short_isolated_first_island": {"orig": sum(1 for x in recs_t if x["status_orig"] == "ok" and x["short_orig"]),
                                              "rep": sum(1 for x in recs_t if x["status_rep"] == "ok" and x["short_rep"])},
              "orig_by_barge": {f"barged={b}": dict(Counter(x["status_orig"] for x in recs_t if x["barged_orig"] == b))
                                for b in (True, False)},
              "rep_by_barge": {f"barged={b}": dict(Counter(x["status_rep"] for x in recs_t if x["barged_orig"] == b))
                               for b in (True, False)}}
    good = [x for x in recs_t if x["status_orig"] == "ok"]
    timing: dict = {"transitions": len(recs_t), "status": status, "used_orig": len(good),
                    "barge_engine_vs_orig": {f"eng={a},orig={b}": v for (a, b), v in
                                             sorted(Counter((bool(x["barged_eng_first"]), x["barged_orig"]) for x in good).items())}}

    def analyse(split: bool) -> dict:
        G = defaultdict(list)
        for x in good:
            G[x["edge"] + ((x["barged_orig"],) if split else ())].append(x)
        acc = defaultdict(lambda: defaultdict(list))
        groups_tab = []
        for k, xs in G.items():
            calls_in = {x["call"] for x in xs}
            vals = [x["d_orig"] for x in xs]
            med = float(np.median(vals))
            tr_vals = [x["d_orig"] for x in xs if x["date"] in TRAIN_DATES]
            med_tr = float(np.median(tr_vals)) if tr_vals else None
            groups_tab.append({"org": ORG_SHORT[k[0]], "from": k[1], "cause": k[2], "to": k[3],
                               **({"barged": k[4]} if split else {}), "records": len(xs), "calls": len(calls_in),
                               "median": round(med, 3), "min": round(min(vals), 3), "max": round(max(vals), 3),
                               "train_records": len(tr_vals)})
            for x in xs:
                others = [y["d_orig"] for y in xs if y["call"] != x["call"]]
                e_loo = x["d_orig"] - float(np.median(others)) if others else None
                for o in (orgname[x["org"]], "ALL"):
                    if len(calls_in) >= 2:
                        acc[o]["within_abs_dev"].append((x["call"], abs(x["d_orig"] - med)))
                    acc[o]["loo_err"].append((x["call"], e_loo))
                    if e_loo is None:
                        acc[o]["loo_undefined"].append((x["call"], 0.0))
                    if x["date"] in TEST_DATES:
                        acc[o]["test_train_err"].append((x["call"], None if med_tr is None else x["d_orig"] - med_tr))
                        if med_tr is None:
                            acc[o]["test_no_train_obs"].append((x["call"], 0.0))
                        acc[o]["test_loo_err"].append((x["call"], e_loo))
                        if x["status_rep"] == "ok" and med_tr is not None:
                            acc[o]["test_emulator_err_same_subset"].append((x["call"], x["d_rep"] - x["d_orig"]))
                            acc[o]["test_train_err_same_subset"].append((x["call"], x["d_orig"] - med_tr))
        res_ = {}
        for o in [orgname[o] for o in orgs] + ["ALL"]:
            d = acc.get(o)
            if not d:
                continue
            res_[o] = {"within_edge_abs_dev": adist(d["within_abs_dev"]),
                       "loo_median_err": sdist(d["loo_err"]), "loo_undefined": len(d["loo_undefined"]),
                       "test_train_median_err": sdist(d["test_train_err"]), "test_no_train_obs": len(d["test_no_train_obs"]),
                       "test_loo_median_err": sdist(d["test_loo_err"]),
                       "test_same_subset": {"emulator_err": sdist(d["test_emulator_err_same_subset"]),
                                            "train_median_err": sdist(d["test_train_err_same_subset"])}}
        return {"by_org": res_, "groups": sorted(groups_tab, key=lambda g: (g["org"], g["from"], g["cause"], g["to"],
                                                                             g.get("barged", False)))}

    timing["groups_pooled"] = analyse(False)
    timing["groups_by_barge"] = analyse(True)
    emu = [x for x in good if x["status_rep"] == "ok"]
    timing["emulator_err"] = {
        "ALL": sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu]),
        "by_org": {orgname[o]: sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu if x["org"] == o]) for o in orgs
                   if any(x["org"] == o for x in emu)},
        "barged_orig": sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu if x["barged_orig"]]),
        "not_barged_orig": sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu if not x["barged_orig"]]),
        "sensitivity_without_short_isolated_first_island": {
            "ALL": sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu if not x["short_orig"]]),
            "barged_orig": sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu if x["barged_orig"] and not x["short_orig"]]),
            "excluded": sum(1 for x in emu if x["short_orig"]),
            "excluded_barged_orig": sum(1 for x in emu if x["short_orig"] and x["barged_orig"])},
        "barge_agree": sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu
                              if bool(x["barged_eng_first"]) == x["barged_orig"]]),
        "barge_disagree": sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu
                                 if bool(x["barged_eng_first"]) != x["barged_orig"]]),
        "by_cause": {"key": sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu if x["cause"] != "input"]),
                     "input": sdist([(x["call"], x["d_rep"] - x["d_orig"]) for x in emu if x["cause"] == "input"])},
        "abs_gt_0.3": sum(1 for x in emu if abs(x["d_rep"] - x["d_orig"]) > 0.3),
        "abs_gt_0.5": sum(1 for x in emu if abs(x["d_rep"] - x["d_orig"]) > 0.5),
        "largest": [{"call": next(r["id"] for r in recs if r["name"] == x["call"]), "edge": f"{x['from']}-{x['cause']}->{x['to']}",
                     "d_orig": x["d_orig"], "d_rep": x["d_rep"], "barged_orig": x["barged_orig"],
                     "barged_eng": x["barged_eng_first"]}
                    for x in sorted(emu, key=lambda x: -abs(x["d_rep"] - x["d_orig"]))[:12]],
    }
    # 측정 정의 확인: 끼어들기가 아닌 무리의 측정 가운뎃값 − 묶음 delay_s 가운뎃값
    G2 = defaultdict(list)
    for x in good:
        if not x["barged_orig"]:
            G2[x["edge"]].append(x["d_orig"])
    diffs = defaultdict(list)
    for (org, src, cause, dst), v in G2.items():
        st = states[org].get(src) or {}
        if cause == "input":
            dd = st.get("input_delay_s")
        else:
            dd = ((st.get("keys") or {}).get(cause[4:]) or {}).get("delay_s")
        sm = dd.get("median") if isinstance(dd, dict) else None
        if isinstance(sm, (int, float)):
            for o in (orgname[org], "ALL"):
                diffs[o].append((f"{org}|{src}|{cause}|{dst}", float(np.median(v)) - sm))
    timing["measured_median_minus_bundle_median"] = {o: sdist(v, boot=False) | {"groups": len(v)} for o, v in diffs.items()}
    # 가상 ARS 지연의 계통 차 분해 (엔진이 끼어들기로 처리하지 않은 키 연결 중 묶음 지연 가운뎃값이 있는 것):
    # 진입 지연 = (상태 진입 − 기준점) − 묶음 가운뎃값 (키를 다음 프레임에서 받는 양자화), 조각 앞 여유 = 가상 시작점 − (진입 − 기준점)
    dec = defaultdict(list)
    for x in good:
        if x["status_rep"] != "ok" or x["cause"] == "input" or x["barged_eng_applied"]:
            continue
        dd = ((states[x["org"]][x["from"]].get("keys") or {}).get(x["cause"][4:]) or {}).get("delay_s")
        sm = dd.get("median") if isinstance(dd, dict) else None
        if not isinstance(sm, (int, float)):
            continue
        rel = x["entry_t"] - x["ref"]
        dec["entry_minus_bundle_median"].append((x["call"], rel - sm))
        dec["replay_onset_minus_entry"].append((x["call"], x["d_rep"] - rel))
        dec["replay_onset_minus_bundle_median"].append((x["call"], x["d_rep"] - sm))
    timing["emulator_bias_decomposition"] = {k: sdist(v) for k, v in dec.items()}
    res["timing"] = timing

    res["leakage"] = ("상태의 정체성·분할·같은 상태 판정은 427통화를 모두 본 분석자가 정했다. 시작 규칙·끼어들기 멈춤·도중 지연 차이·"
                      "입력 결과의 최다 선택·대표 조각 선택·묶음의 지연 가운뎃값·조용한 기다림도 모든 통화에서 정했다. 지지 집합은 실제 "
                      "경로가 아닌 엔진 경로로 만들었고, 변형 오라클은 시험 통화 자신의 소리를 쓴다. 따라서 LOO·시간 분할 덮임은 처음 보는 "
                      "통화에 대한 낙관적 상한(녹음 밖 비율의 하한)이며, 가상 ARS 지연 오차는 표본 내 값이다. 시간 분할의 시험일은 "
                      "일요일(11-08)과 월요일(11-09)뿐이다.")
    res["calls_loo"] = [{k: v for k, v in x.items() if k != "name"} for x in loo_rows if x["offmap"]]
    res["calls_temporal"] = trows
    res["params"] = {"train_dates": TRAIN_DATES, "test_dates": TEST_DATES, "onset_after_s": ONSET_AFTER_S,
                     "tail_max_s": TAIL_MAX_S, "window_s": WINDOW_S, "mask": [-rc.MASK_BEFORE_S, rc.MASK_AFTER_S],
                     "island_rms": rc.ACTIVE_RMS, "boot": {"n": BOOT_N, "seed": BOOT_SEED, "cluster": "call"},
                     "same_m3": SAME_M3, "workers": workers}
    res["elapsed_s"] = round(time.time() - t_start, 1)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=list), encoding="utf-8")

    def brief(x):
        return {k: x.get(k) for k in ("n", "bias_median", "abs_median", "abs_p90", "median", "p90") if k in x}

    tp = timing["groups_pooled"]["by_org"].get("ALL", {})
    return {
        "calls": len(recs), "path_equal_insample": f"{res['path_equal_insample_check']['equal']}/{len(recs)}",
        "trace_vs_path_elements": f"{agree}/{len(recs)}",
        "loo_offmap": loo_summ["ALL"]["loo_offmap"],
        "loo_by_org": {o: f"{v['loo_offmap']['k']}/{v['loo_offmap']['n']}" for o, v in loo_summ.items()},
        "temporal_by_org": {o: (f"{v['covered']['k']}/{v['covered']['n']}" if "covered" in v else v["status"])
                            for o, v in temporal["by_org"].items()},
        "loo_first_kind": loo_summ["ALL"]["first_singleton_kind"], "loo_kept": loo_summ["ALL"]["kept_time_share_cause"],
        "temporal": temporal["ALL_evaluable"]["covered"], "temporal_kept": temporal["ALL_evaluable"]["kept_time_share"],
        "within_edge_abs_dev": brief(tp.get("within_edge_abs_dev", {})), "loo_err": brief(tp.get("loo_median_err", {})),
        "test_train_err": brief(tp.get("test_train_median_err", {})), "emulator": brief(timing["emulator_err"]["ALL"]),
        "spec_agreement": brief(timing["measured_median_minus_bundle_median"].get("ALL", {})),
        "elapsed_s": res["elapsed_s"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=1))
