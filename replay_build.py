"""
녹음 기반 가상 ARS 묶음 생성 (replay_ars.py 가 읽음)

  python replay_build.py <spec 폴더> <출력 폴더> [--roots <녹음 폴더 상위> ...] [--exclude <제외할 녹음 폴더 이름 목록 파일>]

spec_<번호>.json(기관별 녹음 분석 결과)을 읽어, 상태마다 녹음 call.wav 에서 소리 조각을 잘라 μ-law 로 저장하고
bundle.json 을 기록.
  - 소리 조각: 상태의 audio 구간 후보 중 (끊기지 않고 들린 것, 받아쓰기가 있는 것, 길이가 가운뎃값에 가까운 것) 순으로 선택.
    그 통화에서 서버가 보낸 키 톤과 겹치는 구간은 쓰지 않음(키가 구간 뒤쪽이면 키 톤 앞까지 끊긴 조각으로). 앞뒤 무음은 말소리
    경계(10ms RMS 100 초과)까지 다듬음(키 뒤 지연·시간 초과가 말소리 경계로 잰 값이므로). 개인정보 상태(personal)는 소리를 저장하지 않음.
  - 연결: 관측 수가 0 인 추정 연결(spec 의 inferred_from·inferred_next, n=0 입력 결과)은 제외 (녹음에 없는 것은 지어내지
    않음). 연속 무입력 규칙(count_rule)은 third_fail_next 로 옮김. 같은 안내의 자기 반복은 녹음에서 본 횟수까지 허용
    (MAX_PLAYS, 그 밖 2번).
  - 회선 잡음: 녹음의 말 없는 구간이 디지털 무음이면 없음, 아니면 서버 키 톤이 없고 가장 조용한 2초.
  - 시간 초과 기록이 없는 상태: 녹음된 통화를 엔진으로 따라가 안내 끝부터 키까지 조용히 기다린 최대 시간(silent_wait_s)을 기록.
    재생기는 그 시간(+0.5초)과 기관 시간 초과 상한 중 긴 쪽까지 기다린 뒤 녹음 밖 종료.
  - 기관 값: answer_delay_s = 발신 기록부터 연결 기록까지의 가운뎃값(log.json status_changed dialing→listening),
    unseen_timeout_s = 그 기관에서 관측한 입력 시간 초과(3초 이상)의 최댓값, 관측이 없으면 없음(재생기 기본 7초).
  - 받아쓰기(paras): 그 조각을 녹음할 때 서버가 받아쓴 단락 중 조각 안에서 시작한 것. 인식 지연(at)은 단락 소리 길이별
    가운뎃값(모든 녹음의 stt 사건에서 계산)으로 다시 정함(at_rec 은 녹음값).
  - log.json 의 backend 가 clawops 가 아닌 녹음(가상 ARS 통화 등)은 사용하지 않음.
  - --exclude: 검증용으로 남겨 둘 녹음 폴더 이름(한 줄에 하나). 그 녹음의 소리·받아쓰기는 사용하지 않음.

bundle.json
  {"version": 1, "phones": {"15889999": "15889999"},
   "orgs": {"15889999": {"org": "KB국민은행", "start": "S0", "answer_delay_s": 1.0, "unseen_timeout_s": 10.0,
            "first_audio_s": {median, min, max},
            "noise": "15889999/noise.ulaw",
            "states": {"S0": {"clip": "15889999/S0.ulaw", "kind": "menu", "summary": "…", "barge_in": "yes|no|unknown",
                              "keys": {"3": {"next": "S5", "delay_s": {...}, "n": 4}},
                              "after_audio": {"timeout_s": {...}, "next": "S9|hangup", "third_fail_next": "…",
                                              "max_plays": 3},
                              "invalid_key": {"next": "S7", "delay_s": {...}},
                              "input": {"len": 6, "terminator": "#", "digit_timeout_s": 5.0}, "input_next": "S12",
                              "input_delay_s": {...}, "input_value_dependent": false,
                              "personal": false, "clip_from": "녹음 폴더 이름",
                              "paras": [{"text": "…", "start": 0.1, "end": 4.3, "at": 4.6, "at_rec": 5.0,
                                         "forced": false}]}}}}}
  paras: 그 조각이 녹음될 때 서버가 받아쓴 단락(조각 시작 기준 초, start 는 Segmenter 앞 여유 포함, end 는 단락이 닫힌 때 =
  소리 끝 + 1.2초). forced 는 45초 최대 길이에서 잘린 것으로 보는 단락(길이 FORCED_PARA_S 초 이상).
"""

from __future__ import annotations

import argparse
import array
import glob
import json
import logging
import os
import re
import sys
import wave
from functools import lru_cache
from pathlib import Path

ROOTS = [r"C:\Users\User\Documents\ars_combo_live\recordings", r"C:\Users\User\Documents\ars_combo\recordings",
         r"C:\Users\User\Documents\ars_local_jev\recordings", r"C:\Users\User\Documents\ars_local_models\recordings"]
RATE = 8000
WEEKDAYS, SAT, SUN = [0, 1, 2, 3, 4], [5], [6]
# 시간대에 따라 다른 첫 상태. 녹음된 통화 시각(spec 의 variants·start_*)과 안내 문구('평일 오전 9시부터 오후 6시')로 정한 값.
# 경계 시각이 녹음되지 않은 곳은 안내 문구 기준. 규칙에 맞지 않는 시간(예: 일요일 낮)은 START_DEFAULT 의 첫 상태
START_RULES = {
    # KB국민은행: 평일 09~22시 S0, 토요일 09~18시 S1, 그 밖 S2
    "15889999": [{"days": WEEKDAYS, "from": "09:00", "to": "22:00", "state": "S0"},
                 {"days": SAT, "from": "09:00", "to": "18:00", "state": "S1"}],
    "15771000": [{"days": WEEKDAYS, "from": "09:00", "to": "18:00", "state": "S0_DAY"}],
    "15881300": [{"days": WEEKDAYS, "from": "09:00", "to": "18:00", "state": "D0"}],
    "15776000": [{"days": WEEKDAYS, "from": "09:00", "to": "18:00", "state": "S0"}],
    "15882188": [{"days": WEEKDAYS, "from": "09:00", "to": "18:00", "state": "S0_open"}],
}
# 안내 도중 키를 받은 뒤 ARS 가 말을 멈추기까지의 시간(톤 끝 기준, 초). 녹음에서 잰 기관별 값, 없는 기관은 0.17.
# 15771000 은 시간 초과 안내 도중에 잰 값
BARGE_STOP_S = {"15889999": 0.17, "15881300": 0.2, "15880075": 0.17, "15776000": 0.17, "15771000": 0.8}
# 녹음에서 같은 안내가 시간 초과로 연속해서 나온 최대 횟수 (그보다 많은 반복은 녹음에 없음)
MAX_PLAYS = {"15889999": {"S4": 3, "S36": 2}}
# 조각 앞뒤 말소리 경계 판정 (10ms RMS) 과 남기는 여유(초)
SPEECH_EDGE_RMS = 100.0
CLIP_PAD_S = 0.0  # 키 뒤 지연·시간 초과가 말소리 경계로 잰 값이므로 여유 없음 (여유를 두면 다음 안내가 그만큼 늦어짐)
# 서버 Segmenter 가 단락을 닫는 무음 길이(초, config.paragraph_silence_ms) 와 45초 최대 길이에서 잘린 단락으로 보는 길이
PARAGRAPH_SILENCE_S = 1.2
FORCED_PARA_S = 41.0
# 시작 규칙에 맞지 않는 시간의 첫 상태 (spec 의 start 가 낮 시간대 상태인 기관용)
START_DEFAULT = {"15889999": "S2", "15771000": "S0_NIGHT", "15881300": "N0", "15776000": "S0_night", "15882188": "S0_closed"}


def pcm16_to_ulaw(pcm: bytes) -> bytes:
    from clawops.agent._audio import pcm16_to_ulaw as conv

    return conv(pcm)


def find_dir(name: str, roots: list[str], root_hint: str | None = None) -> Path | None:
    cands = [Path(r) / name for r in roots if not root_hint or root_hint in r] + [Path(r) / name for r in roots]
    for c in cands:
        if (c / "call.wav").exists():
            return c
    return None


def read_pcm(d: Path) -> array.array:
    with wave.open(str(d / "call.wav"), "rb") as w:
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
        return a


def key_spans(d: Path) -> list[tuple[float, float]]:
    """그 통화에서 서버가 보낸 키 톤 구간(초). 자리마다 톤 0.15초 + 쉼 0.10초(마지막 쉼 제외), 앞뒤 여유 0.02·0.03초."""
    try:
        j = json.loads((d / "log.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for e in j.get("events", []):
        if e.get("type") == "keys_sent":
            n = len(e.get("keys") or "") or int(e.get("tone_count") or (int(e.get("digit_count") or 0) + 1))
            out.append((float(e["t"]) - 0.02, float(e["t"]) + 0.25 * n - 0.1 + 0.03))
    return out


def backend_of(d: Path) -> str | None:
    try:
        return json.loads((d / "log.json").read_text(encoding="utf-8")).get("backend")
    except (OSError, ValueError):
        return None


def digital_silence(dirs: list[Path]) -> bool:
    """회선의 말 없는 구간이 정확히 0(디지털 무음)인지 여부. 그렇다면 잡음 대신 무음 송출."""
    import numpy as np

    zeros = total = 0
    for d in dirs[:20]:
        pcm = np.frombuffer(read_pcm(d).tobytes(), dtype=np.int16)
        zeros += int(np.sum(pcm == 0))
        total += len(pcm)
    return total > 0 and zeros / total > 0.2


def quiet_noise(dirs: list[Path], seconds: float = 2.0) -> bytes | None:
    import numpy as np

    if digital_silence(dirs):
        return None

    best = None
    n = int(seconds * RATE)
    for d in dirs[:30]:
        pcm = np.frombuffer(read_pcm(d).tobytes(), dtype=np.int16).astype(np.float64)
        if len(pcm) < n * 2:
            continue
        spans = key_spans(d)
        step = RATE // 4
        for i in range(RATE, len(pcm) - n, step):  # 첫 1초(연결 소리)는 건너뜀
            t0, t1 = i / RATE, (i + n) / RATE
            if any(a < t1 and b > t0 for a, b in spans):
                continue
            rms = float(np.sqrt(np.mean(pcm[i:i + n] ** 2)))
            if rms < 1:  # 완전한 0 (기록 공백)은 회선 잡음이 아님
                continue
            if best is None or rms < best[0]:
                best = (rms, d, i)
    if best is None or best[0] > 100:
        return None  # 가장 조용한 구간도 말소리 수준이면(녹음이 적을 때) 잡음 조각을 만들지 않음
    _, d, i = best
    pcm = read_pcm(d)[i:i + n]
    return pcm16_to_ulaw(pcm.tobytes())


def norm_keys(keys) -> dict:
    """키 연결: {키: {next, delay_s, barge_delay_s, n}}. 입력 완료를 나타내는 특수 이름(input·input_alt)은 제외(input_next 로 처리)."""
    out = {}
    for k, e in (keys or {}).items():
        if not isinstance(e, dict) or not e.get("next") or str(k) in ("input", "input_alt"):
            continue
        if e.get("n") == 0 or e.get("inferred_from") or e.get("inferred_next"):
            continue  # 관측 없이 다른 상태에서 옮겨 적은 추정 연결은 제외
        out[str(k)] = {"next": e["next"], "delay_s": e.get("delay_s"), "n": e.get("n"),
                       # 안내 도중 눌렀을 때의 지연 (spec 의 delay_barge_in_s 또는 delay_after_cut_timeout_notice_s)
                       "barge_delay_s": e.get("delay_barge_in_s") or e.get("delay_after_cut_timeout_notice_s")}
    return out


def norm_after(a) -> dict | None:
    """안내 뒤 입력이 없을 때: {next, timeout_s, third_fail_next, third_fail_timeout_s}. 안내 뒤 끊김(hangup_after_audio_s)은
    next=hangup. 녹음에 다음 동작이 없으면 None."""
    if not isinstance(a, dict):
        return None
    out = {"next": a.get("next"), "timeout_s": a.get("timeout_s")}
    if a.get("hangup_after_audio_s") is not None and not out["next"]:
        out = {"next": "hangup", "timeout_s": a["hangup_after_audio_s"]}
    if out["next"] == "hangup" and out["timeout_s"] is None and a.get("hangup_after_audio_s") is not None:
        out["timeout_s"] = a["hangup_after_audio_s"]
    if not a.get("third_fail_next") and a.get("count_rule"):
        # count_rule(예: '두 번째 반복 뒤 시간 초과면 S_final'): next_counts 에서 반복 대상과 다른 끝 상태(S_…)를 세 번째 실패 상태로 사용
        ends = [k for k in (a.get("next_counts") or {}) if k != out["next"] and k.startswith("S_")]
        if ends:
            a = {**a, "third_fail_next": ends[0]}
    if a.get("third_fail_next"):
        out["third_fail_next"] = a["third_fail_next"]
        out["third_fail_timeout_s"] = (a.get("by_next") or {}).get(a["third_fail_next"]) or a.get("timeout_s")
    if isinstance(out.get("timeout_s"), dict) and out["timeout_s"].get("median") is None:
        out["timeout_s"] = None
    return out if out.get("next") else None


def value_dependent(s: dict) -> bool:
    """녹음에서 입력 값에 따라 결과가 갈린 입력 상태인지 여부: 관측 결과가 둘 이상이거나 spec 의 note 에 그렇게 적힌 것."""
    inp = s.get("input") or {}
    outs = [o for o in (inp.get("outcomes") or []) if o.get("next") and o.get("n")]
    return len({o["next"] for o in outs}) >= 2 or "값에 달" in str(inp.get("note") or "")


def input_next(s: dict):
    """입력 상태를 마친 뒤 상태: 특수 키 input, input.outcomes, 끝맺음 키 연결 순으로 확인."""
    keys = s.get("keys") or {}
    if isinstance(keys.get("input"), dict) and keys["input"].get("next") and keys["input"].get("n") != 0:
        return keys["input"]["next"], keys["input"].get("delay_s")
    outs = [o for o in ((s.get("input") or {}).get("outcomes") or [])
            if o.get("next") and o.get("n") != 0 and not o.get("inferred_from")]
    if outs:
        o = max(outs, key=lambda x: x.get("n") or 0)
        return o["next"], o.get("delay_s")
    term = (s.get("input") or {}).get("terminator")
    if term and isinstance(keys.get(term), dict) and keys[term].get("next") and keys[term].get("n") != 0:
        return keys[term]["next"], keys[term].get("delay_s")
    return None, None


@lru_cache(maxsize=None)
def stt_events(d: Path) -> tuple:
    """그 녹음에서 서버가 받아쓴 단락: (소리 시작, 단락 닫힘, 받아쓰기 나온 시각, 문장) 초 단위."""
    try:
        j = json.loads((d / "log.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    out = []
    for e in j.get("events", []):
        a, b, at = e.get("seg_start_t"), e.get("seg_end_t"), e.get("t")
        if e.get("type") == "stt" and e.get("text") and None not in (a, b, at):
            out.append((float(a), float(b), float(at), e["text"]))
    return tuple(out)


# 단락 소리 길이(초)별 인식 지연(단락이 닫힌 뒤 받아쓰기가 나오기까지) 가운뎃값. build() 가 모든 녹음에서 다시 계산
LATENCY_BUCKETS = [(5.0, 0.16), (10.0, 0.22), (20.0, 0.34), (float("inf"), 0.66)]


def latency_for(speech_s: float) -> float:
    for upto, lat in LATENCY_BUCKETS:
        if speech_s < upto:
            return lat
    return LATENCY_BUCKETS[-1][1]


def measure_latency(roots: list[str], exclude: set[str] = frozenset()) -> list[tuple[float, float]]:
    """모든 실통화 녹음의 stt 사건에서 단락 소리 길이(닫힘 − 시작 − 앞 여유 0.2 − 닫힘 무음 1.2)별 인식 지연(t − seg_end_t)
    가운뎃값 계산. 구간: 5초 미만, 5~10, 10~20, 20초 이상."""
    import statistics

    from replay_check import universe

    fixed = universe()
    bins: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
    edges = [5.0, 10.0, 20.0]
    for root in roots:
        for d in Path(root).glob("*"):
            if d.name in exclude or not (d / "call.wav").exists() or backend_of(d) not in (None, "clawops"):
                continue
            if fixed is not None and d.name not in fixed:
                continue
            for a, b, at, _ in stt_events(d):
                speech = b - a - 0.2 - PARAGRAPH_SILENCE_S
                bins[sum(speech >= e for e in edges)].append(at - b)
    out = []
    for i, upto in enumerate(edges + [float("inf")]):
        vals = [v for v in bins[i] if v >= 0]
        out.append((upto, round(statistics.median(vals), 2) if vals else LATENCY_BUCKETS[i][1]))
    return out


def paragraphs(d: Path, t0: float, t1: float) -> list[dict]:
    """구간 [t0, t1] 의 소리를 서버가 받아쓴 단락 (구간 시작 기준 초): start = 단락 소리 시작(200ms 앞 여유 포함),
    end = 서버가 단락을 닫은 때(소리 끝 + 1.2초 무음), at = 닫힘 + 단락 길이별 인식 지연 가운뎃값, at_rec = 녹음의 받아쓰기 시각.
    구간 안에서 시작한(앞 여유 0.5초) 단락은 모두 그 조각 몫. 시간 초과로 다음 안내가 1.2초 안에 이어지면 실제 서버에서 두 안내가
    한 단락이 되므로, 그 단락은 시작한 조각에 붙임. 키에 끊겨 서버가 버린 단락은 녹음에도 없음."""
    out = []
    for a, b, at, text in stt_events(d):
        if not (t0 - 0.5 <= a < t1):
            continue
        speech = b - a - 0.2 - PARAGRAPH_SILENCE_S
        out.append({"text": text, "start": round(a - t0, 2), "end": round(b - t0, 2),
                    "at": round(b - t0 + latency_for(speech), 2), "at_rec": round(at - t0, 2),
                    "forced": (b - a) >= FORCED_PARA_S})
    return out


def speech_bounds(pcm, t0: float, t1: float) -> tuple[float, float]:
    """[t0, t1] 안의 말소리 경계(10ms RMS 가 SPEECH_EDGE_RMS 를 넘는 첫·마지막 칸) ± CLIP_PAD_S. 말소리가 없으면 그대로 반환."""
    import numpy as np

    i0, i1 = int(t0 * RATE), int(t1 * RATE)
    x = np.frombuffer(pcm[i0:i1].tobytes(), dtype=np.int16).astype(np.float64)
    n = 80
    m = len(x) // n
    if m == 0:
        return t0, t1
    e = np.sqrt(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1))
    on = np.nonzero(e > SPEECH_EDGE_RMS)[0]
    if len(on) == 0:
        return t0, t1
    a = max(t0, t0 + on[0] * n / RATE - CLIP_PAD_S)
    b = min(t1, t0 + (on[-1] + 1) * n / RATE + CLIP_PAD_S)
    return a, b


def has_text(a: dict, roots: list[str]) -> bool:
    d = find_dir(a.get("dir", ""), roots, a.get("root"))
    return bool(d and a.get("start_s") is not None and paragraphs(d, float(a["start_s"]), float(a["end_s"])))


def cut_clip(spans: list, out: Path, phone: str, name: str, roots: list[str], exclude: set[str]):
    """구간 후보를 차례로 보며 쓸 수 있는 첫 구간을 잘라 저장. (상대 경로, 녹음 폴더 이름, 끊기지 않은 조각인지, 단락 받아쓰기)
    또는 None 반환."""
    for a in spans:
        if a.get("dir") in exclude:
            continue
        d = find_dir(a["dir"], roots, a.get("root"))
        if d is None or backend_of(d) not in (None, "clawops"):
            continue
        t0, t1 = float(a["start_s"]), float(a["end_s"])
        spans_k = key_spans(d)
        head = [y for x, y in spans_k if x <= t0 < y]
        if head:
            t0 = max(head)  # 앞 상태로 들어온 키 톤 끝이 구간 첫머리에 걸쳐 있으면 톤 뒤부터 사용
        hits = [x for x, y in spans_k if x < t1 and y > t0]
        trimmed = False
        if hits:
            # 서버 키에 끊긴 녹음: 키가 구간 뒤쪽이면 키 톤 앞까지만 사용 (끊긴 조각). 앞·가운데에 섞였으면 사용하지 않음
            cut = min(hits)
            if cut - t0 < 0.5:
                continue
            t1, trimmed = cut, True
        if t1 - t0 < 0.2:
            continue
        whole = read_pcm(d)
        t0, t1b = speech_bounds(whole, t0, t1)
        t1 = t1 if trimmed else t1b  # 키에 끊긴 조각은 끊긴 자리까지 사용 (ARS 가 말하던 중이므로)
        pcm = whole[int(t0 * RATE):int(t1 * RATE)]
        rel = f"{phone}/{name}.ulaw"
        (out / rel).write_bytes(pcm16_to_ulaw(pcm.tobytes()))
        return rel, a["dir"], bool(a.get("complete")) and not trimmed, paragraphs(d, t0, t1)
    return None


def answer_delay(phone: str, roots: list[str], exclude: set[str]) -> float:
    """발신 기록(status dialing)부터 연결 기록(status listening)까지 시각(wall) 차이의 가운뎃값. 녹음이 없으면 0.5초."""
    import statistics

    from replay_check import universe

    fixed = universe()
    vals = []
    for root in roots:
        for d in Path(root).glob(f"*_{phone}_*"):
            if d.name in exclude or not (d / "log.json").exists() or (fixed is not None and d.name not in fixed):
                continue
            try:
                j = json.loads((d / "log.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if j.get("backend") not in (None, "clawops"):
                continue
            walls = {}
            for e in j.get("events", []):
                if e.get("type") == "event:status_changed" and e.get("wall") is not None:
                    walls.setdefault((e.get("data") or {}).get("status"), float(e["wall"]))
            if "dialing" in walls and "listening" in walls and walls["listening"] >= walls["dialing"]:
                vals.append(walls["listening"] - walls["dialing"])
    return round(statistics.median(vals), 2) if vals else 0.5


def inside_git(path: Path) -> bool:
    """path 가 git 작업 폴더 안인지 여부 (개인정보 묶음을 저장소에 만들지 않기 위한 확인)."""
    p = path.resolve()
    return any((q / ".git").exists() for q in [p, *p.parents])


def build(spec_dir: str, out_dir: str, roots: list[str], exclude: set[str], include_personal: bool = False) -> dict:
    global LATENCY_BUCKETS
    out = Path(out_dir)
    if include_personal and inside_git(out.parent if not out.exists() else out):
        raise SystemExit("개인정보 상태를 담은 묶음은 git 저장소 밖에만 만든다: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    # includes_personal: 개인정보 상태(되읽기·조회 결과)의 실제 녹음 소리·받아쓰기를 담은 로컬 전용 묶음. 저장소·외부로 옮기지 않음
    bundle = {"version": 1, "phones": {}, "orgs": {}, "includes_personal": bool(include_personal)}
    report = {}
    LATENCY_BUCKETS = measure_latency(roots, exclude)
    report["_latency_buckets"] = LATENCY_BUCKETS
    for sp in sorted(glob.glob(os.path.join(spec_dir, "spec_*.json"))):
        spec = json.loads(Path(sp).read_text(encoding="utf-8"))
        phone = "".join(ch for ch in str(spec["phone"]) if ch.isdigit())
        (out / phone).mkdir(exist_ok=True)
        states, missing, used_dirs, dropped = {}, [], set(), 0
        for sid, s in spec["states"].items():
            st = {k: s.get(k) for k in ("kind", "summary", "barge_in", "invalid_key", "personal")}
            st["keys"] = norm_keys(s.get("keys"))
            dropped += sum(1 for k, e in (s.get("keys") or {}).items() if isinstance(e, dict) and e.get("next")
                           and str(k) not in ("input", "input_alt") and k not in st["keys"])
            st["after_audio"] = norm_after(s.get("after_audio"))
            if st["after_audio"] and st["after_audio"]["next"] == sid:
                st["after_audio"]["max_plays"] = MAX_PLAYS.get(phone, {}).get(sid)
            inp = s.get("input")
            if inp:
                n = inp.get("len")
                st["input"] = {"len": int(n) if str(n).isdigit() else None, "terminator": inp.get("terminator"),
                               "digit_timeout_s": (inp.get("interdigit_timeout_s") or {}).get("median")}
                nxt, delay = input_next(s)
                if nxt:
                    st["input_next"], st["input_delay_s"] = nxt, delay
                    st["input_value_dependent"] = value_dependent(s)
            if s.get("personal") and not include_personal:
                st["clip"] = None
                states[sid] = st
                continue
            groups: dict[str, list] = {}
            for a in s.get("audio") or []:
                groups.setdefault(a.get("variant") or "base", []).append(a)
            med = (s.get("duration_s") or {}).get("median")
            counts = dict(s.get("variant_counts") or {})
            variants = []
            for vname, spans in groups.items():
                # 대표 조각: 끊기지 않고 들린 것, 그 가운데 길이가 가운뎃값(같은 변형끼리)에 가까운 것
                lens = sorted(a.get("end_s", 0) - a.get("start_s", 0) for a in spans if a.get("complete"))
                target = (s.get("duration_s_by_variant") or {}).get(vname, {}).get("median") or (
                    lens[len(lens) // 2] if lens else med)
                # 받아쓰기 모드(REPLAY_TRANSCRIPTS=1)는 그 녹음의 단락 받아쓰기를 함께 내보내므로 받아쓰기가 있는 구간을 우선
                spans = sorted(spans, key=lambda a: (not a.get("complete"), not has_text(a, roots),
                                                     abs((a.get("end_s", 0) - a.get("start_s", 0)) - (target or 0))))
                got = cut_clip(spans, out, phone, sid if vname == "base" else f"{sid}__{len(variants)}", roots, exclude)
                if got is None:
                    continue
                rel, src, complete, paras = got
                w = counts.get(vname)
                if w is None:
                    tagged = sum(len(v) for k, v in groups.items() if k != "base")
                    w = max(1, ((s.get("duration_s") or {}).get("n") or len(spans)) - tagged) if vname == "base" else len(spans)
                if not paras and (s.get("texts") or s.get("summary")):
                    # 그 구간을 서버가 받아쓰지 않았으면(사용자 응답을 기다리던 중 등) spec 의 texts(없으면 summary 의 첫 문장,
                    # '(…)'·'+ …' 부분 제외)를 소리 끝에 한 단락으로 추가 (approx)
                    clip_s = len((out / rel).read_bytes()) / RATE
                    src_text = str(s["texts"][0]) if s.get("texts") else re.split(r"[(+]", str(s["summary"]))[0]
                    text = re.sub(r"\([^)]*\)", "", src_text).replace(" / ", " ").replace("…", "").strip()
                    if text:
                        end = clip_s - CLIP_PAD_S + PARAGRAPH_SILENCE_S
                        paras = [{"text": text, "start": 0.0, "end": round(end, 2),
                                  "at": round(end + latency_for(clip_s), 2), "approx": True, "forced": False}]
                variants.append({"clip": rel, "name": vname, "weight": w, "from": src, "complete": complete, "paras": paras})
            clip = None
            if variants:
                variants.sort(key=lambda v: -v["weight"])
                clip = variants[0]["clip"]
                st["clip_from"], st["clip_complete"] = variants[0]["from"], variants[0]["complete"]
                st["paras"] = variants[0]["paras"]
                if len(variants) > 1:
                    st["variants"] = [{"clip": v["clip"], "name": v["name"], "weight": v["weight"], "paras": v["paras"],
                                       "from": v["from"]} for v in variants]
                used_dirs.update(find_dir(v["from"], roots) for v in variants if find_dir(v["from"], roots))
            if clip is None:
                missing.append(sid)
            st["clip"] = clip
            states[sid] = st
        # 안내 도중 누른 키의 지연을 잰 연결이 일부뿐이면, 잰 연결들의 '도중 지연 - 끝난 뒤 지연' 가운뎃값을 기관 단위로 더함
        extra = []
        for st in states.values():
            for e in (st.get("keys") or {}).values():
                b, d = e.get("barge_delay_s"), e.get("delay_s")
                if isinstance(b, dict) and isinstance(d, dict) and b.get("median") is not None and d.get("median") is not None:
                    extra.append(b["median"] - d["median"])
        barge_extra = round(sorted(extra)[len(extra) // 2], 2) if extra else None
        noise = quiet_noise(sorted(used_dirs))
        if noise:
            (out / phone / "noise.ulaw").write_bytes(noise)
        bundle["phones"][phone] = phone
        timeouts = [((st.get("after_audio") or {}).get("timeout_s") or {}).get("median") for st in states.values()
                    if (st.get("after_audio") or {}).get("next") not in (None, "hangup")]
        # 3초 미만은 입력 시간 초과가 아니라 다음 안내로 바로 이어지는 것(예: 인사 뒤 곧바로 이어지는 안내)이므로 제외
        timeouts = [t for t in timeouts if isinstance(t, (int, float)) and t >= 3.0]
        bundle["orgs"][phone] = {"org": spec.get("org"), "start": START_DEFAULT.get(phone, spec["start"]),
                                 "start_rules": START_RULES.get(phone, []),
                                 "answer_delay_s": answer_delay(phone, roots, exclude),
                                 "unseen_timeout_s": round(max(timeouts), 2) if timeouts else None,
                                 "barge_stop_s": BARGE_STOP_S.get(phone, 0.17), "barge_extra_s": barge_extra,
                                 "first_audio_s": spec.get("answer_to_first_audio_s"),
                                 "noise": f"{phone}/noise.ulaw" if noise else None, "states": states}
        report[phone] = {"states": len(states), "no_clip": missing, "personal": sum(1 for x in states.values() if x.get("personal")),
                         "dropped_inferred_edges": dropped}
    (out / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    # 시간 초과 기록이 없는 상태가 녹음에서 조용히 기다린 최대 시간 (그동안은 ARS 가 말하지 않는다는 근거). 제외 녹음은 사용하지 않음
    from replay_check import observed_silent_waits

    logging.getLogger("ars.replay").setLevel(logging.ERROR)  # 따라가는 동안의 녹음 밖 경고는 검사(replay_check)에서 확인
    waits = observed_silent_waits(str(out), exclude)
    logging.getLogger("ars.replay").setLevel(logging.NOTSET)
    for phone, per in waits.items():
        for sid, vals in per.items():
            st = bundle["orgs"][phone]["states"].get(sid)
            if st is not None and vals:
                st["silent_wait_s"], st["silent_wait_n"] = round(max(vals), 2), len(vals)
    (out / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--roots", nargs="*", default=ROOTS)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--include-personal", action="store_true",
                    help="개인정보 상태의 실제 녹음까지 담는다 (이 PC 전용, git 저장소 밖 폴더에만)")
    a = ap.parse_args()
    exclude = set()
    if a.exclude:
        exclude = {x.strip() for x in Path(a.exclude).read_text(encoding="utf-8").splitlines() if x.strip()}
    print(json.dumps(build(a.spec_dir, a.out_dir, a.roots, exclude, a.include_personal), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
