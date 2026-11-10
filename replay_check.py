"""
녹음 기반 가상 ARS 검증: 녹음된 실통화와 같은 입력으로 가상 ARS 를 실행해 소리 비교 (실통화 없음)

  python replay_check.py <묶음 폴더> [--only <녹음 폴더 이름 목록 파일>] [--no-oracle] [--out <결과 json>]

측정 방법
  - 대상: 녹음 폴더(ROOTS)의 실통화 녹음 중 폴더 이름의 기관 번호가 묶음에 있고, log.json 의 backend 가 clawops(또는 없음)이며,
    call.wav 가 1초 이상인 것.
  - 입력: 녹음된 키를 같은 시각에 넣음. 키 시각 = keys_sent 의 t(전송 시작) + 0.25×i + 0.15초(자리 i 의 톤 끝). 민감정보
    입력은 값이 녹음에 없으므로 자릿수만큼 '0' 을 넣고, 서버가 덧붙인 끝맺음 '#' 는 n+1 번째 톤 자리에 넣음(tone_count 기록이
    없으면 call.wav 의 1kHz 대체 신호음 수로 판정). 첫 상태는 폴더 이름의 날짜·시각으로 선택. 키 전송이 끝난 때
    (t + 톤 전체 길이) keys_done 호출.
  - 변형: 실제 ARS 가 스스로 고르는 소리 변형은 원래 통화와 가장 닮은 것으로 선택(oracle). --no-oracle 이면 관측 비율대로
    선택(씨앗 = 폴더 이름의 md5 앞 8자리).
  - 구간: 통화 시작·각 키 시각·통화 끝으로 나눔. 0.6초 미만 구간은 제외. 가상 ARS 가 멈춘 뒤 시작하는 구간은 'after_stop'
    으로 따로 계산(원래 녹음에 말소리가 있으면 보수적 집계에서 실패). 양쪽 모두 말소리가 없으면 제외. 한쪽만 있으면 0점.
    서버 키 톤 구간(각 톤 끝 −0.17 ~ +0.05초)은 비교에서 제외.
  - 지표: M3(주 지표) = MFCC c1–c12(25ms 해밍 창, 10ms 이동, 100–3800Hz 멜 40대역) 단위 벡터를 DTW(띠 ±1초)로 맞춘 뒤,
    한쪽 이상 말소리인 프레임 쌍의 코사인 평균(한쪽만 말소리면 0). 같은 소리 = M3 ≥ 0.90. M1(참고 지표) = 50ms 로그
    에너지 윤곽 DTW 상관 ≥ 0.9.
  - 표지: 시작 구간(통화 시작~첫 키), 자기 출처(그 구간의 재생 프레임 중 하나라도 이 통화에서 잘라 온 조각), 개인정보 경계
    (가상 ARS 가 개인정보 상태로 멈춘 구간), 표시 뒤(path 의 첫 미검증 표시 이후).
  - 시작 시각: 원래와 가상의 첫 말소리 덩어리를 짝지어 시작 차(가상 − 원래, 부호 있음).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from audio import dtmf_ulaw, ulaw_to_pcm16
from replay_ars import FRAME_S, KEY_TONE_ON_S, KEY_TONE_S, KST, ReplayBundle, ReplayEngine

ROOTS = [r"C:\Users\User\Documents\ars_combo_live\recordings", r"C:\Users\User\Documents\ars_combo\recordings",
         r"C:\Users\User\Documents\ars_local_jev\recordings", r"C:\Users\User\Documents\ars_local_models\recordings"]
TONE_ON_S = KEY_TONE_ON_S
ENV_S = 0.05
SPEECH_RMS = 300.0  # 말소리로 볼 50ms 에너지 (PCM16 RMS). 덩어리·활동 판정은 그 1/3(100)
ACTIVE_RMS = SPEECH_RMS / 3
MASK_BEFORE_S, MASK_AFTER_S = 0.17, 0.05
MIN_SEG_S = 0.6
SAME_M3 = 0.90
SAME_M1 = 0.9
ENV_BAND = 20      # 50ms × 20 = 1초
SPEC_BAND = 100    # 10ms × 100 = 1초
SPEC_WIN, SPEC_HOP, NFFT, NMEL, FMIN, FMAX = 200, 80, 512, 40, 100.0, 3800.0


# 자료 -------------------------------------------------------------------------
def load_call(d: Path):
    j = json.loads((d / "log.json").read_text(encoding="utf-8"))
    with wave.open(str(d / "call.wav"), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    return j, pcm


def universe() -> set[str] | None:
    """REPLAY_UNIVERSE(녹음 폴더 이름 목록 파일)가 있으면 그 녹음만 검증·묶음 계산에 사용. 녹음이 계속 쌓이므로 검증 대상을
    고정해 같은 결과를 다시 낼 수 있게 함."""
    path = os.environ.get("REPLAY_UNIVERSE", "").strip()
    if not path:
        return None
    return {x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()}


def calls(only: set[str] | None = None, bundle: ReplayBundle | None = None):
    """검증 대상 녹음 폴더 (폴더 이름순, 녹음 루트 순). REPLAY_UNIVERSE 가 있으면 그 안에서만 선택."""
    fixed = universe()
    for root in ROOTS:
        for d in sorted(Path(root).glob("*")):
            if not (d / "log.json").exists() or (only is not None and d.name not in only):
                continue
            if fixed is not None and d.name not in fixed:
                continue
            m = re.search(r"_(\d{3,})_", d.name)
            if not m or (bundle is not None and bundle.org_for(m.group(1)) is None):
                continue
            yield d, m.group(1)


def call_clock(name: str) -> datetime | None:
    """폴더 이름 끝의 날짜·시각 (같은 시각 폴더에 붙는 '-2' 접미사 허용)."""
    m = re.search(r"(\d{8})-(\d{6})(?:-\d+)?$", name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=KST)


def tone_count(pcm: np.ndarray, a: float, b: float, f: float = 1000.0) -> int:
    """[a, b) 의 1kHz 대체 신호음(민감정보 입력 톤 대신 녹음에 넣은 것) 수: 10ms 블록 RMS ≥ 1000 이고 1kHz 성분 비율 > 0.6 인
    블록이 0.1초 이상 이어진 덩어리."""
    n = 80
    ref = np.exp(-2j * np.pi * f * np.arange(n) / 8000)
    runs = r = 0
    for i in range(max(0, int(a * 8000) // n), min(int(b * 8000) // n, len(pcm) // n)):
        x = pcm[i * n:(i + 1) * n]
        e = float(np.dot(x, x))
        if e >= n * 1000 ** 2 and abs(np.dot(x, ref)) ** 2 / (n * e / 2) > 0.6:
            r += 1
        else:
            runs += r >= 10
            r = 0
    return runs + (r >= 10)


def key_events(j: dict, pcm: np.ndarray | None = None) -> list[tuple[float, str, bool, float]]:
    """(ARS 가 받은 시각 = 톤 끝, 키, 민감정보 입력인지, 그 전송이 끝난 시각). 민감정보는 값 대신 '0', 덧붙인 '#' 는 '#'."""
    out = []
    sends = [e for e in j.get("events", []) if e.get("type") == "keys_sent"]
    starts = sorted(float(e["t"]) for e in sends)
    for e in sends:
        t = float(e["t"])
        if e.get("kind") == "input" or e.get("keys") is None:
            n = int(e.get("digit_count") or 0)
            tones = e.get("tone_count")
            if tones is None and pcm is not None:
                # 세는 창은 다음 전송 시작 전에서 끝냄 (한 자리씩 보낸 입력의 다음 자리 신호음을 '#' 로 세지 않게)
                nxt = next((s for s in starts if s > t), math.inf)
                tones = tone_count(pcm, t - 0.02, min(t + KEY_TONE_S * (n + 3), nxt - 0.02))
            seq = "0" * n + ("#" if (tones == n + 1 or (n == 0 and (tones or 0) >= 1)) else "")
            seq = seq or "0"
            secure = True
        else:
            seq, secure = str(e["keys"]), False
        done = t + len(dtmf_ulaw(seq)) / 8000
        for i, ch in enumerate(seq):
            out.append((t + i * KEY_TONE_S + TONE_ON_S, ch, secure, done))
    return sorted(out)


def key_mask(keys) -> list[tuple[float, float]]:
    return [(t - MASK_BEFORE_S, t + MASK_AFTER_S) for t, *_ in keys]


# 윤곽·덩어리 -------------------------------------------------------------------
def envelope(pcm: np.ndarray, mask: list[tuple[float, float]]) -> np.ndarray:
    n = int(ENV_S * 8000)
    m = len(pcm) // n
    e = np.sqrt(np.mean(pcm[: m * n].reshape(m, n) ** 2, axis=1))
    for a, b in mask:  # 서버 키 톤 구간은 비교에서 제외
        e[max(0, int(a / ENV_S)): int(b / ENV_S) + 1] = 0
    return np.log1p(e)


def islands(pcm: np.ndarray, a: float, b: float, mask: list[tuple[float, float]]) -> list[tuple[float, float, np.ndarray]]:
    """구간 [a, b) 의 말소리 덩어리: 10ms 에너지가 SPEECH_RMS/3 을 넘는 곳을 0.15초 미만 쉼은 이어 붙인 것(0.05초 이상).
    (시작, 끝, 10ms 로그 에너지). 서버 키 톤 구간은 제외."""
    n = 80
    i0, i1 = int(a * 8000) // n, int(b * 8000) // n
    x = pcm[i0 * n: i1 * n]
    m = len(x) // n
    if m == 0:
        return []
    e = np.sqrt(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1))
    for ka, kb in mask:
        e[max(0, int(ka * 100) - i0): max(0, int(kb * 100) - i0 + 1)] = 0
    on = e > ACTIVE_RMS
    out, start, gap = [], None, 0
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= 15:
                end = i - gap + 1
                if end - start >= 5:
                    out.append((a + start / 100, a + end / 100, np.log1p(e[start:end])))
                start, gap = None, 0
    if start is not None and len(on) - start >= 5:
        out.append((a + start / 100, a + len(on) / 100, np.log1p(e[start:])))
    return out


def corr(a: np.ndarray, b: np.ndarray, max_lag: int = 10) -> float | None:
    if len(a) < 10 or np.std(a) < 1e-6:
        return None
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
        k = min(len(x), len(y))
        if k < 10 or np.std(x[:k]) < 1e-6 or np.std(y[:k]) < 1e-6:
            continue
        best = max(best, float(np.corrcoef(x[:k], y[:k])[0, 1]))
    return best


def same_islands(orig, rep) -> tuple[int, int, float | None]:
    """원래 덩어리마다 가상 ARS 덩어리를 차례로 맞춤: 시작이 앞 덩어리 어긋남을 빼고 0.5초 안, 길이 차 15% 안, 윤곽 상관 0.85 이상.
    (맞은 수, 원래 수, 첫 덩어리 시작 차 = 가상 − 원래)"""
    j, drift, ok = 0, 0.0, 0
    first = None
    for s0, e0, env0 in orig:
        while j < len(rep) and rep[j][0] - drift < s0 - 0.5:
            j += 1
        if j >= len(rep):
            break
        s1, e1, env1 = rep[j]
        if first is None:
            first = s1 - s0
        if abs((s1 - drift) - s0) <= 0.5 and abs((e1 - s1) - (e0 - s0)) <= 0.15 * max(e0 - s0, 0.2):
            c = corr(env0, env1, 5)
            if c is not None and c >= 0.85:
                ok += 1
                drift = s1 - s0
                j += 1
    return ok, len(orig), first


# DTW (띠, 대각선 단위 벡터 계산; 점화식과 동률 규칙은 대각 → 위 → 왼쪽 순으로 첫 최솟값) -----------------------------------
def _dtw_path(dfun, n: int, m: int, band: int):
    if abs(n - m) > band:
        band = abs(n - m) + 5
    w = 2 * band + 3
    s_n = n + m + 1
    dd = np.full((s_n, w), np.inf)
    for d in range(-band, band + 1):
        lo, hi = max(1, d + 1), min(n, m + d)
        if lo > hi:
            continue
        i = np.arange(lo, hi + 1)
        dd[2 * i - d, d + band + 1] = dfun(i - 1, i - d - 1)
    c = np.full((s_n, w), np.inf)
    c[0, band + 1] = 0.0
    for s in range(2, s_n):
        c[s, 1:-1] = dd[s, 1:-1] + np.minimum(np.minimum(c[s - 1, :-2], c[s - 1, 2:]), c[s - 2, 1:-1])
    cl, off = c.tolist(), band + 1

    def g(i, j):
        if i < 0 or j < 0 or abs(i - j) > band:
            return math.inf
        return cl[i + j][i - j + off]

    i, j, ia, jb = n, m, [], []
    while i > 0 and j > 0:
        ia.append(i - 1)
        jb.append(j - 1)
        dg, up, lf = g(i - 1, j - 1), g(i - 1, j), g(i, j - 1)
        if dg <= up and dg <= lf:
            i, j = i - 1, j - 1
        elif up <= lf:
            i -= 1
        else:
            j -= 1
    return np.array(ia, dtype=np.int64), np.array(jb, dtype=np.int64)


def dtw_corr(a: np.ndarray, b: np.ndarray, band: int = ENV_BAND) -> float | None:
    """M1: 50ms 로그 에너지 윤곽을 DTW(비용 |a−b|, 띠 ±band)로 맞춘 뒤의 Pearson 상관."""
    n, m = len(a), len(b)
    if n < 5 or m < 5 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return None
    ia, jb = _dtw_path(lambda i, j: np.abs(a[i] - b[j]), n, m, band)
    pa, pb = a[ia], b[jb]
    if np.std(pa) < 1e-6 or np.std(pb) < 1e-6:
        return None
    return float(np.corrcoef(pa, pb)[0, 1])


def _mel_fb():
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(mm):
        return 700.0 * (10 ** (mm / 2595.0) - 1.0)

    hz = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), NMEL + 2))
    bins = np.fft.rfftfreq(NFFT, 1 / 8000.0)
    fb = np.zeros((NMEL, len(bins)))
    for k in range(NMEL):
        l, c, r = hz[k], hz[k + 1], hz[k + 2]
        fb[k] = np.maximum(0, np.minimum((bins - l) / (c - l), (r - bins) / (r - c)))
    return fb


_FB = _mel_fb()
_HAM = np.hamming(SPEC_WIN)
_K = np.arange(NMEL)
_DCT = np.sqrt(2.0 / NMEL) * np.cos(np.pi / NMEL * (_K[None, :] + 0.5) * _K[:, None])
_DCT[0] /= np.sqrt(2.0)


def mfcc_frames(x: np.ndarray, start_s: float, mask: list[tuple[float, float]]):
    """구간 신호 → (MFCC c1–c12 단위 벡터 [T,12], 말소리 프레임 [T]). 말소리 = 창 RMS > 100 이고 키 톤 가림과 겹치지 않음."""
    if len(x) < SPEC_WIN:
        return np.zeros((0, 12)), np.zeros(0, bool)
    nf = 1 + (len(x) - SPEC_WIN) // SPEC_HOP
    idx = np.arange(SPEC_WIN)[None, :] + SPEC_HOP * np.arange(nf)[:, None]
    f = x[idx]
    act = np.sqrt(np.mean(f ** 2, axis=1)) > ACTIVE_RMS
    t0 = start_s + np.arange(nf) * SPEC_HOP / 8000.0
    t1 = t0 + SPEC_WIN / 8000.0
    for a, b in mask:
        act &= ~((t0 < b) & (t1 > a))
    lm = np.log((np.abs(np.fft.rfft(f * _HAM, NFFT, axis=1)) ** 2) @ _FB.T + 1.0)
    mf = (lm @ _DCT.T)[:, 1:13]
    return mf / np.maximum(np.linalg.norm(mf, axis=1, keepdims=True), 1e-9), act


def mfcc_sim(xa, aa, xb, ab, band: int = SPEC_BAND) -> float | None:
    """M3: DTW(비용 1 − 유사도, 띠 ±band) 뒤 한쪽 이상 말소리인 쌍의 유사도 평균. 유사도 = 둘 다 말소리면 코사인, 한쪽만이면 0,
    둘 다 아니면 1(평균에서 뺌)."""
    n, m = len(xa), len(xb)
    if n < 5 or m < 5 or (not aa.any() and not ab.any()):
        return None

    def dfun(i, j):
        s = np.einsum("ij,ij->i", xa[i], xb[j])
        return 1.0 - np.where(aa[i] & ab[j], s, np.where(aa[i] | ab[j], 0.0, 1.0))

    ia, jb = _dtw_path(dfun, n, m, band)
    act = aa[ia] | ab[jb]
    if not act.any():
        return None
    s = np.where(aa[ia] & ab[jb], np.einsum("ij,ij->i", xa[ia], xb[jb]), 0.0)
    return float(s[act].mean())


def segment_m3(orig: np.ndarray, rep: np.ndarray, a: float, b: float, mask) -> float | None:
    i0, i1 = int(a * 8000), int(b * 8000)
    xa, aa = mfcc_frames(orig[i0:i1], a, mask)
    xb, ab = mfcc_frames(rep[i0:i1], a, mask)
    return mfcc_sim(xa, aa, xb, ab)


# 엔진 실행 ---------------------------------------------------------------------
def run_engine(bundle: ReplayBundle, org: str, clock, keys, until_s: float, orig_env: np.ndarray | None = None,
               oracle: bool = True, seed=None, on_key=None):
    """녹음된 키를 같은 시각에 넣으며 엔진 실행. (엔진, 가상 PCM, 프레임별 조각 출처, 받아쓰기 [(문장, 시작, 나온 시각)]) 반환.
    on_key(엔진, 키) 는 키를 넣기 직전에 호출."""
    eng = ReplayEngine(bundle, org, clock=clock, seed=seed)
    if oracle and orig_env is not None:
        def pick(state_id, variants):
            # 실제 ARS 가 스스로 고른 변형은 보낸 키와 무관하므로, 원래 통화가 들은 것과 가장 닮은 변형으로 재현
            i0 = int(eng.frame * FRAME_S / ENV_S)
            best, bi = -2.0, 0
            for i, v in enumerate(variants):
                env = envelope(np.frombuffer(ulaw_to_pcm16(bundle.clip(v["clip"])), dtype=np.int16).astype(np.float64), [])
                c = corr(orig_env[i0:i0 + len(env)], env) or -1.0
                if c > best:
                    best, bi = c, i
            return bi
        eng.choose_variant = pick
    frames, srcs, texts, ki = [], [], [], 0
    dones = sorted({k[3] for k in keys})
    di = 0
    while eng.frame * FRAME_S < until_s and not eng.ended:
        t = eng.frame * FRAME_S
        while ki < len(keys) and keys[ki][0] <= t:
            if on_key is not None:
                on_key(eng, keys[ki])
            eng.key(keys[ki][1])
            ki += 1
        while di < len(dones) and dones[di] <= t:
            eng.keys_done()
            di += 1
        frames.append(eng.tick())
        srcs.append(eng.last_src)
        if eng.transcripts:
            texts += [(x, s, round(eng.frame * FRAME_S, 2)) for x, s, _ in eng.transcripts]
            eng.transcripts = []
    pcm = np.frombuffer(ulaw_to_pcm16(b"".join(frames)), dtype=np.int16).astype(np.float64)
    return eng, pcm, srcs, texts


def observed_silent_waits(bundle_dir: str, exclude: set[str] = frozenset()) -> dict[str, dict[str, list[float]]]:
    """녹음된 통화를 엔진으로 따라가며, 입력 시간 초과 기록이 없는 상태에서 안내가 끝난 뒤 키가 올 때까지 기다린 시간을 수집
    ({기관: {상태: [초, …]}}). 그 상태의 시간 초과 동작이 녹음에 없다는 것은 그만큼 기다리는 동안 ARS 가 말하지 않았다는 뜻
    (말했다면 spec 의 after_audio 로 기록됨). 묶음 생성(replay_build)이 상태별 silent_wait_s 로 사용."""
    bundle = ReplayBundle(bundle_dir)
    for o in bundle.orgs.values():  # 수집하는 동안에는 기다림을 끊지 않음
        o["unseen_timeout_s"] = 1e9
        for st in o["states"].values():
            st.pop("silent_wait_s", None)
    out: dict[str, dict[str, list[float]]] = {}
    for d, ph in calls(None, bundle):
        if d.name in exclude:
            continue
        try:
            j, pcm = load_call(d)
        except Exception:
            continue
        if j.get("backend") not in (None, "clawops") or len(pcm) < 8000:
            continue
        org = bundle.org_for(ph)
        keys = key_events(j, pcm)

        def seen(eng, _key, org=org):
            st = eng.states.get(eng.state_id or "", {})
            if eng._waiting_after and not st.get("after_audio") and not (st.get("input") and eng.digits):
                out.setdefault(org, {}).setdefault(eng.state_id, []).append(round((eng.frame - eng._audio_done_frame) * FRAME_S, 2))

        # 검사와 같은 조건(같은 키 시각, 변형은 원래 통화와 가장 닮은 것)으로 실행
        run_engine(bundle, org, call_clock(d.name), keys, len(pcm) / 8000, envelope(pcm, key_mask(keys)), on_key=seen)
    return out


def clip_sources(bundle: ReplayBundle, org: str) -> dict[str, str]:
    """조각 경로 → 그 조각을 잘라 온 녹음 폴더 이름."""
    out = {}
    for st in bundle.orgs[org]["states"].values():
        if st.get("clip") and st.get("clip_from"):
            out[st["clip"]] = st["clip_from"]
        for v in st.get("variants") or []:
            if v.get("from"):
                out[v["clip"]] = v["from"]
    return out


# 검사 -------------------------------------------------------------------------
def check_call(bundle: ReplayBundle, d: Path, org: str, oracle: bool = True) -> dict | None:
    try:
        j, pcm = load_call(d)
    except Exception:
        return None
    if j.get("backend") not in (None, "clawops") or len(pcm) < 8000:
        return None
    keys = key_events(j, pcm)
    end_s = len(pcm) / 8000
    mask = key_mask(keys)
    eo = envelope(pcm, mask)
    seed = int(hashlib.md5(d.name.encode("utf-8")).hexdigest()[:8], 16)
    eng, out, srcs, texts = run_engine(bundle, org, call_clock(d.name), keys, end_s, eo, oracle=oracle, seed=seed)
    er = envelope(out, mask)
    sources = clip_sources(bundle, org)
    stop_s = eng.frame * FRAME_S if eng.ended else end_s
    personal = bool(eng.note and "개인정보" in eng.note)
    flag_times = [p["t"] for p in eng.path if "flag" in p]
    first_flag = min(flag_times) if flag_times else None
    marks = [0.0] + [t for t, *_ in keys if t < end_s] + [end_s]
    segs, after = [], []
    for a, b in zip(marks, marks[1:]):
        if b - a < MIN_SEG_S:
            continue
        if a >= stop_s:
            after.append({"from": round(a, 2), "to": round(b, 2), "orig_sound": bool(islands(pcm, a, b, mask))})
            continue
        b2 = min(b, stop_s)
        io, ir = islands(pcm, a, b2, mask), islands(out, a, b2, mask)
        if not io and not ir:
            continue  # 양쪽 모두 말소리 없음
        ok, n, dt = same_islands(io, ir)
        ia, ib = int(a / ENV_S), int(b2 / ENV_S)
        if bool(io) != bool(ir):
            m1 = m3 = 0.0  # 한쪽만 말소리
        else:
            m1 = dtw_corr(eo[ia:ib], er[ia:ib])
            m3 = segment_m3(pcm, out, a, b2, mask)
            m1 = 0.0 if m1 is None else m1
            m3 = 0.0 if m3 is None else m3
        f0, f1 = int(a / FRAME_S), int(b2 / FRAME_S)
        seg_src = {s for s in srcs[f0:f1] if s}
        states_in = [p["state"] for p in eng.path if "state" in p and a <= p["t"] < b2]
        cur = [p["state"] for p in eng.path if "state" in p and p["t"] < a]
        segs.append({
            "from": round(a, 2), "to": round(b2, 2), "start": a == 0.0,
            "m3": round(m3, 4), "m1": round(m1, 4), "same": m3 >= SAME_M3, "same_m1": m1 >= SAME_M1,
            "onset_dt": None if dt is None else round(dt, 2), "island_ratio": None if n == 0 else round(ok / n, 2),
            "orig_sound": bool(io), "replay_sound": bool(ir),
            "self_src": any(sources.get(s) == d.name for s in seg_src),
            "state_at_start": cur[-1] if cur else None, "states_entered": states_in,
            "personal_boundary": personal and eng.ended and a <= stop_s <= b + 1e-6,
            "after_flag": first_flag is not None and b2 > first_flag,
        })
    return {"dir": d.name, "org": org, "start": eng.start_state, "end_s": round(end_s, 1), "replay_end_s": round(stop_s, 1),
            "offmap": eng.note, "personal": personal, "early": bool(eng.ended and stop_s < end_s - 1.0),
            "flags": eng.flags, "first_flag_t": first_flag, "keys": len(keys),
            "path": [{k: v for k, v in p.items()} for p in eng.path], "segs": segs, "after_stop": after,
            "transcripts": [{"start": round(s, 2), "at": at, "chars": len(x)} for x, s, at in texts]}


def check(bundle_dir: str, only: set[str] | None = None, oracle: bool = True) -> dict:
    bundle = ReplayBundle(bundle_dir)
    res = [r for d, ph in calls(only, bundle) if (r := check_call(bundle, d, bundle.org_for(ph), oracle)) is not None]
    return {"summary": summarize(res), "calls": res}


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def summarize(res: list[dict]) -> dict:
    """구간 단위 같은 소리 비율(M3·M1)을 전체·비시작·비자기출처로, 보수적 비율(녹음 밖 정지 뒤 원래 말소리 구간을 실패로)과 통화
    결과(끝까지 / 녹음 밖 / 개인정보)를 함께 산출."""
    def rate(segs, key="same"):
        k = sum(1 for s in segs if s[key])
        return {"k": k, "n": len(segs), "rate": round(k / len(segs), 4) if segs else None, "wilson": wilson(k, len(segs))}

    out = {}
    orgs = sorted({r["org"] for r in res})
    for org in [None] + orgs:
        rs = [r for r in res if org is None or r["org"] == org]
        segs = [s for r in rs for s in r["segs"]]
        nonstart = [s for s in segs if not s["start"]]
        offmap_after = [a for r in rs if r["offmap"] and not r["personal"] for a in r["after_stop"] if a["orig_sound"]]
        dts = [s["onset_dt"] for s in segs if s["same"] and s["onset_dt"] is not None]
        out[org or "ALL"] = {
            "calls": len(rs),
            "outcome": {"full": sum(1 for r in rs if not r["early"]),
                        "offmap_early": sum(1 for r in rs if r["early"] and r["offmap"] and not r["personal"]),
                        "personal_early": sum(1 for r in rs if r["early"] and r["personal"]),
                        "hangup_early": sum(1 for r in rs if r["early"] and not r["offmap"])},
            "segments": rate(segs), "segments_m1": rate(segs, "same_m1"),
            "nonstart": rate(nonstart), "nonstart_not_self": rate([s for s in nonstart if not s["self_src"]]),
            "start": rate([s for s in segs if s["start"]]),
            "conservative": {"k": sum(1 for s in segs if s["same"]), "n": len(segs) + len(offmap_after)},
            "before_flag_nonstart": rate([s for s in nonstart if not s["after_flag"]]),
            "onset_signed_median": round(statistics.median(dts), 3) if dts else None,
            "onset_abs_median": round(statistics.median([abs(x) for x in dts]), 3) if dts else None,
            "onset_abs_p90": round(sorted(abs(x) for x in dts)[int(len(dts) * 0.9)], 3) if dts else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    only = None
    if a.only:
        only = {x.strip() for x in Path(a.only).read_text(encoding="utf-8").splitlines() if x.strip()}
    res = check(a.bundle, only, oracle=not a.no_oracle)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"].get("ALL"), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
