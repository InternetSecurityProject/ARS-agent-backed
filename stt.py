"""
음성 인식(STT)

Recognizer 프로토콜 구현으로 다른 STT 로 교체 가능.
  - is_local: 서버 밖으로 오디오를 보내지 않으면 True. 민감정보 입력 구간(ARS 가 입력값을 다시 읽어 주는 구간)은
              로컬 STT 일 때만 인식, 결과의 숫자는 마스킹. 외부 STT 이면 해당 구간 인식 생략.

  STT=whisper  로컬 faster-whisper (WHISPER_MODEL: 기본 한국어 small 모델, small / medium / large-v3-turbo ...)
  STT=openai   OpenAI 음성 인식 API (OPENAI_STT_MODEL, 기본 gpt-4o-transcribe). 외부 STT 이므로 is_local=False
  STT=rtzr / clova / gemini / qwen / omni   비교용 후보 엔진 (stt_engines.py)
  STT=none     인식하지 않음 (가상 ARS)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import re
import itertools
import time
import wave
import zlib
from types import SimpleNamespace
from typing import Protocol

from audio import FRAME, rms, ulaw8k_to_pcm16k, voiced_s

log = logging.getLogger("ars.stt")

# 어휘 힌트: 모든 ARS 공통 키 이름만 사용. 특정 기관 안내 문장·업무 낱말(분실신고, 보이스피싱 등)은
# 특정 기관·메뉴에 맞춘 힌트라 다른 기관 ARS 에서 엉뚱한 낱말을 끌어올 수 있어 사용 금지.
# Whisper 문맥: 이 키 이름 힌트 + 같은 단락 앞 조각의 인식 문장(미리 인식)
ARS_KEY_PROMPT = "별표, 샵 버튼, 우물정자."

# 키 이름 오인식 교정 (모든 ARS 공통 낱말만). 해석기의 '#' 선택지 인식에 '샵' 표기 필요
# ('샷 버튼' 등으로 적히면 '#' 선택지가 메뉴에서 누락)
_KEY_NAME_FIXES = [
    (re.compile(r"(샷|샾|섀)\s?(버튼|키)"), r"샵 \2"),
]

# 무음·대기 음악 구간에서 Whisper 가 지어내는 대표적인 문장 (방송 자막 학습의 흔적, ARS 어휘 아님).
# 문장 단위로 검사해 해당 문장만 제거. 방송사 이름만으로는 제거하지 않음 (방송사 이름이 든 방송사 ARS 메뉴 안내 보존)
_HALLUCINATIONS = re.compile(
    r"시청해\s*주셔서|구독(과|\s*와)?\s*좋아요|좋아요와\s*구독|자막\s*(제공|제작)|다음\s*영상에서|"
    r"(MBC|KBS|SBS)\s*뉴스\s*[가-힣]{2,4}(\s*기자)?입니다|(MBC|KBS|SBS)\s*뉴스(데스크)?였습니다|^(MBC|KBS|SBS)\s*뉴스\.?$")
# '감사합니다' 는 단락 전체가 그것뿐일 때만(반복 포함) 제거. 실제 안내 끝의 '…입니다. 감사합니다.' 는 결과·업무시간 외 규칙
# (gemini_interpreter.rule_classify) 이 읽으므로 유지. 미리 인식 조각에는 미적용 (조각마다 제거하면 쉼 뒤 꼬리로
# 따로 인식된 '감사합니다' 만 빠져, 같은 말도 쉼 위치에 따라 결과가 달라짐)
_GREETING_ONLY = re.compile(r"(?:감사합니다[.!]?\s*)+")
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")  # 문장부호 뒤 공백에서만 분리 ("3.5" 같은 소수는 그대로)


def _drop_hallucinations(text: str, piece: bool = False) -> str:
    kept = " ".join(s for s in _SENTENCE_END.split(text) if s and not _HALLUCINATIONS.search(s))
    return "" if not piece and _GREETING_ONLY.fullmatch(kept) else kept


# 메뉴 안내의 번호를 고유어로 적은 경우의 교정 ("보험료 문의는 한 번" → "1번"). Whisper 가 문맥에 따라 "1번" 대신 쓰는 표기.
# "~은/는 N번" 이 두 개 이상인 메뉴 안내일 때만 교정 ("한 번 더 눌러 주세요", "두 번 나누어" 같은 문장은 그대로)
_NATIVE = {"한": "1", "두": "2", "세": "3", "네": "4", "다섯": "5", "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9"}
_NATIVE_KEY = re.compile(r"(?<=[은는])\s(한|두|세|네|다섯|여섯|일곱|여덟|아홉)\s?번(?=을|를|에|이요|입니다|[^가-힣]|$)")
_MENU_ITEM = re.compile(r"[은는]\s?(?:[0-9]|한|두|세|네|다섯|여섯|일곱|여덟|아홉)\s?번(?=을|를|에|이요|입니다|[^가-힣]|$)")


LONG_SPLIT_S = 25.0  # 이보다 긴 음성은 쉼에서 나눠 인식 (Whisper 1회 입력 창 30초보다 짧게)
# 미리 인식 결과 보관 최대 수. 동시 통화가 함께 쓰므로 통화 하나 기준(16)보다 넉넉히 설정
PREFIX_KEEP = 32


def split_at_pauses(ulaw8k: bytes, max_s: float, quiet_rms: float = 500) -> list[bytes]:
    """max_s 초를 넘는 음성을 그 안의 가장 긴 쉼(20ms 무음 프레임이 이어진 곳) 가운데서 분할. 짧으면 그대로 한 조각.
    쉼은 앞 1/3 뒤에서 탐색 (너무 짧은 첫 조각 방지). 쉼이 없으면 max_s 에서 절단."""
    if len(ulaw8k) <= max_s * 8000:
        return [ulaw8k]
    limit = int(max_s * 50)
    best, best_len, run_len = None, 0, 0
    for i in range(limit):
        quiet = rms(ulaw8k[i * FRAME:(i + 1) * FRAME]) <= quiet_rms
        run_len = run_len + 1 if quiet else 0
        if run_len and run_len >= best_len and i > limit // 3:
            best, best_len = i - run_len // 2, run_len
    cut = (best if best is not None else limit) * FRAME
    return [ulaw8k[:cut]] + split_at_pauses(ulaw8k[cut:], max_s, quiet_rms)


def _clean(text: str, piece: bool = False) -> str:
    """인식 문장 정리. piece=True 는 단락의 일부(미리 인식 조각·긴 음성 조각)로, 단락 단위 판정('감사합니다' 뿐인 단락) 생략.
    단락 결과는 recognize 끝에서 piece=False 로 다시 정리."""
    text = _drop_hallucinations(re.sub(r"\s+", " ", text).strip(), piece)
    if not text:
        return ""
    for pat, rep in _KEY_NAME_FIXES:
        text = pat.sub(rep, text)
    if len(_MENU_ITEM.findall(text)) >= 2:
        text = _NATIVE_KEY.sub(lambda m: f" {_NATIVE[m.group(1)]}번", text)
    return text


def _plausible(seg) -> bool:
    """Whisper 조각이 실제 말을 받아 적은 것으로 보이는지 판정. 지어낸 문장의 특징으로 걸러냄.
    - no_speech_prob 이 높음: 말이 아닌 소리로 판단
    - compression_ratio 가 높음: "일어나서 일어나서 …" 같은 반복
    - avg_logprob 이 아주 낮음: 확신 없이 만든 문장"""
    return seg.no_speech_prob < 0.6 and seg.compression_ratio < 2.4 and seg.avg_logprob > -1.0


_LOOP_RE = re.compile(r"(\S.{2,29}?)(?:[\s,.]*\1){2,}")  # 같은 말이 세 번 이상 이어짐
_DIGIT_WORD = set("공영일이삼사오육칠팔구")


def _compression_ratio(text: str) -> float:
    """Whisper 와 같은 방식의 압축률 (높을수록 반복이 많음)."""
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


def _cut_loops(text: str) -> str:
    """같은 말이 세 번 이상 이어진 부분을 한 번으로 축약. 숫자가 아닌 한글 글자가 두 개 이상 든 반복만 대상
    (번호 되읽기 '일 일 일 일' 처럼 같은 숫자가 이어지는 실제 말은 유지)."""
    def rep(m: re.Match) -> str:
        unit = m.group(1)
        words = [c for c in unit if "가" <= c <= "힣" and c not in _DIGIT_WORD]
        return unit if len(words) >= 2 else m.group(0)
    return _LOOP_RE.sub(rep, text)


def _segment_text(seg) -> str:
    """Whisper 구간에서 쓸 문장. 반복 기준(압축률)에만 걸린 구간은 버리지 않고 반복 부분을 줄여 재판정.
    긴 메뉴 안내 끝에 같은 말의 반복 환각이 붙을 때 구간 전체가 버려져 앞의 실제 메뉴까지 누락되는 문제 방지."""
    if _plausible(seg):
        return seg.text.strip()
    if seg.no_speech_prob < 0.6 and seg.avg_logprob > -1.0:
        text = _cut_loops(seg.text.strip())
        # 실제로 반복을 줄였고 줄인 문장이 기준을 통과할 때만 채택 (줄인 것이 없으면 폐기)
        if text != seg.text.strip() and _compression_ratio(text) < 2.4:
            log.info("반복 환각 부분을 줄여 구간을 살림 (압축률 %.2f → %.2f)", seg.compression_ratio, _compression_ratio(text))
            return text
    return ""


def _window_texts(segs) -> list[str]:
    """faster-whisper 는 압축률·확신·말소리 확률을 디코딩 창(30초) 전체 문장으로 한 번 계산해 그 창의 모든 구간에 같은 값을 부여.
    반복 환각이 여러 구간으로 나뉘어 나와도 반복을 줄이고 재판정할 수 있도록 같은 창(seek)의 구간 문장을 합쳐 판정."""
    out = []
    for _, group in itertools.groupby(segs, key=lambda s: getattr(s, "seek", 0)):
        w = list(group)
        joined = SimpleNamespace(text=" ".join(s.text.strip() for s in w), no_speech_prob=w[0].no_speech_prob,
                                 avg_logprob=w[0].avg_logprob, compression_ratio=w[0].compression_ratio)
        t = _segment_text(joined)
        if t:
            out.append(t)
    return out


# 받아쓴 마지막 시각이 조각의 말소리 끝보다 이만큼 앞서면 누락 의심으로 판정. Whisper 는 끝까지 받아 적어도 마지막 문장의
# 끝 시각을 3초쯤 이르게 기록하는 경우가 있어 그보다 넉넉한 값 사용
CUT_SHORT_WARN_S = 6.0
RETRY_GAIN_S = 3.0  # 재인식 결과는 받아 적은 범위가 이만큼 넘게 늘었을 때만 사용


def voiced_end_s(ulaw: bytes, threshold: float) -> float:
    """마지막으로 RMS 가 threshold 를 넘는 20ms 프레임의 끝 시각(초). 말소리가 없으면 0."""
    for i in range(len(ulaw) // FRAME - 1, -1, -1):
        if rms(ulaw[i * FRAME:(i + 1) * FRAME]) > threshold:
            return (i + 1) * FRAME / 8000
    return 0.0


def _warn_if_cut_short(segs, ulaw8k: bytes, quiet_rms: float) -> float | None:
    """Whisper 가 조각 중간에서 끝(EOT)을 내고 나머지 말소리를 받아 적지 않은 것으로 보이면 경고 기록 (동작 변경 없음).
    조각의 첫 문장 뒤 나머지 선택지가 통째로 빠지는 경우의 감지용.
    받아쓴 끝과 말소리 끝의 차이(초) 반환 (경고하지 않으면 None). 재인식은 WhisperRecognizer._run 에서 처리."""
    ends = [s.end for s in segs if getattr(s, "end", None) is not None]
    if not ends:
        return None
    gap = voiced_end_s(ulaw8k, quiet_rms) - max(ends)
    if gap <= CUT_SHORT_WARN_S:
        return None
    log.warning("받아쓰기 누락 의심: 받아쓴 끝 %.1f초, 말소리 끝 %.1f초 (조각 %.1f초)", max(ends), max(ends) + gap,
                len(ulaw8k) / 8000)
    return gap


class Recognizer(Protocol):
    is_local: bool

    async def recognize(self, ulaw8k: bytes) -> str: ...


class NullRecognizer:
    """인식하지 않는 인식기. 가상 ARS(sim) 처럼 백엔드가 문장을 직접 주는 경우에 사용."""

    is_local = True

    async def recognize(self, ulaw8k: bytes) -> str:
        return ""


class WhisperRecognizer:
    """faster-whisper 로컬 인식. 최초 호출 시 모델 내려받기 (small 약 460MB, large-v3-turbo 약 1.6GB)."""

    is_local = True

    def __init__(self, model: str = "small", device: str = "cpu", beam_size: int = 1, quiet_rms: float = 500) -> None:
        self._name, self._device, self._beam = model, device, beam_size
        self.quiet_rms = quiet_rms
        self.vad = os.environ.get("STT_VAD", "1") == "1"  # 말소리 없는 구간 잘라내기 (faster-whisper 내장 Silero VAD)
        self._model = None
        self._lock = asyncio.Lock()
        self._prefix: list[tuple[int, bytes, str]] = []  # 미리 인식 결과 (끝 위치, 앞부분 해시, 누적 문장)

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("Whisper 모델 로드: %s (%s, 빔 %d)", self._name, self._device, self._beam)
            ct = "float16" if self._device == "cuda" else "int8"
            self._model = WhisperModel(self._name, device=self._device, compute_type=ct,
                                       cpu_threads=os.cpu_count() or 4)
        return self._model

    def _run(self, ulaw8k: bytes, context: str = "", beam: int | None = None) -> str:
        import numpy as np

        model = self._load()
        pcm = np.frombuffer(ulaw8k_to_pcm16k(ulaw8k), dtype=np.int16).astype(np.float32) / 32768.0
        # 힌트: 모든 ARS 공통 키 이름(ARS_KEY_PROMPT)만. 이어 붙일 조각이면 같은 단락 앞 조각의 인식 문장 추가
        prompt = f"{ARS_KEY_PROMPT} {context[-120:]}".strip() if context else (ARS_KEY_PROMPT or None)
        # 말소리가 없는 구간(무음·회선 잡음·딸깍·숨소리·DTMF 잔향)은 음성 감지(Silero VAD)로 잘라내 인식 제외.
        # 잘라내지 않으면 Whisper 가 문맥 문장을 닮은 문장을 지어내 AI 가 메뉴로 오인할 수 있음
        # 시각 정보(timestamp) 토큰 사용. 끄면 숫자를 한 자씩 끊어 읽는 구간(입력값 되읽기)을 통째로 건너뜀
        def transcribe() -> list:
            segs, _ = model.transcribe(pcm, language="ko", beam_size=beam or self._beam, vad_filter=self.vad,
                                       vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
                                       initial_prompt=prompt, condition_on_previous_text=False,
                                       temperature=0.0, without_timestamps=False)
            return list(segs)

        segs = transcribe()
        if _warn_if_cut_short(segs, ulaw8k, getattr(self, "quiet_rms", 500)) is not None:
            # 같은 소리를 한 번 더 인식하여 받아 적은 범위가 분명히 늘었을 때만 그 결과로 교체 (덧붙이지 않으므로 겹침·중복 없음).
            # 같은 음성도 다시 인식하면 끝까지 받아 적는 경우가 있어 1회 재시도
            again = transcribe()
            end1 = max([s.end for s in segs if getattr(s, "end", None) is not None] or [0.0])
            end2 = max([s.end for s in again if getattr(s, "end", None) is not None] or [0.0])
            if end2 - end1 > RETRY_GAIN_S:
                log.info("받아쓰기 다시 인식: 받아 적은 끝 %.1f초 → %.1f초", end1, end2)
                segs = again
        return _clean(" ".join(_window_texts(segs)), piece=True)

    def _run_long(self, ulaw8k: bytes, context: str = "", beam: int | None = None) -> str:
        """긴 음성은 25초 안의 가장 긴 쉼에서 나눠 차례로 인식, 앞 조각 문장을 다음 조각의 문맥으로 전달.
        Whisper 에 30초가 넘는 음성을 한 번에 넣으면 앞부분이 버려지는 문제 대응."""
        out = context
        for piece in split_at_pauses(ulaw8k, LONG_SPLIT_S, self.quiet_rms):
            out = f"{out} {self._run(piece, out, beam) if beam else self._run(piece, out)}".strip()
        return out[len(context):].strip() if context else out

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    # 미리 인식 ----------------------------------------------------------------
    # ARS 가 말하는 동안 Segmenter 가 단락 안의 쉼(0.3초)마다 그때까지의 음성 전달(prefetch).
    # 마지막으로 넘긴 뒤의 새 조각만 인식하여 앞 조각 문장에 이어 붙여 보관.
    # 단락 종료로 recognize 호출 시 이미 인식한 앞부분은 건너뛰고 남은 꼬리만 인식.
    # 앞부분 일치는 오디오 바이트의 해시로 확인. 민감정보 재확인 구간은 session 에서 미리 인식을 끄고 단락 전체를 한 번에 인식.
    def _cached(self, ulaw8k: bytes) -> tuple[int, str]:
        """ulaw8k 의 앞부분과 일치하는 가장 긴 미리 인식 결과 (길이, 문장)."""
        best = (0, "")
        for end, digest, text in self._prefix:
            if best[0] < end <= len(ulaw8k) and hashlib.sha1(ulaw8k[:end]).digest() == digest:
                best = (end, text)
        return best

    def _drop_prefix(self, ulaw8k: bytes) -> None:
        """인식을 마친 단락의 미리 인식 결과만 삭제 (그 단락 음성의 앞부분과 해시가 맞는 항목).
        전체를 지우면 동시 통화에서 한 통화의 인식이 다른 통화의 미리 인식을 지워 그 통화가 단락 전체를 다시 인식하게 됨.
        통화가 하나일 때 남는 항목은 다른 음성(버린 단락)의 앞부분이라 다시 맞을 수 없으므로 인식 결과는 동일."""
        self._prefix = [e for e in self._prefix
                        if not (e[0] <= len(ulaw8k) and hashlib.sha1(ulaw8k[:e[0]]).digest() == e[1])]

    async def prefetch(self, prefix: bytes, start: int) -> None:
        async with self._lock:
            done, prev = self._cached(prefix)
            if done >= len(prefix):
                return
            if done < start:  # 앞 조각 없음 (단락 시작 직후) → 처음부터
                done, prev = 0, ""
            t0 = time.monotonic()
            text = await asyncio.to_thread(self._run_long, prefix[done:], prev)
            cum = f"{prev} {text}".strip()
            self._prefix = self._prefix[-(PREFIX_KEEP - 1):] + [(len(prefix), hashlib.sha1(prefix).digest(), cum)]
            log.debug("미리 인식 %.2fs (%.1f~%.1fs) %s", time.monotonic() - t0, done / 8000, len(prefix) / 8000, text)

    async def recognize(self, ulaw8k: bytes) -> str:
        t0 = time.monotonic()
        async with self._lock:  # 모델 하나를 순차 사용 (진행 중인 미리 인식 종료 대기)
            done, prev = self._cached(ulaw8k)
            rest = ulaw8k[done:]
            # 남은 꼬리에 말소리가 0.1초도 없으면 단락 끝 무음뿐으로 판정. 평균 RMS 기준이면 짧은 마지막 항목('상담원 연결은 0번')이
            # 1.2초 무음에 묻혀 기준 아래로 내려가 누락되므로 프레임 단위 유음 길이로 판정
            if done and voiced_s(rest, self.quiet_rms) < 0.1:
                text = prev
            else:
                text = f"{prev} {await asyncio.to_thread(self._run_long, rest, prev)}".strip()
        self._drop_prefix(ulaw8k)
        text = _clean(text)
        log.info("인식 %.2fs (음성 %.1fs, 미리 인식 %.1fs 사용) %s", time.monotonic() - t0, len(ulaw8k) / 8000,
                 done / 8000, text)
        return text

    async def recognize_retry(self, ulaw8k: bytes) -> str:
        """같은 단락을 빔 1(탐욕 디코딩)로 재인식 (미리 인식 결과 미사용). 민감정보 되읽기의 숫자가 입력 자릿수보다 적을 때
        session 에서 호출: 빔 탐색(빔 2 이상)은 단락 시작점에 따라 번호 전체를 건너뛰는 경우가 있어 같은 소리를 빔 1 로 다시 인식.
        설정 빔이 1 이면 같은 디코딩이므로 재인식 없이 빈 문자열 반환."""
        if self._beam <= 1:
            return ""
        async with self._lock:
            text = await asyncio.to_thread(self._run_long, ulaw8k, "", 1)
        return _clean(text)


class OpenAIRecognizer:
    """OpenAI 음성 인식 API. 오디오가 서버 밖으로 나가므로 is_local=False (민감정보 재확인 구간은 인식 제외)."""

    is_local = False

    def __init__(self, model: str = "gpt-4o-transcribe", timeout_s: float = 10.0) -> None:
        from openai import AsyncOpenAI

        self.model, self.timeout_s = model, timeout_s
        # SDK 기본 제한 시간(10분)이 길어 응답이 늦으면 단락 처리 전체가 멈춤. 다른 엔진과 같은 STT_TIMEOUT_S 로 제한
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=1, timeout=timeout_s)

    async def recognize(self, ulaw8k: bytes) -> str:
        if not ulaw8k:
            return ""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(ulaw8k_to_pcm16k(ulaw8k))
        buf.name = "ars.wav"
        buf.seek(0)
        try:
            # SDK 의 timeout 은 연결·읽기 단계별 제한이고 재시도도 있으므로 전체 시간은 wait_for 로 제한
            r = await asyncio.wait_for(self.client.audio.transcriptions.create(
                model=self.model, file=buf, language="ko", prompt=ARS_KEY_PROMPT), self.timeout_s)
        except Exception as e:
            log.warning("STT 실패: %s", str(e) or type(e).__name__)
            return ""
        return _clean(r.text)


def stt_timeout_s() -> float:
    """단락 하나의 인식 제한 시간 (.env STT_TIMEOUT_S, 기본 10초). 외부 API 엔진 공통."""
    v = os.environ.get("STT_TIMEOUT_S", "").strip()
    return float(v) if v else 10.0


def make_recognizer(kind: str, model: str = "small", device: str = "cpu", quiet_rms: float = 500) -> Recognizer:
    if kind == "whisper":
        return WhisperRecognizer(model, device, int(os.environ.get("WHISPER_BEAM", "1")), quiet_rms)
    if kind == "openai":
        return OpenAIRecognizer(os.environ.get("OPENAI_STT_MODEL", "").strip() or "gpt-4o-transcribe", stt_timeout_s())
    if kind in ("rtzr", "clova", "gemini", "qwen", "omni"):
        from stt_engines import make_engine  # 후보 엔진은 선택 시에만 import

        return make_engine(kind, device)
    return NullRecognizer()
