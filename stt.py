"""
음성 인식(STT)

Recognizer 프로토콜을 구현하면 다른 STT 로 교체할 수 있다.
  - is_local: 서버 밖으로 오디오를 보내지 않으면 True. 민감정보 입력 구간(ARS 가 입력값을 다시 읽어 주는 구간)은
              로컬 STT 일 때만 인식하고, 결과의 숫자는 마스킹한다. 외부 STT 이면 해당 구간을 인식하지 않는다.

  STT=whisper  로컬 faster-whisper (WHISPER_MODEL: 기본 한국어 small 모델, small / medium / large-v3-turbo ...)
  STT=openai   OpenAI 음성 인식 API (OPENAI_STT_MODEL, 기본 gpt-4o-transcribe). 외부 STT 이므로 is_local=False
  STT=none     인식하지 않음 (가상 ARS)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import re
import time
import wave
from typing import Protocol

from audio import rms, ulaw8k_to_pcm16k

log = logging.getLogger("ars.stt")

# ARS 안내에 자주 나오는 표현. 인식 시 어휘 힌트로 쓴다.
ARS_PROMPT = ("안녕하십니까. KB국민은행입니다. 잔액조회는 1번, 거래내역조회는 2번, 상담원 연결은 0번, 다른 서비스는 샵 버튼, "
              "이전 메뉴는 별표, 다시 들으시려면 우물정자를 눌러 주십시오. 주민등록번호 앞 여섯 자리를 입력해 주십시오.")

# 무음·대기 음악 구간에서 Whisper 가 지어내는 대표적인 문장
_HALLUCINATIONS = re.compile(
    r"시청해\s*주셔서|구독(과|\s*와)?\s*좋아요|좋아요와\s*구독|MBC\s*뉴스|KBS\s*뉴스|자막\s*(제공|제작)|"
    r"다음\s*영상에서|^감사합니다\.?$")


# 실제 ARS 녹음에서 반복된 오인식을 ARS 표기로 바로잡는다 (키 이름·기관명. 해석기가 기호 키를 찾을 수 있게)
_ARS_FIXES = [
    (re.compile(r"(샷|샾|섀)\s?(버튼|키)"), r"샵 \2"),
    (re.compile(r"(킬로\s?바이트|힐로\s?바이트|케이\s?비|KBE)(?=\s?(국민|청년|고객|스타|\s|$))"), "KB"),
    (re.compile(r"본\s?뱅킹"), "폰뱅킹"),
]


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _HALLUCINATIONS.search(text):
        return ""
    for pat, rep in _ARS_FIXES:
        text = pat.sub(rep, text)
    return text


class Recognizer(Protocol):
    is_local: bool

    async def recognize(self, ulaw8k: bytes) -> str: ...


class NullRecognizer:
    """인식하지 않는다. 가상 ARS(sim) 처럼 백엔드가 문장을 직접 주는 경우에 사용한다."""

    is_local = True

    async def recognize(self, ulaw8k: bytes) -> str:
        return ""


class WhisperRecognizer:
    """faster-whisper 로컬 인식. 최초 호출 시 모델을 내려받는다 (small 약 460MB, large-v3-turbo 약 1.6GB)."""

    is_local = True

    def __init__(self, model: str = "small", device: str = "cpu", beam_size: int = 1, quiet_rms: float = 500) -> None:
        self._name, self._device, self._beam = model, device, beam_size
        self.quiet_rms = quiet_rms
        self._model = None
        self._lock = asyncio.Lock()
        self._prefix: list[tuple[int, bytes, str]] = []  # 미리 인식 결과 (끝 위치, 앞부분 해시, 누적 문장)

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("Whisper 모델 로드: %s (%s)", self._name, self._device)
            ct = "float16" if self._device == "cuda" else "int8"
            self._model = WhisperModel(self._name, device=self._device, compute_type=ct,
                                       cpu_threads=os.cpu_count() or 4)
        return self._model

    def _run(self, ulaw8k: bytes, context: str = "") -> str:
        import numpy as np

        model = self._load()
        pcm = np.frombuffer(ulaw8k_to_pcm16k(ulaw8k), dtype=np.int16).astype(np.float32) / 32768.0
        # 이어 붙일 조각이면 앞 조각의 인식 문장을 문맥으로 준다
        prompt = f"{ARS_PROMPT} {context[-120:]}" if context else ARS_PROMPT
        segs, _ = model.transcribe(pcm, language="ko", beam_size=self._beam, vad_filter=False,
                                   initial_prompt=prompt, condition_on_previous_text=False,
                                   temperature=0.0, without_timestamps=True)
        # 말소리가 아닐 가능성이 높은 구간(대기 음악·잡음)은 버린다
        keep = [s.text.strip() for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -0.8)]
        return _clean(" ".join(keep))

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    # 미리 인식 ----------------------------------------------------------------
    # ARS 가 말하는 동안 Segmenter 가 단락 안의 쉼(0.3초)마다 그때까지의 음성을 넘긴다(prefetch).
    # 마지막으로 넘긴 뒤의 새 조각만 인식하여 앞 조각 문장에 이어 붙여 둔다.
    # 단락이 끝나 recognize 가 불리면 이미 인식한 앞부분은 다시 인식하지 않고 남은 꼬리만 인식한다.
    # 앞부분 일치는 오디오 바이트의 해시로 확인한다. 민감정보 재확인 구간도 로컬 인식이므로 같은 규칙을 따른다.
    def _cached(self, ulaw8k: bytes) -> tuple[int, str]:
        """ulaw8k 의 앞부분과 일치하는 가장 긴 미리 인식 결과 (길이, 문장)."""
        best = (0, "")
        for end, digest, text in self._prefix:
            if best[0] < end <= len(ulaw8k) and hashlib.sha1(ulaw8k[:end]).digest() == digest:
                best = (end, text)
        return best

    async def prefetch(self, prefix: bytes, start: int) -> None:
        async with self._lock:
            done, prev = self._cached(prefix)
            if done >= len(prefix):
                return
            if done < start:  # 앞 조각이 아직 없다 (단락 시작 직후) → 처음부터
                done, prev = 0, ""
            t0 = time.monotonic()
            text = await asyncio.to_thread(self._run, prefix[done:], prev)
            cum = f"{prev} {text}".strip()
            self._prefix = self._prefix[-15:] + [(len(prefix), hashlib.sha1(prefix).digest(), cum)]
            log.debug("미리 인식 %.2fs (%.1f~%.1fs) %s", time.monotonic() - t0, done / 8000, len(prefix) / 8000, text)

    async def recognize(self, ulaw8k: bytes) -> str:
        t0 = time.monotonic()
        async with self._lock:  # CPU 한 개 모델을 순차 사용 (진행 중인 미리 인식이 끝나기를 기다린다)
            done, prev = self._cached(ulaw8k)
            rest = ulaw8k[done:]
            if done and rms(rest) < self.quiet_rms:
                text = prev  # 남은 부분은 단락 끝 무음뿐이다
            else:
                text = f"{prev} {await asyncio.to_thread(self._run, rest, prev)}".strip()
        self._prefix.clear()
        log.info("인식 %.2fs (음성 %.1fs, 미리 인식 %.1fs 사용) %s", time.monotonic() - t0, len(ulaw8k) / 8000,
                 done / 8000, text)
        return _clean(text)


class OpenAIRecognizer:
    """OpenAI 음성 인식 API. 오디오가 서버 밖으로 나가므로 is_local=False (민감정보 재확인 구간은 인식하지 않는다)."""

    is_local = False

    def __init__(self, model: str = "gpt-4o-transcribe") -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=1)

    async def recognize(self, ulaw8k: bytes) -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(ulaw8k_to_pcm16k(ulaw8k))
        buf.name = "ars.wav"
        buf.seek(0)
        try:
            r = await self.client.audio.transcriptions.create(model=self.model, file=buf, language="ko",
                                                              prompt=ARS_PROMPT)
        except Exception as e:
            log.warning("STT 실패: %s", e)
            return ""
        return _clean(r.text)


def make_recognizer(kind: str, model: str = "small", device: str = "cpu", quiet_rms: float = 500) -> Recognizer:
    if kind == "whisper":
        return WhisperRecognizer(model, device, int(os.environ.get("WHISPER_BEAM", "1")), quiet_rms)
    if kind == "openai":
        return OpenAIRecognizer(os.environ.get("OPENAI_STT_MODEL", "gpt-4o-transcribe"))
    return NullRecognizer()
