"""
음성 인식(STT)

Recognizer 프로토콜을 구현하면 다른 STT 로 교체할 수 있다.
  - is_local: 서버 밖으로 오디오를 보내지 않으면 True. 민감정보 입력 구간(ARS 가 입력값을 다시 읽어 주는 구간)은
              로컬 STT 일 때만 인식하고, 결과의 숫자는 마스킹한다. 외부 STT 이면 해당 구간을 인식하지 않는다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from audio import ulaw8k_to_pcm16k

log = logging.getLogger("ars.stt")


class Recognizer(Protocol):
    is_local: bool

    async def recognize(self, ulaw8k: bytes) -> str: ...


class NullRecognizer:
    """인식하지 않는다. 가상 ARS(sim) 처럼 백엔드가 문장을 직접 주는 경우에 사용한다."""

    is_local = True

    async def recognize(self, ulaw8k: bytes) -> str:
        return ""


class WhisperRecognizer:
    """faster-whisper 로컬 인식. 최초 호출 시 모델을 내려받는다(small 약 460MB)."""

    is_local = True
    PROMPT = "잔액조회는 1번, 상담원 연결은 0번, 이전 메뉴는 별표, 다시 들으시려면 우물정자를 눌러 주십시오."

    def __init__(self, model: str = "small", device: str = "cpu") -> None:
        self._name, self._device = model, device
        self._model = None
        self._lock = asyncio.Lock()

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("Whisper 모델 로드: %s (%s)", self._name, self._device)
            ct = "float16" if self._device == "cuda" else "int8"
            self._model = WhisperModel(self._name, device=self._device, compute_type=ct)
        return self._model

    def _run(self, ulaw8k: bytes) -> str:
        import numpy as np

        model = self._load()
        pcm = np.frombuffer(ulaw8k_to_pcm16k(ulaw8k), dtype=np.int16).astype(np.float32) / 32768.0
        segs, _ = model.transcribe(pcm, language="ko", beam_size=1, vad_filter=False,
                                   initial_prompt=self.PROMPT, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    async def recognize(self, ulaw8k: bytes) -> str:
        async with self._lock:  # CPU 한 개 모델을 순차 사용
            return await asyncio.to_thread(self._run, ulaw8k)


def make_recognizer(kind: str, model: str = "small", device: str = "cpu") -> Recognizer:
    if kind == "whisper":
        return WhisperRecognizer(model, device)
    return NullRecognizer()
