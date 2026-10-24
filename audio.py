"""
오디오 처리

  - G.711 μ-law 8kHz 프레임(20ms, 160바이트) 변환
  - 대역 내(in-band) DTMF 톤 생성: 실통화 시험에서 ARS 인식이 확인된 방식
  - 발화 구간 검출(Segmenter): ARS 안내 한 단락을 잘라 STT 로 넘긴다
"""

from __future__ import annotations

import array
import math
import struct

from clawops.agent._audio import pcm16_to_ulaw, resample_pcm16, ulaw_to_pcm16

RATE = 8000
FRAME = 160  # 20ms

DTMF_HZ = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477),
}
VALID_KEYS = set(DTMF_HZ)


def _pcm(freqs: tuple[float, ...], ms: int, amp: float) -> bytes:
    n = RATE * ms // 1000
    out = [int(amp * 32767 * sum(math.sin(2 * math.pi * f * i / RATE) for f in freqs) / len(freqs)) for i in range(n)]
    return struct.pack(f"<{n}h", *out)


def _silence(ms: int) -> bytes:
    return b"\x00\x00" * (RATE * ms // 1000)


_TONE_CACHE: dict[str, bytes] = {}


def dtmf_ulaw(keys: str, tone_ms: int = 150, gap_ms: int = 100) -> bytes:
    """키 문자열을 DTMF 톤 μ-law 오디오로 변환한다."""
    out = bytearray()
    for ch in keys:
        if ch not in DTMF_HZ:
            raise ValueError(f"유효하지 않은 키: {ch}")
        if ch not in _TONE_CACHE:
            _TONE_CACHE[ch] = pcm16_to_ulaw(_pcm(DTMF_HZ[ch], tone_ms, 0.5) + _silence(gap_ms))
        out += _TONE_CACHE[ch]
    return bytes(out)


def hum_ulaw(ms: int = 20) -> bytes:
    """가상 ARS 가 말하는 구간을 표시하는 시험음 (400Hz).
    20ms 프레임에 정확히 8주기가 들어가 프레임 사이가 끊기지 않으며, 오디오 채널 확인 시 들을 수 있는 크기로 둔다."""
    return pcm16_to_ulaw(_pcm((400,), ms, 0.15))


SILENCE_FRAME = b"\xff" * FRAME


def rms(ulaw: bytes) -> float:
    a = array.array("h", ulaw_to_pcm16(ulaw))
    return (sum(x * x for x in a) / max(1, len(a))) ** 0.5


def ulaw8k_to_pcm16k(ulaw: bytes) -> bytes:
    return resample_pcm16(ulaw_to_pcm16(ulaw), from_rate=8000, to_rate=16000)


class Segmenter:
    """RMS 기준 발화 구간 검출.

    100ms 이상 소리가 나면 단락 시작, paragraph_silence_ms 이상 무음이면 단락 종료로 본다.
    mute() 로 지정한 구간(우리가 송출한 DTMF 톤의 반향)은 무시한다.
    """

    def __init__(self, threshold: float = 500, silence_ms: int = 1200, max_s: float = 30.0) -> None:
        self.threshold = threshold
        self.end_frames = silence_ms // 20
        self.max_frames = int(max_s * 50)
        self.reset()
        self._mute_frames = 0

    def reset(self) -> None:
        self.buf = bytearray()
        self.speaking = False
        self.loud_run = 0
        self.quiet_run = 0
        self.frames = 0

    def mute(self, seconds: float) -> None:
        self._mute_frames = max(self._mute_frames, int(seconds * 50) + 5)

    def feed(self, frame: bytes) -> bytes | None:
        """프레임을 넣고, 단락이 끝났으면 그 단락의 μ-law 오디오를 반환한다."""
        if self._mute_frames > 0:
            self._mute_frames -= 1
            return None
        loud = rms(frame) > self.threshold
        if not self.speaking:
            self.loud_run = self.loud_run + 1 if loud else 0
            self.buf.extend(frame)
            if len(self.buf) > FRAME * 10:  # 시작 직전 200ms 만 유지
                del self.buf[: len(self.buf) - FRAME * 10]
            if self.loud_run >= 5:
                self.speaking, self.quiet_run, self.frames = True, 0, len(self.buf) // FRAME
            return None
        self.buf.extend(frame)
        self.frames += 1
        self.quiet_run = 0 if loud else self.quiet_run + 1
        if self.quiet_run >= self.end_frames or self.frames >= self.max_frames:
            seg = bytes(self.buf)
            self.reset()
            return seg
        return None
