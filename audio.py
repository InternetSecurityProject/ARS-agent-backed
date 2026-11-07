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
from collections import deque

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
    if len(out) % FRAME:
        out += b"\xff" * (FRAME - len(out) % FRAME)  # 20ms(160바이트) 프레임 단위로 맞춘다 (마지막 프레임이 잘리지 않게)
    return bytes(out)


def hum_ulaw(ms: int = 20) -> bytes:
    """가상 ARS 가 말하는 구간을 표시하는 시험음 (400Hz).
    20ms 프레임에 정확히 8주기가 들어가 프레임 사이가 끊기지 않으며, 오디오 채널 확인 시 들을 수 있는 크기로 둔다."""
    return pcm16_to_ulaw(_pcm((400,), ms, 0.15))


SILENCE_FRAME = b"\xff" * FRAME


class FrameChunker:
    """받은 오디오를 20ms(160바이트) 프레임으로 나눈다. 160바이트에 못 미치는 나머지는 다음 데이터와 이어 붙인다.
    (앱 녹음 단위가 160의 배수가 아니면 메시지마다 나머지를 버릴 때 소리가 끊긴다. 예: 1024바이트 → 64바이트씩 손실)"""

    def __init__(self) -> None:
        self.rest = b""

    def feed(self, data: bytes) -> list[bytes]:
        buf = self.rest + data
        n = len(buf) - len(buf) % FRAME
        self.rest = buf[n:]
        return [buf[i:i + FRAME] for i in range(0, n, FRAME)]

_ROWS = (697, 770, 852, 941)
_COLS = (1209, 1336, 1477)
_KEY_AT = {hz: k for k, hz in DTMF_HZ.items()}


def _goertzel(x, freq: float) -> float:
    import numpy as np

    n = len(x)
    w = 2 * math.pi * round(n * freq / RATE) / n
    c = 2 * math.cos(w)
    s1 = s2 = 0.0
    for v in x:
        s1, s2 = v + c * s1 - s2, s1
    return float(s1 * s1 + s2 * s2 - c * s1 * s2) / n / max(1.0, float(np.dot(x, x)))


def detect_dtmf(ulaw: bytes, block_ms: int = 40) -> str:
    """μ-law 오디오에서 DTMF 키 열을 찾는다 (Goertzel).
    행·열 주파수가 각각 가장 강하고 신호 에너지 대부분을 차지하는 블록이 2개 이상 이어지면 한 번 누른 것으로 본다.
    가상 ARS 의 키 수신과 톤 생성 검증에 쓴다."""
    import numpy as np

    pcm = np.frombuffer(ulaw_to_pcm16(ulaw), dtype=np.int16).astype(np.float64)
    n = RATE * block_ms // 1000
    out, prev, run = [], None, 0
    for i in range(0, len(pcm) - n + 1, n):
        x = pcm[i:i + n]
        key = None
        if np.sqrt(np.mean(x * x)) > 300:
            rows = [_goertzel(x, f) for f in _ROWS]
            cols = [_goertzel(x, f) for f in _COLS]
            r, c = int(np.argmax(rows)), int(np.argmax(cols))
            # 두 주파수가 에너지의 대부분이고 나머지 행·열보다 뚜렷이 강해야 한다 (말소리와 구분)
            if (rows[r] + cols[c] > 0.15 and rows[r] > 4 * sorted(rows)[-2]
                    and cols[c] > 4 * sorted(cols)[-2]):
                key = _KEY_AT.get((_ROWS[r], _COLS[c]))
        if key is not None and key == prev:
            run += 1
            if run == 2:
                out.append(key)
        else:
            run = 1 if key is not None else 0
        prev = key
    return "".join(out)


def rms(ulaw: bytes) -> float:
    a = array.array("h", ulaw_to_pcm16(ulaw))
    return (sum(x * x for x in a) / max(1, len(a))) ** 0.5


def ulaw8k_to_pcm16k(ulaw: bytes) -> bytes:
    return resample_pcm16(ulaw_to_pcm16(ulaw), from_rate=8000, to_rate=16000)


class Segmenter:
    """RMS 기준 발화 구간 검출.

    100ms 이상 소리가 나면 단락 시작, paragraph_silence_ms 이상 무음이면 단락 종료로 본다.
    mute() 로 지정한 구간(우리가 송출한 DTMF 톤의 반향)은 무시한다.

    미리 인식: on_chunk 가 있으면 단락 안에서 잠깐 쉴 때(pause_ms) 그때까지의 단락 음성과 직전에 넘긴 위치를
    on_chunk(prefix, start) 로 넘긴다. STT 가 ARS 가 말하는 동안 앞부분을 인식해 두어 단락이 끝난 뒤의 인식 시간을 줄인다.
    on_chunk 는 클래스 기본값으로 둘 수 있다 (서버가 시작할 때 STT 에 맞춰 지정, 동시 통화 1건).

    대기 음악: 말소리는 음절·쉼 때문에 크기가 크게 출렁이고(1초 변동계수 실녹음 0.5 이상), 대기 음악은 일정하다
    (0.2 안팎). 변동계수가 music_cv 미만인 소리가 music_ms 이상 이어지면 음악으로 보고 무음처럼 취급한다.
    그래서 대기 안내는 음악이 시작되면 끝나고, 음악은 STT 로 보내지 않으며, 상담원 목소리는 새 단락으로 시작한다.
    음악이 대부분인 단락은 버린다.
    """

    on_chunk = None          # Callable[[bytes, int], None] | None
    pause_ms = 300           # 단락 안의 쉼으로 볼 무음 길이
    min_chunk_ms = 1500      # 미리 인식으로 넘길 최소 길이 (너무 짧으면 낱말이 잘린다)
    music_cv = 0.35          # 1초 창 음량 변동계수가 이보다 작으면 음악 같은 소리
    music_ms = 2000          # 음악 같은 소리가 이만큼 이어지면 대기 음악으로 본다

    def __init__(self, threshold: float = 500, silence_ms: int = 1200, max_s: float = 45.0) -> None:
        self.threshold = threshold
        self.end_frames = silence_ms // 20
        self.max_frames = int(max_s * 50)
        self._levels: deque[float] = deque(maxlen=50)  # 최근 1초 프레임 음량
        self._music_run = 0
        self.music = False       # 지금 대기 음악이 나오는 중
        self.reset()
        self._mute_frames = 0

    def reset(self) -> None:
        self.buf = bytearray()
        self.speaking = False
        self.loud_run = 0
        self.quiet_run = 0
        self.frames = 0
        self.sent = 0  # 미리 인식으로 넘긴 위치 (바이트)
        self.music_frames = 0  # 이 단락에서 음악 같은 소리였던 프레임 수

    def mute(self, seconds: float) -> None:
        self._mute_frames = max(self._mute_frames, int(seconds * 50) + 5)

    def reset_for_key(self) -> None:
        """Begin receiving the next prompt before a key is transmitted."""
        self.reset()
        self._mute_frames=0
        self._levels.clear()
        self._music_run=0
        self.music=False

    def feed(self, frame: bytes) -> bytes | None:
        """프레임을 넣고, 단락이 끝났으면 그 단락의 μ-law 오디오를 반환한다."""
        if self._mute_frames > 0:
            self._mute_frames -= 1
            return None
        level = rms(frame)
        like_music = self._track_music(level)
        loud = level > self.threshold and not self.music
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
        self.music_frames += like_music
        self.quiet_run = 0 if loud else self.quiet_run + 1
        if (self.on_chunk and self.quiet_run == self.pause_ms // 20 and self.quiet_run < self.end_frames
                and len(self.buf) - self.sent >= self.min_chunk_ms * 8
                and self.music_frames < 0.6 * self.frames):  # 음악뿐인 조각은 인식하지 않는다 (CPU 를 상담원 목소리에 남긴다)
            self.on_chunk(bytes(self.buf), self.sent)
            self.sent = len(self.buf)
        if self.quiet_run >= self.end_frames or self.frames >= self.max_frames:
            seg = bytes(self.buf)
            mostly_music = self.music_frames > 0.6 * self.frames
            self.reset()
            return None if mostly_music else seg
        return None

    def _track_music(self, level: float) -> bool:
        """이 프레임이 음악 같은 소리인지(최근 1초 음량이 크고 일정한지) 판단하고 대기 음악 여부를 갱신한다."""
        self._levels.append(level)
        like = False
        if len(self._levels) == self._levels.maxlen:
            mean = sum(self._levels) / len(self._levels)
            if mean > self.threshold:
                sd = (sum((v - mean) ** 2 for v in self._levels) / len(self._levels)) ** 0.5
                like = sd / mean < self.music_cv
        self._music_run = self._music_run + 1 if like else 0
        self.music = self._music_run >= self.music_ms // 20
        return like
