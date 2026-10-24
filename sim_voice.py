"""
음성 가상 ARS (실통화 없음, Windows 전용)

가상 ARS(SimCall)와 같은 메뉴 트리를 쓰되, 안내를 문장이 아닌 소리로 보낸다.
  - 안내 문장을 Windows 한국어 TTS(Microsoft Heami)로 합성해 8kHz μ-law 20ms 프레임으로 실시간 송출한다
    (말하지 않는 구간은 약한 회선 잡음). 서버는 실통화와 같이 Segmenter → STT → 해석 경로로 처리한다.
  - 서버가 보낸 키는 DTMF 톤으로 만든 뒤 detect_dtmf 로 해독하여 받는다 (톤 생성·인식 확인).
  - 안내 중 키를 받으면 안내를 끊고 다음 안내로 넘어간다 (ARS 끼어들기).
합성 음성은 .sim_audio/ 에 저장해 두고 다시 쓴다.

  python app.py --sim --sim-voice
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from audio import FRAME, detect_dtmf, dtmf_ulaw
from clawops.agent._audio import pcm16_to_ulaw
from telephony import CallHandlers, SimCall

log = logging.getLogger("ars.simvoice")
CACHE = Path(__file__).parent / ".sim_audio"


def _noise_frame(level: float = 60.0) -> bytes:
    import struct

    return pcm16_to_ulaw(struct.pack(f"<{FRAME}h", *(int(random.gauss(0, level)) for _ in range(FRAME))))


def _synth(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        src, wav = Path(tmp) / "t.txt", Path(tmp) / "t.wav"
        src.write_text(text, encoding="utf-8")  # 콘솔 입력은 cp949 로 읽혀 한글이 깨지므로 파일로 넘긴다
        ps = ("Add-Type -AssemblyName System.Speech;"
              "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
              "$s.SelectVoice('Microsoft Heami Desktop');"
              "$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(8000,'Sixteen','Mono');"
              f"$s.SetOutputToWaveFile('{wav}',$f);"
              f"$s.Speak((Get-Content -Raw -Encoding UTF8 '{src}'));$s.Dispose()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
        with wave.open(str(wav), "rb") as w:
            return pcm16_to_ulaw(w.readframes(w.getnframes()))


def speech_ulaw(text: str) -> bytes:
    """안내 문장의 합성 음성 (μ-law 8kHz). 없으면 합성하여 저장한다."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / (hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] + ".ulaw")
    if not f.exists():
        f.write_bytes(_synth(text))
    return f.read_bytes()


def prepare(tree: dict) -> None:
    """메뉴 트리의 모든 안내를 미리 합성한다 (첫 실행 시 문장당 1~2초)."""
    texts = [n["say"] for n in tree["nodes"].values()] + [n["human"] for n in tree["nodes"].values() if n.get("human")]
    texts += ["잘못 누르셨습니다.", "입력 시간이 초과되었습니다.", "입력이 없어 통화를 종료합니다.",
              "상담원 김민지입니다. 무엇을 도와드릴까요?"]
    t0 = time.monotonic()
    for t in texts:
        speech_ulaw(t)
    log.info("음성 가상 ARS 준비: 안내 %d개 (%.1fs)", len(texts), time.monotonic() - t0)


class VoiceSimCall(SimCall):
    provides_transcripts = False

    def __init__(self, tree: dict, handlers: CallHandlers, speed: float) -> None:
        super().__init__(tree, handlers, speed, audio=True)
        self._out = bytearray()   # 송출할 안내 음성
        self._noise = [_noise_frame() for _ in range(25)]

    async def _say(self, text: str) -> None:
        if self.ended:
            return
        ulaw = await asyncio.to_thread(speech_ulaw, text)
        self._out += ulaw
        self.speaking = True
        while self._out and not self.ended:  # 다 말하거나 키 입력으로 끊길 때까지
            await asyncio.sleep(0.02)
        self.speaking = False

    async def _audio_loop(self) -> None:
        t0, i = time.monotonic(), 0
        while not self.ended:
            if self._out:
                frame = bytes(self._out[:FRAME]).ljust(FRAME, b"\xff")
                del self._out[:FRAME]
            else:
                frame = self._noise[i % len(self._noise)]
            await self.h.on_audio(frame)
            i += 1
            await asyncio.sleep(max(0.0, t0 + i * 0.02 - time.monotonic()))

    async def send_audio(self, ulaw: bytes) -> None:
        pass  # 사용자 마이크 음성 (상담원 연결 중). 가상 ARS 는 듣지 않는다

    async def send_keys(self, keys: str) -> float:
        tone = dtmf_ulaw(keys)
        heard = detect_dtmf(tone)  # 실제 ARS 처럼 톤을 듣고 키를 알아낸다
        if heard != keys:
            log.warning("DTMF 인식 불일치: 보낸 %s, 들은 %s", keys, heard)
        dur = len(tone) / 8000
        self._spawn(self._keys_after(heard, dur))
        return dur

    async def _keys_after(self, keys: str, dur: float) -> None:
        await asyncio.sleep(dur)
        self._out.clear()  # 끼어들기: 안내 중 키를 누르면 안내를 멈춘다
        for ch in keys:
            await self._key_later(ch)
