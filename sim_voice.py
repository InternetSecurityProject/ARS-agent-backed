"""
음성 가상 ARS (전화 연결 없이 실행, Windows 전용)

가상 ARS(SimCall)와 같은 메뉴 트리 사용, 안내는 문장이 아닌 소리로 전달.
  - 안내 문장을 Windows 한국어 TTS(Microsoft Heami)로 합성해 8kHz μ-law 20ms 프레임으로 실시간 송출
    (말하지 않는 구간은 약한 회선 잡음). 서버는 실제 통화와 같은 Segmenter → STT → 해석 경로로 처리.
  - 서버가 보낸 키는 DTMF 톤으로 만든 뒤 detect_dtmf 로 해독하여 수신 (톤 생성·인식 확인).
  - 안내 중 키를 받으면 안내를 끊고 다음 안내로 이동 (ARS 끼어들기).
합성 음성은 .sim_audio/ 에 저장해 재사용.

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
        src.write_text(text, encoding="utf-8")  # 콘솔 입력의 cp949 해석에 따른 한글 깨짐 방지: 파일로 전달
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
    """안내 문장의 합성 음성 (μ-law 8kHz). 없으면 합성 후 저장."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / (hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] + ".ulaw")
    if not f.exists():
        f.write_bytes(_synth(text))
    return f.read_bytes()


def hold_music(seconds: float) -> bytes:
    """상담원 연결 대기 음악 (단순한 화음 선율, 말소리만큼 크게, 끊김 없이 이어짐)."""
    import math
    import struct

    notes = [523.3, 659.3, 784.0, 659.3, 587.3, 698.5, 880.0, 698.5]  # 도미솔미 레파라파
    n_note = int(0.35 * 8000)
    pcm = []
    for i in range(int(seconds * 8000)):
        f = notes[(i // n_note) % len(notes)]
        k = i % n_note
        env = min(1.0, k / 200, (n_note - k) / 200)  # 음 사이 딸깍 소리 방지
        v = 0.6 * math.sin(2 * math.pi * f * i / 8000) + 0.3 * math.sin(2 * math.pi * f * 1.5 * i / 8000)
        pcm.append(int(9000 * env * v))
    return pcm16_to_ulaw(struct.pack(f"<{len(pcm)}h", *pcm))


def prepare(tree: dict) -> None:
    """메뉴 트리의 모든 안내를 미리 합성 (첫 실행 시 문장당 1~2초)."""
    texts = [n["say"] for n in tree["nodes"].values()] + [n["human"] for n in tree["nodes"].values() if n.get("human")]
    texts += [tree.get("timeout_say", "입력 시간이 초과되었습니다."), "잘못 누르셨습니다.", "입력이 없어 통화를 종료합니다.",
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
        self.said_end: float | None = None  # 마지막 안내를 끝까지 말한 시각 (키 응답 시간 측정)
        self._noise = [_noise_frame() for _ in range(25)]
        self._cut = False
        self.human_at: float | None = None

    async def _say(self, text: str) -> None:
        if self.ended:
            return
        ulaw = await asyncio.to_thread(speech_ulaw, text)
        self._out += ulaw
        self.speaking, self.said_end = True, None
        while self._out and not self.ended:  # 다 말하거나 키 입력으로 끊길 때까지
            await asyncio.sleep(0.02)
        self.speaking = False
        if not self._cut:
            self.said_end = time.monotonic()
        self._cut = False

    async def _play_node(self) -> None:
        n = self.node
        if n.get("type") == "agent" and n.get("hold_s"):
            # 상담원 연결: 대기 안내 → 대기 음악 → 상담원이 받으면 음악이 끊기고 바로 발화 (사이에 무음 없음)
            self._input_gen += 1
            gen = self._input_gen
            await self._say(n["say"])
            self._out += hold_music(n["hold_s"] * self.speed)
            self.speaking = True
            while self._out and not self.ended and gen == self._input_gen:
                await asyncio.sleep(0.02)
            self.speaking = False
            if gen == self._input_gen and not self.ended:
                log.info("가상 ARS 상담원 응답 (대기 음악 %.0f초 후)", n["hold_s"] * self.speed)
                self.human_at = time.monotonic()  # 상담원이 말하기 시작한 시각 (상담원 인식 지연 측정)
                await self._say(n.get("human", "상담원 김민지입니다. 무엇을 도와드릴까요?"))
            return
        await super()._play_node()

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
        pass  # 사용자 마이크 음성 (상담원 연결 중). 가상 ARS 는 듣지 않음

    async def send_keys(self, keys: str) -> float:
        tone = dtmf_ulaw(keys)
        heard = detect_dtmf(tone)  # 실제 ARS 처럼 톤을 듣고 키 판별
        if heard != keys:
            log.warning("DTMF 인식 불일치: 보낸 %d자리, 들은 %d자리", len(keys), len(heard))  # 값은 남기지 않음 (민감정보일 수 있음)
        dur = len(tone) / 8000
        self._spawn(self._keys_after(heard, dur))
        return dur

    def _shown(self, keys: str) -> str:
        """로그에 남길 키: 메뉴의 한 자리 키만 값, 입력 노드(민감정보 입력)·여러 자리는 자릿수만 (한 자씩 입력도 한 자리씩 들어오므로 노드 종류로 구분)."""
        return keys if len(keys) == 1 and self.node.get("type") != "input" else f"{len(keys)}자리"

    async def _keys_after(self, keys: str, dur: float) -> None:
        await asyncio.sleep(dur)
        if self._out:
            log.info("키 응답: 안내 도중 '%s' (끼어들기)", self._shown(keys))
            self._cut = True
        elif self.said_end is not None:
            log.info("키 응답: 안내 끝 → '%s' %.1f초 (입력 대기 %.1f초)", self._shown(keys),
                     time.monotonic() - dur - self.said_end, self.tree.get("timeout_s", 12) * self.speed)
        self._out.clear()  # 끼어들기: 안내 중 키를 누르면 안내 중단
        for ch in keys:
            await self._key_later(ch)
