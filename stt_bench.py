"""
STT 모델 비교 (실통화 없음)

ARS 안내 문장을 Windows 한국어 TTS(Heami)로 합성하고 전화 음질(8kHz μ-law, 약한 잡음)로 바꿔
Whisper 모델별 인식 결과·문자 오류율(CER)·처리 시간을 비교한다. 실제 ARS 음성과는 다르므로 참고용이며,
실통화 녹음(api_check.py --audio 의 WAV)으로도 비교할 수 있다.

  python stt_bench.py --models small large-v3-turbo
  python stt_bench.py --models large-v3-turbo --wav audio_check.wav     # 녹음 파일 인식 (정답 없음)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from clawops.agent._audio import pcm16_to_ulaw

from stt import make_recognizer

HERE = Path(__file__).parent

EXTRA = [
    "국민건강보험공단입니다. 자격 및 보험료 관련 문의는 1번, 건강검진 안내는 2번, 상담사 연결은 9번을 눌러 주십시오.",
    "보험료 납부 확인은 1번, 납부 방법 변경은 2번, 이전 단계로 가시려면 별표를 눌러 주세요.",
    "입력하신 번호는 별 별 별 별 별 별입니다. 맞으면 1번, 아니면 2번을 눌러 주십시오.",
    "지금은 상담 가능 시간이 아닙니다. 평일 오전 9시부터 오후 6시까지 이용해 주시기 바랍니다.",
]


def tts_wav(text: str, path: Path) -> None:
    src = path.with_suffix(".txt")
    src.write_text(text, encoding="utf-8")  # 콘솔 입력은 cp949 로 읽혀 한글이 깨지므로 파일로 넘긴다
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.SelectVoice('Microsoft Heami Desktop');"
        "$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(8000,'Sixteen','Mono');"
        f"$s.SetOutputToWaveFile('{path}',$f);"
        f"$s.Speak((Get-Content -Raw -Encoding UTF8 '{src}'));$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def wav_to_ulaw(path: Path, noise: float = 0.0) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 8000 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: 8kHz 16bit mono WAV 만 지원")
        pcm = bytearray(w.readframes(w.getnframes()))
    if noise:
        import array

        a = array.array("h", bytes(pcm))
        for i in range(len(a)):
            a[i] = max(-32768, min(32767, a[i] + int(random.gauss(0, noise))))
        pcm = bytearray(a.tobytes())
    return pcm16_to_ulaw(bytes(pcm))


def cer(ref: str, hyp: str) -> float:
    r = [c for c in ref if not c.isspace() and c not in ".,"]
    h = [c for c in hyp if not c.isspace() and c not in ".,"]
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(h)] / max(1, len(r))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["small", "large-v3-turbo"])
    ap.add_argument("--wav", help="인식할 8kHz WAV (지정 시 TTS 생략)")
    ap.add_argument("--noise", type=float, default=300.0, help="합성 음성에 더할 잡음 크기 (PCM 표준편차)")
    args = ap.parse_args()

    if args.wav:
        clips = [("", wav_to_ulaw(Path(args.wav)))]
    else:
        tree = json.loads((HERE / "sim_kb.json").read_text(encoding="utf-8"))
        texts = [n["say"] for n in tree["nodes"].values()] + EXTRA
        tmp = Path(tempfile.mkdtemp())
        clips = []
        for i, t in enumerate(texts):
            p = tmp / f"{i}.wav"
            tts_wav(t, p)
            clips.append((t, wav_to_ulaw(p, args.noise)))

    for name in args.models:
        rec = make_recognizer("whisper", name, "cpu")
        t0 = time.monotonic()
        await rec.warmup()
        print(f"\n=== {name} (로드 {time.monotonic() - t0:.1f}s) ===")
        total_cer, total_t, total_audio = 0.0, 0.0, 0.0
        for ref, ulaw in clips:
            t0 = time.monotonic()
            hyp = await rec.recognize(ulaw)
            dt = time.monotonic() - t0
            sec = len(ulaw) / 8000
            total_t, total_audio = total_t + dt, total_audio + sec
            if ref:
                c = cer(ref, hyp)
                total_cer += c
                print(f"[{dt:4.1f}s/{sec:4.1f}s CER {c:4.0%}] {hyp}")
                if c > 0.05:
                    print(f"{'':22}정답: {ref}")
            else:
                print(f"[{dt:4.1f}s/{sec:4.1f}s] {hyp}")
        if clips[0][0]:
            print(f"평균 CER {total_cer / len(clips):.1%}, 처리 {total_t:.1f}s / 음성 {total_audio:.1f}s "
                  f"(실시간 대비 {total_t / total_audio:.2f}배)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
