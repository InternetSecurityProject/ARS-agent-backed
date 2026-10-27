"""
STT 모델 비교 (실통화 없음)

ARS 안내 문장을 전화 음질(8kHz, 약한 잡음)로 만든 시험 음성으로 Whisper 모델별 인식 결과·문자 오류율(CER)·처리 시간을 비교한다.
실제 ARS 음성과는 다르므로 참고용이며, 실통화 녹음(api_check.py --audio 의 WAV)으로도 비교할 수 있다.

  python stt_bench.py --models small kimthegarden/whisper-small-ko-low-qual-voice-ct2              # CPU
  python stt_bench.py --device cuda --models small kimthegarden/whisper-small-ko-low-qual-voice-ct2 large-v3-turbo   # NVIDIA GPU
  python stt_bench.py --models large-v3-turbo --wav audio_check.wav     # 녹음 파일 인식 (정답 없음)

시험 음성: 기본은 저장소의 bench_clips/ (WAV + clips.json, 개인정보 없음, 어느 운영체제에서나 실행).
  bench_clips/ 는 Windows 한국어 TTS(Heami)로 만든다: python stt_bench.py --make-clips bench_clips
결과 마지막의 요약 표를 그대로 공유하면 장치(CPU·GPU)별로 비교할 수 있다.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import logging
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from clawops.agent._audio import pcm16_to_ulaw

import config  # noqa: F401  .env 로드 (WHISPER_DEVICE 기본값)
from stt import make_recognizer

HERE = Path(__file__).parent
CLIPS = HERE / "bench_clips"

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


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 8000 or w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise SystemExit(f"{path}: 8kHz 16bit mono WAV 만 지원")
        return w.readframes(w.getnframes())


def add_noise(pcm: bytes, level: float, seed: int) -> bytes:
    rnd = random.Random(seed)  # 어느 컴퓨터에서 만들어도 같은 잡음
    a = array.array("h", pcm)
    for i in range(len(a)):
        a[i] = max(-32768, min(32767, a[i] + int(rnd.gauss(0, level))))
    return a.tobytes()


def bench_texts() -> list[str]:
    texts = []
    for name in ("sim_kb.json", "sim_kb_real.json"):
        tree = json.loads((HERE / name).read_text(encoding="utf-8"))
        nodes = tree["nodes"].values() if name == "sim_kb.json" else [tree["nodes"]["root"]]  # 실제 KB 첫 메뉴(29초)
        texts += [n["say"] for n in nodes]
    return texts + EXTRA


def make_clips(out: Path, noise: float) -> None:
    """시험 음성 묶음을 만든다 (Windows 전용). 전화 음질 8kHz + 고정 잡음으로 저장한다."""
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(bench_texts()):
            src = Path(tmp) / f"{i}.wav"
            tts_wav(text, src)
            name = f"clip{i:02d}.wav"
            with wave.open(str(out / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(8000)
                w.writeframes(add_noise(read_pcm(src), noise, seed=i))
            meta.append({"file": name, "text": text})
    (out / "clips.json").write_text(json.dumps({"noise": noise, "clips": meta}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(f"시험 음성 {len(meta)}개 저장: {out}")


def cer(ref: str, hyp: str) -> float:
    r = [c for c in ref if not c.isspace() and c not in ".,"]
    h = [c for c in hyp if not c.isspace() and c not in ".,"]
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(h)] / max(1, len(r))


def device_name(device: str) -> str:
    if device == "cuda":
        try:
            import ctranslate2

            return f"cuda (GPU {ctranslate2.get_cuda_device_count()}개)"
        except Exception as e:  # GPU 를 쓸 수 없으면 모델 로드에서 오류가 난다
            return f"cuda (확인 실패: {e})"
    return f"cpu ({os.cpu_count()}코어, {platform.processor() or platform.machine()})"


async def main() -> None:
    ap = argparse.ArgumentParser(description="STT 모델 비교 (실통화 없음)")
    ap.add_argument("--models", nargs="+", default=["small", "kimthegarden/whisper-small-ko-low-qual-voice-ct2"])
    ap.add_argument("--device", default=os.environ.get("WHISPER_DEVICE", "cpu"), choices=["cpu", "cuda"],
                    help="cpu / cuda (NVIDIA GPU). 기본: .env 의 WHISPER_DEVICE")
    ap.add_argument("--clips", default=str(CLIPS), help="시험 음성 폴더 (WAV + clips.json)")
    ap.add_argument("--wav", help="인식할 8kHz WAV 하나 (정답 없음)")
    ap.add_argument("--make-clips", metavar="DIR", help="시험 음성 묶음을 만든다 (Windows 한국어 TTS)")
    ap.add_argument("--noise", type=float, default=300.0, help="--make-clips 때 더할 잡음 크기 (PCM 표준편차)")
    ap.add_argument("--quiet", action="store_true", help="문장별 결과 없이 요약만 출력")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    if args.make_clips:
        make_clips(Path(args.make_clips), args.noise)
        return
    if args.wav:
        clips = [("", pcm16_to_ulaw(read_pcm(Path(args.wav))))]
    else:
        folder = Path(args.clips)
        meta = json.loads((folder / "clips.json").read_text(encoding="utf-8"))["clips"]
        clips = [(m["text"], pcm16_to_ulaw(read_pcm(folder / m["file"]))) for m in meta]

    total_audio = sum(len(u) for _, u in clips) / 8000
    print(f"장치: {device_name(args.device)} / 시험 음성 {len(clips)}개, 합계 {total_audio:.0f}초")
    summary = []
    for name in args.models:
        rec = make_recognizer("whisper", name, args.device)
        t0 = time.monotonic()
        await rec.warmup()
        load = time.monotonic() - t0
        await rec.recognize(clips[0][1][:8000 * 3])  # 첫 인식은 장치 준비 시간이 섞이므로 재지 않는다
        print(f"\n=== {name} (로드 {load:.1f}s) ===")
        cers, times = [], []
        for ref, ulaw in clips:
            t0 = time.monotonic()
            hyp = await rec.recognize(ulaw)
            dt = time.monotonic() - t0
            times.append(dt)
            sec = len(ulaw) / 8000
            if ref:
                c = cer(ref, hyp)
                cers.append(c)
                if not args.quiet:
                    print(f"[{dt:4.2f}s/{sec:4.1f}s CER {c:4.0%}] {hyp}")
                    if c > 0.05:
                        print(f"{'':24}정답: {ref}")
            elif not args.quiet:
                print(f"[{dt:4.2f}s/{sec:4.1f}s] {hyp}")
        summary.append((name, statistics.mean(cers) if cers else None, statistics.median(times), max(times), sum(times)))

    print(f"\n## 요약 ({device_name(args.device)}, 시험 음성 {len(clips)}개 {total_audio:.0f}초)")
    print("| 모델 | 문자 오류율 | 단락당 인식 시간 (중앙값) | 최대 | 전체 처리 |")
    print("| --- | --- | --- | --- | --- |")
    for name, c, med, mx, tot in summary:
        print(f"| {name} | {'—' if c is None else f'{c:.1%}'} | {med:.2f}초 | {mx:.2f}초 | {tot:.1f}초 |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
