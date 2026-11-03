"""녹음 원본 → 단락 WAV + clips.json 초안.

  python -m bench.make_clips bench/raw/K1-1.json
  python -m bench.make_clips bench/raw/*.json --stt gemini --label        # 정답 초안을 Gemini 로, 안내 종류·선택지 초안까지

하는 일
  1. 서버와 같은 단락 판정(RMS 500, 무음 1.2초)으로 녹음을 자른다 (bench/data/<녹음>/<녹음>_NN.wav)
  2. 구역: 첫 번호 입력 전에 시작한 단락은 A, 이후는 B
  3. 판단 지점: 키를 누르기 직전에 끝난 단락에 그 키의 decision 번호를 적는다
  4. 숫자 정답: 번호 입력 직후의 첫 단락에 입력한 가짜 번호를 digits 로 적는다 (되읽기가 아니면 사람이 지운다)
  5. (선택) --stt 로 정답 문장 초안, --label 로 안내 종류·선택지 초안
초안은 verified=false 이다. 사람이 단락을 들으며 text·kind·options·digits·cut 을 고친 뒤 verified=true 로 바꾼다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

from clawops.agent._audio import pcm16_to_ulaw, ulaw_to_pcm16

from audio import FRAME, Segmenter
from bench.data import DATA


def split(ulaw: bytes) -> list[tuple[int, int, bytes]]:
    """(시작 샘플, 끝 샘플, 단락 μ-law). 서버 Segmenter 와 같은 판정. 단락은 소리 시작 200ms 전부터 무음 1.2초 끝까지다."""
    seg = Segmenter()
    out = []
    for i in range(0, len(ulaw) - FRAME + 1, FRAME):
        s = seg.feed(ulaw[i:i + FRAME])
        if s is not None:
            end = i + FRAME
            out.append((end - len(s), end, s))
    return out


async def draft_texts(clips: list[bytes], stt: str, device: str) -> list[str]:
    from stt import make_recognizer
    from stt_bench import engine_of

    kind, model = engine_of(stt)
    rec = make_recognizer(kind, model, device)
    warm = getattr(rec, "warmup", None)
    if warm:
        await warm()
    return [await rec.recognize(u) for u in clips]


async def draft_labels(texts: list[str]) -> list[tuple[str | None, list]]:
    from interpreter import load_interpreter

    interp = load_interpreter("gemini_interpreter:GeminiInterpreter")
    out = []
    for t in texts:
        if not t.strip():
            out.append((None, []))
            continue
        try:
            p = await interp.classify(t)
            out.append((p.kind, [[o.key, o.label] for o in p.all_options()]))
        except Exception:
            out.append((None, []))
    return out


async def make(raw_json: Path, out_root: Path, stt: str | None, label: bool, device: str) -> Path:
    meta = json.loads(raw_json.read_text(encoding="utf-8"))
    name = raw_json.stem
    with wave.open(str(raw_json.with_name(meta["wav"])), "rb") as w:
        ulaw = pcm16_to_ulaw(w.readframes(w.getnframes()))
    parts = split(ulaw)
    acts = [a for a in meta["actions"] if a["type"] in ("key", "input")]
    first_input = next((a["at_sample"] for a in acts if a["type"] == "input"), None)

    clips = []
    for n, (start, end, _) in enumerate(parts, 1):
        clips.append({"file": f"{name}_{n:02d}.wav", "start_s": round(start / 8000, 2), "end_s": round(end / 8000, 2),
                      "text": "", "zone": "B" if first_input is not None and start >= first_input else "A",
                      "kind": None, "options": [], "digits": None, "cut": False, "decision": None, "verified": False})
    # 판단 지점: 키를 누르기 직전에 끝난 단락 (decision 번호 = 시나리오 키 경로의 판단 단계 순서)
    for k, a in enumerate(acts):
        before = [c for (s, e, _), c in zip(parts, clips) if e <= a["at_sample"] + 4000]  # 단락 끝 판정과 키 사이 여유 0.5초
        if before:
            before[-1]["decision"] = k
        if a["type"] == "input" and a.get("digits"):
            after = [c for (s, e, _), c in zip(parts, clips) if s >= a["at_sample"]]
            if after:
                after[0]["digits"] = a["digits"]
    if stt:
        texts = await draft_texts([u for _, _, u in parts], stt, device)
        for c, t in zip(clips, texts):
            c["text"] = t
        if label:
            for c, (kind, opts) in zip(clips, await draft_labels(texts)):
                c["kind"], c["options"] = kind, opts

    folder = out_root / name
    folder.mkdir(parents=True, exist_ok=True)
    for (s, e, u), c in zip(parts, clips):
        with wave.open(str(folder / c["file"]), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(ulaw_to_pcm16(u))
    path = folder / "clips.json"
    path.write_text(json.dumps({"scenario": meta["scenario"], "source": meta["wav"], "draft_stt": stt,
                                "clips": clips}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: 단락 {len(clips)}개 → {path}")
    return path


async def main() -> None:
    import config  # noqa: F401  (.env 로드)

    ap = argparse.ArgumentParser(description="녹음 → 단락 + 정답 초안")
    ap.add_argument("raw", nargs="+", help="bench/raw/<녹음>.json")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--stt", help="정답 문장 초안을 만들 STT (예: gemini)")
    ap.add_argument("--label", action="store_true", help="안내 종류·선택지 초안 (Gemini 해석기)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    for r in a.raw:
        await make(Path(r), Path(a.out), a.stt, a.label, a.device)


if __name__ == "__main__":
    asyncio.run(main())
