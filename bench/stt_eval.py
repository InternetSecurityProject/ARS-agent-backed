"""모델별 음성 인식 평가 (ARS 통화 시나리오 녹음).

  python -m bench.stt_eval --models kimthegarden/whisper-small-ko-low-qual-voice-ct2 gemini rtzr
  python -m bench.stt_eval --device cuda --models large-v3 qwen --scenarios K1,N2 --json bench/out/stt.json
  python -m bench.stt_eval --clips bench_real --models gemini        # clips.json 단락 폴더 직접 지정
  python -m bench.stt_eval --models gemini gemini+interp              # gemini+interp: 받아쓰기+분류 1회 호출 (결합 모드) 속도
  python -m bench.stt_eval --device cuda --clips bench_real --models qwen --prefetch   # 미리 인식을 켠 서버와 같은 조건

지표 (모델마다)
  글자 오류율      잘린 조각(cut)을 뺀 단락의 편집 거리 합 ÷ 정답 글자 수 합. 구역 A·B 별도 산출
  메뉴 키 정확도   메뉴·확인 단락의 (키, 이름) 쌍이 맞게 인식된 비율
  숫자열 정확도    되읽기 단락의 숫자열이 정답과 완전히 같은 비율, 자리 단위 오류율
  누락             정답이 있는데 빈 결과가 나온 단락 수
  환각             정답이 빈 단락(무음·잡음)에서 글자가 나온 단락 수
  속도             단락당 인식 시간 중앙값·95백분위·최대, 음성 1초당 처리 시간
                   --prefetch: 서버의 미리 인식(STT_PREFETCH=1) 재현. 단락을 서버와 같은 Segmenter 에 넣어 쉼(0.3초)마다
                   prefetch 를 호출하고, 인식 장치 하나를 순서대로 쓴다는 가정으로 '단락 종료 → 인식 결과' 시간을 가상 시계로 계산
                   (조각 인식이 다음 조각이나 단락 종료보다 늦게 끝나면 그 지연 가산). 인식 시간 열에 이 값 사용
결과 JSON 에 단락별 인식 문장 포함. interp_eval --hyp 로 '인식 문장 → 해석' 연결 평가에 사용.
외부 엔진(gemini·rtzr·clova·openai)은 단락 음성을 외부 API 로 전송. 벤치 녹음은 가짜 번호만 쓰므로 개인정보 없음.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from clawops.agent._audio import pcm16_to_ulaw

from bench import metrics as M
from bench.data import DATA, load_clip_folder, load_recordings


async def recognize_prefetch(rec, ulaw: bytes, min_chunk_ms: int | None = None) -> tuple[str, float, float]:
    """서버의 미리 인식 재현. (인식 문장, 단락 종료 → 결과 시간, 미리 인식에 쓴 시간 합)."""
    import os

    from audio import FRAME, Segmenter

    events: list[tuple[bytes, int]] = []
    from config import Settings

    seg = Segmenter(float(os.environ.get("VAD_RMS", "500")), 1200)
    # 미리 인식으로 넘길 최소 조각 길이. 지정하지 않으면 서버와 같은 값 (STT_PREFETCH_MIN_S, 기본 5초)
    seg.min_chunk_ms = min_chunk_ms if min_chunk_ms is not None else int(Settings().stt_prefetch_min_s * 1000)
    seg.on_chunk = lambda prefix, start: events.append((prefix, start))
    para = None
    for i in range(0, len(ulaw) - FRAME + 1, FRAME):
        out = seg.feed(ulaw[i:i + FRAME])
        if out is not None and para is None:
            para = out
    if para is None:
        para = bytes(seg.buf) if seg.speaking else ulaw
    events = [(p, st) for p, st in events if len(p) <= len(para) and para.startswith(p)]
    busy = spent = 0.0
    for prefix, start in events:
        t = time.monotonic()
        try:
            await rec.prefetch(prefix, start)
        except Exception as e:  # 서버(app.on_chunk)와 동일: 미리 인식 실패는 무시하고 단락 인식으로 진행
            logging.getLogger("bench.stt_eval").warning("미리 인식 실패: %s: %s", type(e).__name__, e)
        d = time.monotonic() - t
        spent += d
        busy = max(busy, len(prefix) / 8000) + d  # 조각이 들어온 시각 이후, 앞 작업이 끝난 뒤에 시작
    end = len(para) / 8000
    t = time.monotonic()
    hyp = await rec.recognize(para)
    return hyp, max(0.0, busy - end) + (time.monotonic() - t), spent


async def eval_model(name: str, recs, device: str, make=None, prefetch: bool = False,
                     min_chunk_ms: int | None = None) -> dict:
    """한 모델로 모든 단락 인식 후 지표 산출. make 는 시험용 (이름 → 인식기)."""
    interp = None
    if make is not None:
        rec = make(name)
    elif name == "gemini+interp":
        # STT=gemini + GEMINI_STT_COMBINED=1 과 같은 경로: 받아쓰기와 분류·점수를 한 번에. 시간은 '인식+해석' 합계
        from interpreter import load_interpreter
        from stt import make_recognizer

        rec = make_recognizer("gemini", "", device)
        interp = load_interpreter("gemini_interpreter:GeminiInterpreter")
        rec.attach_interpreter(interp)
    else:
        from stt import make_recognizer
        from stt_bench import engine_of

        kind, model = engine_of(name)
        rec = make_recognizer(kind, model, device)
    goals = {}
    if interp is not None:
        from bench.data import load_scenarios

        goals = {k: v.get("goal", "") for k, v in load_scenarios().items()}
    t0 = time.monotonic()
    warm = getattr(rec, "warmup", None)
    if warm:
        await warm()
    load_s = time.monotonic() - t0
    clips = [(r, c) for r in recs for c in r.clips]
    if clips:
        await rec.recognize(pcm16_to_ulaw(clips[0][1].pcm())[:8000 * 3])  # 장치 준비 시간은 측정 제외

    rows = []
    for r, c in clips:
        ulaw = pcm16_to_ulaw(c.pcm())
        if interp is not None:
            interp.goal = goals.get(r.scenario, "")
        t = time.monotonic()
        spent = None
        try:
            if prefetch and hasattr(rec, "prefetch"):
                hyp, took, spent = await recognize_prefetch(rec, ulaw, min_chunk_ms)
            else:
                hyp = await rec.recognize(ulaw)
                took = time.monotonic() - t
            err = None
        except Exception as e:  # 엔진 오류도 결과에 기록
            hyp, err, took = "", f"{type(e).__name__}: {e}", time.monotonic() - t
        rows.append({"recording": r.name, "file": c.path.name, "zone": c.zone, "kind": c.kind, "cut": c.cut,
                     "ref": c.text, "hyp": hyp, "sec": round(len(ulaw) / 8000, 2), "time": round(took, 3),
                     "prefetch_time": None if spent is None else round(spent, 3),
                     "error": err, "options": c.options, "digits": c.digits})
    return summarize(name, rows, load_s)


def summarize(name: str, rows: list[dict], load_s: float = 0.0) -> dict:
    ed = {"all": [0, 0], "A": [0, 0], "B": [0, 0]}
    pairs_ok = pairs_n = 0
    dig_exact = dig_n = dig_edits = dig_len = 0
    missing = halluc = empty_ref = 0
    times, rtf = [], []
    for row in rows:
        times.append(row["time"])
        if row["sec"]:
            rtf.append(row["time"] / row["sec"])
        if not M.norm(row["ref"]):
            empty_ref += 1
            halluc += bool(M.norm(row["hyp"]))
            continue
        if not M.norm(row["hyp"]):
            missing += 1
        if not row["cut"]:
            e, n = M.cer_counts(row["ref"], row["hyp"])
            row["cer"] = round(e / max(1, n), 4)
            for z in ("all", row["zone"]):
                if z in ed:
                    ed[z][0] += e
                    ed[z][1] += n
            if row["options"]:
                ok, n_, detail = M.menu_pairs(row["hyp"], row["options"])
                row["pairs"] = detail
                pairs_ok += ok
                pairs_n += n_
        if row["digits"]:
            d = M.digit_result(row["digits"], row["hyp"])
            row["digit"] = d
            dig_n += 1
            dig_exact += d["exact"]
            dig_edits += d["edits"]
            dig_len += d["expected_len"]
    return {
        "model": name, "clips": len(rows), "load_s": round(load_s, 1),
        "cer": M.ratio(*ed["all"]), "cer_A": M.ratio(*ed["A"]), "cer_B": M.ratio(*ed["B"]),
        "menu_pairs": M.ratio(pairs_ok, pairs_n), "menu_pairs_n": f"{pairs_ok}/{pairs_n}",
        "digits_exact": M.ratio(dig_exact, dig_n), "digits_n": f"{dig_exact}/{dig_n}", "digit_error": M.ratio(dig_edits, dig_len),
        "missing": missing, "halluc": f"{halluc}/{empty_ref}",
        "time_p50": M.pct(times, 0.5), "time_p95": M.pct(times, 0.95), "time_max": max(times) if times else None,
        "rtf": M.pct(rtf, 0.5),
        "rows": rows,
    }


def table(results: list[dict]) -> str:
    head = ("| 모델 | 글자 오류율 | 구역 A | 구역 B | 메뉴 키 | 숫자열 완전 일치 | 숫자 자리 오류 | 누락 | 환각 "
            "| 단락당 중앙값 | 95% | 최대 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines = [head]
    for r in results:
        lines.append(f"| {r['model']} | {M.fmt(r['cer'])} | {M.fmt(r['cer_A'])} | {M.fmt(r['cer_B'])} "
                     f"| {M.fmt(r['menu_pairs'])} ({r['menu_pairs_n']}) | {M.fmt(r['digits_exact'])} ({r['digits_n']}) "
                     f"| {M.fmt(r['digit_error'])} | {r['missing']} | {r['halluc']} | {M.fmt(r['time_p50'], 's')} "
                     f"| {M.fmt(r['time_p95'], 's')} | {M.fmt(r['time_max'], 's')} |")
    return "\n".join(lines)


async def main() -> None:
    import config  # noqa: F401  (.env 로드)

    ap = argparse.ArgumentParser(description="모델별 음성 인식 평가")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--data", default=str(DATA), help="시나리오 녹음 폴더 (bench/data)")
    ap.add_argument("--clips", nargs="+", help="단락 폴더 (bench_real 등 clips.json 형식, 여러 개 가능)")
    ap.add_argument("--scenarios", help="쉼표로 구분한 시나리오 ID 또는 녹음 이름 (예: K1,N2-1)")
    ap.add_argument("--json", help="결과 저장 파일 (단락별 인식 문장 포함)")
    ap.add_argument("--verbose", action="store_true", help="단락별 결과 출력")
    ap.add_argument("--prefetch", action="store_true", help="서버의 미리 인식을 재현하여 '단락 종료 → 결과' 시간을 잰다")
    ap.add_argument("--min-chunk-s", type=float, nargs="+", default=[None],
                    help="--prefetch: 미리 인식 최소 조각 길이(초)를 바꿔 가며 잰다 (예: 1.5 5 8). 기본은 서버 설정값 STT_PREFETCH_MIN_S (5초)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    recs = [load_clip_folder(Path(c)) for c in a.clips] if a.clips else load_recordings(Path(a.data), set(a.scenarios.split(",")) if a.scenarios else None)
    n = sum(len(r.clips) for r in recs)
    if not n:
        raise SystemExit("평가할 단락이 없습니다. bench/data 에 make_clips.py 결과가 있는지 확인하십시오.")
    print(f"녹음 {len(recs)}개, 단락 {n}개, 장치 {a.device}")
    results = []
    for name in a.models:
        for mc in (a.min_chunk_s if a.prefetch else [None]):
            r = await eval_model(name, recs, a.device, prefetch=a.prefetch,
                                 min_chunk_ms=None if mc is None else int(mc * 1000))
            if a.prefetch:
                r["model"] = f"{name} (미리 인식{'' if mc is None else f' {mc:g}초'})"
            results.append(r)
            if a.verbose:
                for row in r["rows"]:
                    mark = f"CER {row['cer']:.0%}" if "cer" in row else ("환각" if not M.norm(row["ref"]) and M.norm(row["hyp"]) else "")
                    print(f"[{r['model']}] {row['recording']}/{row['file']} {row['time']:.2f}s {mark} {row['hyp']}")
    print("\n## 음성 인식 (모델별)\n" + table(results))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n저장: {a.json}")


if __name__ == "__main__":
    asyncio.run(main())
