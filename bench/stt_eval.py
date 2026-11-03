"""모델별 음성 인식 평가 (실통화 시나리오 녹음).

  python -m bench.stt_eval --models kimthegarden/whisper-small-ko-low-qual-voice-ct2 gemini rtzr
  python -m bench.stt_eval --device cuda --models large-v3 qwen --scenarios K1,N2 --json bench/out/stt.json
  python -m bench.stt_eval --clips bench_real --models gemini        # 기존 단락 폴더 형식
  python -m bench.stt_eval --models gemini gemini+interp              # gemini+interp: 받아쓰기+분류 1회 호출 (결합 모드) 속도

지표 (모델마다)
  글자 오류율      잘린 조각(cut)을 뺀 단락의 편집 거리 합 ÷ 정답 글자 수 합. 구역 A·B 따로도 낸다
  메뉴 키 정확도   메뉴·확인 단락의 (키, 이름) 쌍이 맞게 인식된 비율
  숫자열 정확도    되읽기 단락의 숫자열이 정답과 완전히 같은 비율, 자리 단위 오류율
  누락             정답이 있는데 빈 결과가 나온 단락 수
  환각             정답이 빈 단락(무음·잡음)에서 글자가 나온 단락 수
  속도             단락당 인식 시간 중앙값·95백분위·최대, 음성 1초당 처리 시간
결과 JSON 에는 단락별 인식 문장이 들어가며, interp_eval --hyp 로 '인식 문장 → 해석' 연결 평가에 쓴다.
외부 엔진(gemini·rtzr·clova·openai)은 단락 음성을 외부 API 로 전송한다. 벤치 녹음은 가짜 번호만 쓰므로 개인정보가 없다.
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


async def eval_model(name: str, recs, device: str, make=None) -> dict:
    """한 모델로 모든 단락을 인식하고 지표를 낸다. make 는 시험용 (이름 → 인식기)."""
    interp = None
    if make is not None:
        rec = make(name)
    elif name == "gemini+interp":
        # STT=gemini + GEMINI_STT_COMBINED=1 과 같은 경로: 받아쓰기와 분류·점수를 한 번에. 시간은 '인식+해석' 합계다
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
        await rec.recognize(pcm16_to_ulaw(clips[0][1].pcm())[:8000 * 3])  # 장치 준비 시간은 재지 않는다

    rows = []
    for r, c in clips:
        ulaw = pcm16_to_ulaw(c.pcm())
        if interp is not None:
            interp.goal = goals.get(r.scenario, "")
        t = time.monotonic()
        try:
            hyp = await rec.recognize(ulaw)
            err = None
        except Exception as e:  # 엔진 오류도 결과로 남긴다
            hyp, err = "", f"{type(e).__name__}: {e}"
        rows.append({"recording": r.name, "file": c.path.name, "zone": c.zone, "kind": c.kind, "cut": c.cut,
                     "ref": c.text, "hyp": hyp, "sec": round(len(ulaw) / 8000, 2), "time": round(time.monotonic() - t, 3),
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
    ap.add_argument("--clips", help="단락 폴더 하나 (bench_real 등 clips.json 형식)")
    ap.add_argument("--scenarios", help="쉼표로 구분한 시나리오 ID 또는 녹음 이름 (예: K1,N2-1)")
    ap.add_argument("--json", help="결과 저장 파일 (단락별 인식 문장 포함)")
    ap.add_argument("--verbose", action="store_true", help="단락별 결과 출력")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    recs = [load_clip_folder(Path(a.clips))] if a.clips else load_recordings(Path(a.data), set(a.scenarios.split(",")) if a.scenarios else None)
    n = sum(len(r.clips) for r in recs)
    if not n:
        raise SystemExit("평가할 단락이 없습니다. bench/data 에 make_clips.py 결과가 있는지 확인하십시오.")
    print(f"녹음 {len(recs)}개, 단락 {n}개, 장치 {a.device}")
    results = []
    for name in a.models:
        r = await eval_model(name, recs, a.device)
        results.append(r)
        if a.verbose:
            for row in r["rows"]:
                mark = f"CER {row['cer']:.0%}" if "cer" in row else ("환각" if not M.norm(row["ref"]) and M.norm(row["hyp"]) else "")
                print(f"[{name}] {row['recording']}/{row['file']} {row['time']:.2f}s {mark} {row['hyp']}")
    print("\n## 음성 인식 (모델별)\n" + table(results))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n저장: {a.json}")


if __name__ == "__main__":
    asyncio.run(main())
