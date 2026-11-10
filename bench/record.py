"""경로 고정 녹음: 시나리오의 키 경로대로 실제 ARS 에 발신해 통화 음성 녹음.

준비 (녹음 전용 서버 별도 실행. AI 가 키를 누르지 않도록 모의 해석기 사용)
  PowerShell:
    $env:INTERPRETER = "interpreter:MockInterpreter"
    $env:POST_KEY_CAPTION_S = "0"
    python app.py --port 8011
녹음
  python -m bench.record K1 --base http://127.0.0.1:8011 --app-key <APP_KEY>
  python -m bench.record K1 --take 2 ...          # 같은 시나리오 두 번째 녹음
결과
  bench/raw/K1-1.wav   통화 수신 음성 (8kHz 16bit mono)
  bench/raw/K1-1.json  키 전송 시점(녹음 바이트 위치)·입력한 가짜 번호·서버 자막

키를 누르는 시점
  단계마다 다음 중 하나를 기다린 뒤 delay(기본 0.6초) 후에 키 전송.
  - match: 서버 자막이 이 정규식과 맞을 때 (기본: 눌러|누르|입력)
  - para:  녹음 음성에서 단락(1.2초 무음으로 끝나는 소리)이 이 수만큼 끝났을 때. 서버가 자막을 보내지 않는 되읽기 안내 뒤에 사용
  기다리는 안내가 max_wait(기본 45초) 안에 오지 않으면 경로를 멈추고 그때까지의 녹음 저장.

안전 규칙 (bench/scenarios.json 의 rules): 존재할 수 없는 가짜 번호만 입력, 발급·발송으로 이어지는 확인에는 '맞음' 누름 금지.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from api_check import AudioProbe, Client, finish, open_session
from audio import Segmenter
from bench.data import BENCH, load_scenarios

DEFAULT_MATCH = r"눌러|누르|입력"


class ParagraphWatch:
    """수신 음성에 서버와 같은 단락 판정(RMS 500, 무음 1.2초)을 적용한 단락 종료 횟수 집계."""

    def __init__(self, probe: AudioProbe) -> None:
        from clawops.agent._audio import pcm16_to_ulaw  # noqa: F401  (의존성 확인)

        self.probe, self.seg, self.pos, self.ends = probe, Segmenter(), 0, 0

    def poll(self) -> int:
        frames = self.probe.frames
        while self.pos < len(frames):
            if self.seg.feed(frames[self.pos]) is not None:
                self.ends += 1
            self.pos += 1
        return self.ends


async def wait_trigger(c: Client, watch: ParagraphWatch, step: dict, seen_caps: int, seen_paras: int,
                       max_wait: float) -> tuple[bool, int]:
    """단계의 조건이 맞을 때까지 대기. (성공 여부, 지금까지 본 자막 수)."""
    end = time.monotonic() + step.get("max_wait", max_wait)
    rx = re.compile(step.get("match", DEFAULT_MATCH))
    while time.monotonic() < end:
        if c.seen(lambda e: e["event"] == "ended"):
            return False, seen_caps
        if "para" in step:
            if watch.poll() - seen_paras >= step["para"]:
                return True, seen_caps
        else:
            caps = [e for e in c.events if e["event"] == "caption"]
            for e in caps[seen_caps:]:
                if rx.search(e["data"].get("text", "")):
                    return True, len(caps)
        await asyncio.sleep(0.1)
    return False, seen_caps


async def send(c: Client, keys: str) -> None:
    """키 전송. 서버의 key 조작은 4자까지이므로 나눠 보내고, 서버가 톤을 실시간으로 보내는 동안 대기."""
    for i in range(0, len(keys), 4):
        part = keys[i:i + 4]
        await c.call("POST", f"/v1/sessions/{c.sid}/actions", {"type": "key", "key": part})
        await asyncio.sleep(len(part) * 0.25 + 0.15)


async def record(scenario: dict, base: str, app_key: str, out: Path, take: int, max_wait: float, force: bool) -> Path:
    c = Client(base, app_key)
    _, h = await c.request("GET", "/v1/health")
    if "Mock" not in str(h.get("interpreter")) and not force:
        raise SystemExit("녹음 서버가 모의 해석기로 실행되지 않았습니다 (INTERPRETER=interpreter:MockInterpreter). "
                         "AI 가 키를 누르면 경로가 섞입니다. 계속하려면 --force")
    name = f"{scenario['id']}-{take}"
    out.mkdir(parents=True, exist_ok=True)
    probe = AudioProbe(str(out / f"{name}.wav"))
    listener = await open_session(c, scenario["phone"], scenario["goal"], probe)
    watch = ParagraphWatch(probe)
    actions, seen_caps = [], 0
    t0 = time.monotonic()
    ok = True
    for i, step in enumerate(scenario["route"]):
        if "observe" in step:
            await asyncio.sleep(step["observe"])
            continue
        hit, seen_caps = await wait_trigger(c, watch, step, seen_caps, watch.poll(), max_wait)
        if not hit:
            print(f"\n[경로 중단] {i + 1}단계 {step} 를 기다리던 안내가 오지 않았습니다. 지금까지 녹음을 저장합니다.")
            ok = False
            break
        await asyncio.sleep(step.get("delay", 0.6))
        keys = step.get("key") or (step["input"] + step.get("terminator", ""))
        at = sum(len(f) for f in probe.frames)  # 녹음에서 키를 누른 위치 (바이트, μ-law 1바이트 = 1샘플)
        actions.append({"step": i, "type": "key" if "key" in step else "input", "keys": keys,
                        "digits": step.get("input"), "at_sample": at, "t": round(time.monotonic() - t0, 2)})
        await send(c, keys)
    await finish(c, listener)
    await probe.stop()
    probe.save_wav()
    captions = [{"text": e["data"].get("text", ""), "at": e.get("at")} for e in c.events if e["event"] == "caption"]
    meta = {"scenario": scenario["id"], "take": take, "wav": f"{name}.wav", "complete": ok,
            "seconds": round(probe.seconds, 1), "actions": actions, "captions": captions}
    path = out / f"{name}.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out / (name + '.wav')} ({probe.seconds:.1f}초), {path} (키 {len(actions)}회, 경로 {'완료' if ok else '중단'})")
    return path


async def main() -> None:
    ap = argparse.ArgumentParser(description="경로 고정 녹음")
    ap.add_argument("scenario", help="시나리오 ID (bench/scenarios.json)")
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    ap.add_argument("--app-key", default="dev-app-key")
    ap.add_argument("--take", type=int, default=1)
    ap.add_argument("--out", default=str(BENCH / "raw"))
    ap.add_argument("--max-wait", type=float, default=45)
    ap.add_argument("--force", action="store_true", help="모의 해석기가 아닌 서버에서도 녹음")
    a = ap.parse_args()
    sc = load_scenarios().get(a.scenario)
    if not sc:
        raise SystemExit(f"시나리오 {a.scenario} 가 없습니다")
    await record(sc, a.base, a.app_key, Path(a.out), a.take, a.max_wait, a.force)


if __name__ == "__main__":
    asyncio.run(main())
