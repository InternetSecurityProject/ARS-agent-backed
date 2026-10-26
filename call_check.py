"""
실통화 점검 (통화 1건으로 통화 계층을 최대한 확인)

실행 중인 서버(python app.py, 실통화)에 앱과 같은 방식으로 세션 1건을 만들고, 이벤트·오디오 채널을 받아
시간순으로 기록한 뒤 통화 계층 확인 항목을 통과/실패로 출력한다. 받은 음성은 WAV 로 저장한다.

  python call_check.py --phone 15889999 --goal "상담원 연결해줘" --sec 60 --wav _local/real/call1.wav

안전 규칙 (자동 종료)
  - 민감정보 입력 요청(secure_input)이 오면 아무것도 입력하지 않고 바로 종료한다.
  - 상담원 연결 대기(waiting_agent)에 들어가면 바로 종료한다 (실제 상담원을 붙잡지 않는다).
  - 재질문(clarification)이 오면 답하지 않고 종료한다. 결과 요약을 받으면 2초 뒤 종료한다.
  - --sec 이 지나면 종료한다. 실통화 서버이면 발신 전에 번호·목적·제한 시간을 보여 주고 확인을 받는다(--yes 로 생략).
기록: 4자리 이상 숫자는 가려서 출력·저장한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx
import websockets

from api_check import WS_OPTS, AudioProbe, Checks, audio_checks

T0 = time.monotonic()
LOG: list[str] = []


def mask(text: str) -> str:
    return re.sub(r"\d[\d,\s\-]{2,}\d",
                  lambda m: "*" * len(re.sub(r"\D", "", m.group())) if len(re.sub(r"\D", "", m.group())) >= 4 else m.group(),
                  text)


def say(kind: str, text: str) -> None:
    line = f"[{time.monotonic() - T0:6.1f}s] {kind:<8} {mask(text)}"
    LOG.append(line)
    print(line, flush=True)


async def main() -> int:
    ap = argparse.ArgumentParser(description="실통화 점검 (통화 1건)")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--app-key", default="dev-app-key")
    ap.add_argument("--phone", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--sec", type=float, default=60, help="최대 통화 시간(초)")
    ap.add_argument("--wav", default="call_check.wav", help="받은 음성 저장 파일")
    ap.add_argument("--yes", action="store_true", help="실통화 발신 확인을 생략")
    a = ap.parse_args()

    async with httpx.AsyncClient(base_url=a.base, timeout=30) as c:
        health = (await c.get("/v1/health")).json()
        real = health.get("backend") != "sim"
        say("서버", f"backend={health.get('backend')} stt={health.get('stt')} 해석기={health.get('interpreter')}")
        if health.get("active_session"):
            say("중단", "다른 통화가 진행 중입니다.")
            return 1
        if real and not a.yes:
            ans = input(f"\n실제 발신합니다: {a.phone} / 목적 \"{a.goal}\" / 최대 {a.sec:.0f}초. 계속할까요? [y/N] ")
            if ans.strip().lower() != "y":
                say("중단", "발신하지 않았습니다.")
                return 1

        r = await c.post("/v1/sessions", headers={"X-App-Key": a.app_key},
                         json={"phone_number": a.phone, "goal": a.goal})
        d = r.json()
        if r.status_code != 201:
            say("오류", f"세션 생성 {r.status_code} {d}")
            return 1
        sid, tok = d["session_id"], d["session_token"]
        auth = {"Authorization": f"Bearer {tok}"}
        say("발신", f"{d['phone_number']} 목적 요약: {d.get('goal_summary')}")
        probe = AudioProbe(a.wav)
        probe.start(f"{d['audio_url']}?token={tok}")

        # progress: 키 전송 뒤 ARS 가 다음 단계로 넘어갔다는 신호 (자막, 민감정보 요청·재질문, 결과, 상담원 대기)
        obs = {"answered": None, "captions": [], "path": [], "rejected": [], "statuses": [], "progress": [], "stop": None}
        canceled = False

        async def cancel(why: str) -> None:
            nonlocal canceled
            if not canceled:
                canceled = True
                obs["stop"] = why
                say("종료요청", why)
                await c.post(f"/v1/sessions/{sid}/actions", headers=auth, json={"type": "cancel"})

        async with websockets.connect(f"{d['events_url']}?token={tok}", **WS_OPTS) as ws:
            end = time.monotonic() + a.sec + 30  # 상대가 받기까지 여유
            while time.monotonic() < end:
                if obs["answered"] and time.monotonic() - obs["answered"] > a.sec:
                    await cancel(f"최대 {a.sec:.0f}초 도달")
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 1.0))
                except asyncio.TimeoutError:
                    continue
                e, data, now = m["event"], m["data"], time.monotonic()
                if e == "status_changed":
                    obs["statuses"].append(data["status"])
                    say("상태", f"{data['status']} {data['message']}")
                    if data["status"] == "listening" and obs["answered"] is None:
                        obs["answered"] = now
                    if data["status"] in ("awaiting_user", "listening_result", "waiting_agent"):
                        obs["progress"].append(now)
                    if data["status"] in ("waiting_agent", "connected_to_agent"):
                        await cancel("상담원 단계 도달 (상담원을 붙잡지 않도록 종료)")
                elif e == "caption":
                    obs["captions"].append((now, data["text"]))
                    obs["progress"].append(now)
                    say("자막", data["text"])
                elif e == "path_updated":
                    obs["path"].append((now, data))
                    say("경로", f"{data['text']} (신뢰도 {data['selected']['confidence']}, {data['actor']})")
                elif e == "input_rejected":
                    obs["rejected"].append((now, data))
                    say("입력오류", f"{data['keys']} → {data['ars_prompt']}")
                elif e == "request":
                    say("요청", f"{data['kind']} {data.get('reason') or ''} {data['message']}")
                    await cancel("민감정보 요청 → 입력하지 않고 종료" if data["kind"] == "secure_input"
                                 else "재질문 → 답하지 않고 종료")
                elif e == "summary":
                    say("요약", json.dumps(data, ensure_ascii=False))
                    if data["kind"] == "result":
                        await asyncio.sleep(2)
                        await cancel("결과 요약 수신")
                elif e == "ended":
                    say("종료", f"{data['end_reason']} {data['message']}")
                    break
            else:
                await cancel("시간 초과")
        await probe.stop()
        res = await c.get(f"/v1/sessions/{sid}/result", headers=auth)
        result = res.json() if res.status_code == 200 else {}

    ck = Checks()
    ck.add("발신·연결 (상대가 받음)", obs["answered"] is not None)
    audio_checks(ck, probe, sim=not real)
    ck.add("ARS 안내 자막 수신", bool(obs["captions"]), f"{len(obs['captions'])}개")
    for t, p in obs["path"]:
        nxt = [x for x in obs["progress"] if x > t]
        rej = [x for x, _ in obs["rejected"] if x > t and (not nxt or x <= nxt[0] + 0.5)]
        ck.add(f"ARS 가 '{p['selected']['key']}' 키를 받음 (다음 안내로 진행)", bool(nxt) and not rej,
               "입력 오류 안내" if rej else (f"{nxt[0] - t:.1f}초 뒤 다음 안내" if nxt else "다음 안내 없음"))
    ck.add("종료 사유", result.get("end_reason") is not None,
           f"{result.get('end_reason')} / 통화 {result.get('duration_sec')}초 / 종료 계기: {obs['stop'] or 'ARS·상대'}")
    code = ck.report("실통화 점검 결과")
    log = Path(a.wav).with_suffix(".txt")
    log.write_text("\n".join(LOG) + "\n\n" + "\n".join(f"[{m}] {n} - {dt}" for m, n, dt in ck.items) + "\n",
                   encoding="utf-8")
    print(f"기록 저장: {log}")
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main()))
