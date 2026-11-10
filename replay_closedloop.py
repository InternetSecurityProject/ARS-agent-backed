"""
녹음 기반 가상 ARS 폐회로 시험: 실제 서버(판단·받아쓰기 전부)가 가상 ARS 와 통화하게 하고, 녹음된 원래 통화와 비교 (실통화 없음)

  python replay_closedloop.py scenarios <출력 json>                         # 녹음에서 시나리오 목록 생성
  python replay_closedloop.py run <시나리오 json> <결과 폴더> --group day|night [--base <시험 서버 주소>] [--limit N] [--conc N]
  python replay_closedloop.py report <결과 폴더> <REPLAY_LOG> [--out <요약 json>] [--scenarios <시나리오 json>]

시험 서버: CALL_BACKEND=replay 로 띄운 서버. 서버는 실통화와 같은 경로로 가상 ARS 의 소리를
듣고(STT) 판단하고 키를 보냄. 가상 ARS 는 통화마다 path·녹음 밖 종료·미검증 표시를 REPLAY_LOG 에 기록.

시나리오: 녹음된 실통화(backend clawops)의 (전화번호, 목적) 쌍마다 하나. 함께 담는 것
  - 원래 통화들의 시각(낮/밤 구분: 원래 통화 시각으로 고른 첫 상태가 기관의 낮 시작 규칙 상태인지)
  - 원래 사용자가 재질문에서 고른 선택지 이름(시험 도구가 사용자 역할을 할 때 같은 뜻의 선택지를 고르는 데 사용)
  - 원래 민감정보 입력의 자릿수(값은 녹음에 없고 사용하지도 않음. 시험은 같은 자릿수의 '0' 을 송출)
  - 원래 통화의 결과(끝난 이유, 결과 요약 종류)
사용자 역할(시험 도구 전용, 제품 코드에는 넣지 않음)
  - 재질문: 원래 사용자가 고른 이름과 2글자 Jaccard 가 0.5 이상인 선택지, 없으면 맞다는 선택지(되읽기 확인), 그것도 없으면 목적 문장과
    가장 겹치는 선택지(겹침 0 이면 끊음)
  - 민감정보 입력 요청: 원래 자릿수(없으면 안내 문구의 'N자리', 그것도 없으면 6)만큼 '0' 을 송출
  - 상담원 대기·연결 상태가 되면 끊음. 결과 요약을 받으면 1초 뒤 끊음. 한 통화 최대 240초
  - 같은 장비의 실통화용 서버(LIVE_BASE)에 진행 중인 통화가 있으면 새 통화를 걸지 않고 대기 (모델 자원을 나눠 쓰므로)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAX_CALL_S = 240
LIVE_BASE = "http://127.0.0.1:8000"


def mask(s: str) -> str:
    return re.sub(r"\d{4,}", lambda m: "*" * len(m.group()), s or "")


def bigrams(s: str) -> set[str]:
    t = re.sub(r"[^가-힣0-9A-Za-z]", "", s or "")
    return {t[i:i + 2] for i in range(len(t) - 1)}


def jaccard(a: str, b: str) -> float:
    x, y = bigrams(a), bigrams(b)
    return len(x & y) / len(x | y) if x and y else 0.0


# 시나리오 ---------------------------------------------------------------------
def build_scenarios(out: str) -> None:
    import replay_check as rc
    from replay_ars import ReplayBundle, pick_start

    bundle = ReplayBundle(r"C:\Users\User\Documents\ars_run\replay\bundle")
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for d, ph in rc.calls(None, bundle):
        j = json.loads((d / "log.json").read_text(encoding="utf-8"))
        if j.get("backend") not in (None, "clawops"):
            continue
        goal = (j.get("goal") or "").strip()
        if not goal:
            continue
        opts_by_req, prefs, digits, keys = {}, [], [], []
        end_reason, summary_kind = None, None
        for e in j.get("events", []):
            t, data = e.get("type"), e.get("data") or {}
            if t == "event:request" and data.get("kind") == "clarification":
                opts_by_req[data.get("request_id")] = {o.get("key"): o.get("label") for o in data.get("options") or []}
            elif t == "user" and e.get("action") == "select":
                lab = (opts_by_req.get(e.get("request_id")) or {}).get(e.get("key"))
                if lab:
                    prefs.append(lab)
            elif t == "user" and e.get("action") == "secure_input":
                digits.append(int(e.get("digit_count") or 0))
            elif t == "user" and e.get("action") == "secure_key" and e.get("done") is not None:
                if not digits or digits[-1] < 0:
                    digits.append(0)
                digits[-1] += int(e.get("digit_count") or 0)
                if e.get("done"):
                    digits[-1] = -digits[-1]  # 한 입력의 끝을 음수로 표시 (아래에서 양수로 되돌림)
            elif t == "keys_sent" and e.get("kind") != "input" and e.get("keys"):
                keys.append(str(e["keys"]))
            elif t == "event:ended":
                end_reason = data.get("end_reason") or data.get("reason")
            elif t == "event:summary":
                summary_kind = data.get("kind")
        clock = rc.call_clock(d.name)
        org = bundle.org_for(ph)
        o = bundle.orgs[org]
        day_states = {r["state"] for r in o.get("start_rules") or []}
        start = pick_start(o, clock) if clock else o["start"]
        groups[(ph, goal)].append({"dir": d.name, "day": start in day_states if day_states else True,
                                   "start": start, "prefs": prefs, "digits": [abs(x) for x in digits if x], "keys": keys,
                                   "end_reason": end_reason, "summary": summary_kind})
    scen = []
    for (ph, goal), calls in sorted(groups.items()):
        day = sum(c["day"] for c in calls) >= len(calls) / 2
        prefs = [p for c in calls for p in c["prefs"]]
        digits = next((c["digits"] for c in calls if c["digits"]), [])
        scen.append({"id": f"s{len(scen) + 1:03d}", "phone": ph, "goal": goal, "group": "day" if day else "night",
                     "prefs": list(dict.fromkeys(prefs)), "digits": digits, "originals": calls})
    Path(out).write_text(json.dumps(scen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"시나리오 {len(scen)}개 (낮 {sum(s['group'] == 'day' for s in scen)}, 밤 {sum(s['group'] == 'night' for s in scen)})")


# 실행 -------------------------------------------------------------------------
def app_key() -> str:
    os.environ.setdefault("ARS_ENV_FILE", r"C:\Users\User\Documents\ars_local_jev\.env")
    from config import Settings  # 키 값은 출력하지 않음
    return Settings().app_key


def choose(options: list[dict], prefs: list[str], goal: str) -> str | None:
    best, key = 0.0, None
    for o in options:
        for p in prefs:
            s = jaccard(o.get("label") or "", p)
            if s >= 0.5 and s > best:
                best, key = s, o.get("key")
    if key is not None:
        return key
    # 되읽기 확인처럼 맞는지 묻는 재질문: 사용자는 맞다고 답함 (시험은 원래 통화와 같은 자릿수를 보냈으므로)
    for o in options:
        if re.search(r"맞|확인|예|네", o.get("label") or "") and not re.search(r"아니|틀|다시|재입력|취소", o.get("label") or ""):
            return o.get("key")
    best = 0.0
    for o in options:
        s = jaccard((o.get("label") or "") + " " + (o.get("category") or ""), goal)
        if s > best:
            best, key = s, o.get("key")
    return key if best > 0 else None


def secure_len(sc: dict, idx: int, req: dict) -> int:
    if idx < len(sc["digits"]) and sc["digits"][idx]:
        return sc["digits"][idx]
    m = re.search(r"(\d{1,2})\s*자리", (req.get("label") or "") + " " + (req.get("message") or ""))
    return int(m.group(1)) if m else 6


async def live_busy(client) -> bool:
    try:
        r = await client.get(f"{LIVE_BASE}/v1/health", timeout=5)
        return int(r.json().get("active_sessions") or 0) > 0
    except Exception:  # noqa: BLE001 — 실통화용 서버가 꺼져 있으면 대기하지 않음
        return False


async def run_one(sc: dict, base: str, akey: str, out_dir: Path, sem: asyncio.Semaphore) -> dict:
    import httpx
    import websockets

    ws_base = base.replace("http", "ws", 1)
    async with sem:
        async with httpx.AsyncClient(timeout=15) as c:
            while await live_busy(c):
                await asyncio.sleep(10)
            t0 = time.monotonic()
            rec = {"id": sc["id"], "phone": sc["phone"], "goal": sc["goal"], "group": sc["group"],
                   "created_at": datetime.now().astimezone().isoformat(timespec="seconds"), "events": [],
                   "selects": [], "secure": [], "agent": False, "summary": None, "end_reason": None, "cancel_why": None}

            def note(what, **kw):
                rec["events"].append({"t": round(time.monotonic() - t0, 2), "kind": what, **kw})

            r = await c.post(f"{base}/v1/sessions", headers={"x-app-key": akey},
                             json={"phone_number": sc["phone"], "goal": sc["goal"], "record": True})
            if r.status_code != 201:
                rec["outcome"] = f"create_failed {r.status_code}"
                return rec
            d = r.json()
            sid, token = d["session_id"], d["session_token"]
            rec["session_id"] = sid
            auth = {"authorization": f"Bearer {token}"}

            async def act(body):
                try:
                    rr = await c.post(f"{base}/v1/sessions/{sid}/actions", headers=auth, json=body)
                    note("action", type=body.get("type"), status=rr.status_code)
                except Exception as e:  # noqa: BLE001
                    note("action_error", error=str(e)[:80])

            cancelled = False

            async def cancel(why):
                nonlocal cancelled
                if not cancelled:
                    cancelled = True
                    rec["cancel_why"] = why
                    note("cancel", why=why)
                    await act({"type": "cancel"})

            try:
                async with websockets.connect(f"{ws_base}/v1/sessions/{sid}/events?token={token}", open_timeout=10) as ws:
                    while True:
                        left = MAX_CALL_S - (time.monotonic() - t0)
                        if left <= 0:
                            await cancel("time_limit")
                            left = 20
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=left)
                        except asyncio.TimeoutError:
                            if cancelled:
                                break
                            continue
                        except websockets.ConnectionClosed:
                            break
                        msg = json.loads(raw)
                        ev, data = msg.get("event"), msg.get("data") or {}
                        if ev == "caption":
                            note("caption", chars=len(data.get("text") or ""))
                        elif ev == "path_updated":
                            sel = data.get("selected") or {}
                            note("press", key=sel.get("key"), actor=data.get("actor"))
                        elif ev == "status_changed":
                            note("status", status=data.get("status"))
                            if data.get("status") in ("waiting_agent", "connected_to_agent"):
                                rec["agent"] = True
                                await cancel("agent")
                        elif ev == "input_rejected":
                            note("input_rejected", retry=data.get("retry_count"))
                        elif ev == "request":
                            kind = data.get("kind")
                            if cancelled:
                                continue
                            if kind == "secure_input":
                                n = secure_len(sc, len(rec["secure"]), data)
                                rec["secure"].append({"label": data.get("label"), "digits": n})
                                note("secure_input", digits=n)
                                await act({"type": "secure_input", "request_id": data.get("request_id"), "digits": "0" * n})
                            elif kind == "clarification":
                                opts = data.get("options") or []
                                k = choose(opts, sc["prefs"], sc["goal"])
                                note("ask", reason=data.get("reason"), n_options=len(opts), chose=k)
                                if k is None:
                                    await cancel("no_matching_option")
                                else:
                                    rec["selects"].append(k)
                                    await act({"type": "select", "request_id": data.get("request_id"), "key": k})
                        elif ev == "summary":
                            rec["summary"] = data.get("kind")
                            note("summary", summary_kind=data.get("kind"))
                            await asyncio.sleep(1.0)
                            await cancel("summary")
                        elif ev == "ended":
                            rec["end_reason"] = data.get("end_reason") or data.get("reason")
                            note("ended", reason=rec["end_reason"])
                            break
            except Exception as e:  # noqa: BLE001
                note("ws_error", error=str(e)[:120])
                await cancel("ws_error")
        rec["duration_s"] = round(time.monotonic() - t0, 1)
        rec["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        (out_dir / f"{sc['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        presses = [e.get("key") for e in rec["events"] if e["kind"] == "press"]
        print(f"[{sc['id']}] {sc['phone']} {mask(sc['goal'])[:24]}: 끝={rec['end_reason']} 요약={rec['summary']} "
              f"끊음={rec['cancel_why']} 누름={presses} ({rec['duration_s']}s)", flush=True)
        return rec


async def run_all(scen_path: str, out_dir: str, group: str, base: str, limit: int | None, conc: int) -> None:
    scen = [s for s in json.loads(Path(scen_path).read_text(encoding="utf-8")) if s["group"] == group]
    if limit:
        scen = scen[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out.glob("s*.json")}
    scen = [s for s in scen if s["id"] not in done]
    akey = app_key()
    sem = asyncio.Semaphore(conc)
    tasks = []
    for s in scen:
        tasks.append(asyncio.create_task(run_one(s, base, akey, out, sem)))
        await asyncio.sleep(3.0)  # 같은 번호 통화가 겹쳐도 발신 순서로 기록을 맞출 수 있도록 간격을 둠
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"끝: {len(scen)}통화", flush=True)


# 요약 -------------------------------------------------------------------------
def report(out_dir: str, log_path: str, out: str | None, scen_path: str | None) -> dict:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(out_dir).glob("s*.json"))]
    logs = [json.loads(x) for x in Path(log_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    scen = {s["id"]: s for s in json.loads(Path(scen_path).read_text(encoding="utf-8"))} if scen_path else {}
    used = set()
    rows = []
    for r in sorted(recs, key=lambda x: x["created_at"]):
        t = datetime.fromisoformat(r["created_at"]).timestamp()
        cands = [(i, l) for i, l in enumerate(logs) if i not in used and l.get("phone") == r["phone"]
                 and 0 <= datetime.fromisoformat(l["dialed_at"]).timestamp() - t <= 90]
        lg = None
        if cands:
            i, lg = min(cands, key=lambda x: datetime.fromisoformat(x[1]["dialed_at"]).timestamp())
            used.add(i)
        flags = [p for p in (lg or {}).get("path", []) if "flag" in p]
        states = [p["state"] for p in (lg or {}).get("path", []) if "state" in p]
        presses = [str(e.get("key")) for e in r["events"] if e["kind"] == "press" and e.get("key")]
        origs = scen.get(r["id"], {}).get("originals", [])
        # 원래 통화들과 누른 키 순서 비교: 같은 순서가 있으면 same, 한쪽이 다른 쪽의 앞부분이면 prefix, 아니면 처음 갈라진 자리
        route, diverge = "differ", None
        for o in origs:
            ok = o.get("keys") or []
            if ok == presses:
                route = "same"
                break
            k = min(len(ok), len(presses))
            if ok[:k] == presses[:k]:
                route = "prefix" if route != "same" else route
            else:
                i = next(i for i in range(k) if ok[i] != presses[i])
                diverge = i if diverge is None else max(diverge, i)
        rows.append({"id": r["id"], "phone": r["phone"], "group": r["group"], "summary": r["summary"],
                     "end_reason": r["end_reason"], "cancel_why": r["cancel_why"], "agent": r["agent"],
                     "secure": len(r["secure"]), "asks": sum(1 for e in r["events"] if e["kind"] == "ask"),
                     "presses": presses,
                     "duration_s": r["duration_s"], "matched_log": lg is not None,
                     "offmap": (lg or {}).get("offmap"), "first_flag": flags[0]["flag"] if flags else None,
                     "first_flag_t": flags[0]["t"] if flags else None, "flags": (lg or {}).get("flags"),
                     "states": states, "evidence_ok": lg is not None and not (lg or {}).get("offmap") and not flags,
                     "route_vs_original": route, "diverge_at": None if route != "differ" else diverge,
                     "orig_keys": [o.get("keys") for o in origs],
                     "orig_end": [o.get("end_reason") for o in scen.get(r["id"], {}).get("originals", [])],
                     "orig_summary": [o.get("summary") for o in scen.get(r["id"], {}).get("originals", [])]})
    summ = {"calls": len(rows), "matched_log": sum(r["matched_log"] for r in rows),
            "evidence_ok": sum(r["evidence_ok"] for r in rows),
            "server_outcome": Counter(r["summary"] or r["cancel_why"] or r["end_reason"] for r in rows),
            "offmap": Counter((r["offmap"] or "")[:20] for r in rows if r["offmap"]),
            "first_flag": Counter(r["first_flag"] for r in rows if r["first_flag"]),
            "route_vs_original": Counter(r["route_vs_original"] for r in rows)}
    res = {"summary": summ, "rows": rows}
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scenarios")
    a.add_argument("out")
    b = sub.add_parser("run")
    b.add_argument("scenarios")
    b.add_argument("out_dir")
    b.add_argument("--group", required=True, choices=["day", "night"])
    b.add_argument("--base", default="http://127.0.0.1:8002")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--conc", type=int, default=3)
    c = sub.add_parser("report")
    c.add_argument("out_dir")
    c.add_argument("replay_log")
    c.add_argument("--out", default=None)
    c.add_argument("--scenarios", default=None)
    x = ap.parse_args()
    if x.cmd == "scenarios":
        build_scenarios(x.out)
    elif x.cmd == "run":
        asyncio.run(run_all(x.scenarios, x.out_dir, x.group, x.base, x.limit, x.conc))
    else:
        report(x.out_dir, x.replay_log, x.out, x.scenarios)


if __name__ == "__main__":
    main()
