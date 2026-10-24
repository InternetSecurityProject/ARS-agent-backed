"""
실시간 듣기·디버깅 콘솔 (스마트폰 앱 대역)

서버 API 를 앱과 같은 방식으로 호출하면서, 오디오 채널(4.5)의 통화 음성을 PC 스피커로 실시간 재생하고
상태·ARS 자막·번호 선택·재질문을 시간순으로 출력한다. 터미널 명령으로 앱의 조작(4.3)을 그대로 사용한다.
음성 입력(목적, 재질문 답변)은 앱의 단말 STT 결과 대신 텍스트로 입력한다.
AI 나 사용자가 번호를 누르면(path_updated) 그 번호의 DTMF 톤을 재생음에 섞어 들려준다.

  1) 서버 실행 (별도 터미널)
       python app.py --sim        가상 ARS (음성 없이 안내 구간에 400Hz 시험음)
       python app.py              ClawOps 실통화 (실제 ARS 음성)
  2) 콘솔 실행
       python live.py                                             받는 번호와 원하는 업무(목적)를 입력하면 전화를 건다
       python live.py --goal "통장 잔액 알고 싶어"                 목적을 바로 지정하여 전화를 건다
       python live.py --session ses_xxx --token st_xxx           앱이 만든 세션에 붙어서 함께 듣기·조작

  명령 (입력 후 Enter)
       a <문장>   재질문에 답변 (answer, 음성 답변을 텍스트로 대신)
       s <번호>   재질문 후보 선택 (select)
       k <키>     직접 선택 (key). 예: k 3, k *, k 0
       p          민감정보 직접 입력 (secure_input). 숫자는 화면에 표시하지 않는다
       r <문장>   다시 말하기 (revise_goal, 첫 번호를 누르기 전까지)
       m          소리 켜기·끄기 (오디오 채널 연결·해제)
       q          통화 종료 (cancel)
       x          콘솔만 종료 (통화 유지)
       ?          도움말

  소리 출력: --player auto(기본, sounddevice → ffplay 순서) / sounddevice / ffplay / none
             sounddevice 는 pip 로 설치되며(Windows·macOS), ffplay 는 FFmpeg 를 설치하면 함께 설치된다.
  --wav 파일명: 재생한 소리(DTMF 톤 포함)를 파일로도 저장한다.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
import wave

import httpx
import numpy as np
import websockets

import api_check as ac

RATE = 8000
OUT_RATE = 48000   # sounddevice 출력 표본율. 8kHz 를 지원하지 않는 장치가 있어 48kHz 로 올려 재생한다
STALL_S = 0.5      # 이 시간 이상 프레임이 오지 않으면 끊김으로 표시한다
MAX_QUEUE = 10     # 재생 대기 프레임이 이보다 많아지면(200ms) 오래된 프레임을 버려 지연이 쌓이지 않게 한다
FFPLAY_INPUT = ["-f", "s16le", "-sample_rate", "8000", "-i", "-"]  # 16bit PCM 8kHz mono 를 표준 입력으로 받는다

DTMF_HZ = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477),
}

HELP = ("명령: a <답변>  s <번호>  k <키>  p(민감정보)  r <목적>  m(소리 켜기·끄기)  q(통화 종료)  x(콘솔만 종료)  ?(도움말)")


def say(text: str) -> None:
    print(f"{ac.ts()} {text}", flush=True)


# ---------------------------------------------------------------------------
# 소리 처리
# ---------------------------------------------------------------------------
_ULAW_DEC = np.array(ac.ULAW_TABLE, dtype=np.int16)


def ulaw_decode(data: bytes) -> np.ndarray:
    return _ULAW_DEC[np.frombuffer(data, dtype=np.uint8)]


def dtmf_tone(keys: str, tone_ms: int = 120, gap_ms: int = 80, amp: float = 8000.0) -> np.ndarray:
    """번호를 누를 때 나는 DTMF 톤 (8kHz float). 시작·끝 5ms 를 서서히 키우고 줄여 딸깍 소리를 없앤다."""
    n, ramp = RATE * tone_ms // 1000, RATE * 5 // 1000
    env = np.ones(n, dtype=np.float32)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    t = np.arange(n) / RATE
    parts = []
    for ch in keys:
        if ch in DTMF_HZ:
            lo, hi = DTMF_HZ[ch]
            parts += [amp * env * (np.sin(2 * np.pi * lo * t) + np.sin(2 * np.pi * hi * t)) / 2,
                      np.zeros(RATE * gap_ms // 1000, dtype=np.float32)]
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)


class SoundDeviceSink:
    """sounddevice(PortAudio) 로 재생한다. 8kHz → 48kHz 선형 보간."""

    name = "sounddevice"

    def __init__(self) -> None:
        import sounddevice as sd  # 없거나 PortAudio 를 찾지 못하면 예외

        self.stream = sd.RawOutputStream(samplerate=OUT_RATE, channels=1, dtype="int16", latency="low")
        self.stream.start()
        self.stream.write(np.zeros(OUT_RATE // 10, dtype=np.int16).tobytes())  # 100ms 여유: 도착 간격 흔들림을 흡수
        self.last = 0.0
        self.k = OUT_RATE // RATE

    def write(self, pcm: np.ndarray) -> None:
        x = np.concatenate(([self.last], pcm.astype(np.float32)))
        y = np.interp(np.arange(1, len(pcm) * self.k + 1) / self.k, np.arange(len(x)), x)
        self.last = float(pcm[-1])
        self.stream.write(y.astype(np.int16).tobytes())

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


class FFplaySink:
    """FFmpeg 의 ffplay 로 재생한다. 16bit PCM 8kHz 를 표준 입력으로 넘긴다."""

    name = "ffplay"

    def __init__(self, cmd: list[str]) -> None:
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def command(exe: str) -> list[str]:
        return [exe, "-hide_banner", "-loglevel", "error", "-nodisp", "-autoexit",
                "-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0",
                *FFPLAY_INPUT]

    def write(self, pcm: np.ndarray) -> None:
        self.p.stdin.write(pcm.astype("<i2").tobytes())
        self.p.stdin.flush()

    def close(self) -> None:
        try:
            self.p.stdin.close()
            self.p.wait(3)
        except Exception:
            self.p.kill()


def open_sink(player: str):
    """재생 장치를 연다. 반환: (sink 또는 None, 설명)"""
    errors = []
    if player in ("auto", "sounddevice"):
        try:
            return SoundDeviceSink(), "sounddevice (48kHz)"
        except Exception as e:
            errors.append(f"sounddevice: {type(e).__name__} {e}")
    if player in ("auto", "ffplay"):
        exe = shutil.which("ffplay")
        if exe:
            return FFplaySink(FFplaySink.command(exe)), f"ffplay ({exe})"
        errors.append("ffplay: PATH 에서 찾지 못함")
    if player != "none":
        say("! 소리를 재생하지 못합니다. " + " / ".join(errors))
        say("  pip install sounddevice 로 설치하거나 FFmpeg(ffplay)를 설치한다. "
            "Windows: winget install Gyan.FFmpeg, macOS: brew install ffmpeg")
    return None, "재생 안 함"


class Speaker:
    """오디오 채널 프레임(μ-law 8kHz)을 실시간 재생한다. 번호가 눌리면 DTMF 톤을 섞는다."""

    def __init__(self, sink=None, wav_path: str | None = None) -> None:
        self.sink = sink
        self.q: queue.Queue = queue.Queue()
        self.tone = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.wav = None
        if wav_path:
            self.wav = wave.open(wav_path, "wb")
            self.wav.setnchannels(1)
            self.wav.setsampwidth(2)
            self.wav.setframerate(RATE)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def feed(self, ulaw: bytes) -> None:
        self.q.put(ulaw)

    def clear_tone(self) -> None:
        with self.lock:
            self.tone = np.zeros(0, dtype=np.float32)

    def press(self, keys: str) -> None:
        tone = dtmf_tone(keys)
        with self.lock:
            self.tone = np.concatenate((self.tone, tone))

    def process(self, ulaw: bytes) -> np.ndarray:
        """한 프레임을 PCM 으로 풀고, 대기 중인 DTMF 톤을 섞는다."""
        pcm = ulaw_decode(ulaw).astype(np.float32)
        with self.lock:
            n = min(len(self.tone), len(pcm))
            if n:
                pcm[:n] += self.tone[:n]
                self.tone = self.tone[n:]
        return np.clip(pcm, -32768, 32767).astype(np.int16)

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            while self.q.qsize() > MAX_QUEUE:  # 재생이 밀리면 오래된 프레임을 버린다 (실시간 유지)
                nxt = self.q.get()
                if nxt is None:
                    return
                item = nxt
            out = self.process(item)
            if self.wav:
                self.wav.writeframes(out.astype("<i2").tobytes())
            if self.sink:
                try:
                    self.sink.write(out)
                except Exception as e:
                    say(f"! 재생 장치 오류로 재생을 멈춥니다: {type(e).__name__} {e}")
                    self.sink = None

    def close(self) -> None:
        self.q.put(None)
        self.thread.join(5)
        if self.sink:
            try:
                self.sink.close()
            except Exception:
                pass
        if self.wav:
            self.wav.close()


# ---------------------------------------------------------------------------
# 이벤트 출력
# ---------------------------------------------------------------------------
def describe(msg: dict) -> list[str]:
    """이벤트 하나를 사람이 읽는 줄로 바꾼다 (번호 선택·재질문·결과를 눈에 띄게)."""
    ev, d = msg.get("event"), msg.get("data") or {}
    if ev == "snapshot":
        return [f"· 연결됨: 상태 {d.get('status')}, 경로 {len(d.get('path') or [])}건, 목적 \"{d.get('goal')}\""]
    if ev == "status_changed":
        return [f"· 상태 {d.get('status')}  {d.get('message', '')}"]
    if ev == "caption":
        return [f"ARS  {d.get('text', '')}"]
    if ev == "path_updated":
        sel = d.get("selected") or {}
        who = "AI" if d.get("actor") == "ai" else "사용자"
        conf = sel.get("confidence")
        heard = " | ".join(f"[{o['key']}] {o['label']}" for o in d.get("options") or [])
        lines = [f"▶ {who}: [{sel.get('key')}] 누름 - {sel.get('label')}" + (f" (신뢰도 {conf:.2f})" if conf is not None else "")]
        if heard:
            lines.append(f"    들은 선택지: {heard}")
        return lines
    if ev == "request":
        if d.get("kind") == "secure_input":
            term = f" (종료 키 {d['terminator']} 는 서버가 붙임)" if d.get("terminator") else ""
            return [f"★ 민감정보 직접 입력: {d.get('label')}{term}",
                    "    → p 를 입력한 뒤 숫자를 입력 (화면에 표시하지 않음)"]
        opts = " | ".join(f"[{o['key']}] {o['label']}" + (f" {o['confidence']:.2f}" if o.get("confidence") is not None else "")
                          for o in d.get("options") or [])
        how = "s <번호> (음성 답변 불가)" if d.get("reason") == "critical_action" else "a <답변> 또는 s <번호>"
        lines = [f"★ 재질문 ({d.get('reason')}): {d.get('message')}"]
        if opts:
            lines.append(f"    후보: {opts}")
        lines.append(f"    → {how}")
        return lines
    if ev == "input_rejected":
        return [f"! 입력 오류 ({d.get('retry_count')}회째, {d.get('actor')} [{d.get('keys')}]): {d.get('ars_prompt')}"]
    if ev == "summary":
        if d.get("kind") == "handoff":
            return [f"★ 상담원 연결: {d.get('summary')}", f"    상담사 전달 문장: {d.get('handoff_script')}"]
        return [f"★ 결과: {d.get('summary')}"] + [f"    {k['label']}: {k['value']}" for k in d.get("key_info") or []]
    if ev == "ended":
        return [f"■ 통화 종료: {d.get('end_reason')} - {d.get('message', '')}"
                + (f" ({d['detail']})" if d.get("detail") else "")]
    return [f"{ev} {json.dumps(d, ensure_ascii=False)}"]


# ---------------------------------------------------------------------------
# 콘솔
# ---------------------------------------------------------------------------
class Console:
    def __init__(self, base: str, app_key: str, speaker: Speaker | None, raw: bool = False) -> None:
        self.c = ac.Client(base, app_key)
        self.speaker = speaker
        self.raw = raw
        self.pending: dict | None = None
        self.status: str | None = None
        self.ended = asyncio.Event()
        self.quit = asyncio.Event()
        self.owner = False  # 이 콘솔이 만든 세션이면 Ctrl+C 시 통화를 끊는다
        self.audio_task: asyncio.Task | None = None
        self.frames = 0
        self.events_task: asyncio.Task | None = None

    # 연결 ------------------------------------------------------------------
    async def create(self, phone: str, goal: str) -> bool:
        st, out = await self.c.request("POST", "/v1/sessions", {"phone_number": phone, "goal": goal})
        if st != 201:
            return False
        self.c.sid, self.c.token = out["session_id"], out["session_token"]
        self.c.urls = {"events": out["events_url"], "audio": out["audio_url"]}  # 응답의 URL 을 그대로 사용한다 (4.1)
        self.owner = True
        say(f"세션 {self.c.sid} 생성: {phone} 로 발신, 목적 \"{goal}\"")
        return True

    def attach(self, sid: str, token: str) -> None:
        self.c.sid, self.c.token = sid, token
        ws_base = self.c.base.replace("http", "ws", 1)
        self.c.urls = {"events": f"{ws_base}/v1/sessions/{sid}/events", "audio": f"{ws_base}/v1/sessions/{sid}/audio"}
        say(f"세션 {sid} 에 연결 (앱이 만든 세션. x 로 나가면 통화는 유지된다)")

    def start(self, sound: bool = True) -> None:
        self.events_task = asyncio.create_task(self._events())
        if sound:
            self.sound_on()

    async def _events(self) -> None:
        try:
            async with websockets.connect(f"{self.c.urls['events']}?token={self.c.token}", **ac.WS_OPTS) as ws:
                async for raw in ws:
                    self.on_event(json.loads(raw))
                    if self.ended.is_set():
                        return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            say(f"! 이벤트 채널 오류: {type(e).__name__} {e}")
            self.quit.set()

    def on_event(self, msg: dict) -> None:
        ev, d = msg.get("event"), msg.get("data") or {}
        if ev in ("snapshot", "status_changed"):
            self.status = d.get("status")
            if ev == "snapshot":
                self.pending = d.get("pending_request")
            elif self.status != "awaiting_user":
                self.pending = None
        elif ev == "request":
            self.pending = d
        elif ev == "path_updated" and self.speaker and self.sound_is_on:
            self.speaker.press(str((d.get("selected") or {}).get("key") or ""))
        for line in describe(msg):
            say(line)
        if self.raw:
            say(f"  (raw) {json.dumps(msg, ensure_ascii=False)}")
        if ev == "ended":
            self.ended.set()

    # 소리 (오디오 채널) -------------------------------------------------------
    @property
    def sound_is_on(self) -> bool:
        return bool(self.audio_task and not self.audio_task.done())

    def sound_on(self) -> None:
        if self.audio_task and not self.audio_task.done():
            return
        self.audio_task = asyncio.create_task(self._audio())

    async def sound_off(self) -> None:
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except (asyncio.CancelledError, Exception):
                pass
            if self.speaker:
                self.speaker.clear_tone()
            say("· 소리 꺼짐 (오디오 채널 연결 해제)")

    async def _audio(self) -> None:
        try:
            async with websockets.connect(f"{self.c.urls['audio']}?token={self.c.token}", **ac.WS_OPTS) as ws:
                say("· 소리 켜짐 (오디오 채널 연결)")
                last = None
                async for msg in ws:
                    if isinstance(msg, str):
                        continue
                    now = time.monotonic()
                    if last is not None and now - last >= STALL_S:
                        say(f"! 오디오 {now - last:.1f}초 동안 끊김")
                    last = now
                    self.frames += 1
                    if self.speaker:
                        self.speaker.feed(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            say(f"! 오디오 채널 오류: {type(e).__name__} {e}")

    # 명령 (앱 조작) ------------------------------------------------------------
    async def act(self, body: dict, secret: bool = False) -> tuple[int, dict]:
        return await self.c.request("POST", f"/v1/sessions/{self.c.sid}/actions", body, secret=secret)

    async def command(self, line: str) -> None:
        cmd, _, arg = line.strip().partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        req = self.pending
        if not cmd:
            return
        if cmd in ("?", "h", "help"):
            say(HELP)
        elif cmd == "a":
            if not arg:
                say("  사용법: a <답변 문장>")
            elif not req or req.get("kind") != "clarification":
                say("  대기 중인 재질문이 없습니다.")
            else:
                await self.act({"type": "answer", "request_id": req["request_id"], "text": arg})
        elif cmd == "s":
            if not req or req.get("kind") != "clarification":
                say("  대기 중인 재질문이 없습니다.")
            elif not arg:
                say("  사용법: s <후보 번호>")
            else:
                await self.act({"type": "select", "request_id": req["request_id"], "key": arg})
        elif cmd == "k":
            if not arg:
                say("  사용법: k <키>  예: k 3, k *, k 0")
            else:
                await self.act({"type": "key", "key": arg.replace(" ", "")})
        elif cmd == "r":
            if not arg:
                say("  사용법: r <새 목적 문장>")
            else:
                await self.act({"type": "revise_goal", "text": arg})
        elif cmd == "m":
            if self.sound_is_on:
                await self.sound_off()
            else:
                self.sound_on()
        elif cmd == "q":
            await self.act({"type": "cancel"})
        elif cmd == "x":
            self.quit.set()
        elif cmd == "p":
            say("  p 만 입력하고 Enter 를 누르면 숫자를 화면에 표시하지 않고 받는다.")
        else:
            say(f"  알 수 없는 명령: {cmd}. {HELP}")

    async def secure(self, digits: str) -> None:
        req = self.pending
        if not req or req.get("kind") != "secure_input":
            say("  대기 중인 민감정보 입력 요청이 없습니다.")
            return
        if not digits:
            say("  입력이 비어 있습니다.")
            return
        await self.act({"type": "secure_input", "request_id": req["request_id"], "digits": digits}, secret=True)

    async def close(self, hang_up: bool) -> None:
        if hang_up and not self.ended.is_set() and self.c.sid:
            await self.act({"type": "cancel"})
            try:
                await asyncio.wait_for(self.ended.wait(), 5)
            except asyncio.TimeoutError:
                pass
        await self.sound_off()
        if self.events_task:
            self.events_task.cancel()
        if self.ended.is_set():
            st, d = await self.c.request("GET", f"/v1/sessions/{self.c.sid}/result")
        await self.c.http.aclose()


def _read_stdin(loop: asyncio.AbstractEventLoop, q: asyncio.Queue) -> None:
    """표준 입력을 읽어 이벤트 루프로 넘긴다. p 는 숫자를 숨김 입력으로 받는다."""
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            return
        if not line:
            return  # EOF: 명령 없이 듣기만 계속한다
        if line.strip().lower() == "p":
            try:
                digits = getpass.getpass("  민감정보 번호 입력 (화면에 표시하지 않음): ")
            except Exception:
                digits = ""
            loop.call_soon_threadsafe(q.put_nowait, ("secure", digits.strip()))
        else:
            loop.call_soon_threadsafe(q.put_nowait, ("line", line))


async def run(a: argparse.Namespace) -> int:
    sink, sink_name = open_sink(a.player)
    speaker = Speaker(sink, a.wav)
    con = Console(a.base, a.app_key, speaker, a.raw)
    try:
        if a.session:
            if not a.token:
                say("--session 에는 --token 이 필요합니다.")
                return 2
            con.attach(a.session, a.token)
        elif not await con.create(a.phone, a.goal):
            return 1
        say(f"· 소리 출력: {sink_name}" + (". 번호가 눌리면 DTMF 톤을 섞어 들려준다" if sink else "")
            + (f". 들은 소리(DTMF 톤 포함)를 {a.wav} 에 저장" if a.wav else ""))
        say(HELP)
        con.start(sound=not a.no_sound)
        cmds: asyncio.Queue = asyncio.Queue()
        threading.Thread(target=_read_stdin, args=(asyncio.get_running_loop(), cmds), daemon=True).start()
        while not (con.ended.is_set() or con.quit.is_set()):
            get = asyncio.create_task(cmds.get())
            stop = [asyncio.create_task(con.ended.wait()), asyncio.create_task(con.quit.wait())]
            done, _ = await asyncio.wait([get, *stop], return_when=asyncio.FIRST_COMPLETED)
            for t in stop:
                t.cancel()
            if get in done:
                kind, value = get.result()
                await (con.secure(value) if kind == "secure" else con.command(value))
            else:
                get.cancel()
        await asyncio.sleep(0.3)  # 마지막 소리 재생
        await con.close(hang_up=False)
        return 0
    except httpx.ConnectError:
        say(f"서버에 연결할 수 없습니다 ({a.base}). 다른 터미널에서 python app.py --sim 으로 서버를 먼저 실행하십시오.")
        return 2
    except (KeyboardInterrupt, asyncio.CancelledError):
        say("중단: " + ("통화를 종료합니다." if con.owner else "콘솔만 종료합니다 (통화 유지)."))
        try:
            await con.close(hang_up=con.owner)
        except Exception:
            pass
        return 130
    finally:
        speaker.close()


def ask_goal(base: str, phone: str, read=input) -> tuple[str, str | None]:
    """받는 번호와 원하는 업무(목적)를 입력받는다. 목 데이터 해석기이면 처리할 수 있는 목적 예시를 번호로 보여 준다."""
    try:
        h = httpx.get(base + "/v1/health", timeout=5).json()
    except httpx.HTTPError:
        print(f"서버에 연결할 수 없습니다 ({base}). 다른 터미널에서 python app.py --sim 으로 서버를 먼저 실행하십시오.")
        return phone, None
    backend = {"sim": "가상 ARS", "clawops": "ClawOps 실통화"}.get(h.get("backend"), h.get("backend"))
    print(f"서버: {backend}, 해석기: {h.get('interpreter')}")
    examples = (h.get("mock_goals") or []) if "Mock" in str(h.get("interpreter")) else []
    if examples:
        print("목적 예시 (AI 자리가 목 데이터라 아래 문장만 처리한다. 번호로 고를 수 있다)")
        for i, g in enumerate(examples, 1):
            print(f"  {i:2}) {g}")
    try:
        phone = read(f"받는 번호 [{phone}]: ").strip() or phone
        while True:
            s = read("원하는 업무 (예시 번호 또는 문장): ").strip()
            goal = examples[int(s) - 1] if s.isdigit() and 1 <= int(s) <= len(examples) else s
            if goal:
                break
            print("  목적을 입력해야 전화를 걸 수 있습니다.")
    except EOFError:
        print("목적이 필요합니다. --goal 로 지정하십시오.")
        return phone, None
    print(f'→ {phone} 로 "{goal}" 목적의 전화를 겁니다.')
    return phone, goal


def main() -> int:
    try:  # Windows 에서 출력을 파일로 돌려도 글자가 깨지지 않도록 UTF-8 로 출력한다
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="실시간 듣기·디버깅 콘솔 (스마트폰 앱 대역)")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--app-key", default="dev-app-key")
    ap.add_argument("--phone", default="15889999")
    ap.add_argument("--goal", help="목적 (앱의 음성 입력을 텍스트로 대신). 없으면 실행 후 입력받는다")
    ap.add_argument("--session", help="앱이 만든 세션에 붙을 때 세션 ID")
    ap.add_argument("--token", help="앱이 만든 세션에 붙을 때 세션 토큰")
    ap.add_argument("--player", choices=["auto", "sounddevice", "ffplay", "none"], default="auto")
    ap.add_argument("--wav", help="재생한 소리(DTMF 톤 포함)를 저장할 WAV 파일")
    ap.add_argument("--no-sound", action="store_true", help="처음에는 소리를 끈 상태로 시작 (m 으로 켬)")
    ap.add_argument("--raw", action="store_true", help="이벤트 원문(JSON)도 출력")
    a = ap.parse_args()
    try:
        if not a.session and not a.goal:  # 앱과 같이 목적을 먼저 정한 뒤 전화를 건다
            a.phone, a.goal = ask_goal(a.base, a.phone)
            if not a.goal:
                return 2
        return asyncio.run(run(a))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
