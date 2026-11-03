"""
통화 기록 (시험용, 선택 기능). 세션을 만들 때 record=true 로 켠 경우에만 남긴다.

담는 것
  - 소리: 받은 ARS 소리와 서버가 보낸 DTMF 톤을 같은 시간축에 합친 8kHz WAV
  - 기록: 단락별 인식 문장(엔진·소요 시간), 안내 분류, 키 전송 시각(누가 눌렀는지), 요청·사용자 응답, 상태 변화, 결과 요약
민감정보
  - 사용자가 입력한 번호는 값을 남기지 않는다 (자릿수만). 그 DTMF 소리도 값을 알 수 없는 같은 길이의 신호음으로 대신 넣는다
  - ARS 가 입력값을 다시 읽어 주는 소리와 그 인식 문장은 그대로 남는다
보관
  - 서버 메모리에만 둔다 (디스크에 쓰지 않음). 세션 주인(session_token)만 내려받을 수 있다
  - 주인이 지우거나(DELETE), 통화 종료 뒤 RECORDING_KEEP_S(기본 30분)가 지나거나, 서버가 다시 시작되면 사라진다
기록 파일(log.json)은 bench/make_clips.py 의 녹음 메타데이터 형식(wav, actions[at_sample], captions)을 따른다.
"""

from __future__ import annotations

import array
import io
import json
import math
import time
import wave
import zipfile
from datetime import datetime, timedelta, timezone

from audio import dtmf_ulaw, ulaw_to_pcm16

RATE = 8000
KST = timezone(timedelta(hours=9))


def _pcm(data: bytes) -> array.array:
    a = array.array("h")
    a.frombytes(data)
    return a


def _mask_tone(n_keys: int) -> array.array:
    """민감정보 입력 DTMF 대신 넣는 신호음. 키마다 DTMF 와 같은 길이(150ms 톤 + 100ms 쉼)의 1kHz 단음이라 값을 알 수 없다."""
    one = [int(0.25 * 32767 * math.sin(2 * math.pi * 1000 * i / RATE)) for i in range(RATE * 150 // 1000)]
    one += [0] * (RATE * 100 // 1000)
    return array.array("h", one * n_keys)


def _clock(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


class CallRecorder:
    def __init__(self, meta: dict) -> None:
        self.meta = {**meta, "created_at": datetime.now(KST).isoformat(timespec="seconds")}
        self.pcm = array.array("h")
        self.tones: list[tuple[int, array.array]] = []
        self.events: list[dict] = []
        self.actions: list[dict] = []
        self.captions: list[dict] = []
        self.t0 = time.monotonic()

    @property
    def pos(self) -> int:
        return len(self.pcm)

    def _stamp(self) -> dict:
        return {"t": round(self.pos / RATE, 2), "at_sample": self.pos, "wall": round(time.monotonic() - self.t0, 2)}

    def note(self, etype: str, **data) -> None:
        self.events.append({**self._stamp(), "type": etype, **data})

    # 소리 ------------------------------------------------------------------
    def audio(self, frame: bytes) -> None:
        self.pcm.extend(_pcm(ulaw_to_pcm16(frame)))

    def keys(self, keys: str, actor: str, secure: bool) -> None:
        """키 전송 시작 시각에 톤을 기록한다. 민감정보는 값 대신 자릿수와 대체 신호음만 남긴다."""
        if secure:
            self.tones.append((self.pos, _mask_tone(len(keys))))
            act = {"type": "input", "actor": actor, "digits": None, "digit_count": sum(c.isdigit() for c in keys)}
        else:
            self.tones.append((self.pos, _pcm(ulaw_to_pcm16(dtmf_ulaw(keys)))))
            act = {"type": "key", "actor": actor, "keys": keys}
        self.actions.append({**act, **self._stamp()})
        self.note("keys_sent", **{k: v for k, v in act.items() if k != "type"}, kind=act["type"])

    # 기록 ------------------------------------------------------------------
    def event(self, name: str, data: dict) -> None:
        """서버가 앱으로 보낸 이벤트 (요청·자막·경로·상태·요약·종료). 민감정보 값은 이벤트에 들어 있지 않다."""
        if name == "caption":
            self.captions.append({**self._stamp(), "text": data.get("text", "")})
        self.note("event:" + name, data=data)

    def stt(self, text: str, end_sample: int, seg_samples: int, seconds: float, engine: str) -> None:
        self.note("stt", text=text, engine=engine, stt_s=round(seconds, 2),
                  seg_start_t=round(max(0, end_sample - seg_samples) / RATE, 2), seg_end_t=round(end_sample / RATE, 2))

    def user_action(self, body: dict) -> None:
        t = str(body.get("type", ""))
        data = {"action": t, "request_id": body.get("request_id")}
        if t in ("select", "key"):
            data["key"] = body.get("key")
        elif t in ("answer", "revise_goal"):
            data["text"] = body.get("text")
        elif t == "secure_input":
            data["digit_count"] = sum(c.isdigit() for c in str(body.get("digits", "")))
        self.note("user", **data)

    # 내보내기 --------------------------------------------------------------
    def wav_bytes(self) -> bytes:
        mix = array.array("h", self.pcm)
        end = max([p + len(t) for p, t in self.tones], default=0)
        if end > len(mix):
            mix.extend([0] * (end - len(mix)))
        for p, tone in self.tones:
            for i, v in enumerate(tone):
                s = mix[p + i] + v
                mix[p + i] = 32767 if s > 32767 else -32768 if s < -32768 else s
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(mix.tobytes())
        return buf.getvalue()

    def log(self) -> dict:
        return {"version": 1, **self.meta, "wav": "call.wav", "seconds": round(self.pos / RATE, 1),
                "actions": self.actions, "captions": self.captions, "events": self.events}

    def text_log(self) -> str:
        lines = [f"ARS 통화 기록  {self.meta.get('created_at', '')}",
                 f"번호 {self.meta.get('phone', '')} · 목적 {self.meta.get('goal', '')}",
                 f"STT {self.meta.get('stt', '')} · 해석기 {self.meta.get('interpreter', '')}",
                 "시각은 통화 연결부터 잰 시간 (call.wav 와 같은 시간축)", ""]
        wall = self.pos == 0  # 소리를 받지 못한 통화(가상 ARS 등)는 실제 경과 시간으로 적는다
        for e in self.events:
            t, k = _clock(e["wall"] if wall else e["t"]), e["type"]
            if k == "stt":
                lines.append(f"[{t}] 인식({e['engine']}, {e['stt_s']}초, 음성 {_clock(e['seg_start_t'])}~{_clock(e['seg_end_t'])}) {e['text']}")
            elif k == "stt_skipped":
                lines.append(f"[{t}] 인식 안 함: {e.get('reason', '')}")
            elif k == "classified":
                opts = ", ".join(e.get("options") or [])
                lines.append(f"[{t}] 분류 {e.get('kind')}" + (f" · {opts}" if opts else ""))
            elif k == "keys_sent":
                who = "AI" if e.get("actor") == "ai" else "사용자"
                what = f"민감정보 {e.get('digit_count')}자리 (값 기록 안 함)" if e.get("kind") == "input" else f"키 {e.get('keys')}"
                lines.append(f"[{t}] {who} → {what}")
            elif k == "user":
                extra = e.get("key") or e.get("text") or (f"{e.get('digit_count')}자리" if e.get("digit_count") is not None else "")
                lines.append(f"[{t}] 사용자 응답 {e.get('action')} {extra}".rstrip())
            elif k.startswith("event:"):
                name, d = k[6:], e.get("data") or {}
                if name == "caption":
                    lines.append(f"[{t}] 자막 {d.get('text', '')}")
                elif name == "status_changed":
                    lines.append(f"[{t}] 상태 {d.get('status')} {d.get('message', '')}".rstrip())
                elif name == "request":
                    opts = ", ".join(f"{o.get('key')}:{o.get('label')}" for o in d.get("options") or [])
                    lines.append(f"[{t}] 요청 {d.get('kind')}/{d.get('reason') or d.get('label') or ''} {d.get('message', '')}"
                                 + (f" [{opts}]" if opts else ""))
                elif name == "path_updated":
                    sel = d.get("selected") or {}
                    c = sel.get("confidence")
                    lines.append(f"[{t}] 경로 {d.get('step')}단계 {sel.get('label')} ({sel.get('key')})"
                                 + (f" 신뢰도 {c:.2f}" if isinstance(c, (int, float)) else ""))
                elif name == "input_rejected":
                    lines.append(f"[{t}] 입력 거절 {d.get('message', '')}")
                elif name == "summary":
                    lines.append(f"[{t}] 요약 {d.get('summary', '')}")
                elif name == "ended":
                    lines.append(f"[{t}] 종료 {d.get('end_reason')} {d.get('message', '')}".rstrip())
        return "\n".join(lines) + "\n"

    def export_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("call.wav", self.wav_bytes())
            z.writestr("log.json", json.dumps(self.log(), ensure_ascii=False, indent=1))
            z.writestr("log.txt", self.text_log())
        return buf.getvalue()

    def wipe(self) -> None:
        self.pcm = array.array("h")
        self.tones.clear()
        self.events.clear()
        self.actions.clear()
        self.captions.clear()
        self.meta = {}
