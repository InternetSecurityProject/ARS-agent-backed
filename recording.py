"""
통화 기록 (시험용, 선택 기능). 세션 생성 시 record=true 로 켠 경우에만 기록.

담는 것
  - 소리: 받은 ARS 소리와 서버가 보낸 DTMF 톤을 같은 시간축에 합친 8kHz WAV
  - 기록: 단락별 인식 문장(엔진·소요 시간), 안내 분류, 키 전송 시각(누가 눌렀는지), 요청·사용자 응답, 상태 변화, 결과 요약
민감정보
  - 사용자가 입력한 번호는 값 대신 자릿수만 기록. 그 DTMF 소리도 값을 알 수 없는 같은 길이의 신호음으로 대체
  - ARS 가 입력값을 다시 읽어 주는 소리와 그 인식 문장은 그대로 남음
보관
  - 서버 메모리: 세션 주인(session_token)만 내려받기 가능. 주인의 삭제(DELETE), 통화 종료 뒤 RECORDING_KEEP_S(기본 30분) 경과,
    서버 재시작 시 소멸
  - 디스크(RECORDING_SAVE=1, 기본): 통화 종료 시 서버의 RECORDING_DIR(기본 서버 폴더의 recordings/) 아래
    <목적>_<전화번호>_<기관>_<날짜-시각>/ 에 call.wav·log.json·log.txt 저장. 기관 이름은 bench/scenarios.json 에
    등록된 번호에만 추가. recordings/ 는 git 관리 대상 아님(.gitignore). 메모리 기록을 지워도 디스크 파일은 남으므로
    시험 후 직접 삭제 필요 (되읽기 소리·인식 문장 포함)
  - RECORD_ALWAYS=1 이면 앱이 record 를 보내지 않아도(Android 앱 등) 모든 통화 기록 (기본 끔)
기록 파일(log.json)은 bench/make_clips.py 의 녹음 메타데이터 형식(wav, actions[at_sample], captions) 준수.
"""

from __future__ import annotations

import array
import io
import json
import math
import re
import time
import wave
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from audio import dtmf_ulaw, ulaw_to_pcm16

RATE = 8000
KST = timezone(timedelta(hours=9))
HERE = Path(__file__).parent
_BAD_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f\s]+')  # 폴더 이름에 쓸 수 없는 글자와 공백
_ORGS: dict[str, str] | None = None
_KEY_RE = re.compile(r"[0-9*#]{1,4}")  # 서버가 받는 키 형식 (명세 key: 1~4자)


def org_of(phone: str) -> str:
    """전화번호의 기관 이름 (bench/scenarios.json 에 등록된 번호만). 모르면 빈 문자열."""
    global _ORGS
    if _ORGS is None:
        try:
            d = json.loads((HERE / "bench" / "scenarios.json").read_text(encoding="utf-8"))
            _ORGS = {re.sub(r"\D", "", str(s.get("phone", ""))): str(s.get("org", "")) for s in d.get("scenarios", [])}
        except Exception:
            _ORGS = {}
    return _ORGS.get(re.sub(r"\D", "", phone or ""), "")


def folder_name(goal: str, phone: str, created: datetime) -> str:
    """기록 폴더 이름: <목적>_<전화번호>_<기관>_<날짜-시각> (기관을 모르면 생략, 날짜-시각은 KST 기준 YYYYMMDD-HHMMSS)."""
    def clean(s: str, n: int) -> str:
        return _BAD_NAME.sub("", s or "").strip(".")[:n]

    parts = [clean(goal, 30) or "목적없음", re.sub(r"\D", "", phone or "")[:20] or "번호없음", clean(org_of(phone), 20),
             created.astimezone(KST).strftime("%Y%m%d-%H%M%S")]
    return "_".join(p for p in parts if p)


def save_files(root: str | Path, name: str, files: dict[str, bytes]) -> Path:
    """root/name/ 에 파일 저장. 같은 이름의 폴더가 있으면 -2, -3 … 접미사 추가. 만든 폴더 반환."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path, n = root / name, 1
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            n += 1
            path = root / f"{name}-{n}"
    for fname, data in files.items():
        (path / fname).write_bytes(data)
    return path


def _pcm(data: bytes) -> array.array:
    a = array.array("h")
    a.frombytes(data)
    return a


def _mask_tone(n_keys: int) -> array.array:
    """민감정보 입력 DTMF 대신 넣는 신호음. 키마다 DTMF 와 같은 길이(150ms 톤 + 100ms 쉼)의 1kHz 단음이므로 값 식별 불가."""
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
        """키 전송 시작 시각 기준 톤 기록. 민감정보는 값 대신 자릿수와 대체 신호음만 기록."""
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
        """서버가 앱으로 보낸 이벤트 (요청·자막·경로·상태·요약·종료). 민감정보 값은 이벤트에 포함되지 않음."""
        if name == "caption":
            self.captions.append({**self._stamp(), "text": data.get("text", "")})
        self.note("event:" + name, data=data)

    def stt(self, text: str, end_sample: int, seg_samples: int, seconds: float, engine: str) -> None:
        self.note("stt", text=text, engine=engine, stt_s=round(seconds, 2),
                  seg_start_t=round(max(0, end_sample - seg_samples) / RATE, 2), seg_end_t=round(end_sample / RATE, 2))

    def user_action(self, body: dict, mask: bool = False) -> None:
        """앱이 보낸 행동. 서버 검사 전에 기록하므로, 민감정보 입력 요청이 열려 있을 때(mask=True)나 키 형식이 틀린 값은
        값 대신 자릿수만 기록 (입력 요청 중 키패드로 누른 숫자가 409 로 거절돼도 기록·디스크에 남지 않도록)."""
        t = str(body.get("type", ""))
        data = {"action": t, "request_id": body.get("request_id")}
        if t in ("select", "key"):
            k = str(body.get("key", ""))
            if not mask and _KEY_RE.fullmatch(k):
                data["key"] = k
            else:
                data["digit_count"] = sum(c.isdigit() for c in k)
        elif t in ("answer", "revise_goal"):
            data["text"] = None if mask else body.get("text")
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
        wall = self.pos == 0  # 소리를 받지 못한 통화(가상 ARS 등): 실제 경과 시간 기준 표기
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

    def files(self) -> dict[str, bytes]:
        """디스크에 저장할 파일 내용 (call.wav, log.json, log.txt). 통화 종료 직후 이벤트 루프에서 바로 생성
        (그 뒤 주인의 삭제 요청·보관 시간 만료로 지워져도 저장 내용이 비지 않도록)."""
        return {"call.wav": self.wav_bytes(),
                "log.json": json.dumps(self.log(), ensure_ascii=False, indent=1).encode("utf-8"),
                "log.txt": self.text_log().encode("utf-8")}

    def wipe(self) -> None:
        self.pcm = array.array("h")
        self.tones.clear()
        self.events.clear()
        self.actions.clear()
        self.captions.clear()
        self.meta = {}
