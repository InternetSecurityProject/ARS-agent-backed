"""벤치마크 데이터 읽기.

구조
  bench/scenarios.json          시나리오 정의 (번호, 목적, 키 경로, 목적 변형별 기대 판단)
  bench/raw/<ID>-<n>.wav, .json 경로 고정 녹음 원본과 키 전송 기록 (record.py)
  bench/data/<ID>-<n>/          단락 WAV 와 clips.json (make_clips.py, 사람이 정답 확정)

clips.json 단락 필드
  file      단락 WAV (8kHz 16bit mono)
  text      정답 문장 (말소리 없는 단락은 빈 문자열)
  zone      A(첫 민감정보 입력 전) / B(이후)
  kind      menu, confirm, sensitive, result, agent_wait, input_error, voice_request, out_of_hours, message_sent, info
  options   메뉴·확인 안내의 [[키, 이름], ...]
  digits    숫자 되읽기 단락이면 정답 숫자열
  cut       키 전송에 잘린 조각이면 true (글자 오류율·메뉴 키 집계에서 뺀다)
  decision  이 단락 뒤에 누른 키(시나리오 키 경로의 decision point 번호). 없으면 null
  verified  사람이 정답을 확인했으면 true
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field
from pathlib import Path

BENCH = Path(__file__).parent
SCENARIOS = BENCH / "scenarios.json"
DATA = BENCH / "data"


@dataclass
class Clip:
    path: Path
    text: str
    zone: str = "A"
    kind: str | None = None
    options: list = field(default_factory=list)
    digits: str | None = None
    cut: bool = False
    decision: int | None = None
    verified: bool = False

    def pcm(self) -> bytes:
        with wave.open(str(self.path), "rb") as w:
            if w.getframerate() != 8000 or w.getsampwidth() != 2 or w.getnchannels() != 1:
                raise SystemExit(f"{self.path}: 8kHz 16bit mono WAV 만 지원")
            return w.readframes(w.getnframes())


@dataclass
class Recording:
    name: str          # 폴더 이름 (예: K1-1)
    scenario: str      # 시나리오 ID (예: K1)
    clips: list[Clip]


def load_scenarios(path: Path = SCENARIOS) -> dict[str, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {s["id"]: s for s in d["scenarios"]}


def decision_steps(scenario: dict) -> list[dict]:
    """키 경로 가운데 판단이 일어나는 단계 (키 누르기·번호 입력). observe 단계는 제외."""
    return [s for s in scenario.get("route", []) if "key" in s or "input" in s]


def gold_expect(scenario: dict) -> list[str]:
    """시나리오 기본 경로의 기대 판단 (키는 press:키, 번호 입력은 secure)."""
    return [f"press:{s['key']}" if "key" in s else "secure" for s in decision_steps(scenario)]


def load_recordings(data: Path = DATA, only: set[str] | None = None) -> list[Recording]:
    out = []
    for f in sorted(Path(data).glob("*/clips.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sid = d.get("scenario") or f.parent.name.split("-")[0]
        if only and sid not in only and f.parent.name not in only:
            continue
        clips = [Clip(path=f.parent / c["file"], text=c.get("text", ""), zone=c.get("zone", "A"), kind=c.get("kind"),
                      options=c.get("options") or [], digits=c.get("digits"), cut=bool(c.get("cut")),
                      decision=c.get("decision"), verified=bool(c.get("verified")))
                 for c in d["clips"]]
        out.append(Recording(f.parent.name, sid, clips))
    return out


def load_clip_folder(folder: Path) -> Recording:
    """기존 bench_clips/·bench_real/ 형식(clips.json 의 file·text 만 있는 폴더)도 읽는다."""
    d = json.loads((Path(folder) / "clips.json").read_text(encoding="utf-8"))
    clips = [Clip(path=Path(folder) / c["file"], text=c.get("text", ""), zone=c.get("zone", "A"), kind=c.get("kind"),
                  options=c.get("options") or [], digits=c.get("digits"), cut=bool(c.get("cut")),
                  decision=c.get("decision"), verified=bool(c.get("verified")))
             for c in d["clips"]]
    return Recording(Path(folder).name, d.get("scenario", Path(folder).name), clips)
