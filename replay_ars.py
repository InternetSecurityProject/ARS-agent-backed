"""
녹음 기반 가상 ARS (시험용, 실통화 없음)

실통화 녹음(recordings/*/call.wav·log.json)에서 뽑은 상태 그래프와 실제 ARS 소리 조각으로, 그 기관 ARS 를 녹음에 있는
갈래 안에서 녹음과 같은 소리·시간으로 재생.

  CALL_BACKEND=replay REPLAY_BUNDLE=<묶음 폴더> python app.py      # 묶음은 replay_build.py 로 생성

동작
  - 상태 = ARS 가 그 상태에 들어선 뒤 다음 입력을 기다릴 때까지 말하는 소리 전체. 녹음의 소리 조각을 20ms μ-law 프레임으로
    송출. 말하지 않는 구간은 녹음과 같이 디지털 무음(묶음에 회선 잡음 조각이 있으면 그 잡음)을 송출.
  - 시간은 프레임 수로 계산(ReplayEngine). 서버에서는 20ms 마다 한 프레임씩(실시간), 검증(replay_check.py)에서는 같은 엔진을
    시계 없이 실행. 두 경우 모두 같은 코드로 같은 시간표 생성.
  - 키: 실통화(ClawOps 톤 방식)처럼 자리마다 톤 150ms + 쉼 100ms 를 실시간으로 보낸 뒤 반환(반환값 0). 톤을
    detect_dtmf 로 해독해 받고, 자리마다 톤이 끝난 시각에 ARS 가 받은 것으로 처리.
  - 끼어들기, 키 뒤 다음 안내까지의 지연, 안내가 끝난 뒤 입력이 없을 때의 동작(after_audio), 연속 무입력 규칙은 녹음에서 잰
    값 사용(기본은 가운뎃값, REPLAY_JITTER=1 이면 녹음 범위 안에서 무작위). 실제 ARS 가 스스로 고르는 소리 변형은 관측 비율대로
    선택(REPLAY_SEED 로 고정).
  - 녹음에 없는 갈래는 지어내지 않고 통화 종료(녹음 밖 종료): 본 적 없는 키(잘못 누른 키의 안내도 녹음에 없음), 녹음에 없는
    입력 시간 초과 동작, 녹음에서 본 것보다 많은 같은 안내 반복, 입력 뒤 상태가 녹음에 없음. 개인정보가 들어 있을 수 있는
    상태(민감정보 입력 뒤 되읽기·조회 결과)도 재생하지 않고 종료. 단, 개인정보 포함 묶음(replay_build.py
    --include-personal, bundle.json includes_personal)에서는 그 상태의 실제 녹음을 그대로 재생(되읽기는 녹음 당시의 실제
    번호를 읽으며, 서버가 보낸 가짜 번호와 자릿수만 같음).
  - 녹음에 근거가 부족한 동작은 계속 진행하되 path 에 표시(flag)를 남김: 끼어들기 여부를 모르는 상태에서 안내 도중 받은 키
    (barge_unverified), 안내 끝 4.5초 뒤의 늦은 키(late_key), 다음 안내 전에 미리 누른 키(type_ahead, 앞 상태의 키로
    처리하지 않고 폐기), 지연을 재지 못한 연결(unmeasured_delay), 입력 값에 따라 결과가 갈리는 입력(value_dependent),
    입력 도중 시간 초과를 재지 못한 입력(unmeasured_digit_timeout). 시험 결과에서는 첫 녹음 밖 종료·표시 이후를 근거로 보지 않음.
  - 통화마다 지나온 상태·키·표시·종료 이유를 REPLAY_LOG(jsonl, 기본 replay_log.jsonl)에 기록. 서버에는 녹음 밖 종료도 실제
    ARS 끊음과 같은 remote_hangup 으로 보이므로 이 기록으로 구분.
  - 가상 ARS 통화는 묶음을 만드는 자료로 다시 쓰지 않음 (replay_build.py 는 log.json 의 backend 가 clawops 인 녹음만 사용).
  - 받아쓰기 모드(REPLAY_TRANSCRIPTS=1): 서버가 소리를 인식하지 않고(STT 없음), 소리 조각이 녹음될 때 서버가 실제로 받아쓴 문장을
    같은 단락 경계로 전달(on_transcript). 소리는 그대로 송출(앱으로 들림). 실제 서버의 STT 경로와 같게: 키 전송이 끝날 때 아직
    닫히지 않은 단락은 폐기(서버가 말하던 단락을 버림), 상태가 바뀌어도 이미 시작된 단락은 유지, ARS 가 끊으면 듣던
    단락을 전달(flush_transcripts). 45초에서 잘린 단락은 forced 로 넘겨 세션이 다음 단락과 이어 붙임. 인식 지연은 단락 소리
    길이별로 녹음에서 잰 가운뎃값, 인식 문장은 그 조각이 녹음될 때의 한 번의 결과(인식 오류 변동은 재현하지 않음).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from audio import FRAME, detect_dtmf, dtmf_ulaw
from telephony import CallHandlers

log = logging.getLogger("ars.replay")

FRAME_S = FRAME / 8000  # 20ms
KST = timezone(timedelta(hours=9))

# 녹음에 입력 시간 초과 동작이 없는 상태에서 '녹음 밖' 으로 끝내기까지 기다리는 시간(초)의 기본값. 묶음의 기관별 값
# (unseen_timeout_s = 그 기관에서 관측한 시간 초과의 최댓값)이 우선. 그 시간이 지나면 실제 ARS 는 어떤 동작을 했을 것으로 간주
UNSEEN_TIMEOUT_S = 7.0
# 상태별로 녹음에서 조용히 기다린 최대 시간(묶음 silent_wait_s)에 더하는 여유(초)
SILENT_MARGIN_S = 0.5
# 두 자리 이상 키의 앞자리만 받고 다음 자리를 기다리는 최대 시간(초). 녹음에서 재지 않고 임의로 정한 값
KEY_WAIT_S = 3.0
# 자리 간격(톤 150ms + 쉼 100ms)과 톤 소리 길이. 실통화 톤 방식(audio.dtmf_ulaw 기본값)과 같음
KEY_TONE_ON_S = 0.15
KEY_TONE_S = 0.25
# 안내가 끝난 뒤 이 시간이 지나 받은 메뉴 키는 '늦은 키' 로 표시. 그보다 늦은 키는 실제 ARS 가 무시하거나 시간 초과 안내를
# 먼저 낼 수 있음
LATE_KEY_S = 4.5
# 같은 안내가 시간 초과로 연속해서 다시 나오는 횟수의 기본 상한 (묶음에 max_plays 가 없을 때). 자기 반복 연결이 녹음에 있으면
# 적어도 두 번은 관측된 것임
DEFAULT_MAX_PLAYS = 2
# 받아쓰기 모드의 단락 시작 시각 보정. 녹음의 seg_start_t 는 서버 Segmenter 의 앞 여유 200ms 를 포함하고, 실제 서버 경로의
# seg_started_at 은 소리를 확인해 단락을 연 시각임
DETECT_LAG_S = 0.2


def pick_start(org: dict, now: datetime) -> str:
    """시간대에 따라 다른 첫 상태(평일 낮 메뉴·밤 메뉴 등). org['start_rules']: [{"days": [0..6], "from": "09:00", "to": "18:00",
    "state": "S0"}] 앞에서부터 처음 맞는 것, 없으면 org['start']. days 는 월=0 … 일=6. 공휴일은 구분하지 않음."""
    hm = now.strftime("%H:%M")
    for r in org.get("start_rules") or []:
        if now.weekday() in r.get("days", range(7)) and r.get("from", "00:00") <= hm < r.get("to", "24:00"):
            return r["state"]
    return org["start"]


def _frames(raw: bytes) -> list[bytes]:
    out = [raw[i:i + FRAME] for i in range(0, len(raw), FRAME)]
    if out and len(out[-1]) < FRAME:
        out[-1] = out[-1].ljust(FRAME, b"\xff")
    return out


def _measured(d) -> bool:
    """녹음에서 잰 지연인지 여부 (가운뎃값이 있는 기록). 기록이 없거나 가운뎃값이 비면 기본값을 쓰고 unmeasured_delay 로 표시."""
    return isinstance(d, dict) and isinstance(d.get("median"), (int, float))


def _meaningful(text: str) -> bool:
    """실제 서버의 STT 경로와 같은 거르기: 글자·숫자가 세 자 미만인 인식 결과는 폐기."""
    return len(re.sub(r"[^가-힣0-9A-Za-z]", "", text or "")) >= 3


class ReplayBundle:
    """묶음 폴더: bundle.json + 소리 조각(.ulaw, 8kHz μ-law). 형식은 replay_build.py 참고."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "bundle.json").read_text(encoding="utf-8"))
        self.orgs: dict = self.meta["orgs"]
        self.phones: dict[str, str] = self.meta["phones"]
        self._clips: dict[str, bytes] = {}

    def clip(self, rel: str) -> bytes:
        if rel not in self._clips:
            self._clips[rel] = (self.dir / rel).read_bytes()
        return self._clips[rel]

    def org_for(self, phone: str) -> str | None:
        digits = "".join(ch for ch in str(phone) if ch.isdigit())
        return self.phones.get(digits)


class ReplayEngine:
    """프레임 단위로 도는 녹음 기반 ARS 상태 기계 (동기식, 시계 없음).
    tick() 이 다음 20ms 프레임을 반환하고, key() 로 그 순간 받은 키를 넣음. 키 전송이 끝나면 keys_done() 호출.
    통화가 끝나면 ended 와 end_reason 이 설정됨."""

    def __init__(self, bundle: ReplayBundle, org: str, *, jitter: bool = False, seed: int | None = None,
                 start: str | None = None, clock: datetime | None = None) -> None:
        self.b, self.org_id = bundle, org
        self.org = bundle.orgs[org]
        self.states: dict = self.org["states"]
        self.rand = random.Random(seed)
        self.jitter = jitter
        self.frame = 0                       # 지금까지 낸 프레임 수 (통화 연결부터)
        self.state_id: str | None = None
        self.out: list[bytes] = []           # 남은 안내 프레임
        self.speaking = False
        self.ended = False
        self.end_reason: str | None = None   # remote_hangup
        self.note: str | None = None         # 녹음 밖 종료 이유
        self.digits = 0
        # 지나온 상태·키·표시: {"t", "state", "via", "variant"?} / {"t", "key"} / {"t", "flag"} / {"t", "end"}.
        # 키 값은 메뉴 키만 남기고 민감정보 입력은 '*입력*'
        self.path: list[dict] = []
        self.flags: list[str] = []
        self._timers: list[tuple[int, int, str, str | None, str | None]] = []  # (프레임, 세대, 할 일, 대상 상태, 표시)
        self._gen = 0
        self._pending_key: str | None = None
        self._pending_enter = False          # 키 연결의 다음 상태 진입을 기다리는 중
        self._waiting_after = False          # 안내가 끝나 입력을 기다리는 중
        self._audio_done_frame = 0
        self._keybuf = ""                    # 두 자리 이상 키('10'·'77')를 모으는 중
        self._timeouts = 0                   # 연속 입력 시간 초과 횟수 (키를 받으면 0)
        self._plays = 0                      # 지금 상태가 시간 초과로 연속해서 나온 횟수 (처음 들어오면 1)
        self.choose_variant = None           # (상태, 변형 목록) → 고를 번호. 없으면 관측 비율대로 선택
        self._paras: list[dict] = []         # 아직 내보내지 않은 단락 받아쓰기 (프레임 단위 start·end·at)
        self.transcripts: list[tuple[str, float, bool]] = []  # 나온 받아쓰기 (문장, 단락 시작 초, forced). 호출하는 쪽이 가져감
        self._unseen_s = float(self.org.get("unseen_timeout_s") or UNSEEN_TIMEOUT_S)
        self.cur_clip: str | None = None     # 지금 상태의 소리 조각 (검증에서 프레임별 출처 구분용)
        self.last_src: str | None = None     # 방금 낸 프레임의 조각 (무음이면 None)
        noise = self.org.get("noise")
        self._noise = _frames(bundle.clip(noise)) if noise else [b"\xff" * FRAME]
        self.start_state = start or pick_start(self.org, clock or datetime.now(KST))
        self._schedule(self._delay(self.org.get("first_audio_s"), 0.5), "enter", self.start_state, via="start")

    # 도구 --------------------------------------------------------------------
    @property
    def t(self) -> float:
        return round(self.frame * FRAME_S, 2)

    def _delay(self, d, default: float) -> float:
        if not isinstance(d, dict):
            return float(d) if isinstance(d, (int, float)) else default
        if self.jitter and d.get("min") is not None and d.get("max") is not None:
            return self.rand.uniform(float(d["min"]), float(d["max"]))
        v = d.get("median")
        return float(v) if isinstance(v, (int, float)) else default  # 녹음에서 재지 못한 값은 기본값

    def _schedule(self, after_s: float, what: str, target: str | None = None, flag: str | None = None,
                  via: str | None = None) -> None:
        # via 는 대상 상태로 들어가는 까닭(start·key·timeout·third_fail·input·digit_timeout·invalid). 표시 칸과 함께 한 문자열로 저장
        self._timers.append((self.frame + max(0, round(after_s / FRAME_S)), self._gen, what, target,
                             f"{via or ''}|{flag or ''}"))

    def _flag(self, name: str) -> None:
        self.path.append({"t": self.t, "flag": name})
        if name not in self.flags:
            self.flags.append(name)

    @property
    def text_pending(self) -> bool:
        """소리가 시작됐지만 받아쓰기가 아직 나오지 않은 단락이 있는지 여부 (실제 서버의 '단락을 듣는 중·인식 중' 에 해당)."""
        return any(p["start"] <= self.frame for p in self._paras)

    def keys_done(self) -> None:
        """키 전송이 끝난 때: 아직 닫히지 않은 단락의 받아쓰기는 내보내지 않음 (실제 서버는 키를 보낸 뒤 Segmenter 를 비워 말하던
        단락을 버림). 이미 닫힌 단락은 인식 중이므로 늦게라도 나옴."""
        self._paras = [p for p in self._paras if p["end"] <= self.frame]

    def flush(self) -> list[tuple[str, float, bool]]:
        """통화가 끝날 때 듣던 단락 (실제 서버는 ARS 가 끊으면 Segmenter 에 남은 단락을 인식해 넘김). 소리가 시작된 것만 반환."""
        out = [(p["text"], p["start"] * FRAME_S + DETECT_LAG_S, p["forced"]) for p in self._paras if p["start"] <= self.frame]
        self._paras = []
        return out

    def _end(self, note: str | None = None) -> None:
        if self.ended:
            return
        self.ended, self.end_reason, self.note = True, "remote_hangup", note
        self.out.clear()
        self.speaking = False
        self.path.append({"t": self.t, "end": note or "hangup"})
        if note:
            log.warning("가상 ARS(%s) 녹음 밖: %s (상태 %s, %.1f초)", self.org_id, note, self.state_id, self.t)

    # 상태 --------------------------------------------------------------------
    def _enter(self, state_id: str, via: str = "key") -> None:
        if state_id == "hangup":
            self._end()
            return
        if state_id not in self.states:
            self._end(f"모르는 상태 {state_id}")
            return
        # 시간 초과(안내 뒤 무입력·입력 도중 멈춤)로 같은 상태에 다시 들어오면 같은 안내의 반복으로 계산
        self._plays = self._plays + 1 if (via in ("timeout", "digit_timeout") and state_id == self.state_id) else 1
        self._gen += 1
        self.state_id, self.digits, self._pending_key, self._waiting_after = state_id, 0, None, False
        self._pending_enter = False
        s = self.states[state_id]
        self.path.append({"t": self.t, "state": state_id, "via": via})
        if s.get("personal"):
            if not (self.b.meta.get("includes_personal") and (s.get("clip") or s.get("variants"))):
                self._end("개인정보가 들어 있을 수 있는 상태는 재생하지 않음")
                return
            # 개인정보 포함 묶음(includes_personal): 되읽기·조회 결과 녹음을 그대로 재생 (내용은 기록에 남기지 않음)
            self.path[-1]["personal"] = True
        if not s.get("clip") and not s.get("variants"):
            # 이 상태의 소리가 묶음을 만든 녹음에 없음 (replay_build --exclude 로 일부 녹음을 빼고 만든 묶음에서 발생 가능). 무음으로 지나가지 않음
            self._end("소리 조각이 없는 상태")
            return
        clip, paras = s.get("clip"), s.get("paras") or []
        if s.get("variants"):
            # 실제 ARS 가 스스로 고르는 변형(예: 혼잡할 때만 붙는 안내 문장): 관측 비율대로 선택 (검증은 choose_variant 로 지정)
            vs = s["variants"]
            i = self.choose_variant(state_id, vs) if self.choose_variant else (
                self.rand.choices(range(len(vs)), weights=[v.get("weight", 1) for v in vs])[0])
            clip, paras = vs[i]["clip"], vs[i].get("paras") or []
            self.path[-1]["variant"] = vs[i].get("name")
        self.cur_clip = clip
        f0 = self.frame
        # 앞 상태에서 이미 시작된 단락은 유지 (상태가 바뀌어도 실제 서버의 Segmenter 는 단락을 버리지 않음)
        self._paras += [{"text": p["text"], "start": f0 + round(p["start"] / FRAME_S), "end": f0 + round(p["end"] / FRAME_S),
                         "at": f0 + round(p["at"] / FRAME_S), "forced": bool(p.get("forced"))} for p in paras]
        self.out = _frames(self.b.clip(clip)) if clip else []
        self.speaking = bool(self.out)
        if not self.out:
            self._audio_done()

    def _audio_done(self) -> None:
        """안내를 다 말한 뒤: 기다리던 키, 입력 시간 초과 동작, 녹음에 없으면 일정 시간 뒤 녹음 밖 종료."""
        self.speaking = False
        self._audio_done_frame = self.frame
        if self._pending_key is not None:
            key, self._pending_key = self._pending_key, None
            self._apply_key(key)
            return
        self._waiting_after = True
        st = self.states[self.state_id]
        if st.get("input") and self.digits:
            return  # 입력 중: 시간 초과는 마지막 자리부터 잼 (key)
        after = st.get("after_audio") or {}
        nxt, wait, via = after.get("next"), after.get("timeout_s"), "timeout"
        limit = int(after.get("max_plays") or DEFAULT_MAX_PLAYS)
        if after.get("third_fail_next") and (self._timeouts >= 2 or (nxt == self.state_id and self._plays >= limit)):
            # 연속 세 번째 입력 없음(무입력 3회 규칙), 또는 자기 반복이 상한에 닿음(같은 안내가 정해진 횟수 나온 다음 종료 안내로
            # 가는 ARS)
            nxt, wait, via = after["third_fail_next"], after.get("third_fail_timeout_s") or wait, "third_fail"
        if nxt:
            self._schedule(self._delay(wait, 5.0), "timeout", nxt, via=via)
        else:
            # 시간 초과 동작이 녹음에 없음: 녹음에서 이 상태가 조용히 기다린 최대 시간(+0.5초)과 기관의 시간 초과 상한 중 긴 쪽까지
            # 대기한 뒤 녹음 밖 종료
            seen = st.get("silent_wait_s")
            self._schedule(max(self._unseen_s, float(seen) + SILENT_MARGIN_S) if seen else self._unseen_s, "unseen")

    # 키 ----------------------------------------------------------------------
    def key(self, ch: str) -> None:
        """ARS 가 키 한 자리를 받은 순간 (톤이 끝난 시각)."""
        if self.ended or self.state_id is None:
            return
        s = self.states[self.state_id]
        inp = s.get("input")
        self.path.append({"t": self.t, "key": "*입력*" if inp else ch})
        keys = s.get("keys") or {}
        continuing = bool(self._keybuf) and any(k.startswith(self._keybuf + ch) for k in keys)
        if self._pending_enter and not continuing:
            # 다음 안내가 나오기 전에 누른 키: 실제 ARS 가 버리는지 다음 메뉴로 넘기는지 녹음에 없음. 앞 상태의 키로 처리하지 않고
            # 폐기하며 표시. 입력 상태에서 정한 자릿수를 다 입력한 뒤 서버가 덧붙인 '#' 은 표시 없이 폐기
            if not (inp and ch == "#"):
                self._flag("type_ahead")
            return
        if self.speaking and s.get("barge_in") != "yes":
            self._flag("barge_unverified")  # 이 상태에서 안내 도중 키를 받았을 때의 ARS 동작이 녹음에 없음
        if (not inp and self._waiting_after and not continuing
                and (self.frame - self._audio_done_frame) * FRAME_S > LATE_KEY_S):
            self._flag("late_key")
        if inp:
            if self.speaking:
                # 입력이 시작되면 안내 중단 (서버는 사용자가 누르는 대로 한 자리씩 바로 송출)
                keep = int(float(self.org.get("barge_stop_s") or 0) / FRAME_S)
                self.out = self.out[:keep]
                self.speaking = bool(self.out)
            term = inp.get("terminator")
            n = inp.get("len") if isinstance(inp.get("len"), int) else None
            done = ch == term or (ch.isdigit() and n is not None and self.digits + 1 >= n and not term)
            if ch.isdigit():
                self.digits += 1  # 값은 남기지 않음
            if done:
                nxt = s.get("input_next")
                if not nxt:
                    self._end("입력 뒤 상태가 녹음에 없음")
                    return
                if s.get("input_value_dependent"):
                    self._flag("value_dependent")  # 녹음에서 입력 값에 따라 결과가 갈림: 가장 많이 본 결과로 진행
                if not _measured(s.get("input_delay_s")):
                    self._flag("unmeasured_delay")
                self.out, self.speaking = [], False
                self._gen += 1
                self._timeouts = 0
                self._pending_enter = True
                self._schedule(self._delay(s.get("input_delay_s"), 0.5), "enter", nxt, via="input")
                return
            # 다 입력하지 않았으면 마지막 자리부터 입력 시간 초과를 다시 잼
            self._gen += 1
            after = s.get("after_audio") or {}
            if after.get("next"):
                measured = inp.get("digit_timeout_s")
                wait = measured or self._delay(after.get("timeout_s"), 5.0)
                nxt, via = after["next"], "digit_timeout"
                limit = int(after.get("max_plays") or DEFAULT_MAX_PLAYS)
                if after.get("third_fail_next") and (self._timeouts >= 2 or (nxt == self.state_id and self._plays >= limit)):
                    # 입력 도중 멈춤도 실패 한 번: 연속 세 번째이거나 같은 안내가 상한에 닿으면 세 번째 규칙의 상태로 이동
                    nxt, via = after["third_fail_next"], "third_fail"
                    wait = measured or self._delay(after.get("third_fail_timeout_s") or after.get("timeout_s"), 5.0)
                self._schedule(float(wait), "timeout", nxt,
                               flag=None if measured else "unmeasured_digit_timeout", via=via)
            else:
                self._schedule(self._unseen_s, "unseen")  # 입력 도중 멈췄을 때의 동작이 녹음에 없음
            return
        if self.speaking and s.get("barge_in") == "no":
            self._pending_key = ch  # 안내 도중 키를 받아도 말을 멈추지 않는 ARS: 안내가 끝난 뒤 처리
            return
        barged = self.speaking
        # 끼어들기: 실제 ARS 는 키를 받은 뒤 잠깐(barge_stop_s) 더 말하다 멈춤
        keep = int(float(self.org.get("barge_stop_s") or 0) / FRAME_S) if barged else 0
        self.out = self.out[:keep]
        self.speaking = bool(self.out)
        buf = self._keybuf + ch
        if any(len(k) > len(buf) and k.startswith(buf) for k in keys):
            # 두 자리 이상 키의 앞자리: 한 자리 키 연결을 예약해 두고, 그 전에 다음 자리가 오면 두 자리 키로 처리
            self._keybuf = buf
            self._apply_key(buf if buf in keys else ch, barged)
            return
        self._keybuf = ""
        self._apply_key(buf if buf in keys else ch, barged)

    def _apply_key(self, ch: str, barged: bool = False) -> None:
        s = self.states[self.state_id]
        self._gen += 1  # 이 상태의 시간 초과·먼저 예약한 한 자리 키 연결은 무효화
        self._timeouts = 0
        edge = (s.get("keys") or {}).get(ch) or {}
        if edge.get("next"):
            if barged and _measured(edge.get("barge_delay_s")):
                wait = self._delay(edge["barge_delay_s"], 0.6)
            else:
                if not _measured(edge.get("delay_s")):
                    self._flag("unmeasured_delay")
                wait = self._delay(edge.get("delay_s"), 0.6)
                if barged and self.org.get("barge_extra_s"):
                    wait += float(self.org["barge_extra_s"])  # 이 연결의 도중 지연을 재지 못했으면 기관 단위 차이를 더함
            self._pending_enter = True
            self._schedule(wait, "enter", edge["next"], via="key")
            return
        if self._keybuf:
            self._schedule(KEY_WAIT_S, "keybuf")  # 앞자리만으로는 아직 키가 아님: 다음 자리 대기
            return
        inv = s.get("invalid_key") or {}
        if inv.get("next"):
            self._pending_enter = True
            self._schedule(self._delay(inv.get("delay_s"), 0.6), "enter", inv["next"], via="invalid")
            return
        self._end(f"녹음에 없는 키 '{ch}'")

    # 프레임 -------------------------------------------------------------------
    def _timeout_fires(self, target: str) -> bool:
        """시간 초과 연결을 따라가도 되는지 여부. 같은 안내가 녹음에서 본 것보다 많이 반복되면 녹음 밖 종료."""
        if target != self.state_id:
            return True
        after = self.states[self.state_id].get("after_audio") or {}
        limit = int(after.get("max_plays") or DEFAULT_MAX_PLAYS)
        if self._plays + 1 > limit:
            self._end(f"같은 안내가 녹음에서 본 {limit}번보다 많이 반복됨")
            return False
        return True

    def tick(self) -> bytes:
        """다음 20ms 프레임. 예약된 일을 먼저 처리."""
        if not self.ended:
            due = [x for x in self._timers if x[0] <= self.frame]
            if due:
                self._timers = [x for x in self._timers if x[0] > self.frame]
                for _, gen, what, target, extra in sorted(due, key=lambda x: (x[0], x[1])):
                    if self.ended:
                        break
                    via, _, flag = (extra or "|").partition("|")
                    if what in ("enter", "timeout") and gen == self._gen:
                        if what == "timeout":
                            if not self._timeout_fires(target):
                                break
                            self._timeouts += 1
                        if flag:
                            self._flag(flag)
                        self._keybuf = ""
                        self._enter(target, via or what)
                    elif what == "unseen" and gen == self._gen and self._waiting_after:
                        self._end("입력이 없을 때의 동작이 녹음에 없음")
                    elif what == "keybuf" and gen == self._gen and self._keybuf:
                        self._end(f"녹음에 없는 키 '{self._keybuf}'")
        if self._paras and not self.ended:
            # 이 틱에 내보낸 단락은 이 프레임 다음에 세션에 닿음(세션의 소리 시계 = frame+1). 정한 시각 at 에 닿도록 한 프레임 먼저 내보냄
            due = [p for p in self._paras if p["at"] <= self.frame + 1]
            if due:
                self._paras = [p for p in self._paras if p["at"] > self.frame + 1]
                self.transcripts += [(p["text"], p["start"] * FRAME_S + DETECT_LAG_S, p["forced"]) for p in due]
        if self.out and not self.ended:
            frame = self.out.pop(0)
            self.last_src = self.cur_clip
            if not self.out:
                self.frame += 1
                self._audio_done()
                return frame
        else:
            frame = self._noise[self.frame % len(self._noise)]
            self.last_src = None
        self.frame += 1
        return frame


class ReplayCall:
    """서버용 통화: ReplayEngine 을 20ms 마다 한 프레임씩 실시간으로 실행. transcripts=True 면 받아쓰기 모드."""

    def __init__(self, bundle: ReplayBundle, org: str, handlers: CallHandlers, *, jitter: bool = False,
                 log_path: str | None = None, clock: datetime | None = None, transcripts: bool = False,
                 seed: int | None = None, phone: str = "") -> None:
        self.h, self.log_path, self.phone = handlers, log_path, phone
        self.provides_transcripts = transcripts
        self.clock = clock
        self.eng = ReplayEngine(bundle, org, jitter=jitter, clock=clock, seed=seed)
        self.ended = False
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._answer_delay = float(bundle.orgs[org].get("answer_delay_s", 0.5))
        self._t0: float | None = None   # 첫 프레임을 낸 시각 (monotonic). 단락 시작 시각의 기준
        self._dialed = datetime.now(KST)

    @property
    def speaking(self) -> bool:
        """세션의 '말하는 중' 판단에 더해지는 값. 실통화(ClawOpsCall)에는 이 속성이 없고 서버가 소리로 판단하므로, 소리 모드에서는
        항상 거짓. 받아쓰기 모드는 서버가 소리를 듣지 않으므로 단락 소리가 시작돼 받아쓰기가 나올 때까지를 알림
        (실제 서버 경로의 Segmenter.speaking 또는 인식 중에 해당)."""
        return self.provides_transcripts and self.eng.text_pending

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _write_log(self, reason: str) -> None:
        if not self.log_path:
            return
        rec = {"dialed_at": self._dialed.isoformat(timespec="seconds"), "phone": self.phone, "org": self.eng.org_id,
               "clock": self.clock.isoformat(timespec="minutes") if self.clock else None,
               "start": self.eng.start_state, "transcripts": self.provides_transcripts, "end_reason": reason,
               "offmap": self.eng.note, "flags": self.eng.flags, "path": self.eng.path}
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("가상 ARS 기록 실패: %s", e)

    async def answer(self) -> None:
        await asyncio.sleep(self._answer_delay)
        if self.ended:
            return
        await self.h.on_answer()
        self._spawn(self._audio_loop())

    async def _audio_loop(self) -> None:
        t0, i = time.monotonic(), 0
        self._t0 = t0
        while not self.ended:
            async with self._lock:
                frame = self.eng.tick()
            if self.eng.transcripts:
                got, self.eng.transcripts = self.eng.transcripts, []
                if self.provides_transcripts:
                    for text, start, forced in got:
                        if _meaningful(text):
                            # 판단이 오래 걸려도 소리는 계속 나가야 하므로 별도 작업으로 실행 (세션 잠금이 순서 보장)
                            self._spawn(self.h.on_transcript(text, t0 + start, forced=forced))
            if self.eng.ended:
                await self._finish()
                return
            await self.h.on_audio(frame)
            i += 1
            await asyncio.sleep(max(0.0, t0 + i * FRAME_S - time.monotonic()))

    def flush_transcripts(self) -> list[tuple[str, float, bool]]:
        """ARS 가 끊은 뒤 세션이 호출 (session._after_hangup): 끊긴 순간 듣던 단락. 받아쓰기 모드가 아니면 빈 목록."""
        if not self.provides_transcripts or self._t0 is None:
            return []
        return [(text, self._t0 + start, forced) for text, start, forced in self.eng.flush() if _meaningful(text)]

    async def _finish(self) -> None:
        if not self.ended:
            self.ended = True
            self._write_log("remote_hangup")
            await self.h.on_end("remote_hangup")

    async def send_audio(self, ulaw: bytes) -> None:
        pass  # 사용자 마이크 음성 (상담원 연결 중). 녹음 기반 가상 ARS 는 듣지 않음

    async def send_keys(self, keys: str) -> float:
        """실통화 톤 방식처럼 톤을 실시간으로 송출. 자리 i 의 톤이 끝난 때(시작 + 0.25i + 0.15초) ARS 가 받음. 시각은 전송
        시작부터 누적해 계산(자리마다 따로 잠들면 지연 초과분이 쌓이므로). 모든 톤(20ms 프레임 단위로 채운 길이)을 보낸 뒤 반환. 반환값 0."""
        t0 = time.monotonic()
        heard = detect_dtmf(dtmf_ulaw(keys))  # 실제 ARS 처럼 톤을 듣고 키를 해독
        if heard != keys:
            log.warning("DTMF 인식 불일치: 보낸 %d자리, 들은 %d자리", len(keys), len(heard))
        for i, ch in enumerate(heard):
            await asyncio.sleep(max(0.0, t0 + i * KEY_TONE_S + KEY_TONE_ON_S - time.monotonic()))
            async with self._lock:
                self.eng.key(ch)
        await asyncio.sleep(max(0.0, t0 + len(dtmf_ulaw(keys)) / 8000 - time.monotonic()))
        async with self._lock:
            self.eng.keys_done()
        return 0.0

    async def hangup(self) -> None:
        if not self.ended:
            self.ended = True
            self._write_log("local_hangup")
            for t in list(self._tasks):
                if t is not asyncio.current_task():
                    t.cancel()
            await self.h.on_end("local_hangup")


class ReplayBackend:
    """CALL_BACKEND=replay. 전화번호로 묶음 안의 기관을 선택. 묶음에 없는 번호는 거절."""

    def __init__(self, bundle_dir: str, lines: int = 1, jitter: bool = False, log_path: str | None = None,
                 transcripts: bool = False) -> None:
        self.bundle = ReplayBundle(bundle_dir)
        self.capacity = max(1, int(lines))
        # 녹음 밖 종료·미검증 표시를 가려내는 기록은 항상 남김 (서버에는 녹음 밖 종료도 ARS 끊음으로 보임)
        self.jitter, self.log_path, self.transcripts = jitter, log_path or "replay_log.jsonl", transcripts

    def line_name(self, i: int) -> str:
        return f"번호{i + 1}"

    def line_rank(self, i: int) -> int:
        return 0 if 0 <= i < self.capacity else 2

    def usable_count(self) -> int:
        return self.capacity

    async def start(self) -> None:
        detect_dtmf(dtmf_ulaw("1"))  # numpy 를 미리 불러 첫 키가 늦지 않게 함
        log.info("녹음 기반 가상 ARS: %s (기록 %s)", ", ".join(f"{o['org']}({len(o['states'])}상태)" for o in self.bundle.orgs.values()),
                 self.log_path)

    async def stop(self) -> None:
        pass

    async def dial(self, to: str, handlers: CallHandlers, line: int | None = None) -> ReplayCall:
        org = self.bundle.org_for(to)
        if org is None:
            raise RuntimeError("녹음 기반 가상 ARS 에 없는 번호입니다")
        # REPLAY_CLOCK(YYYY-MM-DDTHH:MM): 시간대에 따라 다른 첫 메뉴를 시험할 때 그 시각 기준으로 선택 (비우면 현재 시각)
        raw = os.environ.get("REPLAY_CLOCK", "").strip()
        clock = datetime.fromisoformat(raw).replace(tzinfo=KST) if raw else None
        seed_raw = os.environ.get("REPLAY_SEED", "").strip()
        c = ReplayCall(self.bundle, org, handlers, jitter=self.jitter, log_path=self.log_path, clock=clock,
                       transcripts=self.transcripts, seed=int(seed_raw) if seed_raw else None,
                       phone="".join(ch for ch in to if ch.isdigit()))
        c._spawn(c.answer())
        return c
