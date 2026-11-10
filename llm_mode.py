"""로컬 모델 판단 조합 (combo.interp = llm, 화면 이름 '로컬 모델 (규칙 없음)').

정규식 규칙 대신 로컬 모델이 메뉴 단락을 판단.
이 모듈은 re 미사용. 코드는 모델 답의 형식(키 글자, 원문에 그대로 있는지, 길이)만 확인.

범위
- 모델이 판단하는 것: 단락 분류·선택지·이동 키·상담원 키 뽑기(규칙 추출·키 이름 바꾸기·모델 답 보정 규칙 대신), 선택지 점수,
  탐색기의 상담원 대안 질문·같은 종류 질문·목적의 상담원 요청 판별·상담원 선택지 판별(LlmNavigator), 확인 질문에 글로 한 답 풀이.
  확인 질문은 기본적으로 사용자에게 질문 (LLM_AUTO_CONFIRM=1 이면 목적으로 분명할 때 모델이 선택).
- 세션: 이 해석기(rule_free)로 하는 통화는 통화 기록에 사용자 키를 자릿수로만 기록 (입력 요청을 놓쳤을 때 대비).
- 모델 판단으로 대체하지 않은 것(정규식 포함): 받아쓰기 정리(stt.py), 세션의 되읽기 판별·숫자 가림·입력 오류·업무시간 문장 등 세션 규칙,
  목적 요약·답 합치기·결과 요약·인계 문구·막다른 안내 판단·상담원 대기 확인(로컬 해석기에 위임), 탐색기의 상담원 업무 부분 찾기.
"""
from __future__ import annotations

import json
import logging
import os
import time

from gemini_interpreter import COMMON, CONFIRM, KINDS, SCORE, _block
from interpreter import AGENT_CATEGORY, Option, Prompt
from navigator import Navigator

log = logging.getLogger("ars.llm")

KEY_CHARS = "0123456789*#"

LLM_CLASSIFY = COMMON + """
할 일: ARS 안내 한 단락(<ars>)을 분류하고, 누를 수 있는 키와 그 뜻을 뽑습니다.
kind (하나만):
  menu 업무를 골라 번호를 누르는 메뉴 / confirm 맞는지·진행할지 묻는 확인 질문 / sensitive 주민등록번호·생년월일·계좌·카드·비밀번호·
  인증번호·전화번호 등 자기 번호를 지금 입력하라는 요청 / result 조회 결과·처리 완료를 알리거나, 목적(<goal>)으로 알고 싶어 한 내용을 알려 주는 안내 /
  agent_wait 사람 상담원에게 연결하려고 기다리게 하는 안내 (다른 자동응답으로 넘기는 것은 info) / agent_human 상담원이 직접 말함 /
  input_error 잘못 눌렀거나 입력 시간이 지났다는 안내 / voice_request 말로 대답하라는 요청 / out_of_hours 업무시간이 아니라 이용할 수 없고 고를 업무도 없는 안내 /
  message_sent 문자·알림을 이미 보냈고 서비스를 마친다는 안내 / rejected 본인 확인 실패 등 거절 / info 그 밖의 안내
규칙:
- <ars> 는 음성 인식 원문입니다. 키는 말(한국어 숫자·'한 번' 같은 고유어·영어 숫자·별표·샵·우물정자)로 적혔거나 발음이 비슷한 말로
  잘못 적혔을 수 있습니다. key 는 0-9, *, # 로 적습니다 (두 자리 번호는 그대로). 같은 말이 되풀이되어 적힌 부분은 한 번 말한 것으로,
  전화 안내와 무관한 문장은 없는 것으로 봅니다.
- keys_said: 안내가 누르라고 말한 키(이동 키 포함)를 말한 순서대로 '키글자|원문표기' 로 적습니다 (예: '*|별표'). 키글자는 0-9, *, #
  이고 칸 이름을 쓰지 않습니다. 원문표기는 그 키를 말한 부분을 <ars> 에 적힌 그대로 옮깁니다. 횟수('한 번 더')·기간·전화번호·금액·날짜·
  시각의 숫자는 키가 아닙니다.
- options·이동 키·agent_key 에는 keys_said 에 적은 키만 씁니다. 업무 선택지는 options 에, 이전 메뉴·처음으로·다시 듣기 키는 back_key·home_key·
  repeat_key 에 적습니다 (업무를 고르는 선택지는 이름에 비슷한 낱말이 있어도 options). 상담원 연결 선택지는 options 에 넣고 agent_key 에도
  적으며 category 를 '상담원 연결' 로 씁니다.
- label 은 그 키를 누르면 되는 일의 이름을 안내 문장에 근거해 짧게 씁니다. '~을 원하시면' 같은 조건·조사 꼬리와 번호는 넣지 않습니다.
  확인 질문의 선택지 label 은 그 키가 뜻하는 대답입니다. category 는 기관마다 다른 표현을 묶은 짧은 업무 이름입니다.
- 이름은 들렸지만 번호가 들리지 않은 선택지는 unnumbered 에 적습니다.
- 업무 선택지가 있고 그중에서 고르라고 하면, 안내 앞부분에 결과·발송·공지·상담 가능 시간 같은 말이 있어도 menu 입니다.
- 한 키는 한 가지 뜻입니다. 다시 듣기 키를 이전 메뉴·처음으로 키에 함께 적지 않습니다.
  업무 선택지 없이 이동 키만 남은 안내가 목적의 답을 알려 주면 result 입니다.
- input_request: 지금 사용자가 자기 번호를 직접 눌러 입력하라고 요구하면 true (메뉴 번호를 고르라는 말, 이미 입력한 값이 맞는지 묻는 말은 false).
  그때 input_label 에 입력 항목 이름, terminator 에 입력을 끝내는 키(* 또는 #, 없으면 null).
- critical: 결제·송금·이체·계약 체결·가입·해지의 최종 승인 단계일 때만 true."""

GOAL_AGENT = COMMON + """
할 일: 사용자 목적(<goal>)이 사람(상담원·직원)과 통화하기를 원하는지 답합니다. 원하면 wants_agent=true."""
GOAL_AGENT_SCHEMA = {"type": "object", "properties": {"wants_agent": {"type": "boolean"}}, "required": ["wants_agent"]}

CONFIRM_ANSWER = COMMON + """
할 일: 사용자가 ARS 확인 질문(<ars>)에 글로 답했습니다(<answer>). 그 답이 뜻하는 선택지(<options>)의 key 를 고릅니다.
- 답이 어느 선택지를 뜻하는지 분명하지 않으면 key=null (사용자에게 다시 묻습니다)."""

LLM_GOAL_RULES = """
목적(<goal>)이 있으므로 아래도 적습니다 (kind 가 menu·confirm 이 아니면 scores 는 빈 목록):
- scores: 목적이 options 의 각 선택지와 맞는 정도 0~1. 0.8 이상 직접 일치 / 0.5~0.8 목적으로 가는 상위 메뉴 / 0.3~0.5 관련은 있으나
  불확실 / 0.3 미만 무관. <path> 는 지금까지 고른 메뉴입니다. 목적이 여러 선택지에 똑같이 해당하면 같은 값을 줍니다. 상담원 연결은 목적이
  상담원 연결이거나 다른 길이 없어 보일 때만 높게 줍니다. 모든 options 키에 값을 줍니다.
- wants_agent: 목적이 사람(상담원·직원)과 통화하기를 원하면 true.
- same_kind: 점수가 가장 높은 선택지와 같은 대상의 다른 종류를 고르는 선택지들인데 목적이 어느 종류인지 말하지 않았으면 그 키들 (없으면 빈 목록).
- agent_or_self: 업무 선택지 하나와 상담원 연결만 있을 때, 그 업무 선택지가 목적을 직접 처리하지 않고 목적을 이루는 한 가지 방법(다른
  경로로 안내를 받는 등)일 뿐이며 목적이 그 방법을 말하지 않았으면 true. 사용자가 말하지 않은 처리 방식을 대신 고르지 않습니다."""

_KEY_ITEM = {"type": "string"}
_SN = {"type": ["string", "null"]}


def classify_schema(with_goal: bool) -> dict:
    props = {
        "kind": {"type": "string", "enum": KINDS},
        "keys_said": {"type": "array", "items": _KEY_ITEM},
        "options": {"type": "array", "items": {"type": "object", "properties": {
            "key": _KEY_ITEM, "label": {"type": "string"}, "category": {"type": "string"}}, "required": ["key", "label", "category"]}},
        "unnumbered": {"type": "array", "items": {"type": "string"}},
        "back_key": _SN, "home_key": _SN, "repeat_key": _SN, "agent_key": _SN,
        "input_request": {"type": "boolean"}, "input_label": _SN, "terminator": _SN, "critical": {"type": "boolean"},
    }
    req = list(props)
    if with_goal:
        props.update({
            "scores": {"type": "array", "items": {"type": "object", "properties": {
                "key": _KEY_ITEM, "confidence": {"type": "number"}}, "required": ["key", "confidence"]}},
            "wants_agent": {"type": "boolean"},
            "same_kind": {"type": "array", "items": _KEY_ITEM},
            "agent_or_self": {"type": "boolean"},
        })
        req += ["scores", "wants_agent", "same_kind", "agent_or_self"]
    return {"type": "object", "properties": props, "required": req}


def _enum_keys(keys) -> dict:
    return {"type": "string", "enum": list(keys)}


def score_schema(keys) -> dict:
    return {"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "object", "properties": {
        "key": _enum_keys(keys), "confidence": {"type": "number"}}, "required": ["key", "confidence"]}}}, "required": ["scores"]}


def confirm_schema(keys) -> dict:
    return {"type": "object", "properties": {"key": {"type": ["string", "null"], "enum": list(keys) + [None]}}, "required": ["key"]}


# 형식 확인 (정규식 없음) --------------------------------------------------------------------------------
def ws(text: str) -> str:
    return " ".join((text or "").split())


def valid_key(k) -> bool:
    return isinstance(k, str) and 1 <= len(k) <= 3 and all(c in KEY_CHARS for c in k)


def norm_label(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def parse_keys_said(items, text: str) -> set[str]:
    """{key, said}(또는 '키|원문표기') 가운데 키 형식이 맞고 원문표기가 안내 문장에 그대로 있는 키 (지어낸 키 입력 방지)."""
    out = set()
    for item in items or []:
        if isinstance(item, dict):
            k, said = item.get("key"), item.get("said")
        elif isinstance(item, str):
            k, sep, said = item.partition("|")
            if not sep:
                continue
        else:
            continue
        k, said = str(k or "").strip(), str(said or "").strip()
        if valid_key(k) and said and said in text:
            out.add(k)
    return out


def _clip(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else min(1.0, max(0.0, v))


def to_prompt(d: dict, text: str) -> tuple[Prompt, dict, dict[str, float] | None]:
    """모델 답 → (Prompt, 탐색 힌트, 점수). 형식이 맞고 원문에 근거가 있는 키만 사용."""
    kind = d.get("kind") if d.get("kind") in KINDS else "info"
    said = parse_keys_said(d.get("keys_said"), text)
    nav = {}
    for f in ("back_key", "home_key", "repeat_key"):
        k = d.get(f)
        nav[f] = k if valid_key(k) and k in said else None
    for f in ("back_key", "home_key"):
        if nav[f] and nav[f] == nav["repeat_key"]:
            nav[f] = None                              # 한 키에 두 뜻: 누르면 되돌아가는 쪽을 빼고 다시 듣기로만 취급
    nav_keys = {k for k in nav.values() if k}
    agent_key = d.get("agent_key") if valid_key(d.get("agent_key")) and d.get("agent_key") in said else None
    opts: list[Option] = []
    unnumbered = [str(x).strip() for x in (d.get("unnumbered") or []) if isinstance(x, str) and str(x).strip()]
    by_key: dict[str, Option] = {}
    dup: set[str] = set()
    dropped: set[str] = set()                          # 원문 근거가 없거나 이름이 엇갈려 뺀 키 (점수 확인용)
    for o in d.get("options") or []:
        if not isinstance(o, dict):
            continue
        k, label = o.get("key"), str(o.get("label") or "").strip()
        cat = str(o.get("category") or "").strip() or None
        if not valid_key(k) or k not in said or not label:
            if label:
                unnumbered.append(label)
            if valid_key(k):
                dropped.add(k)
            continue
        if k in nav_keys and k != agent_key:
            continue                                   # 이동 키는 업무 선택지가 아님
        if k in by_key:
            if norm_label(by_key[k].label) != norm_label(label):
                dup.add(k)                             # 같은 번호에 다른 이름 둘: 받아쓰기 오류로 보고 둘 다 제외
            continue
        by_key[k] = Option(k, label[:60], cat)
    for k in dup:
        unnumbered.append(by_key.pop(k).label)
        dropped.add(k)
    opts = list(by_key.values())
    if agent_key:
        hit = by_key.get(agent_key)
        if hit is not None:
            hit.category = AGENT_CATEGORY
        elif kind == "menu" and opts:
            opts.append(Option(agent_key, "상담원 연결", AGENT_CATEGORY))
    input_request = d.get("input_request") is True
    tasks = sum(o.category != AGENT_CATEGORY for o in opts)
    if input_request and (kind == "info" or (kind == "menu" and tasks < 2)):
        # 자기 번호 입력 요청은 상담원 선택지 하나쯤 함께 있어도 입력 요청. 고를 업무가 둘 이상인 메뉴는 메뉴로 유지
        # (모델이 input_request 를 잘못 켜도 메뉴 선택지가 사라지지 않게 함)
        kind = "sensitive"
    elif kind in ("info", "out_of_hours") and sum(o.category != AGENT_CATEGORY for o in opts) >= 2:
        kind = "menu"                                  # 고를 업무가 둘 이상 남은 안내는 메뉴 (gemini_interpreter 의 MENU_KEEP_KINDS 와 같은 기준)
    if kind in ("menu", "confirm") and not opts:
        kind = "info"
    if kind == "sensitive":
        opts = []
    term = d.get("terminator") if d.get("terminator") in ("#", "*") else None
    label = str(d.get("input_label") or "").strip()[:30] or None
    p = Prompt(kind=kind, text=text, options=opts, back_key=nav["back_key"], home_key=nav["home_key"], repeat_key=nav["repeat_key"],
               agent_key=agent_key if any(o.key == agent_key for o in opts) else None,
               label=label if kind == "sensitive" else None, terminator=term if kind == "sensitive" else None,
               critical=d.get("critical") is True)
    keys = {o.key for o in opts}
    scores = None
    if "scores" in d and kind in ("menu", "confirm"):
        got = {}
        for s in d.get("scores") or []:
            if isinstance(s, dict) and s.get("key") in keys:
                got[s["key"]] = _clip(s.get("confidence"))
        scores = {k: got.get(k, 0.0) for k in keys} if got else None   # 점수가 비면 따로 질문 (종류를 바꾼 단락 등)
        if scores:
            lost = [_clip(s.get("confidence")) for s in d.get("scores") or []
                    if isinstance(s, dict) and s.get("key") in dropped - keys]
            if lost and max(lost) >= max(max(scores.values()), 0.3):   # 뺀 선택지가 낮은 점수끼리 같았을 뿐이면 그대로 (맞는 갈래 없음)
                # 근거가 없어 뺀 선택지가 1위였으면 남은 선택지를 스스로 누르지 않고 사용자에게 질문. 모두 0.4(gemini_interpreter 의
                # FAIL_SCORE 와 같은 값)로 두어 '확신 없음' 재질문으로 처리 (낮은 점수 그대로면 맞는 선택지가 없다고 보고 되돌아갈 수 있음)
                scores = {k: 0.4 for k in scores}
    hint = {"wants_agent": d.get("wants_agent") is True, "agent_or_self": d.get("agent_or_self") is True,
            "same_kind": [k for k in (d.get("same_kind") or []) if k in keys],
            "top": max(scores, key=scores.get) if scores else None, "unnumbered": list(dict.fromkeys(unnumbered))[:8]}
    return p, hint, scores


class LlmInterpreter:
    """로컬 모델 판단 해석기. 메뉴 판단은 이 클래스가 정규식 없이 수행, 그 밖의 일(목적 요약·답 합치기·결과 요약·인계 문구·
    막다른 안내·상담원 대기 확인)은 로컬 해석기(base)에 위임."""
    rule_free = True
    fix_mode = "0"
    fix_transcript = False

    def __init__(self, client, base, cache: dict | None = None) -> None:
        self.client = client          # local_llm.OpenAICompatClient
        self.base = base              # 같은 연결을 쓰는 로컬 해석기 (위임 대상)
        self.goal: str | None = None
        self.last_menu: Prompt | None = None
        self.last_path: list[str] = []
        self._cache = cache if cache is not None else {}       # (안내, 목적) → 모델 답 (같은 조합 통화끼리 공유)
        self._scores: dict[tuple[str, str], dict[str, tuple[str, float]]] = {}   # (목적, 메뉴) → 키 → (이름, 점수)
        self._hints: dict[str, dict] = {}                       # 메뉴 signature → 탐색 힌트 (지금 목적 기준)
        self._answer: tuple[str, str | None] | None = None      # (사용자 글 답, 그 답을 받은 안내의 signature)
        self._asked: Prompt | None = None                       # 세션이 알려 준, 지금 글 답을 받은 안내 (note_asked)
        self._noted = False                                     # 세션이 note_asked 를 불렀는지 (None 을 알려 준 경우와 구별)
        self._goal_agent: dict[str, bool | None] = {}            # 목적 → 상담원 통화를 원하는지 (목적이 바뀐 뒤 힌트가 없을 때)

    @property
    def decide_timeout_s(self) -> float:
        return getattr(self.base, "decide_timeout_s", 4.0)

    async def _gen(self, system: str, user: str, schema: dict, timeout_s: float | None) -> dict:
        kw = {"timeout_s": timeout_s, "retries": 0}
        try:
            import inspect
            if "max_tokens" in inspect.signature(self.client.generate).parameters:
                kw["max_tokens"] = 1200   # 여유분. 선택지가 많은 메뉴에서 keys_said 가 객체 형식으로 와도 답이 잘리지 않게 함
        except (TypeError, ValueError):
            pass
        return await self.client.generate(system, user, schema, **kw)

    # 해석기 프로토콜 --------------------------------------------------------------------------------------
    async def summarize_goal(self, goal: str) -> str:
        self.goal = (goal or "").strip()
        self._scores.clear()
        self._hints.clear()
        return await self.base.summarize_goal(goal)

    def set_goal(self, goal: str) -> None:
        self.goal = goal
        self.base.goal = goal
        self._scores.clear()
        self._hints.clear()

    async def classify(self, text: str) -> Prompt:
        # 글 답은 그 답을 받은 판단(세션의 같은 답 처리 안)에서만 사용. 새 안내를 들으면 남은 답 폐기
        # (이름이 같은 다음 확인 질문 '예는 1번, 아니요는 2번' 에 앞 답이 쓰이지 않게 함)
        self._answer = self._asked = None
        self._noted = False
        text = ws(text)
        goal = self.goal or ""
        key = (text, goal)
        d = self._cache.get(key)
        t0 = time.monotonic()
        if d is None:
            system = LLM_CLASSIFY + (LLM_GOAL_RULES if goal else "")
            blocks = {"ars": text, **({"goal": goal, "path": self.last_path} if goal else {})}
            user = "\n".join(_block(k, v) for k, v in blocks.items())
            try:
                d = await self._gen(system, user, classify_schema(bool(goal)), self.decide_timeout_s)
            except Exception as e:
                log.warning("로컬 모델 분류 실패 (%s). 일반 안내로 둔다", type(e).__name__)
                return Prompt(kind="info", text=text)
            if not isinstance(d, dict):
                return Prompt(kind="info", text=text)
            if len(self._cache) >= 64:
                self._cache.clear()
            self._cache[key] = d
        p, hint, scores = to_prompt(d, text)
        log.info("로컬 모델 분류 %.0fms kind=%s 선택지=%d 이동키=%s/%s/%s", 1000 * (time.monotonic() - t0), p.kind, len(p.options),
                 p.back_key, p.home_key, p.repeat_key)
        if p.kind == "agent_wait":
            # 상담원 연결 선택지가 있는 메뉴를 상담원 대기로 본 것인지 로컬 해석기의 상담원 대기 확인(모델 질문)에 위임.
            # 메뉴로 바뀌면 점수는 따로 질문 (분류 때는 메뉴가 아니어서 점수가 없음)
            try:
                m = await self.base.agent_wait_menu(text, p)
            except Exception as e:
                log.info("상담원 대기 확인 실패 (%s): 상담원 대기로 둔다", type(e).__name__)
                m = None
            if m is not None:
                p, scores = m, None
                hint = {**hint, "top": None, "same_kind": []}
        if p.kind in ("menu", "confirm"):
            self.last_menu = p
            sig = p.signature()
            self._hints[sig] = hint
            if scores is not None:
                names = {o.key: norm_label(o.label) for o in p.options}
                self._scores.pop((goal, sig), None)
                self._scores[(goal, sig)] = {k: (names[k], v) for k, v in scores.items()}
        p.llm_hint, p.llm_hint_goal = hint, goal
        return p

    def hint_for(self, prompt: Prompt | None) -> dict | None:
        """분류 때 받은 탐색 힌트. 목적이 바뀐 뒤(재질문 답 등)에는 옛 목적의 힌트를 쓰지 않음."""
        if prompt is None:
            return None
        if getattr(prompt, "llm_hint_goal", None) == (self.goal or ""):
            return getattr(prompt, "llm_hint", None)
        return self._hints.get(prompt.signature())

    async def score(self, goal: str, options: list[Option], path: list[str] | None = None) -> dict[str, float]:
        if not options:
            return {}
        keys = [o.key for o in options]
        self.last_path = list(path or [])
        for (g, _sig), sc in reversed(list(self._scores.items())):
            # 같은 메뉴(키와 이름이 모두 같음)에서 받은 점수만 사용. 키만 겹치는 다른 메뉴의 점수는 쓰지 않음
            if g == (goal or "") and all(k in sc and sc[k][0] == norm_label(o.label) for k, o in zip(keys, options)):
                return {k: sc[k][1] for k in keys}
        listed = [{"key": o.key, "label": o.label, "category": o.category} for o in options]
        user = "\n".join(_block(k, v) for k, v in (("goal", goal or ""), ("path", self.last_path), ("options", listed)))
        try:
            d = await self._gen(SCORE, user, score_schema(keys), self.decide_timeout_s)
            got = {s["key"]: _clip(s.get("confidence")) for s in (d.get("scores") or []) if isinstance(s, dict) and s.get("key") in keys}
        except Exception as e:
            log.warning("로컬 모델 점수 실패 (%s). 불확실로 둔다", type(e).__name__)
            return {k: 0.4 for k in keys}
        return {k: got.get(k, 0.0) for k in keys}

    async def answer_confirm(self, goal: str, prompt: Prompt) -> str | None:
        """확인 질문의 키. None 이면 사용자에게 질문.
        - 결제·해지 등 최종 승인(critical)이면 늘 None.
        - 사용자가 바로 이 질문에 글로 답했으면('네'·'아니요'·'다시 입력할게요') 그 답이 뜻하는 키를 모델이 선택.
          다른 질문에 했던 답은 쓰지 않음 (메뉴 질문에 쓴 '1번' 이 뒤의 되읽기 확인 '맞으면 1번' 에 쓰이지 않게 함).
        - 그 밖에는 사용자에게 질문. LLM_AUTO_CONFIRM=1 일 때만 목적으로 분명한 확인 질문에 모델이 고른 키를 누름
          (규칙 안전망이 없어 모델이 critical 을 놓치면 승인 키를 스스로 누를 수 있으므로 기본은 끔)."""
        answer, self._answer = self._answer, None
        if getattr(self.base, "last_answer", None) is not None:
            self.base.last_answer = None
        if prompt.critical or not prompt.options:
            return None
        keys = [o.key for o in prompt.options]
        listed = [{"key": o.key, "label": o.label} for o in prompt.options]
        if answer and answer[0] and answer[1] == prompt.signature():
            user = "\n".join(_block(k, v) for k, v in (("ars", prompt.text), ("options", listed), ("answer", answer[0])))
            try:
                d = await self._gen(CONFIRM_ANSWER, user, confirm_schema(keys), self.decide_timeout_s)
            except Exception as e:
                log.warning("로컬 모델 확인 답 해석 실패 (%s). 사용자에게 묻는다", type(e).__name__)
                return None
            k = d.get("key") if isinstance(d, dict) else None
            return k if k in keys else None
        if os.environ.get("LLM_AUTO_CONFIRM", "0").strip() != "1":
            return None
        user = "\n".join(_block(k, v) for k, v in (("goal", goal or ""), ("ars", prompt.text), ("options", listed)))
        try:
            d = await self._gen(CONFIRM, user, confirm_schema(keys), self.decide_timeout_s)
        except Exception as e:
            log.warning("로컬 모델 확인 질문 실패 (%s). 사용자에게 묻는다", type(e).__name__)
            return None
        k = d.get("key") if isinstance(d, dict) else None
        return k if k in keys else None

    def note_asked(self, prompt: Prompt | None, question: str | None = None) -> None:
        """세션이 글 답을 받기 직전에 호출: 그 답을 받은 안내 (세션이 되읽기·입력 오류 처리로 새로 만든 확인 질문 포함, 모르면 None)."""
        self._asked, self._noted = prompt, True

    async def goal_wants_agent(self, goal: str) -> bool:
        """목적이 사람(상담원·직원)과 통화하기를 원하는지. 목적이 바뀐 뒤 같은 안내를 다시 판단할 때(분류 때 받은 힌트가 옛 목적의
        것일 때) 탐색기가 질문. 실패하면 False."""
        g = (goal or "").strip()
        if g not in self._goal_agent:
            try:
                d = await self._gen(GOAL_AGENT, _block("goal", g), GOAL_AGENT_SCHEMA, self.decide_timeout_s)
                self._goal_agent[g] = d.get("wants_agent") is True if isinstance(d, dict) else None
            except Exception as e:
                log.warning("로컬 모델 상담원 요청 판단 실패 (%s). 상담원 요청 없음으로 둔다", type(e).__name__)
                return False
        return bool(self._goal_agent[g])

    # 맡기는 일 (로컬 해석기) --------------------------------------------------------------------------------
    async def merge_answer(self, goal: str, question: str, answer: str) -> str:
        # 답을 받은 안내를 함께 기억. answer_confirm 은 바로 그 안내에만 사용.
        # 세션이 note_asked 로 알려 주었으면 그 안내만 확인 (None 이면 막다른 안내·음성 응답 요구처럼 지금 안내가 아닌 질문의 답:
        # 앞서 분류한 확인 질문은 쓰지 않음). 알려 주지 않았으면(세션 밖 호출) 가장 최근 메뉴·확인 질문
        if self._noted:
            asked = self._asked
            menu = asked if asked is not None else (self.last_menu if self.last_menu is not None and self.last_menu.kind == "menu" else None)
        else:
            asked = menu = self.last_menu
        self._asked, self._noted = None, False
        self._answer = ((answer or "").strip(), asked.signature() if asked is not None else None)
        if asked is not None and asked.kind == "confirm":
            # 확인 질문의 답('네'·'1번')은 목적을 바꾸지 않음. 키는 answer_confirm 이 이 답으로 선택
            # (로컬 해석기의 번호 풀이에 넘기면 목적이 선택지 이름 '맞음' 이나 앞 메뉴의 1번 이름으로 바뀜)
            return goal
        # 로컬 해석기에 그 메뉴·경로 전달 ('2번'·'그거' 풀이용. 확인 질문은 넘기지 않음)
        self.base.last_menu, self.base.last_path = menu, list(self.last_path)
        new = await self.base.merge_answer(goal, question, answer)
        self.goal = new
        self._scores.clear()
        self._hints.clear()
        return new

    async def summarize_result(self, goal: str, captions: list[str]) -> dict:
        return await self.base.summarize_result(goal, captions)

    async def handoff_script(self, goal: str, path_texts: list[str]) -> str:
        return await self.base.handoff_script(goal, path_texts)

    async def answers_goal(self, goal, text, chosen=None, siblings=None):
        return await self.base.answers_goal(goal, text, chosen, siblings)

    async def agent_wait_menu(self, text: str, prompt: Prompt | None = None):
        return await self.base.agent_wait_menu(text, prompt)

    async def warmup(self) -> None:
        await self.base.warmup()


class LlmNavigator(Navigator):
    """탐색기의 규칙 판단(상담원 대안 질문·같은 종류 질문·목적의 상담원 요청·상담원 선택지 판별)을 분류 때 받은 모델 판단으로 대체.
    점수 기준(0.50·0.10·0.30)과 되돌아가기·끝내기 흐름은 Navigator 와 동일."""

    @classmethod
    def like(cls, base: Navigator, interp) -> "LlmNavigator":
        return cls(interp, base.threshold, base.margin, base.max_revisit, base.max_presses, base.low)

    async def decide(self, st, current, history, arrived: bool = True, revised: bool = False, user_branch=None):
        self._cur = current
        self._goal_agent = None
        if self.interp.hint_for(current) is None and (st.goal or "").strip():
            # 분류 때 받은 힌트가 없음 (목적이 바뀐 뒤 같은 안내를 다시 판단, 세션이 만든 확인 질문 등).
            # 상담원 요청 여부만 지금 목적으로 모델에 다시 질문. 같은 종류·상담원 대안 질문은 하지 않음 (사용자가 이미 답했음)
            self._goal_agent = await self.interp.goal_wants_agent(st.goal)
        return await super().decide(st, current, history, arrived, revised=revised, user_branch=user_branch)

    def _hint(self) -> dict:
        return self.interp.hint_for(getattr(self, "_cur", None)) or {}

    def _agent_wanted(self) -> bool:
        h = self.interp.hint_for(getattr(self, "_cur", None))
        return bool(h.get("wants_agent")) if h is not None else bool(getattr(self, "_goal_agent", None))

    def _is_agent(self, o: Option) -> bool:
        return o.category == AGENT_CATEGORY

    def _wants_agent(self, goal: str) -> bool:
        return self._agent_wanted()

    def _goal_names_agent(self, goal: str) -> bool:
        return self._agent_wanted()

    def _twins(self, goal: str, best: Option, rest: list[Option]) -> list[Option]:
        h = self._hint()
        if h.get("top") != best.key:
            return []
        keys = set(h.get("same_kind") or []) - {best.key}
        return [o for o in rest if o.key in keys]

    def _agent_alt(self, goal: str, best: Option, options: list[Option]) -> Option | None:
        if len(options) != 2 or self._is_agent(best) or not self._hint().get("agent_or_self"):
            return None
        return next((o for o in options if o is not best and self._is_agent(o)), None)
