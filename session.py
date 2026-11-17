"""
통화 Session (상태 머신)

통화 1건의 상태·자막·경로·재질문·입력 요청 관리와 이벤트 전송.
전화 제어(키 전송·종료)는 이 모듈에서만 수행. 해석기(AI)는 판단 결과만 반환.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from audio import FRAME, VALID_KEYS, Segmenter, rms
from config import Settings
from gemini_interpreter import _NAV_PHRASE, _PRESS_END, _TERMINATED_INPUT, rule_classify, rule_keys, rule_menu
from jev_interpreter import extract_prompt, rule_critical
from interpreter import AGENT_LABEL_RE, Interpreter, Option, Prompt, is_agent_option
from navigator import (AGENT_UNASKED_ASK, AGENT_UNASKED_ONE_ASK, NO_AGENT_ASK, _GENERIC, Layer, NavState, Navigator, _chunks,
                       agent_goal, agent_topic, better_sibling, chosen_names_goal, wants_agent)
from recording import CallRecorder, folder_name, save_files
from som_mode import MenuRecoveryBudgets, display_fixes_for, is_som, som_navigator_like
from stt import Recognizer
from telephony import Backend, Call, CallHandlers

log = logging.getLogger("ars.session")
KST = timezone(timedelta(hours=9))
# 키 톤 송출 뒤 반향을 추가로 무시하는 시간(초). Segmenter.mute 가 이미 0.1초를 더하므로 0 이면 톤 종료 뒤 0.1초만 차단
TONE_ECHO_TAIL_S = 0.0
# ARS 발화 도중 키 전송 시 진행 중이던 안내는 폐기(Segmenter.reset), 이후 소리는 새 단락으로 처리. 키 전송 뒤 이 시간(초) 안에
# 시작된 단락은 잘린 안내의 뒷부분일 가능성 있음 (예: '입력시간이 초과되었습니다' 도중 키 전송 시 뒷부분만 '추가되었습니다'로
# 인식되어 결과 안내로 오판). 이런 조각은 결과·종료 안내로 취급하지 않음
CUT_TAIL_S = 1.0
# 잘린 뒷부분으로 보는 단락의 최대 글자 수(한글·숫자·영문). 잘린 꼬리는 짧은 조각('추가되었습니다')임. 사용자가 ARS 재안내 도중에 답하면
# 키 뒤 남은 소리 때문에 다음 안내 전체가 키 직후 시작한 단락으로 잡히는데, 그것까지 결과·종료로 보지 않으면 실제 결과를 놓침
# (2026-11-09·10 실통화: 결과 안내 2건을 버리고, 문자 발송 종료 2건을 failed 로 기록)
CUT_TAIL_MAX_CHARS = 24
# 이동 키만 남은 안내(막다른 안내 후보) 수신 뒤 판단 전 대기 시간(초). ARS 가 잠시 쉰 뒤 메뉴·결과를 이어 말하면 판단 생략
DEAD_END_WAIT_S = 1.5
# 상담원을 찾으러 업무 메뉴로 들어간 뒤 결과 안내를 들으면, 뒤이은 메뉴(상담원 연결 선택지)를 기다리는 시간(초). 새 안내가 없으면 사용자에게 질문
AGENT_SEEK_RESULT_WAIT_S = 4.0
AGENT_SEEK_RESULT_ASK = ("말씀하신 업무의 안내는 나왔지만 상담원 연결 선택지는 찾지 못했습니다. 안내로 충분하면 통화를 끝내시고, "
                         "상담원이 필요하면 원하시는 것을 말씀해 주세요.")
# 첫 메뉴를 듣기 전의 결과 안내를 확정하기 전에 기다리는 시간(초). 그 사이 메뉴·입력 요청·상담원 대기 등이 오면 결과로 보지 않고,
# 일반 안내가 오면 그 안내 뒤부터 다시 대기
EARLY_RESULT_WAIT_S = 3.0
# 최대 길이에서 잘린 단락(Segmenter.last_forced) 보류 중, 이어지는 소리의 종료 여부 재확인 간격(초)
HELD_HEAD_CHECK_S = 1.0


def now_iso(offset_s: float = 0) -> str:
    return (datetime.now(KST) + timedelta(seconds=offset_s)).isoformat(timespec="seconds")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.reason = reason   # 같은 code 안에서 원인 구분 값 (예: no_free_line). None 이면 응답에 미포함


STATUS_TEXT = {
    "dialing": "전화를 걸고 있습니다.",
    "listening": "안내를 듣고 있습니다.",
    "selecting": "누를 번호를 고르고 있습니다.",
    "awaiting_user": "확인이 필요합니다.",
    "listening_result": "조회 결과를 듣고 있습니다.",
    "agent_done": "업무를 마쳤습니다. 결과를 확인한 뒤 통화를 종료하세요.",
    "waiting_agent": "상담원 연결을 기다리고 있습니다.",
    "connected_to_agent": "상담원이 연결되었습니다. 직접 말씀하세요.",
    "ended": "통화가 종료되었습니다.",
}

END_TEXT = {
    "completed": "업무를 마쳐 통화를 종료했습니다.",
    "agent_call_ended": "상담원과의 통화가 종료되었습니다.",
    "user_input_timeout": "입력이 없어 ARS 가 통화를 종료했습니다.",
    "message_sent": "ARS 가 문자로 안내를 보내고 통화를 종료했습니다.",
    "out_of_hours": "상담 가능 시간이 아니어서 통화를 종료했습니다.",
    "ars_rejected": "ARS 가 요청을 처리하지 못해 통화를 종료했습니다.",
    "loop_detected": "같은 메뉴가 반복되어 통화를 종료했습니다.",
    "not_found": "원하시는 메뉴를 찾지 못해 통화를 종료했습니다.",
    "user_canceled": "통화를 종료했습니다.",
    "no_answer": "상대방이 전화를 받지 않습니다.",
    "busy": "상대방이 통화 중입니다.",
    "failed": "통화가 끊겼습니다.",
    "max_duration": "최대 통화 시간이 지나 통화를 종료했습니다.",
}

# 입력값 되읽기: "입력하신 번호는 123456, 맞으면 1번" 처럼 입력 표현과 확인 표현이 함께 있는 안내
_READBACK_RE = re.compile(r"(입력하신|누르신|입력한).*(맞으|맞습니까|확인|정확)|(맞으|맞습니까|확인|정확).*(입력하신|누르신|입력한)")
_DIGITS_RE = re.compile(r"[0-9][0-9\s,\-]{2,}[0-9]|(?:[공영일이삼사오육칠팔구]\s*){4,}")
_NO_ANSWER_RE = re.compile(r"틀|아니|다시|재입력|취소|정정|수정")  # 되읽기 확인에서 '아니요' 쪽 선택지 이름
_SPOKEN_RUN_RE = re.compile(r"(?:[공영일이삼사오육칠팔구]\s*){6,}")  # 한국어 숫자를 여섯 자 이상 이어 읽음 (금액·시각은 이렇게 읽지 않음)
# 숫자에 바로 붙은 단위: 이런 숫자는 입력값이 아닌 금액·날짜·개수 (mask_digits 의 자릿수 일치 가림에서 제외).
# 띄어 쓴 경우는 단위로 보지 않음 ('1234 원하시면', '1234 분실신고' 의 '원'·'분' 은 다음 낱말의 첫 글자일 가능성)
_QUANTITY_AFTER_RE = re.compile(r"(?:원|만|천|억|조|달러|엔|위안|유로|%|퍼센트|개|건|회|명|년|월|일|시|분|초|점|포인트)")
# "입력하신" 없이 숫자만 읽고 확인을 묻는 되읽기 ("[번호]번입니다. 맞으면 1번, 다시 입력하시려면 2번" 형태).
# 민감정보 입력 직후(되읽기 구간)에 확인 표현과 재입력 표현이 함께 있으면 되읽기로 판정. 전화 음질에서는 숫자 인식이 깨지므로
# 숫자 인식 여부는 조건에서 제외
_CONFIRM_RE = re.compile(r"맞으(?:면|시면)|맞습니까|맞는지|맞나요")
# 한 자씩 입력 중 이 안내가 오면 ARS 가 입력을 끝내고 다음 단계로 넘어간 것으로 보고 요청 종료
_STREAM_END_KINDS = ("result", "message_sent", "out_of_hours", "rejected", "agent_wait", "agent_human")
_REENTER_RE = re.compile(r"다시\s*입력|재입력|틀리(?:면|시면)|틀립니까|아니(?:면|시면)|정정")
_OPTION_KEY_RE = re.compile(r"([가-힣][가-힣 ]{0,14}?(?:으면|시면|려면|면))\s*([0-9*#])\s*번")
_ANY_DIGIT_RE = re.compile(r"[0-9공영일이삼사오육칠팔구]")


def _obj(word: str) -> str:
    """목적격 조사 (을/를)."""
    ch = word.strip()[-1:] or "가"
    return "을" if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 else "를"


def _choice_words(label: str, others: list[str], field: str = "") -> tuple[list[str], str]:
    """고른 선택지를 들었는지 판정할 증거: (구별 낱말, 붙여 쓴 이름 전체). 구별 낱말은 이름의 낱말 조각(두 글자 이상, 일반 낱말 제외) 중
    같은 메뉴의 다른 선택지 이름에 없고(앞자리만 들어가 형제 선택지로 간 안내의 '서식'·'신고서' 를 증거로 쓰지 않도록) 입력 항목
    이름(field) 첫머리가 아닌 곳에 들어 있지 않은 것('휴대전화 번호 변경' 뒤 '주민등록번호를 …' 의 '번호'). 항목 이름은 띄어쓰기를
    지우고 비교 ('사업장 관리 번호'·'사업장관리번호' 가 같은 결과). 이름의 이웃 두 조각을 붙인 말('팩스번호'·'휴대전화번호')도
    다른 선택지 이름에 없으면 증거에 추가 (바른 갈래의 '새로 등록하실 팩스 번호' 를 들은 것으로 판정하도록)."""
    other = {c for o in others for c in _chunks(o or "")}
    other_flat = [re.sub(r"[\s,·]", "", o or "") for o in others]
    fld = re.sub(r"[^가-힣A-Za-z0-9]", "", field or "")
    chunks = [c for c in _chunks(label or "") if c]
    words = [c for c in chunks if len(c) >= 2 and c not in _GENERIC and c not in other
             and not (c in fld and not fld.startswith(c))]
    for a, b in zip(chunks, chunks[1:]):
        pair = a + b
        if len(a) >= 2 and len(b) >= 2 and not any(pair in o for o in other_flat) and pair not in words:
            words.append(pair)
    return words, re.sub(r"[\s,·]", "", label or "")


def _choice_said(label: str, others: list[str], texts: tuple, field: str = "") -> bool:
    """안내(원문·화면용 고친 문장)에 고른 선택지 이름이 나왔는지: 구별 낱말 하나, 또는 붙여 쓴 이름 전체(세 글자 이상) (띄어쓰기 무시)."""
    words, whole = _choice_words(label, others, field)
    for t in texts:
        flat = re.sub(r"[\s,·]", "", t or "")
        if flat and (any(w in flat for w in words) or (len(whole) >= 3 and whole in flat)):
            return True
    return False


def _digit_count(s: str) -> int:
    return len(re.sub(r"[^0-9공영일이삼사오육칠팔구]", "", s))


# 되읽기 문장: 입력 표현과 확인 표현이 함께 있음. '입력하신 후·다음·뒤' 처럼 입력을 시키는 말은 제외
_READBACK_STRICT_RE = re.compile(r"(입력하신|누르신|입력한)(?!\s*(?:후|다음|뒤)).*(맞으|맞습니까|확인|정확)"
                                 r"|(맞으|맞습니까|확인|정확).*(입력하신|누르신|입력한)(?!\s*(?:후|다음|뒤))")
# 번호 일부만 되읽음 ('끝 네 자리는', '뒤 4자리')
_PARTIAL_READBACK_RE = re.compile(r"(?:앞|뒤|끝|마지막)\s*(?:[0-9]+|[한두세네]|다섯|여섯|일곱|여덟|아홉)?\s*자리")
# 메뉴 키 숫자('맞으면 1번', '○○는 2번')와 'N자리' 의 숫자: 번호 자릿수에서 뺌
_MENU_KEY_DIGIT_RE = re.compile(r"(?:으면|시면|려면|면|[은는])\s*,?\s*([0-9])\s*번(?!호)|([0-9]+)\s*자리")
_KO_DIGIT_RUN_RE = re.compile(r"(?:[공영일이삼사오육칠팔구][\s,.·\-]*){4,}")


def _is_readback(text: str) -> bool:
    """입력값 되읽기 확인 문장인지 (입력 표현 + 확인 표현, 또는 확인 표현 + 재입력 표현)."""
    return bool(_READBACK_STRICT_RE.search(text or "") or (_CONFIRM_RE.search(text or "") and _REENTER_RE.search(text or "")))


def _readback_digits(text: str) -> int:
    """되읽기 단락의 번호 자릿수: 아라비아 숫자 + 넷 이상 이어진 한국어 숫자(쉼표·공백 끊김 허용) − 메뉴 키·'N자리' 숫자."""
    t = text or ""
    n = len(re.findall(r"[0-9]", t)) - sum(len(a or b) for a, b in _MENU_KEY_DIGIT_RE.findall(t))
    n += sum(len(re.sub(r"[^공영일이삼사오육칠팔구]", "", m)) for m in _KO_DIGIT_RUN_RE.findall(t))
    return max(0, n)


def _same_prompt(a: Prompt | None, b: Prompt | None) -> bool:
    """같은 안내인지 여부: 같은 객체이거나, ARS 가 다시 읽은 같은 메뉴(종류·선택지가 같음. 선택지가 없으면 원문이 같음)."""
    if a is b:
        return True
    if a is None or b is None or a.kind != b.kind:
        return False
    return a.signature() == b.signature() if (a.options or b.options) else a.text == b.text


def _field_name(label: str) -> str:
    """입력 항목명에서 뒤의 자릿수 말을 뗀 이름 ('주민등록번호 앞 6자리' → '주민등록번호')."""
    return re.sub(r"\s*(?:앞|뒤)?\s*(?:[0-9]+|열\s?(?:[한두세네]|다섯|여섯|일곱|여덟|아홉)?|[한두세네]|다섯|여섯|일곱|여덟|아홉)\s*자리\s*$", "",
                  label or "").strip() or (label or "")


_KO_COUNT = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}


def _field_key(label: str) -> tuple[str, str, int | None]:
    """입력 항목 비교용 열쇠: (이름, 부분 '앞'·'뒤'·'', 자릿수). '앞 여섯 자리' 와 '앞 6자리' 는 같고 '뒤 7자리' 는 다름."""
    t = label or ""
    name = re.sub(r"\s", "", _field_name(t))
    m = re.search(r"(앞|뒤)?\s*([0-9]+|열\s?(?:[한두세네]|다섯|여섯|일곱|여덟|아홉)?|[한두세네]|다섯|여섯|일곱|여덟|아홉)\s*자리\s*$", t)
    if not m:
        return name, "", None
    w = re.sub(r"\s", "", m.group(2))
    count = int(w) if w.isdigit() else (10 + _KO_COUNT.get(w[1:], 0) if w.startswith("열") else _KO_COUNT.get(w))
    return name, m.group(1) or "", count


def _same_field(a: str, b: str) -> bool:
    """같은 입력 항목인지: 이름이 같고, 둘 다 부분·자릿수를 말했으면 그것도 같음 (한쪽만 말하면 같은 항목으로 봄)."""
    ka, kb = _field_key(a), _field_key(b)
    if ka[0] != kb[0]:
        return False
    if (ka[1] or ka[2] is not None) and (kb[1] or kb[2] is not None):
        return ka[1:] == kb[1:]
    return True


# 같은 단락에 통화를 끝내는 말이 있으면 메뉴 재낭독으로 보지 않음
_ENDING_RE = re.compile(r"다시\s*이용해\s*주시기|이용해\s*주셔서\s*감사|통화를\s*종료|종료하겠습니다|끊겠습니다|마칩니다|마치겠습니다")
_OPTION_WORD_RE = re.compile(r"[0-9]\s*번(?!호)")  # 번호 선택지 말 ('1번', '10번')
# 입력 오류 문장 ('입력 시간이 초과되었습니다.', '잘못 누르셨습니다.'): 메뉴 추출 전에 뺌
_ERR_SENT_RE = re.compile(r"[^.?!]*(?:입력\s*시간이\s*(?:초과|경과|지났)|잘못\s*누르셨|입력이\s*되지\s*않았)[^.?!]*[.?!]\s*")


_HOURS_RE = re.compile(r"\d{1,2}\s?시(?!간)")


def hours_sentence(text: str) -> str | None:
    """업무시간 외 안내에서 상담 가능 시간을 말한 문장. '부터'·'까지' 가 함께 있는 시각 문장('평일 오전 9시부터 오후 6시까지')을 먼저,
    없으면 시각이 둘이거나 '부터'·'까지' 하나가 붙은 문장. 모든 ARS 공통 표현만 확인. 없으면 None."""
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text or "") if _HOURS_RE.search(s)]
    pick = next((s for s in sents if "부터" in s and "까지" in s), None) or next(
        (s for s in sents if len(_HOURS_RE.findall(s)) >= 2 or re.search(r"부터|까지", s)), None)
    if pick is None:
        return None
    return pick if len(pick) <= 80 else pick[:80].rstrip() + "…"


def _rule_critical(norm: str) -> bool:
    """결제·해지 등 최종 승인으로 보이는 문장 (jev_interpreter 중요 선택 규칙과 같은 기준)."""
    return rule_critical(norm)


def key_name(key: str) -> str:
    """키를 읽는 말: 3 → '3번', * → '별표', # → '샵'."""
    return {"*": "별표", "#": "샵"}.get(key, f"{key}번")


# 문장 안의 키 말 ('3번', '별표', '샵 버튼', '우물정자', 영어 'press 6')
_KEY_MENTION = re.compile(r"[0-9]\s*번(?!호)|별\s?표|샵|우물\s?정자|[Pp]ress\s*[0-9*#]")


def menu_preface(text: str) -> str:
    """메뉴 단락에서 첫 선택지(키 말이 든 문장) 앞의 안내 문장. '지금은 상담 업무 시간이 종료되어 … 분실 신고는 별표, …' 의 앞부분."""
    out = []
    for s in re.split(r"(?<=[.!?。])\s+", (text or "").strip()):
        if _KEY_MENTION.search(s):
            break
        out.append(s)
    return " ".join(out).strip()


def _same_menu(a: Prompt, b: Prompt, tolerant: bool = False) -> bool:
    """두 메뉴의 동일 여부 (ARS 의 같은 메뉴 재낭독 여부). 키 구성이 같고, 받아쓰기 차이('분실신고 금융사기' / '분실 신고, 금융 사기',
    'For' / 'for')를 무시했을 때 이름이 같은 선택지가 70% 이상이면 같은 메뉴로 판정.
    tolerant: 같은 메뉴를 두 번 읽을 때 추출이 키 하나를 놓칠 가능성 있음(예: 앞 안내가 붙은 단락에서 기호 키 누락). 두 키 집합의 차이가
    키 하나 이하이고 공통 키가 3개 이상이며 공통 키 이름의 70% 이상이 같으면 같은 메뉴로 판정."""
    ka, kb = {o.key: o.label for o in a.options}, {o.key: o.label for o in b.options}
    norm = lambda s: re.sub(r"[^0-9a-z가-힣]", "", (s or "").lower())  # noqa: E731
    if tolerant:
        common = set(ka) & set(kb)
        if len(common) < 3 or len(set(ka) ^ set(kb)) > 1:
            return False
        return sum(norm(ka[k]) == norm(kb[k]) for k in common) >= 0.7 * len(common)
    if not ka or set(ka) != set(kb):
        return False
    return sum(norm(ka[k]) == norm(kb[k]) for k in ka) >= 0.7 * len(ka)


def caption_text(p: Prompt, text: str) -> str:
    """앱에 보내는 자막(caption.text). 앱은 해석 없이 그대로 표시하므로 서버에서 표시 형태로 생성.
    선택지가 있는 안내(메뉴·확인 질문)는 STT 원문 대신 AI 가 뽑은 선택지 목록으로 압축:
    "잔액조회 3번 · 거래내역조회 4번 · 상담직원 연결 0번 · 다시 듣기 별표". 그 밖의 안내(결과·오류·일반 안내)는 원문 그대로.
    결과 요약(summarize_result)이 captions 를 입력으로 쓰므로 결과 안내는 원문 유지 필요."""
    opts = p.all_options() if p.options and p.kind in ("menu", "confirm") else []
    if not opts:
        # 결과·일반 안내에 '다시 들으시려면 1번' 같은 선택지가 붙어 와도 원문 사용 (선택지 목록으로 바꾸면 결과 본문이
        # 결과 요약 입력에서 누락. 받아쓰기 고침 사용 시 결과 단락에 선택지가 붙는 경우 있음)
        return text
    # 받아쓰기 고침을 따로 받은 안내(LLM_FIX_TRANSCRIPT=separate)는 선택지 이름이 원문이므로 화면 표시 시에만 고침 적용
    return " · ".join(f"{fix_words(o.label, p.display_fixes)} {key_name(o.key)}" for o in opts)


def fix_words(s, fixes) -> str:
    """화면 글에 받아쓰기 고침(잘못 적힌 말 → 고친 말) 적용. 문자열이 아니거나 고침이 없으면 그대로 반환.
    고친 말이 잘못 적힌 말을 품는 경우('발급'→'재발급') 이미 고친 말인 자리는 다시 고치지 않음
    (모델이 선택지 이름을 스스로 '카드 재발급' 으로 써 둔 경우의 '카드 재재발급' 방지)."""
    if not isinstance(s, str) or not fixes:
        return s
    for a, b in fixes:
        if not a:
            continue
        s = b.join(part.replace(a, b) for part in s.split(b)) if a in b else s.replace(a, b)
    return s


def mask_digits(text: str, length: int | None = None) -> tuple[str, bool]:
    """ARS 가 입력값을 다시 읽어 주는 문장의 숫자열 가림.
    length 지정 시 사용자가 입력한 자릿수와 같은 숫자열만 가림 (금액·날짜 같은 다른 숫자는 유지).
    이때 단위가 바로 붙은 숫자('2000원', '36만 2000원' 의 '2000', '2026년', '15건')는 자릿수가 같아도 입력값이 아닌 수량으로 보고
    유지 (예: 비밀번호 네 자리 입력 뒤 '출금 가능액은 ○○만 ○○○○원' 의 네 자리 오가림 방지).
    입력값을 확인하는 되읽기 문장은 mask_readback 이 단위와 무관하게 모두 가림."""
    def rep(m: re.Match) -> str:
        n = _digit_count(m.group())
        if length is not None and (n != length or _QUANTITY_AFTER_RE.match(text, m.end())):
            return m.group()
        return "*" * (n or 4) + (" " if m.group().endswith(" ") else "")

    masked, n = _DIGITS_RE.subn(rep, text)
    return masked, masked != text


_CONFIRM_KEY_RE = re.compile(r"[\s,]*[0-9*#]\s*번")  # 확인 표현 바로 뒤의 선택 키 ('맞으면 1번')


def _readback_split(text: str) -> int:
    """되읽기 문장에서 '읽어 준 값' 과 '선택지(맞으면 1번 …)' 의 경계.
    1) 선택 키가 바로 붙은 확인 표현('맞으면 1번') — '…맞는지 확인합니다. [번호]. 맞으면 1번' 처럼 확인 표현이 번호보다
       먼저 나와도 번호는 앞부분에 포함
    2) 확인 표현 목록에 없는 선택지('정확하면 1번', '확인하시려면 1번')의 마지막 낱말 시작
    3) 마지막 확인 표현  4) 없으면 문장 끝"""
    ms = list(_CONFIRM_RE.finditer(text))
    keyed = next((m for m in ms if _CONFIRM_KEY_RE.match(text, m.end())), None)
    if keyed:
        return keyed.start()
    k = _OPTION_KEY_RE.search(text)
    if k:
        return k.start(1) + k.group(1).rfind(" ") + 1
    return ms[-1].start() if ms else len(text)


_NUM_TOKEN_RE = re.compile(r"[0-9]+|(?<![가-힣])[공영일이삼사오육칠팔구]+(?![가-힣])")  # 낱말 속 글자가 아닌 숫자


def mask_readback(text: str) -> str:
    """되읽기 문장에서 읽어 준 값을 형태와 무관하게 가리고 선택 키('맞으면 1번', '정확하면 1번' 의 키)만 유지.
    경계(_readback_split) 뒤에 선택 키가 아닌 숫자가 남아 있으면 경계를 그 숫자 뒤로 이동 (선택지가 번호보다 먼저 나오거나
    되읽기가 한 단락에 두 번 들어간 경우). 경계 앞은 숫자·한국어 숫자 글자를 한 글자씩 모두 가림(인식이 깨진 "1. 영 1. 영 구해 3. 영"
    포함, 낱말 속 글자도 가려질 수 있음). 경계 뒤는 낱말은 두고 네 자 이상 숫자열만 한 번 더 가림."""
    keys = {m.start(2) for m in _OPTION_KEY_RE.finditer(text)}
    i = _readback_split(text)
    for m in _NUM_TOKEN_RE.finditer(text, i):
        if m.start() not in keys:
            i = m.end()
    parts, last = [], 0
    for k in sorted(p for p in keys if p < i) + [i]:  # 선택 키 사이 구간별: 숫자열을 묶어 가린 뒤 남은 숫자 글자도 가림
        seg, _ = mask_digits(text[last:k])
        parts.append(_ANY_DIGIT_RE.sub("*", seg) + (text[k] if k < i else ""))
        last = k + 1
    tail, _ = mask_digits(text[i:])
    return "".join(parts) + tail


_SPACED_DIGITS_RE = re.compile(r"(?<=[0-9])[ \t]+(?=[0-9])")


def join_readback_digits(text: str) -> str:
    """사용자에게 보여 주는 되읽기 원문에서 숫자 묶음 사이 공백 제거 ('010 12 345 678 번' → '01012345678 번').
    STT 가 ARS 의 숫자 낭독 중 쉰 곳에 공백을 넣으므로 묶음 위치가 통화마다 다름. 선택지('맞으면 1번') 앞부분의
    아라비아 숫자 사이 공백만 제거 (한국어 숫자 글자는 낱말 속 글자와 구별할 수 없어 유지, '-' 같은 기호도 유지).
    경계는 가리기(mask_readback)와 동일."""
    i = _readback_split(text)
    return _SPACED_DIGITS_RE.sub("", text[:i]) + text[i:]


def readback_options(text: str) -> list[Option]:
    """되읽기 확인 문장의 선택지 ('맞으면 1번', '다시 입력하시려면 2번'). AI 가 확인 질문으로 분류하지 못했을 때 사용."""
    out: list[Option] = []
    for label, key in _OPTION_KEY_RE.findall(text):
        if all(o.key != key for o in out):
            out.append(Option(key, label.strip()))
    return out


# ---------------------------------------------------------------------------
# 이벤트 전송
# ---------------------------------------------------------------------------
class EventHub:
    def __init__(self) -> None:
        self.queues: set[asyncio.Queue] = set()
        self.history: list[dict] = []  # 시험·디버깅용

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.queues.discard(q)

    def publish(self, event: str, data: dict) -> None:
        msg = {"event": event, "at": now_iso(), "data": data}
        self.history.append(msg)
        for q in list(self.queues):
            q.put_nowait(msg)


@dataclass
class Deps:
    settings: Settings
    backend: Backend
    recognizer: Recognizer
    interp: Interpreter
    navigator: Navigator
    # 통화별 해석기 생성기 (combos.ComboFactory). None 이면 모든 통화가 interp 하나를 공유
    combos: object | None = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, phone: str, deps: Deps, line: int | None = None, combo: dict | None = None,
                 repick: Callable[[], int | None] | None = None) -> None:
        self.id = "ses_" + secrets.token_hex(4)
        self.token = "st_" + secrets.token_urlsafe(24)
        self.phone = phone
        self.d = deps
        self.s = deps.settings
        self.line = line      # 이 통화를 거는 발신 번호 순번 (번호가 여럿일 때만. None 이면 백엔드 기본 번호)
        self.combo = combo    # 이 통화의 판단 조합 {"interp", "fix", "result"} (조합을 고를 수 있는 서버만)
        self._repick = repick  # 발신 직전에 고른 번호를 쓸 수 없을 때 빈 번호를 다시 고르는 함수 (SessionManager)
        self.hub = EventHub()
        self.audio_out: set[asyncio.Queue] = set()   # 앱 오디오 채널 (서버 → 앱)

        self.status = "created"
        self.status_message = ""
        self.agent_active = False
        self.goal: str | None = None
        self.goal_summary: str | None = None

        self.call: Call | None = None
        self.answered_at: float | None = None
        self.ended_at: float | None = None
        self.end_reason: str | None = None
        self.created_mono = time.monotonic()
        self.recorder: CallRecorder | None = None   # 시험용 통화 기록 (record=true 또는 RECORD_ALWAYS=1)
        self.recording_path: str | None = None       # 디스크에 저장한 기록 폴더 (RECORDING_SAVE=1, 통화 종료 뒤)
        self._save_task: asyncio.Task | None = None

        self.captions: list[dict] = []
        # 번호 선택지를 목록으로 줄인 자막(메뉴·확인 질문). 결과 요약 입력에서 제외 (지나온 메뉴의 '1번' 을 결과로 적는 오류 방지)
        self._choice_caps: set[int] = set()
        self.path: list[dict] = []
        self.step = 0
        self.layers: list[Layer] = []        # 키를 눌러 지나온 메뉴 (사용자·AI 공통)
        self.last_press_layer: Layer | None = None  # 마지막 키가 쌓은 계층 (그 키가 거부되면 되돌림)
        # 두 자리 이상 메뉴 키를 누른 뒤 그 선택지 이름을 아직 듣지 못함: (키, 이름, 같은 메뉴 다른 선택지 이름들, 메뉴).
        # 바로 입력 안내가 오면 입력 요청에 경고 (_press, _paragraph, _ask_secure)
        self.unconfirmed_multi: tuple[str, str, list[str], Prompt] | None = None
        self.secure_warned: dict | None = None  # 경고를 붙인 입력 요청 (같은 요청 재알림 때 이름을 들었으면 경고 제거)
        self.rejected_layer: Layer | None = None    # 거부로 보인 키의 계층: 같은 메뉴를 다시 들으면 되돌림, 다른 메뉴면 유지
        self.redo_by_user = False                   # 거부로 보인 키를 사용자가 눌렀는지
        self.redo_menu: Prompt | None = None        # 거부로 보인 키를 누른 메뉴
        self.readback_retry_armed = False           # 민감정보를 보낸 뒤 되읽기 다시 인식을 쓸 수 있는 구간
        self.secure_typed_req: str | None = None    # 한 자씩 입력(secure_key) 중인 요청
        self.secure_first_sent_at: float | None = None  # 그 요청의 첫 키가 ARS 로 나간 시각 (이 뒤에 시작된 안내는 보낸 키에 대한 반응일 수 있음)
        self.secure_req_pub_at = 0.0                # 한 자씩 입력 중 요청을 다시 알린 시각 (기한 연장을 화면에 알림)
        self.secure_buf = ""                        # ARS 가 말하는 동안 모아 둔 한 자씩 입력 (보내면 바로 비움, 기록하지 않음)
        self.current: Prompt | None = None   # 가장 최근에 들은 메뉴·확인 질문
        self.current_fresh = False           # 마지막 키 전송 이후에 들은 안내 여부
        self.since_key_texts: list[str] = []  # 마지막 키 전송 뒤 들은 일반 안내 문장 (막다른 안내 판단용)
        self.para_seq = 0                     # 처리한 단락 수 (막다른 안내 판단 대기 중 새 안내 수신 여부 확인용)
        self.dead_end_rid: str | None = None  # 막다른 안내 재질문의 request_id
        self.dead_end_prompt: Prompt | None = None  # 그 막다른 안내 (사용자가 고른 이동 키를 이 안내 기준으로 처리)
        self.last_key_actor = "ai"
        self.nav: NavState | None = None
        self.key_queue: list[str] = []       # 이전 계층 복귀 시 차례로 누를 키 (처음으로 + 지나온 키)
        self.redo_same_menu = False          # 입력 오류 뒤 같은 메뉴 재판단 여부
        self.after_backtrack = False         # 이전 계층 복귀 중 여부
        self.input_errors = 0
        self.private_steps: set[int] = set()  # 통화 기록에 값을 남기지 않는 경로 단계 (규칙 안전망이 없는 해석기의 키패드 키)
        self.private_run = ""                 # 이어서 누른 그 키패드 번호 (기록에서 지울 값 생성, _press)
        # 거부를 센 뒤 일반 안내를 들음: 그 뒤 입력 오류는 키를 누른 메뉴를 다시 읽을 때만 횟수 누적 (다음 단계로 이어지지 않도록)
        self.info_since_reject = False
        self.answer_revised = False          # 재질문에 글로 답해 목적이 바뀐 뒤의 재판단 (navigator 의 사용자 갈래 규칙)

        self.pending: dict | None = None
        # 재질문 id → AI(merge_answer)에 줄 질문. 되읽기 확인은 사용자에게 숫자가 든 원문, AI 에는 가린 문장 제공
        # (요청 dict 는 앱으로 그대로 전송되므로 별도 보관)
        self.ai_questions: dict[str, str] = {}
        self.req_seq = 0
        self.secure_window = False
        self.ai_hold_until = 0.0
        self.summary: dict | None = None
        self.result_heard = False            # 결과 안내를 들음 (요약을 마치기 전에 끊겨도 completed)
        self.agent_goal_at_caption = 0       # AI 시작 시점의 자막 위치 (결과 요약 범위)
        self.agent_request = False           # 상담원 연결 버튼으로 건 통화 (create agent=true) 또는 통화 중 상담원 연결 요청 완료
        self.routing_mode = "task"           # task: 업무 목적으로 탐색, human_agent: 상담원 선택지 우선 (업무는 task_goal)
        self.task_goal = ""                  # human_agent 탐색의 업무 목적 (상담원 말을 붙이기 전). 스냅샷 goal 로 표시
        self.handoff_prev: dict | None = None  # 업무 목적 입력 요청을 열 때 닫은 재질문 (입력 취소 시 복원)
        self.hours_note: str | None = None   # 업무시간 외 안내에서 뽑은 상담 가능 시간 문장 (종료 문구에 덧붙임)
        self.hangup_seen = False             # 상대가 끊음. 남은 안내를 처리한 뒤 종료한다
        self.heard_since_press = False       # 마지막 키 전송 뒤 ARS 가 소리를 냈는지
        self.key_outstanding = False         # 키를 보냈고 아직 ARS 의 응답 안내를 듣지 못함 (입력 오류 안내의 귀속 판단)
        self.last_key_at: float | None = None  # 마지막 키 전송 시각. 그 전에 시작된 안내는 키 이전 상태의 안내
        self.early_result_wait = False       # 첫 메뉴 전 결과 안내를 확정하려고 기다리는 중 (_early_result_check)
        self.result_upto: int | None = None  # 보류한 결과 단락까지의 자막 수. 확정 때 요약·원문 대체를 여기까지로 (뒤에 온 일반 안내 제외)
        self.early_end_kind: str | None = None  # 첫 메뉴 전 보류 중인 끝맺음 종료 사유 (해석기 hold_early_end). None 이면 결과 보류
        self._dead_end_pending: tuple | None = None  # 판단 대기 중인 막다른 안내 후보 (안내, 그때의 last_key_at). 해석기 dead_end_on_timeout
        self.key_cut_speech = False          # 마지막 키를 ARS 발화 도중에 보냈는지 여부 (그 안내의 뒷부분이 새 단락으로 들림)
        self.seg_started_at: float | None = None  # 지금 듣고 있는 단락의 소리 시작 시각
        self.last_send_secure = False        # 마지막으로 보낸 키가 민감정보 입력값인지 여부
        self.secure_queued = False           # 민감정보 입력값을 받아 ARS 안내 종료를 기다리는 중 (값은 전송 작업 안에만 존재)
        self.secure_abort = False            # 대기 중 ARS 가 메뉴·결과 등 다른 안내로 넘어감 → 입력값 전송 취소
        self.secure_len: int | None = None   # 마지막 민감정보 입력의 자릿수 (값은 저장하지 않음)
        self.secure_label: str = "입력 번호"   # 마지막 민감정보 요청의 항목명 (되읽기 확인 단계의 경로 표시용)
        self.secure_req: tuple[str | None, str | None] = (None, None)  # 마지막 민감정보 요청의 (항목명, 종료 키). 입력 오류 뒤 재요청 시 사용
        self.secure_sent_at = 0.0
        self.secure_errors = 0               # 민감정보 입력 뒤 연속 입력 오류 횟수
        self.queued_key: str | None = None   # 이전 계층 복귀 중 마지막으로 누른 대기열 키 (입력 오류 시 재전송)
        self.queued_err = False              # 그 대기 키 뒤 키 반응이 아닌 입력 오류 단락을 들음 (같은 메뉴가 다시 오면 다시 누름)
        self.closed = False                  # ended 이벤트 전송 완료. 이후 이벤트 전송 없음

        self.segmenter = Segmenter(self.s.vad_rms, self.s.paragraph_silence_ms)
        self.segmenter.min_chunk_ms = int(getattr(self.s, "stt_prefetch_min_s", 5.0) * 1000)  # 미리 인식 조각 최소 길이
        self.stt_q: asyncio.Queue = asyncio.Queue()
        self.stt_inflight = 0                # 인식을 기다리거나 처리 중인 단락 수
        # som 방식(해석기 system_recovery) 전용: 시스템 재판단·다시 듣기 복구 예산과, 지금 판단 중인 단락의 인식 작업
        self._system_recovery_budget = MenuRecoveryBudgets()
        self._processing_stt_task: asyncio.Task | None = None
        # 최대 길이에서 강제로 끊긴 단락의 인식 문장 (문장, 소리 시작 시각, 그때의 마지막 키 시각). 다음 단락과 이어 붙여 판단
        self._held_head: tuple[str, float | None, float | None] | None = None
        self._timers: dict[str, asyncio.Task] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        # 발행 대기열 (LLM_FIX_TRANSCRIPT=separate): 받아쓰기 고침을 기다리는 자막이 있는 동안 모든 이벤트를 발행 순서대로
        # 쌓아 두고 한 작업이 차례로 전송. 상태 변경은 즉시, 앱 전송만 지연 (잠금 안에서 대기하지 않음)
        self._outbox: list[tuple] = []
        self._drainer: asyncio.Task | None = None
        self._para_done: asyncio.Event | None = None  # 지금 처리 중인 단락의 판단 종료 여부 (고침 호출 시작 신호)
        self._unsent: set[int] = set()  # 대기열에 있어 아직 보내지 않은 자막 (id)

    # 공통 ------------------------------------------------------------------
    def spawn(self, coro: Awaitable) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        t.add_done_callback(self._log_exc)
        return t

    @staticmethod
    def _log_exc(t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception():
            log.error("세션 작업 오류", exc_info=t.exception())

    async def _ai(self, name: str, coro: Awaitable, fallback):
        """AI 모듈 호출. 실패(시간 초과·API 오류) 시에도 통화가 멈추지 않도록 대체값 사용."""
        try:
            return await coro
        except Exception:
            log.exception("AI 모듈 %s 실패. 대체값으로 진행", name)
            return fallback

    def timer(self, name: str, delay: float, fn: Callable[[], Awaitable[None]]) -> None:
        self.cancel_timer(name)

        async def run():
            await asyncio.sleep(delay)
            self._timers.pop(name, None)
            await fn()

        self._timers[name] = self.spawn(run())

    def cancel_timer(self, name: str) -> None:
        t = self._timers.pop(name, None)
        if t and t is not asyncio.current_task():
            t.cancel()

    @property
    def ended(self) -> bool:
        return self.status == "ended"

    def publish(self, event: str, data: dict, src: Prompt | None = None) -> None:
        """이벤트 전송. ended 가 마지막 이벤트이며, 종료 뒤 늦게 끝난 처리의 이벤트는 전송하지 않음.
        src: 이 이벤트를 만든 안내. 그 안내에 화면용 받아쓰기 고침(display_fixes)이 있으면 전송 시 화면 글에 적용.
        발행 대기열이 처리 중이면(받아쓰기 고침을 기다리는 자막이 앞에 있음) 뒤에 붙여 발행 순서 유지."""
        if self.closed:
            return
        if self._outbox_busy():
            self._outbox.append(("event", event, data, src))
            return
        self._send(event, data, src)

    def _send(self, event: str, data: dict, src: Prompt | None) -> None:
        if src is not None and src.display_fixes:
            self._display_fix(event, data, src)
        if self.recorder is not None:
            self.recorder.event(event, self._recorded(event, data))
        self.hub.publish(event, data)

    def _recorded(self, event: str, data: dict) -> dict:
        """통화 기록에 남길 사본. 값을 남기지 않는 경로 단계(private_steps, 규칙 안전망이 없는 해석기의 키패드 키)는 키·이름 대신
        자릿수만 남김. 앱에 보내는 이벤트와 self.path 는 그대로 둠 (같은 사전을 다시 읽으므로 고치지 않고 사본 생성)."""
        if not self.private_steps:
            return data
        if event == "path_updated" and data.get("step") in self.private_steps:
            sel = data.get("selected") or {}
            n = sum(c.isdigit() for c in str(sel.get("key") or ""))
            return {**data, "text": f"{n}자리 입력 (값 기록 안 함)",
                    "selected": {**sel, "key": None, "label": None, "digit_count": n}}
        if event == "input_rejected" and data.get("keys") and self.path and self.path[-1].get("step") in self.private_steps:
            return {**data, "keys": None, "digit_count": sum(c.isdigit() for c in str(data["keys"]))}
        return data

    def _display_fix(self, event: str, data: dict, src: Prompt) -> None:
        """화면 글에 받아쓰기 고침 적용. 상태에 저장된 같은 사전(경로·요청)도 함께 고쳐짐.
        고치는 대상은 인식 문장에서 온 값뿐: 그 안내의 선택지 이름(경로 문장·상태 문구·요청 문구 안에 든 이름 포함)과
        안내 원문(ars_prompt, unknown_prompt 문구). 분류 항목(category)·이동 키 이름·서버가 만든 문구('5 입력')는 고치지 않음."""
        fixes = src.display_fixes
        names: dict[str, str] = {}
        for o in src.options:
            new = fix_words(o.label, fixes)
            if new != o.label:
                names[o.label] = new

        def sub(s):
            if not isinstance(s, str):
                return s
            if src.text and s == src.text:
                return fix_words(s, fixes)
            for raw in sorted(names, key=len, reverse=True):
                s = s.replace(raw, names[raw])
            return s

        if event == "status_changed":
            new = sub(data.get("message"))
            if self.status_message == data.get("message"):
                self.status_message = new
            data["message"] = new
            return
        for k in ("text", "message", "ars_prompt"):
            if k in data:
                data[k] = sub(data[k])
        for o in data.get("options") or []:
            if isinstance(o, dict) and o.get("label") in names:
                o["label"] = names[o["label"]]
        sel = data.get("selected")
        if isinstance(sel, dict) and sel.get("label") in names:
            sel["label"] = names[sel["label"]]

    def _outbox_busy(self) -> bool:
        return self._drainer is not None and not self._drainer.done()

    def events_done(self) -> bool:
        """마지막 이벤트(ended)까지 앱 전송 완료 여부. 상태가 ended 여도 발행 대기열에 이벤트가 남아 있으면 미완료."""
        return self.closed and not self._outbox_busy()

    def _queue_fix_caption(self, p: Prompt, cap: dict, text: str) -> None:
        """받아쓰기 고침을 따로 받을 자막(LLM_FIX_TRANSCRIPT=separate)을 발행 대기열에 추가.
        자막은 self.captions 에 원문으로 이미 들어 있고, 고침이 오면 같은 사전의 글을 바꿔 전송."""
        self._outbox.append(("fix_caption", p, cap, text, self._para_done))
        self._unsent.add(id(cap))  # 전송 전에는 상태 조회(snapshot)에서 제외 (다시 연결한 앱의 중복 표시 방지)
        if not self._outbox_busy():
            self._drainer = self.spawn(self._drain())

    async def _drain(self) -> None:
        while self._outbox:
            item = self._outbox.pop(0)
            if item[0] == "fix_caption":
                await self._fix_caption(*item[1:])
                continue
            _, event, data, src = item
            try:
                self._send(event, data, src)
            except Exception:
                log.exception("발행 대기열: 이벤트 전송 오류 (%s)", event)

    async def _fix_caption(self, p: Prompt, cap: dict, text: str, gate: asyncio.Event | None) -> None:
        """그 단락의 판단 종료를 기다린 뒤(판단 모델과의 자원 경쟁 방지) 고침을 따로 요청하고 고친 자막 전송.
        어떤 오류가 나도 자막은 전송 (고치지 못하면 원문)."""
        rec: dict = {}
        try:
            if gate is not None:
                try:
                    await asyncio.wait_for(gate.wait(), 15)
                except asyncio.TimeoutError:
                    rec = {"source": "gate_timeout"}
            if not rec and p.kind not in ("menu", "info"):
                rec = {"source": "skipped_kind"}  # 판단 뒤 결과로 바뀐 안내(막다른 안내가 목적의 답) 등은 고치지 않음
            if not rec:
                rec = await self.d.interp.fix_prompt(p.text, p.options, p.kind)
            applied = list(rec.get("applied") or [])
            if applied:
                p.display_fixes = applied
                p.shown = fix_words(text, applied)
                cap["text"] = caption_text(p, p.shown)
        except Exception:
            log.exception("받아쓰기 고침(따로) 처리 오류. 원문 자막을 보낸다")
            rec = rec or {"source": "error"}
        finally:
            self._unsent.discard(id(cap))
            try:
                if self.recorder is not None:
                    m = lambda s: re.sub(r"\d", "#", str(s))  # noqa: E731  기록에 숫자를 남기지 않음
                    self.recorder.note("fixes", source=rec.get("source"), ms=rec.get("ms"),
                                       proposed=[f"{m(a)}→{m(b)}" for a, b in rec.get("proposed") or []],
                                       applied=[f"{m(a)}→{m(b)}" for a, b in rec.get("applied") or []],
                                       rejected=[f"{m(a)}→{m(b)} ({why})" for a, b, why in rec.get("rejected") or []])
                self._send("caption", cap, None)
            except Exception:
                log.exception("발행 대기열: 자막 전송 오류")

    def set_status(self, status: str, message: str | None = None, src: Prompt | None = None) -> None:
        if self.ended:
            return  # 종료 뒤 늦게 끝난 처리의 상태 되돌림 방지
        msg = message or STATUS_TEXT.get(status, "")
        if status == self.status and msg == self.status_message:
            return
        self.status, self.status_message = status, msg
        self.publish("status_changed", {"status": status, "message": msg}, src=src)

    def idle_status(self) -> str:
        """사용자 입력이 끝난 뒤 돌아갈 상태. AI 탐색 중이면 listening, 업무를 마친 뒤면 agent_done."""
        return "listening" if self.agent_active else "agent_done"

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "message": self.status_message,
            # human_agent 탐색은 내부 목적('<업무> 상담원 연결') 대신 업무 목적 표시. 업무가 없으면 내부 목적
            "goal": (self.task_goal or self.goal) if self.routing_mode == "human_agent" else self.goal,
            "routing_mode": self.routing_mode,
            "goal_summary": self.goal_summary,
            "path": self.path,
            "captions": [c for c in self.captions if id(c) not in self._unsent] if self._unsent else self.captions,
            "pending_request": self.pending,
            "summary": self.summary,
            "end_reason": self.end_reason,
        }

    # 생성·발신 ---------------------------------------------------------------
    async def create(self, goal: str | None, record: bool = False, agent: bool = False) -> dict:
        # AI 도우미 사용 여부와 목적은 통화 시작 시 결정. 승인 없이 바로 발신하고 첫 메뉴부터 탐색.
        # agent: 상담원 연결 요청(앱의 상담원 연결 버튼). 함께 입력한 업무로 '<업무> 상담원 연결' 목적 생성 (navigator.agent_goal)
        out = {"session_id": self.id, "session_token": self.token, "phone_number": self.phone}
        self.agent_request = bool(agent)
        self.goal = agent_goal(goal) if self.agent_request else (goal or "").strip()
        if self.agent_request:
            self.routing_mode, self.task_goal = "human_agent", agent_topic(self.goal)
        if record or getattr(self.s, "record_always", False):  # RECORD_ALWAYS=1: 앱 요청과 무관하게 기록 (시험용)
            meta = {"session_id": self.id, "phone": self.phone, "goal": self.goal,
                    "backend": self.s.backend, "stt": self.s.stt, "interpreter": self.s.interpreter}
            if self.agent_request:
                meta["agent_request"] = True
            if self.combo is not None:
                meta["combo"] = dict(self.combo)
            if self.line is not None:
                meta["line"] = f"번호{self.line + 1}"
            self.recorder = CallRecorder(meta)
        # 해석기의 통화 단위 상태 초기화 (hybrid 해석기의 '입력 요청 뒤' 상태 등). summarize_goal 은 목적 다시 말하기에서도
        # 호출되므로 새 통화 시작은 별도 통지. 이 메서드가 없는 해석기는 그대로
        start_call = getattr(self.d.interp, "start_call", None)
        if callable(start_call):
            start_call()
        self.goal_summary = await self._ai("summarize_goal", self.d.interp.summarize_goal(self.goal),
                                           f"'{self.goal}' 업무를 찾아 드릴게요.")
        self.agent_active = True
        self.nav = NavState(self.goal)
        out["goal_summary"] = self.goal_summary
        await self.dial()
        out["status"] = self.status
        return out

    async def dial(self) -> None:
        self.set_status("dialing")
        handlers = CallHandlers(self._on_answer, self._on_audio, self._on_transcript, self._on_end)
        if not self._line_ready():
            await self.finish("failed")
            return
        try:
            if self.line is None:
                self.call = await self.d.backend.dial(self.phone, handlers)
            else:
                self.call = await self.d.backend.dial(self.phone, handlers, line=self.line)
        except Exception as e:
            # 예외 문구는 기록하지 않음: SDK 오류 문구에 발신 번호 포함 가능. 계정 한도 거절 등을 구분할 status·code 만 기록
            log.error("세션 %s 발신 실패%s: %s (status=%s, code=%s)", self.id,
                      f" (번호{self.line + 1})" if self.line is not None else "", type(e).__name__,
                      getattr(e, "status", None), getattr(e, "code", None))
            await self.finish("failed")
            return
        if self.answered_at is None and self.end_reason is None and self.s.dial_timeout_s > 0:
            self.timer("dial_timeout", self.s.dial_timeout_s, self._dial_timeout)
        self.spawn(self._stt_worker())

    def _line_ready(self) -> bool:
        """번호가 여럿일 때: Session 생성 시 고른 번호를 발신 직전까지(목적 요약 대기 중) 잃었으면 빈 번호 재선택.
        재선택할 번호가 없으면 False."""
        rank = getattr(self.d.backend, "line_rank", None)
        if self.line is None or rank is None or rank(self.line) < 2:
            return True
        old, new = self.line, (self._repick() if self._repick else None)
        if new is None:
            log.error("세션 %s 발신 전 번호%d 을 쓸 수 없게 되었고 빈 번호가 없다", self.id, old + 1)
            return False
        self.line = new
        if self.recorder is not None:
            self.recorder.meta["line"] = f"번호{new + 1}"
        log.warning("세션 %s 발신 전 번호%d 을 쓸 수 없게 되어 번호%d 로 건다", self.id, old + 1, new + 1)
        return True

    async def _dial_timeout(self) -> None:
        if self.answered_at is None and self.end_reason is None:
            log.warning("발신 뒤 %.0f초 동안 상대가 받지 않아 끊는다", self.s.dial_timeout_s)
            await self.finish("no_answer")

    async def _on_answer(self) -> None:
        if self.end_reason is not None:
            # 발신 제한 시간·사용자 취소로 끝난 뒤 상대가 늦게 받은 경우. 통화 상태를 되살리지 않고 연결된 통화 종료
            if self.call is not None:
                try:
                    await self.call.hangup()
                except Exception as e:
                    log.warning("늦게 받은 통화 종료 요청 실패: %s", e)
            return
        self.cancel_timer("dial_timeout")
        self.answered_at = time.monotonic()
        self.timer("max_duration", self.s.max_call_s, lambda: self.finish("max_duration"))
        self.set_status("listening", self.goal_summary)

    async def _on_end(self, reason: str) -> None:
        if self.end_reason is not None or self.hangup_seen:
            return
        if reason == "remote_hangup":
            # 상대가 끊음. 끊기 직전까지 들은 안내와 결과 요약을 마친 뒤 종료 사유 결정 (명세 7장 7항).
            # 백엔드의 종료 알림은 즉시 반환.
            self.hangup_seen = True
            self.spawn(self._after_hangup())
            return
        await self.finish("user_canceled" if reason == "local_hangup" else reason, hangup=False)

    async def _after_hangup(self) -> None:
        # 단락 끝(무음)을 기다리지 못한 마지막 안내. 결과 안내 직후 바로 끊는 ARS 가 많음
        if self.segmenter.speaking:
            self._queue_segment(bytes(self.segmenter.buf), self.seg_started_at)
            self.segmenter.reset()
        flush = getattr(self.call, "flush_transcripts", None) if self.call is not None else None
        if flush is not None:
            # 받아쓰기를 직접 주는 백엔드: 끊긴 순간 듣던 단락 처리 (소리 인식 경로의 위 마무리 인식과 같은 역할)
            for text, started, _ in flush():
                await self._deliver(text, started, False)
        deadline = time.monotonic() + self.s.hangup_flush_s
        while ((self.stt_inflight or self._handed_over() or self._lock.locked() or self._held_head is not None)
               and time.monotonic() < deadline):
            if self._held_head is not None and not self.stt_inflight and not self._handed_over() and not self._lock.locked():
                await self._flush_head(now=True)  # 잘린 앞부분 뒤로 이어 들을 소리 없음
                continue
            await asyncio.sleep(0.05)
        if self.early_result_wait:
            # 첫 메뉴 전 결과 안내를 확정하려고 기다리던 중 ARS 가 끊은 경우: 인식할 소리가 더 없고 메뉴가 이어지지 않았으므로 즉시
            # 확정. 타이머가 이미 울려 확인이 잠금을 기다리는 중이어도 표시로 구분(두 확인은 표시로 한 번만 확정).
            # 요약은 남은 시간 안에서 대기(결과 확정까지 최소 1초), 넘기면 아래에서 안내 원문으로 대체. 제한 시간까지 끝나지 않은
            # 인식·판단(동시 통화로 밀린 경우 등)은 기다리지 않음(force). 그래도 확인을 마치지 못했으면(잠금을 얻지 못함) 끊김을 근거로
            # 결과로 확정 (요약은 안내 원문)
            self.cancel_timer("early_result")
            t = self.spawn(self._early_result_check(force=True))
            end = max(deadline, time.monotonic() + 1.0)
            while not t.done() and time.monotonic() < end:
                await asyncio.sleep(0.05)
            if self.early_result_wait and self.early_end_kind is not None:
                # 보류한 것이 끝맺음 안내(업무시간 외 등)인데 확인 미완료: 메뉴가 이어지지 않고 끊겼으므로 그 사유로 종료
                kind = self.early_end_kind
                log.warning("끊긴 뒤 첫 메뉴 전 끝맺음 안내의 확인을 마치지 못해 그 사유(%s)로 끝낸다", kind)
                self.early_result_wait = False
                self.early_end_kind = None
                await self.finish(kind, hangup=False)
                return
            if self.early_result_wait and not self.result_heard:
                log.warning("끊긴 뒤 첫 메뉴 전 결과 안내의 확인을 마치지 못해 결과로 확정한다 (안내 원문 요약)")
                self.early_result_wait = False
                self.result_heard = True
        if self.summary is None and (self.result_heard or self.status == "connected_to_agent"):
            # 결과(또는 상담원 인계) 요약이 제한 시간 안에 끝나지 않은 경우 (AI 응답 지연). 요약 없이 끝내지 않고
            # 안내 원문·기본 문장으로 summary 전송 뒤 ended 전송. 늦게 끝난 AI 요약은 폐기 (명세 7장 7항, 9.3)
            log.warning("끊긴 뒤 %.0f초 안에 요약이 끝나지 않아 안내 원문으로 대신한다", self.s.hangup_flush_s)
            if self.result_heard:
                res = self._result_fallback()
                self.summary = {"kind": "result", "summary": res["summary"], "key_info": [], "handoff_script": None}
            else:
                self.summary = {"kind": "handoff", "summary": "상담원 연결까지 진행했습니다.",
                                "handoff_script": self._handoff_fallback()}
            self.publish("summary", self.summary)
        if self.status == "connected_to_agent":
            reason = "agent_call_ended"
        elif self.result_heard:
            reason = "completed"
        elif self.pending is not None:  # 민감정보 직접 입력·재질문 응답 대기 중
            reason = "user_input_timeout"
        else:
            reason = "failed"
        await self.finish(reason, hangup=False)

    async def finish(self, reason: str, hangup: bool = True, note: str = "") -> None:
        if self.end_reason is not None:
            return
        self.end_reason = reason
        self.ended_at = time.monotonic()
        self.pending = None
        self.agent_active = False
        for name in list(self._timers):
            self.cancel_timer(name)
        if hangup and self.call is not None:
            try:
                await self.call.hangup()
            except Exception as e:
                log.warning("통화 종료 요청 실패: %s", e)
        end_text = END_TEXT.get(reason, "")
        if reason == "out_of_hours" and self.hours_note:
            end_text = f"{end_text} (ARS 안내: {self.hours_note})"  # ARS 가 말한 상담 가능 시간 문장
        self.set_status("ended", end_text)
        data = {"end_reason": reason, "message": end_text}
        if note:
            data["detail"] = note
        self.publish("ended", data)
        self.closed = True
        if self.recorder is not None:
            self._save_recording()
            asyncio.get_running_loop().call_later(self.s.recording_keep_s, self.drop_recording)
        for q in list(self.audio_out):
            q.put_nowait(None)
        self.stt_q.put_nowait(None)  # 음성 인식 작업 종료

    # 오디오 ----------------------------------------------------------------
    async def _on_audio(self, frame: bytes) -> None:
        if self.ended:
            return  # 종료된 통화 (늦게 받은 통화를 끊는 사이 들어온 소리 등)
        if self.recorder is not None and not self.closed:
            self.recorder.audio(frame)
        for q in list(self.audio_out):
            if q.qsize() < 50:  # 1초 이상 밀리면 폐기
                q.put_nowait(frame)
        if (self.call is None or self.call.provides_transcripts or self.status == "connected_to_agent"
                or self.hangup_seen):
            return
        # 되읽기 구간에서는 미리 인식 없이 단락 전체를 한 번에 인식. ARS 가 번호를 한 자씩 끊어 읽으므로(짧은 소리와 쉼의 반복)
        # 조각별 인식 시 숫자 누락·오인식 발생
        if self.secure_window:
            self.segmenter.on_chunk = None            # 인스턴스 값이 서버가 연결한 미리 인식(클래스 값)을 가림
        elif "on_chunk" in vars(self.segmenter):
            del self.segmenter.on_chunk              # 구간이 닫히면 다시 미리 인식
        was_speaking = self.segmenter.speaking
        seg = self.segmenter.feed(frame)
        if self.segmenter.speaking and not was_speaking:
            self.seg_started_at = time.monotonic()  # 이 단락의 소리 시작 판정 시각 (키 이전 안내 판별용)
        if seg is not None or self.segmenter.speaking:
            self.heard_since_press = True
        if seg is not None:
            self._queue_segment(seg, self.seg_started_at, forced=self.segmenter.last_forced)
            if self.segmenter.speaking:  # 최대 길이에서 쉼 자리로 잘라 뒷소리를 넘겨받은 다음 단락: 그 소리의 시작 시각
                self.seg_started_at = time.monotonic() - len(self.segmenter.buf) / 8000

    def _queue_segment(self, seg: bytes, started_at: float | None, forced: bool = False) -> None:
        self.stt_inflight += 1
        self.stt_q.put_nowait((seg, started_at, self.recorder.pos if self.recorder else 0, forced))

    def _ars_active(self) -> bool:
        """ARS 안내 진행 중이거나 끝난 안내를 아직 인식·처리 중인지 여부 (가상 ARS 는 call.speaking 으로 통지)."""
        return bool(getattr(self.call, "speaking", False) or self.segmenter.speaking or self.stt_inflight
                    or self._handed_over() or self._held_head is not None)

    def _handed_over(self) -> int:
        """받아쓰기를 직접 주는 백엔드(가상 ARS 받아쓰기 모드)가 넘겼으나 세션이 아직 잠금 아래로 가져가지 않은 단락 수.
        소리 인식 경로는 단락이 닫히는 순간 stt_inflight 를 올리므로, 넘김과 가져감 사이도 같은 'ARS 활동' 으로 집계. 실통화는 늘 0."""
        n = getattr(self.call, "transcripts_inflight", 0) if self.call is not None else 0
        return n if isinstance(n, int) else 0

    def _ars_active_for_recovery(self) -> bool:
        """som 방식 복구용: 지금 판단 중인 단락 자신의 인식은 빼고, 다른 인식 대기나 들어오는 말소리가 있는지 여부."""
        active = self._ars_active()
        if (active and self._processing_stt_task is not None
                and self._processing_stt_task is asyncio.current_task()):
            return bool(getattr(self.call, "speaking", False) or self.segmenter.speaking or self.stt_inflight > 1
                        or self._handed_over() or self._held_head is not None)
        return active

    @property
    def _som(self) -> bool:
        """판단 해석기가 som 방식(시스템 복구 흐름을 쓰는 해석기, system_recovery)인지 여부."""
        return bool(getattr(self.d.interp, "system_recovery", False))

    async def _classify_text(self, interp, text: str) -> Prompt:
        """분류 요청. som 방식 해석기는 지나온 계층과 함께 분류 (classify_history).
        som 해석기가 원문 구절 그대로 뽑은 선택지 이름('…을 원하시면')은 화면용 고침(display_fixes)으로만 다듬음 (판단·JEV 요청은 원래 이름)."""
        historical = getattr(interp, "classify_history", None) if getattr(interp, "system_recovery", False) else None
        if getattr(interp, "wants_layer_path", False):
            # 지나온 계층의 범주를 분류 전에 넘김 (해석기가 원하는 경우만, jevllm): 방금 누른 키의 범주를 이번 판단 경로에 포함
            interp.layer_path = [layer.prompt.category_of(layer.key) for layer in self.layers]
        if historical is None:
            return await interp.classify(text)
        p = await historical(text, self.goal or "", self.layers)
        if p is not None and p.options:
            p.display_fixes = display_fixes_for(p)
        return p

    async def _stt_worker(self) -> None:
        while not self.ended:
            item = await self.stt_q.get()
            if item is None:
                break  # 통화 종료
            seg, started_at, end_sample, forced = (*item, 0, False)[:4] if len(item) == 2 else (*item, False)[:4]
            del item
            rec = self.recorder
            try:
                if self.secure_window and not self.d.recognizer.is_local:
                    # 외부 STT 에는 입력값을 다시 읽어 주는 구간을 보내지 않음. 문장을 받지 못하면 구간을 닫을 계기가 없으므로
                    # 입력 전송 뒤 secure_skip_s 경과 시 구간을 닫고 인식 재개 (그 사이의 되읽기 확인은 사용자가 소리를 듣고 직접 입력)
                    if time.monotonic() - self.secure_sent_at < self.s.secure_skip_s:
                        if rec:
                            rec.note("stt_skipped", reason="민감정보 입력 뒤 외부 STT 로 보내지 않는 구간",
                                     seg_end_t=round(end_sample / 8000, 2))
                        continue
                    log.info("외부 STT: 되읽기 건너뛰기 %.0f초가 지나 인식을 재개한다", self.s.secure_skip_s)
                    self.secure_window = False
                voiced = sum(1 for i in range(0, len(seg) - FRAME + 1, FRAME) if rms(seg[i:i + FRAME]) > self.s.vad_rms) * FRAME / 8000
                if voiced < self.s.min_voiced_s:
                    # 키 전송(DTMF)에 끊긴 조각·딸깍음·잔향. 인식하면 "콩" 같은 글자가 자막으로 남음
                    log.info("소리 %.2f초뿐인 단락은 인식하지 않는다", voiced)
                    if rec:
                        rec.note("stt_skipped", reason=f"소리 {voiced:.2f}초뿐인 단락", seg_end_t=round(end_sample / 8000, 2))
                    continue
                t_stt = time.monotonic()
                text = await self.d.recognizer.recognize(seg)
                if rec:
                    rec.stt(text, end_sample, len(seg), time.monotonic() - t_stt, self.s.stt, forced=forced)
                if text and self.secure_window and self.secure_len and self.d.recognizer.is_local:
                    text = await self._readback_retry(seg, text)
                del seg
                if text and len(re.sub(r"[^가-힣0-9A-Za-z]", "", text)) < 3:
                    log.info("두 글자 이하 인식 결과는 버린다: %s", text)
                    if not forced:
                        await self._flush_head(now=True)
                    continue
                if text:
                    self._processing_stt_task = asyncio.current_task()  # som 방식 복구가 이 단락 자신의 인식을 진행 중으로 세지 않도록
                    try:
                        await self._deliver(text, started_at, forced)
                    finally:
                        self._processing_stt_task = None
                elif not forced:
                    await self._flush_head(now=True)
            except Exception:
                log.exception("안내 처리 오류")  # 한 단락의 오류로 인식 작업이 멈추지 않도록 처리
            finally:
                self.stt_inflight -= 1

    async def _readback_retry(self, seg: bytes, text: str) -> str:
        """민감정보를 보낸 뒤 되읽기 단락의 번호가 입력 자릿수(secure_len, 값은 저장하지 않음)보다 적으면 같은 소리를 빔 1 로 한 번 더
        인식. 다시 인식 문장도 되읽기 확인 문장이고 확인 질문 선택지를 모두 담고 있으며, 번호가 더 많고 입력 자릿수에 더 가까울
        때만 그 문장을 씀 (아니면 처음 문장). 번호 일부만 되읽는 문장('끝 네 자리')은 다시 인식하지 않음. 숫자를 채워 넣지 않음.
        다시 인식이 실패하면 처음 문장."""
        retry = getattr(self.d.recognizer, "recognize_retry", None)
        if retry is None or not self.readback_retry_armed or not _is_readback(text):
            return text
        want, have = self.secure_len or 0, _readback_digits(text)
        if have >= want or (have and _PARTIAL_READBACK_RE.search(text)):
            return text
        t0 = time.monotonic()
        try:
            alt = await retry(seg)
        except Exception:
            log.warning("되읽기 다시 인식 실패, 처음 결과 사용", exc_info=True)
            if self.recorder is not None:
                self.recorder.note("stt_retry", digits_before=have, digits_after=None, entered=want, used=False, error=True)
            return text
        if not alt:
            return text  # 빔 1 인식기 등: 같은 디코딩이라 다시 인식하지 않음
        got = _readback_digits(alt)
        opts_kept = {o.key for o in readback_options(text)} <= {o.key for o in readback_options(alt)}
        use = _is_readback(alt) and opts_kept and got > have and abs(got - want) < abs(have - want)
        secs = time.monotonic() - t0
        log.info("되읽기 숫자 %d/%d자리 → 빔 1 다시 인식 %d자리 (%.2fs, %s)", have, want, got, secs,
                 "다시 인식 결과 사용" if use else "처음 결과 유지")
        if self.recorder is not None:
            self.recorder.note("stt_retry", digits_before=have, digits_after=got, entered=want, used=use, seconds=round(secs, 2))
        return alt if use else text

    async def _deliver(self, text: str, started_at: float | None, forced: bool = False,
                       taken: Callable[[], None] | None = None) -> None:
        """인식한 단락을 안내 처리로 전달. 무음이 아닌 최대 길이(Segmenter max_s)에서 끊긴 단락은 발화 도중에 잘린
        앞부분이므로 바로 판단하지 않고 다음 단락과 이어 붙여 한 단락으로 판단 (2026-11-07 실통화: 45초가 넘는 하나카드 안내가
        메뉴 중간에서 잘려, 앞부분은 선택지 없는 일반 안내로, 뒷부분은 뒤쪽 선택지만 있는 메뉴로 판단되었다).
        그 사이에 키를 보냈으면 이어 붙이지 않고 앞부분을 먼저 따로 처리.
        다음 단락이 오지 않으면(ARS 발화 종료) 앞부분만으로 처리 (_flush_head).
        taken: 받아쓰기 백엔드의 넘김 표시를 해제하는 함수. 이 단락을 잠금 아래로 가져갈 때 호출 (on_paragraph)."""
        head = self._held_head
        if head is not None:
            self._held_head = None
            self.cancel_timer("held_head")
            h_text, h_start, h_key = head
            if h_key == self.last_key_at:
                text, started_at = f"{h_text} {text}", h_start
            else:
                await self.on_paragraph(h_text, h_start)
        if forced and not self.hangup_seen and not self.ended:
            self._held_head = (text, started_at, self.last_key_at)
            self.timer("held_head", HELD_HEAD_CHECK_S, self._flush_head)
            return
        await self.on_paragraph(text, started_at, taken=taken)

    async def _flush_head(self, now: bool = False) -> None:
        """잡아 둔 잘린 앞부분 단독 처리. 이어지는 소리를 아직 듣거나 인식 중이면 잠시 더 대기."""
        if self._held_head is None or self.ended:
            return
        # 받아쓰기를 직접 주는 백엔드(녹음 기반 가상 ARS 받아쓰기 모드)는 Segmenter 를 쓰지 않으므로 통화가 알리는 '단락을 듣는 중'
        # (call.speaking) 으로 이어지는 소리 대기
        busy = self.segmenter.speaking or self.stt_inflight or self._handed_over() or (
            self.call is not None and self.call.provides_transcripts and bool(getattr(self.call, "speaking", False)))
        if not now and busy and not self.hangup_seen:
            self.timer("held_head", HELD_HEAD_CHECK_S, self._flush_head)
            return
        h_text, h_start, _ = self._held_head
        self._held_head = None
        self.cancel_timer("held_head")
        await self.on_paragraph(h_text, h_start)

    async def mic_audio(self, frame: bytes) -> None:
        """앱 마이크 음성. 상담원 연결 중에만 통화로 전송 (그 외에는 사용자 음성 중계 없음)."""
        if self.call is None or self.ended or self.status != "connected_to_agent":
            return
        await self.call.send_audio(frame)

    async def _on_transcript(self, text: str, started_at: float | None = None, forced: bool = False,
                             taken: Callable[[], None] | None = None) -> None:
        # 소리 인식 경로(_stt_worker)와 같게 _deliver 로 전달: 최대 길이에서 잘린 단락(forced)은 다음 단락과 이어 붙임.
        # 가상 ARS(sim)는 forced 를 주지 않으므로 바로 on_paragraph 로 진행. taken: 넘김 표시를 해제하는 함수 (가상 ARS 받아쓰기 모드)
        await self._deliver(text, started_at, forced, taken)

    # 안내 처리 ---------------------------------------------------------------
    async def on_paragraph(self, text: str, started_at: float | None = None,
                           taken: Callable[[], None] | None = None) -> None:
        """ARS 안내 한 단락. started_at 은 그 안내의 소리 시작 시각(monotonic). 모르면 None.
        taken: 잠금을 얻은 순간 호출하는 함수 (받아쓰기 백엔드의 넘김 표시 해제. 단락 자신의 판단 중에는 넘김으로 세지 않음)."""
        if self.ended or self.status == "connected_to_agent":
            return
        async with self._lock:
            if taken is not None:
                taken()
            # 이 단락의 판단(키 전송·재질문 발행 포함) 종료를 받아쓰기 고침 작업에 통지 (고침 호출은 그 뒤에 시작)
            self._para_done = done = asyncio.Event()
            try:
                await self._paragraph(text, started_at)
            finally:
                done.set()
                self._para_done = None

    async def _paragraph(self, text: str, started_at: float | None = None) -> None:
        interp = self.d.interp
        self.cancel_timer("no_response")
        self.para_seq += 1
        self.cancel_timer("dead_end")  # 새 안내 수신: 앞 단락의 막다른 안내 판단은 취소 (ARS 가 말을 이어 감)
        pending_dead_end, self._dead_end_pending = self._dead_end_pending, None   # 새 안내가 무입력 안내면 앞 후보 판단 (아래)
        timeout_notice = False
        if self.early_result_wait:
            # 첫 메뉴 전 결과 안내를 확정하려고 기다리는 중 새 안내 수신: 이 안내를 마친 뒤부터 다시 대기. 이 안내가 메뉴·입력
            # 요청·상담원 대기 등이면 그 상태가 결과 확정을 막음 (_early_result_check). 일반 안내('이용해 주셔서 감사합니다')면 결과 유지
            self.timer("early_result", EARLY_RESULT_WAIT_S, self._early_result_check)
        self.heard_since_press = True  # ARS 의 키 응답
        # 키 전송 전에 시작된 안내는 키 이전 상태에 대한 안내 (예: AI 판단이 ARS 입력 대기보다 늦어
        # ARS 가 같은 메뉴를 다시 읽기 시작한 뒤 키를 누른 경우). 메뉴·확인 질문·입력 오류로 다루지 않고 자막만 기록.
        stale = started_at is not None and self.last_key_at is not None and started_at < self.last_key_at
        replying = self.key_outstanding and not stale  # 이 안내가 보낸 키에 대한 ARS 의 첫 반응인지 여부
        # ARS 발화 도중에 보낸 키 직후 시작된 단락: 잘린 안내의 뒷부분일 수 있어 결과·종료 판단에서 제외 (CUT_TAIL_S)
        cut_tail = (not stale and self.key_cut_speech and started_at is not None and self.last_key_at is not None
                    and started_at - self.last_key_at < CUT_TAIL_S)
        if not stale:
            self.key_outstanding = False

        # 되읽기 구간이 닫혔더라도 "입력하신 … 맞으면" 문장에 한국어 숫자를 길게 읽는 부분이 있으면 입력값 되읽기로 판정
        readback = bool(_READBACK_RE.search(text)) and (self.secure_window or bool(_SPOKEN_RUN_RE.search(text)))
        # 되읽기 구간에서 '맞으면 N번 … 다시 입력(틀리면)' 확인 질문이면 "입력하신" 이 없어도 되읽기로 판정 (숫자만 읽는 ARS)
        readback = readback or (self.secure_window and bool(_CONFIRM_RE.search(text)) and bool(_REENTER_RE.search(text)))
        pre: Prompt | None = None  # 되읽기 분기에서 이미 분류한 안내 (재분류 생략)
        if self.secure_window or readback:
            # ARS 가 입력값을 다시 읽어 주는 안내("입력하신 번호는 … 맞으면 1번")는 숫자를 가린 뒤 판단만 하고 자막 미전송
            if readback:
                masked = mask_readback(text)
                # 판단(AI)·자막·로그에는 숫자를 가린 문장 사용. 사용자에게 묻는 문구(요청 message·상태 문구)만 원문 사용:
                # 사용자가 입력한 값의 정상 전달 여부를 화면에서 확인하려면 되읽힌 숫자 표시 필요.
                # 서버 메모리에는 요청이 닫히면 남지 않음. 단, 시험용 통화 기록(record)을 켜면 되읽기 소리·인식 문장과 함께
                # 이 문구도 기록에 남음. 숫자 묶음 사이 공백은 제거 (STT 가 쉼마다 넣음)
                shown = join_readback_digits(text)
                del text
                p = await self._ai("classify", self._classify_text(interp, masked), Prompt(kind="info", text=masked))
                if self.recorder is not None:
                    self.recorder.note("classified", kind=p.kind, text=masked, readback=True,
                                       options=[f"{o.key}:{o.label}" for o in (p.options or [])],
                                       **({"fixes": [f"{a}→{b}" for a, b in p.fixes]} if p.fixes else {}))
                if p.kind not in ("menu", "confirm", "sensitive"):
                    # AI 가 되읽기 확인 질문을 놓쳐도(가린 숫자 때문에 일반 안내로 분류 등) 사용자에게 질문.
                    # 묻지 않으면 ARS 는 응답을 기다리다 '입력 시간이 경과하였습니다'로 넘어감
                    opts = readback_options(masked)
                    if opts:
                        log.info("되읽기 확인 질문을 AI 가 %s 로 분류 — 규칙으로 선택지를 찾아 사용자에게 묻는다", p.kind)
                        p = Prompt(kind="confirm", text=masked, options=opts)
                if p.kind in ("menu", "confirm"):
                    p.readback_of = self.secure_label  # 경로 표시용: 어떤 입력을 확인한 단계인지
                if p.kind in ("menu", "confirm", "sensitive", "info"):
                    if p.kind != "info":
                        # 모아 둔 한 자씩 입력 키가 되읽기 확인 질문의 답·새 입력의 앞자리로 나가지 않도록 폐기
                        self._drop_stream_buf("ARS가 다음 안내로 넘어가 누르신 번호를 보내지 않았습니다. 입력 안내가 다시 나오면 다시 입력해 주세요.")
                    if p.kind == "sensitive":
                        await self._ask_secure(p)
                    elif p.kind in ("menu", "confirm"):
                        self.current, self.current_fresh = p, True
                        if self._streamed_open():
                            # 한 자씩 입력 뒤 ARS 가 되읽기 확인으로 넘어감 (입력 완료 전). 열어 두면 직접 누르기가 막힘
                            self._close_request()
                            self.set_status(self.idle_status())
                        if self.agent_active:
                            # 되읽힌 숫자를 포함한 원문으로 질문. 사용자가 글로 답해 AI 가 답을 해석할 때는 가린 문장 제공
                            await self._ask(p, "unknown_prompt", p.options, shown, ai_question=masked)
                    return
                # 되읽기 표현('입력하신 … 확인')이 있어도 결과·문자 발송·거절·입력 오류 같은 다음 단계 안내는 버리지 않고
                # 가린 문장으로 아래 공통 처리에 전달 (버리면 결과 요약이 없고, 끊긴 뒤 completed 대신 failed 로 종료)
                pre, text, hit, hit2 = p, masked, 0, 0
            else:
                # 되읽기 문장이 아니어도, 입력한 자릿수와 같은 숫자열이 들리면 입력값으로 보고 가림 (금액·날짜는 자릿수가 달라 유지).
                # 인식이 한두 글자 틀려 자릿수가 달라져도, 한국어 숫자를 여섯 자 이상 이어 읽는 것은 입력값 되읽기이므로 가림
                text, hit = mask_digits(text, self.secure_len)
                text, hit2 = _SPOKEN_RUN_RE.subn(lambda m: "*" * _digit_count(m.group()), text)
                if hit or hit2:
                    log.info("민감정보 입력 뒤 숫자열을 가렸다")
            closing_secure = True
        else:
            closing_secure = False

        if pre is None:
            p = await self._ai("classify", self._classify_text(interp, text), Prompt(kind="info", text=text))
            if self.recorder is not None:
                self.recorder.note("classified", kind=p.kind, text=text,
                                   options=[f"{o.key}:{o.label}" for o in (p.options or [])],
                                   **({"fixes": [f"{a}→{b}" for a, b in p.fixes]} if p.fixes else {}))
            if (p.kind in ("menu", "confirm") and p.options and not stale and _ERR_SENT_RE.match(text)
                    and self.agent_active and self.pending is None and not self.secure_queued
                    and not self.last_send_secure and not self._stream_live(started_at)
                    and (replying or self._own_key_menu(p, text)) and self._rereads_key_menu(p, text)):
                # 입력 오류 문장('입력시간이 지났습니다'·'잘못 누르셨습니다')으로 시작하는, 방금 키를 누른 메뉴의 재낭독을 해석기가
                # 메뉴로만 분류한 경우. 메뉴로 판단하면 다시 누르기는 하지만 거부로 세지 않아 '2회 연속이면 묻기' 가 걸리지 않고,
                # 같은 메뉴 계층이 쌓임 (같은 메뉴를 거쳐 온 것으로 보여 '처음으로' 복귀 뒤 실패한 갈래로 다시 들어감). 입력 오류로
                # 바꾸어 아래 입력 오류 처리(거부로 세고 계층을 되돌린 뒤 바로 다시 판단, 2회면 재질문)에 맡김. 해석기가 같은 단락을
                # 입력 오류로 분류할 때와 같은 결과 (모델에 따른 분류 차이 제거).
                # 바꾸지 않는 경우: 다른 메뉴(키가 받아들여짐: 메뉴 그대로 자막·되돌아옴 판단), 업무 완료 뒤, 사용자 요청이 열림
                # (재질문 중 다시 듣기 키 등), 민감정보 입력값 전송 대기 중(메뉴로 넘어가면 입력값을 버려야 함)
                log.info("입력 오류 문장으로 시작하는 단락(%s 로 분류)을 방금 누른 키의 입력 오류로 처리", p.kind)
                p.kind = "input_error"
            if p.kind == "input_error":
                p = self._reprompt_after_error(p, text)
        else:
            p = pre
        # 키에 대한 첫 반응은 아니지만(키 뒤에 다른 안내가 먼저 옴) 방금 키를 누른 메뉴를 입력 오류와 함께 다시 읽음: 그 키는 받아들여지지
        # 않음. 일반 안내·메뉴로 보지 않고 거부로 집계 (메뉴로 보면 다시 누를 때마다 계층이 쌓이고 '2회 연속이면 묻기' 가 늦어짐)
        own_retry = (p.kind == "input_error" and not replying and not stale and not self.last_send_secure
                     and not self._stream_live(started_at) and self._own_key_menu(p, text))
        if self.secure_queued and (p.kind not in ("sensitive", "input_error", "info")
                                   or (p.kind == "input_error" and p.options and not self._same_input_reprompt(text))):
            # 입력 오류 안내와 함께 메뉴를 다시 읽은 단락(선택지가 있는 입력 오류)도 ARS 가 입력 단계를 떠난 것으로 판정. 다만 같은 입력
            # 항목의 재안내('…누르신 후 우물정자를 눌러 주시고, 모르시면 1번')는 입력 단계에 머문 것이므로 입력값 전송
            # (다른 항목의 입력 안내로 넘어갔으면 앞 항목 값은 전송하지 않음)
            log.info("입력값 전송 대기 중 ARS 가 다른 안내(%s)로 넘어감. 입력값을 보내지 않는다", p.kind)
            self.secure_abort = True
        # 민감정보 전송 전에 시작된 안내(입력 대기 중의 '입력시간이 경과하였습니다' 재안내 등)로는 되읽기 구간을 닫지 않음.
        # 그 안내의 판단이 끝나기 전에 입력값이 전송될 수 있고, 되읽기는 전송 뒤에 옴
        before_send = started_at is not None and self.secure_sent_at > started_at
        if closing_secure and (p.kind != "info" or hit or hit2) and not before_send:
            # 되읽기 구간은 메뉴·결과·입력 요청처럼 다음 단계가 분명한 안내가 와야 닫음.
            # "잠시만 기다려 주십시오" 같은 짧은 안내로 닫으면 그 뒤에 오는 실제 되읽기 문장의 숫자가 가려지지 않음
            self.secure_window = False
        if p.kind == "input_error" and (not replying or stale) and not self._stream_live(started_at) and not own_retry:
            # 보낸 키에 대한 반응이 아닌 '입력 시간이 초과되었습니다'·'잘못 누르셨습니다'(사용자 응답 대기 중
            # ARS 의 대기 시간 경과 등)는 입력 오류가 아닌 ARS 의 안내로 취급 (명세 5.2).
            # 그 뒤에 메뉴를 다시 읽었으면(선택지가 있으면) 그 메뉴로 판단 ('입력이 되지 않았습니다. ○○는 5번 … 눌러주세요' 를
            # 반복해 듣고도 판단하지 않아 ARS 가 끊는 문제 방지). 키보다 먼저 시작된 안내는 그대로 일반 안내
            if self.queued_key and not p.options and not stale:
                self.queued_err = True  # 복귀 대기 키 뒤 따로 온 입력 오류: 다음에 그 메뉴를 다시 읽으면 그 키를 다시 누름
            p.kind = ("confirm" if _CONFIRM_RE.search(text) else "menu") if (p.options and not stale) else "info"
            timeout_notice = p.kind == "info" and not stale
        if stale and p.kind in ("menu", "confirm"):
            p.kind = "info"
        if p.kind == "menu" and pre is None and not cut_tail:
            p = await self._bounced_notice(p, text)
        if (cut_tail and p.kind in ("result", "message_sent", "out_of_hours", "rejected")
                and len(re.sub(r"[^가-힣0-9A-Za-z]", "", text)) < CUT_TAIL_MAX_CHARS):
            log.info("키가 ARS 안내 도중에 들어가 잘린 뒷부분(%s 로 분류, %d자)은 결과·종료로 보지 않는다", p.kind,
                     len(re.sub(r"[^가-힣0-9A-Za-z]", "", text)))
            p.kind = "info"
        if p.kind not in ("input_error", "info") and not self.redo_same_menu:
            # 직전 키가 받아들여짐(메뉴·결과 등 다음 단계). 입력 오류 '2회 연속'을 다시 집계. 일반 안내로는 바로 초기화하지 않음:
            # '잠시만 기다려 주십시오' 뒤 시간 초과 + 같은 메뉴가 되풀이되면 매번 1회로만 세어 재질문이 나오지 않기 때문.
            # 일반 안내 뒤의 입력 오류는 키를 누른 메뉴를 다시 읽을 때만 횟수 누적 (info_since_reject, 입력 오류 처리)
            self.input_errors = 0
            self.info_since_reject = False
        # 복귀 대기 키가 받아들여지지 않은 채 그 키를 누른 메뉴를 다시 들음: 다시 듣기 키(무응답 복구) 뒤의 재낭독이거나, 따로 온 입력
        # 오류 단락 뒤의 재낭독. 다음 대기 키 대신 그 키를 다시 누름 (다음 대기 키를 이 메뉴에서 누르면 엉뚱한 갈래로 들어감)
        requeue = (self.queued_key if (p.kind == "menu" and self.queued_key and (self.redo_same_menu or self.queued_err)
                                       and self.layers and self.layers[-1].key == self.queued_key
                                       and self._menu_matches(p, self.layers[-1].prompt, text)) else None)
        if p.kind != "input_error":
            if p.kind != "info":
                # 다음 단계 안내(메뉴·결과 등) 수신: 복귀 중 누른 대기 키는 받아들여짐. 일반 안내('잠시만 기다려 주십시오')로는 지우지
                # 않음: 그 뒤 시간 초과와 함께 같은 메뉴가 오면(거부) 그 키를 다시 눌러야 함 (지우면 다음 대기 키를 그 메뉴에서 누름)
                self.queued_key = None
                self.queued_err = False
            if p.kind != "sensitive":
                self.secure_errors = 0
        self.current_fresh = True if p.kind in ("menu", "confirm") else self.current_fresh
        # 키 전송에 잘린 안내(키보다 먼저 시작된 단락, 또는 키 전송 뒤 post_key_caption_s 안에 끝난 일반 안내)는 자막 미전송.
        # "이번에 입력해 주십시오" 같은 조각이 남는 것을 방지. 메뉴·결과·확인처럼 다음 단계가 분명한 안내는 그대로 전송
        cut = p.kind == "info" and (stale or (self.last_key_at is not None
                                             and time.monotonic() - self.last_key_at < self.s.post_key_caption_s))
        if cut:
            log.info("키 전송에 잘린 안내는 자막으로 보내지 않는다: %s", text[:40])
        # 마지막 키 뒤에 들은 일반 안내 (막다른 안내가 여러 단락으로 나뉘어 와도 함께 판단하기 위함). 메뉴 수신 시 초기화.
        # 키에 잘린 조각(앞 안내의 꼬리)은 제외
        if p.kind in ("menu", "confirm"):
            self.since_key_texts = []
        elif p.kind == "info" and not (stale or cut or cut_tail):
            self.since_key_texts = (self.since_key_texts + [text])[-4:]
            if self.input_errors and not self.redo_same_menu:
                # 다시 누른 키가 받아들여진 뒤의 일반 안내만 집계. 거부 뒤 메뉴를 다시 읽기 전의 안내('메뉴를 다시 안내해 드리겠습니다')는
                # 같은 메뉴 거부의 일부라 집계하지 않음 (세면 같은 메뉴의 두 번째 거부를 1회로 다시 세어 재질문하지 않음)
                self.info_since_reject = True
            if self.secure_warned and _choice_said(self.secure_warned["label"], self.secure_warned["others"], (text, p.shown),
                                                   self.secure_warned["field"]):
                self.secure_warned["heard"] = True  # 경고를 붙인 요청이 열린 동안 고른 선택지 이름을 들음: 다음 재알림에서 경고 제거
        if p.kind not in ("sensitive", "input_error") and not cut:
            # 받아쓰기 고침(LLM_FIX_TRANSCRIPT=1)이 있으면 고친 문장을 자막·결과 요약에 사용. 판단·기록에는 원문 사용
            cap = {"step": self.step + 1, "speaker": "ars", "text": caption_text(p, p.shown or text), "at": now_iso()}
            self.captions.append(cap)
            if p.kind in ("menu", "confirm"):
                self._choice_caps.add(id(cap))
            if (pre is None and not closing_secure and p.kind in ("menu", "info") and not self.closed
                    and getattr(interp, "fix_mode", "0") == "separate"):
                # 고침을 따로 받는 경우 (LLM_FIX_TRANSCRIPT=separate): 이 단락의 판단 종료 뒤 고침을 요청하고 자막은 그 뒤에 전송.
                # 되읽기·민감정보 입력 뒤 구간(가리지 못한 숫자가 고침으로 완성될 가능성)·확인 질문·민감정보·입력 오류·
                # 결과·종료·상담원 안내는 고치지 않음
                self._queue_fix_caption(p, cap, text)
            else:
                self.publish("caption", cap)

        if self.unconfirmed_multi is not None and not stale:
            keys_m, label_m, others_m, menu_m = self.unconfirmed_multi
            field = (p.label or "") if p.kind == "sensitive" else ""
            if _choice_said(label_m, others_m, (text, p.shown), field):
                self.unconfirmed_multi = None  # 고른 선택지의 안내 수신 (키가 제대로 들어감)
            elif p.kind == "sensitive":
                # 두 자리 키를 누른 뒤 그 선택지 이름 없이 바로 입력 안내: 앞자리만 들어가 다른 메뉴의 입력 단계일 가능성.
                # 입력 요청은 그대로 열되(같은 입력 단계일 수도 있음) 사용자가 ARS 안내를 확인하도록 알림 (_ask_secure)
                log.info("%s자리 키를 누른 뒤 고른 선택지 안내 없이 입력 안내가 나와 입력 요청에 경고를 붙인다", len(keys_m))
                p.choice_warning = (keys_m, label_m, others_m, menu_m)
                self.unconfirmed_multi = None
            elif p.kind not in ("info", "input_error"):
                self.unconfirmed_multi = None  # 메뉴·결과 등 다음 단계로 넘어감

        # AI 사용 여부와 관계없이 처리하는 안내
        if p.kind == "sensitive":
            live = self._stream_live(started_at)
            if stale and self.last_send_secure and not live:
                # 사용자가 번호를 누르는 동안 ARS 가 같은 입력 안내를 되풀이한 것이 뒤늦게 인식된 경우.
                # 이미 입력을 보냈으므로 새 입력 요청 생성 생략 (생성 시 사용자가 같은 번호를 또 입력해 '잘못된 입력' 반복)
                log.info("입력 전송 전에 시작된 입력 안내가 뒤늦게 도착. 무시한다")
                return
            if self.secure_queued and not live:
                # 입력값을 받아 ARS 안내 종료를 기다리는 중에 같은 입력 안내가 다시 들린 경우. 안내가 끝나면 그 값을 전송
                log.info("입력값 전송 대기 중 입력 안내가 다시 들림. 새 입력 요청을 만들지 않는다")
                return
            if live:
                # 한 자씩 입력으로 키를 보낸 뒤 시작된 새 입력 안내: 다음 항목(종료 키 없는 고정 자릿수 항목)이거나,
                # ARS 가 앞서 누른 번호를 버리고 다시 받는 경우. 같은 요청을 이어 쓰면 항목 이름·자릿수가 어긋나므로 요청을 새로 엶.
                # 이 안내 중에 누른 키(모아 둔 키)는 전송하지 않음 (새 입력 칸에서 다시 누름)
                log.info("한 자씩 입력 뒤 새 입력 안내. 입력 요청을 새로 연다")
                self._drop_stream_buf("새 입력 안내가 나와 누르신 번호를 보내지 않았습니다. 안내에 맞춰 다시 입력해 주세요.")
                self._close_request()
                await self._ask_secure(p)
                return
            if self.pending and self.pending["kind"] == "secure_input":
                # 입력하는 동안 ARS 가 안내를 되풀이함. 요청은 그대로 두고 만료 시각만 늘려 같은 요청 재통지
                # 새 사전으로 바꾸어 이미 보낸 요청 이벤트의 기록(기록 파일·이벤트 기록)은 그대로 유지
                self.pending = {**self.pending, "expires_at": now_iso(self.s.secure_expire_s)}
                self._drop_stale_warning(text, p)
                self.publish("request", self.pending)
                return
            await self._ask_secure(p)
            return
        if p.kind == "input_error":
            if self.last_send_secure:
                # 민감정보 입력값에 대한 오류. 입력값은 어디에도 남기지 않으며(keys=null), 메뉴 키 오류로 세지 않음.
                # ARS 가 입력 안내를 다시 하면 새 secure_input 요청 발행 (명세 6.3 처리 규칙 6)
                self.secure_errors += 1
                self.publish("input_rejected", {
                    "keys": None, "actor": "user", "ars_prompt": text, "retry_count": self.secure_errors,
                    "message": "입력하신 번호가 인식되지 않았습니다. 안내에 따라 다시 입력해 주세요.",
                })
                # ARS 는 대개 같은 단락에서 재입력을 요구 ('잘못 누르셨습니다. … 번호를 누르신 후 우물정자를 눌러주십시오').
                # AI 가 이 단락을 입력 요청이 아닌 입력 오류로 분류하면 입력 안내가 따로 오지 않아 사용자에게 다시 묻지 못하고,
                # ARS 는 입력 시간 초과를 되풀이하다 끊음. 같은 항목으로 새 입력 요청을 즉시 생성.
                # ARS 가 입력 없이 메뉴로 넘어가면 그 메뉴에서 요청 종료 (아래 메뉴 처리)
                if not p.options:
                    if self._stream_live(started_at):
                        # 한 자씩 입력으로 보낸 번호를 ARS 가 거부: 다시 누를 번호는 새 요청으로 수신 (자릿수가 이어지지 않도록).
                        # 오류 안내 중에 누른 키는 전송하지 않음 (위 알림이 재입력을 안내)
                        self._drop_stream_buf(None)
                        self._close_request()
                        label, term = self.secure_req
                        await self._ask_secure(Prompt(kind="sensitive", text=text, label=label, terminator=term))
                        return
                    if not self.secure_queued and not (self.pending and self.pending["kind"] == "secure_input"):
                        label, term = self.secure_req
                        await self._ask_secure(Prompt(kind="sensitive", text=text, label=label, terminator=term))
                    return
                # 거부 안내와 메뉴 재낭독이 한 단락('잘못 누르셨습니다. ○○는 1번 …'): 입력 요청을 다시 열지 않고 그 메뉴로 판단
                log.info("민감정보 입력 거부 뒤 같은 단락의 메뉴(선택지 %d개)로 판단", len(p.options))
                p.kind = "confirm" if _CONFIRM_RE.search(text) else "menu"
                self.current_fresh = True
                self.since_key_texts = []
            else:
                layer = self._key_layer()
                key_menu = layer.prompt if layer is not None else self.current
                if p.options and key_menu is not None and not self._menu_matches(p, key_menu, text):
                    # 같은 단락의 메뉴가 키를 누른 메뉴와 다름: 키는 받아들여졌고 시간 초과는 다음 메뉴의 것 (첫 낭독을 못 들은 경우).
                    # 거부로 세지 않고 계층을 유지한 채 새 메뉴로 판단. 이전 계층 복귀 중이던 키도 받아들여졌으므로 대기 키 삭제
                    # (남기면 다음 거부 때 다른 메뉴에서 그 키를 다시 누름)
                    log.info("입력 오류 안내 뒤 다른 메뉴(선택지 %d개): 직전 키는 받아들여진 것으로 보고 새 메뉴로 판단", len(p.options))
                    self.last_press_layer = None
                    self.input_errors = 0
                    self.info_since_reject = False
                    self.queued_key = None
                    p.kind = "confirm" if _CONFIRM_RE.search(text) else "menu"
                    self.current_fresh = True
                    self.since_key_texts = []
                else:
                    if self.info_since_reject and not (p.options and key_menu is not None
                                                       and self._menu_matches(p, key_menu, text)):
                        # 앞 거부 뒤 일반 안내를 들은 다음 단계의 오류: 키를 누른 메뉴를 다시 읽은 것이 아니면 횟수를 처음부터 집계
                        # (앞 단계의 거부 1회가 남아 첫 오류에서 지난 메뉴 선택지로 재질문하는 문제 방지)
                        self.input_errors = 0
                    self.info_since_reject = False
                    self.publish("input_rejected", {
                        "keys": self.path[-1]["selected"]["key"] if self.path else None,
                        "actor": self.last_key_actor, "ars_prompt": text, "retry_count": self.input_errors + 1,
                        "message": "번호가 인식되지 않아 다시 시도합니다." if self.agent_active else "번호가 인식되지 않았습니다.",
                    })
                    if not self.agent_active:
                        return
                    self.last_press_layer = None
                    self.input_errors += 1
                    self.redo_same_menu = True
                    self.redo_by_user = self.last_key_actor == "user"
                    self.redo_menu = key_menu
                    self.rejected_layer = layer  # 같은 메뉴를 다시 들으면 되돌림 (남기면 '처음으로' 복귀가 실패한 갈래로 다시 들어감)
                    if self.queued_key:
                        self.key_queue.insert(0, self.queued_key)  # 이전 계층 복귀 중 거부된 키는 메뉴가 다시 나오면 재전송
                        self.queued_key = None
                    if self.input_errors >= 2 and self.current and not self._handoff_open():
                        # 같은 단계 2회 연속 거부: 사용자에게 질문 (업무 목적 입력 중이면 그 요청을 덮지 않음). 이전 계층 복귀 중이었으면 복귀 중단 (답 뒤에 거부된 대기 키를
                        # 또 누르지 않도록)
                        self.key_queue.clear()
                        self.queued_key = None
                        self.after_backtrack = False
                        await self._ask(self.current, "input_rejected", self.current.options)
                        return
                    if not p.options:
                        return  # 메뉴를 따로 다시 읽으면 그 메뉴로 다시 판단 (redo_same_menu, _decide_heard)
                    # 입력 오류 안내와 같은 메뉴 재낭독이 한 단락('입력 시간이 지났습니다. ○○는 1번 …'): 다음 단락을 기다리지 않고
                    # 이 메뉴로 바로 다시 판단 (기다리면 ARS 가 다시 시간 초과로 넘어가 끊김). 같은 단계 2회 연속이면 위에서 재질문
                    log.info("입력 오류 안내 뒤 같은 단락의 메뉴(선택지 %d개)로 다시 판단", len(p.options))
                    # 같은 안내를 다시 읽은 것이므로 키를 누른 안내의 종류(메뉴·확인 질문)를 따름. 확인 질문을 메뉴로 판단하면
                    # 확인 답(answer_confirm) 대신 점수로 고르고 확인 질문이 메뉴 계층에 쌓임
                    p.kind = (key_menu.kind if key_menu is not None and key_menu.kind in ("menu", "confirm")
                              else ("confirm" if _CONFIRM_RE.search(text) else "menu"))
                    self.current_fresh = True
                    self.since_key_texts = []
        if p.kind in _STREAM_END_KINDS and self._streamed_open():
            # 종료 키 없는 고정 자릿수 항목은 마지막 자리를 받으면 ARS 가 스스로 넘어감. 사용자가 입력 완료를 누르기 전에
            # 결과·종료·상담원 안내가 오면 요청을 닫고 그 안내를 처리 (열어 두면 결과를 버리고 user_input_timeout 으로 끝남).
            # 일반 안내(info)로는 닫지 않음: 입력 도중의 짧은 안내로 닫으면 다음 키가 거절됨
            log.info("한 자씩 입력 뒤 ARS 가 다음 단계(%s)로 넘어감. 입력 요청을 닫는다", p.kind)
            self._close_request()
            self.set_status(self.idle_status())
        if p.kind == "agent_human":
            if self.agent_active or self.status == "waiting_agent":
                await self._handoff()
            return
        if p.kind in ("menu", "confirm"):
            self.current = p
            if self.pending and self.pending["kind"] == "secure_input":
                # 입력을 기다리던 ARS 가 입력 없이 메뉴로 돌아간 경우. 민감정보 입력 요청을 닫고 메뉴 판단
                log.info("민감정보 입력 대기 중 ARS 가 메뉴로 넘어감. 입력 요청을 닫는다")
                self._drop_stream_buf("ARS가 다음 안내로 넘어가 누르신 번호를 보내지 않았습니다. 입력 안내가 다시 나오면 다시 입력해 주세요.")
                self._close_request()
                self.secure_window = False
                self.set_status(self.idle_status())

        if not self.agent_active:
            return  # 업무 완료(agent_done) 뒤의 안내는 자막만 전송

        # 이하 AI 탐색 중
        if self._handoff_open() and p.kind in ("result", "message_sent", "out_of_hours", "rejected", "agent_wait"):
            # 업무 목적 입력 요청은 메뉴와 관계없이 열림(키를 누른 직후 등). 결과·종료·상담원 대기 안내가 오면 요청을 닫고
            # 그 안내를 처리 (열어 두면 안내가 버려지고, 상담원 대기 중에 답·만료로 탐색이 다시 시작됨)
            log.info("업무 목적 입력 중 %s 안내: 입력 요청을 닫고 안내를 처리한다", p.kind)
            self._close_request()
            self.handoff_prev = None
            self.set_status(self.idle_status())
        if self.pending is not None:
            return  # 사용자 응답 대기 중 ARS 가 안내를 다시 읽는 경우. 계속 청취.
        if (self.key_queue or requeue) and p.kind == "menu":
            if not self.hangup_seen:
                self._settle_rejected_layer(p)
                self.redo_same_menu = False  # _decide_heard 를 거치지 않으므로 여기서 정리 (남으면 다음 단계의 입력 오류 횟수가 이어짐)
                if requeue:
                    log.info("복귀 대기 키 %s 가 받아들여지지 않아 같은 메뉴에서 다시 누른다", requeue)
                    self.layers.pop()  # 그 키가 쌓은 계층 (다시 누르면 다시 쌓임)
                    key = requeue
                else:
                    key = self.key_queue.pop(0)
                try:
                    await self._press(key, "ai", p)
                    self.queued_key = key  # 전송 뒤 기억 (_send_keys 는 새 키마다 앞의 대기 키를 지움)
                except ApiError as e:
                    log.warning("이전 계층 복귀 키 전송 실패: %s", e.message)
            return
        if (timeout_notice and pending_dead_end is not None and getattr(interp, "dead_end_on_timeout", False)
                and pending_dead_end[1] == self.last_key_at and time.monotonic() >= self.ai_hold_until
                and self._dead_end_candidate(pending_dead_end[0])):
            # 막다른 안내 후보의 판단 대기 중에 우리 키 없이 ARS 가 입력 대기 시간 경과·무입력 안내를 함: ARS 가 그 안내에서
            # 입력을 기다렸다는 뜻 (판단 대기의 목적이 확인됨). 앞 후보를 즉시 판단 (ARS 가 곧바로 시간 초과 안내를 시작하면
            # 발화 중이라 _dead_end_check 가 판단을 미루다가 이 안내 수신으로 취소되므로 여기서 판단)
            log.info("막다른 안내 후보 뒤 우리 키 없이 무입력 안내가 옴: 앞 후보를 지금 판단한다")
            if not await self._dead_end(pending_dead_end[0]):
                await self._on_result()
            return
        if self._dead_end_candidate(p) and not (stale or cut or cut_tail):
            # 즉시 판단하지 않음: ARS 가 잠시 쉰 뒤 메뉴·결과를 이어 말할 가능성. 잠시 새 안내가 없으면 판단
            # (사용자가 직접 누른 직후 보류 시간이면 보류가 끝난 뒤)
            wait = max(getattr(self.s, "dead_end_wait_s", DEAD_END_WAIT_S), self.ai_hold_until - time.monotonic() + 0.2)
            seq = self.para_seq
            self._dead_end_pending = (p, self.last_key_at)
            self.timer("dead_end", wait, lambda: self._dead_end_check(p, seq))
            return
        if p.kind in ("menu", "confirm"):
            if self.hangup_seen or time.monotonic() < self.ai_hold_until:
                return  # 끊긴 뒤에 인식한 메뉴이거나, 사용자가 직접 누른 직후 보류 시간 (보류가 끝나면 가장 최근 메뉴로 판단)
            await self._decide_heard(p)
        elif p.kind == "result":
            if self.nav is not None and self.nav.agent_seek and AGENT_LABEL_RE.search(self.goal or ""):
                # 상담원을 찾으러 업무 메뉴로 들어간 뒤 결과 안내: 업무 완료로 끝내지 않음 (사용자는 상담원 연결을 원함). 뒤이어 상담원
                # 선택지가 있는 메뉴가 오면 그것을 판단하고, 잠시 새 안내가 없으면 상담원 연결을 찾지 못했다고 질문
                log.info("상담원 목적인데 업무 부분으로 들어간 갈래에서 결과 안내: 업무 완료로 보지 않고 다음 안내를 기다린다")
                seq = self.para_seq
                self.timer("agent_seek_result", AGENT_SEEK_RESULT_WAIT_S, lambda: self._agent_seek_result_check(p, seq))
            elif (self.current is None and not self.layers and not self.path and self.last_key_at is None
                  and self.status in ("listening", "selecting")):
                # 첫 메뉴를 듣기 전, 어떤 키(메뉴 키·민감정보 입력·사용자 키)도 보내기 전의 결과: 인사말 뒤의 이용 방법 안내
                # ('진료 예약 및 변경은 홈페이지 또는 모바일 앱으로도 …')를 목적의 답으로 오판할 가능성. 바로 확정하지 않고,
                # ARS 가 이어서 말하면(메뉴 등) 그 안내로 판단. 메뉴 없이 잠시 조용하거나 ARS 가 끊으면 결과로 판정.
                # 상담원 대기 중(waiting_agent)에 들은 결과는 바로 확정 (보류를 막는 상태가 이미 있음)
                log.info("첫 메뉴 전의 결과 안내: ARS 가 이어서 말하는지 보고 확정한다")
                self.early_result_wait = True
                self.early_end_kind = None  # 앞서 보류한 끝맺음이 있어도 가장 최근 안내(결과)로 확정
                self.result_upto = len(self.captions)  # 이 결과 단락의 자막까지 (자막은 판단 앞에서 쌓임)
                self.timer("early_result", EARLY_RESULT_WAIT_S, self._early_result_check)
            else:
                await self._on_result()
        elif p.kind in ("message_sent", "out_of_hours", "rejected"):
            if p.kind == "out_of_hours":
                self.hours_note = hours_sentence(text)
            if (getattr(interp, "hold_early_end", False) and self.current is None and not self.layers and not self.path
                    and self.last_key_at is None and self.status in ("listening", "selecting") and not self.hangup_seen):
                # 첫 메뉴 전·키 전송 전의 끝맺음 판단: 첫 메뉴 전 결과 보류와 같은 방식으로 잠시 대기. 인사말의 혼잡·연결 지연
                # 안내를 업무시간 외로 오판할 가능성 있음. ARS 가 이어서 메뉴 등을 말하면 그 안내로 판단하고, 메뉴 없이 조용하거나
                # 끊으면 이 사유로 종료 (_early_result_check, _after_hangup). 끊긴 뒤에 인식한 끝맺음은 기다릴 것이 없어 바로 종료
                log.info("첫 메뉴 전의 끝맺음 안내(%s): ARS 가 이어서 말하는지 보고 확정한다", p.kind)
                self.early_result_wait = True
                self.early_end_kind = {"rejected": "ars_rejected"}.get(p.kind, p.kind)
                self.timer("early_result", EARLY_RESULT_WAIT_S, self._early_result_check)
                return
            await self.finish({"rejected": "ars_rejected"}.get(p.kind, p.kind), hangup=not self.hangup_seen)
        elif p.kind == "agent_wait":
            self.set_status("waiting_agent")
        elif p.kind == "voice_request" and not self.hangup_seen:
            # [검증 필요] 전환 문구 음성 송출(TTS) 미구현. 사용자에게 알리고 응답 수신.
            await self._ask(p, "unknown_prompt", p.options,
                            "ARS 가 말로 대답하라고 합니다. 음성으로 답하는 기능은 아직 지원하지 않습니다. "
                            "번호로 고를 수 있으면 골라 주세요.")

    def _result_captions(self) -> list[str]:
        """결과 요약 입력: 이번 탐색의 자막 중 메뉴·확인 질문(선택지 목록으로 줄인 자막)을 뺀 안내.
        (막다른 안내를 결과로 요약할 때 지나온 메뉴 자막 '… 1번 · …' 을 보고 '1번을 선택해야 함' 처럼 잘못 적는 문제 방지.)
        남는 것이 없으면 전체 사용."""
        # 첫 메뉴 전 결과 보류로 확정한 결과는 그 결과 단락까지만 사용 (뒤에 온 '이용해 주셔서 감사합니다' 가 원문 대체가 되지 않도록)
        caps = self.captions[self.agent_goal_at_caption:self.result_upto]
        return [c["text"] for c in caps if id(c) not in self._choice_caps] or [c["text"] for c in caps]

    def _result_fallback(self) -> dict:
        """AI 요약을 쓸 수 없을 때의 결과 요약: 마지막 결과 안내 원문 (명세 9.3)."""
        caps = self._result_captions()
        return {"summary": caps[-1] if caps else "", "key_info": []}

    async def _handoff(self) -> None:
        # 상담원 연결을 먼저 통지 (앱 마이크 중계 즉시 시작). 상황 설명 문장은 AI 응답(최대 12초)을 기다린 뒤
        # summary 로 전송. 설명 문장 생성 중 끊기면 _after_hangup 이 기본 문장으로 summary 전송
        self.agent_active = False
        self.pending = None
        self.set_status("connected_to_agent")
        texts = [p["text"] for p in self.path]
        script = (await self._ai("handoff_script", self.d.interp.handoff_script(self.goal, texts),
                                 self._handoff_fallback()) if self.goal else None)
        if self.ended or self.summary is not None:
            return
        self.summary = {"kind": "handoff", "summary": "상담원 연결까지 진행했습니다.", "handoff_script": script}
        self.publish("summary", self.summary)

    def _handoff_fallback(self) -> str | None:
        return f"{self.goal} 건으로 연락드렸습니다." if self.goal else None

    # 판단 ------------------------------------------------------------------
    def _reprompt_after_error(self, p: Prompt, text: str) -> Prompt:
        """입력 오류로 분류된 단락에 ARS 의 다음 요구가 이어질 때 규칙으로 그 내용을 살림 (해석기와 무관하게 같은 기준).
        해석기가 입력 오류로만 답하면 입력 창도, 같은 메뉴 재시도도 없이 ARS 가 시간 초과를 되풀이하다 끊는 문제 방지.
        - 민감정보 입력 안내: 강한 입력 신호(끝맺음 키 입력 안내 또는 'N자리')가 있고 번호 선택지 말('1번')이 없을 때만,
          해석기 선택지가 없고 열린 입력 요청이 없으면 입력 요청으로 처리. 직전 키가 입력값이면 다른 항목의 안내일 때만
          (같은 항목이면 입력 오류 처리에서 입력 거부 알림 뒤 같은 항목 재요청)
        - 메뉴 재낭독: 해석기가 선택지를 비운 경우, 방금 키를 누른 메뉴와 같으면 그 메뉴의 선택지(두 자리 키 포함)를,
          아니면 규칙이 찾은 선택지를 붙임 (종류는 입력 오류 그대로, 종료 안내가 든 단락 제외)
        - 중요 선택: 같은 메뉴 재낭독이면 직전 메뉴의 판정을 물려받고, 규칙으로 붙인 메뉴는 원문으로 계산"""
        r = rule_classify(text)
        norm = rule_keys(text)
        if r.kind == "sensitive":
            body = _NAV_PHRASE.sub(" ", norm)
            strong = bool(_TERMINATED_INPUT.search(body) or "자리" in norm) and not _OPTION_WORD_RE.search(body)
            if not strong or p.options or (self.pending and self.pending["kind"] == "secure_input"):
                return p
            if self.last_send_secure and (not r.label or _same_field(r.label, self.secure_req[0] or self.secure_label)):
                return p  # 보낸 항목과 같은 항목의 재안내: 입력 오류 처리에서 입력 거부 알림 뒤 같은 항목 재요청
            log.info("입력 오류 안내 뒤 민감정보 입력 안내(%s)가 이어져 입력 요청으로 처리", r.label or "항목 이름 없음")
            return r
        src = None if p.options else self._reread_menu(r, norm, text)
        if src is not None:
            menu, how = src
            log.info("입력 오류 안내 뒤 메뉴 재낭독: %s 선택지 %d개를 붙임", "직전 메뉴" if how == "current" else "규칙",
                     len(menu.options))
            p.options = [copy.copy(o) for o in menu.options]
            p.back_key = p.back_key or menu.back_key
            p.home_key = p.home_key or menu.home_key
            p.repeat_key = p.repeat_key or menu.repeat_key
            p.agent_key = p.agent_key or menu.agent_key
            if how == "rule":
                p.critical = p.critical or menu.critical or _rule_critical(norm)
        if p.options and self.current is not None and self._same_menu_as_current(p.options, norm, text):
            p.critical = p.critical or self.current.critical
        return p

    def _same_menu_as_current(self, options: list[Option], norm: str, text: str = "") -> bool:
        """선택지가 직전 메뉴(self.current)를 다시 읽은 것인지: 번호가 모두 직전 메뉴에 있고, 직전 메뉴 선택지 이름의 절반 이상이
        단락에 들어 있음 (번호만 맞추면 다른 메뉴와 섞임). 해석기가 이름을 원문과 다르게 적은 경우를 위해, 두 단락의 규칙 추출
        선택지가 같은 메뉴(번호 같음, 이름 70% 이상 같음)인지도 봄."""
        cur = self.current
        if cur is None or not cur.options or not options:
            return False
        keys = {o.key for o in cur.all_options()}
        if not {o.key for o in options} <= keys:
            return False
        flat = re.sub(r"\s", "", norm)
        names = [re.sub(r"\s", "", o.label) for o in cur.options if o.label]
        if names and sum(1 for n in names if n and n in flat) * 2 >= len(names):
            return True
        return bool(text) and self._same_rule_menu(text, cur.text)

    @staticmethod
    def _same_rule_menu(a: str, b: str) -> bool:
        """두 단락의 규칙 추출 선택지가 같은 메뉴인지 (입력 오류 문장은 빼고 추출)."""
        ea, eb = extract_prompt(_ERR_SENT_RE.sub(" ", a or "", count=1)), extract_prompt(_ERR_SENT_RE.sub(" ", b or "", count=1))
        return bool(ea.options) and _same_menu(ea, eb)

    def _menu_matches(self, p: Prompt, menu: Prompt, text: str) -> bool:
        """p(방금 들은 단락)가 menu 를 다시 읽은 것인지."""
        return _same_menu(p, menu) or self._same_rule_menu(text, menu.text)

    def _same_input_reprompt(self, text: str) -> bool:
        """단락이 지금 입력 요청과 같은 항목의 입력 안내인지 (규칙 분류가 입력 요청이고, 항목 이름이 없거나 같은 항목)."""
        r = rule_classify(text)
        return r.kind == "sensitive" and (not r.label or _same_field(r.label, self.secure_req[0] or self.secure_label))

    def _key_layer(self) -> Layer | None:
        """마지막 키가 쌓은 계층 (그 뒤 계층이 바뀌지 않았을 때만)."""
        layer = self.last_press_layer
        return layer if layer is not None and self.layers and self.layers[-1] is layer else None

    def _rereads_key_menu(self, p: Prompt, text: str) -> bool:
        """p 가 마지막 키를 누른 안내(계층이 있으면 그 메뉴, 없으면 현재 안내)를 다시 읽은 것인지."""
        layer = self._key_layer()
        key_menu = layer.prompt if layer is not None else self.current
        return key_menu is not None and self._menu_matches(p, key_menu, text)

    def _own_key_menu(self, p: Prompt, text: str) -> bool:
        """p 가 방금 서버가 키를 누른 메뉴를 다시 읽은 단락인지 (키에 대한 첫 반응이 아니어도). 키를 누른 뒤 다른 메뉴·확인 질문을
        듣지 않았고(현재 메뉴가 그 메뉴 그대로), AI 가 탐색 중이며(업무 완료 뒤 아님), 사용자 요청이 열려 있지 않고, 민감정보 입력값
        전송 대기 중이 아닐 때만. 오래전에 받아들여진 키·결과를 들은 뒤의 키를 거부로 세지 않도록 제한."""
        layer = self._key_layer()
        return (self.agent_active and self.pending is None and not self.secure_queued and layer is not None
                and self.current is layer.prompt and bool(p.options) and self._menu_matches(p, layer.prompt, text))

    def _settle_rejected_layer(self, p: Prompt) -> None:
        """거부로 보인 키의 계층: 같은 메뉴를 다시 들었으면 되돌리고, 다른 메뉴면(키가 받아들여짐) 그대로 둠."""
        layer, self.rejected_layer = self.rejected_layer, None
        if layer is not None and self.layers and self.layers[-1] is layer and self._menu_matches(p, layer.prompt, p.text):
            self.layers.pop()

    def _reread_menu(self, r: Prompt, norm: str, text: str) -> tuple[Prompt, str] | None:
        """입력 오류 단락 속 메뉴 재낭독의 선택지 출처: (직전 메뉴, 'current') 또는 (규칙 추출, 'rule'). 없으면 None."""
        if _ENDING_RE.search(norm):
            return None
        ext = extract_prompt(_ERR_SENT_RE.sub(" ", text, count=1))  # 두 자리 키(10번)까지 뽑는 규칙 추출 (입력 오류 문장 제외)
        cand = ext if ext.kind in ("menu", "confirm") and ext.options else (
            r if r.kind in ("menu", "confirm") and r.options else None)
        if cand is not None and self._same_menu_as_current(cand.options, norm, text):
            return self.current, "current"
        if cand is None and self.current is not None and self.current.options:
            # 두 자리 키를 한국어로 읽는 등 규칙이 선택지를 못 찾은 경우: 직전 메뉴의 번호를 말했거나 누르라는 말로 끝나고,
            # 직전 메뉴 이름이 단락에 있을 때만 직전 메뉴로 봄 (이름만 나오는 안내 제외)
            keys = {o.key for o in self.current.all_options()}
            said = set(re.findall(r"(?<![0-9])([0-9]{1,2})\s*번(?!호)", norm))
            if (keys & said or _PRESS_END.search(norm.strip())) and self._same_menu_as_current(self.current.options, norm):
                return self.current, "current"
            return None
        if cand is None:
            return None
        as_menu = cand if cand.kind == "menu" else dataclasses.replace(cand, kind="menu")
        return (cand, "rule") if rule_menu(as_menu, norm) else None

    async def _bounced_notice(self, p: Prompt, text: str) -> Prompt:
        """AI 가 키를 누른 뒤 ARS 가 안내 한마디와 함께 이미 지나온 메뉴로 되돌려 보내면(같은 단락), 그 안내가 목적에 대한 답인지 먼저
        판단 (예: '지금은 상담 업무 시간이 종료되어 상담을 해드릴 수가 없습니다 …' 뒤에 첫 메뉴를 다시 읽는 경우, 같은 경로를
        한 번 더 누른 뒤 반복 감지로 멈추면서 그 안내를 버리는 문제 방지. 영어 서비스의 'We are not in service now …' 도 동일).
        답이면 그 안내만 결과로 처리(안내 문장이 결과 요약에 포함), 아니거나 모르면 메뉴 그대로 판단.
        막다른 안내 판단(answers_goal)과 같은 판단 사용."""
        if not (self.agent_active and self.pending is None and self.layers and self.last_key_actor == "ai"
                and not self.after_backtrack and not self.redo_same_menu and not self.key_queue and not self.hangup_seen):
            return p
        tolerant = bool(getattr(self.d.interp, "tolerant_menu_match", False))
        if not any(_same_menu(p, layer.prompt, tolerant) for layer in self.layers):
            return p
        notice = menu_preface(text)
        fn = getattr(self.d.interp, "answers_goal", None)
        if len(re.sub(r"\s", "", notice)) < 15 or fn is None:
            return p
        layer = self.layers[-1]
        chosen = layer.prompt.label_of(layer.key)
        siblings = [o.label for o in layer.prompt.options if o.key != layer.key]
        verdict = await self._ai("answers_goal", fn(self.goal or "", notice, chosen, siblings), None)
        log.info("지나온 메뉴로 되돌아옴 (직전 선택 '%s'): 앞 안내는 %s", chosen,
                 {True: "목적에 대한 답 → 결과", False: "다른 갈래"}.get(verdict, "판단 못 함 → 메뉴로 판단"))
        if verdict is True and not self.ended and self.agent_active and self.pending is None:
            # 자막·결과 요약에는 안내 문장만 사용 (뒤에 다시 읽은 메뉴는 제외)
            return Prompt(kind="result", text=notice, shown=notice)
        return p

    def _dead_end_candidate(self, p: Prompt) -> bool:
        """막다른 안내 후보: 메뉴 키를 누른 뒤 메뉴·확인 질문을 듣지 않았고, 업무 선택지 없이 이동 키만 남은 일반 안내.
        AI 탐색 중(듣는 중·고르는 중)에만 판정. 상담원 대기·연결, 사용자 응답 대기 중에는 판정하지 않음."""
        return (p.kind == "info" and bool(p.back_key or p.home_key) and not self.current_fresh and bool(self.layers)
                and self.agent_active and self.pending is None and not self.hangup_seen
                and self.status in ("listening", "selecting"))

    async def _early_result_check(self, force: bool = False) -> None:
        """첫 메뉴 전의 결과 안내 뒤 메뉴 없이 잠시 조용하면 결과로 확정. 보류 표시(early_result_wait)가 꺼졌으면(키 전송·
        목적 다시 말하기·이미 확정) 생략. 그 사이 메뉴·확인 질문(current), 키(path·last_key_at), 입력 요청·재질문(pending),
        상담원 대기·연결(status) 이 있었으면 보류 종료. ARS 가 아직 말하는 중이거나 들은 소리를 인식하는 중이면 연기.
        끊긴 뒤면 인식 중인 단락(메뉴일 수 있음)까지 기다린 뒤 들은 안내로 결과 생성 (_after_hangup).
        force: 끊긴 뒤 제한 시간까지 기다린 _after_hangup 이 호출. 인식·말하는 중이어도 연기하지 않음."""
        async with self._lock:
            if not self.early_result_wait:
                return
            if (self.ended or not self.agent_active or self.pending is not None or self.path or self.current is not None
                    or self.last_key_at is not None or self.result_heard or self.status not in ("listening", "selecting")):
                self.early_result_wait = False
                self.result_upto = None
                self.early_end_kind = None
                return
            busy = ((self.stt_inflight or self._handed_over() or self._held_head is not None) if self.hangup_seen
                    else self._ars_active())
            if busy and not force:
                self.timer("early_result", EARLY_RESULT_WAIT_S, self._early_result_check)
                return
            self.early_result_wait = False
            if self.early_end_kind is not None:
                kind, self.early_end_kind = self.early_end_kind, None
                await self.finish(kind, hangup=not self.hangup_seen)
                return
            await self._on_result(held=True)

    async def _agent_seek_result_check(self, p: Prompt, seq: int) -> None:
        """상담원을 찾던 갈래의 결과 안내 뒤 잠시 새 안내가 없으면 사용자에게 질문 (업무 완료로 끝내지 않음)."""
        async with self._lock:
            if seq != self.para_seq or self.ended or self.hangup_seen or not self.agent_active or self.pending is not None:
                return
            if self._ars_active():
                self.timer("agent_seek_result", AGENT_SEEK_RESULT_WAIT_S, lambda: self._agent_seek_result_check(p, seq))
                return
            self.result_heard = True  # 사용자가 여기서 끝내거나 ARS 가 끊으면 들은 안내로 요약 (completed)
            await self._ask(p, "unknown_prompt", [], message=AGENT_SEEK_RESULT_ASK)

    async def _dead_end_check(self, p: Prompt, seq: int) -> None:
        """막다른 안내 후보 수신 뒤 잠시 새 안내가 없으면 판단. 그 사이 새 단락이 오면(para_seq) 판단 생략."""
        async with self._lock:
            if seq != self.para_seq or self.ended or not self._dead_end_candidate(p):
                return
            if time.monotonic() < self.ai_hold_until or self._ars_active():
                # 아직 발화 중이거나 사용자 직접 키 보류 중이면 잠시 더 대기
                wait = max(getattr(self.s, "dead_end_wait_s", DEAD_END_WAIT_S), self.ai_hold_until - time.monotonic() + 0.2)
                self.timer("dead_end", wait, lambda: self._dead_end_check(p, seq))
                return
            self._dead_end_pending = None
            if not await self._dead_end(p):
                await self._on_result()

    async def _dead_end(self, p: Prompt, after_answer: bool = False) -> bool:
        """막다른 안내: 메뉴 키를 누른 뒤 업무 선택지 없이 '이전 단계'·'처음으로' 같은 이동 키만 남은 안내.
        (예: "… 휴대전화로 다시 이용하시거나 앱을 이용하세요. 전 단계로 돌아가시려면 우물정자" 를 일반 안내로 보고
        아무 처리도 하지 않으면 ARS 가 입력을 기다리다 종료.) 해석기에 목적에 대한 답인지 질의:
        - 답이면 결과 안내로 전환 (False 를 반환하고 호출한 쪽이 결과로 처리: p.kind='result').
          단, 목적 낱말이 지나온 메뉴·안내에 없고 다른 선택지 하나의 이름에 있으면(better_sibling) 결과로 보지 않고 질문
          (예: 목적과 다른 발급 갈래의 본인인증 안내를 모델이 답으로 판단하는 경우). 낱말 규칙은 같은 뜻의
          다른 말을 모르므로 질문까지만 하고 되돌아가지는 않음. 사용자가 고른 갈래이거나 재질문에 글로 답한 뒤에는 적용하지 않음
        - 다른 갈래면 이전 메뉴로 복귀. 돌아간 메뉴에서 이번에만 그 갈래를 빼고 선택 (거절 목록에는 넣지 않음:
          판단이 틀렸을 때 사용자가 다시 고를 수 있도록). 사용자가 직접 고른 갈래이거나, 이번 탐색에서 이미 되돌아간 적이 있으면
          (막다른 안내·맞는 선택지 없음 모두. 탐색당 자동 되돌아가기 1회) 되돌아가지 않고 질문
        - 모르면(판단 실패 포함) 이동 키를 선택지로 사용자에게 질문
        True 면 여기서 처리 완료."""
        fn = getattr(self.d.interp, "answers_goal", None)
        layer = self.layers[-1]
        chosen = layer.prompt.label_of(layer.key)
        siblings = [o.label for o in layer.prompt.options if o.key != layer.key]
        text = " ".join(self.since_key_texts) or p.text
        # 마지막 키가 지난 메뉴의 선택지로 기록되지 않은 사용자 키(새 메뉴를 듣기 전 키패드 입력, 민감정보 입력)였으면
        # 지나온 계층이 지금 ARS 위치와 불일치. 마지막 계층의 메뉴로 판단하지 않고 질문
        blind = self.last_key_actor == "user" and layer.actor != "user"
        user_branch = layer.actor == "user" or blind
        verdict = None
        if fn and not blind:
            verdict = await self._ai("answers_goal", fn(self.goal or "", text, chosen, siblings), None)
        if self.ended or self.hangup_seen or self.pending is not None or not self.agent_active:
            return True
        better = None
        if verdict is True and not user_branch and not after_answer:
            path = " ".join(x.prompt.label_of(x.key) for x in self.layers)  # 지나온 메뉴 이름 모두 (윗단계에서 고른 대상 포함)
            better = better_sibling(self.goal or "", path, text, siblings)
            if better is not None:
                log.info("막다른 안내: 모델은 목적에 대한 답으로 판단했으나 목적 낱말이 다른 선택지 이름에 있어 사용자에게 묻는다")
                verdict = None
        if verdict is False and not user_branch and chosen_names_goal(self.goal or "", chosen):
            # 반대 경우도 동일: 모델은 다른 갈래로 판단했지만 고른 메뉴 이름이 목적의 중심 낱말을 담으면 스스로 되돌아가지 않고 질문
            log.info("막다른 안내: 모델은 다른 갈래로 판단했으나 고른 메뉴 이름이 목적 낱말을 담아 사용자에게 묻는다")
            verdict = None
        log.info("막다른 안내 (직전 선택 '%s'): %s", chosen, {True: "목적에 대한 답", False: "다른 갈래"}.get(verdict, "판단 못 함"))
        if verdict is True:
            p.kind = "result"
            return False
        if verdict is False and not user_branch and not self.nav.backtracked:
            self.nav.backtracked = True
            self.nav.exclude, self.nav.exclude_at = {layer.key}, layer.prompt.signature()
            self.after_backtrack = True  # 돌아간 메뉴 도착은 반복으로 세지 않음
            keys = p.back_key or p.home_key
            if not p.back_key:
                self.key_queue = [x.key for x in self.layers[:-1]]  # '처음으로' 뒤 지나온 키를 다시 눌러 직전 메뉴로 이동
            self.set_status("selecting", f"'{chosen}' 은 찾는 업무가 아니어서 이전 메뉴로 돌아갑니다.", src=layer.prompt)
            try:
                await self._press(keys, "ai", p)
            except ApiError as e:
                log.warning("막다른 안내에서 되돌아가기 키 전송 실패: %s", e.message)
            return True
        opts, seen = [], set()
        for k, name in ((p.back_key, "이전 메뉴"), (p.home_key, "처음으로"), (p.repeat_key, "다시 듣기")):
            if k and k not in seen:
                seen.add(k)
                opts.append(Option(k, name))
        # 힌트 속 다른 선택지 이름은 ARS 메뉴가 말한 이름 그대로
        if better:
            hint = f"찾는 업무가 '{better}' 쪽이면 이전 메뉴를 골라 주세요. "
        elif verdict is False:  # 다른 갈래로 판단했지만 사용자가 고른 갈래이거나 이미 한 번 되돌아가서 스스로 되돌아가지 않는 경우
            hint = f"'{chosen}' 은 찾는 업무와 다른 것 같습니다. "
        else:
            hint = ""
        await self._ask(p, "unknown_prompt", opts, f"이 안내에서 더 고를 메뉴가 없습니다. {hint}목적을 마쳤으면 통화를 끝내고, "
                                                   "다른 메뉴가 필요하면 이전 메뉴나 처음으로를 골라 주세요.",
                        ai_question=f"{hint}{p.text}")  # 글 답('그 메뉴로요') 해석 시 그 선택지 이름도 참고
        # 사용자가 고른 이동 키(선택·키패드)는 이 안내 기준으로 처리 (이전 메뉴면 지나온 계층에서 한 단계 제거)
        self.dead_end_rid, self.dead_end_prompt = self.pending["request_id"], p
        return True

    async def _on_result(self, held: bool = False) -> None:
        """결과 안내: AI 만 마치고 통화는 유지. 사용자가 통화를 종료하면 end_reason=completed.
        요약을 마치기 전에 ARS 가 끊어도 completed 이며, 요약을 마친 뒤 summary → ended 순서로 전송 (_after_hangup)"""
        self.result_heard = True
        self.early_result_wait = False
        self.early_end_kind = None
        if not held:
            # 첫 메뉴 전 결과 보류에서 온 확정이 아니면(상담원 대기 중에 들은 다른 결과 등) 보류한 단락으로 요약 범위를 자르지 않음
            self.result_upto = None
        self.set_status("listening_result")
        try:
            res = await self.d.interp.summarize_result(self.goal or "", self._result_captions())
        except Exception:
            log.exception("결과 요약 실패")
            res = self._result_fallback()  # 결과 안내 원문으로 대체
        if self.ended or self.summary is not None:
            return  # 요약 중 통화가 끊기고 제한 시간이 지나 _after_hangup 이 원문 요약을 이미 전송한 경우
        self.summary = {"kind": "result", "summary": res.get("summary", ""),
                        "key_info": res.get("key_info", []), "handoff_script": None}
        self.publish("summary", self.summary)
        self.agent_active = False
        self.key_queue.clear()
        if not self.hangup_seen:
            self.set_status("agent_done")  # 통화가 이미 끊겼으면 바로 ended 로 전환

    async def _decide_heard(self, p: Prompt) -> None:
        """새로 들은 메뉴·확인 질문 판단.
        입력 오류 뒤 같은 메뉴, 이전 계층 복귀로 도착한 메뉴는 반복으로 세지 않음."""
        redo = self.redo_same_menu
        if redo and self.redo_menu is not None and not self._menu_matches(p, self.redo_menu, p.text):
            # 거부로 보았던 키가 실제로는 받아들여졌고 다른 메뉴가 왔음: 계층 유지, 새로 도착한 메뉴로 판단
            redo, self.input_errors, self.rejected_layer = False, 0, None
        self._settle_rejected_layer(p)
        user_redo = redo and self.redo_by_user and self.input_errors > 0
        arrived = not (redo or self.after_backtrack)
        self.redo_same_menu = self.after_backtrack = False
        self.redo_by_user, self.redo_menu = False, None
        if user_redo:
            # 사용자가 고른 키가 거부된 뒤의 같은 메뉴: AI 가 다시 고르지 않고 사용자에게 다시 물음
            # (AI 가 다른 갈래를 누르거나, 이 계층의 재질문을 이미 써서 not_found 로 끝나는 문제 방지)
            if self.agent_active and self.pending is None and not (self.ended or self.hangup_seen):
                await self._ask(p, "critical_action" if p.critical else "input_rejected", p.options)
            return
        await self._decide(p, arrived)

    async def _decide(self, p: Prompt, arrived: bool) -> None:
        # 글 답으로 목적이 바뀐 표시는 이번 판단에만 사용 (일찍 끝나는 경로·확인 질문에서도 지워, 나중의 다른 메뉴 판단에 남지 않도록)
        revised, self.answer_revised = self.answer_revised, False
        if not self.agent_active or self.pending is not None or self.ended or self.hangup_seen or self.nav is None:
            return
        # 메뉴·확인 질문 판단 시작 시 selecting (명세 3.1).
        # 결과에 따라 listening(키 전송)·awaiting_user(재질문)·ended(not_found) 로 전환
        if p.kind != "confirm":
            self.set_status("selecting")  # 확인 질문은 답을 정한 뒤에만 selecting (대부분 사용자 질문으로 이어지므로 상태 깜빡임 방지)
        try:
            await self._decide_inner(p, arrived, revised)
        except ApiError as e:
            log.warning("키 전송 실패: %s", e.message)  # 통화가 끊기는 중. 종료 처리는 _on_end 에서 수행

    async def _decide_inner(self, p: Prompt, arrived: bool, revised: bool = False) -> None:
        if p.kind == "confirm":
            if p.critical:  # 결제·계약·해지 최종 승인 등은 확인 질문이어도 키 전송 금지 (명세 1.1 원칙 5)
                await self._ask(p, "critical_action", p.options)
                return
            key = await self._ai("answer_confirm", self.d.interp.answer_confirm(self.goal or "", p), None)
            if self.ended or self.hangup_seen:
                return  # 판단 중 통화 끊김
            if key:
                self.set_status("selecting")
                await self._press(key, "ai", p)
            elif self._som:
                # som 방식: 검증된 대안이 둘 이상일 때만 질문, 아니면 시스템 재판단·다시 듣기 복구
                route = await self._ai("confirm_route", self.d.interp.assess_route(
                    self.goal or "", p, p.options, [layer.prompt.category_of(layer.key) for layer in self.layers]), None)
                if self.ended or self.hangup_seen or self.pending is not None or not self.agent_active:
                    return
                relevant = [o for o in p.options if o.key in getattr(route, "clarification_keys", ())]
                if getattr(route, "status", None) == "needs_info" and len(relevant) >= 2:
                    await self._ask(p, "ambiguous", relevant, message=f"{' / '.join(o.label for o in relevant)} 중 어떤 쪽인가요?")
                else:
                    await self._recover_system(p)
            else:
                await self._ask(p, "unknown_prompt", p.options)
            return
        # 사용자가 고른 갈래인지: 마지막 계층을 사용자가 골랐거나, 마지막 키가 계층에 기록되지 않은 사용자 키(새 메뉴를 듣기 전 키패드
        # 입력)이면 참 (막다른 안내 판단 _dead_end 와 같은 기준). 글로 답해 목적이 바뀐 뒤의 재판단인지도 전달
        user_branch = bool(self.layers) and (self.layers[-1].actor == "user" or self.last_key_actor == "user")
        try:
            dec = await self.d.navigator.decide(self.nav, p, self.layers, arrived, revised=revised, user_branch=user_branch)
        except Exception:
            log.exception("AI 모듈 score 실패. 사용자에게 묻는다")
            if not (self.ended or self.hangup_seen):
                if self._som:
                    await self._recover_system(p)   # som 방식: 사용자에게 묻지 않고 시스템 복구
                else:
                    await self._ask(p, "unknown_prompt", p.options,
                                    "안내를 해석하지 못했습니다. 원하시는 메뉴를 말씀하거나 직접 골라 주세요.")
            return
        if self.ended or self.hangup_seen:
            return  # 판단 중 통화 끊김
        log.info("판단 %s %s %s", dec.action, dec.key or dec.keys or dec.reason, dec.note)
        if dec.action == "press":
            self.nav.presses += 1
            self.nav.exclude.clear()
            self.nav.exclude_at = None
            label = p.label_of(dec.key)
            # 이름이 '…메뉴' 로 끝나면 '메뉴' 를 덧붙이지 않음 ("'폰뱅킹 전용 메뉴' 메뉴를" 같은 중복 표시 방지)
            self.set_status("selecting", f"'{label}'를 선택합니다." if label.strip().endswith("메뉴")
                            else f"'{label}' 메뉴를 선택합니다.", src=p)
            await self._press(dec.key, "ai", p, confidence=dec.scores.get(dec.key))
        elif dec.action == "ask":
            await self._ask(p, dec.reason, dec.options, message=dec.message or None, scores=dec.scores)
        elif dec.action == "recover":
            # som 방식 탐색기만 내는 판단. 반복·키 한도면 재판단 없이 다시 듣기 복구만
            await self._recover_system(p, reanalyze=dec.reason != "navigation_limit")
        elif dec.action == "backtrack":
            # '이전 메뉴' 키를 누르면 _press 가 마지막 계층 제거.
            # '처음으로' 키를 쓰는 경우 지나온 키를 key_queue 로 다시 누르며 계층 재구성.
            self.nav.backtracked = True
            self.nav.exclude = {self.layers[-1].key}
            self.nav.exclude_at = self.layers[-1].prompt.signature()
            self.after_backtrack = True
            keys = dec.keys
            if keys[0] == p.home_key and len(keys) > 1:
                self.key_queue = list(keys[1:])
                keys = keys[0]
            self.set_status("selecting", "맞는 메뉴가 없어 이전 메뉴를 확인합니다.")
            await self._press(keys, "ai", p)
        elif dec.action == "end":
            if self._som and dec.reason == "not_found":
                self.set_status("selecting", "말씀하신 업무로 진행할 수 있는 메뉴를 찾지 못했어요. 통화를 종료할게요.")
            await self.finish(dec.reason, note=dec.note)

    async def _recover_system(self, p: Prompt, *, reanalyze: bool = True) -> None:
        """som 방식 전용 복구. AI·STT 실패에 사용자 목적을 묻지 않고 키를 추측하지 않음.
        관찰한 메뉴마다 내부 재판단 1회와 검증된 다시 듣기 1회, 계층당 공유 한도. 계속 실패하면 듣는 중으로 유지."""
        if (self.ended or self.hangup_seen or not self.agent_active or self.pending is not None
                or p is not self.current or not self.current_fresh):
            return
        key = (self.goal, tuple((layer.prompt.text, layer.key) for layer in self.layers))
        budget = self._system_recovery_budget.for_prompt(key, p)
        if budget is None:
            self.set_status("listening", "다음 안내를 기다리고 있어요.")
            return
        if reanalyze and not budget.analyzed:
            budget.analyzed = True
            self.set_status("selecting", "안내와 이전 경로를 다시 확인하고 있어요.")
            start = time.monotonic()
            try:
                fixed = await self.d.interp.recover_context(p, self.goal or "",
                                                            [layer.prompt.category_of(layer.key) for layer in self.layers])
            except Exception as exc:
                log.warning("시스템 재판단 실패: %s", type(exc).__name__)
                fixed = p
            if self.ended or self.hangup_seen or self.pending is not None or not self.agent_active:
                return
            log.info("시스템 재판단 %.0fms", (time.monotonic() - start) * 1000)
            if fixed is not p and fixed.options:
                fixed.display_fixes = display_fixes_for(fixed)   # 화면용 이름 다듬기 (분류 때와 같음)
            self.current = fixed
            if self.recorder is not None:
                self.recorder.note("system_reanalysis", duration_s=round(time.monotonic() - start, 3),
                                   kind=fixed.kind, options=[f"{o.key}:{o.label}" for o in fixed.options])
            if fixed.kind in ("menu", "confirm"):
                await self._decide_inner(fixed, arrived=False)
                return
            p = fixed  # 재판단이 메뉴가 아니면(입력 요청·결과 등) 여기서 실행하지 않고 다음 단락 처리에 맡김
        if self._ars_active_for_recovery():
            self.set_status("listening", "새 안내를 듣고 다시 확인하고 있어요.")
            return
        if p.repeat_key and not p.critical and not budget.replayed:
            budget.replayed = True
            self.redo_same_menu = True
            self.set_status("listening", "안내를 다시 듣고 확인할게요.")
            await self._press(p.repeat_key, "ai", p)
            # _press → _send_keys 가 redo_same_menu 를 지우므로 다시 켬 (다시 들은 메뉴는 같은 메뉴의 재판단)
            self.redo_same_menu = True
            retry = getattr(self.d.interp, "retry_uncertain_after_replay", None)
            if retry:
                retry(p)
            return
        self.set_status("listening", "안내를 정확히 확인하지 못해 자동 선택을 보류했어요. 새 안내를 기다리고 있어요.")

    async def _send_keys(self, keys: str, actor: str = "ai", secure: bool = False, private: bool = False) -> float:
        """키(DTMF 톤) 전송. 키를 누르면 ARS 는 하던 안내를 끊거나 다음 안내로 넘어가므로,
        말하던 중인 단락은 버리고(앞뒤 안내가 한 자막으로 섞이지 않게) 보낸 톤의 반향은 듣지 않는다."""
        if self.call is None or self.ended or self.hangup_seen:
            raise ApiError(409, "invalid_state", "통화가 연결되지 않았습니다.")
        # 새 키: 이전 키의 거부·재시도 상태 정리 (계층은 _press 가 다시 기억). 다시 듣기 키 뒤의 redo 는 _no_response 가 다시 켬
        self.last_press_layer = self.rejected_layer = None
        self.unconfirmed_multi = None
        self.queued_key = None  # 다른 키를 보내면 앞서 복귀 중 누른 대기 키는 다시 누를 대상에서 제외 (사용자 키 뒤 거부에 끼어들지 않도록)
        self.queued_err = False
        self.redo_same_menu, self.redo_by_user, self.redo_menu = False, False, None
        self.readback_retry_armed = secure  # 민감정보를 보낸 뒤부터 다음 일반 키까지
        self.last_key_at = time.monotonic()  # 전송 시작 시각. 이보다 먼저 시작된 안내는 키 이전 상태의 안내
        self.key_cut_speech = self._ars_speaking()  # ARS 발화 도중이면 그 안내의 뒷부분이 키 뒤 새 단락으로 들림
        self.since_key_texts = []
        # 키 전송 시 ARS 는 그 안내를 벗어남. 대기 중이던 막다른 안내 판단 취소 (판단 중 되돌아가기 키는 이미 타이머 밖이라 영향 없음)
        self.cancel_timer("dead_end")
        # 첫 메뉴 전 결과 보류도 종료: 키(민감정보 입력·사용자 키 포함)를 보낸 뒤의 안내는 그 결과 단락과 무관
        if self.early_result_wait:
            self.early_result_wait = False
            self.result_upto = None
            self.early_end_kind = None
        self.cancel_timer("early_result")
        if secure:
            # 민감정보를 실제로 보내는 순간 되읽기 구간을 (다시) 엶. ARS 안내 종료를 기다렸다 보내는 동안 다른 안내
            # ('입력시간이 경과하였습니다' 재안내 등)로 구간이 닫혔을 가능성 (닫힌 채 두면 되읽기 숫자가 자막에 남음)
            self.secure_window = True
            self.secure_sent_at = self.last_key_at
        if self.recorder is not None:
            # 민감정보는 값 대신 자릿수·대체 신호음만 남김. private: 규칙 안전망이 없는 해석기의 키패드 키 (_press)
            self.recorder.keys(keys, actor, secure or private)
        try:
            dur = await self.call.send_keys(keys)
        except Exception as e:
            log.warning("키 전송 실패: %s", e)
            raise ApiError(409, "invalid_state", "키를 전송하지 못했습니다. 통화가 끊겼을 수 있습니다.")
        self.segmenter.reset()
        # 보낸 톤의 반향 무시: 반환 뒤에도 남은 톤 시간(dur) + 전화망 반향 여유. 실시간 송출은 dur=0 이므로
        # 반향 여유만 차단 (송출 시간만큼 추가로 막으면 ARS 되읽기 앞부분이 버려짐)
        self.segmenter.mute(dur + TONE_ECHO_TAIL_S)
        self.heard_since_press = False
        self.key_outstanding = True
        self.last_send_secure = False
        if secure and self._som:
            self._system_recovery_budget.clear()  # som 방식: 민감정보를 보내면 복구 예산을 새로 시작
        return dur

    def _user_left(self, layer: Layer) -> None:
        """사용자가 재질문에서 이동 키로 갈래를 벗어난 경우. 그 메뉴에서 그 갈래를 거절한 것으로 누적 (AI 가 키를 눌러도 유지.
        자동 복귀·두 번째 거절 뒤 재진입 방지). 돌아간 메뉴에서 다시 물을 수 있도록 재질문 기록을 지우고,
        돌아간 메뉴 도착은 반복으로 세지 않음 (AI 복귀와 동일)."""
        sig = layer.prompt.signature()
        self.nav.rejected.setdefault(sig, set()).add(layer.key)
        self.nav.low_asked.discard(sig)
        self.nav.sibling_asked.discard(sig)  # 같은 종류 재질문도 다시 가능하도록 (나온 갈래 말고 다른 종류를 사용자가 고르도록)
        self.after_backtrack = True

    async def _press(self, keys: str, actor: str, p: Prompt | None, confidence: float | None = None,
                     keypad: bool = False) -> None:
        # 규칙 안전망이 없는 해석기(rule_free, llm 조합)는 입력 요청을 놓치면 사용자가 키패드로 누른 번호가 민감정보일 가능성 있음.
        # 그 키는 통화 기록(키 전송·소리·경로·입력 거절)에 자릿수로만 남김. 버튼으로 고른 메뉴(select)는 그대로 남김
        private = keypad and actor == "user" and bool(getattr(self.d.interp, "rule_free", False))
        if self.recorder is not None:
            # 이어서 누른 키패드 번호(한 자씩 눌러도 이어 붙임)가 3자리 이상이면 이 뒤의 모든 기록에서 그 값을 지움
            # (막다른 안내 문구·인계 요약·되읽기 받아쓰기 등에 섞여 남지 않도록, recording.private_values)
            self.private_run = self.private_run + "".join(c for c in keys if c.isdigit()) if private else ""
            if len(self.private_run) >= 3:
                self.recorder.private_values.add(self.private_run)
        await self._send_keys(keys, actor, private=private)
        if self._som and (p is None or keys != p.repeat_key):
            # som 방식: 다시 듣기가 아닌 실제 키 입력은 복구 예산을 새로 시작
            self._system_recovery_budget.clear()
        self.current_fresh = False
        self.last_key_actor = actor
        self.last_press_layer = None
        if p is not None:
            if keys == p.back_key:
                if self.layers:
                    left = self.layers.pop()
                    if actor == "user":
                        # 사용자가 재질문에서 '이전 메뉴' 선택. 돌아간 메뉴에서 방금 나온 갈래는 다시 누르지 않음
                        self.nav.exclude = {left.key}
                        self.nav.exclude_at = left.prompt.signature()
                        self._user_left(left)
            elif keys == p.home_key:
                if actor == "user" and self.layers:
                    # '처음으로': 첫 메뉴에서 나온 갈래를 같은 방식으로 기록. 막다른 안내에서 고른 것이면 그 막다른 갈래만 기록
                    # (첫 메뉴의 갈래 전체를 거절하면 같은 갈래 안의 맞는 메뉴에 다시 도달 불가)
                    self._user_left(self.layers[-1] if p.kind == "info" else self.layers[0])
                self.layers.clear()
            elif keys != p.repeat_key and p.kind == "menu":
                self.layers.append(Layer(p, keys, actor))
                self.last_press_layer = self.layers[-1]
        self.step += 1
        label = p.label_of(keys) if p else keys
        if (p is not None and p.kind == "menu" and len(keys) >= 2 and keys.isdigit() and label != keys
                and any(o.key == keys for o in p.options)):
            # 두 자리 메뉴 키: ARS 가 앞자리만 먼저 받아 다른 메뉴로 갈 가능성 (예: 10 을 누른 뒤
            # 10번 선택지 안내 없이 다른 메뉴의 입력 시간 초과 안내). 이 선택지 이름을 듣기 전에 입력 안내가 오면 입력 요청에 경고 부착.
            # 이름을 들었는지 확인할 증거(구별 낱말 또는 세 글자 이상 이름)가 없으면 확인 생략 (늘 경고가 붙는 것 방지)
            others = [o.label for o in p.options if o.key != keys]
            words, whole = _choice_words(label, others)
            if words or len(whole) >= 3:
                self.unconfirmed_multi = (keys, label, others, p)
        if p is not None and getattr(p, "readback_of", None) and any(o.key == keys for o in p.options):
            # 입력값 되읽기 확인 단계: 경로에는 "확인" 같은 답이 아닌 확인 대상을 기록
            label = f"{p.readback_of} {'재입력' if _NO_ANSWER_RE.search(label) else '확인'}"
        names = [o.label for o in (p.options if p else [])]
        if label == keys:
            text = f"{keys} 입력"  # 안내에서 듣지 못한 키 (직접 선택)
        elif len(names) >= 2 and any(o.key == keys for o in p.options):
            # 명세 6.2: 들은 선택지 중 고른 선택지 (예: "잔액조회·거래내역조회·상담직원 연결 중 잔액조회(3) 선택")
            text = f"{'·'.join(names[:5])}{' 등' if len(names) > 5 else ''} 중 {label}({keys}) 선택"
        else:
            text = f"{label}({keys}) 선택"  # 이전 메뉴·처음으로·다시 듣기, 선택지가 하나인 안내
        sel = next((o for o in p.all_options() if o.key == keys), None) if p else None
        step = {
            "step": self.step, "text": text,
            "options": [o.as_dict() for o in (p.all_options() if p else [])],
            # 이동 키(이전 메뉴·처음으로·다시 듣기)와 안내에 없던 키는 category=null (options 와 동일)
            "selected": {"key": keys, "label": label, "category": sel.category if sel else None,
                         "confidence": confidence},
            "actor": actor,
        }
        self.path.append(step)
        if private:
            self.private_steps.add(self.step)
        self.publish("path_updated", step, src=p)
        if self.agent_active and self.pending is None:
            self.set_status("listening")  # 재질문에 select·key 로 답한 경우도 awaiting_user 해제
        if self.agent_active:
            # 무응답 복구(다시 듣기 키 재전송)는 AI 탐색 중에만 수행. 업무 완료·상담원 연결 뒤
            # 사용자가 직접 누른 키에는 자동 재전송 없음 (상담원 통화에 DTMF 끼어듦 방지).
            self.timer("no_response", self.s.no_response_s, lambda: self._no_response(p))

    async def _no_response(self, p: Prompt | None) -> None:
        """키 전송 후 일정 시간(기본 8초) 새 안내가 없을 때 (명세 7장 3항).
        그동안 ARS 가 아무 소리도 내지 않았으면 키가 인식되지 않아 같은 메뉴에서 대기 중인 것으로 보고,
        같은 키를 재전송하지 않고 키를 누른 그 메뉴의 다시 듣기 키 전송.
        소리를 냈으면(다음 안내 중이거나 인식 중) 누르지 않고 안내 종료를 대기.
        이전 메뉴의 다시 듣기 키는 다음 메뉴에서 다른 뜻일 가능성 있음."""
        async with self._lock:
            if (self.ended or self.hangup_seen or not self.agent_active or p is None
                    or not p.repeat_key or self.current_fresh or self.pending is not None):
                return
            if self.heard_since_press or self._ars_active():
                return
            if self._som:
                # som 방식: 이 메뉴의 다시 듣기 복구를 이미 썼으면 다시 듣기 키를 또 보내지 않음
                recovery_key = (self.goal, tuple((layer.prompt.text, layer.key) for layer in self.layers))
                budget = self._system_recovery_budget.for_prompt(recovery_key, p, create=False)
                if budget is not None and budget.replayed:
                    self.set_status("listening", "다시 듣기 응답을 확인하지 못해 자동 선택을 보류했어요. 새 안내를 기다리고 있어요.")
                    return
            keep, keep_q = self.unconfirmed_multi, self.queued_key  # 다시 듣기 키는 고른 선택지·복귀 대기 키를 바꾸지 않음
            try:
                await self._send_keys(p.repeat_key)
            except ApiError:
                return
            self.unconfirmed_multi, self.queued_key = keep, keep_q
            self.redo_same_menu = True  # 다시 듣기 키로 다시 나온 메뉴는 반복으로 세지 않음 (명세 7장 5항)

    # 사용자에게 보내는 요청 ---------------------------------------------------
    def _new_request(self, kind: str, expire_s: float, **fields) -> dict:
        self.req_seq += 1
        req = {"request_id": f"req_{self.req_seq}", "kind": kind, **fields, "expires_at": now_iso(expire_s)}
        self.pending = req
        return req

    async def _ask(self, p: Prompt, reason: str, options: list[Option], message: str | None = None,
                   scores: dict[str, float] | None = None, ai_question: str | None = None) -> None:
        labels = [o.label for o in options]
        scores = scores or {}
        # 확신 없음 문구에 이름을 대는 선택지는 점수가 단독 최고일 때만. 1·2위가 같은 점수면 목록 맨 앞은 선택지 목록 순서일 뿐
        # (모든 선택지가 같은 점수일 때 목록 맨 앞 메뉴가 맞는지 묻는 문구가 되는 문제 방지)
        top = (scores.get(options[0].key) or 0.0) if options else 0.0
        named = bool(options) and (len(options) == 1 or round(top - (scores.get(options[1].key) or 0.0), 6) > 0)
        if p.kind == "menu" and self.layers and reason in ("ambiguous", "low_confidence", "loop", "input_rejected"):
            # 한 단계 이상 들어온 메뉴의 재질문에는 이동 키도 선택지로 표시. 잘못 들어간 갈래에서 사용자가 돌아갈 수 있도록
            # (하위 메뉴 재질문에 '이전 메뉴' 가 없으면 복귀 불가)
            seen = {o.key for o in options}
            extra: list[Option] = []
            for k, name in ((p.back_key, "이전 메뉴"), (p.home_key, "처음으로")):
                if k and k not in seen:  # 이전 메뉴·처음으로가 같은 키면 한 번만
                    seen.add(k)
                    extra.append(Option(k, name))
            options = list(options) + extra
        default = {
            "ambiguous": f"{' / '.join(labels)} 중 어느 쪽인가요? 원하시는 업무를 말씀해 주세요.",
            # 1·2위 점수가 같으면(판단기가 어느 쪽도 고르지 못함. JEV 가 확신하지 못하면 모두 0.4) 정하지 못했다고 알림. 맞는 선택지가
            # 없다는 뜻은 아니므로 '찾지 못했다' 고 하지 않음 (2026-11-09 실통화: 같은 문구만 되풀이되어 무엇을 고르라는 건지 알 수 없었음)
            "low_confidence": (f"'{labels[0]}' 메뉴가 맞는지 확실하지 않습니다. " if named
                               else "어느 선택지가 맞는지 정하지 못했습니다. 아래에서 고르시거나 ")
                              + "원하시는 업무를 조금 더 자세히 말씀해 주세요.",
            "critical_action": "결제·계약·해지처럼 중요한 단계입니다. 진행하려면 화면에서 번호를 직접 골라 주세요.",
            "loop": "같은 메뉴가 반복되고 있습니다. 원하시는 업무를 조금 더 자세히 말씀해 주세요.",
            "unknown_prompt": p.text,  # 안내문 그대로 표시 (앞뒤 설명을 붙이거나 자르지 않음)
            "input_rejected": "번호가 계속 인식되지 않습니다. 원하시는 메뉴를 말씀하거나 직접 골라 주세요.",
        }
        req = self._new_request("clarification", self.s.clarify_expire_s, reason=reason,
                                message=message or default.get(reason, "확인이 필요합니다."),
                                options=[{**o.as_dict(), "confidence": scores.get(o.key)} for o in options],
                                ars_prompt=p.text)
        # 답 해석(merge_answer)에 쓰는 질문은 원문 유지. 요청 사전의 message 는 전송 시 화면용 받아쓰기 고침이
        # 적용될 가능성 (LLM_FIX_TRANSCRIPT=separate). 판단 입력은 원문 필요
        self.ai_questions = {req["request_id"]: ai_question if ai_question is not None else req["message"]}
        self.set_status("awaiting_user", req["message"], src=p)
        self.publish("request", req, src=p)
        rid = req["request_id"]
        self.timer("clarify_expire", self.s.clarify_expire_s, lambda: self._clarify_expire(rid))
        self.timer("clarify_drop", self.s.clarify_drop_s, lambda: self._clarify_drop(rid))

    async def _clarify_expire(self, rid: str) -> None:
        """30초: ARS 가 통화를 끊지 않도록 지금 메뉴의 다시 듣기 키를 1회 전송. 요청은 유지.
        ARS 안내 중이면 안내 종료 뒤 전송 (안내 중에 누르면 안내가 끊기고 자막이 섞임)."""
        async with self._lock:
            if not (self.pending and self.pending["request_id"] == rid) or self.ended or self.hangup_seen:
                return
            if self._ars_active():
                self.timer("clarify_expire", 0.5, lambda: self._clarify_expire(rid))
                return
            # 막다른 안내 재질문이면 그 안내의 다시 듣기 키만 전송. 지난 메뉴의 다시 듣기 키는 막다른 안내에서 다른 뜻일 가능성
            # ('#' 이 지난 메뉴의 다시 듣기이고 막다른 안내의 처음으로인 경우 등)
            dead = self._dead_end_open()
            key = (dead.repeat_key if dead is not None else (self.current.repeat_key if self.current else None))
            if key:
                keep, keep_q = self.unconfirmed_multi, self.queued_key  # 다시 듣기 키는 고른 선택지·복귀 대기 키를 바꾸지 않음
                try:
                    await self._send_keys(key)
                except ApiError:
                    pass
                self.unconfirmed_multi, self.queued_key = keep, keep_q

    async def _clarify_drop(self, rid: str) -> None:
        """60초: CLARIFY_DROP_TO_AGENT=1 이면 요청을 무효로 하고 상담원 연결 메뉴로 진행.
        0(기본)이거나 상담원 연결 선택지가 없으면 요청을 유지한 채 계속 청취 (ARS 가 통화를 끊으면 user_input_timeout)."""
        async with self._lock:
            if not (self.pending and self.pending["request_id"] == rid) or self.ended:
                return
            cur = self.current
            if self._dead_end_open() is not None:
                return  # 막다른 안내에서는 지난 메뉴의 상담원 키를 누르지 않음 (그 안내에서 다른 뜻일 가능성)
            if cur and cur.agent_key and self.s.drop_to_agent:
                self._close_request()
                self.key_queue.clear()  # 이전 계층 복귀 중이었으면 복귀 중단 (상담원 연결로 진행)
                self.queued_key = None
                self.after_backtrack = False
                self.set_status("selecting", "응답이 없어 상담원 연결 메뉴로 진행합니다.")
                try:
                    await self._press(cur.agent_key, "ai", cur)
                except ApiError:
                    pass

    async def _ask_secure(self, p: Prompt) -> None:
        self.secure_window = True
        label = p.label or "요청 정보"
        self.secure_req = (label, p.terminator)
        # "주민등록번호 앞 6자리"·"계좌 비밀번호 네 자리" → "주민등록번호"·"계좌 비밀번호" (경로에 보일 항목명)
        self.secure_label = _field_name(label)
        note = getattr(self.d.interp, "note_input_request", None)
        if callable(note):
            note()  # hybrid: 해석기 판정 밖에서 연 입력 요청도 '입력 요청 뒤 모두 로컬 판단' 상태로 전환
        base = f"{label}{_obj(label)} 직접 눌러 주세요. 입력하신 번호는 저장되지 않습니다."
        message, menu = base, None
        warning = getattr(p, "choice_warning", None)  # 두 자리 키 뒤 고른 선택지 안내 없이 온 입력 안내 (_paragraph)
        if warning:
            keys_m, label_m, others_m, menu = warning
            message = (f"{keys_m}번을 누른 뒤 '{label_m}' 안내 없이 바로 입력 안내가 나왔습니다. 번호가 일부만 들어가 "
                       f"다른 메뉴로 갔을 수 있으니 ARS 안내를 확인한 뒤 입력해 주세요. {base}")
        req = self._new_request("secure_input", self.s.secure_expire_s, label=label, ars_prompt=p.text,
                                terminator=p.terminator, message=message)
        self.secure_warned = ({"rid": req["request_id"], "label": label_m, "others": others_m, "field": label, "base": base}
                              if warning else None)
        self.set_status("awaiting_user", "직접 입력이 필요합니다.")
        # 경고 속 선택지 이름은 그 메뉴의 화면용 받아쓰기 고침을 따름 (경로 표시와 같은 이름)
        self.publish("request", req, src=menu)

    def _drop_stale_warning(self, text: str, p: Prompt) -> None:
        """경고를 붙인 입력 요청을 재알림하기 전에, 지금 안내가 고른 선택지 이름을 말했으면(키가 제대로 들어감) 경고 제거."""
        w = self.secure_warned
        if (w and self.pending and self.pending.get("request_id") == w["rid"]
                and (w.get("heard") or _choice_said(w["label"], w["others"], (text, p.shown), w["field"]))):
            # 재안내 단락이나, 요청이 열린 뒤 들은 일반 안내(heard)에 고른 선택지 이름이 있었던 경우
            log.info("경고를 붙인 입력 요청의 재안내에서 고른 선택지 이름을 들어 경고를 뺀다")
            self.pending = {**self.pending, "message": w["base"]}
            self.secure_warned = None

    # 사용자 조작 (4.3) ---------------------------------------------------------
    def _check_request(self, body: dict, kind: str) -> dict:
        req = self.pending
        if not req or req["kind"] != kind or body.get("request_id") != req["request_id"]:
            raise ApiError(409, "invalid_state", "이미 응답했거나 만료된 요청입니다.")
        return req

    async def action(self, body: dict) -> dict:
        t = body.get("type")
        if self.ended:
            raise ApiError(409, "invalid_state", "이미 종료된 통화입니다.")
        if self.recorder is not None:
            # 민감정보 입력 요청이 열려 있으면 key·answer 값도 기록하지 않음 (요청 중 키패드로 누른 숫자는 409 로 거절되지만
            # 이 기록은 검사 전에 남기 때문)
            secure_open = bool(self.pending and self.pending.get("kind") == "secure_input")
            # 규칙 안전망이 없는 해석기(llm 조합)는 키패드 키(key)를 늘 자릿수로만 남김 (위 키 전송 기록과 같은 이유)
            keypad = t == "key" and getattr(self.d.interp, "rule_free", False)
            self.recorder.user_action(body, mask=secure_open or keypad)
        if t == "client_note":
            # 앱이 버린 누름(꺼진 버튼·바뀐 질문·끊긴 연결 등)에 대한 앱의 알림. 위에서 통화 기록에만 남기고 처리 없음
            return {"accepted": True}
        async with self._lock:
            if t == "cancel":
                if self.status == "connected_to_agent":
                    reason = "agent_call_ended"
                elif self.result_heard:
                    reason = "completed"
                else:
                    reason = "user_canceled"
                await self.finish(reason, hangup=not self.hangup_seen)
                return {"accepted": True}
            if self.hangup_seen:
                raise ApiError(409, "invalid_state", "상대가 통화를 끊었습니다.")
            if t == "request_handoff":
                return await self._act_request_handoff()
            if t == "cancel_handoff":
                return await self._act_cancel_handoff(body)
            if t == "answer":
                if self.pending and self.pending["kind"] == "handoff_goal":
                    return await self._act_handoff_answer(body)
                return await self._act_answer(body)
            if t == "select":
                req = self._check_request(body, "clarification")
                key = str(body.get("key", ""))
                if key not in {o["key"] for o in req["options"]}:
                    raise ApiError(400, "invalid_request", "요청에 포함된 후보가 아닙니다.")
                dead = self._dead_end_open()  # 막다른 안내 재질문이면 고른 이동 키를 그 안내 기준으로 처리
                self._close_request()
                if req["reason"] == "loop" and self.nav:
                    self.nav.reset_loop()
                self.input_errors = 0
                self.key_queue.clear()  # 이전 계층 복귀 중이었으면 복귀 중단 (직접 키 입력과 동일)
                self.after_backtrack = False
                await self._press(key, "user", dead or self.current)
                return {"accepted": True, "key_sent": key}
            if t == "key":
                return await self._act_key(body)
            if t == "secure_input":
                return await self._act_secure(body)
            if t == "secure_key":
                return await self._act_secure_key(body)
            if t == "revise_goal":
                return await self._act_revise_goal(body)
            raise ApiError(400, "invalid_request", f"알 수 없는 type: {t}")

    def _streamed_open(self) -> bool:
        """한 자씩 입력(secure_key)으로 키를 받은 민감정보 요청이 열려 있음. 한 번에 보내는 secure_input 은 해당 없음."""
        return bool(self.pending and self.pending["kind"] == "secure_input"
                    and self.secure_typed_req == self.pending["request_id"])

    def _stream_live(self, started_at: float | None) -> bool:
        """한 자씩 입력 요청이 열려 있고, 이 안내가 그 요청의 첫 키가 ARS 로 나간 뒤에 시작됨 (보낸 키에 대한 ARS 반응일 수 있음).
        마지막 키가 아니라 첫 키를 기준으로 판정: 사용자가 오류 안내 중에도 계속 누르면 그 안내가 마지막 키보다 먼저 시작되기 때문.
        첫 키보다 먼저 시작된 안내(누르기 전부터 ARS 가 하던 말)는 제외. 오류 안내 중에 입력 완료까지 눌러 요청이
        닫혔어도 그 키들이 아직 모아 둔 채면 해당 (그대로 보내면 앞자리가 빠진 번호가 ARS 의 새 시도로 들어감)."""
        active = self._streamed_open() or bool(self.secure_buf)  # 입력 완료로 요청은 닫혔어도 아직 보내지 않은 키가 있음
        return (active and self.secure_first_sent_at is not None
                and (started_at is None or started_at >= self.secure_first_sent_at))

    def _drop_stream_buf(self, message: str | None) -> None:
        """모아 둔 한 자씩 입력 키를 보내지 않고 버림 (값은 남기지 않음). message 가 있으면 버렸다고 알림."""
        if self.secure_queued:
            self.secure_abort = True
        if self.secure_buf:
            self.secure_buf = ""
            if message:
                self.secure_errors += 1
                self.publish("input_rejected", {"keys": None, "actor": "user", "ars_prompt": "",
                                                "retry_count": self.secure_errors, "message": message})

    def _close_request(self, keep_secure: bool = False) -> None:
        if keep_secure and self.pending and self.pending["kind"] == "secure_input":
            return
        self.pending = None
        self.dead_end_rid = self.dead_end_prompt = None
        self.cancel_timer("clarify_expire")
        self.cancel_timer("clarify_drop")

    def _dead_end_open(self) -> Prompt | None:
        """지금 열린 요청이 막다른 안내 재질문이면 그 안내."""
        if self.pending is not None and self.dead_end_rid == self.pending.get("request_id"):
            return self.dead_end_prompt
        return None

    async def _act_answer(self, body: dict) -> dict:
        text = str(body.get("text", "")).strip()
        if not text:
            raise ApiError(400, "invalid_request", "text 가 비어 있습니다.")
        req = self._check_request(body, "clarification")
        if req["reason"] == "critical_action":
            raise ApiError(409, "invalid_state", "중요 선택은 번호를 직접 골라야 합니다 (select).")
        question = self.ai_questions.pop(req["request_id"], req["message"])  # 되읽기 확인이면 숫자를 가린 문장
        dead = self._dead_end_open()
        self._close_request()
        before = (self.goal or "").strip()
        note = getattr(self.d.interp, "note_asked", None)
        if note is not None:
            # 이 답을 받은 안내(세션이 되읽기·입력 오류 처리로 새로 만든 확인 질문 포함)를 해석기에 통지. 그 안내를 다시 판단할 때만
            # 이 답을 사용 (note_asked 는 llm 조합 해석기에만 있음)
            note(self.current if (dead is None and self.current is not None and self.current_fresh) else None, question)
        self.goal = await self._ai("merge_answer", self.d.interp.merge_answer(self.goal or "", question, text),
                                   text)
        merged = self.goal
        if question == NO_AGENT_ASK and AGENT_LABEL_RE.search(before) and not AGENT_LABEL_RE.search(self.goal or ""):
            # '상담원 연결 선택지가 없다' 질문에 업무로 답한 경우('1번'·'카드 분실'): 그 업무 안에서 상담원을 찾는다는 뜻이므로 상담원 연결
            # 목적 유지 (목적 합치기가 선택지 이름으로 바꾸면 상담원 말이 사라져 하위 메뉴의 상담원 선택지를 고르지 않음)
            self.goal = f"{(self.goal or '').strip()} 상담원 연결".strip()
        if self.agent_request:
            self.goal = agent_goal(self.goal)  # 상담원 연결 버튼 통화: 답을 합쳐도 상담원 연결 목적을 유지 (_act_revise_goal 과 같음)
            self.task_goal = agent_topic(self.goal) or self.task_goal  # 업무 말이 없는 답('네')이면 앞 업무 목적 유지
        sync = getattr(self.d.interp, "set_goal", None)
        if sync is not None and self.goal != merged:
            sync(self.goal)  # 해석기(set_goal 이 있는 해석기)가 기억한 목적도 고친 목적으로 맞춤 (상담원 목적 판단에 사용)
        if question in (AGENT_UNASKED_ASK, AGENT_UNASKED_ONE_ASK) and self.current is not None and not wants_agent(self.goal):
            # '상담원 연결이 가장 가까운 선택지' 질문에 상담원이 아닌 답: 이 계층에서는 상담원 선택지를 빼고 다시 판단
            # (빼지 않으면 같은 점수로 상담원을 다시 1위로 골라 누름)
            agents = {o.key for o in self.current.options if is_agent_option(o)}
            if agents:
                self.nav.exclude, self.nav.exclude_at = agents, self.current.signature()
        self.nav.goal = self.goal
        if req["reason"] == "loop":
            self.nav.reset_loop()
        self.input_errors = 0
        self.key_queue.clear()  # 이전 계층 복귀 중이었으면 복귀 중단 (사용자 답으로 재판단)
        self.queued_key = None
        self.after_backtrack = False
        if dead is not None and self.layers and self.agent_active and not self.ended:
            # 막다른 안내 재질문에 글로 답한 경우: 고친 목적으로 그 안내를 재판단 (답이면 결과, 다른 갈래면 되돌아가기, 모르면 재질문).
            # 지난 메뉴로 판단하면 지금 안내와 맞지 않는 키를 누르게 됨
            # 목적이 바뀐 답이면 사용자 판단으로 보고 낱말 확인 없이 재판단. 목적이 그대로인 답('네')은 낱말 확인 재적용
            # (낱말 확인으로 물은 안내에 '네' 라고 답했는데 확인을 끄면 같은 모델 판단으로 결과가 되는 문제 방지)
            self.set_status("selecting")
            if not await self._dead_end(dead, after_answer=(self.goal or "").strip() != before):
                await self._on_result()
        elif self.current is not None and self.current_fresh:
            self.answer_revised = (self.goal or "").strip() != before  # 목적이 바뀐 답: 사용자 갈래에서도 이전 계층 확인 사용
            await self._decide(self.current, arrived=False)
        else:
            # 음성 응답 요구처럼 지금 안내가 메뉴가 아니면 지난 메뉴로 판단하지 않고 다음 안내 대기
            self.set_status(self.idle_status())
        return {"accepted": True}

    async def _act_key(self, body: dict) -> dict:
        key = str(body.get("key", ""))
        if not key or len(key) > 4 or any(ch not in VALID_KEYS for ch in key):
            raise ApiError(400, "invalid_request", "key 는 0-9, *, # 1~4자입니다.")
        if self.pending and self.pending["kind"] == "secure_input":
            raise ApiError(409, "sensitive_input_blocked", "민감정보는 secure_input 으로 입력하십시오.")
        if self.call is None or self.answered_at is None:
            raise ApiError(409, "invalid_state", "통화가 연결되지 않았습니다.")
        dead = self._dead_end_open()  # 막다른 안내 재질문 중 키패드로 누른 이동 키도 그 안내 기준으로 처리
        self._close_request()
        self.key_queue.clear()          # 이전 계층 복귀 중이었으면 복귀 중단
        self.after_backtrack = False
        if self.agent_active:
            self.ai_hold_until = time.monotonic() + self.s.user_key_hold_s
            self.timer("ai_hold", self.s.user_key_hold_s, self._resume_after_hold)
        # 마지막 키 전송 뒤 새 메뉴를 듣지 못한 상태(민감정보 안내 중, 안내가 끝나기 전 등)에서 누른 키는
        # 지난 메뉴의 선택지로 해석하지 않고 "N 입력" 으로만 기록한다 (6.2)
        await self._press(key, "user", dead or (self.current if self.current_fresh else None), keypad=True)
        return {"accepted": True, "key_sent": key}

    async def _resume_after_hold(self) -> None:
        async with self._lock:
            if self.agent_active and self.current_fresh and self.current and self.pending is None:
                await self._decide_heard(self.current)

    async def _act_secure(self, body: dict) -> dict:
        req = self._check_request(body, "secure_input")
        digits = str(body.get("digits", ""))
        if not re.fullmatch(r"[0-9*#]{1,20}", digits):
            raise ApiError(400, "invalid_request", "digits 는 0-9, *, # 1~20자입니다.")
        term = req.get("terminator") or ""
        keys = digits if (term and digits.endswith(term)) else digits + term  # 사용자가 종료 키까지 누른 경우 중복 방지
        self.secure_len = _digit_count(digits)  # 자릿수만 기억 (되읽기 마스킹용)
        self.secure_sent_at = time.monotonic()  # 외부 STT 는 이 뒤 secure_skip_s 동안 음성 미전송
        del digits
        self.pending = None
        self.current_fresh = False
        self.last_key_actor = "user"
        if self._ars_speaking() and self.s.secure_wait_max_s > 0:
            # ARS 안내 중. '입력 시간이 초과되었습니다' 처럼 키를 받지 않는 안내 중에 보내면 입력이 무시되므로
            # 안내 종료 뒤 전송. 대기 중 다른 처리(안내 인식·종료 요청)가 막히지 않도록 잠금 밖에서 대기
            self.secure_queued, self.secure_abort = True, False
            self.set_status(self.idle_status())
            self._spawn_secure_send(keys)
            del keys
            return {"accepted": True}
        await self._send_keys(keys, "user", secure=True)  # 메모리에서 톤으로 변환하여 전송
        del keys
        self.last_send_secure = True
        self.set_status(self.idle_status())
        return {"accepted": True}

    def _save_recording(self) -> None:
        """통화 기록을 서버 디스크에 저장 (RECORDING_SAVE=1). 실패해도 통화 처리에는 영향 없음 (경고 로그만).
        종료 순간 인식·판단 중이던 단락이 있으면 그 결과까지 담도록 처리 종료를 기다린 뒤(최대 hangup_flush_s) 내용 생성.
        그동안 메모리 기록 삭제 요청이 와도 저장이 끝난 뒤에 삭제 (drop_recording)."""
        if not getattr(self.s, "recording_save", False) or self.recorder is None:
            return
        rec = self.recorder
        try:
            created = datetime.fromisoformat(rec.meta["created_at"]) if rec.meta.get("created_at") else datetime.now()
            name = folder_name(self.goal or "", self.phone, created)
        except Exception as e:
            log.warning("통화 기록 저장 준비 실패: %s", e)
            return

        async def write() -> None:
            deadline = time.monotonic() + self.s.hangup_flush_s
            # 발행 대기열(받아쓰기 고침을 기다리는 자막·이벤트)도 모두 전송한 뒤 생성 (고침 기록·마지막 이벤트 누락 방지)
            while ((self.stt_inflight or self._lock.locked() or self._outbox_busy())
                   and time.monotonic() < deadline):
                await asyncio.sleep(0.05)
            try:
                files = rec.files()  # 이벤트 루프에서 생성 (기록이 더 바뀌지 않는 시점)
                path = await asyncio.to_thread(save_files, self.s.recording_dir, name, files)
                self.recording_path = str(path)
                log.info("통화 기록 저장: %s", path)
            except Exception as e:
                log.warning("통화 기록 저장 실패 (%s): %s", self.s.recording_dir, e)

        self._save_task = asyncio.get_running_loop().create_task(write())

    def drop_recording(self) -> None:
        """통화 기록을 메모리에서 삭제 (주인의 삭제 요청 또는 보관 시간 경과). 디스크 사본(recordings/)은 유지."""
        if self._save_task is not None and not self._save_task.done():
            self._save_task.add_done_callback(lambda _t: self.drop_recording())  # 디스크 저장 종료 뒤 삭제
            return
        if self.recorder is not None:
            self.recorder.wipe()
            self.recorder = None
            log.info("통화 기록을 지웠다 (%s)", self.id)

    def _ars_speaking(self) -> bool:
        """ARS 소리 발생 여부 (단락 종료 무음 paragraph_silence_ms 전까지는 발화 중으로 판정)."""
        return bool(getattr(self.call, "speaking", False) or self.segmenter.speaking)

    def _spawn_secure_send(self, keys: str) -> None:
        task = asyncio.create_task(self._send_secure_when_quiet(keys))
        self._secure_task = task  # 작업 참조 유지 (값은 작업 종료 시 소멸)

    async def _send_secure_when_quiet(self, keys: str) -> None:
        t0 = time.monotonic()
        try:
            while (self._ars_speaking() and not self.ended and not self.hangup_seen and not self.secure_abort
                   and time.monotonic() - t0 < self.s.secure_wait_max_s):
                await asyncio.sleep(0.05)
            if self.ended or self.hangup_seen:
                return
            if self.secure_abort or self._ars_speaking():
                # 다른 안내로 넘어갔거나, 입력 안내 되풀이로 보기에는 발화가 너무 긴 경우(첫 메뉴로 돌아간 경우 등).
                # 입력값을 메뉴 키로 잘못 보내지 않도록 폐기하고, 입력 안내가 다시 나오면 새로 입력 수신
                log.info("ARS 가 다른 안내로 넘어가 입력값을 보내지 않았다 (%.1f초 기다림)", time.monotonic() - t0)
                self.secure_errors += 1
                self.publish("input_rejected", {
                    "keys": None, "actor": "user", "ars_prompt": "", "retry_count": self.secure_errors,
                    "message": "ARS가 다른 안내로 넘어가 입력하신 번호를 보내지 않았습니다. 입력 안내가 다시 나오면 다시 입력해 주세요.",
                })
                return
            log.info("ARS 안내가 끝나 입력값을 보낸다 (%.1f초 기다림)", time.monotonic() - t0)
            # 잠금 없이 전송: 안내 인식·AI 판단이 잠금을 잡고 있으면 ARS 의 입력 대기 시간 초과 우려
            self.last_send_secure = True       # 전송 중에 처리되는 같은 입력 안내는 무시 (stale)
            self.secure_sent_at = time.monotonic()
            await self._send_keys(keys, "user", secure=True)
            self.last_send_secure = True
        except ApiError:
            pass
        except Exception:
            log.exception("입력값 전송 실패")
        finally:
            del keys
            self.secure_queued = False

    async def _act_secure_key(self, body: dict) -> dict:
        """민감정보 한 자씩 입력: 사용자가 누르는 대로 바로 ARS 로 보냄 (전화 키패드와 같음. 번호를 다 누를 때까지 기다리다 ARS 가
        시간 초과로 넘어가는 문제 방지). body: request_id, key(0-9*# 한 자), done(입력 끝, true 면 key 생략 가능).
        - 끝(done 또는 요청의 종료 키)이면 종료 키를 붙이고(이미 눌렀으면 생략) 요청을 닫음. 그 전에는 요청을 열어 둠(기한 연장).
        - ARS 가 말하는 중이면 모았다가 안내가 끝나면 보냄 (말하는 중에 보낸 키는 무시될 수 있음, secure_input 과 같은 규칙).
          ARS 가 다른 안내로 넘어가면 모은 키를 버리고 알림.
        - 값은 기록·저장하지 않고 자릿수만 기억 (되읽기 가리기·다시 인식용)."""
        req = self._check_request(body, "secure_input")
        key = str(body.get("key") or "")
        done = body.get("done") is True
        if not (re.fullmatch(r"[0-9*#]", key) or (key == "" and done)):
            raise ApiError(400, "invalid_request", "key 는 0-9, *, # 한 자입니다 (끝낼 때는 done=true).")
        term = req.get("terminator") or ""
        if self.secure_typed_req != req["request_id"]:
            self.secure_typed_req, self.secure_len, self.secure_first_sent_at = req["request_id"], 0, None
            self.secure_req_pub_at = time.monotonic()
        finished = done or (bool(term) and key == term)
        keys = key + (term if done and term and key != term else "")
        self.secure_sent_at = time.monotonic()
        self.last_key_actor = "user"
        if finished:
            self.pending = None
            self.current_fresh = False
        else:
            req["expires_at"] = now_iso(self.s.secure_expire_s)
            if time.monotonic() - self.secure_req_pub_at > 15:
                # 화면은 요청의 기한(expires_at)이 지나면 입력 칸을 닫음. 입력이 길어지면 늘린 기한을 다시 알림 (자주 보내지 않음)
                self.secure_req_pub_at = time.monotonic()
                self.publish("request", req)
        if keys:
            # 모아 둔 키가 있거나 보내는 중(secure_queued)이면 그 뒤에 추가: 따로 보내면 두 전송이 겹쳐 순서가 바뀜.
            # ARS 가 말하는 중이거나 들은 안내를 아직 처리하는 중이어도 모음 (그 안내가 다른 안내로 넘어간 것이면 버림)
            if self.secure_buf or self.secure_queued or (self._ars_active() and self.s.secure_wait_max_s > 0):
                self.secure_buf += keys
                if not self.secure_queued:
                    self.secure_queued, self.secure_abort = True, False
                    self._secure_task = asyncio.create_task(self._flush_secure_buf())
            else:
                await self._send_keys(keys, "user", secure=True)  # 메모리에서 톤으로 변환하여 전송
                self.last_send_secure = True
                self._stream_sent(keys)
        del key, keys
        if finished:
            self.set_status(self.idle_status())
        return {"accepted": True, "done": finished}

    async def _flush_secure_buf(self) -> None:
        """한 자씩 입력 중 모아 둔 키를 보냄. ARS 가 조용해지고 들은 안내의 처리(인식·판단)가 끝난 뒤에 전송: 그 안내가 다른 안내·
        입력 오류·새 입력 안내였으면 판단 쪽이 secure_abort 를 세우거나 모은 키를 비움. 보내는 동안 새로 모인 키도 같은 규칙으로
        이어 전송 (이 작업 하나만 보내므로 순서가 바뀌지 않음). 잠금 밖에서 대기."""
        t0 = time.monotonic()
        try:
            while True:
                while (self._ars_active() and not self.ended and not self.hangup_seen and not self.secure_abort
                       and time.monotonic() - t0 < self.s.secure_wait_max_s):
                    await asyncio.sleep(0.05)
                if self.ended or self.hangup_seen:
                    return
                if self.secure_abort or self._ars_active():
                    dropped = bool(self.secure_buf)
                    self.secure_buf = ""
                    if dropped:
                        log.info("ARS 가 다른 안내로 넘어가 모아 둔 입력을 보내지 않았다 (%.1f초 기다림)", time.monotonic() - t0)
                        self.secure_errors += 1
                        self.publish("input_rejected", {
                            "keys": None, "actor": "user", "ars_prompt": "", "retry_count": self.secure_errors,
                            "message": "ARS가 다른 안내로 넘어가 입력하신 번호를 보내지 않았습니다. 입력 안내가 다시 나오면 다시 입력해 주세요.",
                        })
                        async with self._lock:
                            if self._streamed_open() and not (self.ended or self.hangup_seen):
                                # 같은 요청을 그대로 두면 화면 입력 칸이 버린 키까지 센 채로 남음. 같은 항목으로 새 요청을 엶
                                old = self.pending
                                self._close_request()
                                label, term = self.secure_req
                                await self._ask_secure(Prompt(kind="sensitive", text=old.get("ars_prompt") or "",
                                                              label=label, terminator=term))
                    return
                if not self.secure_buf:
                    return
                keys, self.secure_buf = self.secure_buf, ""
                self.last_send_secure = True
                self.secure_sent_at = time.monotonic()
                await self._send_keys(keys, "user", secure=True)
                self.last_send_secure = True
                self._stream_sent(keys)
                del keys
        except ApiError:
            pass
        except Exception:
            log.exception("입력값 전송 실패")
        finally:
            self.secure_buf = ""
            self.secure_queued = False

    def _stream_sent(self, keys: str) -> None:
        """한 자씩 입력 키를 ARS 로 보낸 뒤: 보낸 자릿수만 집계 (버린 키는 세지 않음), 그 요청의 첫 전송 시각 기록."""
        self.secure_len = (self.secure_len or 0) + _digit_count(keys)
        if self.secure_first_sent_at is None:
            self.secure_first_sent_at = self.last_key_at

    async def _act_revise_goal(self, body: dict) -> dict:
        """다시 말하기: 목적 설정 단계, 즉 첫 번호를 누르기 전까지만 수락.
        새 목적으로 바꾸고 지금 머물러 있는 메뉴(첫 메뉴)에서 재판단."""
        text = str(body.get("text", "")).strip()
        if not text:
            raise ApiError(400, "invalid_request", "text(목적)가 비어 있습니다.")
        if self.path or self.status == "connected_to_agent":
            raise ApiError(409, "invalid_state",
                           "목적은 첫 번호를 누르기 전까지만 바꿀 수 있습니다. 이후에는 재질문에 답하거나 직접 선택하십시오.")
        # 상담원 연결 버튼으로 건 통화는 다시 말한 업무로도 상담원을 찾음 (업무만 말해도 상담원 연결 요청을 잃지 않도록)
        if self.agent_request:
            self.task_goal = agent_topic(text)
        text = agent_goal(text) if self.agent_request else text
        self.goal = text
        self.goal_summary = await self._ai("summarize_goal", self.d.interp.summarize_goal(text),
                                           f"'{text}' 업무를 찾아 드릴게요.")
        self.summary = None
        self.result_heard = False
        # 앞 목적으로 기다리던 첫 메뉴 전 결과 보류 종료 (새 목적으로는 아직 들은 안내 없음. 타이머가 울려 잠금을 기다리는 확인도 표시로 차단)
        self.early_result_wait = False
        self.result_upto = None
        self.early_end_kind = None
        self.cancel_timer("early_result")
        self._close_request(keep_secure=True)
        self.key_queue.clear()
        await self._activate_agent()
        return {"accepted": True, "status": self.status, "goal_summary": self.goal_summary}

    # 통화 중 상담원 연결 (request_handoff → answer 또는 cancel_handoff) ------------------------
    async def _act_request_handoff(self) -> dict:
        """통화 중 상담원 연결 요청: 자동 선택을 멈추고 상담받을 업무 목적을 질문 (request 이벤트 kind=handoff_goal).
        답(answer)을 받으면 그 업무로 상담원 선택지를 우선해 탐색 (routing_mode=human_agent).
        같은 요청이 열려 있으면 새로 열지 않고 그 요청 번호를 반환 (중복 요청으로 요청·키 전송이 겹치지 않도록)."""
        req = self.pending
        if req and req["kind"] == "handoff_goal":
            return {"accepted": True, "request_id": req["request_id"], "status": self.status, "duplicate": True}
        if self.status in ("waiting_agent", "connected_to_agent"):
            raise ApiError(409, "invalid_state", "상담원 연결 대기·연결 중에는 요청할 수 없습니다.")
        if (req and req["kind"] == "secure_input") or self.secure_queued or self.secure_buf:
            raise ApiError(409, "sensitive_input_blocked",
                           "민감정보 입력 중에는 상담원 연결로 바꿀 수 없습니다. 입력을 마친 뒤 다시 요청하십시오.")
        if req and req.get("reason") == "critical_action":
            raise ApiError(409, "invalid_state", "중요 선택 확인 중에는 상담원 연결로 바꿀 수 없습니다. 번호를 직접 고르십시오.")
        if self.call is None or self.answered_at is None:
            raise ApiError(409, "invalid_state", "통화가 연결되지 않았습니다.")
        # 열려 있던 재질문은 닫아 두고, 입력 취소 시 다시 열기. 요청 시점의 메뉴를 함께 남겨 그 사이 새 메뉴를 들었는지 확인
        self.handoff_prev = {"current": self.current, "req": req,
                             "question": self.ai_questions.get(req["request_id"]) if req else None,
                             "dead_prompt": self._dead_end_open()}
        self._close_request()
        self.key_queue.clear()  # 이전 계층 복귀 중이었으면 복귀 중단 (다른 사용자 조작과 같음)
        self.queued_key = None
        self.after_backtrack = False
        req = self._new_request("handoff_goal", self.s.clarify_drop_s, options=[],
                                message="상담원에게 연결해 드릴게요. 어떤 업무로 상담받으실지 말씀해 주세요.")
        rid = req["request_id"]
        self.set_status("awaiting_user", req["message"])
        self.publish("request", req)
        self.timer("clarify_expire", self.s.clarify_expire_s, lambda: self._handoff_expire(rid))
        self.timer("clarify_drop", self.s.clarify_drop_s, lambda: self._handoff_drop(rid))
        return {"accepted": True, "request_id": rid, "status": self.status}

    def _handoff_open(self) -> bool:
        return bool(self.pending and self.pending["kind"] == "handoff_goal")

    async def _act_handoff_answer(self, body: dict) -> dict:
        text = str(body.get("text", "")).strip()
        if not text:
            raise ApiError(400, "invalid_request", "text 가 비어 있습니다.")
        self._check_request(body, "handoff_goal")
        await self._start_handoff(text)
        return {"accepted": True, "status": self.status, "goal_summary": self.goal_summary,
                "routing_mode": self.routing_mode}

    async def _act_cancel_handoff(self, body: dict) -> dict:
        """업무 목적 입력 취소: 요청을 닫고 앞의 목적·탐색으로 복귀 (routing_mode 유지). 통화 종료는 cancel."""
        self._check_request(body, "handoff_goal")
        prev, self.handoff_prev = self.handoff_prev, None
        self._close_request()
        same = prev is not None and _same_prompt(prev["current"], self.current)  # 같은 메뉴를 다시 읽은 것도 같음
        if same and prev["req"] is not None:
            self._reopen_request(prev)  # 요청 사이 새 메뉴를 듣지 않음: 닫아 둔 재질문을 다시 열기
        elif self.agent_active and self.current is not None and self.current_fresh:
            if same:
                await self._decide(self.current, arrived=False)
            else:
                await self._decide_heard(self.current)  # 요청 중에 들어 판단하지 않은 메뉴: 새로 들은 메뉴로 판단
        else:
            self.set_status(self.idle_status())
        if self.pending is None and not self.ended and self.status == "awaiting_user":
            self.set_status(self.idle_status())
        return {"accepted": True, "status": self.status, "routing_mode": self.routing_mode}

    def _reopen_request(self, prev: dict) -> None:
        """업무 목적 입력 요청을 열 때 닫은 재질문을 새 요청 번호로 다시 열기 (만료 시각·시간 처리도 새로 설정)."""
        old = prev["req"]
        self.req_seq += 1
        req = {**old, "request_id": f"req_{self.req_seq}", "expires_at": now_iso(self.s.clarify_expire_s)}
        self.pending = req
        rid = req["request_id"]
        self.ai_questions = {rid: prev["question"] if prev["question"] is not None else req["message"]}
        if prev["dead_prompt"] is not None:
            self.dead_end_rid, self.dead_end_prompt = rid, prev["dead_prompt"]
        self.set_status("awaiting_user", req["message"])
        self.publish("request", req)
        self.timer("clarify_expire", self.s.clarify_expire_s, lambda: self._clarify_expire(rid))
        self.timer("clarify_drop", self.s.clarify_drop_s, lambda: self._clarify_drop(rid))

    async def _handoff_expire(self, rid: str) -> None:
        """업무 목적 입력 30초: 지금 머문 메뉴(마지막 키 뒤에 들은 메뉴)가 있을 때만 다시 듣기 키 전송 (_clarify_expire).
        이 요청은 메뉴와 관계없이 열리므로, 결과 안내 중이거나 키를 누른 뒤 새 메뉴를 듣지 못했으면 지난 메뉴의 키는 누르지 않음."""
        if self.agent_active and self.current is not None and self.current_fresh:
            await self._clarify_expire(rid)

    async def _handoff_drop(self, rid: str) -> None:
        """업무 목적 입력 만료: 상담원 연결 요청대로, 지금까지의 업무 목적으로 상담원 연결 탐색."""
        async with self._lock:
            if not (self.pending and self.pending["request_id"] == rid) or self.ended or self.hangup_seen:
                return
            await self._start_handoff(self.task_goal if self.routing_mode == "human_agent" else (self.goal or ""))

    async def _start_handoff(self, text: str) -> None:
        """업무 목적 text 로 상담원 우선 탐색 시작: 목적 '<업무> 상담원 연결' (세션 생성 agent=true 와 같은 탐색)."""
        self._close_request()
        self.handoff_prev = None
        self.agent_request = True
        self.routing_mode = "human_agent"
        self.goal = agent_goal(text)
        self.task_goal = agent_topic(self.goal)
        self.goal_summary = await self._ai("summarize_goal", self.d.interp.summarize_goal(self.goal),
                                           f"'{self.goal}' 업무를 찾아 드릴게요.")
        self.summary = None
        self.result_heard = False
        self.early_result_wait = False  # 앞 목적으로 기다리던 첫 메뉴 전 결과 보류 종료 (_act_revise_goal 과 같음)
        self.result_upto = None
        self.early_end_kind = None
        self.cancel_timer("early_result")
        self.key_queue.clear()
        self.queued_key = None
        await self._activate_agent()

    async def _activate_agent(self) -> None:
        """새 목적으로 탐색 (재)시작. 지금 머물러 있는 계층(가장 최근 메뉴)에서 판단."""
        self.agent_active = True
        self.nav = NavState(self.goal or "")
        self.agent_goal_at_caption = len(self.captions)
        self.redo_same_menu = self.after_backtrack = False
        self.input_errors = 0
        if self.pending and self.pending["kind"] == "secure_input":
            return  # 직접 입력이 끝나면 다음 안내부터 탐색
        if self.answered_at is None:
            return  # 상대가 받으면 listening 으로 전환
        self.set_status("listening", self.goal_summary)
        if self.current is not None and self.current_fresh:
            await self._decide(self.current, arrived=True)
        # 마지막 키 전송 이후 아직 메뉴를 듣지 못했으면 다음 안내 대기 (ARS 가 다시 읽거나 다음 메뉴를 안내)

    # 결과 (4.4) ---------------------------------------------------------------
    def result(self) -> tuple[int, dict]:
        if not self.ended:
            return 202, {"status": self.status}
        outcome = {"completed": "success", "agent_call_ended": "partial", "message_sent": "partial"}.get(
            self.end_reason or "", "failed")
        dur = round((self.ended_at or 0) - self.answered_at) if self.answered_at else 0
        return 200, {
            "outcome": outcome,
            "end_reason": self.end_reason,
            "summary": (self.summary or {}).get("summary"),
            "key_info": (self.summary or {}).get("key_info", []),
            "handoff_script": (self.summary or {}).get("handoff_script"),
            "path": self.path,
            "duration_sec": dur,
        }


class SessionManager:
    """Session 목록과 발신 번호 배정. 번호가 하나(capacity 1)면 진행 중인 통화가 있을 때 새 통화 차단.
    번호가 여럿이면 통화마다 빈 번호를 하나 선택. 선택과 Session 등록 사이에 await 가 없어 동시에 온 생성 요청의
    같은 번호 배정 방지."""

    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.sessions: dict[str, Session] = {}

    @property
    def combos(self):
        return getattr(self.deps, "combos", None)

    @property
    def capacity(self) -> int:
        try:
            return max(1, int(getattr(self.deps.backend, "capacity", 1) or 1))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _holds_line(s: Session) -> bool:
        # 상대가 끊은 뒤 마지막 안내를 처리 중(hangup_seen)인 Session 은 회선 미사용
        return not s.ended and not s.hangup_seen

    def active(self) -> Session | None:
        # 상대가 끊은 뒤 마지막 안내를 처리 중(hangup_seen)인 Session 은 회선을 쓰지 않으므로 새 통화를 막지 않음
        return next((s for s in self.sessions.values() if self._holds_line(s)), None)

    def active_sessions(self) -> list[Session]:
        """끝나지 않은 Session (마지막 안내 처리 중 포함). 판단 모델·음성 인식을 함께 쓰는 통화 수 표시용."""
        return [s for s in self.sessions.values() if not s.ended]

    def lines_in_use(self) -> int:
        return sum(1 for s in self.sessions.values() if self._holds_line(s))

    def _pick_line(self) -> int | None:
        """빈 번호 하나 (번호가 여럿일 때). 선택 순서: 통화 없음 → 이전 통화 정리 중, 같으면 번호 순.
        쓸 수 없는 번호(다른 프로세스가 가져감·연결 끊김)는 제외. 없으면 None."""
        held = [s for s in self.sessions.values() if self._holds_line(s)]
        if len(held) >= self.capacity:
            return None
        taken = {s.line for s in held}
        rank = getattr(self.deps.backend, "line_rank", None)
        best: tuple[int, int] | None = None
        for i in range(self.capacity):
            if i in taken:
                continue
            r = rank(i) if rank else 0
            if r >= 2:
                continue
            if best is None or r < best[0]:
                best = (r, i)
        return best[1] if best else None

    def _call_deps(self, interp, fix, result=None) -> tuple[Deps, dict | None]:
        """이 통화가 쓸 의존 객체. 조합을 고를 수 있는 서버면 해석기·탐색기를 통화마다 새로 생성.
        result: 결과 안내 판단 (jev/local, hybrid 만. 비었으면 서버 기본값)."""
        f = self.combos
        if f is None:
            if interp not in (None, "") or fix not in (None, "") or result not in (None, ""):
                raise ApiError(400, "invalid_request", "이 서버는 판단 조합을 고를 수 없습니다.")
            return self.deps, None
        from combos import ComboError

        try:
            i, fx = f.normalize(interp, fix)
            r = f.normalize_result(i, result)
        except ComboError as e:
            raise ApiError(400, "invalid_request", str(e))
        it = f.build(i, fx, r)
        if is_som(it):
            nav = som_navigator_like(self.deps.navigator, it)  # som 방식은 som_jev 탐색기 (기준값은 서버 탐색기와 같음)
        elif getattr(it, "navigator_class", None) is not None:
            # 해석기가 자기 탐색기를 지정한 경우 (jevllm 조합의 JevLlmNavigator). 지정이 없는 해석기는 아래 규칙 그대로
            nav = it.navigator_class.like(self.deps.navigator, it)
        elif getattr(it, "rule_free", False):
            from llm_mode import LlmNavigator
            nav = LlmNavigator.like(self.deps.navigator, it)   # llm 조합: 탐색 규칙 판단을 분류 때 받은 모델 판단으로 대체
        else:
            nav = copy.copy(self.deps.navigator)
            nav.interp = it
        return dataclasses.replace(self.deps, interp=it, navigator=nav), {"interp": i, "fix": fx, "result": r}

    def get(self, sid: str, token: str | None) -> Session:
        s = self.sessions.get(sid)
        if s is None:
            raise ApiError(404, "not_found", "세션이 없습니다.")
        if not token or not secrets.compare_digest(token, s.token):
            raise ApiError(401, "unauthorized", "세션 토큰이 올바르지 않습니다.")
        if s.ended and time.monotonic() - (s.ended_at or 0) > self.deps.settings.token_ttl_s:
            raise ApiError(401, "unauthorized", "만료된 세션 토큰입니다.")
        return s

    async def create(self, phone: str, goal: str | None, record: bool = False,
                     interp: str | None = None, fix: str | None = None, agent: bool = False,
                     result: str | None = None) -> dict:
        # 상담원 연결 요청(agent)은 업무를 입력하지 않아도 발신 가능 (목적 '상담원 연결': 첫 메뉴에 상담원이 없으면 업무 질문)
        if not agent and (not goal or not str(goal).strip()):
            raise ApiError(400, "invalid_request", "goal(목적)이 필요합니다. AI 도우미 사용과 목적은 발신 시 정합니다.")
        deps, combo = self._call_deps(interp, fix, result)
        line: int | None = None
        if self.capacity == 1:
            if self.active() is not None:
                raise ApiError(409, "invalid_state", "다른 통화가 진행 중입니다. (동시 통화 1건)")
        else:
            line = self._pick_line()
            if line is None:
                raise ApiError(409, "invalid_state", "빈 번호가 없습니다. 다른 분이 시험 중일 수 있습니다. 잠시 뒤 다시 걸어 주십시오.",
                               reason="no_free_line")
        digits = "".join(ch for ch in phone if ch.isdigit())
        if not 3 <= len(digits) <= 13:
            raise ApiError(400, "invalid_request", "전화번호 형식이 올바르지 않습니다.")
        s = Session(digits, deps, line=line, combo=combo, repick=self._pick_line if line is not None else None)
        self.sessions[s.id] = s
        if line is not None or combo is not None:
            log.info("세션 %s 시작%s%s", s.id, f" (번호{line + 1})" if line is not None else "",
                     f" 판단={combo['interp']} 고침={combo['fix']} 결과={combo['result']}" if combo else "")
        return await s.create(goal, record, agent=agent)
