"""
메뉴 탐색 판단 (API 명세 7장 처리 흐름의 '선택지 안내' 분류)

탐색 규칙 (v0.3)
  1. 현재 머물러 있는 ARS 계층(가장 최근에 들은 메뉴) 기준 판단. 처음에는 첫 메뉴, 다시 말하기 후에는 그 시점의 메뉴.
  2. 신뢰도(AI score: 목적을 선택지 카테고리에 맞춘 값) 구간별 판단.
       0.50 이상           일치. 1·2순위 차이 0.10 이상이면 키 전송, 미만이면 재질문(ambiguous)
       0.30 이상 0.50 미만  불확실. 재질문(low_confidence), 계층당 1회
       0.30 미만           불일치 (low_confidence 답변 뒤에도 0.50 미만이면 불일치로 판정)
  3. 불일치 시 바로 이전 계층의 다른 선택지 확인.
     이전 계층에 0.50 이상인 선택지가 있으면 '이전 메뉴' 키(없으면 '처음으로' 키 + 지나온 키)로 복귀 후 재판단.
  4. 이전 계층에도 없으면 탐색 종료 (end_reason=not_found). 이전 계층 복귀는 탐색당 1회만 허용.

판단 결과(Decision)만 반환. 키 전송·이벤트 전송은 session 모듈에서 수행.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from interpreter import AGENT_LABEL_RE, Interpreter, Option, Prompt, is_agent_option

_GENERIC = {"서비스", "안내", "업무", "문의", "연결", "메뉴", "기타", "다른", "신청", "조회", "발급", "변경", "확인"}


def _chunks(label: str) -> list[str]:
    """선택지 이름의 낱말 조각 분리: '신용 체크카드' → ['신용', '체크', '카드'], '현금카드' → ['현금', '카드']."""
    import re

    out: list[str] = []
    for w in re.split(r"[\s·,/및과와]+", label):
        w = re.sub(r"[^가-힣A-Za-z0-9]", "", w)
        # 끝에 붙은 조사 제거: '다른은행으로 송금' 의 '다른은행으로' 가 목적 '다른 은행 계좌로 …' 에 그대로 없어
        # 목적이 종류를 가리키는데도 가리키지 않는다고 보고 재질문하는 문제 방지. 남는 말이 두 글자 이상일 때만
        w = re.sub(r"^(.{2,}?)(?:으로|에서|에게|을|를|은|는)$", r"\1", w)
        if not w:
            continue
        # '체크카드'·'현금카드' 처럼 붙은 복합어는 뒤의 공통 명사 분리 (뒤 명사 2글자, 앞부분 2글자 이상)
        m = re.match(r"^(.{2,})(카드|주소|계좌|통장|보험|요금|번호|예금|대출|수표)$", w)
        out += [m.group(1), m.group(2)] if m else [w]
    return out


def _gap(a: float, b: float) -> float:
    """점수 차이. 모델 점수가 0.1 단위이므로 소수 오차(0.6 - 0.5 = 0.0999…)로 같은 0.1 차이가 다르게 판정되지 않도록 반올림."""
    return round(a - b, 6)


SHOWN_MAX = 5  # 확신 없음·맞는 선택지 없음 재질문에 점수 순으로 보여 줄 선택지 수 (limit 번째와 점수 차이가 margin 미만인 선택지는 넘쳐도 모두 표시)


def shown_options(ranked: list[tuple[Option, float]], margin: float, limit: int = SHOWN_MAX) -> list[Option]:
    """확신이 없을 때 사용자에게 보여 줄 선택지. 점수 순 limit 개에 더해, limit 번째 선택지와 점수 차이가
    1·2위 차이 기준(margin) 미만인 선택지 모두 포함. 같은 점수끼리는 넣고 뺄 근거가 없으므로 묶음째 표시
    (여러 선택지가 같은 점수를 받을 때 목록 앞 limit 개만 보여 정답 선택지가 빠지는 문제 방지).
    0점 묶음도 동일: 해석기가 점수를 받지 못한 선택지에도 0 을 주므로 0 이 '아님'만을 뜻하지는 않음.
    기준은 limit 번째 점수에 고정: 차이가 이어지며 목록이 끝없이 늘어나는 것 방지."""
    if len(ranked) <= limit:
        return [o for o, _ in ranked]
    cut = ranked[limit - 1][1]
    return [o for i, (o, s) in enumerate(ranked) if i < limit or _gap(cut, s) < margin]


def sibling_choices(goal: str, best: Option, others: list[Option]) -> list[Option]:
    """best 가 다른 선택지와 같은 대상의 '종류'를 고르는 선택지이고(신용·체크카드 / 현금카드, 집 주소 / 회사 주소,
    당행 송금 / 타행 송금) 목적에 그 종류를 가르는 말이 없으면, AI 가 대신 고르지 않고 사용자에게 함께 물을 선택지 반환.
    같은 대상 = 선택지 이름의 마지막 명사(머리말)가 같음. '결제일 변경 / 결제 대금 조회' 처럼 머리말이 다르면 다른 업무."""
    g = goal.replace(" ", "")

    def same(a: str, b: str) -> bool:  # '주소지'·'주소' 처럼 한쪽이 다른 쪽으로 시작하면 같은 명사로 판정
        return len(a) >= 2 and len(b) >= 2 and (a.startswith(b) or b.startswith(a))

    bc = _chunks(best.label)
    if len(bc) < 2 or bc[-1] in _GENERIC:
        return []
    twins: list[Option] = []
    for o in others:
        oc = _chunks(o.label)
        if len(oc) < 2 or not same(bc[-1], oc[-1]):
            continue
        if any(c in g for c in bc[:-1] if len(c) >= 2):
            continue  # 목적이 best 쪽을 가리킴 ("신용카드 …" → 신용 체크카드)
        if any(c in g for c in oc[:-1] if len(c) >= 2):
            continue  # 목적이 다른 쪽을 가리킴
        twins.append(o)
    return twins


SIBLING_REST_NAMED = 3  # 같은 종류 재질문에서 이름을 밝혀 안내할 나머지 선택지 수 (초과 시 '그 밖의 선택지' 로만 안내)


def sibling_message(twins: list[Option], rest: list[Option]) -> str:
    """같은 종류 재질문 문구: 같은 종류 선택지끼리 묻고, 같은 메뉴의 나머지 업무 선택지가 있으면 그것도 선택 가능하다고 안내."""
    head = f"{' / '.join(o.label for o in twins)} 중 어느 쪽인가요? "
    if rest and len(rest) <= SIBLING_REST_NAMED:
        head += f"다른 업무라면 {', '.join(o.label for o in rest)}도 고를 수 있습니다. "  # 이름 안의 '·' 와 구분되도록 쉼표 사용
    elif rest:
        head += "그 밖의 선택지도 아래에서 고를 수 있습니다. "
    return head + "원하시는 업무를 말씀해 주세요."


def agent_alternative(goal: str, best: Option, options: list[Option]) -> Option | None:
    """메뉴의 선택지가 best 와 상담원 연결 둘뿐이고 목적에 best 를 가리키는 말이 없으면 그 상담원 선택지 반환.
    '문자로 받기 / 상담원 연결' 처럼 사람과 통화할지 자동으로 처리할지는 사용자가 정할 일이므로 AI 가 대신 선택하지 않음
    (예: 목적 '카드 발급 방법 알려줘' 에서 '문자 수신' 쪽 점수가 더 높다는 이유만으로 자동 선택하는 문제 방지).
    목적이 best 의 낱말('문자로 받고 싶어')을 담으면 질문 없음. 일반 낱말(서비스·안내·문의 등)은 가리키는 말로 보지 않음."""
    if len(options) != 2 or is_agent_option(best):
        return None
    agent = next((o for o in options if o is not best and o.key != best.key and is_agent_option(o)), None)
    if agent is None:
        return None
    g = goal.replace(" ", "").lower()
    a = agent.label.replace(" ", "").lower()  # 두 선택지가 함께 가진 낱말('카드 발급 문자 안내 / 카드 발급 상담원 연결')은 best 를 가리키는 말에서 제외
    if any(len(c) >= 2 and c not in _GENERIC and c.lower() not in a and c.lower() in g for c in _chunks(best.label)):
        return None
    return agent


# 목적 문장의 낱말 끝에서 떼는 조사, 그리고 서술어·어미로 끝나 대상 이름이 아닌 낱말 ('발급받고'·'싶어'·'궁금해'·'뭔지'·'발급받을래')
_PARTICLE = re.compile(r"(?:에서|으로|이랑|부터|까지|을|를|이|가|은|는|에|의|로|도|만|과|와|랑)$")
_PREDICATE = re.compile(r"(?:고|어|아|요|줘|게|지|해|면|다|니|까|데|려|래|서요|나요|세요|려서|어서|아서|해서|돼서|여서|으러)$")
# 붙여 쓴 서술어 꼬리 ('해지하고'·'재발급받고'·'납부하려는데'·'발급하는'). 떼고 남은 앞부분을 목적 낱말 줄기로 사용
_LIGHT_VERB = re.compile(r"(?:하려고|하려는데|하고싶어|하고|하는|하기|하면|해서|해야|해줘|했어|할래|하|받고|받을래|받으려고|받는|받|되고|돼서|되는|된|되)$")
# 대상 이름이 아닌 일반 명사·부사 (일반 낱말에 더해 '방법'·'관련'·'다시' 처럼 목적 문장에 흔히 붙는 말)
_GOAL_FILLER = _GENERIC | {"방법", "관련", "내용", "정보", "경우", "여부", "하나", "같은", "어떻게", "얼마", "언제", "어디",
                           "다시", "지금", "빨리", "혹시", "새로", "바로", "제가", "저는", "내가", "나는"}


def _strip_particle(w: str) -> str:
    """조사 제거. 떼고 한 글자만 남는 '도·과·의·만·가·이·와' 는 조사가 아닌 낱말의 일부로 판정 ('한도'·'결과'·'속도'·'명의').
    '는·은·을·를' 로 끝나는 두 글자('받는'·'없는')는 꾸미는 말이므로 떼어 한 글자로 남김 (이후 단계에서 제외)."""
    t = _PARTICLE.sub("", w)
    return t if len(t) >= 2 or not re.search(r"[도과의만가이와]$", w) else w


def goal_words(goal: str) -> list[str]:
    """목적 문장에서 대상 이름으로 볼 낱말 목록 (순서 유지): '운전면허증 발급받고 싶어' → ['운전면허증'],
    '이사 가는데 우편물을 새 주소로 받는 방법 알고 싶어' → ['이사', '우편물', '주소']. 조사 제거, 서술어·일반 낱말·한 글자 제외.
    중심 낱말(마지막 낱말) 선택용. 메뉴 이름·안내가 목적을 말했는지 판단할 때는 줄기까지 넓게 보는 _goal_mention_words 사용."""
    out: list[str] = []
    for w in re.findall(r"[가-힣A-Za-z0-9]+", goal or ""):
        if _PREDICATE.search(w) and not _PARTICLE.search(w):
            continue
        w = _strip_particle(w)
        if len(w) >= 2 and w.lower() not in _GOAL_FILLER and not _PREDICATE.search(w):
            out.append(w.lower())
    return out


def _goal_mention_words(goal: str) -> tuple[list[str], list[str]]:
    """메뉴 이름·안내가 목적을 말했는지 볼 낱말 두 묶음.
    (1) 대상 낱말과 붙여 쓴 서술어 꼬리를 뗀 줄기 ('해지하고' → '해지', '재발급받고' → '재발급', '한도를' → '한도'): 메뉴 이름·안내 전체에서 검색.
    (2) 서술어 어미 모양으로 끝나 대상 낱말에서 빠진 낱말 ('신고'·'해지'·'정지'·'알고'·'싶어'): 명사 여부를 알 수 없으므로
        지나온 메뉴 이름에 그대로 있을 때만 언급으로 판정 (안내 문장의 '발급받을 수' 같은 서술어와의 겹침 방지)."""
    loose: list[str] = list(goal_words(goal))
    label_only: list[str] = []
    for w in re.findall(r"[가-힣A-Za-z0-9]+", goal or ""):
        w = _strip_particle(w).lower()
        t = _LIGHT_VERB.sub("", w)
        if t != w:
            if len(t) >= 2 and t not in _GOAL_FILLER:
                loose.append(t)
        elif len(w) >= 2 and w not in _GOAL_FILLER and w not in loose:
            label_only.append(w)
    return loose, label_only


def _mentions(word: str, text: str) -> bool:
    """text(공백 없앤 글)의 목적 낱말 포함 여부. 세 글자 이상이면 앞 세 글자로 비교 ('업무시간' ↔ '업무 시간 안내')."""
    return word in text or (len(word) >= 3 and word[:3] in text)


def _names_head(head: str, name: str) -> bool:
    """선택지 이름의 중심 낱말 포함 여부: 낱말 그대로, 또는 네 글자 이상이면 끝 글자를 뺀 앞부분 ('운전면허증' ↔ '운전면허 갱신').
    앞 세 글자만으로는 비교하지 않음 ('주민등록등본' ↔ '주민등록증 재발급', '서비스센터' ↔ '부가서비스' 는 다른 대상)."""
    s = re.sub(r"\s", "", name).lower()
    return head in s or (len(head) >= 4 and head[:-1] in s)


def better_sibling(goal: str, chosen: str, heard: str, siblings: list[str]) -> str | None:
    """막다른 안내가 목적의 답이라는 모델 판단의 낱말 기반 재확인. 목적의 낱말(줄기 포함)이 지나온 메뉴 이름(chosen)에도
    그 뒤 안내(heard)에도 하나도 없고, 목적의 중심 낱말(목적 문장의 마지막 대상 낱말)을 이름에 담은 다른 업무 선택지가
    딱 하나 있으면 그 선택지 이름 반환 (다른 갈래에서 들은 본인인증 같은 안내를 모델이 목적의 답으로 오판하는 경우 대비).
    낱말 일치만 보는 규칙이므로 같은 뜻의 다른 말('택배'·'배송')을 쓰는 맞는 갈래에서도 반환 가능. 따라서 호출하는 쪽은
    이 값을 '다른 갈래'로 쓰지 않고, 결과로 처리하는 대신 사용자에게 묻는 근거로만 사용.
    None 인 경우: 목적 낱말이 지나온 메뉴 이름·안내에 하나라도 있음(고른 메뉴 이름의 조각이 목적 낱말 안에 있는 것 포함,
    '보험료' ↔ '자동차보험료'), 중심 낱말을 담은 선택지가 없거나 둘 이상(그 낱말로 갈래 구분 불가). 상담원 연결 선택지는 제외."""
    words = goal_words(goal)
    if not words:
        return None
    path = re.sub(r"\s", "", chosen).lower()
    ctx = path + re.sub(r"\s", "", heard or "").lower()
    loose, label_only = _goal_mention_words(goal)
    if any(_mentions(w, ctx) for w in loose) or any(w in path for w in label_only):
        return None
    pieces = [c.lower() for c in _chunks(chosen) if len(c) >= 2 and c not in _GENERIC]
    if any(c in w for c in pieces for w in loose):
        return None
    head = words[-1]
    hits = [name for name in siblings if not is_agent_option(Option("", name)) and _names_head(head, name)]
    return hits[0] if len(hits) == 1 else None


def chosen_names_goal(goal: str, chosen: str) -> bool:
    """고른 메뉴 이름의 목적 중심 낱말(목적 문장의 마지막 대상 낱말) 포함 여부 ('청약통장 해지하면 다시 만들 수 있나요' ↔ '청약통장 안내').
    모델이 막다른 안내를 다른 갈래로 판단해도 이 경우에는 낱말 근거와 엇갈리므로 자동 복귀 대신 질문하는 근거로 사용."""
    words = goal_words(goal)
    return bool(words) and bool(chosen) and _names_head(words[-1], chosen)


@dataclass
class Layer:
    """키를 눌러 지나온 메뉴 한 단계."""
    prompt: Prompt
    key: str
    actor: str  # ai / user


@dataclass
class Decision:
    action: str                 # press / ask / backtrack / end
    key: str | None = None      # press: 누를 키
    keys: str | None = None     # backtrack: 돌아가기 위해 누를 키 열
    reason: str = ""            # ask: 재질문 사유 / end: end_reason
    note: str = ""              # 사람이 읽는 판단 근거
    options: list[Option] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    message: str = ""           # ask: 기본 문구 대신 쓸 질문 (비어 있으면 reason 의 기본 문구)


NOT_FOUND_ASK = "원하시는 업무와 맞는 메뉴를 찾지 못했습니다. 아래에서 고르시거나 원하시는 업무를 조금 더 자세히 말씀해 주세요."
# 목적에 상담원 요청이 없는데 상담원 연결이 가장 맞는 선택지로 나왔을 때의 질문 (선택지가 상담원 하나뿐이면 둘째 문구)
AGENT_UNASKED_ASK = ("상담원 연결이 가장 가까운 선택지로 보입니다. 상담원과 통화하시려면 고르시고, 아니면 다른 선택지를 고르시거나 "
                     "원하시는 업무를 조금 더 자세히 말씀해 주세요.")
AGENT_UNASKED_ONE_ASK = "이 안내에는 상담원 연결 번호만 있습니다. 상담원과 통화하시려면 고르시고, 아니면 원하시는 업무를 조금 더 자세히 말씀해 주세요."
# 상담원이라는 말 없이 사람과의 통화를 원하는 목적 ('사람이랑 얘기할래', '직원 연결', '상담 받고 싶어'). 모든 ARS 공통 표현
_HUMAN_REQ_RE = re.compile(r"사람\s?(?:이랑|하고|과|와|한테)|직원|상담\s?(?:받|하고|좀)")


def wants_agent(goal: str | None) -> bool:
    """목적이 사람(상담원)과의 통화를 원하는지: 상담원 말(AGENT_LABEL_RE) 또는 사람·직원 요청 표현."""
    g = goal or ""
    return bool(AGENT_LABEL_RE.search(g) or _HUMAN_REQ_RE.search(g))
NO_AGENT_ASK = ("이 메뉴에는 상담원 연결 선택지가 없습니다. ARS 안내(상담 가능 시간 등)를 확인해 주세요. "
                "상담하실 업무를 말씀해 주시거나 아래에서 메뉴를 고르시면 그 안에서 상담원 연결을 찾겠습니다.")

# 목적의 '상담원 연결' 말 (공통 표현: 상담원·상담사·상담직원 + 연결·통화 등의 서술). 업무 부분만 남길 때 제거
_AGENT_GOAL_RE = re.compile(r"(?:" + AGENT_LABEL_RE.pattern + r")[가-힣]*(?:\s*(?:연결|통화|대화|바꿔|이어|부탁)[가-힣]*)*")


AGENT_TOPIC_MAX = 2  # 상담원 선택지가 없을 때 업무 부분으로 메뉴에 들어가는 최대 횟수 (탐색당). 초과 시 상담원을 찾지 못했다고 질문


def agent_topic(goal: str) -> str:
    """목적이 상담원 연결일 때 그 상담의 업무 부분: 목적에서 상담원 말만 뺀 문장 ('카드 분실 관련 상담원 연결해줘' → '카드 분실 관련',
    '교통사고 접수 상담원 연결해줘' → '교통사고 접수'). 업무 말은 거르지 않음 (점수를 매길 목적 문장이므로). 대상 이름 낱말이 없으면
    빈 문자열 ('상담원 연결해줘'·'상담사랑 통화하고 싶어' → '')."""
    t = re.sub(r"\s+", " ", _AGENT_GOAL_RE.sub(" ", goal or "")).strip()
    return t if goal_words(t) else ""


def agent_goal(goal: str | None) -> str:
    """상담원 연결 요청(세션 생성 agent=true)의 목적: 사용자가 함께 입력한 업무 뒤에 '상담원 연결' 을 붙임
    ('카드 분실' → '카드 분실 상담원 연결'). 이미 상담원 말이 있으면 그대로, 업무를 입력하지 않았으면 '상담원 연결'. 첫 메뉴에 상담원
    선택지가 없으면 업무 부분(agent_topic)으로 맞는 메뉴에 들어가 그 안에서 상담원 연결을 탐색(_agent_topic_press). 업무가 없으면
    상담원 선택지가 없는 메뉴에서 사용자에게 업무를 질문(NO_AGENT_ASK)."""
    g = re.sub(r"\s+", " ", goal or "").strip()
    if not g:
        return "상담원 연결"
    return g if AGENT_LABEL_RE.search(g) else f"{g} 상담원 연결"


@dataclass
class NavState:
    goal: str
    backtracked: bool = False           # 이전 계층 복귀 사용 여부
    exclude: set[str] = field(default_factory=set)  # 복귀한 계층에서 제외할 키 (이미 시도한 키)
    exclude_at: str | None = None       # exclude 를 적용할 계층 (메뉴 signature). 다른 메뉴에는 미적용
    presses: int = 0                    # AI 가 현재 탐색에서 누른 업무 선택 키 수
    visits: Counter = field(default_factory=Counter)
    revisits: int = 0                   # 키 전송 결과로 이미 지나온 메뉴에 다시 도착한 횟수
    low_asked: set[str] = field(default_factory=set)  # low_confidence 재질문을 이미 한 계층
    sibling_asked: set[str] = field(default_factory=set)  # 같은 대상의 종류를 고르는 재질문을 이미 한 계층
    agent_asked: set[str] = field(default_factory=set)    # '선택지 하나 / 상담원 연결' 재질문을 이미 한 계층
    # 사용자가 재질문에서 '이전 메뉴'·'처음으로'로 나온 갈래 (메뉴 signature → 키). AI 가 키를 눌러도 유지
    rejected: dict[str, set[str]] = field(default_factory=dict)
    # 상담원 목적: 상담원 선택지가 없는 메뉴에서 업무 부분으로 누른 횟수, 그렇게 들어간 탐색인지 여부 (결과 안내를 업무 완료로 보지 않음)
    topic_presses: int = 0
    agent_seek: bool = False

    def reset_loop(self) -> None:
        """반복(loop) 재질문에 사용자가 응답한 뒤 반복 판정 횟수 초기화."""
        self.visits.clear()
        self.revisits = 0
        self.presses = 0


class Navigator:
    def __init__(self, interp: Interpreter, threshold: float = 0.5, margin: float = 0.1,
                 max_revisit: int = 2, max_presses: int = 5, low_threshold: float = 0.3) -> None:
        self.interp = interp
        self.threshold, self.margin, self.low = threshold, margin, low_threshold
        self.max_revisit, self.max_presses = max_revisit, max_presses

    async def _rank(self, goal: str, opts: list[Option], path: list[str]) -> tuple[list[tuple[Option, float]], dict[str, float]]:
        scores = await self.interp.score(goal, opts, path) if opts else {}
        # 동점이면 업무 선택지를 상담원 연결보다 앞에 배치 (그 밖에는 선택지 목록 순서). 1·2위 차이 기준이 0 보다 크면
        # (config 의 0.10 고정값) 1위 동점은 언제나 차이 재질문이므로 누를 키는 그대로, 재질문 선택지 순서만 변경
        ranked = sorted(((o, scores.get(o.key, 0.0)) for o in opts),
                        key=lambda x: (x[1], not is_agent_option(x[0])), reverse=True)
        return ranked, scores

    async def _agent_topic_press(self, st: NavState, current: Prompt, opts: list[Option], path: list[str],
                                 ranked: list[tuple[Option, float]], no_agent: bool) -> Decision | None:
        """목적이 상담원 연결인데 이 메뉴에 상담원 선택지가 없으면, 목적의 업무 부분('카드 분실 관련')으로 맞는 업무 메뉴를 골라 들어가
        그 안에서 상담원 연결 탐색. 상담원 말이 든 목적 전체로 점수를 매기면 업무 선택지가 모두 낮게 나오기 때문. None 이면 원래 판단 유지:
        - 원래 목적으로 정해지면(일치 기준·1·2위 차이 통과) 그대로 따름 ('상담을 원하시면 0번' 처럼 이름 정규식이 상담원 선택지로 못 본
          선택지를 원래 목적 점수가 맞게 고른 경우를 덮어쓰지 않도록, 그리고 추가 점수 요청이 이 경우에만 생기도록)
        - 메뉴 단락에 상담원 말이 있는데 선택지가 없으면('상담사 연결은 평일 업무시간에 …') 상담 불가 안내일 수 있어 들어가지 않고 안내
        - 업무 부분이 없거나('상담원 연결해줘'), 업무 부분으로도 확신이 없거나, 같은 종류 선택지면 질문
        - 업무 부분으로 누르는 것은 탐색당 AGENT_TOPIC_MAX 번까지 (초과 시 상담원을 찾지 못했다고 질문)"""
        if not no_agent or st.topic_presses >= AGENT_TOPIC_MAX:
            return None
        if ranked and ranked[0][1] >= self.threshold and _gap(ranked[0][1], ranked[1][1] if len(ranked) > 1 else 0.0) >= self.margin:
            return None
        if AGENT_LABEL_RE.search(current.text or ""):
            return None
        topic = agent_topic(st.goal)
        if not topic:
            return None
        t_ranked, t_scores = await self._rank(topic, opts, path)
        if not t_ranked or t_ranked[0][1] < self.threshold:
            return None
        best, s1 = t_ranked[0]
        if _gap(s1, t_ranked[1][1] if len(t_ranked) > 1 else 0.0) < self.margin:
            return None
        if sibling_choices(topic, best, [o for o, _ in t_ranked[1:]]):
            return None
        st.topic_presses += 1
        st.agent_seek = True
        return Decision("press", key=best.key, scores=t_scores,
                        note=f"상담원 선택지 없음: 업무 부분 '{topic}' 으로 '{best.label}' 일치율 {s1:.2f} (그 안에서 상담원 연결을 찾음)")

    # 규칙 판단 훅. 기본은 규칙 함수 그대로 호출 (llm 조합의 LlmNavigator 만 모델 판단으로 대체, llm_mode.py)
    def _is_agent(self, o: Option) -> bool:
        return is_agent_option(o)

    def _goal_names_agent(self, goal: str) -> bool:
        return bool(AGENT_LABEL_RE.search(goal or ""))

    def _wants_agent(self, goal: str) -> bool:
        return wants_agent(goal)

    def _twins(self, goal: str, best: Option, rest: list[Option]) -> list[Option]:
        return sibling_choices(goal, best, rest)

    def _agent_alt(self, goal: str, best: Option, options: list[Option]) -> Option | None:
        return agent_alternative(goal, best, options)

    async def decide(self, st: NavState, current: Prompt, history: list[Layer], arrived: bool = True,
                     revised: bool = False, user_branch: bool | None = None) -> Decision:
        """arrived=True: 키 전송 결과로 새로 도착한 메뉴. 재질문 답변 후 같은 메뉴를 다시 판단할 때는 False.
        revised: 재질문에 글로 답해 목적이 바뀐 뒤의 재판단. user_branch: 지금 메뉴가 사용자가 고른 갈래인지 (None 이면 마지막 계층 기준으로 판정)."""
        sig = current.signature()
        excluded = (st.exclude if st.exclude_at == sig else set()) | st.rejected.get(sig, set())
        opts = [o for o in current.options if o.key not in excluded]

        # 반복 감지: 키를 눌러 이미 지나온 메뉴에 다시 도착한 경우만 집계 (API 명세 7장 5항).
        # 명세 6.3: '이미 지나온 메뉴가 다시 나오는 상황이 2회 발생' 또는 키 전송 5회
        # 확인 질문("맞으면 1번/틀리면 2번")은 여러 단계에서 같은 모양으로 되풀이되므로 반복 집계 제외
        if arrived and current.kind == "menu":
            if st.visits[sig]:
                st.revisits += 1
            st.visits[sig] += 1
        # MAX_REVISIT=0 / MAX_PRESSES=0 이면 그 판정 끔
        if (self.max_revisit and st.revisits >= self.max_revisit) or (self.max_presses and st.presses >= self.max_presses):
            note = (f"지나온 메뉴에 다시 도착 {st.revisits}회" if st.revisits >= self.max_revisit
                    else f"키 {st.presses}회 입력 후 미도달")
            return Decision("ask", reason="loop", note=note, options=current.options)

        if current.critical:
            return Decision("ask", reason="critical_action", note="중요 선택", options=current.options)

        path = [layer.prompt.category_of(layer.key) for layer in history]
        no_agent = self._goal_names_agent(st.goal or "") and not any(self._is_agent(o) for o in current.options)
        ranked, scores = await self._rank(st.goal, opts, path)
        topic_dec = await self._agent_topic_press(st, current, opts, path, ranked, no_agent)
        if topic_dec is not None:
            return topic_dec
        if ranked and ranked[0][1] >= self.threshold:
            best, s1 = ranked[0]
            s2 = ranked[1][1] if len(ranked) > 1 else 0.0
            if _gap(s1, s2) >= self.margin:
                twins = self._twins(st.goal, best, [o for o, _ in ranked[1:]]) if sig not in st.sibling_asked else []
                if twins:
                    # 같은 대상의 종류를 고르는 선택지이고 목적에 그 구분 없음. 모델 점수가 높아도 사용자 선택 (계층당 1회).
                    # 같은 메뉴의 나머지 업무 선택지도 점수 순으로 뒤에 붙여 선택 가능하게 함 (머리말이 다른 업무 선택지가 질문에서 빠지는 것 방지).
                    # 질문 문구는 같은 종류끼리만 견주고 나머지는 있다는 사실만 안내
                    st.sibling_asked.add(sig)
                    rest = [o for o, _ in ranked if o is not best and all(o is not t for t in twins)]
                    return Decision("ask", reason="ambiguous", note=f"'{best.label}' 과 '{twins[0].label}' 중 목적만으로 구분 불가",
                                    options=[best] + twins + rest, scores=scores, message=sibling_message([best] + twins, rest))
                agent = self._agent_alt(st.goal, best, opts) if sig not in st.agent_asked else None
                if agent:
                    # 남은 선택지가 이것과 상담원 연결뿐. 사람과 통화할지 자동으로 처리할지는 사용자 선택 (계층당 1회)
                    st.agent_asked.add(sig)
                    return Decision("ask", reason="ambiguous", note=f"'{best.label}' 과 '{agent.label}' 중 사용자가 고를 일",
                                    options=[best, agent], scores=scores)
                if self._is_agent(best) and not self._wants_agent(st.goal) and sig not in st.agent_asked:
                    # 목적에 상담원 요청이 없는데 상담원 연결이 1위: 받아쓰기가 업무 선택지를 빠뜨렸거나 맞는 업무가 없는 메뉴일 가능성 있음.
                    # 사람과 통화할지는 사용자가 고른다 (계층당 1회. 2026-11-10 우체국 실통화: 첫 메뉴 1~5번이 받아쓰기에서 빠져 '운영시간'
                    # 목적에 상담사 연결 0.50 을 바로 누름)
                    st.agent_asked.add(sig)
                    return Decision("ask", reason="ambiguous", note=f"목적에 상담원 요청이 없는데 '{best.label}' 이 1위 ({s1:.2f})",
                                    options=[o for o, _ in ranked], scores=scores,
                                    message=AGENT_UNASKED_ASK if len(ranked) > 1 else AGENT_UNASKED_ONE_ASK)
                return Decision("press", key=best.key, note=f"'{best.label}' 일치율 {s1:.2f}", scores=scores)
            # 1위와 차이가 기준 미만인 선택지 모두 (2위가 일치 기준 0.5 미만이어도 포함. 빼면 선택지 하나짜리 질문이 됨)
            cands = [o for o, s in ranked if _gap(s1, s) < self.margin]
            return Decision("ask", reason="ambiguous", note=f"1·2순위 차이 {s1 - s2:.2f}", options=cands, scores=scores)

        # 불확실 → 사용자에게 재질문 (계층당 1회)
        best_s = ranked[0][1] if ranked else 0.0
        # 확신이 없거나 맞는 선택지가 없을 때 사용자에게 보여 줄 선택지: 점수 순 5개, 5번째와 차이 0.1 미만인 선택지는 모두
        # (0.3 이상만 보이면 숨은 하위 메뉴를 담은 선택지('각종 안내' 등)를 사용자가 고를 수 없는 문제 방지)
        shown = shown_options(ranked, self.margin)
        if not shown and st.rejected.get(sig):
            shown = list(current.options)  # 사용자가 갈래를 모두 거절한 경우. 종료 전에 전체 선택지를 보여 주고 질문 (점수가 없어 자를 근거도 없음)
        # 목적이 상담원 연결인데 이 메뉴에 상담원 선택지가 없으면(no_agent) 그 사실을 안내 (상담 가능 시간 밖 등으로 ARS 가 상담원 연결을
        # 제공하지 않을 때 일반 재질문 문구로만 묻는 문제 방지). 상담원 연결은 모든 ARS 공통 표현
        if best_s >= self.low and sig not in st.low_asked:
            st.low_asked.add(sig)
            return Decision("ask", reason="low_confidence", note=f"최고 신뢰도 {best_s:.2f} (기준 {self.threshold:.2f} 미만)",
                            options=shown, scores=scores, message=NO_AGENT_ASK if no_agent else "")

        def ask_or_end(note: str) -> Decision:
            # 맞는 선택지 없음. 통화 종료 전에 이 계층에서 1회 사용자에게 질문 (첫 계층에서 바로 종료되던 문제 방지)
            if shown and sig not in st.low_asked:
                st.low_asked.add(sig)
                return Decision("ask", reason="low_confidence", note=f"{note} — 끝내기 전에 묻는다", options=shown,
                                scores=scores, message=NO_AGENT_ASK if no_agent else NOT_FOUND_ASK)
            return Decision("end", reason="not_found", note=note, scores=scores)

        # 불일치 → 이전 계층 확인
        ub = user_branch if user_branch is not None else bool(history) and history[-1].actor == "user"

        def user_ask(note: str) -> Decision:
            # 사용자가 고른 갈래(키패드로 새 메뉴를 듣기 전에 누른 키·민감정보 입력 뒤 포함)에서 맞는 선택지 없음: 스스로 되돌아가
            # 사용자가 고르지 않은 갈래를 누르지 않고 질문 (막다른 안내의 사용자 갈래 규칙과 동일. 사용자가 고른 업무 선택지의 하위 메뉴 점수가
            # 낮다는 이유로 이전 메뉴로 돌아가 1위 선택지를 누르는 것 방지). 이 계층에서 이미 물었어도,
            # 이번 탐색에서 이미 되돌아갔어도 종료하지 않고 하위 선택지와 이동 키로 다시 질문
            if shown or current.options:
                st.low_asked.add(sig)
                return Decision("ask", reason="low_confidence", note=f"{note} (최고 {best_s:.2f})",
                                options=shown or list(current.options), scores=scores,
                                message=NO_AGENT_ASK if no_agent else NOT_FOUND_ASK)
            return ask_or_end(note)

        # 재질문에 글로 답해 목적이 바뀌었으면(revised) 사용자의 새 말이 앞선 선택보다 우선이므로 아래의 이전 계층 확인(탐색당 1회
        # 되돌아가기) 사용. 되돌아갈 수 없으면(이전 계층에도 맞는 선택지 없음·돌아가는 키 없음) 종료하지 않고 다시 질문
        if ub and history and not (revised and not st.backtracked):
            return user_ask("사용자가 고른 갈래에 일치 선택지 없음")
        if st.backtracked:
            return ask_or_end(f"이전 계층 확인 후에도 일치 선택지 없음 (최고 {best_s:.2f})")
        if not history:
            return ask_or_end(f"현재 계층에 일치 선택지 없음, 이전 계층 정보 없음 (최고 {best_s:.2f})")

        prev = history[-1]
        prev_sig = prev.prompt.signature()
        prev_excluded = (st.exclude if st.exclude_at == prev_sig else set()) | st.rejected.get(prev_sig, set())
        alts = [o for o in prev.prompt.options if o.key != prev.key and o.key not in prev_excluded]  # 사용자가 거절한 갈래는 대안에서 제외
        # 상담원을 찾으러 업무 부분으로 들어온 갈래면, 이전 계층의 상담원 없는 메뉴 대안도 같은 업무 부분 기준으로 점수 산정
        alt_goal = st.goal
        if st.agent_seek and not any(is_agent_option(o) for o in prev.prompt.options):
            alt_goal = agent_topic(st.goal) or st.goal
        alt_ranked, _ = await self._rank(alt_goal, alts, path[:-1])
        if not alt_ranked or alt_ranked[0][1] < self.threshold:
            return user_ask("사용자 갈래·이전 계층 모두 일치 선택지 없음") if ub else ask_or_end("현재·이전 계층 모두 일치 선택지 없음")

        if current.back_key:
            keys = current.back_key
        elif current.home_key and all(layer.key for layer in history[:-1]):
            keys = current.home_key + "".join(layer.key for layer in history[:-1])
        else:
            note = "이전 계층에 후보가 있으나 돌아가는 키(이전 메뉴·처음으로)가 안내되지 않음"
            return user_ask(note) if ub else ask_or_end(note)
        return Decision("backtrack", keys=keys,
                        note=f"이전 계층의 '{alt_ranked[0][0].label}' 확인을 위해 복귀", scores=scores)
