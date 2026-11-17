"""ARS 안내 한 단락에서 누를 수 있는 키·역할·이름, 발화 종류, 입력 요청·끝맺음 키를 정규식으로 추출.

extract(text) -> {"kind", "keys": [{"key", "role", "label"}], "input": {"request", "terminator"}, "critical",
                  "certain", "uncertain_reasons"}
판단은 모두 정규식으로 하고, 파이썬은 나누기·순서·짝짓기만 담당. 기관 이름·업무 이름은 사용하지 않음.
certain 이 참이면 정규식 결과만으로 정해도 되는 발화, 거짓이면 로컬 모델에 물을 발화 (아래 6. 확실함 판정).
certain 이 참이면 critical 은 거짓이고, 거래·확정·승인 단서도 없음 (최종 승인 판단은 정규식이 확정하지 않음).
uncertain_reasons 는 확실하지 않은 까닭 목록.
"""
from __future__ import annotations

import re

# ===========================================================================
# 1. 정규화: 말로 한 키 이름을 기호·숫자로 변환
# ===========================================================================
# 받아적기 주석 괄호 ('(?)', '(<설명>)'): ARS 가 말한 글이 아니므로 삭제 (30자 이하, 키 표현이 없는 것만)
_PAREN = re.compile(r"\((?![^()]*(?:\d\s*번|별표|샵|우물\s?정자))[^()]{0,30}\)")
# 무대 지시 '(<N> 입력 후)'·'(<항목> 입력 후)': 맨 앞이면 직전 단계 표시라 지우고, 글 중간이면 키를 누른 뒤의 다음 안내가 이어
# 붙은 것이라 그 앞 안내만 남김
_STAGE = re.compile(r"\([^()]{0,20}?(?:입력|누른|누르신|선택)\s?후\)")
# 여러 자리 한자어 숫자 + '번' ('일일일 번' → '111번')
_DIGIT_RUN = re.compile(r"(?<![가-힣])([공영일이삼사오육칠팔구]{2,3})\s?번(?=\s*(?:(?P<topic>은|는)|을|를|,|\.|$|누르|눌러))")
# 누르라는 말 (세 자리 번호 키, '…은/는' 앞 한자어 숫자 키의 조건)
_PRESS_WORD = re.compile(r"누르|눌러|선택|입력|press", re.I)
# '#' 의 공통 이름: 샵·샾(+버튼·키), 우물정자, 해시. '샵' 은 낱말 첫머리이고 뒤에 한글이 바로 붙지 않을 때만
# (조사·'버튼' 은 허용. '샵<글자>' 처럼 낱말 속 '샵' 은 키가 아님)
_HASH = re.compile(r"(?:(?<![가-힣A-Za-z])|(?<=[은는]))(?:샵|샾)(?:\s?(?:버튼|키))?(?=$|[^가-힣]|[을를은는이과와]|으로|로)"
                   r"|우물\s?정\s?자(?:\s?(?:버튼|키))?|해시\s?(?:키|버튼)|#\s?(?:버튼|키)")
# '*' 의 공통 이름: 별표·별 모양(+버튼·키), 낱말 첫머리의 '별 버튼·별키' ('특별 …' 같은 낱말 속 '별' 은 제외)
_STAR = re.compile(r"별\s?(?:표|모양)(?:\s?(?:버튼|키))?|(?<![가-힣])별\s?(?:버튼|키)|\*\s?(?:버튼|키)")
# 한자어 숫자 한 글자 + '번' ('삼 번' → '3번'). 앞에 한글이 붙은 말('이번'·'사번')은 제외, 뒤가 조사·구두점·누르라는 말일 때만.
# 뒤가 주제 조사 '은/는' 이면 문장 첫머리의 '이번은 …' 과 겹치므로, 발화에 누르라는 말이 있을 때만 변환 (normalize)
_SINO = {"공": "0", "영": "0", "빵": "0", "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5", "육": "6", "륙": "6",
         "칠": "7", "팔": "8", "구": "9"}
_SINO_KEY = re.compile(r"(?<![가-힣])([공영빵일이삼사오육륙칠팔구])\s?번(?=\s*(?:(?P<topic>은|는)|을|를|,|\.|$|누르|눌러))")
# '…는 한 번을 누르세요' 처럼 조사 '은/는' 바로 뒤의 고유어 숫자 + '번', 바로 뒤에 누르라는 말이 올 때만
# ('다시 한 번 듣기', '<명사>는 한 번, …' 같은 횟수는 그대로)
_NATIVE = {"한": "1", "두": "2", "세": "3", "네": "4", "다섯": "5", "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9"}
_NATIVE_KEY = re.compile(r"([은는])\s*(한|두|세|네|다섯|여섯|일곱|여덟|아홉)\s?번(?=\s*(?:을|를)?\s*(?:누르|눌러))")
# 영어 키 표현, 이름이 앞: '<이름>, press 6' · 'press number 9' → '<이름>는 6번'
_PRESS_AFTER = re.compile(r"(?<=[A-Za-z가-힣])\s*,?\s*\b[Pp]ress\s+(?:number\s+)?([0-9])\b(?!\s+for\b)")
# 영어 키 표현, 키가 앞: 'press 2 for English' · 'or 1 for Korean' → ', English는 2번'
_PRESS_FOR = re.compile(r"\b(?:[Pp]ress|or)\s+(?:number\s+)?([0-9])\s+for\s+([A-Za-z][A-Za-z ]{1,30}?)(?=\s*(?:,|\.|or\b|$))")
# '숫자 <N>와 …', '숫자 <N>를 누르신 후' : 숫자 이름으로 말한 키 (명령형 문장에서만). 값의 조건을 말하는
# '숫자 <N>을 포함한·숫자 <N>과 <M> 사이·<N>부터·<N> 이상' 은 키가 아님
_DIGIT_WORD = re.compile(r"숫자\s?([0-9])(?=\s*(?:와|과|를|을|,))(?![^.,]{0,8}?(?:포함|사이|부터|까지|이상|이하))")
# 숫자 하나 + '을/를 누르' ('1을 누르세요', '0을 눌러'): '번' 을 붙여 키 표현으로 통일 ('숫자 N' 꼴과 금액·번호 끝자리는 제외)
_BARE_DIGIT = re.compile(r"(?<![\d\-.,:/~○])(?<!숫자)(?<!숫자\s)(\d)(?=\s*(?:을|를)\s*(?:눌러|누르))")


def normalize(text: str) -> str:
    t = (text or "").replace("\n", " ")
    for m in _STAGE.finditer(t):
        if t[:m.start()].strip():
            t = t[:m.start()]
            break
    t = _STAGE.sub(" ", t)
    t = _PAREN.sub(" ", t)
    press = bool(_PRESS_WORD.search(t))
    t = _DIGIT_RUN.sub(lambda m: m.group(0) if m.group("topic") and not press
                       else "".join(_SINO[c] for c in m.group(1)) + "번", t)
    t = _SINO_KEY.sub(lambda m: m.group(0) if m.group("topic") and not press else _SINO[m.group(1)] + "번", t)
    t = _BARE_DIGIT.sub(r"\1번", t)
    t = _NATIVE_KEY.sub(lambda m: f"{m.group(1)} {_NATIVE[m.group(2)]}번", t)
    t = _HASH.sub("#", t)
    t = _STAR.sub("*", t)
    t = _PRESS_FOR.sub(lambda m: f", {m.group(2).strip()}는 {m.group(1)}번", t)
    t = _PRESS_AFTER.sub(lambda m: f"는 {m.group(1)}번", t)
    return re.sub(r"[ \t]+", " ", t)


# ===========================================================================
# 2. 키 찾기
# ===========================================================================
# 요청 문장의 끝맺음 (명령형·청유형): '…주십시오/주십시요/주세요', '…십시오', '…세요', '…바랍니다', '…주시기', '…시면 됩니다'.
# 받아쓰기 띄어쓰기('주 십 시 오')와 '십시요' 철자도 허용
_IMP_END = r"(?:주\s?십\s?시\s?[오요]|주\s?세\s?요|십시[오요]|세요|바랍니다|주시기|시면\s?됩니다)"
# 누르라는 지시 (서술격 꼬리 키, 쉼표 없는 나열의 조건): 누르다·입력하다·선택하다 + 요청 끝맺음, '누르시면', 영어 press.
# '입력하신·선택하신' 같은 설명 말은 지시가 아님
_PRESS_ORDER = re.compile(r"(?:누르|눌러|입력|선택)[가-힣\s]{0,6}?" + _IMP_END + r"|누르시면|눌러\s?주시면|press", re.I)
# 키 표현: 숫자(1~3자리)+'번', 또는 기호 '*'·'#'. 앞에 숫자·하이픈·마침표가 붙은 숫자(전화번호·금액 끝자리)와
# '번호·번째·번지·번 출구', 횟수('3번 이상') 는 키가 아님
_KEY = re.compile(r"(?<![\d\-.,:/~○])(?<!\d\s)(?P<num>\d{1,3})\s*번"
                  r"(?!호|째|지|길|\s*출구|\s*(?:이상|이하|이내|까지|만에|연속|정도|씩)|\s*(?:으로|로)\s*(?:전화|연락|걸))|(?<![*#\d])(?P<sym>[*#])(?![*#\d])")
# 키 바로 뒤에 올 수 있는 말: 조사·'버튼'·구두점·받아적기 주석 대괄호·누르라는 말·다른 키·끝. 이 밖의 말(명사)이 바로 오면
# 서비스 이름 속 번호('<NNN>번 <명사>')일 수 있어 나열 조건(find_keys)을 따로 확인
_AFTER_OK = re.compile(r"\s*(?:(?:을|를|은|는|이|가|과|와|이나|만|으로|로|버튼|키)\s*)*"
                       r"(?:$|[,.!?;·)\[\]…/*#]|(?:누르|눌러|선택|입력)|또는|혹은|그리고|or\b)")
# 서술격 꼬리('A는 N번입니다', 구어 'A는 N번이요')로 끝나는 키: 공지 속 순번·정답 번호와 겹치므로, 발화에 누르라는 지시가 있거나
# 같은 문장 60자 안 앞에 키가 있거나 뒤로 나열이 이어지거나 이름이 이동·상담원 이름(순번이 될 수 없음)일 때만 키
_AFTER_COPULA = re.compile(r"\s*(?:(?:을|를|은|는|이|가|과|와|이나|만|입니다|이에요|예요|이고|이요|요|으로|로|버튼|키)\s*)*"
                           r"(?:$|[,.!?;·)\[\]…/*#]|(?:누르|눌러|선택|입력)|또는|혹은|그리고|or\b)")
# 키 앞이 이름+주제·조건 조사 ('A는 1번', 'A면 1번', 'A로 가시려면, #'): 뒤에 무엇이 와도 키 (받아쓰기에서 쉼표가 빠진 경우)
_PARTICLE_BEFORE = re.compile(r"[가-힣A-Za-z\]]\s*(?:은|는|면|으로|로|:)\s*,?\s*$")
# 나열 사이 구간: 문장 끝(마침표·물음표·느낌표)이 끼면 나열이 끊김, 이름(한글·영문 글자)이 있어야 다음 항목
_SENT_BREAK = re.compile(r"[.!?]")
_HAS_NAME = re.compile(r"[가-힣A-Za-z]")
# 키 앞에 이 말이 붙으면 키가 아니라 번호 값 ('오류번호 <N>번', '코드 <NNN>번', '총 <N>번')
_BEFORE_VALUE = re.compile(r"(?:번호|코드|오류|(?<![가-힣])제|(?<![가-힣])약|(?<![가-힣])총|(?<![가-힣])매)\s*$")
# 끝맺음 키 앞말: '입력 항목+와/과·다음·뒤' 이거나, 누르다·입력하다·선택하다 뒤 8자 안의 순서 말('후(에)·다음(에)·뒤(에)·나서·고')
# 이나 다 마친 조건('…셨으면·…했으면·끝나면·마치면'), 또는 '마지막에·마지막으로·끝에'
# ('<항목>번호와 #', '<N>자리 뒤에 #', '누르신 다음에 #', '입력한 뒤 #', '입력해 주시고 #', '다 입력하셨으면 #', '<N>번과 #').
# 뜻('…하시려면')이 아니라 마친 조건만 허용 ('다른 업무를 선택하시려면 #' 은 선택지 키).
# 마친 조건 갈래(_TERM_DONE)는 '잘못 누르셨으면 *' 와 겹치므로 입력 요청이 있을 때만 끝맺음으로 취급
_TERM_SEQ = (r"(?:(?:번호|자리|금액|일자|날짜|기간|코드|값|암호|\d\s*번)\s*(?:와|과|다음에?|뒤에?)"
             r"|(?:누르|누른|입력|눌러|선택)[가-힣\s]{0,8}?(?:후에?|다음에?|뒤에?|나서|고)"
             r"|마지막(?:에|으로)|끝에)")
_TERM_DONE = r"(?:누르|누른|입력|눌러|선택)[가-힣\s]{0,8}?(?:(?:셨|었|았|였|했)으면|(?:끝나|마치)(?:시)?면)"
_TERM_LEAD = r"(?:" + _TERM_SEQ + "|" + _TERM_DONE + ")"
# 끝맺음 키 자리: 키 바로 앞이 끝맺음 앞말
_BEFORE_TERM = re.compile(_TERM_SEQ + r"\s*,?\s*$")
_BEFORE_TERM_DONE = re.compile(_TERM_DONE + r"\s*,?\s*$")
# 번호 먼저 말투 ('<N>번은 <이름>'): 키 뒤 '은/는' + 이름 (서술어로 끝나지 않는 명사구)
_KEY_FIRST = re.compile(r"\s*(?:은|는|:)\s*([^,.!?*#\d][^,.!?*#\d]{1,30}?)\s*(?=$|[,.!?]|\d{1,3}\s*번|[*#])")
# 조사 없는 번호 먼저 말투 ('1번 A, 2번 B, 0번 C를 눌러 주세요'): 키 뒤 이름이 쉼표·마침표·'을 눌러' 앞까지. 앞 키도 번호 먼저
# 말투였거나 이름 뒤에 바로 다음 키가 오고, 같은 문장에 누르라는 말이 있을 때만 (나열이 아닌 '<N>번 <명사> …' 는 키가 아님)
# 쉼표 없는 받아쓰기에서는 이름 뒤에 공백만 두고 다음 키가 옴 ('1번 A 2번 B 0번 C를 눌러')
_KEY_FIRST_BARE = re.compile(r"\s+([가-힣A-Za-z][^,.!?*#\d]{1,30}?)\s*(?=$|[,.!?]|(?:을|를)\s*(?:눌러|누르)|\s\d{1,3}\s*번|\s[*#])")
_NEXT_IS_KEY = re.compile(r"\s+[가-힣A-Za-z][^,.!?*#\d]{1,30}?(?:\s*[,.]\s*|\s+)(?:\d{1,3}\s*번|[*#])")
# 번호 먼저 말투 이름 끝의 서술격 ('9번은 이전 메뉴입니다')
_COPULA_END = re.compile(r"\s*(?:입니다|이에요|예요)$")
# 이름이 '…후·뒤·도중에' 로 끝남: 다른 단계에서 누를 키 ('상담 연결 후 0번을 누르시면')
_LATER_STAGE = re.compile(r"(?:후|뒤|다음에|도중에|중에)\s*$")
_PREDICATE = re.compile(r"(?:니다|세요|십시[오요]|어요|아요|해요|습니까|ㅂ니다)$|(?:은|는|이|가)\s")
# 문장 끝 (이름은 마지막 문장 안에서만 찾음). 마침표 없는 받아쓰기는 '…니다·…세요' 뒤 공백을 문장 끝으로 봄
_SENT_END = re.compile(r"[.!?。](?=\s|$)|(?:니다|세요|십시[오요]|바랍니다)\s+")
# 쉼표로 이어진 이름 조각을 뒤에서부터 이을 때 멈추는 조각: 서술어·연결 어미·조사로 끝나는 조각 (앞 문장의 일부)
_CLAUSE_END = re.compile(r"(?:니다|세요|어요|아요|에요|해요|이요|하고|시고|며|면|까지|부터|에는|에서|께서|님|후|을|를|고)$"
                         r"|(?:은|는)\s+\S")
# 이름 앞에 남은 앞 키 표현의 꼬리 ('을 눌러 주시고', '그리고', '번')와, 이름 앞의 때 부사구 ('<명사> 도중', '<명사> 중에')
_LEAD_JUNK = re.compile(r"^(?:\s|[,·]|을|를|번|눌러\s*주(?:시고|십시[오요]|세요)|눌러|누르시고|그리고|또는|및|이나|혹은"
                        r"|(?:[가-힣]+\s){0,2}[가-힣]+\s?(?:도중|중에)\s+)+")
# 이름 끝의 조건·주제 꼬리 ('…을 원하시면', '…하시려면', '…으시다면', '…셨으면', '…은', '…으로').
# 이동 말 바로 뒤의 '(으)로' 는 이동의 근거라 남기고('처음으로는' → '처음으로'), '화면' 의 '면' 은 꼬리가 아님
_NAV_HEAD = "처음|초기|최초|메인|시작|이전|직전|상위"     # 두 글자 이동 말 (뒤보기 고정 폭)
_TAIL = re.compile(r"\s*(?:(?:을|를)?\s*원하(?:시|실)(?:면|는\s?경우|\s?경우)|하시려면|으시려면|시려면|려면|으시다면|시다면|하실\s?때|"
                   r"으시면|시면|이시면|이라면|이시라면|으면|(?<!화)면|은|는"
                   r"|(?<!" + _NAV_HEAD + r")으로"
                   r"|(?<!" + "|".join(w + "으" for w in _NAV_HEAD.split("|")) + r")(?<!" + _NAV_HEAD + r")(?<!뒤)(?<!바)(?<!새)(?<!따)로"
                   r"|:)\s*$")
# ('바로·새로·따로' 의 '로' 는 조사가 아니라 부사의 일부라 떼지 않음)
# 서술격 조건 '…이면' ('<명사>이면 1번'): 받침 있는 말 뒤의 '이면' 을 제거
_COPULA = re.compile(r"(?<=[가-힣])이(?:시)?면\s*$")
# 입력 값 안의 글자를 기호로 누르라는 말 ('<번호> 중 하이픈은 별표'): 선택지 키가 아님
_SYMBOL_IN_VALUE = re.compile(r"하이픈|대시|붙임표|(?:^|\s)-\s*$")
# 명령형 문장 ('…눌러 주십시오', '…누르세요', '…누르시면 됩니다'): '숫자 N' 키는 이런 문장에서만 키
_IMPERATIVE = re.compile(_IMP_END + r"\s*[.!]?\s*$")


def _has_batchim(ch: str) -> bool:
    return "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0


def clean_label(s: str) -> str:
    s = _LEAD_JUNK.sub("", s or "").strip()
    m = _COPULA.search(s)
    if m and m.start() > 0 and _has_batchim(s[m.start() - 1]):
        s = s[:m.start()]
    for _ in range(3):
        s2 = _TAIL.sub("", s).strip()
        if s2 == s:
            break
        s = s2
    return s.strip(" ,·")


def _label_before(t: str, lo: int, hi: int) -> str:
    """키 앞 이름: 앞 키 끝(lo)과 키 시작(hi) 사이의 마지막 문장, 쉼표 조각을 뒤에서부터 이음.
    받아쓰기에서 쉼표가 이름 뒤에 잘못 붙은 꼴('… 가시려면, #')은 끝 쉼표를 떼고 처리."""
    chunk = _SENT_END.split(t[lo:hi])[-1].rstrip(" ,")
    pieces = [p.strip() for p in re.split(r"[,，]\s*", chunk)]
    keep = [pieces[-1]] if pieces[-1] else []
    for p in reversed(pieces[:-1] if keep else []):
        if not p or _CLAUSE_END.search(p) or len(keep) >= 6:
            break
        keep.insert(0, p)
    return ", ".join(keep).strip()


def _sentence_at(t: str, pos: int) -> str:
    """pos 가 든 문장."""
    starts = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", t[:pos])]
    s = starts[-1] if starts else 0
    m = re.search(r"[.!?](?=\s|$)", t[pos:])
    e = pos + m.end() if m else len(t)
    return t[s:e]


def _list_continues(t: str, e: int, window: int = 60, depth: int = 0) -> bool:
    """e 뒤 같은 문장 window 자 안에 이름과 다음 키가 오고, 그 다음 키도 진짜 키인지 (뒤가 키 뒤 말·서술격 꼬리이거나 다시
    나열이 이어짐, 또는 이름+주제 조사 뒤의 기호 키). '<명사>은 3번 4번 <명사>' 의 3, '<N>번 <명사> … <M>번 출구' 의 N 은
    나열이 아님."""
    for km in _KEY.finditer(t, e):
        gap = t[e:km.start()]
        if km.start() - e > window or _SENT_BREAK.search(gap) or not _HAS_NAME.search(gap):
            return False
        if km.group("num") and _BEFORE_VALUE.search(gap):
            return False
        if _AFTER_COPULA.match(t, km.end()) or (km.group("sym") and _PARTICLE_BEFORE.search(gap)):
            return True
        return depth < 40 and _list_continues(t, km.end(), window, depth + 1)
    return False


def _key_just_before(t: str, out: list, s: int) -> bool:
    """같은 문장 60자 안 앞에 이미 찾은 키가 있는지."""
    return bool(out) and s - out[-1][1] <= 60 and not re.search(r"[.!?](?=\s|$)", t[out[-1][1]:s])


def find_keys(t: str, request: bool = False, trace: list | None = None):
    """키 후보 [(key, pos, label, kind)] — kind: 'opt' 선택지, 'term' 끝맺음 키 (말한 순서).
    '…셨으면 #' 꼴의 끝맺음 키 자리는 입력 요청이 있을 때만 확인 ('잘못 누르셨으면 *' 는 입력 요청이 없으면 선택지 키).
    trace 를 주면 키 표현마다 받아들인·버린 갈래 이름을 (key, pos, end, 판정, 갈래) 로 덧붙임 (판단은 바뀌지 않음)."""
    out = []
    prev = 0
    key_first_prev = False
    note = (lambda *a: trace.append(a)) if trace is not None else (lambda *a: None)
    for m in _KEY.finditer(t):
        key = m.group("num") or m.group("sym")
        s, e = m.start(), m.end()
        before, after = t[prev:s], t[e:e + 40]
        if m.group("num") and _BEFORE_VALUE.search(before):
            note(key, s, e, "drop", "before_value")
            continue
        if m.group("num") and len(key) >= 3 and not _PRESS_WORD.search(_sentence_at(t, s)):
            note(key, s, e, "drop", "three_digit_no_press")
            continue  # 세 자리 번호는 누르라는 말이 같은 문장에 있을 때만 키 ('<A> 문의는 <NNN>번, <B> 문의는 <MMM>번으로 …')
        if key in "*#" and (_BEFORE_TERM.search(t[max(0, s - 25):s]) or (request and _BEFORE_TERM_DONE.search(t[max(0, s - 25):s]))):
            note(key, s, e, "term", "term_seq" if _BEFORE_TERM.search(t[max(0, s - 25):s]) else "term_done")
            out.append((key, s, None, "term"))
            prev = e
            continue
        raw = _label_before(t, prev, s)
        if _LATER_STAGE.search(raw):
            note(key, s, e, "drop", "later_stage")
            continue  # '<다른 단계> 후 0번을 누르시면' : 지금 누를 키가 아님
        label = clean_label(raw)
        named = bool(re.search(r"[가-힣A-Za-z]", label))
        nxt = e
        key_first = False
        if named:
            # 뒤에 명사가 바로 오면 (쉼표가 없거나 빠진 나열), 다음 셋 가운데 하나일 때만 키:
            #  - 앞이 주제·조건 조사이고 같은 문장 60자 안에 진짜 다음 키가 이어짐 ('A은 <N>번 B는 <M>번')
            #  - 발화에 누르라는 지시가 있고 같은 문장 30자 안에 이름과 진짜 다음 키가 옴 ('A <N>번 B <M>번을 눌러')
            #  - 기호 키('*'·'#')이고 앞이 주제·조건 조사 (기호는 이름 속 번호·순번·정답 번호가 될 수 없어, 서술격 꼬리
            #    조건도 거치지 않음)
            # ('<항목>번호는 <NNN>번 <명사>을 통해 …', '<명사>은 3번 4번 <명사>' 는 키가 아님)
            ok_after = bool(_AFTER_OK.match(after))
            how = "after_ok" if ok_after else None
            if not ok_after and _AFTER_COPULA.match(after):
                ok_after = bool(_PRESS_ORDER.search(t) or _key_just_before(t, out, s) or _list_continues(t, e)
                                or role_of(label) != "choice")
                how = "after_copula" if ok_after else None
            particle = bool(_PARTICLE_BEFORE.search(before))
            if not ok_after:
                if particle and _list_continues(t, e):
                    how = "particle_list"
                elif m.group("sym") and particle:
                    how = "sym_particle"
                elif _PRESS_ORDER.search(t) and _list_continues(t, e, 30):
                    how = "press_list"
            if not (ok_after or (particle and _list_continues(t, e)) or (m.group("sym") and particle)
                    or (_PRESS_ORDER.search(t) and _list_continues(t, e, 30))):
                note(key, s, e, "drop", "named_no_cue")
                continue
            if _SYMBOL_IN_VALUE.search(label):
                note(key, s, e, "drop", "symbol_in_value")
                continue
        else:
            kf = _KEY_FIRST.match(after)
            how = "key_first" if kf else None
            if not kf and (key_first_prev or _NEXT_IS_KEY.match(t, e)) and _PRESS_WORD.search(_sentence_at(t, s)):
                kf = _KEY_FIRST_BARE.match(after)  # '1번 A, 2번 B' : 조사 없는 번호 먼저 나열
                how = "key_first_bare" if kf else None
            kl = _COPULA_END.sub("", kf.group(1).strip()) if kf else ""
            if kf and kl and not _PREDICATE.search(kl):
                label = clean_label(kl)
                nxt = e + kf.end()  # 다음 키의 이름은 이 이름 뒤에서부터 찾음
                key_first = True
            elif re.match(r"\s*(?:을|를)?\s*(?:누르|눌러)", after):
                label = ""
                how = "bare_press"
            else:
                note(key, s, e, "drop", "unnamed_no_cue")
                continue
        note(key, s, e, "opt", how)
        out.append((key, s, label or None, "opt"))
        prev = nxt
        key_first_prev = key_first
    for m in _DIGIT_WORD.finditer(t):
        if _IMPERATIVE.search(_sentence_at(t, m.start())):
            note(m.group(1), m.start(), m.end(), "opt", "digit_word")
            out.append((m.group(1), m.start(), clean_label(_label_before(t, 0, m.start())) or None, "opt"))
        else:
            note(m.group(1), m.start(), m.end(), "drop", "digit_word_not_imperative")
    out.sort(key=lambda x: x[1])
    return out


# ===========================================================================
# 3. 역할 (이름 모양)
# ===========================================================================
# 상담원 연결: 이름이 상담원·상담사·상담직원·상담관·직원(+연결·통화)으로 끝남, '상담 연결', '상담원에게 문의·상담원과 (직접) 통화',
# 이름 전체가 '전화 상담'. 'AI 상담사' 는 사람 상담원이 아님 (자동 응답 안내의 일반 말). 영어 이름(representative·agent·operator)도 같음
_AGENT = re.compile(r"(?<!AI)(?<!AI\s)(?:상담\s?(?:원|사|직원|관)|(?<![가-힣])직원)\s*(?:과|와|을|를|에게|님)?\s*"
                    r"(?:의\s*)?(?:연결|통화|상담|연결\s?요청)?\s*(?:\(.*\))?$"
                    r"|(?<!AI)(?<!AI\s)상담\s?연결(?:\s?요청)?$"
                    r"|(?<!AI)(?<!AI\s)상담\s?(?:원|사)(?:에게|과|와)[가-힣\s]{0,6}(?:문의|통화|상담|연결)$"
                    r"|^전화\s?상담$"
                    r"|(?:representative|operator|agent|counselor)s?$", re.I)
# 이동 말 단위 (이름 전체가 이 단위로만 이루어질 때 이동 키). listen 다시 듣기, home 처음, back 이전, target 메뉴·단계,
# mod 꾸밈 말, move 가다·돌아가다, end 조사·어미 (아무 음절이나 이어 붙지 않도록 낱글자 묶음이 아닌 어미 목록)
_NAV_UNIT = re.compile(
    r"\s*(?:"
    r"(?P<listen>다시\s?듣기|다시\s?들(?:으|을|어)?|다시\s?듣|재청취|반복\s?청취|반복|청취|듣기|듣|들으)"
    r"|(?P<home>맨\s?처음|처음|초기|최초|메인|첫\s?화면|시작|전체\s?메뉴)"
    r"|(?P<back>이전|직전|전\s?단계|앞\s?단계|윗\s?단계|상위|뒤로)"
    r"|(?P<target>메뉴|단계|화면)"
    r"|(?P<mod>안내|내용|멘트|부분|다시|바로|현재|지금|방금|맨|한\s?번\s?더|한\s?번|더|전체)"
    r"|(?P<move>(?:되\s?)?돌아\s?가|돌아\s?갈|넘어\s?가|들어\s?가|이동|가|갈)"
    r"|(?P<back2>전)"
    r"|(?P<end>하시려면|시려면|려면|기|고|를|을|는|은|으로|로|의|에서|부터|까지|싶|실\s?때|겠|주세요|세요|습니다)"
    r")")
# 이름 끝이 '처음 이동' 구 ('<꾸밈 말> 처음으로 이동', '<꾸밈 말> 초기 화면으로 돌아가'): 앞의 꾸밈 말과 관계없이 home.
# 이동 말 앞에 띄어쓰기가 있고, 뒤에 '(으)로'(+관형격 '의')와 가다·돌아가다·이동이 이름 끝에 있어야 함
# ('처음으로 가입' 은 아님). '이전' 은 업무 명사(옮김)와 겹치므로 이 규칙에 쓰지 않음
_HOME_PHRASE = re.compile(r"(?:^|\s)(?:맨\s?처음|처음|초기|메인|첫\s?화면)\s?(?:메뉴|단계|화면)?\s?(?:으로|로)\s?(?:의\s?)?"
                          r"(?:(?:되\s?)?돌아\s?가|넘어\s?가|이동|가)(?:하기|기|고)?$")
# '다시' 와 함께 쓴 듣기 말은 다시 듣기 ('<대상>을 다시 들으시려면', '다시 한번 <대상>을 듣고 싶으시면', '처음부터 다시 듣기')
_REPEAT = re.compile(r"다시\s?(?:한\s?번\s?)?(?:[가-힣]+\s){0,2}?(?:듣|들(?!어\s?가)|청취)|반복\s?(?:해서\s?)?(?:듣|청취)|재청취|repeat|listen\s+again", re.I)
_REPEAT_WORD = re.compile(r"다시|반복|재청취|한\s?번\s?더")
# 다시 입력 (다른 업무 선택지가 없으면 이전 단계로 돌아가는 이동 키, 있으면 대답 선택지)
_REINPUT = re.compile(r"^(?:다시|재)\s?입력")
# 그 밖의 이동 (업무가 아닌 이동): 다른·기타 서비스·업무·메뉴, 외국어(영어) 안내 전환, 통화·이용 종료
_OTHER_NAV = re.compile(r"^(?:기타\s?)?(?:다른|타)\s?(?:서비스|업무|메뉴)|^기타\s?(?:서비스|업무|메뉴)$"
                        r"|^(?:for\s+)?(?:(?:a|an|the)\s+)?(?:외국어|foreign|english|영어|different\s+language)"
                        r"|^(?:(?:이용|통화|전화|상담)\s?)?종료$", re.I)


def nav_role(label: str) -> str | None:
    if _REPEAT.search(label) and not re.search(r"(?:이전|처음|초기|메인|상위)\s?메뉴\s?(?:로|으로)", label):
        return "repeat"
    kinds, pos = [], 0
    while pos < len(label):
        m = _NAV_UNIT.match(label, pos)
        if not m or m.end() == pos:
            return None
        kinds.append(m.lastgroup)
        pos = m.end()
    anchored = any(k in ("listen", "move", "target") for k in kinds) or bool(re.search(r"으로|로|단계|화면|부터", label))
    if not anchored:
        return None
    if "listen" in kinds and (_REPEAT_WORD.search(label) or not ({"home", "back", "back2"} & set(kinds))):
        return "repeat"
    if "home" in kinds:
        return "home"
    if "back" in kinds or "back2" in kinds:
        return "back"
    return None


def role_of(label: str | None) -> str:
    if not label:
        return "choice"
    lab = label.strip()
    if _AGENT.search(lab):
        return "agent"
    nr = nav_role(lab)
    if nr:
        return nr
    if _HOME_PHRASE.search(lab):
        return "home"
    if _REINPUT.search(lab):
        return "back"
    if _OTHER_NAV.search(lab):
        return "other"
    return "choice"


# ===========================================================================
# 4. 입력 요청·끝맺음 키
# ===========================================================================
# 입력 항목 낱말: 개인 정보·번호·날짜·금액·자릿수. '원하시는 번호'·'서비스 번호'·'메뉴·업무·항목·선택 번호'·'<명사>의 번호' 는
# 선택지 번호라 제외
_FIELD = (r"(?:주민\s?(?:등록\s?)?번호|생년\s?월일|계좌\s?번호|카드\s?번호|비밀\s?번호|인증\s?번호|전화\s?번호|휴대\s?(?:폰|전화)\s?번호"
          r"|핸드폰\s?번호|사업자\s?(?:등록\s?)?번호"
          r"|(?<![가-힣])(?!(?:원하시는|원하는|서비스|메뉴|업무|항목|선택하실|선택|해당)\s?번호)(?![가-힣]{1,6}의\s?번호)[가-힣]{1,6}\s?번호"
          r"|[가-힣]{0,4}(?:일자|날짜|기간|금액|코드|암호)|[가-힣]{1,4}일(?=\s?(?:을|를))"
          r"|(?:앞|뒤)?\s?(?:\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s?자리)")
# 입력을 요청하는 동사 ('눌러 주십시오', '입력해 주세요', '누르신 후'). '입력하신·입력되지·잘못 입력' 은 요청이 아님
_REQ_VERB = r"(?:눌러|누른\s?후|누르(?:신|시고|십|세|고)|입력\s?(?:해|하여|하십|하세|하고|하시고|하신\s?후|후|바랍))"
# 항목과 동사 사이: 문장 끝·선택지 키가 끼지 않음. 끝맺음 키 자리의 기호('와 #', '후(에) #', '다음(에) #', '뒤(에) #', '고 #',
# '…셨으면 #', '마지막으로 #')만 낄 수 있음. 뜻 말 뒤의 기호('<항목> 변경을 원하시면 #', '<항목>을 다시 들으시려면 *')는
# 선택지 키라 입력 요청이 아님
_INPUT_REQ = re.compile(_FIELD + r"(?:을|를|와|과|만|의|로)?(?:(?:와|과|후에?|다음에?|고|뒤에?|(?:셨|었|았|였|했)으면|마지막(?:에|으로)|끝에)"
                        r"\s*[#*]|[^.?!*#]){0,30}?" + _REQ_VERB)
# 끝맺음 키: 끝맺음 앞말 뒤의 '#'·'*' 를 누르라는 말 ('번호와 #을 눌러', '누르신 후 #', '입력한 뒤 #', '9번과 #, ')
_TERM = re.compile(_TERM_LEAD + r"\s*,?\s*([#*])\s*(?:(?:을|를)\s*)?(?:,|또는|혹은|눌러|누르|입력)")


def input_info(t: str, trace: list | None = None):
    """(입력 요청 여부, 끝맺음 키). 요청은 명령형 문장 안에서 항목+동사 사이에 선택지 키가 없을 때만.
    trace 를 주면 입력 요청 후보마다 (start, end, 판정) 을 덧붙임 (판단은 바뀌지 않음)."""
    req = False
    note = (lambda *a: trace.append(a)) if trace is not None else (lambda *a: None)
    for m in _INPUT_REQ.finditer(t):
        gap = t[m.start():m.end()]
        if re.search(r"\d{1,3}\s*번", gap[len(re.match(_FIELD, gap).group(0)):]):
            note(m.start(), m.end(), "key_in_gap")
            continue  # '<항목>번호 변경은 2번을 눌러' 는 선택지 고르기
        if not re.search(_IMP_END, _sentence_at(t, m.end() - 1)):
            note(m.start(), m.end(), "not_imperative")
            continue  # '<항목>을 입력하여 … 서비스입니다' 같은 설명
        note(m.start(), m.end(), "request")
        req = True
        break
    tm = _TERM.search(t)
    term = next(g for g in tm.groups() if g) if tm else None
    return req, term


# ===========================================================================
# 5. 발화 종류
# ===========================================================================
# 확인 질문의 대답 키 ('맞으면 1번', '틀리면 2번', '동의하시면 1번', '원하지 않으시면 2번')
_CONFIRM_ANS = re.compile(r"(?:맞(?:으시|으)?면|틀리(?:시)?면|동의하(?:시)?면|동의하지\s?않으(?:시)?면|원하지\s?않으(?:시)?면)\s*,?\s*"
                          r"(?:\d{1,3}\s*번|[*#])")
# 말로 묻는 확인 질문 ('… 맞으십니까?', '<동작> 도와드릴까요?', '…하시겠습니까?'). 받아적기에서 물음표가 마침표로 적히거나
# 빠지는 경우가 있어 물음표·마침표·끝을 모두 허용
_Q_END = r"\s*(?:\?|\.|$)"
_CONFIRM_Q = re.compile(r"맞(?:으십|습|으세요|나요|으신가요)[니까]*" + _Q_END + r"|맞습니까|(?:드릴|할)까요" + _Q_END
                        + r"|하시겠습니까" + _Q_END)
# 대답 낱말 이름 ('예는 1번, 아니오는 2번', '1번은 확인, 2번은 취소'): 이런 이름의 키가 둘 이상이면 확인 질문
_ANSWER_LABEL = re.compile(r"^(?:예|네|아니(?:요|오)|확인|취소|동의|비동의|동의\s?안\s?함)$")
# 의문사 질문 ('무엇을 도와드릴까요?')은 예·아니요 확인이 아님
_WH = re.compile(r"무엇|뭘|어떤|어떻게|언제|어디|몇|무슨|누구|어느")
# 진행 승인 말 (업무 선택지가 하나이고 이동 키가 '이전·처음' 이 아니면 확인). 명사로 쓴 '진행 상황·시작일·계속 들으시려면' 은 제외
_PROCEED = re.compile(r"(?:진행|시작|계속)(?!\s?(?:상황|현황|일|사항|중|여부|들))")
# 말로 대답·선택하라는 요청
_VOICE = re.compile(r"말씀(?:하시거나|해\s?주|하여\s?주|해주|하시면|하셔도)|(?:이|하)신가요\s*\?\s*$|선택(?:해\s?주시거나|하시거나)"
                    r"|선택해\s?주세요\.?$")
# 문자·알림 발송 안내
_MSG = re.compile(r"(?:문자|메시지|SMS|링크|URL)[^.?!]{0,25}(?:보내\s?드리겠|보내\s?드렸|보내드릴|"
                  r"발송\s?(?:되었|됐|했|하였|해\s?드리겠|해\s?드렸|해드리겠|해드렸|됩니다))")
# 발송 뒤에도 통화를 이어 가는 안내 ('전화를 끊지 마시고')
_CONTINUE = re.compile(r"끊지\s?마")
# 업무시간이 아니라 이용 불가
_HOURS = re.compile(r"(?:업무|상담|근무|운영|영업)\s?(?:가능\s?)?시간이?\s?(?:이\s?)?(?:아닙니다|아니(?:므로|어서|라서|오니)|종료되)"
                    r"|가능한\s?시간이\s?아닙니다")
# 상담원 연결 대기 (상담원 말이 있을 때만): '연결해 드리겠습니다·연결 중·통화 중·연결됩니다·연결될 예정'
_WAIT = re.compile(r"상담\s?(?:원|사|관|직원)[^.?!]{0,8}(?:연결\s?(?:해|하여)\s?드리겠|연결하겠|연결\s?중|통화\s?중|연결(?:됩니다|될\s?예정))")
# 결과·처리 완료 (고객 자신의 조회 값·처리 상태). '다음과 같습니다' 같은 일반 공지 말은 쓰지 않음
_RESULT = re.compile(r"조회하신|조회된|(?:현재\s?)?잔액은|(?:완료|접수|처리)\s?되었|발송\s?(?:되었|됩니다|완료)"
                     r"|(?:내역|명세)(?:이|가|은|는)?\s?없습니다")
# 주기 발송 공지 ('매월 <N>일에 … 발송됩니다'): 이미 보낸 결과가 아니라 앞으로 되풀이되는 안내라 결과·발송 판단에서 제외
_PERIODIC_SEND = re.compile(r"(?:매월|매주|매일|매년|말일|\d+\s?일에)[^.?!]{0,25}발송\s?됩니다")
# 약한 결과 신호: 고객 자신을 가리키는 말(고객님·님께서)과 함께일 때만 결과 ('확인 결과 고객님께서 …')
_RESULT_WEAK = re.compile(r"(?:조회|확인)\s?결과")
_CUSTOMER = re.compile(r"고객님|님께서|손님")
# 거절 (본인 확인 실패·미가입·미등록·등록 정보 없음)
_REJECT = re.compile(r"(?:등록|가입|신청)(?:이)?\s?되어\s?있지\s?않|미(?:가입|등록)\s?(?:상태|고객)|일치하지\s?않|확인되지\s?않"
                     r"|실패(?:하였|했)|(?:일치하는|등록된|해당)[가-힣\s]{0,8}(?:정보|고객)(?:가|이)?\s?없")
# 입력 오류 (잘못 누름·시간 초과·입력 없음·못 알아들음)
_ERROR = re.compile(r"잘못\s?(?:누르|눌러|눌렀|입력|선택)|입력\s?시간이?\s?(?:초과|경과|지났)|입력하지\s?않으|입력되지\s?않"
                    r"|확인(?:하신)?\s?후\s?다시\s?(?:이용|걸어)|이해하지\s?못했")
# 사람 상담원이 직접 말함 ('상담원 <이름>입니다'). 이름 자리에 상태 말('연결중·부재중·대기중·통화중')이 오면 제외
_HUMAN = re.compile(r"(?:상담원|상담사|상담직원)\s*(?![가-힣]*(?:중|연결|대기|부재|부족|통화))[가-힣]{2,4}\s*입니다")
# 최종 승인: 금액과 함께 이체·송금·결제·해지·가입·계약을 지금 확정하는지
# ('<금액>원을 이체하시겠습니까', '<금액>원을 결제하시려면 1번', '<금액>원을 송금합니다. 맞으시면 1번', '<금액>원 … . 이체를 진행하시려면 1번').
# 금액은 숫자('50,000원', '30만 원')와 한글 수('오만 원', '삼십만 원': 십·백·천·만 단위 글자가 있을 때만)를 받고,
# 금액과 승인 말 사이에 문장 경계 하나까지 허용
_AMOUNT = r"(?:\d[\d,]*\s*(?:만\s*)?|(?<![가-힣])[일이삼사오육칠팔구]?(?:[십백천만][일이삼사오육칠팔구]?)+\s*)원"
_DEAL = r"(?:이체|송금|결제|해지|가입|계약)"
_SPAN = r"[^.?!]{0,40}(?:[.?!]\s*[^.?!]{0,40})?"
_CRITICAL = re.compile(_AMOUNT + _SPAN + _DEAL + r"(?:를|을)?\s*(?:최종\s?)?(?:승인|확정|진행)?\s*(?:하시겠습니까|하시려면|하시면)"
                       r"|" + _AMOUNT + _SPAN + _DEAL + r"[^.?!]{0,15}[.]?\s*(?:맞으시면|맞으면|진행하시려면|승인하시려면)")
# 메뉴 꼴 최종 승인의 대답 키: 업무 선택지 둘 가운데 하나가 취소·거절 ('…이체하시려면 1번, 취소는 2번')
_CANCEL_LABEL = re.compile(r"취소|아니|중단|거절|원하지\s?않")


def classify(t: str, keys: list, request: bool, trace: list | None = None) -> str:
    """발화 종류. trace 를 주면 종류를 정한 갈래 이름을 덧붙임 (판단은 바뀌지 않음)."""
    note = (lambda a: trace.append(a)) if trace is not None else (lambda a: None)
    biz = [k for k in keys if k["role"] == "choice"]
    navs = {k["role"] for k in keys}
    if request:
        note("request_menu" if len(biz) >= 2 else "request_sensitive")
        return "menu" if len(biz) >= 2 else "sensitive"
    answers = [k for k in biz if _ANSWER_LABEL.match(k["label"] or "")]
    if biz and _CONFIRM_ANS.search(t):
        note("confirm_answer_key")
        return "confirm"
    if biz and len(answers) >= 2:
        note("confirm_answer_labels")
        return "confirm"
    if biz and len(biz) == 1 and _PROCEED.search(biz[0]["label"] or "") and not navs & {"back", "home"}:
        note("confirm_proceed")
        return "confirm"
    if biz:
        note("menu_choice")
        return "menu"
    if not keys and _VOICE.search(t):
        note("voice_nokey")
        return "voice_request"
    if not keys and _CONFIRM_Q.search(t) and not _WH.search(t):
        note("confirm_question")
        return "confirm"
    if _MSG.search(t) and not _CONTINUE.search(t):
        note("msg_result" if keys else "msg_sent")
        return "result" if keys else "message_sent"
    if _HOURS.search(t):
        note("hours")
        return "out_of_hours"
    if _HUMAN.search(t):
        note("human")
        return "agent_human"
    if _WAIT.search(t) and not keys:
        note("wait")
        return "agent_wait"
    if _REJECT.search(t):
        note("reject")
        return "rejected"
    if _ERROR.search(t):
        note("error")
        return "input_error"
    if _RESULT.search(_PERIODIC_SEND.sub(" ", t)):
        note("result")
        return "result"
    if _RESULT_WEAK.search(t) and _CUSTOMER.search(t):
        note("result_weak")
        return "result"
    if _VOICE.search(t):
        note("voice")
        return "voice_request"
    note("info_default")
    return "info"


def _extract(text: str, tr: dict | None = None) -> dict:
    """추출 본체. tr 을 주면 판단 갈래 기록(확실함 판정용)을 채움."""
    t = normalize(text)
    if tr is not None:
        tr.update(t=t, keys=[], req=[], kind=[], cands=[], removed_by_term=[])
    request, term = input_info(t, tr["req"] if tr is not None else None)
    keys, seen = [], set()
    cands = find_keys(t, request, tr["keys"] if tr is not None else None)
    for key, _pos, label, kind in cands:
        if kind != "opt" or key in seen:
            continue
        seen.add(key)
        keys.append({"key": key, "role": role_of(label), "label": label})
    # '다시 입력' 은 다른 업무 선택지와 함께면 대답 선택지
    if any(k["role"] == "choice" for k in keys):
        for k in keys:
            if k["role"] == "back" and _REINPUT.search(k["label"] or ""):
                k["role"] = "choice"
    # 끝맺음 키와 같은 키는 끝맺음으로 봄 (단 입력을 버리고 나가는 '다른 서비스' 이동은 남김)
    if request and term:
        if tr is not None:
            tr["removed_by_term"] = [k for k in keys if k["key"] == term and k["role"] != "other"]
        keys = [k for k in keys if not (k["key"] == term and k["role"] != "other")]
    kind = classify(t, keys, request, tr["kind"] if tr is not None else None)
    biz = [k for k in keys if k["role"] == "choice"]
    approval_menu = kind == "menu" and len(biz) == 2 and any(_CANCEL_LABEL.search(k["label"] or "") for k in biz)
    critical = (kind == "confirm" or approval_menu) and bool(_CRITICAL.search(t))
    if tr is not None:
        tr.update(cands=cands, request=request, term=term)
    return {"kind": kind, "keys": keys, "input": {"request": request, "terminator": term if request else None},
            "critical": critical}


def explain(text: str) -> tuple[dict, dict]:
    """(추출 결과, 판단 갈래 기록) — 개발·점검용."""
    tr: dict = {}
    out = _extract(text, tr)
    return out, tr


def extract(text: str) -> dict:
    tr: dict = {}
    out = _extract(text, tr)
    try:
        reasons = certainty_reasons(text, out, tr)
    except Exception as ex:  # 판정 오류는 '확실하지 않음' 으로
        reasons = [f"판정 오류: {type(ex).__name__}"]
    out["certain"] = not reasons
    out["uncertain_reasons"] = reasons
    return out


# ===========================================================================
# 6. 확실함 판정: 정규식만으로 정해도 되는 발화인지 (아니면 로컬 모델에 물음)
# ===========================================================================
# 판정 원칙: 키 표현·누르라는 말·종류 단서가 모두 강한 갈래로 설명될 때만 확실함. 틀린 '확실함' 은 잘못 누름·입력 요청
# 놓침으로 이어지고, '확실하지 않음' 은 지연만 생기므로 애매하면 확실하지 않음으로 처리.

# 받아적기 편집 표시: 대괄호 주석·자리표시('[불명확]', '[금액]'), 괄호 주석('(?)', '(… 입력 후)'), 말줄임, 동그라미 가림
_MARK = re.compile(r"[\[\]()（）]|…|\.{3}|[○●◯]")
# 강한 갈래 (그 밖의 갈래는 쉼표가 빠진 나열·서술격 꼬리·조사 없는 나열을 되찾는 보조 규칙이라 약함)
_STRONG_KEY = {"after_ok", "key_first", "term_seq"}
_LIST_KEY = {"particle_list", "sym_particle"}   # 쉼표가 빠진 나열·조사 뒤 기호 (확실함 판정에서 인정하는 보조 갈래)
_STRONG_DROP = {"before_value"}            # 키 앞 '번호·코드·오류·제·약·총·매' (번호 값)
# 키 표현 찾기에 걸리지 않은 키 같은 말 (정규화 뒤 글에서 찾음)
_RES_NUMKEY = re.compile(r"(?<!\d)\d{1,3}\s*번")
# 숫자+'번' 뒤가 이 말이면 키가 아닌 번호·횟수 (_KEY 의 뒤보기 제외와 같은 말)
_NOT_KEY_AFTER = re.compile(r"호|째|지|길|\s*출구|\s*(?:이상|이하|이내|까지|만에|연속|정도|씩)|\s*(?:으로|로)\s*(?:전화|연락|걸)")
# 숫자+'번' 앞이 전화번호·금액 꼴 ('<NNNN>-<NNN>번')
_PHONE_BEFORE = re.compile(r"\d-\s*$")
_RES_SYM = re.compile(r"[*#]")
# 한자어 숫자 + '번' 이 정규화에서 숫자로 바뀌지 않고 남은 것: 띄어 쓴 꼴('일 번 <명사>')은 뒤가 무엇이든, 붙여 쓴 꼴은 뒤가
# 조사·구두점·누르라는 말일 때 ('이번 <명사>' 같은 때 말은 제외). '<N> 번 출구' 같은 번호는 제외
_RES_SINO = re.compile(r"(?<![가-힣])[공영빵일이삼사오육륙칠팔구](?:\s번(?![호째지길])(?!\s*출구)|번(?=\s*(?:은|는|을|를|,|\.|$|누르|눌러)))")
# 숫자가 빠진 '번' ('<이름>은 번, …', '…시려면 번 <다음 이름>'): 앞 낱말이 숫자·횟수 말(한·두·몇·여러·이·그·첫·매 …)이 아닌
# 낱말 하나로 홀로 선 '번'
_RES_BARE_BEON = re.compile(r"(?:^|(?<=[\s,.]))(?<!\d\s)(?<!한\s)(?<!두\s)(?<!세\s)(?<!네\s)(?<!몇\s)(?<!여러\s)(?<!이\s)(?<!그\s)(?<!첫\s)(?<!매\s)"
                            r"번(?=$|[\s,.]|을|를|은|는|누르|눌러)")
# 키 낱말이 빠진 자리: 앞말 없이 홀로 선 조사 ('… 후 를 눌러', '<항목>와 를 입력'), 이름+주제·조건 꼬리 바로 뒤의 쉼표·마침표
# ('<이름>는 , <이름>는 #'), 뒤가 키 표현이면 제외
_RES_LONE_PARTICLE = re.compile(r"(?:^|(?<=[\s,]))(?:을|를|은|는|와|과|으로|로)(?=\s*(?:[,.]|눌러|누르|입력|$))")
_RES_EMPTY_ITEM = re.compile(r"[가-힣](?:은|는|려면|으면|시면|이면)(?:\s*[,.](?!\s*(?:\d{1,3}\s*번|[*#]))|\s*$)")
# 글 끝이 문장 끝맺음·키 표현이 아님 (발화가 중간에 잘린 조각)
# 홀로 선 한 글자('… . 다')는 끝맺음으로 치지 않도록, 한 글자 끝맺음 앞에는 한글 음절이 붙어 있어야 함
_COMPLETE_END = re.compile(r"(?:니다|세요|십시[오요]|[어아에예해이군네]요|[가-힣](?:죠|까|다|요)|\d\s*번|[*#])\s*[.?!]?\s*$")
_RES_NATIVE = re.compile(r"(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉)\s?번\s*(?:을|를)?\s*(?:누르|눌러)")
# 키 이름 낱말: 영어 키 말(press·sharp·pound·hash·star …), '#' 의 이름과 받아쓰기 변형(샵·샾·샤프·우물 정·정짜·해시·해쉬·파운드),
# 버튼, 홀로 선 '별'
_RES_WORD = re.compile(r"\b(?:press|push|dial|sharp|pound|hash|star)\b|샵|샾|샤프|우물|정\s?짜|해시|해쉬|파운드"
                       r"|버튼|(?<![가-힣])별(?=\s*(?:을|를|은|는|$|[,.]|누르|눌러))", re.I)
_RES_BARE_DIGIT = re.compile(r"\d\s*(?:을|를|만)?\s*(?:누르|눌러)")
# 단위·조사 없이 홀로 선 한 자리 숫자 (받아쓰기에서 '번' 이 빠진 키일 수 있음)
_LONE_DIGIT = re.compile(r"(?:(?<=^)|(?<=[\s,]))\d(?=$|[\s,.?!])")
# 누르라는 말: 누르다·입력하다·선택하다의 요청·조건 꼴 ('눌러 주세요', '입력하시면', '누르면'). '입력하신·입력되지 않았습니다' 같은
# 설명 꼴은 요청이 아님
_PRESS_VERB = re.compile(r"(?:누르|눌러|입력|선택)[가-힣\s]{0,8}?(?:" + _IMP_END + r"|면)|press", re.I)
# 메뉴 머리말: '원하시는 <서비스> 번호를 눌러 주세요', '<대상>을 선택해 주십시오' (뒤에 선택지 나열이 올 때)
_MENU_INTRO = re.compile(r"(?:원하시는|원하는|해당)\s?(?:[가-힣]+\s?){0,2}?(?:번호|메뉴|서비스|업무|항목)|(?:서비스|메뉴|업무|항목)\s?번호"
                         r"|선택[가-힣\s]{0,6}?" + _IMP_END)
# 이동·상담원 낱말 (선택지로 판정된 이름에 이 말이 있으면 역할이 애매함)
# 상담원 쪽: 사람 낱말(상담원·상담사·직원·담당자), 연결, 주제 없는 '상담' ('상담', '전화 상담', '상담 신청').
# 주제 명사가 앞에 붙은 '<주제> 상담' 은 업무 분류라 낱말로 치지 않음
# 이동 쪽: 처음·초기·메인·상위·돌아가다·이동·다시·반복·듣기·종료·외국어 전환, '…(으)로 가', '이전·전·앞·첫·다른 + 메뉴·단계',
# '메뉴·단계로'. 주제 명사가 붙은 '<주제> 메뉴'·'<주제> 이전'·'단계별' 은 업무 이름이라 낱말로 치지 않음
_NAV_HINT = re.compile(r"상담\s?(?:원|사|직원|관)|직원|담당자|연결|^(?:전화|직접|유선|일반)?\s?상담(?:\s?(?:요청|신청|접수|하기))?$"
                       r"|처음|초기|메인|상위|이전\s?(?:메뉴|단계|화면|으로|로)|^이전$|(?:(?<![가-힣])전|앞|첫|다른)\s?(?:메뉴|단계)"
                       r"|(?<![가-힣])주\s?메뉴|(?:단계|메뉴)\s?(?:으로|로)|^(?:단계|메뉴)$|돌아|이동|다시|반복|듣|들으|청취|종료|외국어|영어"
                       r"|english|(?:으로|로)\s?가", re.I)
# 이름 모양이 깨짐: 다른 키 표현·서술어 끝맺음이 이름 안에 들어옴
_LABEL_BROKEN = re.compile(r"\d\s*번|[*#]|(?:니다|세요|십시[오요])(?:\s|$)|(?<![가-힣])[공영빵일이삼사오육륙칠팔구]\s?번(?![호째지길])"
                           r"|(?:^|\s)번(?:\s|$)")
# 확인 질문 낱말 (메뉴 판정에 섞이면 확인과 메뉴 사이가 애매함)
_CONFIRM_HINT = re.compile(r"맞(?:으|으시)?면|틀리(?:시)?면|동의|맞(?:으십|습|나요|으신가요|으세요)|하시겠습니까")
# 종류 단서 무리 (단서 이름, 묶음, 정규식)
_KIND_CUES = (("voice", "other", _VOICE), ("confirm", "confirm", None), ("msg", "end", None), ("hours", "end", _HOURS),
              ("human", "agent", _HUMAN), ("wait", "agent", _WAIT), ("reject", "end", _REJECT), ("error", "other", _ERROR),
              ("result", "end", None))
# 키 없는 기본 안내(info)의 묶음별 의심 말: 각 종류 단서의 느슨한 꼴. 하나라도 걸리면 기본 안내로 확정하지 않음
_NEAR = (
    ("메시지", re.compile(r"(?:문자|메시지|알림|SMS|링크|URL)|보내\s?드|발송|전송", re.I)),
    ("결과", re.compile(r"결과|잔액|내역|명세|현황|조회하신|조회된|고객님(?:의|께서|은|이)|님께서|손님의?"
                        r"|[가-힣]\s?(?:가\s?)?(?:되었|됐|하였|되셨)"
                        r"|없습니다|\d[\d,.]*\s*(?:원|건|명|석|회|개|점|분|초|시|일|월|년|%)")),
    ("업무시간", re.compile(r"(?:업무|상담|근무|운영|영업|이용|서비스)\s?(?:가능\s?)?시간|시간\s?(?:외|이외|이후|이전)|휴무|마감")),
    ("거절", re.compile(r"(?:가입|등록|신청|일치|확인|조회)[가-힣\s]{0,6}(?:않|없|안\s?되)|불가|실패|만\s?이용|제한|어렵")),
    ("상담원", re.compile(r"상담\s?(?:원|사|직원|관)|직원|담당자|연결|대기|기다려|기다리")),
    ("확인 질문", re.compile(r"\?|[습읍]니까|까요|[가나]요\s*[.?]?\s*$|맞(?:으|습|나|는지)|틀리|동의")),
    ("입력", re.compile(_FIELD + r"|입력|누르|눌러|누른|자리")),
    ("메뉴", re.compile(r"선택|버튼|원하시(?:면|는)|시려면|(?<!\d)\d{1,3}\s?번|번(?:을|를|은|는)")),
    ("오류·음성", re.compile(r"잘못|초과|다시|말씀(?:하|해)|이해하지")),
)
# 키 나열 안의 키 없는 항목: '<이름>은/는·…면·…경우·…때 <짧은 낱말>(을)' 로 끝나는 조각 (받아쓰기에서 키 낱말이 엉뚱한 말로 적힌 꼴)
# 낱말 자리에 숫자가 섞인 꼴('<이름>은 2분', '<이름>면 2편')도 포함 (받아쓰기에서 '번' 이 다른 글자로 적힘)
_ORPHAN_ITEM = re.compile(r"[가-힣A-Za-z]\s*(?:은|는|면|경우|때)\s+([가-힣A-Za-z0-9]{1,4}?)\s*(?:을|를)?\s*$")
# 조사 없는 나열의 키 없는 항목: '<이름> <숫자><'번' 이 아닌 한두 글자>' ('조회 1번, 변경 2분, …' 의 '변경 2분')
_ORPHAN_DIGIT = re.compile(r"[가-힣A-Za-z]\s+(?<![\d.,])\d{1,2}\s*(?!번)[가-힣]{1,2}\s*(?:을|를)?\s*$")
# 누르라는 말의 목적어: 키 표현·입력 항목·숫자 말이어야 함 ('<엉뚱한 말>을 눌러' 는 키 낱말이 잘못 적힌 꼴일 수 있음)
_PRESS_OBJ = re.compile(r"([^\s,.]+)\s*(?:을|를)\s*(?:눌러|누르|입력)")
_OBJ_OK = re.compile(r"(?:\d\s*번|[*#]|번호|자리|숫자|버튼|키|코드|금액|일자|날짜|기간|암호|생년월일|\d)$")
_NOT_ORPHAN_END = re.compile(r"(?:다|요|죠|까|오|네|데|며|고|서|면|만|도|를|을|이|가|에|의|로|와|과)$")
# 선택지가 하나뿐일 때 확인(진행 승인)과 메뉴 사이가 애매한 승인·진행 말: 종류·critical 판단에 이미 쓰는 진행 말(_PROCEED),
# 거래 말(_DEAL), 승인 말, 대답 낱말(확인·동의), 취소·거절 말(_CANCEL_LABEL)을 모은 것
_APPROVE_HINT = re.compile(_PROCEED.pattern + r"|" + _DEAL + r"|승인|확정|확인|동의|" + _CANCEL_LABEL.pattern)
# 누르라는 말 (원문·정규화 글 어디에든): 키가 있는데 이 말이 전혀 없으면 공지 속 순번·번호일 수 있음
_ANY_PRESS = re.compile(r"누르|눌러|누른|선택|입력|press", re.I)
# 누르지 말라는 말 ('누르지 마시고'): 키 표현과 함께면 지금 누를 키인지 애매함
_NO_PRESS = re.compile(r"(?:누르|입력|선택)(?:하)?지\s?마")
# 끝맺음 키가 섞인 거래 승인 말
_DEAL_HINT = re.compile(_AMOUNT + r"|" + _DEAL)
_STRONG_KIND = {"request_sensitive", "confirm_answer_key", "confirm_answer_labels", "menu_choice", "voice_nokey",
                "confirm_question", "msg_sent", "msg_result", "hours", "human", "wait", "reject", "error", "result"}

# --- 보강 조건 ---
# 말 단위 경계: 마침표·물음표·느낌표, 또는 '…니다·…세요·…십시오·…바랍니다' 뒤 공백 (받아쓰기에서 마침표가 빠져도 나뉨)
_CLAUSE_BREAK = re.compile(r"[.!?。](?=\s|$)|(?:니다|세요|십시[오요]|바랍니다)(?=\s)")
# 키 앞 횟수·한도 말 ('하루 최대 3번', '연 4번', '한 달에 2번'): 키가 아니라 횟수일 수 있음
_COUNT_BEFORE = re.compile(r"(?:최대|최소|최고|최다|하루|매일|매주|매달|매월|매년|한\s?달에?|일주일에?|일\s?년에?"
                           r"|(?<![가-힣])(?:연|월|주|일|년|달|총|약|각|모두|합계|누적))\s*(?:에\s*)?$")
# 순서·횟수·등급 값 이름 (선택지 이름이 이 말로 끝나면 '<순서>는 N번' 같은 공지 속 값일 수 있음)
_VALUE_LABEL = re.compile(r"(?:순서|순번|차례|대기\s?번호|등급|회차|횟수|번째)$")
# 고객 자신을 가리키는 말 (선택지 이름에 있으면 '고객님의 <값>은 N번' 같은 공지 속 값일 수 있음)
_CUSTOMER_LABEL = re.compile(r"고객님|손님|귀하|회원님|님의|님께서")
# 다른 단계·나중을 가리키는 말 ('…끝난 후에는', '…뒤에', '다음부터는', '이후', '나중에', '<명사> 시', '…연결되면')
_LATER_CTX = re.compile(
    r"(?<![가-힣])(?:후|뒤)(?:에|에는|엔|부터|부터는|로는|에도)?(?![가-힣])"
    r"|이후|직후|그\s?다음|다음(?:에|에는|엔|부터|부터는)|다음\s?(?:번(?!호)|통화|이용|전화|단계|화면|안내)"
    r"|(?<=[난친된신한은린른던운간온쓴])\s다음(?![가-힣]|\s?(?:중|과|의))"
    r"|부터는|나중에|추후|향후|앞으로는|이따가?|잠시\s?후"
    r"|(?<=[가-힣]\s)시(?:에|에는)?(?=[\s,])"
    r"|(?<!다음\s)(?<!다음)(?<![가-힣])(?:도중|중)(?:에|에는|에도)?(?=[\s,])"
    r"|(?:끝나|마치|마쳤|끝났|완료되|연결되|확인되|인증되|접수되|처리되|나오|들리|이어지)(?:시|셨으|었으|으)?면"
    r"|(?:완료하|끝내|마무리하)(?:시)?면|(?<=\s)되(?:시)?면|(?<![가-힣])전에(?:는)?(?![가-힣])"
    r"|예를\s?들|예컨대|예시|예로")      # 예시로 든 키도 지금 누를 키가 아님
# 이름에 든 '<이름>은/는/…면 <숫자><'번' 이 아닌 글자>' ('변경은 2분 해지': 쉼표가 빠진 채 받아쓰기로 깨진 항목이 이름에 붙음)
_LABEL_ODD_DIGIT = re.compile(r"(?:은|는|면)\s*,?\s*(?<![\d.,])\d{1,2}\s*(?!번)[가-힣]")
# 끝맺음 키 자리: 누르다·입력하다 뒤 순서 말('…후·다음·뒤·고')·마친 조건·'마지막에' 뒤 두 낱말 안에 누르라는 말
# ('입력하신 후 <낱말> 누르세요'). 입력 요청인데 이 자리의 키를 못 찾으면 끝맺음 키 이름이 낯선 말로 적힌 것.
# 그 자리에 입력 항목이 오면 ('… 누르시고 비밀번호를 눌러') 두 번째 입력이라 세지 않음
_TERM_SLOT = re.compile(r"(?:(?:누르|누른|입력|눌러|선택)[가-힣\s]{0,8}?(?:후에?|다음에?|뒤에?|나서|고)|마지막(?:에|으로)|끝에|"
                        + _TERM_DONE + r")\s*,?\s*(?P<slot>(?:[^\s,.]+\s*){0,2}?)(?:눌러|누르)")
# 조사 없이 누르라는 말 바로 앞에 선 낱말 ('<낱말> 누르세요'): 키 표현·부사·연결 어미가 아니면 키 이름이 낯선 말일 수 있음
_BARE_OBJ = re.compile(r"(?:^|(?<=[\s,]))([가-힣A-Za-z]+)\s*(?=(?:눌러|누르))")
_BARE_OBJ_OK = re.compile(r"^(?:다시|먼저|바로|직접|모두|함께|꼭|반드시|천천히|정확히|차례로|계속|지금|이어서|한번|이제|해당|번|잘못)$"
                          r"|(?:고|서|여|후|에|게|히|로|면|며|지|도|만|요|다|께|와|과|을|를|은|는|가|나|시|이|번|자리|번호|숫자)$")
# 선택지 이름이 상태 조건 ('정확하면', '맞으면', '다르면', '<명사>이면'): 높임 '…시면'·'…려면' 이 아닌 '…(으)면'
_STATE_COND = re.compile(r"(?<![시려화])(?:으)?면\s*,?\s*$")
# 조건 꼴 이름 ('…면', '…려면', '…경우', '…때'로 끝남)
_COND_LABEL = re.compile(r"(?<!화)(?:면|경우|때)\s*,?\s*$")
# 대답 말 (선택지 이름에 있으면 메뉴가 아니라 확인 질문일 수 있음).
#  강한 대답 말: 맞음·틀림·정확·올바름·다름·같음·일치·재입력·이대로·동의·아님 (업무 이름으로 드묾) → 선택지 수와 관계없이 걸림
#  약한 대답 말: 취소·수정·정정·계속·진행·중단·같은 (업무 이름으로도 흔함) → 업무 선택지가 둘뿐일 때만 걸림
_ANSWER_STRONG = re.compile(r"맞|틀리|틀린|정확|올바르|옳|다르|같으|같다|일치|재입력|다시\s?(?:입력|선택|녹음)"
                            r"|이대로|그대로|위\s?내용|아니|동의|원하지\s?않|좋으|싫으|괜찮|그렇|그러시|^(?:예|네)$")
_ANSWER_WEAK = re.compile(r"취소|수정|정정|계속|중단|그만|같은|진행(?!\s?(?:상황|현황|사항|중|여부))")
# 되읽기 말 (입력·선택한 값을 다시 읽어 주는 꼴): 메뉴 꼴이어도 확인 질문일 수 있음
_READBACK = re.compile(r"입력하신|입력한|입력된|말씀하신|누르신|선택하신|등록하신|확인하신|고르신|위\s?내용|이\s?내용으로")
# 키 없는 기본 안내로 확정해도 되는 말 끝: 인사·감사·환영만 ('<이름>입니다' 꼴은 '<…> 끝입니다' 같은 결과 명사와 가를 수 없어 받지 않음)
_INFO_OK_END = re.compile(r"(?:(?:감사|고맙)(?:합니다|드립니다|습니다)|안녕하(?:십니까|세요)|반갑습니다|환영합니다)\s*[.!?]?\s*$")
# 기본 안내 말 끝 앞부분에 있으면 안 되는 말: 지난 일·못함·없음·아님·예정·할 수 있음/없음·처리·때·시간·요청 말
_INFO_BAD = re.compile(r"했|였|었|았|됐|되었|되셨|드렸|드릴|못|없|않|아니|아닙|예정|겠|수\s?(?:있|없)|불가|어렵|완료|처리|접수|발송|도착"
                       r"|마감|종료|휴무|휴일|쉽니다|중입니다|때|시간|부담|필요|제한|바랍|주십|주세|하십|하세|드리|합니다"
                       r"|점검|장애|지연|대기|중단|정지|공사|만료|초과|오류|실패|결과|상태|대상|해당|부재|연결")
# 말로 대답하는 방법 설명 ('첫 번째, …', '예를 들어 …라고 하시면 됩니다'): 지금 대답을 요청하는지 설명인지 애매함
_EXPLAIN = re.compile(r"예를\s?들|[가-힣]\s?번째")
# 조건 말 ('원하시면 … 보내 드리겠습니다'): 보낸 안내가 아니라 제안일 수 있음 (발송 말이 든 말 단위의 처음부터 발송 말까지에서 찾음)
_COND_BEFORE = re.compile(r"(?:시|으)?면\s|려면|경우|때")
# 다시 걸라는 말 (상담원 대기가 아니라 통화를 마치는 안내일 수 있음)
_CALL_AGAIN = re.compile(r"다시\s?(?:걸|전화|연락|이용)|나중에|추후")
# 가능 여부·못함·불가 말 ('수 있·없', 못, 불가 …: 조회 결과가 아니라 거절·오류일 수 있음). '내역이 없습니다' 는 결과 단서라 세지 않음
_RESULT_UNABLE = re.compile(r"수\s?(?:없|있)|못|불가|않|어렵|없어|없으")

# --- 추가 보강 조건 ---
# 키 바로 뒤의 누르라는 말 (조사·쉼표 뒤 첫 낱말이 누르다·선택하다·입력하다·고르다 꼴일 때 그 낱말과 나머지)
_POST_VERB = re.compile(r"\s*(?:을|를|은|는|만|이나|나)?\s*,?\s*(?P<v>(?:누르|눌|선택|입력|고르|골라)[가-힣]*)(?P<rest>[^.?!]*)")
# 키 뒤 동사의 꼴과, 같은 말 단위에서 그 뒤에 와도 되는 나머지 (허용 목록. 이 밖의 나머지는 키의 뜻이 키 뒤에 있는 꼴이라 거절)
#  끝맺음 꼴 ('누르세요·눌러주세요·누르십시오·누르시오'): 나머지가 없거나 다음 키가 이어짐
#  '…시기' : 나머지가 '바랍니다'
#  연결 꼴 ('눌러·선택해·입력해·골라'): 나머지가 '주세요·주십시오·주시기 바랍니다·주시면 됩니다', 또는 '주시고' 뒤에 다음 키가 이어짐
#  나열 꼴 ('…시고·…고'): 나머지에 다음 키가 이어짐 ('누르고 기다리시면 <뜻>' 은 거절)
#  고르기 꼴 ('…시거나·…거나'): 나머지에 다음 키가 이어지거나 다른 방법을 요청하는 끝맺음
#  조건 꼴 ('누르시면·누르면'): 나머지가 '됩니다' 뿐 (지난 일 꼴 '누르셨다면' 은 거절)
_POST_VERB_END = re.compile(r"(?:세요|십시[오요]|시오|주세요|주십시[오요])$")
_POST_VERB_LINK = re.compile(r"(?:눌러|선택해|입력해|골라)$")
_POST_VERB_COND = re.compile(r"(?:시|으시)?면$")
_POST_REST_EMPTY = re.compile(r"^[\s,]*$")
_POST_REST_WISH = re.compile(r"^\s*바랍니다\s*$")
_POST_REST_LINK = re.compile(r"^\s*주(?:세요|십시[오요]|시기\s?바랍니다|시면\s?됩니다)\s*$")
_POST_REST_COND = re.compile(r"^\s*,?\s*(?:됩니다|돼요|되십니다|된다)\s*$")
_POST_REST_ALT = re.compile(r"(?:말씀|선택|입력|누르|눌러)[^.?!]*?(?:주세요|주십시[오요]|바랍니다)\s*$")
_NEXT_KEY = re.compile(r"\d{1,3}\s*번|[*#]")
# 예시 표지 (글 어디에든 있으면 예로 든 키일 수 있음): 예·보기·일례·사례 + 를 들어·(으)로, 예컨대, 예시, 이를테면, 가령 (띄어쓰기·
# 받아쓰기 철자 변형 포함)
_EXAMPLE_ANY = re.compile(r"(?<![가-힣])(?:(?:예|보\s?기|일\s?례|사\s?례)\s?(?:를\s?들|로|으로)|예\s?컨\s?[대데]|예시|이\s?를\s?테\s?면|가령)")
# 다른 장치·화면 말 (ARS 버튼이 아닌 화면·앱·웹의 번호·입력일 수 있음). '화면·창' 은 장소 조사가 붙을 때만.
# 입력 요청은 같은 문장에 이 말이 있으면, 키는 같은 말 단위 앞에 이 말 + 장소 조사('…에서·…의·…에')가 있으면 걸림
# (서비스 이름 속 장치 말 '<장치> <명사>은 1번' 은 키 판정에서 세지 않음)
_DEVICE_WORD = r"(?:화면|창|앱|어플|애플리케이션|링크|홈\s?페이지|웹|사이트|인터넷|키오스크|단말기|모바일|브라우저|메신저|이\s?메일)"
_DEVICE = re.compile(r"(?:화면|창)\s?(?:에서|의|에)|앱|어플|애플리케이션|링크|홈\s?페이지|웹|사이트|인터넷|키오스크|단말기|모바일|접속"
                     r"|로그인|브라우저|메신저|이\s?메일", re.I)
_DEVICE_LOC = re.compile(_DEVICE_WORD + r"\s?(?:에서|의|에)(?![가-힣])|접속|로그인", re.I)
# 이름 꼴 끝 (선택지 이름이 끝나는 주제·조건 꼬리: '…은/는', '…(으)면', '…려면', '…(으)로', ':')
_NAME_END = re.compile(r"(?:은|는|면|려면|으로|로|:)\s*,?\s*$")
# 이름 끝이 장소 조사 '…에서' ('이전 메뉴에서 1번을', '앱에서 1번을'): 그 장소에서 누를 키라 지금 이 안내의 키인지 알 수 없음
_LABEL_LOC = re.compile(r"에서(?:는|도)?\s*,?\s*$")
# 이름 끝이 신분 서술 ('상담원이시면', '<명사>이신 분'): 이동·상담원 역할과 맞지 않음 (고객 구분 선택지일 수 있음)
_LABEL_IDENTITY = re.compile(r"(?:이(?:시)?(?:면|라면)|인\s?(?:분|경우)|이신\s?(?:분|경우)|이세요|이십니까)\s*,?\s*$")
# 상담원 역할 이름의 사람 낱말과, 연결·통화를 뜻하는 말
_PERSON = re.compile(r"상담\s?(?:원|사|직원|관)|직원")
_CONNECT = re.compile(r"연결|통화|상담|문의|대화|요청|바꿔|이야기")
# 영어 상담원 이름이 복수('agents')이거나 신분 서술('if you are an agent')이면 고객 구분일 수 있음
_EN_AGENT_ODD = re.compile(r"\b(?:agents|representatives|operators|counselors)\b|\b(?:are|am|is)\s+(?:an?\s+)?(?:agent|representative|operator|counselor)\b", re.I)
# 업무 선택지 이름에 있으면 상담원·이동·종료·언어 전환일 수 있는 말 (_NAV_HINT 낱말 밖): 통화·사람·요원·담당·대화 같은 사람과
# 이야기하는 말, 주제가 붙었든 아니든 '상담', 되풀이·재생·홈·맨 앞·첫·화면 같은 이동 말, 끊다, 언어 이름
# '센터' 는 이름이 고객·상담·콜·안내 센터(연결)로 끝날 때만 ('<주제>센터'·'<센터> 업무 시간' 은 업무 이름)
_NAV_HINT2 = re.compile(r"(?:과|와|하고)\s?(?:직접\s?)?(?:통화|대화|이야기)|사람|요원|안내원|선생님|담당|부서|(?:고객|상담|콜|안내)\s?센터(?:\s?(?:연결|통화|으로|로))?$|직통"
                        r"|되풀이|재생|(?<![가-힣])홈(?![가-힣])|맨\s?앞|(?<![가-힣])첫(?![째])|화면|끊|언어|[가-힣]{1,4}어\s?(?:안내|서비스|상담)", re.I)
# '상담' 이 든 업무 선택지 이름: 다른 키가 상담원 역할이거나 '상담' 이름의 선택지가 둘 이상(주제별 상담 나열)일 때만 업무로 봄.
# 하나뿐이면 상담원 연결인지 업무인지 애매함 ('상담을 받으시려면 0번', '<꾸밈 말> 상담은 0번')
_SANGDAM = re.compile(r"상담")
# 입력 항목 목적어 + 입력 동사 ('<항목>을 입력', '<N>자리를 누르'): 이 자리가 입력 요청 자리 안에 있어야 함
_INPUT_OBJ = re.compile(r"(?:" + _FIELD + r"|[가-힣]*(?:번호|숫자|자리|코드|금액|일자|날짜|암호))\s*(?:을|를|만)?\s*(?P<vb>입력|누르|누른|눌)")
# 입력 항목이 일반 '<낱말> 번호' 갈래일 때 그 낱말이 관형 꼴 ('해당하는 번호', '고르실 번호', '들으실 <명사> 번호'): 선택 번호일 수 있음
_NAMED_FIELD = re.compile(r"주민\s?(?:등록\s?)?번호|생년\s?월일|계좌\s?번호|카드\s?번호|비밀\s?번호|인증\s?번호|전화\s?번호"
                          r"|휴대\s?(?:폰|전화)\s?번호|핸드폰\s?번호|사업자\s?(?:등록\s?)?번호|\d+\s?자리|(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s?자리")
# 거래 말 (확인 질문·두 갈래 메뉴의 최종 승인 판단): 추출기의 거래 말에 탈퇴·구매·주문·출금·입금·납부·충전·환전·해약을 더함
_DEAL_X = re.compile(_DEAL + r"|탈퇴|구매|주문|출금|입금|납부|충전|환전|해약")
# 확정·승인 말 (금액이 없어도 최종 승인일 수 있음)
_COMMIT = re.compile(r"확정|승인|실행|최종|체결|철회|정말")
_COMMIT_LABEL = re.compile(r"확정|승인|실행|최종|체결|철회|완료")
# 행동 제안 말 (확인 질문이 값 확인이 아니라 어떤 일을 할지 묻는 꼴: 최종 승인인지 글로 가를 수 없음)
_PROPOSE = re.compile(r"하시겠습니까|하실까요|할까요|드릴까요|진행|신청|등록|발급|해제")
# 원 밖의 금액 ('100달러', '50유로', '3천 포인트')
_AMOUNT_X = re.compile(r"\d[\d,.]*\s*(?:천\s*|만\s*)?(?:달러|불|엔|유로|위안|파운드|포인트|마일)")
# 되읽기 꼴 문장 ('요청하신 날짜는 <값>입니다', '알려 주신 번호는 <값>입니다', '선택한·등록된 <명사>는 <값>입니다'): 한 일을 가리키는
# 관형 꼴('…신·…주신·…한·…된') + 명사 + 주제 조사 + 서술격 끝맺음
_READBACK_STMT = re.compile(r"[가-힣](?:신|주신|한|된)\s(?:[가-힣]+\s){0,2}?[가-힣]+(?:은|는|이|가)\s[^.?!]{0,40}?(?:입니다|이에요|예요)")
# 값 알림 문장 ('<명사>는 <값>입니다'): 업무 선택지 둘인 메뉴 앞에 오면 그 값을 확인하는 질문일 수 있음
_VALUE_STMT = re.compile(r"[가-힣](?:은|는)\s[^.?!]{1,40}?(?:입니다|이에요|예요)")
# 가리킴 말 + '(으)로' 이름 ('이 번호로 발송', '위 주소로 진행'): 앞에서 말한 값을 받는 확인 질문일 수 있음
_POINT_LABEL = re.compile(r"(?<![가-힣])(?:이|위|그|해당|같은)\s?[가-힣]{1,5}(?:으로|로)(?=\s|$)")
# 키 없는 강한 종류 단서 뒤에 와도 되는 낱말 나머지: 끝맺음('…니다·…세요·…요'), 이유·나열 연결('…므로·…니·…어서·…고·…며·
# …지만'), '…해 주십시오' 의 앞 꼴. 관형('…은·…는·…을'), 조건('…면'), 인용('…다는'), 안긴 물음('…는지'), 명사형('…기') 은 없음
_CUE_TAIL_OK = re.compile(r"^(?:[가-힣]*(?:니다|니까|세요|십시[오요]|어요|아요|에요|예요|해요|이요|죠|므로|오니|니|어서|아서|여서|라서|해서"
                          r"|고|며|지만|는데|은데|거나|라|아|어|여|해))?$")
# 단서가 낱말 끝에서 끝날 때 다음 낱말로 와도 되는 끝맺음
_CUE_NEXT_OK = re.compile(r"^\s*(?:[.!?,]|$|(?:입니다|입니까|이오니|이니|이므로|이라|이어서|됩니다|되었습니다|됐습니다|드리겠습니다|주십시[오요]"
                          r"|주세요|주시기)(?![가-힣]))")
# 결과 단서 가운데 관형·주제 꼴 ('조회하신 <명사>', '조회된 <명사>', '잔액은'): 같은 말 단위 뒤쪽에 값(숫자)이 있어야 결과
_RESULT_ADN = re.compile(r"조회하신|조회된|(?:현재\s?)?잔액은")
# 처리 완료 결과 단서 ('…완료되었·접수되었·처리되었'): 고객의 요청을 가리키는 말이 없으면 일반 공지('<시설> 점검이 완료되었습니다')일 수 있음
_RESULT_DONE = re.compile(r"(?:완료|접수|처리)\s?되었")
_CUSTOMER_REF = re.compile(r"고객님|님(?:의|께서|께)|[가-힣](?:하|주|드|받으)신|신청|요청|주문")
# 기준·한도 말 ('최소 잔액은 …'): 고객의 값이 아니라 기준 안내일 수 있음
_RESULT_POLICY = re.compile(r"최소|최대|최저|최고|기준|한도|이상|이하|필요|유지")
# 사람에게 말하라는 꼴 ('상담원에게 말씀하시면'): 지금 ARS 에 말하라는 요청이 아닐 수 있음
_TO_PERSON = re.compile(r"(?:상담\s?(?:원|사|직원|관)|직원|담당자)\s?(?:에게|께|한테)")
# 다시 걸기·연락 말 (상담원 대기 경계에 더함)
_CALL_BACK = re.compile(r"연락|남겨|회신|예약")
# 기다리라는 말이 아닌 요청 (상담원 대기 안내에 다른 요청이 섞임)
_WAIT_ONLY_REQ = re.compile(r"기다려\s?주|기다리")


def _adnominal(w: str) -> bool:
    """낱말이 관형 꼴로 끝나는지: '…는·…던', 받침 ㄹ ('…실·…할·…을'), '…신·…된' ('선택하신·등록된')."""
    if not w:
        return False
    ch = w[-1]
    return ch in "는던신된" or ("가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 == 8)


def _fired_cues(t: str) -> dict:
    """종류 단서 이름 -> (묶음, 걸린 자리 끝 목록). classify 와 같은 조건."""
    fired = {}
    for name, group, rx in _KIND_CUES:
        if name == "confirm":
            ends = [m.end() for m in _CONFIRM_Q.finditer(t)] if not _WH.search(t) else []
        elif name == "msg":
            ends = [m.end() for m in _MSG.finditer(t)] if not _CONTINUE.search(t) else []
        elif name == "result":
            ends = [m.end() for m in _RESULT.finditer(_PERIODIC_SEND.sub(lambda x: " " * len(x.group(0)), t))]
        else:
            ends = [m.end() for m in rx.finditer(t)]
        if ends:
            fired[name] = (group, ends)
    return fired


def _request_spans(t: str) -> list:
    """입력 요청으로 인정되는 모든 자리 (input_info 와 같은 조건, 첫 자리에서 멈추지 않음)."""
    out = []
    for m in _INPUT_REQ.finditer(t):
        gap = t[m.start():m.end()]
        if re.search(r"\d{1,3}\s*번", gap[len(re.match(_FIELD, gap).group(0)):]):
            continue
        if not re.search(_IMP_END, _sentence_at(t, m.end() - 1)):
            continue
        out.append((m.start(), m.end()))
    return out


def _cut_after_menu(t: str, ev: list) -> bool:
    """글 끝이 잘렸어도, 키를 고르라는 말이 끝맺은 문장 안에서 끝나고 잘린 꼬리에 키·누르라는 말이 없으면 잘림은 키와 무관
    ('…7번을 눌러주십시오. <업무 이름>, 우': 다시 듣기 안내의 첫머리가 잘림). 키가 없는 글은 해당 없음."""
    starts = [s for _k, s, _e, v, _h in ev if v in ("opt", "term")]
    if not starts:
        return False
    ends = [m.end() for m in re.finditer(r"(?:니다|세요|십시[오요]|바랍니다)\s*[.?!]", t)]
    if not ends:
        return False
    cut = ends[-1]
    tail = t[cut:]
    if max(starts) >= cut or not _PRESS_VERB.search(t[:cut]):
        return False
    #  꼬리가 '<이름>은'·'…시려면' 으로 끝나면 다음 선택지의 첫머리가 잘린 것일 수 있음 ('…눌러주십시오. 상담원 연결은')
    return not (_RES_NUMKEY.search(tail) or _RES_SYM.search(tail) or _RES_WORD.search(tail) or _ANY_PRESS.search(tail)
                or _LONE_DIGIT.search(tail) or _RES_SINO.search(tail) or re.search(r"(?:은|는|면|으로|로)\s*[,.]?\s*$", tail))


def _press_by_chain(t: str, key: str, s: int, e: int, key_spans: list, biz_starts: list) -> bool:
    """키 뒤 말 단위에 누르라는 말이 없어도, 그 뒤 누르라는 말까지가 키 항목의 이어진 나열이면 그 말이 이 키에도 적용
    ('… 4번. <이름>은 5번, <이름>은 6번을 눌러주십시오': 받아쓰기가 나열 중간에 마침표를 찍은 꼴).
    조건
    - 키 바로 뒤는 ('버튼' 과) 구두점·공백뿐 ('<값>은 3번입니다. …' 는 키 뒤가 서술어라 해당 없음)
    - 이 키가 나열 안에 있음 (같은 문장 안에 다른 키가 있음), 또는 기호 키이고 이어지는 키 가운데 업무 선택지가 있음
      ('이용하실 버스는 5번. 다시 들으시려면 별표, …'·'이용하실 창구는 5번. 조회는 1번, …' 처럼 문장 첫 번호는 공지 속 번호일 수 있음)
    - 그 뒤 누르라는 말(요청 꼴) 앞까지 쉼표·마침표로 나눈 조각마다 키 표현이 있고 문장 끝맺음이 없음 (쉼표로 이어지는 이름 조각
      '<이름 앞부분>, <이름 뒷부분>은 10번' 의 앞 조각은 서술어 끝맺음·숫자가 없으면 이름의 일부로 봄)
    - 누르라는 말이 든 조각의 키가 누르라는 말 바로 앞에 있음 ('<이름>은 7번을 눌러')"""
    m = re.match(r"\s*(?:버튼)?[\s,.]*", t[e:])
    if not m or not re.search(r"[,.]", m.group(0)):
        return False
    pv = _PRESS_VERB.search(t, e)
    if not pv:
        return False
    starts = [x for x, _y in key_spans if e <= x < pv.start()]
    if not starts:
        return False
    sent_lo, sent_hi = _sentence_span(t, s)
    if not (any(sent_lo <= x < sent_hi and x != s for x, _y in key_spans) or (key in ("*", "#") and any(e <= x < pv.start() for x in biz_starts))):
        return False
    pos = e + m.end()
    for seg in re.finditer(r"[^,.]+", t[pos:pv.start()]):
        a, b = pos + seg.start(), pos + seg.end()
        body = seg.group(0).strip()
        if not body:
            continue
        if _CLAUSE_BREAK.search(seg.group(0)):
            return False
        if any(a <= x < b for x in starts):
            continue
        if t[b:b + 1] == "," and not re.search(r"\d", body) and not _PRED_END.search(body):
            continue
        return False
    last = max(starts)
    return not re.search(r"[,.]", t[last:pv.start()])


# 관형 꼴 동사로 시작하는 이름 ('이용하실 버스'·'배정된 창구'·'받으실 창구'): 나열 맨 앞 특별 항목으로 인정하지 않음 (공지 속 값일 수 있음)
_VALUE_HEAD = re.compile(r"\s*[가-힣]*(?:하실|으실|하신|으신|된|되는|하는|있는|없는|던|드릴|드린)\s")
# 서술어 끝맺음 (쉼표 앞 조각이 이름 조각이 아니라 문장인지)
_PRED_END = re.compile(r"(?:니다|세요|십시[오요]|[가-힣]요|[가-힣]다|[가-힣]까)$")


def _sentence_span(t: str, pos: int) -> tuple:
    starts = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", t[:pos])]
    s = starts[-1] if starts else 0
    m = re.search(r"[.!?](?=\s|$)", t[pos:])
    return s, (pos + m.end() if m else len(t))


def certainty_reasons(text: str, out: dict, tr: dict) -> list:
    """확실하지 않은 까닭 목록 (빈 목록이면 확실함). 사유는 '조건 이름: 짧은 설명' 꼴."""
    rs = []
    add = (lambda r: rs.append(r) if r not in rs else None)
    t = tr["t"]
    raw = (text or "")
    keys = out["keys"]
    biz = [k for k in keys if k["role"] == "choice"]
    kind = out["kind"]
    request = out["input"]["request"]

    # (1) 편집 표시·잘린 글
    if _MARK.search(raw):
        add("편집 표시: 대괄호·괄호·말줄임·가림 표시")

    if t.strip() and not _COMPLETE_END.search(t) and not _cut_after_menu(t, tr["keys"]):
        add("잘린 글: 문장 끝맺음 없이 끝남")

    # (2) 키 표현이 모두 강한 갈래로 설명되는지
    ev = tr["keys"]
    covered = [(x[1], x[2]) for x in ev]
    #  보조 갈래 가운데 인정하는 것: 이름 뒤 주제·조건 조사 + 키 + 다음 항목 ('<이름>은 1번 <이름>은 2번': 받아쓰기가 쉼표를 뺀 나열),
    #  주제·조건 조사 뒤 기호 키 (기호는 공지 속 값이 될 수 없음), 서술격 꼬리 키('<이름>은 0번입니다')는 같은 글에 강한 갈래·위 두
    #  갈래의 선택지 키가 둘 이상 있을 때 (나열의 끝 항목)
    #  쉼표가 빠진 나열 키가 업무 선택지 역할이면 나열 안에 있어야 함: 같은 문장 앞에 다른 키가 있거나 뒤에 업무 선택지가 이어짐
    #  ('이용하실 버스는 5번 다시 들으시려면 별표, …': 홀로 선 번호 뒤에 이동 키만 이어지면 공지 속 번호일 수 있음)
    lab_at = {p2: l2 for _k2, p2, l2, kd in tr["cands"]}
    all_starts = [x[1] for x in ev if x[3] in ("opt", "term")]
    biz_at = [p2 for _k2, p2, l2, kd in tr["cands"] if kd == "opt" and role_of(l2) == "choice"]

    def _in_list(s, e):
        lo = _sentence_span(t, s)[0]
        return any(lo <= x < s for x in all_starts) or any(x > e for x in biz_at)

    listed = sum(1 for _k, _s, _e, v, h in ev if v == "opt" and (h in _STRONG_KEY or h in _LIST_KEY))
    for key, s, e, verdict, how in ev:
        ok_weak = (how in _LIST_KEY and (how == "sym_particle" or role_of(lab_at.get(s)) != "choice" or _in_list(s, e))) \
            or (how == "after_copula" and listed >= 2)
        if verdict in ("opt", "term") and how not in _STRONG_KEY and not ok_weak:
            add(f"보조 갈래 키: {how}")
        if verdict == "drop" and how not in _STRONG_DROP:
            if how == "three_digit_no_press" and not _PRESS_WORD.search(t):
                continue  # 누르라는 말이 전혀 없는 발화의 세 자리 번호는 안내 속 번호
            add(f"버린 키 표현: {how}")
    for x in tr["req"]:
        if x[2] not in ("request", "key_in_gap"):
            add(f"입력 요청 후보 버림: {x[2]}")

    # (3) 키 표현 찾기에 걸리지 않은 키 같은 말
    def _is_covered(a, b):
        return any(a < ce and cs < b for cs, ce in covered)

    for m in _RES_NUMKEY.finditer(t):
        if _is_covered(m.start(), m.end()):
            continue
        if _NOT_KEY_AFTER.match(t, m.end()) or _PHONE_BEFORE.search(t[:m.start()]):
            continue
        add("설명 안 된 키 표현: 숫자+번")
    for m in _RES_SYM.finditer(t):
        if not _is_covered(m.start(), m.end()):
            add("설명 안 된 키 표현: 기호")
    if _RES_SINO.search(t):
        add("설명 안 된 키 표현: 한자어 숫자+번")
    if _RES_LONE_PARTICLE.search(t):
        add("설명 안 된 키 표현: 홀로 선 조사")
    if _RES_EMPTY_ITEM.search(t):
        add("설명 안 된 키 표현: 키 없이 끝난 이름")
    if _RES_BARE_BEON.search(t):
        add("설명 안 된 키 표현: 숫자 빠진 번")
    if _RES_NATIVE.search(t):
        add("설명 안 된 키 표현: 고유어 숫자+번")
    for m in _RES_WORD.finditer(t):
        #  키 표현 바로 뒤의 '버튼' ('<이름>은 # 버튼') 은 그 키 표현의 일부
        if m.group(0) == "버튼" and any(ce <= m.start() <= ce + 1 for _cs, ce in covered):
            continue
        add("설명 안 된 키 표현: 키 이름 낱말")
        break
    if _RES_BARE_DIGIT.search(t):
        add("설명 안 된 키 표현: 누르라는 말 앞 숫자")
    for m in _LONE_DIGIT.finditer(t):
        if not _is_covered(m.start(), m.end()):
            add("설명 안 된 키 표현: 홀로 선 숫자")
            break
    for m in _PRESS_OBJ.finditer(t):
        if _is_covered(m.start(1), m.end(1)) or _OBJ_OK.search(m.group(1)) or re.fullmatch(_FIELD, m.group(1)):
            continue
        add("누를 대상이 키·입력 항목이 아님")
        break
    opt_sp = [(x[1], x[2]) for x in ev if x[3] in ("opt", "term")]
    if opt_sp:
        # 나열 구간: 첫 키 문장의 바로 앞 문장부터 마지막 키까지 (깨진 키가 맨 앞 항목일 수 있음)
        lo, hi = _sentence_span(t, opt_sp[0][0])[0], opt_sp[-1][1]
        lo = _sentence_span(t, max(0, lo - 2))[0] if lo > 0 else 0
        pos = 0
        for seg in re.split(r"([,.!?])", t):
            a, b = pos, pos + len(seg)
            pos = b
            if a < lo or b > hi or any(a <= ks < b for ks, _ke in opt_sp):
                continue
            m = _ORPHAN_ITEM.search(seg)
            if m and not _NOT_ORPHAN_END.search(m.group(1)):
                add("키 나열 안의 키 없는 항목")
                break
            if _ORPHAN_DIGIT.search(seg):
                add("키 나열 안의 숫자+낯선 글자")
                break

    # (4) 누르라는 말이 키·입력 요청·오류 안내로 설명되는지
    #   키·끝맺음 키가 든 문장, 입력 요청 자리(모두), 오류 안내 자리, 선택지 둘 이상 앞의 메뉴 머리말 문장.
    #   입력 요청 발화에서는 '다시 입력해 주세요' 같은 항목 없는 입력 말도 그 요청의 일부로 취급
    gov = [_sentence_span(t, s) for key, s, e, verdict, how in ev if verdict in ("opt", "term")]
    gov += [(x[0], x[1]) for x in tr["req"] if x[2] == "key_in_gap"]
    req_spans = _request_spans(t)
    gov += req_spans
    gov += [(m.start(), m.end()) for m in _ERROR.finditer(t)]
    opt_pos = [s for key, s, e, verdict, how in ev if verdict == "opt"]
    for m in _MENU_INTRO.finditer(t):
        a, b = _sentence_span(t, m.start())
        if sum(1 for p in opt_pos if p >= b) >= 2:
            gov.append((a, b))
    for m in _PRESS_VERB.finditer(t):
        if request and not m.group(0).startswith("선택"):
            continue
        if not any(a <= m.start() < b for a, b in gov):
            add("설명 안 된 누르라는 말")
            break

    # (5) 같은 키 중복·끝맺음 충돌
    #   같은 키를 두 번 말했을 때: 역할이 다르면 충돌. 선택지 역할이면 이름도 같아야 함 (한쪽이 다른 쪽을 품으면 같음)
    opts: dict = {}
    for key, _pos, label, kd in tr["cands"]:
        opts.setdefault(key, []).append((role_of(label) if kd == "opt" else "term", re.sub(r"\W", "", label or "")))
    for key, lst in opts.items():
        if len(lst) < 2:
            continue
        if len({r for r, _l in lst}) > 1:
            add("같은 키 다른 역할")
        elif lst[0][0] == "choice" and any(not (a in b or b in a) for _r, a in lst for _r2, b in lst):
            add("같은 키 다른 이름")
    if not request and any(kd == "term" for _k, _p, _l, kd in tr["cands"]):
        add("입력 요청 없는 끝맺음 키")
    if request:
        term_ev = {k for k, _p, _l, kd in tr["cands"] if kd == "term"}
        if (tr["term"] and term_ev != {tr["term"]}) or (not tr["term"] and term_ev):
            add("끝맺음 키 판정 불일치")
        if tr["removed_by_term"]:
            add("끝맺음 키와 같은 선택지 키")

    if keys and not (_ANY_PRESS.search(t) or _ANY_PRESS.search(raw)):
        add("누르라는 말 없는 키")
    if _NO_PRESS.search(t):
        add("누르지 말라는 말")
    if any(len(k["key"]) >= 3 for k in keys):
        add("세 자리 이상 키")

    # (6) 역할 근거
    for k in keys:
        lab = k["label"] or ""
        if k["role"] == "choice":
            if not re.search(r"[가-힣A-Za-z]", lab):
                add("이름 없는 선택지 키")
            elif len(re.findall(r"[가-힣A-Za-z]", lab)) < 2 and not _ANSWER_LABEL.match(lab):
                add("선택지 이름 너무 짧음")
            if _NAV_HINT.search(lab):
                add("선택지 이름에 이동·상담원 낱말")
            #  기호 키 선택지: 이름이 있는 업무 선택지 셋 이상의 메뉴 안이면 인정 ('<업무>는 별표, <업무>는 1번, …')
            if k["key"] in "*#" and (request or len(biz) < 3):
                add("기호 키가 선택지")
            if _REINPUT.search(lab):
                add("다시 입력 역할 바꿈")
        if _LABEL_BROKEN.search(lab):
            add("이름 모양 깨짐")

    # (7) 종류 근거
    kt = tr["kind"][0] if tr["kind"] else ""
    #  이동 키(다시 듣기·이전·처음·기타)만 있는 글: 상담원 키가 있으면 상담원 연결 메뉴일 수 있어 해당 없음
    nav_only = bool(keys) and all(k["role"] in ("repeat", "back", "home", "other") for k in keys)
    fired = _fired_cues(t)
    groups = {g for g, _e in fired.values()}
    if kt not in _STRONG_KIND:
        if kt == "info_default" and not keys:
            for name, rx in _NEAR:
                if rx.search(t):
                    add(f"기본 안내 의심 말: {name}")
            if len(re.sub(r"\s", "", t)) < 20:
                add("기본 안내: 짧은 글")
        elif not (kt == "info_default" and nav_only):
            add(f"약한 종류 갈래: {kt}")
    if kt == "request_sensitive":
        if biz:
            add("입력 요청과 선택지 함께")
        # 앞 문장의 오류·결과·거절 안내 뒤에 다시 입력을 요청하는 꼴은 입력 요청 (메뉴의 '앞 공지 뒤 선택지' 와 같은 규칙).
        # 단서가 입력 요청 자리 뒤나 같은 자리에 걸리면 (예: '말씀하시거나 입력') 애매함
        first_req = min((a for a, _b in req_spans), default=0)
        for name, (group, ends) in fired.items():
            if group == "confirm" or any(e > _sentence_span(t, first_req)[0] for e in ends):
                add(f"입력 요청과 다른 종류 단서: {name}")
        if _CONFIRM_ANS.search(t) or _CONFIRM_HINT.search(t):
            add("입력 요청과 확인 질문 낱말")
    elif kt == "menu_choice":
        if "confirm" in fired or _CONFIRM_HINT.search(t) or any(_ANSWER_LABEL.match(k["label"] or "") for k in biz):
            add("메뉴와 확인 질문 낱말")
        if len(keys) == 1:
            add("선택지 하나뿐")
        elif len(biz) == 1 and _APPROVE_HINT.search(biz[0]["label"] or ""):
            add("선택지 하나에 승인·진행 말")
    elif kt.startswith("confirm"):
        if len(groups - {"confirm"}) > 0 and kt == "confirm_question":
            add("확인 질문과 다른 종류 단서")
    elif kt in _STRONG_KIND:
        if len(groups) > 1:
            add("종류 단서 여러 묶음")
        if any(k["role"] == "other" for k in keys):
            add("기타 이동 키와 안내")
    #  이동 키만 있는 기본 안내('<안내>입니다. 다시 들으시려면 별표, …')는 확실함으로 인정 (종류는 JEV 가 다시 정함).
    #  상담원 키가 있으면 상담원 연결 메뉴일 수 있어 확실하지 않음
    if keys and not biz and kt == "info_default" and not nav_only:
        add("이동 키만 있는 기본 안내")
    #  이동 키만 있는 안내에 대기 말과 상담원 말: 번호 없이 들린 '기다리시면 상담원 연결' 갈래가 있을 수 있음
    if nav_only and kt == "info_default" and re.search(r"기다려|기다리|대기", t) and re.search(r"상담|연결|직원", t):
        add("이동 키만 있는 안내에 상담원·대기 말")

    # (8) 거래 승인. 종류가 확인·메뉴이면 금액만 있어도 걸림 (_DEAL_HINT 가 금액도 받으므로 거래 말은 없어도 됨)
    if out["critical"] or (kind in ("confirm", "menu") and _DEAL_HINT.search(t) and re.search(_AMOUNT, t)):
        add("금액 거래 승인")

    # (9) 보강 조건: 공지 속 번호, 다른 단계 키, 받아쓰기로 깨진 '번', 낯선 끝맺음 키 이름, 메뉴 꼴 확인 질문, 키 없는 안내
    bounds = sorted({0, len(t)} | {m.end() for m in _CLAUSE_BREAK.finditer(t)})

    def _clause(pos):
        a = max(x for x in bounds if x <= pos)
        b = min((x for x in bounds if x > pos), default=len(t))
        return a, b

    prev_end = 0
    cond_keys = set()
    key_spans = [(s, e) for _k, s, e, v, _h in ev if v in ("opt", "term")]
    biz_starts = [p2 for _k2, p2, l2, kd in tr["cands"] if kd == "opt" and role_of(l2) == "choice"]
    for key, s, e, verdict, how in ev:
        if verdict == "term":
            prev_end = e
            continue
        if verdict != "opt":
            continue
        a, b = _clause(s)
        #  키 뒤 같은 말 단위 안에 누르라는 말이 와야 함 ('<값>은 3번, … 입니다' 의 3 은 공지 속 번호).
        #  예외: 원문에서 영어 'press <키>' 로 말한 키 (누르라는 말이 키 앞에 붙음), 이동·상담원 역할 키 (이름 모양이 공지 속
        #  값이 될 수 없음) 는 글 뒤쪽 어디든 누르라는 말이 있으면 됨
        lab_s = next((l for k2, p2, l, kd in tr["cands"] if p2 == s), None)
        role = role_of(lab_s)
        #  조사 없는 나열('<이름> <N>번, …')에서 이름이 쉼표 조각 여럿으로 이어짐: 앞 조각이 키 낱말이 엉뚱한 말로 적힌 항목일
        #  수 있음 ('조회 1번, 변경 <엉뚱한 말>, 해지 3번')
        if role == "choice" and "," in (lab_s or "") and not _PARTICLE_BEFORE.search(t[:s]):
            add("조사 없는 나열의 여러 조각 이름")
        eng = key.isdigit() and re.search(r"\bpress\s+(?:number\s+)?" + key + r"\b", raw, re.I)
        if not (_PRESS_VERB.search(t, e, b) or eng or (role != "choice" and _PRESS_VERB.search(t, e))
                or _press_by_chain(t, key, s, e, key_spans, biz_starts)):
            add("누르라는 말이 뒤따르지 않는 키")
        #  키 앞 횟수·한도 말
        if key.isdigit() and _COUNT_BEFORE.search(t[a:s]):
            add("횟수·한도 말 뒤 키")
        #  같은 말 단위에서 키 앞에 다른 단계·나중을 가리키는 말
        if _LATER_CTX.search(t[a:s]):
            add("다른 단계 말 뒤 키")
        #  선택지 이름이 상태 조건 ('…이 맞으면', '정확하면', '다르면'): 확인 질문의 대답 키일 수 있음
        raw_lab = _label_before(t, prev_end, s)
        if not kt.startswith("confirm") and _STATE_COND.search(raw_lab):
            add("선택지 이름이 상태 조건")
        if role == "choice" and _COND_LABEL.search(raw_lab):
            cond_keys.add(key)
        prev_end = e
    digits = [int(k["key"]) for k in biz if k["key"].isdigit() and k["key"] != "0"]
    lab_of = {k["key"]: k["label"] or "" for k in biz}
    #  번호 순서: 맨 앞 특별 항목(분실·사기 신고 같은 급한 업무, 3개까지)을 먼저 말하고 본 나열이 오름차순인 꼴은 인정
    #  ('<급한 업무>는 8번, <업무>는 1번, 2번, …'). 그 밖에 순서가 어긋나면 키와 이름이 밀렸을 수 있음
    #  맨 앞 항목 이름이 관형 꼴 말로 시작하면 ('이용하실 버스는 7번, 조회는 1번, …') 공지 속 값일 수 있어 인정하지 않음
    run = 1
    while run < len(digits) and digits[-run - 1] < digits[-run]:
        run += 1
    head = [str(d) for d in digits[:len(digits) - run]]
    if len(head) > (3 if run >= 3 else 0) or any(_VALUE_HEAD.match(lab_of.get(k, "")) for k in head):
        add("선택지 번호 순서 어긋남")
    for k in biz:
        lab = k["label"] or ""
        if _CUSTOMER_LABEL.search(lab):
            add("선택지 이름에 고객 가리킴 말")
        if _VALUE_LABEL.search(lab):
            add("선택지 이름이 순서·횟수 값")
        if _LABEL_ODD_DIGIT.search(lab):
            add("이름 속 숫자+낯선 글자")
    #  키가 있는 발화에서 누르라는 말이 든 쉼표 조각에 키·끝맺음 키·입력 요청·오류 안내·메뉴 머리말이 없음
    #  ('…, 해지는 3분 눌러 주세요': 마지막 항목의 키 낱말이 깨지면 누르라는 말이 앞 키의 문장에 묻혀 설명된 것처럼 보임)
    key_starts = [s for _k, s, _e, v, _h in ev if v in ("opt", "term")]
    if key_starts:
        segb = sorted(set(bounds) | {m.end() for m in re.finditer(r"[,，]", t)})
        ok_spans = list(req_spans) + [(m.start(), m.end()) for m in _ERROR.finditer(t)]
        for m in _MENU_INTRO.finditer(t):
            a, b = _sentence_span(t, m.start())
            if sum(1 for p in key_starts if p >= b) >= 2:
                ok_spans.append((a, b))
        for m in _PRESS_VERB.finditer(t):
            a = max(x for x in segb if x <= m.start())
            b = min((x for x in segb if x > m.start()), default=len(t))
            if any(a <= p < b for p in key_starts) or any(x < b and a < y for x, y in ok_spans):
                continue
            add("누르라는 말이 든 조각에 키 없음")
            break
    #  입력 요청인데 끝맺음 키 자리('입력하신 후 <낱말> 누르세요')의 키를 못 찾음
    if request and not tr["term"] and any(not re.search(_FIELD, m.group("slot")) for m in _TERM_SLOT.finditer(t)):
        add("끝맺음 자리 키 없음")
    for m in _BARE_OBJ.finditer(t):
        if _BARE_OBJ_OK.search(m.group(1)) or _is_covered(m.start(1), m.end(1)):
            continue
        add("누르라는 말 앞 맨 낱말")
        break
    #  메뉴 꼴 확인 질문: 선택지 이름의 대답 말, 되읽기 말, 조건 꼴 이름의 선택지 둘
    #  ('이상 없으시면 1번, 고치시려면 2번': 대답 말 목록 밖의 동사로 말한 확인 질문과 두 갈래 메뉴를 글로 가를 수 없음)
    if kt == "menu_choice" and len(biz) == 2 and cond_keys & {k["key"] for k in biz}:
        add("선택지 둘에 조건 꼴 이름")
    if kt == "menu_choice":
        if any(_ANSWER_STRONG.search(k["label"] or "") for k in biz) or (
                len(biz) == 2 and any(_ANSWER_WEAK.search(k["label"] or "") for k in biz)):
            add("선택지 이름에 대답 말")
        if _READBACK.search(t):
            add("되읽기 말과 메뉴")
    #  키 없는 기본 안내: 모든 말 단위가 인사·감사·환영이나 짧은 이름 소개로 끝나고, 결과·불가·예정·요청 말이 없을 때만 확정
    if kt == "info_default" and not keys:
        for a, b in zip(bounds, bounds[1:]):
            c = t[a:b].strip()
            if not c:
                continue
            m = _INFO_OK_END.search(c)
            if m:
                body = c[:m.start()]
            else:
                add("기본 안내 맺음 아님: 인사·감사·환영이 아님")
                break
            if _INFO_BAD.search(body):
                add("기본 안내 결과·불가·예정·요청 말")
                break
    #  말로 대답 요청인데 방법 설명 꼴, 키 없는 결과인데 발송 말 (결과와 발송 안내 사이)
    if kt in ("voice_nokey", "voice") and _EXPLAIN.search(t):
        add("말로 대답 요청과 설명 말")
    if kt == "result" and not keys and _NEAR[0][1].search(t):
        add("결과와 발송 말")
    #  그 밖의 강한 종류 갈래의 경계: 나중 단계의 입력 요청, 조건이 붙은 발송 제안, 다시 걸라는 대기 안내, 못함이 섞인 결과
    if request and req_spans:
        a0, _b0 = _clause(req_spans[0][0])
        if _LATER_CTX.search(t[a0:req_spans[0][0]]):
            add("다른 단계 말 뒤 입력 요청")
    if kt in ("msg_sent", "msg_result"):
        for m in _MSG.finditer(t):
            if _COND_BEFORE.search(t[_clause(m.start())[0]:m.end()]):
                add("발송 안내에 조건 말")
                break
    if kt == "wait" and _CALL_AGAIN.search(t):
        add("상담원 대기와 다시 걸기 말")
    if kt == "result" and _RESULT_UNABLE.search(t):
        add("결과와 못함 말")

    # (10) 추가 보강 조건: 지금 누를 키인지(키 뒤 말·예시·장소·지난 일), 모든 역할의 이름 근거, 번호 구조, 입력 항목, 금액 없는 승인,
    #      되읽기 꼴, 키 없는 강한 종류 단서의 쓰임
    opt_ev = [(key, s, e) for key, s, e, verdict, how in ev if verdict == "opt"]
    prev_end = 0
    for key, s, e, verdict, how in ev:
        if verdict == "term":
            prev_end = e
            continue
        if verdict != "opt":
            continue
        a, b = _clause(s)
        lab_s = next((l for k2, p2, l, kd in tr["cands"] if p2 == s), None) or ""
        role = role_of(lab_s)
        raw_lab = _label_before(t, prev_end, s)
        prev_end = e
        #  키 바로 뒤 누르라는 말이 지금 누르라는 요청 꼴이 아님 ('1번을 누르시면 <뜻>', '누르셨다면', '누르신 분', '누르시는')
        pm = _POST_VERB.match(t, e, b)
        if pm:
            v, rest = pm.group("v"), pm.group("rest")
            nxt_key = bool(_NEXT_KEY.search(rest))
            if _POST_VERB_END.search(v):
                ok_v = bool(_POST_REST_EMPTY.match(rest)) or nxt_key
            elif v.endswith("시기"):
                ok_v = bool(_POST_REST_WISH.match(rest))
            elif _POST_VERB_LINK.search(v):
                ok_v = bool(_POST_REST_LINK.match(rest)) or (bool(re.match(r"\s*주시고", rest)) and nxt_key)
            elif re.search(r"고$", v):
                ok_v = nxt_key
            elif re.search(r"거나$", v):
                ok_v = nxt_key or bool(_POST_REST_ALT.search(rest))
            elif _POST_VERB_COND.search(v) and not re.search(r"(?:셨|었|았|였|렀|했)", v):
                ok_v = bool(_POST_REST_COND.match(rest))
            else:
                ok_v = False
            if not ok_v:
                add("키 뒤 누르라는 말이 요청 꼴이 아님")
        #  이름 끝이 장소 조사 '…에서'
        if _LABEL_LOC.search(raw_lab):
            add("장소 말 뒤 키")
        #  같은 말 단위에서 키 앞에 다른 장치·화면 말: 장소 조사 꼴이거나, 장치 말이 든 이름이 이름 꼴(주제·조건 꼬리)로 끝나지 않음
        #  ('앱을 켜고 1번을': 이름이 아니라 다른 장치에서 할 일)
        if _DEVICE_LOC.search(t[a:s]) or (re.search(_DEVICE_WORD, t[a:s], re.I) and not _NAME_END.search(raw_lab)):
            add("다른 장치 말 뒤 키")
        #  이동·상담원 역할 근거 (위 (6) 은 선택지 역할만 보므로 따로 봄): 신분 서술 이름, 상담원 이름의 꾸밈 말·연결 말 없음,
        #  영어 상담원 이름의 복수·신분 서술, 이동 이름 안의 이동 말 밖 낱말 ('녹음 메시지 다시 듣기')
        if role != "choice" and _LABEL_IDENTITY.search(raw_lab):
            add("이동·상담원 이름이 신분 서술")
        if role == "agent":
            if _EN_AGENT_ODD.search(lab_s) or _EN_AGENT_ODD.search(raw_lab):
                add("영어 상담원 이름이 복수·신분 서술")
            rest_lab = _PERSON.sub(" ", lab_s)
            if re.search(r"[가-힣]", lab_s) and not re.search(r"[A-Za-z]", lab_s) and not _CONNECT.search(rest_lab) \
                    and (re.search(r"[가-힣]", rest_lab) or key != "0"):
                add("상담원 이름에 연결 말 없음")
        if role in ("repeat", "home", "back") and not _HOME_PHRASE.search(lab_s):
            pos, okp = 0, True
            while pos < len(lab_s):
                mu = _NAV_UNIT.match(lab_s, pos)
                if not mu or mu.end() == pos:
                    okp = False
                    break
                pos = mu.end()
            if not okp:
                add("이동 이름에 다른 낱말")
    if opt_ev and _EXAMPLE_ANY.search(t):
        add("예시 말과 키")
    #  번호 구조: 0번이 업무 선택지, 업무 선택지 8·9번이 바로 앞 번호 없이 떨어져 있음 ('1번, 2번, … 9번': 0·8·9 는 대개 상담원·이동 키)
    if any(k["key"] == "0" for k in biz):
        add("0번이 업무 선택지")
    bd = {int(k["key"]) for k in biz if k["key"].isdigit() and len(k["key"]) == 1 and k["key"] != "0"}
    #  떨어진 8·9번이 나열 맨 앞 특별 항목이면 인정 (이동 키는 대개 나열 끝에 옴)
    #  또는 '<긴 이름>은 9번' 처럼 여섯 글자 이상 이름 뒤 주제 조사로 말함 (이동 키는 대개 '<이동>하시려면 9번' 처럼 조건 꼴로 말함)
    first_biz = biz[0]["key"] if biz else None
    pos_of = {k2: p2 for k2, p2, _l, kd in tr["cands"] if kd == "opt"}

    def _cond_said(key):
        #  조건 꼴로 말했거나 이름이 짧음 ('쳐음 매뉴는 9번': 받아쓰기가 이동 이름을 깨뜨린 꼴일 수 있음)
        p = pos_of.get(key)
        return p is None or bool(re.search(r"면\s*,?\s*$", t[:p])) or len(re.findall(r"[가-힣A-Za-z]", lab_of.get(key, ""))) < 6             or bool(_VALUE_HEAD.match(lab_of.get(key, "")))

    if any(d in (8, 9) and d - 1 not in bd and len(bd) > 1 and str(d) != first_biz and _cond_said(str(d)) for d in bd):
        add("업무 선택지 8·9번이 앞 번호와 떨어짐")
    for k in biz:
        if _NAV_HINT2.search(k["label"] or ""):
            add("선택지 이름에 상담원·이동 말 (2)")
            break
    sd = [k for k in biz if _SANGDAM.search(k["label"] or "")]
    #  '상담' 이름 선택지가 하나뿐: 주제 명사가 앞에 붙은 '<주제> 상담' (상담이 이름 맨 앞이 아님, 사람 낱말 없음, 0번 아님)은 업무로 인정
    if len(sd) == 1 and not any(k["role"] == "agent" for k in keys) and (
            sd[0]["key"] == "0" or re.match(r"\s*(?:전화|직접|유선|일반)?\s?상담", sd[0]["label"] or "")
            or re.search(r"상담\s?(?:원|사|직원|관)", sd[0]["label"] or "")):
        add("선택지 하나에만 상담 말")
    #  입력 항목 + 입력 동사가 입력 요청 자리 밖에 있음 ('<항목>을 입력하시거나, …')
    ok_in = list(req_spans)
    near = [(m.start(), m.end()) for m in _ERROR.finditer(t)] + [_sentence_span(t, m.start()) for m in _MENU_INTRO.finditer(t)]
    for m in _INPUT_OBJ.finditer(t):
        if any(x <= m.start() and m.end() <= y for x, y in ok_in) or any(x < m.end() and m.start() < y for x, y in near):
            continue
        verb = re.match(r"[가-힣]*", t[m.start("vb"):]).group(0)
        if _adnominal(verb) or re.fullmatch(r"(?:입력(?:하|되)|누르|눌리)지", verb):
            continue  # '<항목>을 입력할·입력하신·입력하지' : 설명 말
        add("입력 항목이 입력 요청 밖")
        break
    if request:
        if any(_DEVICE.search(t[_sentence_span(t, x)[0]:_sentence_span(t, max(x, y - 1))[1]]) for x, y in req_spans):
            add("다른 장치 말과 입력 요청")
        for x, y in req_spans:
            f = re.match(_FIELD, t[x:y]).group(0)
            if _NAMED_FIELD.search(f) or not re.search(r"번호\s*$", f):
                continue
            words = re.findall(r"[가-힣]+", re.sub(r"번호\s*$", "", f.strip())) + re.findall(r"([가-힣]+)\s*$", t[max(0, x - 12):x])
            if any(_adnominal(w) for w in words):
                add("입력 항목이 관형 꼴 + 번호")
                break
    #  금액 없는 최종 승인: 확인 질문에 거래·확정·행동 제안 말, 메뉴 선택지 이름에 확정·승인 말, 메뉴에 거래 말과 확정 말, 원 밖의 금액
    if kind == "confirm" and (_DEAL_X.search(t) or _COMMIT.search(t) or _PROPOSE.search(t)):
        add("확인 질문에 거래·확정·행동 제안 말")
    if kind == "menu":
        if any(_COMMIT_LABEL.search(k["label"] or "") for k in biz) or (_DEAL_X.search(t) and _COMMIT.search(t)):
            add("메뉴에 확정·승인 말")
    if kind in ("confirm", "menu") and _AMOUNT_X.search(t):
        add("원 밖의 금액")
    #  되읽기 꼴 문장·가리킴 말 이름과 메뉴
    if kind == "menu":
        if _READBACK_STMT.search(t):
            add("되읽기 꼴 문장과 메뉴")
        if len(biz) == 2 and _VALUE_STMT.search(t):
            add("값 알림 문장과 두 갈래 메뉴")
        if any(_POINT_LABEL.search(k["label"] or "") for k in biz):
            add("선택지 이름에 가리킴 말")
    #  키 없는 강한 종류: 사람 응답은 확정하지 않음, 단서 뒤 낱말 나머지가 끝맺음·이유 연결이 아님, 종류별 경계
    if kt == "human":
        add("사람 상담원 말 종류")
    cue_rx = {"hours": _HOURS, "wait": _WAIT, "reject": _REJECT, "error": _ERROR, "msg_sent": _MSG, "msg_result": _MSG}.get(kt)
    if kt == "result":
        cue_rx = _RESULT
    if cue_rx is not None:
        for m in cue_rx.finditer(t):
            if kt == "result" and _RESULT_ADN.fullmatch(m.group(0)):
                a, b = _clause(m.start())
                if not re.search(r"\d", t[m.end():b]):
                    add("결과 관형 단서에 값 없음")
                continue
            tail_m = re.match(r"[가-힣]*", t[m.end():])
            tail = tail_m.group(0)
            if tail:
                if not _CUE_TAIL_OK.match(tail):
                    add("종류 단서 뒤 꼴이 끝맺음 아님")
            elif not (re.search(r"(?:니다|요)$", m.group(0)) or _CUE_NEXT_OK.match(t, m.end())):
                add("종류 단서 뒤 꼴이 끝맺음 아님")
    if kt in ("voice_nokey", "voice"):
        for m in _VOICE.finditer(t):
            a, b = _clause(m.start())
            tail = re.match(r"[가-힣]*", t[m.end():]).group(0)
            if re.search(r"서$", tail) or re.search(r"감사|고맙", t[a:b]) or _TO_PERSON.search(t[a:m.start()]):
                add("말로 대답 단서가 지금 요청이 아님")
                break
    if kt == "confirm_question":
        cl = [t[x:y] for x, y in zip(bounds, bounds[1:]) if t[x:y].strip()]
        if cl and not _CONFIRM_Q.search(cl[-1]):
            add("확인 질문 뒤에 다른 말")
    if kt == "result":
        if any(_RESULT_DONE.search(m.group(0)) for m in _RESULT.finditer(t)) and not _CUSTOMER_REF.search(t):
            add("완료 결과에 고객 가리킴 없음")
        for m in _RESULT.finditer(t):
            if _RESULT_POLICY.search(t[_clause(m.start())[0]:m.start()]):
                add("결과와 기준·한도 말")
                break
    if kt == "wait":
        if _CALL_BACK.search(t):
            add("상담원 대기와 연락 말")
        for x, y in zip(bounds, bounds[1:]):
            c = t[x:y]
            if re.search(_IMP_END + r"|주시면", c) and not _WAIT_ONLY_REQ.search(c):
                add("상담원 대기와 다른 요청")
                break
    if kt == "reject" and re.search(_FIELD, t):
        add("거절과 입력 항목")
    return rs
