"""Source-proven button wording only; never routing evidence or free text.

Private audio/transcripts stay at the caller. Return a small vocabulary of
display labels with the EXISTING keys, without copying values into labels.
"""
from __future__ import annotations

from collections import Counter
import re

from interpreter import Option

_KEY = re.compile(r'(?<![0-9])(?:(?P<number>[0-9])\s*번|(?P<symbol>[*#])(?:\s*(?:번|버튼))?)')
_END = r'(?:을|를)?\s*(?:원하시면|원하실\s*경우|하시려면|하려면|하실\s*경우|받으시려면|받으려면|은|는)'
_ACTIONS = {
    '다시 입력': '다시 입력하기', '재입력': '다시 입력하기',
    '재발급': '재발급하기', '발급': '발급하기', '취소': '취소하기',
    '확인': '확인하기', '신청': '신청하기', '조회': '조회하기', '변경': '변경하기',
}
_ACTION = re.compile(r'(?<![가-힣])(?P<action>'+ '|'.join(map(re.escape,_ACTIONS)) +r')'+_END+r'\s*$')
_DECLINE = re.compile(r'(?:원하지|원치)\s*않으(?:면|시면)\s*$')
_IMPORTANT = re.compile(r'결제|송금|이체|계약|해지|동의')
_KEY_NEGATION = re.compile(r'번\s*(?:이|가|은|는)?\s*(?:아니|아닙)|'
    r'(?:누르|선택하|입력하)지|누르(?:시면|면)\s*안')
_RECEIVE = r'(?:발급(?:을|받기|받으시려면|받으려면)?|받으시려면|받으려면)'
_ADDRESS_END = r'\s*(?:원하시면|원하실\s*경우|하시려면|은|는)?\s*$'
_ADDRESS = [
    (re.compile(r'주민\s*등록(?:\s*상(?:의)?)?\s*주소(?:지)?(?:로|에서)?\s*'+_RECEIVE+_ADDRESS_END), '주민등록 주소로 받기'),
    (re.compile(r'(?:회사|직장)\s*주소(?:지)?(?:로|에서)?\s*'+_RECEIVE+_ADDRESS_END), '회사 주소로 받기'),
    (re.compile(r'사업장\s*주소(?:지)?(?:로|에서)?\s*'+_RECEIVE+_ADDRESS_END), '사업장 주소로 받기'),
    (re.compile(r'(?:등록(?:된)?|기존|현재)\s*주소(?:지)?(?:로|에서)?\s*'+_RECEIVE+_ADDRESS_END), '등록된 주소로 받기'),
    (re.compile(r'주소(?:지)?(?:를)?\s*변경\s*(?:후|한\s*후|하여|하고|해서)?\s*'+_RECEIVE+_ADDRESS_END), '주소 바꿔 받기'),
    (re.compile(r'(?:다른|새로운|새)\s*주소(?:지)?(?:로|에서)?\s*'+_RECEIVE+_ADDRESS_END), '다른 주소로 받기'),
    (re.compile(r'선택(?:한)?\s*주소(?:지)?(?:로|에서)?\s*'+_RECEIVE+_ADDRESS_END), '주소 직접 정해 받기'),
]


def choice_display_options(text: str, options: list[Option]) -> list[Option]:
    """All-or-nothing presentation rewrite from unambiguous current clauses.

    Reject duplicate keys, identical output labels and unsupported/negated
    grammar. Financial/contract approval wording retains the full original.
    In particular, declining a new issuance is not cancelling an application.
    """
    original=list(options)
    if len(original)<2 or len(text)>5000 or _IMPORTANT.search(text) or _KEY_NEGATION.search(text):
        return original
    from gemini_interpreter import normalize_keys
    text=normalize_keys(text)
    matches=list(_KEY.finditer(text))
    counts=Counter(m.group('number') or m.group('symbol') for m in matches)
    keys=[o.key for o in original]
    if len(keys)!=len(set(keys)) or any(counts[k]!=1 for k in keys):
        return original
    found={};declines=[];actions={};start=0
    for match in matches:
        key=match.group('number') or match.group('symbol')
        body=text[start:match.start()].strip();start=match.end()
        if key not in keys:
            continue
        body=re.split(r'[,，.!?。;\n]',body)[-1].strip()
        # The WHOLE keyed destination clause must be supported. A suffix
        # match could silently drop "A and B" or a negated destination prefix.
        address=next((label for pattern,label in _ADDRESS if pattern.fullmatch(body)),None)
        if address:
            found[key]=address
            continue
        if '주소' in body:
            # A destination qualifier (including a negated one) must not be
            # erased by falling back to the generic "발급하기" action below.
            return original
        action=_ACTION.search(body)
        if action:
            actions[key]=action['action'];found[key]=_ACTIONS[action['action']]
        elif _DECLINE.search(body):
            declines.append((key,body))
    if declines:
        # Only a single explicit action plus its directly stated refusal can
        # establish the missing subject of "원하지 않으시면". No yes/no guessing.
        if len(keys)!=2 or len(actions)!=1 or len(declines)!=1:
            return original
        action=next(iter(actions.values()))
        key,body=declines[0]
        refusal=_DECLINE.search(body)
        subject=body[:refusal.start()].strip()
        # An explicitly different subject (e.g. refusing postal receipt or
        # counseling) is not refusal of the preceding issuance action.
        if (subject and not re.fullmatch(re.escape(action)+r'(?:을|를)?',subject)
                or action not in ('발급','재발급','신청','조회')):
            return original
        found[key]=action+'하지 않기'
    if set(found)!=set(keys) or len(set(found.values()))!=len(keys):
        return original
    # No category: display text cannot become semantic route evidence.
    return [Option(o.key,found[o.key]) for o in original]
