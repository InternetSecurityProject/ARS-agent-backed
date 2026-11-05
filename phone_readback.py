"""Explicit, metadata-only consent for one OpenAI phone-readback window."""
from __future__ import annotations

import re
import time

PHONE_FIELDS = {'휴대폰번호', '휴대전화번호', '핸드폰번호', '전화번호', '연락처'}
OTHER_PRIVATE_FIELD = re.compile(
    r'주민(?:\s*등록)?\s*번호|생년\s*월일|계좌\s*번호|카드\s*번호|비밀\s*번호|인증\s*번호|'
    r'성명|성함|이름|주소|이메일|생일|우편\s*번호|보안\s*(?:카드|코드)|일회용|OTP|PIN|CVC|유효\s*기간', re.I)
PHONE_MENTION = re.compile(
    r'(?:[0-9]{1,2}\s*자리\s*)?'
    r'(?:(?:휴대\s*(?:폰|전화)|핸드폰|전화)\s*번호|연락처)'
    r'(?:\s*[0-9]{1,2}\s*자리)?')
_JOIN = r'(?:그리고|하고|또는|혹은|및|와|과|and|or|[,，/+&])'
_JOIN_END = re.compile(_JOIN + r'\s*$', re.I)
_KEY = r'(?:샵|샾|샤프|우물정자|별표|star|hash|pound|[*#]|[0-9]\s*번)\s*(?:버튼|키)?\s*(?:을|를)?\s*'
_POLITE = r'(?:주세요|주십시오|주시기\s*바랍니다)'
_INPUT = r'(?:입력(?:해|하여)\s*' + _POLITE + r'|입력하(?:세요|십시오)|입력\s*부탁드립니다)'
_PRESS = r'(?:눌러\s*' + _POLITE + r'|누르(?:세요|십시오)|누르시기\s*바랍니다)'
_THEN = r'입력(?:하(?:시고|고)|(?:하신|한)?\s*(?:후|뒤))\s*(?:(?:이어서|마지막으로|그리고)\s*)?'
_PHONE_INSTRUCTION = re.compile(
    r'^\s*(?:을|를|은|는)?\s*(?:' + _INPUT + '|' + _PRESS + '|'
    + _JOIN + r'\s*' + _KEY + _PRESS + '|' + _THEN + _KEY + _PRESS + r')[\s.!?。]*$', re.I)
_PHONE_DESCRIPTOR = re.compile(
    r'(?:받아보실|받으실|받을|수신하실|수신할|연락\s*가능한|수신\s*가능한|'
    r'사용하시는|사용\s*중인|명의의|본인의|본인|고객님의|고객님|위해서|위해)\s*$')
_PREFIX_COORDINATION = re.compile(_JOIN, re.I)


def phone_only_request(label, source):
    phones = list(PHONE_MENTION.finditer(source))
    if (re.sub(r'\s+', '', label) not in PHONE_FIELDS or len(phones) != 1
            or OTHER_PRIVATE_FIELD.search(source)):
        return False
    phone = phones[0]
    before, after = source[:phone.start()], source[phone.end():]
    if (re.search(r'번호|자리', before + after) or _JOIN_END.search(before)
            or re.search(r'포함|더해|더하여|추가|함께|입력|제출', before)
            or _PREFIX_COORDINATION.search(before)
            or (before.strip() and not _PHONE_DESCRIPTOR.search(before))):
        return False
    # Positively accept only understood single-phone instructions, optionally
    # followed by a named keypad terminator. Unknown residual noun phrases or
    # constructions (e.g. "including phone plus another value") stay private.
    return bool(_PHONE_INSTRUCTION.fullmatch(after))


class PhoneReadbackPermit:
    """No entered value is stored. Each new boundary invalidates old work."""
    seconds = 30.0

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.generation = 0
        self.until = 0.0
        self.eligible = False

    def revoke(self):
        self.generation += 1
        self.until = 0.0
        self.eligible = False

    def arm(self, label, source):
        self.revoke()
        self.eligible = self.enabled and phone_only_request(label, source)

    def begin(self, keys):
        eligible = self.eligible and bool(re.fullmatch(r'[0-9]{9,15}[*#]?', keys))
        self.revoke()
        if eligible:
            self.until = time.monotonic() + self.seconds
        return eligible

    def active(self):
        return self.enabled and time.monotonic() < self.until

    def valid(self, generation):
        return generation == self.generation and self.active()
