"""Readback display only: preserve digit order, prove confirmation keys locally.

Never use these presentation labels as automatic menu-selection evidence.
"""
from __future__ import annotations

import re

from interpreter import Option

_DIGITS = '0123456789공영일이삼사오육칠팔구'
_SEPARATOR = r'[\s,，.·\-]*'
DIGIT_RUN = re.compile(f'[{_DIGITS}](?:{_SEPARATOR}[{_DIGITS}]){{3,}}')
SPOKEN_LONG_RUN = re.compile(f'[{_DIGITS}](?:{_SEPARATOR}[{_DIGITS}]){{5,}}')
_VALUES = str.maketrans({'공':'0','영':'0','일':'1','이':'2','삼':'3','사':'4',
                        '오':'5','육':'6','칠':'7','팔':'8','구':'9'})
_CONDITION_KEY = re.compile(
    r'(?P<condition>맞으(?:면|시면)|정확하(?:면|시면)|틀리(?:면|시면)|'
    r'(?:다시\s*입력|재입력)(?:하시려면|하려면|하실\s*경우|을\s*원하시면|은|는)?)'
    r'\s*(?P<key>[0-9*#])\s*번')


def _digits(text):
    return re.sub(r'[^0-9]', '', text.translate(_VALUES))


def _phone(digits):
    if not digits.startswith('0'):
        return digits
    if digits.startswith('02') and len(digits) in (9, 10):
        return f'{digits[:2]}-{digits[2:-4]}-{digits[-4:]}'
    if len(digits) in (10, 11):
        return f'{digits[:3]}-{digits[3:-4]}-{digits[-4:]}'
    return digits


def readback_message(text, field_label=None, expected_length=None):
    """Format recognized digits; never make them match the expected value/size."""
    lengths = []
    phone_field = bool(re.search(r'휴대|전화|연락처', field_label or text))
    if not phone_field:
        return text  # This change is scoped to phone readback, not other private fields.
    def replace(match):
        digits = _digits(match.group())
        lengths.append(len(digits))
        return _phone(digits) if phone_field else digits
    message = DIGIT_RUN.sub(replace, text)
    message = re.sub(r'(?<=[0-9])\.\s*(?=번(?:입니다|이|을|를|$))', ' ', message)
    if not lengths:
        message += '\n번호를 정확히 받아쓰지 못했어요. ARS 소리로 확인하거나 다시 입력해 주세요.'
    elif expected_length and lengths[0] != expected_length:
        message += '\n읽어드린 번호의 자릿수가 입력한 값과 달라요. 다시 확인해 주세요.'
    return message


def readback_options(text, prompt):
    """Readable labels only when both conditional/key pairs are unambiguous."""
    if prompt.critical:
        return list(prompt.options)
    positive, negative = [], []
    for match in _CONDITION_KEY.finditer(text):
        condition, key = match.group('condition', 'key')
        if condition.startswith(('맞', '정확')):
            positive.append((key, '맞아요'))
        else:
            negative.append((key, '다시 입력' if re.search('입력', condition) else '틀려요'))
    if len(positive) != 1 or len(negative) != 1 or positive[0][0] == negative[0][0]:
        return list(prompt.options)
    keys = [key for key, _ in positive + negative]
    observed = re.findall(r'(?<![0-9])([0-9*#])\s*번', text)
    if any(observed.count(key) != 1 for key in keys):
        return list(prompt.options)
    # Category is deliberately absent: this is not model source evidence.
    return [Option(key, label) for key, label in positive + negative] + [
        o for o in prompt.options if o.key not in keys]


def phone_confirmation_source(text):
    """Allowlist control clauses, not phone values, for Jev/recording input.

    Values can be Arabic, Hangul, English or malformed. Regex redaction of
    digit forms alone cannot guarantee the same privacy boundary.
    """
    from interpreter import Prompt
    choices = readback_options(text, Prompt('confirm', ''))
    if len(choices) != 2:
        return '전화번호 재확인 안내입니다.'
    return '전화번호 확인 안내입니다. ' + ', '.join(m.group() for m in _CONDITION_KEY.finditer(text))
