"""Temporary phone-only input comparison. Never persist or use as AI evidence."""
from __future__ import annotations

import re
import time

from phone_readback import phone_only_request
from readback_display import DIGIT_RUN, _digits, _phone, phone_confirmation_source


class PhoneComparison:
    seconds=30.

    def __init__(self,clock=time.monotonic):
        self.clock=clock
        self.generation=0
        self.clear()

    def __repr__(self):
        return '<PhoneComparison>'

    def clear(self):
        self.generation+=1
        self.eligible=False
        self.terminator=''
        self._value=None
        self.until=0.

    def arm(self,label,source,terminator=''):
        self.clear()
        # Numeric delimiters cannot be distinguished from a typed final digit
        # by the existing payload contract. Do not guess an entered value.
        self.eligible=phone_only_request(label,source) and (terminator or '') in ('','#','*')
        self.terminator=terminator or ''

    def begin(self,keys):
        eligible,term=self.eligible,self.terminator
        self.clear()
        value=keys[:-len(term)] if term and keys.endswith(term) else keys
        if eligible and re.fullmatch(r'[0-9]{9,15}',value):
            self._value=value
            self.until=self.clock()+self.seconds
            return True
        return False

    @property
    def has_value(self):
        if self._value is not None and self.clock()>=self.until:self.clear()
        return self._value is not None

    def take(self):
        value=self._value if self.has_value else None
        self.clear()
        return value


def comparison_message(text,entered):
    candidates=[_digits(m.group()) for m in DIGIT_RUN.finditer(text)]
    recognized=candidates[0] if len(candidates)==1 else None
    lines=[f'입력한 번호: {_phone(entered)}',
           '음성 인식 번호: '+(_phone(recognized) if recognized else '정확히 읽지 못했어요')]
    if recognized is None:
        lines.append('ARS 소리를 켜서 확인하거나 다시 입력해 주세요.')
    elif recognized!=entered:
        lines.append('두 번호가 달라요. 음성 인식 오류일 수도 있어요.')
        lines.append('ARS 소리를 켜서 확인하거나 다시 입력해 주세요.')
    else:
        lines.append('입력한 번호와 같게 인식됐어요. 안내를 확인해 주세요.')
    # Only actual confirmation controls, never a corrected/reconstructed value.
    lines.append(phone_confirmation_source(text))
    return '\n'.join(lines)
