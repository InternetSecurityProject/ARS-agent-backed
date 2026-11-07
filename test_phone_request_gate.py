"""Phone-only readback routing; synthetic audio/values, no network or calls."""
import json

import pytest

from recording import CallRecorder
from test_jev_phrase import run
from test_openai_prefetch import settle
from test_phone_readback_audio import setup, ask_and_send, readback_frames, finish, NUMBER


@pytest.mark.parametrize('source',[
    '경과의 모바일 신청 URL을 받아보실 휴대폰 번호와 샵 버튼을 눌러 주십시오.',
    '신청 결과를 받아보실 휴대폰 번호와 샵 버튼을 눌러 주십시오.',
    '신청 결과가 있으니 받으실 휴대폰 번호를 입력해 주세요.',
    '온라인 안내를 받으실 휴대폰 번호를 입력해 주세요.',
])
def test_words_containing_connectives_do_not_disable_phone_readback(source):
    async def go():
        s,rec,ai=setup();worker=s.spawn(s._stt_worker())
        rec.text='입력하신 번호는 01000000001입니다. 맞으면 1번, 다시 입력하시려면 2번을 눌러 주세요.'
        s.recorder=CallRecorder({'test':'synthetic'})
        try:
            await ask_and_send(s,source=source)
            await readback_frames(s)
            await settle(lambda:s.stt_inflight==0)
            assert len(rec.calls)==1  # real routing to the external recognizer boundary
            assert s.pending and s.pending['kind']=='clarification'
            message=s.pending['message']
            assert '입력한 번호: 010-0000-0000' in message
            assert '음성 인식 번호: 010-0000-0001' in message
            assert '달라' in message
            assert s.call.keys==[NUMBER+'#']  # confirmation remains user-owned
            assert s.pending['reason']=='critical_action'
            for data in (s.recorder.events,list(s.hub.history),ai.calls):
                text=json.dumps(data,ensure_ascii=False)
                assert all(value not in text for value in
                           (NUMBER,'01000000001','010-0000-0000','010-0000-0001'))
        finally:await finish(s,worker)
    run(go())


@pytest.mark.parametrize('source',[
    '성명과 결과를 받으실 휴대폰 번호를 입력해 주세요.',
    '인증번호와 결과를 받으실 휴대폰 번호를 입력해 주세요.',
    '계좌번호와 결과를 받으실 휴대폰 번호를 입력해 주세요.',
    '주민등록번호와 결과를 받으실 휴대폰 번호를 입력해 주세요.',
    '고향과 본인의 휴대폰 번호를 입력해 주세요.',
    '고향과본인의 휴대폰 번호를 입력해 주세요.',
    '고향과본인명의의 휴대폰 번호를 입력해 주세요.',
    '고향과현재 사용하시는 휴대폰 번호를 입력해 주세요.',
    '고향과고객님 본인의 휴대폰 번호를 입력해 주세요.',
    '생년과문자를 받으실 휴대폰 번호를 입력해 주세요.',
    '고향과결과를 받으실 휴대폰 번호를 입력해 주세요.',
    '와이파이 안내를 받으실 휴대폰 번호를 입력해 주세요.',
    '비상식별값 및 고객님의 휴대폰 번호를 입력해 주세요.',
    '비상식별값및고객님의 휴대폰 번호를 입력해 주세요.',
    '비상식별값 and 고객님의 휴대폰 번호를 입력해 주세요.',
    '비상식별값 or 본인의 휴대폰 번호를 입력해 주세요.',
    '결과를 받을 휴대폰 번호와 인증번호를 입력해 주세요.',
    '결과를 받을 휴대폰 번호를 포함하여 비상식별값을 입력해 주세요.',
])
def test_mixed_or_unknown_additional_fields_still_never_upload(source):
    async def go():
        s,rec,_=setup();worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,source=source)
            assert not s._phone_comparison.has_value
            await readback_frames(s)
            await settle(lambda:s.stt_inflight==0)
            assert not rec.calls
            assert s.pending is None
        finally:await finish(s,worker)
    run(go())
