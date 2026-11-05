"""Protected audio routing must create secure forms without any external calls."""
import asyncio
import time
import pytest
from interpreter import Prompt
from recording import CallRecorder
from test_phone_readback_audio import setup,ask_and_send,readback_frames,finish
from test_openai_prefetch import settle,feed,speech_frames
from audio import SILENCE_FRAME
from test_jev_phrase import run

ACCOUNT='고객님의 계좌번호를 누르신 후 샵 버튼을 눌러 주십시오.'

class LocalRecognizer:
    is_local=True
    def __init__(self,text=ACCOUNT,gate=None):
        self.text=text;self.gate=gate;self.started=asyncio.Event();self.calls=0
    async def recognize(self,audio):
        self.calls+=1;self.started.set()
        if self.gate:await self.gate.wait()
        return self.text

@pytest.mark.parametrize('text,label,term',[
 (ACCOUNT,'계좌번호','#'),
 ('계좌 번호 열네 자리를 입력해 주세요.','계좌번호 14자리',None),
 ('계좌 비밀번호 네 자리를 입력하신 후 별표를 눌러 주세요.','계좌비밀번호 4자리','*'),
 ('생년월일 앞 여섯 자리를 입력하십시오.','생년월일 앞 6자리',None),
 ('인증번호를 입력하세요.','인증번호',None),
 ('입력하신 번호는 123456입니다. 다음으로 계좌번호를 입력하고 우물정자를 눌러 주세요.','계좌번호','#'),
])
def test_local_directive_yields_only_allowlisted_field_metadata(text,label,term):
    from protected_input import protected_prompt
    p=protected_prompt(text)
    assert p and p.kind=='sensitive' and p.label==label and p.terminator==term
    assert '123456' not in p.text and '고객님의' not in p.text

@pytest.mark.parametrize('text',[
 '입력하신 계좌번호는 1234567890입니다. 맞으면 1번, 틀리면 0번을 눌러 주세요.',
 '계좌번호 변경은 2번, 잔액 조회는 3번을 눌러 주세요.',
 '계좌번호를 입력하지 마세요.',
 '계좌번호를 입력하셨습니다.',
 '잠시 기다려 주십시오.',
 '계좌번호가 필요하면 상담원에게 알려 주세요.',
 '입력하신 계좌번호는 1234567890번입니다. 다시 입력하시려면 2번을 눌러 주세요.',
])
def test_readbacks_menus_and_unsupported_text_never_become_secure_input(text):
    from protected_input import protected_prompt
    p=protected_prompt(text)
    assert p is None or p.kind!='sensitive'


def test_next_account_audio_creates_keypad_without_openai_or_jev():
    async def go():
        s,external,ai=setup();s.d.private_recognizer=LocalRecognizer()
        s.recorder=CallRecorder({})
        worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='주민등록번호')
            await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert s.pending and s.pending['kind']=='secure_input'
            assert s.pending['label']=='계좌번호' and s.pending['terminator']=='#'
            assert s.status=='awaiting_user' and not external.calls and not ai.calls
            assert any(e['type']=='event:request' and e['data'].get('label')=='계좌번호' for e in s.recorder.events)
        finally:await finish(s,worker)
    run(go())


def test_readback_never_reopens_keypad_and_does_not_leak_to_models_or_recording():
    async def go():
        s,external,ai=setup()
        s.d.private_recognizer=LocalRecognizer('입력하신 계좌번호는 1234567890입니다. 맞으면 1번, 틀리면 0번을 눌러 주세요.')
        s.recorder=CallRecorder({});worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='계좌번호');await readback_frames(s)
            await settle(lambda:s.stt_inflight==0)
            assert s.pending and s.pending['kind']=='clarification'
            assert {o['key']:o['label'] for o in s.pending['options']}=={'1':'맞아요','0':'틀려요'}
            assert '1234567890' in s.pending['message']  # local ephemeral UI only
            assert not external.calls and not ai.calls
            assert '1234567890' not in repr(s.recorder.events) and not s.captions
        finally:await finish(s,worker)
    run(go())


def test_unrecognized_protected_audio_stays_local_after_twenty_seconds():
    async def go():
        s,external,ai=setup();s.d.private_recognizer=LocalRecognizer('개인정보가 포함된 알 수 없는 안내')
        worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='계좌번호')
            s.secure_sent_at=time.monotonic()-100
            await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert not external.calls and not ai.calls and s.pending is None
            assert '개인정보' in s.status_message
        finally:await finish(s,worker)
    run(go())


def test_old_local_result_cannot_replace_new_request():
    async def go():
        s,external,ai=setup();gate=asyncio.Event();local=LocalRecognizer(gate=gate)
        s.d.private_recognizer=local;worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='주민등록번호');await readback_frames(s)
            await settle(local.started.is_set)
            await s._ask_secure(Prompt('sensitive','인증번호를 입력해 주세요.',label='인증번호'))
            rid=s.pending['request_id'];gate.set();await settle(lambda:s.stt_inflight==0)
            assert s.pending['request_id']==rid and s.pending['label']=='인증번호'
            assert not external.calls and not ai.calls
        finally:gate.set();await finish(s,worker)
    run(go())


def test_local_failure_does_not_fall_back_to_external():
    class Broken(LocalRecognizer):
        async def recognize(self,audio):raise RuntimeError('1234567890 private')
    async def go():
        s,external,ai=setup();s.d.private_recognizer=Broken();worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='주민등록번호');await readback_frames(s)
            await settle(lambda:s.stt_inflight==0)
            assert not external.calls and not ai.calls and s.pending is None
            assert '개인정보' in s.status_message
        finally:await finish(s,worker)
    run(go())

@pytest.mark.parametrize('text,term',[
 ('계좌번호를 입력해 주세요, 취소는 별표를 눌러 주세요.',None),
 ('계좌번호를 입력하고 샵 버튼을 눌러 주세요. 취소는 별표를 눌러 주세요.','#'),
 ('계좌번호와 별표를 눌러 주세요.','*'),
])
def test_terminator_is_bound_to_input_completion_not_cancel(text,term):
    from protected_input import protected_prompt
    assert protected_prompt(text).terminator==term


def test_raw_private_confirmation_is_not_kept_in_debug_history_or_sent_by_voice_answer():
    from session import ApiError
    async def go():
        s,external,ai=setup();s.d.private_recognizer=LocalRecognizer('입력하신 계좌번호는 1234567890입니다. 맞으면 1번, 틀리면 0번을 눌러 주세요.')
        worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='계좌번호');await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            rid=s.pending['request_id']
            with pytest.raises(ApiError):await s.action({'type':'answer','request_id':rid,'text':'맞아요'})
            await s.action({'type':'select','request_id':rid,'key':'1'})
            assert '1234567890' not in repr(s.hub.history) and not external.calls and not ai.calls
        finally:await finish(s,worker)
    run(go())


def test_result_after_local_confirmation_displays_locally_without_cloud_summary():
    async def go():
        s,external,ai=setup();local=LocalRecognizer('입력하신 계좌번호는 1234567890입니다. 맞으면 1번, 틀리면 0번을 눌러 주세요.')
        s.d.private_recognizer=local;s.recorder=CallRecorder({});worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='계좌번호');await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            await s.action({'type':'select','request_id':s.pending['request_id'],'key':'1'})
            local.text='고객님의 잔액은 12345원입니다.'
            await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert s.result_heard and s.status=='agent_done'
            assert s.summary['summary']=='조회 결과를 화면에서 확인해 주세요.'
            assert s.summary['key_info'][0]['value']==local.text
            assert '12345' not in repr(s.recorder.events) and '12345' not in repr(s.hub.history)
            assert not external.calls and not ai.calls and not s.captions
        finally:await finish(s,worker)
    run(go())


def test_menu_after_local_input_can_be_read_and_selected_without_cloud_egress():
    async def go():
        s,external,ai=setup();s.d.private_recognizer=LocalRecognizer('잔액 조회는 3번, 거래내역 조회는 4번을 눌러 주세요.')
        worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='계좌번호');await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert s.pending and s.pending['kind']=='clarification'
            assert {o['key'] for o in s.pending['options']}=={'3','4'} and '잔액 조회' in s.pending['message']
            await s.action({'type':'select','request_id':s.pending['request_id'],'key':'3'})
            assert s.call.keys[-1]=='3' and not external.calls and not ai.calls
        finally:await finish(s,worker)
    run(go())


def test_local_timeout_menu_invalidates_old_input_form():
    async def go():
        s,external,ai=setup();s.d.private_recognizer=LocalRecognizer('잔액 조회는 3번, 거래내역 조회는 4번을 눌러 주세요.')
        worker=s.spawn(s._stt_worker())
        try:
            await s._ask_secure(Prompt('sensitive',ACCOUNT,label='계좌번호',terminator='#'))
            old=s.pending['request_id'];await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert s.pending and s.pending['request_id']!=old and s.pending['kind']=='clarification'
            assert not external.calls and not ai.calls and not s.call.keys
        finally:await finish(s,worker)
    run(go())


def test_queued_input_waits_for_local_classification_and_aborts_on_menu():
    async def go():
        s,external,ai=setup();gate=asyncio.Event()
        s.d.private_recognizer=LocalRecognizer('잔액 조회는 3번, 거래내역 조회는 4번을 눌러 주세요.',gate)
        s.call.speaking=True;worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='계좌번호')
            await readback_frames(s);await settle(s.d.private_recognizer.started.is_set)
            s.call.speaking=False;await asyncio.sleep(.07)
            assert not s.call.keys  # Do not send before the protected paragraph is classified.
            gate.set();await settle(lambda:s.stt_inflight==0)
            await settle(lambda:not s.secure_queued)
            assert not s.call.keys and s.pending['kind']=='clarification'
            assert not external.calls and not ai.calls
        finally:gate.set();await finish(s,worker)
    run(go())


def test_final_private_result_is_preserved_when_remote_hangup_occurs_during_local_stt():
    async def go():
        s,external,ai=setup();gate=asyncio.Event()
        local=LocalRecognizer('고객님의 잔액은 12345원입니다.',gate);s.d.private_recognizer=local
        worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='계좌번호');await readback_frames(s)
            await settle(local.started.is_set);s.hangup_seen=True;gate.set()
            await settle(lambda:s.stt_inflight==0)
            assert s.result_heard and s.summary['key_info'][0]['value']==local.text
            assert not external.calls and not ai.calls
        finally:gate.set();await finish(s,worker)
    run(go())


def test_old_form_submission_waits_for_inflight_protected_menu_recognition():
    async def go():
        s,external,ai=setup();gate=asyncio.Event()
        local=LocalRecognizer('잔액 조회는 3번, 거래내역 조회는 4번을 눌러 주세요.',gate)
        s.d.private_recognizer=local;worker=s.spawn(s._stt_worker())
        try:
            await s._ask_secure(Prompt('sensitive',ACCOUNT,label='계좌번호',terminator='#'))
            rid=s.pending['request_id'];await readback_frames(s);await settle(local.started.is_set)
            assert not s._ars_speaking()
            await s.action({'type':'secure_input','request_id':rid,'digits':'1234567890'})
            assert not s.call.keys and s.secure_queued
            gate.set();await settle(lambda:s.stt_inflight==0);await settle(lambda:not s.secure_queued)
            assert not s.call.keys and s.pending['kind']=='clarification'
        finally:gate.set();await finish(s,worker)
    run(go())


def test_queued_value_is_not_sent_when_wait_limit_expires_during_local_recognition():
    async def go():
        s,external,ai=setup();s.s.secure_wait_max_s=.02;gate=asyncio.Event()
        local=LocalRecognizer(ACCOUNT,gate);s.d.private_recognizer=local
        s.call.speaking=True;worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label='계좌번호');await readback_frames(s);await settle(local.started.is_set)
            s.call.speaking=False;await asyncio.sleep(.08)
            assert not s.call.keys and not s.secure_queued
            assert any(m['event']=='input_rejected' for m in s.hub.history)
        finally:gate.set();await finish(s,worker)
    run(go())
