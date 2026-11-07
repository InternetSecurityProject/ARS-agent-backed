"""Audio frames -> privacy gate -> mocked OpenAI -> existing readback request.

No real API/audio upload/phone calls; only synthetic tones and fictional digits.
"""
import asyncio
import time
from types import SimpleNamespace
import pytest

from audio import SILENCE_FRAME
from interpreter import Prompt
from navigator import NavState
from session import ApiError, Session
from stt_prefetch import PrefetchedSegment
from test_flows import make_settings
from test_openai_prefetch import feed, speech_frames, settle
from test_system_recovery import RecoveryAI, LocalCall
from test_jev_phrase import item,engine,run

NUMBER='01000000000'
READBACK='입력하신 번호는 공 일 공 공 공. 공, 공, 공, 공, 공, 공. 번입니다. 맞으면 1번, 다시 입력하시려면 2번을 눌러 주십시오.'

class FakeOpenAI:
    is_local=False
    def __init__(self,gate=None):self.calls=[];self.gate=gate;self.started=asyncio.Event();self.text=READBACK
    async def recognize(self,data):
        self.calls.append(data);self.started.set()
        if self.gate:await self.gate.wait()
        return self.text
    def new_segment(self):
        return PrefetchedSegment(self.recognize,min_chunk_s=1.5)

class FakePhone(LocalCall):
    provides_transcripts=False
    dtmf_mode='sdk'
    speaking=False


def setup(*,enabled=True,provider='openai',gate=None):
    from app import build_deps
    st=make_settings(stt_prefetch=True,allow_openai_phone_readback=enabled,
                     clarify_expire_s=60,clarify_drop_s=120,no_response_s=60)
    deps=build_deps(st);st.stt=provider
    rec=FakeOpenAI(gate);deps.recognizer=rec
    ai=RecoveryAI({'1':item('맞으면',confidence=.3),'2':item('다시 입력하시려면',confidence=.3)},kind='confirm')
    deps.interp=engine(ai)
    s=Session('5550000',deps);s.call=FakePhone();s.answered_at=time.monotonic()
    s.goal='문자 안내';s.nav=NavState(s.goal);s.agent_active=True;s.status='listening'
    return s,rec,ai

async def ask_and_send(s,label='휴대폰번호',source=None):
    await s._ask_secure(Prompt('sensitive',source or f'{label}를 입력하고 샵 버튼을 눌러 주세요.',
                               label=label,terminator='#'))
    rid=s.pending['request_id']
    await s.action({'type':'secure_input','request_id':rid,'digits':NUMBER})

async def finish(s,worker=None):
    await s.finish('user_canceled',hangup=False)
    if worker:await worker

async def readback_frames(s):
    await feed(s,speech_frames());await feed(s,[SILENCE_FRAME]*60)


def test_approved_phone_audio_reaches_stt_and_creates_visible_confirmation():
    async def go():
        s,rec,ai=setup();worker=s.spawn(s._stt_worker())
        try:
            await s._ask_secure(Prompt('sensitive','휴대폰 번호를 입력하고 샵을 눌러 주세요.',label='휴대폰번호',terminator='#'))
            # Pending-input privacy tail must not poison NEW audio after the real send boundary.
            await feed(s,[SILENCE_FRAME]*12)
            await s.action({'type':'secure_input','request_id':s.pending['request_id'],'digits':NUMBER})
            await feed(s,speech_frames())
            assert not rec.calls  # private readback is never uploaded speculatively as prefix chunks
            await feed(s,[SILENCE_FRAME]*60)
            await settle(lambda:s.pending is not None)
            req=s.pending
            assert len(rec.calls)==1
            assert req['kind']=='clarification' and '010-0000-0000' in req['message']
            assert {o['key']:o['label'] for o in req['options']}=={'1':'맞아요','2':'다시 입력'}
            assert all('공 일 공' not in repr(state) and NUMBER not in repr(state) for state,q in ai.calls)
            assert '공 일 공' not in req['ars_prompt'] and not s.captions
            await s.action({'type':'select','request_id':req['request_id'],'key':'1'})
            assert s.call.keys==[NUMBER+'#','1']
        finally:await finish(s,worker)
    run(go())


@pytest.mark.parametrize('enabled,provider,label,source',[
 (False,'openai','휴대폰번호',None),
 (True,'gemini','휴대폰번호',None),
 (True,'rtzr','휴대폰번호',None),
 (True,'openai','주민등록번호',None),
 (True,'openai','계좌번호',None),
 (True,'openai','비밀번호',None),
 (True,'openai','카드번호',None),
 (True,'openai','인증번호',None),
 (True,'openai','입력 번호',None),
 (True,'openai','휴대폰번호','휴대폰 번호와 인증번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','휴대폰 번호와 주민번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','성명과 휴대폰 번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','휴대폰 번호와 고객번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','휴대폰 번호와 운전면허번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','휴대폰 번호와 고객식별 8자리를 입력해 주세요.'),
 (True,'openai','휴대폰번호','휴대폰 번호와 이메일을 입력해 주세요.'),
 (True,'openai','휴대폰번호','휴대폰 번호와 생일을 입력해 주세요.'),
 (True,'openai','휴대폰번호','고향과 휴대폰 번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','휴대폰 번호와 비상식별값을 입력해 주세요.'),
 (True,'openai','휴대폰번호','휴대폰 번호를 포함하여 비상식별값을 입력해 주세요.'),
 (True,'openai','휴대폰번호','비상식별값에 이어 휴대폰 번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','고향과 본인의 휴대폰 번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','비상식별값 및 고객님의 휴대폰 번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','비상식별값 and 고객님의 휴대폰 번호를 입력해 주세요.'),
 (True,'openai','휴대폰번호','비상식별값 OR 본인의 휴대폰 번호를 입력해 주세요.'),
])
def test_nonconsented_vendor_or_nonphone_audio_stays_private(enabled,provider,label,source):
    async def go():
        s,rec,_=setup(enabled=enabled,provider=provider);worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,label,source)
            await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert not rec.calls and s.pending is None
        finally:await finish(s,worker)
    run(go())


def test_queued_input_does_not_open_phone_exception_before_sending():
    async def go():
        s,rec,_=setup();worker=s.spawn(s._stt_worker());s.call.speaking=True
        try:
            await ask_and_send(s)
            assert s.secure_queued and not s.call.keys
            await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert not rec.calls
            s.call.speaking=False
            await settle(lambda:bool(s.call.keys))
            await readback_frames(s);await settle(lambda:s.pending is not None)
            assert s.pending['kind']=='clarification' and len(rec.calls)==1
        finally:await finish(s,worker)
    run(go())


def test_send_failure_never_leaves_a_phone_exception_open():
    async def go():
        s,rec,_=setup();worker=s.spawn(s._stt_worker())
        async def broken(_):raise RuntimeError('synthetic failure')
        s.call.send_keys=broken
        try:
            with pytest.raises(ApiError):await ask_and_send(s)
            await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert not rec.calls
        finally:await finish(s,worker)
    run(go())


def test_old_queued_phone_audio_is_not_reauthorized_by_new_secure_request():
    async def go():
        s,rec,_=setup()
        try:
            await ask_and_send(s);await readback_frames(s)
            assert s.stt_inflight
            await s._ask_secure(Prompt('sensitive','주민등록번호를 입력해 주세요.',label='주민등록번호'))
            rid=s.pending['request_id'];worker=s.spawn(s._stt_worker())
            await settle(lambda:s.stt_inflight==0)
            assert not rec.calls and s.pending['request_id']==rid
        finally:await finish(s,locals().get('worker'))
    run(go())


def test_revoked_inflight_phone_result_never_overwrites_new_request():
    async def go():
        gate=asyncio.Event();s,rec,ai=setup(gate=gate);worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s);await readback_frames(s)
            await settle(lambda:rec.started.is_set())
            await s._ask_secure(Prompt('sensitive','계좌번호를 입력해 주세요.',label='계좌번호'))
            rid=s.pending['request_id'];gate.set()
            await settle(lambda:s.stt_inflight==0)
            assert s.pending['request_id']==rid and not ai.calls
        finally:gate.set();await finish(s,worker)
    run(go())


def test_phone_confirmation_closes_private_window_for_the_public_result():
    async def go():
        s,rec,_=setup();worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.pending is not None)
            await s.action({'type':'select','request_id':s.pending['request_id'],'key':'1'})
            rec.text='입력하신 휴대폰 번호로 잠시 후 신청 문자를 발송해 드리겠습니다. 이용해 주셔서 감사합니다.'
            await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert s.result_heard and s.status=='agent_done' and s.pending is None
        finally:await finish(s,worker)
    run(go())


def test_expired_phone_permission_drops_whole_readback_without_upload():
    async def go():
        s,rec,_=setup();worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s);await feed(s,speech_frames())
            s._phone_readback.until=time.monotonic()-1
            await feed(s,[SILENCE_FRAME]*60);await settle(lambda:s.stt_inflight==0)
            assert not rec.calls and s.pending is None
        finally:await finish(s,worker)
    run(go())


def test_private_recorder_transcript_stays_masked():
    from recording import CallRecorder
    async def go():
        s,rec,_=setup();s.recorder=CallRecorder({'test':'synthetic'});worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.pending is not None)
            texts=[e['text'] for e in s.recorder.events if e['type']=='stt']
            assert texts and all('공 일 공' not in t and NUMBER not in t for t in texts)
        finally:await finish(s,worker)
    run(go())


def test_wrong_length_arabic_phone_never_reaches_jev_or_captions():
    async def go():
        s,rec,ai=setup();worker=s.spawn(s._stt_worker())
        rec.text='입력하신 번호는 010000000000입니다. 잠시 기다려 주세요.'
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert rec.calls and ai.calls
            assert all('010000000000' not in repr(state) for state,q in ai.calls)
            assert '010000000000' not in repr(s.captions)
        finally:await finish(s,worker)
    run(go())


def test_result_waiting_for_lock_is_revalidated_after_new_request():
    async def go():
        s,rec,ai=setup();worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s)
            async with s._lock:
                await readback_frames(s);await settle(lambda:rec.started.is_set())
                await s._ask_secure(Prompt('sensitive','계좌번호를 입력해 주세요.',label='계좌번호'))
                rid=s.pending['request_id']
            await settle(lambda:s.stt_inflight==0)
            assert s.pending['request_id']==rid and not ai.calls
        finally:await finish(s,worker)
    run(go())


def test_english_spelled_phone_value_is_not_forwarded_to_jev():
    async def go():
        s,rec,ai=setup();worker=s.spawn(s._stt_worker())
        rec.text='입력하신 번호는 zero one zero five five five five zero zero zero zero 입니다. 맞으면 1번, 다시 입력하시려면 2번.'
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert s.pending and all('zero one zero' not in repr(state) for state,q in ai.calls)
        finally:await finish(s,worker)
    run(go())


def test_missing_value_but_explicit_confirmation_keys_still_asks_transparently():
    async def go():
        s,rec,ai=setup();worker=s.spawn(s._stt_worker());ai.kind='info'
        rec.text='맞으면 1번, 다시 입력하시려면 2번을 눌러 주세요.'
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert s.pending and '정확히 읽지 못했어요' in s.pending['message']
            assert '입력한 번호: 010-0000-0000' in s.pending['message']
            # The separately labelled recognized value is never filled from input.
            assert '음성 인식 번호: 정확히 읽지 못했어요' in s.pending['message']
            assert {o['key']:o['label'] for o in s.pending['options']}=={'1':'맞아요','2':'다시 입력'}
        finally:await finish(s,worker)
    run(go())


@pytest.mark.parametrize('source',[
 '모바일 신청 URL을 받아보실 휴대폰 번호와 샵 버튼을 눌러 주십시오.',
 '문자를 받으실 11자리 휴대폰 번호를 입력해 주세요.',
 '본인 명의의 휴대폰 번호를 입력한 후 샵을 눌러 주세요.',
])
def test_supported_single_phone_request_grammars_create_confirmation(source):
    async def go():
        s,rec,_=setup();worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s,source=source);await readback_frames(s)
            await settle(lambda:s.stt_inflight==0)
            assert rec.calls and s.pending and s.pending['kind']=='clarification'
        finally:await finish(s,worker)
    run(go())


def test_waiting_for_phone_confirmation_does_not_upload_repeated_private_audio():
    async def go():
        s,rec,_=setup();worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.pending is not None)
            rid=s.pending['request_id'];s.secure_sent_at=time.monotonic()-100
            await feed(s,[SILENCE_FRAME]*12)  # clear the earlier200ms tail before the repeat
            await readback_frames(s);await settle(lambda:s.stt_inflight==0)
            assert len(rec.calls)==1 and s.pending['request_id']==rid
        finally:await finish(s,worker)
    run(go())
