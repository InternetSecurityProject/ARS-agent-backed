"""Synthetic receive boundaries and resuming recovered menus; no live phone/API."""
import asyncio
import array
import pytest

from audio import FRAME, SILENCE_FRAME, _pcm, dtmf_ulaw
from clawops.agent._audio import pcm16_to_ulaw, ulaw_to_pcm16
from test_flows import manual, wait_for
from test_jev_phrase import answer, item, run
from test_system_recovery import RecoveryAI, make_session


PREFIX=pcm16_to_ulaw(_pcm((311,),160,.25))
TAIL=pcm16_to_ulaw(_pcm((427,),800,.2))


class CaptureSTT:
    is_local=True
    def __init__(self):self.audio=[]
    async def recognize(self,audio):
        self.audio.append(audio)
        return '새로운 안내입니다.'


async def feed(s,audio):
    for pos in range(0,len(audio),FRAME):
        await s._on_audio(audio[pos:pos+FRAME])


@pytest.mark.parametrize('during_send',[True,False])
def test_receive_keeps_first_speech_during_and_immediately_after_key_send(during_send):
    async def go():
        rec=CaptureSTT();s,call=await manual('안내',stt=rec,paragraph_silence_ms=200)
        async def send(_):
            if during_send:await feed(s,PREFIX)
            return .26
        call.send_keys=send
        try:
            await s._send_keys('5')
            if not during_send:await feed(s,PREFIX)
            await feed(s,TAIL+SILENCE_FRAME*15)
            await wait_for(s,lambda _:bool(rec.audio))
            assert PREFIX in rec.audio[0]
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_outgoing_echo_is_removed_without_discarding_new_speech():
    async def go():
        rec=CaptureSTT();s,call=await manual('안내',stt=rec,paragraph_silence_ms=200)
        echo=dtmf_ulaw('5')[:FRAME*6]
        async def send(_):
            await feed(s,echo)
            await feed(s,PREFIX)
            return .26
        call.send_keys=send
        try:
            await s._send_keys('5');await feed(s,TAIL+SILENCE_FRAME*15)
            await wait_for(s,lambda _:bool(rec.audio))
            assert PREFIX in rec.audio[0]
            assert echo not in rec.audio[0]
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_speech_mixed_with_key_echo_is_retained_instead_of_muted():
    tone=array.array('h',ulaw_to_pcm16(dtmf_ulaw('5')[:FRAME*6]))
    voice=array.array('h',ulaw_to_pcm16(pcm16_to_ulaw(_pcm((311,),120,.3))))
    mixed=pcm16_to_ulaw(array.array('h',(max(-32768,min(32767,a+b)) for a,b in zip(tone,voice))).tobytes())
    async def go():
        rec=CaptureSTT();s,call=await manual('안내',stt=rec,paragraph_silence_ms=200)
        async def send(_):await feed(s,mixed);return .26
        call.send_keys=send
        try:
            await s._send_keys('5');await feed(s,TAIL+SILENCE_FRAME*15)
            await wait_for(s,lambda _:bool(rec.audio))
            assert mixed in rec.audio[0]
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


class CorroboratedAI(RecoveryAI):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.safe_menu=True
    async def decide(self,state,questions):
        standard={k:v for k,v in questions.items() if k!='safe_current_menu'}
        out=await super().decide(state,standard)
        if 'safe_current_menu' in questions:
            out['safe_current_menu']={'type':'noul','noul':getattr(self,'safe_strength',.99) if self.safe_menu else .01}
        return out


def test_fresh_corroborated_menu_after_a_timeout_resumes_without_new_user_prompt():
    async def go():
        c=CorroboratedAI({'3':item('문의는')},route='unknown',key='3');s=make_session(c)
        s.goal='카드 신규 발급';s.nav.goal=s.goal
        try:
            await s.on_paragraph('문의는 3번.')
            assert not s.call.keys
            c.items={'3':item('카드 문의는'),'1':item('수표 조회는')}
            c.route='proceed';c.kind='input_error';c.kind_conf=.4
            await s.on_paragraph('입력 시간이 경과했습니다. 카드 문의는 3번, 수표 조회는 1번.')
            assert s.call.keys==['3'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_timeout_and_current_menu_can_both_be_true_without_exclusive_kind_agreement():
    async def go():
        c=CorroboratedAI({'3':item('진료 예약 변경은'),'1':item('검사 결과는')},
                        kind='input_error',kind_conf=.95,key='3')
        c.safe_strength=.82
        s=make_session(c)
        try:
            await s.on_paragraph('입력 시간이 경과했습니다. 진료 예약 변경은 3번, 검사 결과는 1번.')
            assert s.call.keys==['3'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_low_kind_without_independent_actionable_evidence_cannot_press():
    async def go():
        c=CorroboratedAI({'3':item('진료 문의는')},key='3',kind_conf=.4);c.safe_menu=False
        s=make_session(c)
        try:
            await s.on_paragraph('진료 문의는 3번.')
            assert not s.call.keys and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_corroboration_cannot_override_sensitive_classification():
    async def go():
        c=CorroboratedAI({'1':item('번호 확인은')},kind='sensitive')
        s=make_session(c)
        try:
            await s.on_paragraph('번호 확인은 1번.')
            assert not s.call.keys and s.pending['kind']=='secure_input'
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_recovery_replay_is_not_blocked_by_the_current_stt_job_itself():
    async def go():
        c=CorroboratedAI({'3':item('예약 문의는') ,'9':item('다시 듣기는','repeat')},route='unknown',key='3')
        s=make_session(c);s.call.provides_transcripts=False
        class Rec:
            is_local=True
            async def recognize(self,_):return '예약 문의는 3번, 다시 듣기는 9번.'
        s.d.recognizer=Rec()
        task=s.spawn(s._stt_worker())
        try:
            s._queue_segment(PREFIX+TAIL,None)
            await wait_for(s,lambda _:s.stt_inflight==0)
            assert s.call.keys==['9'] and s.pending is None
        finally:
            await s.finish('user_canceled',hangup=False);await task
    run(go())


def test_queued_next_stt_segment_still_blocks_premature_replay():
    async def go():
        c=CorroboratedAI({'3':item('예약 문의는'),'9':item('다시 듣기는','repeat')},route='unknown',key='3')
        s=make_session(c);s.stt_inflight=2
        try:
            await s.on_paragraph('예약 문의는 3번, 다시 듣기는 9번.')
            assert not s.call.keys and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


@pytest.mark.parametrize('key,label,goal',[('4','입원 보험금 청구는','입원 보험금 청구'),('7','검진 일정 변경은','검진 날짜 변경')])
def test_decisive_source_choice_is_not_vetoed_by_weak_duplicate_global_assessment(key,label,goal):
    class Decisive(CorroboratedAI):
        async def decide(self,state,questions):
            out=await super().decide(state,questions)
            if 'route_status' in out:
                out['route_status']=answer(questions['route_status'],'proceed',.55)
            return out
    async def go():
        s=make_session(Decisive({key:item(label),'2':item('주차 안내는')},key=key))
        s.goal=goal;s.nav.goal=goal
        try:
            await s.on_paragraph(f'{label} {key}번, 주차 안내는 2번.')
            assert s.call.keys==[key] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


@pytest.mark.parametrize('weak',['key','service','detail','detail_middle','raw_unknown'])
def test_weak_key_or_service_or_conflicting_user_fact_never_uses_decisive_override(weak):
    class Weak(CorroboratedAI):
        async def decide(self,state,questions):
            out=await super().decide(state,questions)
            if 'route_status' not in out:return out
            out['route_status']=answer(questions['route_status'],'unknown' if weak=='raw_unknown' else 'proceed',.55)
            if weak=='key':
                out['matching_key']=answer(questions['matching_key'],'dtmf_1',.7)
            if weak=='service':out['has_matching_service']={'type':'noul','noul':.65}
            if weak in ('detail','detail_middle'):
                certainty=.8 if weak=='detail_middle' else .99
                out['clarification_pair']=answer(questions['clarification_pair'],'pair_0',certainty)
                out['user_detail_required']={'type':'noul','noul':certainty}
            return out
    async def go():
        c=Weak({'1':item('예약 문의는'),'2':item('주차 안내는')})
        s=make_session(c)
        try:
            await s.on_paragraph('예약 문의는 1번, 주차 안내는 2번.')
            assert not s.call.keys
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_private_input_rejection_during_send_keeps_secure_attribution():
    from test_flows import secure_pending, ERR, ev
    async def go():
        s,call=await secure_pending();s.pending=None;s.secure_queued=True
        async def send(_):
            await s.on_paragraph(ERR)
            return 0.
        call.send_keys=send
        try:
            await s._send_secure_when_quiet('123456')
            rejected=ev(s,'input_rejected')[-1]
            assert rejected['keys'] is None and rejected['actor']=='user'
            assert s.secure_errors==1 and s.input_errors==0
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
