"""Only fictional phone values; no provider calls or real telephony."""
import json
import pytest

from interpreter import Prompt
from recording import CallRecorder
from test_jev_phrase import run
from test_phone_readback_audio import setup,ask_and_send,readback_frames,finish,NUMBER
from test_openai_prefetch import settle

WRONG='01000000001'
SOURCE='입력하신 번호는 01000000001번입니다. 맞으면 1번, 다시 입력하시려면 2번을 눌러 주세요.'


def test_mismatch_shows_both_numbers_and_does_not_repair_recognized_digits():
    async def go():
        s,rec,_=setup();rec.text=SOURCE;worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.pending is not None)
            msg=s.pending['message']
            assert '입력한 번호' in msg and '음성 인식 번호' in msg
            assert '010-0000-0000' in msg and '010-0000-0001' in msg
            assert '달라' in msg and '오류' in msg
            assert s.call.keys==[NUMBER+'#']  # no automatic approval or correction
            await s.action({'type':'select','request_id':s.pending['request_id'],'key':'2'})
            assert s.call.keys==[NUMBER+'#','2']
            assert not s._phone_comparison.has_value
        finally:await finish(s,worker)
    run(go())


def test_all_text_records_and_history_hide_both_phone_values():
    import io
    import zipfile
    async def go():
        s,rec,ai=setup();rec.text=SOURCE;s.recorder=CallRecorder({'test':'synthetic'})
        worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.pending is not None)
            assert '010-0000-0001' in s.pending['message']  # live owner display remains useful
            for data in (s.recorder.events,list(s.hub.history),ai.calls):
                dump=json.dumps(data,ensure_ascii=False)
                assert all(value not in dump for value in (NUMBER,WRONG,'010-0000-0000','010-0000-0001'))
            with zipfile.ZipFile(io.BytesIO(s.recorder.export_zip())) as archive:
                for name in ('log.json','log.txt'):
                    text=archive.read(name).decode('utf-8')
                    assert all(value not in text for value in (NUMBER,WRONG,'010-0000-0000','010-0000-0001'))
        finally:await finish(s,worker)
    run(go())


def test_closed_or_expired_private_request_drops_the_live_phone_message():
    async def go():
        s,rec,_=setup();rec.text=SOURCE;worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.pending is not None)
            req=s.pending
            await s._clarify_expire(req['request_id'])
            assert '010-' not in req['message']
            assert '010-' not in s.pending['message']
        finally:await finish(s,worker)
    run(go())


@pytest.mark.parametrize('label,source',[
    ('계좌번호','계좌번호를 입력해 주세요.'),
    ('휴대폰번호','휴대폰 번호와 인증번호를 입력해 주세요.'),
])
def test_nonphone_or_combined_private_fields_are_never_retained(label,source):
    from phone_comparison import PhoneComparison
    holder=PhoneComparison()
    holder.arm(label,source,'#')
    assert not holder.begin(NUMBER+'#') and not holder.has_value


def test_phone_holder_consumes_once_and_expires_without_readback():
    from phone_comparison import PhoneComparison
    now=[0.];holder=PhoneComparison(clock=lambda:now[0])
    holder.arm('휴대폰번호','휴대폰 번호를 입력하고 샵 버튼을 눌러 주세요.','#')
    assert holder.begin(NUMBER+'#')
    assert NUMBER not in repr(holder)
    assert holder.take()==NUMBER and holder.take() is None
    holder.arm('휴대폰번호','휴대폰 번호를 입력하고 샵 버튼을 눌러 주세요.','#')
    assert holder.begin(NUMBER+'#')
    now[0]=31.
    assert not holder.has_value and holder.take() is None


def test_missing_or_multiple_recognized_numbers_are_not_filled_from_input():
    from phone_comparison import comparison_message
    for source in ('번호를 다시 확인해 주세요. 맞으면 1번, 다시 입력하시려면 2번.',
                   '01000000001 또는 01000000002입니다. 맞으면 1번, 다시 입력하시려면 2번.'):
        message=comparison_message(source,NUMBER)
        assert '입력한 번호: 010-0000-0000' in message
        assert '음성 인식 번호: 정확히 읽지 못했어요' in message


def test_replacement_private_field_and_old_expiry_cannot_reuse_phone():
    async def go():
        s,_,_=setup()
        try:
            await ask_and_send(s)
            old=s._phone_comparison.generation
            assert s._phone_comparison.has_value
            await s._ask_secure(Prompt('sensitive','계좌번호를 입력해 주세요.',label='계좌번호'))
            assert not s._phone_comparison.has_value
            await ask_and_send(s)
            await s._expire_phone_comparison(old)
            assert s._phone_comparison.has_value
            await s._expire_phone_comparison(s._phone_comparison.generation)
            assert not s._phone_comparison.has_value
        finally:await finish(s)
    run(go())


def test_send_failure_clears_comparison_and_does_not_log_input(caplog):
    from session import ApiError
    async def go():
        s,_,_=setup()
        async def broken(keys):raise RuntimeError('private input '+NUMBER)
        s.call.send_keys=broken
        try:
            with pytest.raises(ApiError):await ask_and_send(s)
            assert not s._phone_comparison.has_value
            assert NUMBER not in caplog.text
        finally:await finish(s)
    run(go())


def test_queued_number_is_not_available_for_comparison_before_sending():
    async def go():
        s,_,_=setup();blocked=[True]
        s._secure_send_blocked=lambda:blocked[0]
        s.s.secure_wait_max_s=1.
        try:
            await ask_and_send(s)
            assert s.secure_queued and s.call.keys==[]
            assert not s._phone_comparison.has_value
            blocked[0]=False
            await settle(lambda:bool(s.call.keys))
            assert s._phone_comparison.has_value
        finally:await finish(s)
    run(go())


def test_end_and_request_close_sanitize_the_private_live_message():
    async def go():
        s,rec,_=setup();rec.text=SOURCE;worker=s.spawn(s._stt_worker())
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.pending is not None)
            req=s.pending
            assert '010-' in req['message']
            await s.finish('user_canceled',hangup=False)
            assert '010-' not in req['message']
            assert s.pending is None and not s._phone_comparison.has_value
        finally:await finish(s,worker)
    run(go())


def test_phone_confirmation_cannot_forward_a_spoken_private_reply():
    from session import ApiError
    async def go():
        s,rec,ai=setup();rec.text=SOURCE;worker=s.spawn(s._stt_worker())
        s.recorder=CallRecorder({'test':'synthetic'})
        try:
            await ask_and_send(s);await readback_frames(s);await settle(lambda:s.pending is not None)
            before=len(ai.calls)
            with pytest.raises(ApiError):
                await s.action({'type':'answer','request_id':s.pending['request_id'],
                                'text':'내 번호는 '+NUMBER})
            assert len(ai.calls)==before
            assert s.pending is not None and s.call.keys==[NUMBER+'#']
            assert NUMBER not in json.dumps(s.recorder.events,ensure_ascii=False)
            assert NUMBER not in s.recorder.text_log()
        finally:await finish(s,worker)
    run(go())


def test_valid_public_answer_is_still_recorded():
    from interpreter import Option
    async def go():
        s,_,_=setup();s.recorder=CallRecorder({'test':'synthetic'})
        try:
            p=Prompt('menu','어떤 고객인가요?',[Option('1','개인 고객'),Option('2','기업 고객')])
            await s._ask(p,'ambiguous',p.options)
            await s.action({'type':'answer','request_id':s.pending['request_id'],'text':'개인 고객이에요'})
            assert '개인 고객이에요' in s.recorder.text_log()
        finally:await finish(s)
    run(go())


def test_ambiguous_numeric_terminator_does_not_retain_a_guessed_input():
    from phone_comparison import PhoneComparison
    holder=PhoneComparison()
    holder.arm('휴대폰번호','휴대폰 번호와 1번 버튼을 눌러 주세요.','1')
    assert not holder.begin('01000000001')
    assert not holder.has_value


@pytest.mark.parametrize('entered,recognized,same',[
    ('01000000000','01000000000',True),
    ('07000000000','07000000001',False),
    ('0299990000','0299990001',False),
    ('01000000000','0100000000',False),
])
def test_comparison_preserves_matching_mismatching_and_wrong_length_values(entered,recognized,same):
    from phone_comparison import comparison_message
    from readback_display import _phone
    message=comparison_message('입력하신 번호는 '+recognized+'입니다. 맞으면 1번, 다시 입력하시려면 2번.',entered)
    assert '입력한 번호: '+_phone(entered) in message
    assert '음성 인식 번호: '+_phone(recognized) in message
    assert ('달라' in message)==(not same)


def test_local_private_readback_compares_before_closing_old_request():
    from types import SimpleNamespace
    async def go():
        s,rec,ai=setup(enabled=False)
        async def recognize(audio):return SOURCE
        s.d.private_recognizer=SimpleNamespace(recognize=recognize)
        try:
            await ask_and_send(s)
            assert s._phone_comparison.has_value
            await s._protected_segment(b'local-synthetic',None,s._private_generation)
            assert '입력한 번호: 010-0000-0000' in s.pending['message']
            assert '음성 인식 번호: 010-0000-0001' in s.pending['message']
            assert not rec.calls and not ai.calls
        finally:await finish(s)
    run(go())
