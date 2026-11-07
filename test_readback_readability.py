"""Readable readback without digit correction or model privacy regressions."""
import pytest
from interpreter import Option,Prompt
from session import mask_digits
from test_jev_phrase import run
from test_system_recovery import RecoveryAI,make_session


def test_punctuation_split_spoken_digits_are_fully_masked():
    text='입력하신 번호는 공 일 공 일 이. 삼, 사, 오, 육, 칠, 팔. 번입니다. 맞으면 1번, 다시 입력은 2번.'
    masked,hit=mask_digits(text)
    assert hit and '삼, 사' not in masked and '칠, 팔' not in masked
    assert '1번' in masked and '2번' in masked


@pytest.mark.parametrize('text,wanted',[
 ('입력하신 전화번호는 공 일 공 일 이 삼 사 오 육 칠 팔 입니다. 맞으면 1번, 틀리면 0번.', '010-1234-5678'),
 ('입력하신 전화번호는 영 일 영 일 이. 삼, 사, 오, 육, 칠, 팔. 번입니다. 맞으면 3번, 다시 입력은 7번.', '010-1234-5678'),
 ('입력하신 전화번호는 01012345678입니다. 맞으면 1번, 틀리면 0번.', '010-1234-5678'),
])
def test_readback_user_display_formats_exact_spoken_digits(text,wanted):
    from readback_display import readback_message
    assert wanted in readback_message(text,'휴대폰번호',11)


def test_wrong_length_is_not_repaired_or_formatted_as_a_valid_phone():
    from readback_display import readback_message
    text='입력하신 번호는 공 일 공 일 이 삼 사 오 육 칠 팔 구 구 입니다. 맞으면 1번, 틀리면 0번.'
    result=readback_message(text,'휴대폰번호',11)
    assert '0101234567899' in result and '자릿수' in result
    assert '010-1234-5678' not in result


@pytest.mark.parametrize('good,bad',[('1','0'),('3','7'),('1','2')])
def test_display_labels_keep_actual_confirmation_keys(good,bad):
    from readback_display import readback_options
    text=f'입력하신 번호가 맞으면 {good}번, 다시 입력하시려면 {bad}번.'
    p=Prompt('confirm',text,[Option(good,f'{good}번 · 메뉴 확인 필요'),Option(bad,f'{bad}번 · 메뉴 확인 필요')])
    options=readback_options(text,p)
    assert {o.key:o.label for o in options}=={good:'맞아요',bad:'다시 입력'}
    assert p.options[0].category is None  # display proof does not invent AI category evidence


def test_conflicting_or_critical_instructions_are_not_simplified():
    from readback_display import readback_options
    p=Prompt('confirm','',[Option('1','원문 확인'),Option('2','원문 거절')])
    assert readback_options('맞으면 1번, 틀리면 1번.',p)==p.options
    p.critical=True
    assert readback_options('결제에 동의하고 번호가 맞으면 1번, 틀리면 2번.',p)==p.options


def test_session_shows_digits_but_sends_only_masked_source_to_interpreter():
    async def go():
        c=RecoveryAI({'1':{'role':'business','label':'맞으면','role_conf':.3,'span_conf':.3},
                      '2':{'role':'business','label':'다시 입력하시려면','role_conf':.3,'span_conf':.3}},kind='confirm')
        s=make_session(c);s.secure_window=True;s.secure_len=11;s.secure_label='휴대폰번호'
        text='입력하신 번호는 공 일 공 일 이. 삼, 사, 오, 육, 칠, 팔. 번입니다. 맞으면 1번, 다시 입력하시려면 2번.'
        try:
            await s.on_paragraph(text)
            req=s.pending
            assert req and '010-1234-5678' in req['message']
            assert {o['key']:o['label'] for o in req['options']}=={'1':'맞아요','2':'다시 입력'}
            assert '삼, 사' not in req['ars_prompt']
            assert all('공 일 공' not in repr(state) and '삼, 사' not in repr(state) for state,q in c.calls)
            assert not s.captions
            await s.action({'type':'select','request_id':req['request_id'],'key':'2'})
            assert s.call.keys==['2']
            assert s.path[-1]['selected']['label']=='휴대폰번호 재입력'
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_final_notice_is_not_another_secure_input_request():
    from gemini_interpreter import rule_classify
    text='입력하신 휴대폰 번호로 잠시 후 카드 발급 신청 문자를 발송해 드리겠습니다. 이용해 주셔서 감사합니다.'
    assert rule_classify(text).kind=='result'
    assert rule_classify('휴대폰 번호를 다시 입력해 주세요.').kind=='sensitive'


def test_other_private_fields_keep_their_existing_display():
    from readback_display import readback_message
    source='입력하신 생년월일은 구 오 공 일 공 일 입니다. 맞으면 1번, 틀리면 2번.'
    assert readback_message(source,'생년월일',6)==source


@pytest.mark.parametrize('text',[
 '휴대폰 번호를 입력하신 후 잠시 기다려 주세요.',
 '계좌번호를 입력한 뒤 기다려 주세요.',
 '전화번호를 입력하십시오.',
])
def test_sequential_entry_instructions_still_open_secure_input(text):
    from gemini_interpreter import rule_classify
    assert rule_classify(text).kind=='sensitive'


def test_negated_or_unsupported_correction_condition_keeps_original_labels():
    from readback_display import readback_options
    p=Prompt('confirm','',[Option('1','확인 안내'),Option('2','원문 확인 필요')])
    assert readback_options('번호가 맞으면 1번, 다시 입력하지 않으려면 2번.',p)==p.options


def test_wrong_length_mixed_readback_without_confirmation_is_masked():
    async def go():
        c=RecoveryAI({},kind='info');s=make_session(c)
        s.secure_window=True;s.secure_len=11
        try:
            await s.on_paragraph('입력하신 번호는 공1공1이3사5육7팔9입니다. 잠시 기다려 주세요.')
            assert all('공1공1이3사5육7팔9' not in repr(state) for state,q in c.calls)
            assert '공1공1이3사5육7팔9' not in repr(s.captions)
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_clear_final_delivery_notice_does_not_depend_on_ai_kind_confidence():
    async def go():
        s=make_session(RecoveryAI({},kind='info'))
        try:
            await s.on_paragraph('입력하신 휴대폰 번호로 잠시 후 신청 문자를 발송해 드리겠습니다. 이용해 주셔서 감사합니다.')
            assert not s.pending and s.result_heard and s.status=='agent_done'
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_delivery_notice_with_an_explicit_next_choice_is_not_forced_terminal():
    async def go():
        s=make_session(RecoveryAI({},kind='info'))
        try:
            await s.on_paragraph('신청 문자를 발송해 드리겠습니다. 다른 업무를 원하시면 1번.')
            assert not s.result_heard
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


@pytest.mark.parametrize('text',[
 '결제에 동의하시면 신청 문자를 보내 드리겠습니다.',
 '본인 인증을 완료하시면 신청 문자를 보내 드리겠습니다.',
 '신청을 완료한 후 신청 문자를 보내 드리겠습니다.',
])
def test_conditional_delivery_promise_is_not_forced_to_result(text):
    async def go():
        s=make_session(RecoveryAI({},kind='info'))
        try:
            await s.on_paragraph(text)
            assert not s.result_heard
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
