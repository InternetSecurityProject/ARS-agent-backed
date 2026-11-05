"""Presentation changes never change keys, AI source or approval requirements."""
import pytest
from interpreter import Option,Prompt
from test_jev_phrase import run
from test_system_recovery import make_session,RecoveryAI

def labels(text,options=None):
    from choice_display import choice_display_options
    options=options or [Option('1','1번'),Option('2','2번')]
    return {o.key:o.label for o in choice_display_options(text,options)}

@pytest.mark.parametrize('text,wanted',[
 ('발급을 원하시면 1번, 취소하시려면 2번을 눌러 주세요.',{'1':'발급하기','2':'취소하기'}),
 ('발급을 원하시면 1번, 원하지 않으시면 2번.',{'1':'발급하기','2':'발급하지 않기'}),
 ('다시 입력하시려면 2번, 확인하시려면 1번.',{'2':'다시 입력하기','1':'확인하기'}),
 ('등록된 주소로 발급받으시려면 1번, 주소 변경 후 발급받으시려면 2번.',{'1':'등록된 주소로 받기','2':'주소 바꿔 받기'}),
 ('기존 주소로 받으시려면 1번, 다른 주소로 받으시려면 2번.',{'1':'등록된 주소로 받기','2':'다른 주소로 받기'}),
 ('주민등록 주소지 발급은 1번, 선택 주소지 발급은 2번.',{'1':'주민등록 주소로 받기','2':'주소 직접 정해 받기'}),
 ('등록된 주소지로 발급받으시려면 1번, 주소지를 변경하여 발급받으시려면 2번.',{'1':'등록된 주소로 받기','2':'주소 바꿔 받기'}),
])
def test_actions_are_named_by_the_actual_keyed_source(text,wanted):
    assert labels(text)==wanted

@pytest.mark.parametrize('text',[
 '발급을 원하지 않으시면 1번, 취소하지 않으려면 2번.',
 '발급은 1번, 취소는 1번. 다시 입력은 2번.',
 '발급은 3번, 취소는 4번.',
 '맞으면 1번, 아니면 2번.',
 '발급은 1번, 발급은 2번.',
 '결제 금액 10000원에 동의하시면 1번, 취소는 2번.',
 '발급을 원하시면 1번, 우편 수령을 원하지 않으시면 2번.',
 '발급을 원하시면 1번, 상담을 원하지 않으시면 2번.',
 '발급은 1번이 아닙니다. 취소는 2번입니다.',
 '발급을 원하시면 1번을 누르지 마세요. 취소는 2번입니다.',
])
def test_unsupported_or_ambiguous_source_keeps_original_labels(text):
    assert labels(text)=={'1':'1번','2':'2번'}

def test_key_order_and_original_model_options_are_unchanged():
    from choice_display import choice_display_options
    opts=[Option('7','원문 재입력','원문 재입력'),Option('3','원문 확인','원문 확인')]
    shown=choice_display_options('확인하시려면 3번, 다시 입력하시려면 7번.',opts)
    assert [(o.key,o.label) for o in shown]==[('7','다시 입력하기'),('3','확인하기')]
    assert all(o.category is None for o in shown)
    assert [(o.key,o.label,o.category) for o in opts]==[('7','원문 재입력','원문 재입력'),('3','원문 확인','원문 확인')]

def test_private_value_is_never_copied_into_labels():
    assert labels('입력 번호는 01012345678입니다. 발급을 원하시면 1번, 취소는 2번.')=={'1':'발급하기','2':'취소하기'}

def test_protected_keyed_menu_has_readable_labels_without_cloud_or_auto_press():
    async def go():
        s=make_session(RecoveryAI({}));s.secure_window=True
        class Local:
            async def recognize(self,_):return '발급을 원하시면 1번, 원하지 않으시면 2번.'
        s.d.private_recognizer=Local()
        try:
            await s._protected_segment(b'fake',None,s._private_generation)
            req=s.pending
            assert req and req['reason']=='critical_action'
            assert {o['key']:o['label'] for o in req['options']}=={'1':'발급하기','2':'발급하지 않기'}
            assert s.call.keys==[] and s.interp.client.calls==[]
            await s.action({'type':'select','request_id':req['request_id'],'key':'2'})
            assert s.call.keys==['2']
        finally:await s.finish('user_canceled',hangup=False)
    run(go())

def test_public_confirmation_presentation_preserves_original_prompt_and_number():
    async def go():
        s=make_session(RecoveryAI({}));p=Prompt('confirm','発給')
        p.text='발급을 원하시면 1번, 취소하시려면 2번.'
        p.options=[Option('1','발급을 원하시면','발급을 원하시면'),Option('2','취소하시려면','취소하시려면')]
        s.current=p;s.current_fresh=True
        try:
            await s._ask(p,'unknown_prompt',p.options)
            req=s.pending
            assert {o['key']:o['label'] for o in req['options']}=={'1':'발급하기','2':'취소하기'}
            assert p.options[0].label=='발급을 원하시면'
            await s.action({'type':'select','request_id':req['request_id'],'key':'1'})
            assert s.call.keys==['1']
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
