"""Only the explicitly spoken destination may name a displayed address choice."""
import pytest
from interpreter import Option
from choice_display import choice_display_options
from test_jev_phrase import run
from test_system_recovery import RecoveryAI,make_session

@pytest.mark.parametrize('source,first,second',[
 ('주민등록상 주소지 발급은 1번. 회사 주소지 발급은 2번을 눌러주십시오.','주민등록 주소로 받기','회사 주소로 받기'),
 ('주민 등록상 주소지 발급은 1번. 회사 주소지 발급은 2번을 눌러주십시오.','주민등록 주소로 받기','회사 주소로 받기'),
 ('주민등록상의 주소지로 발급받으시려면 1번, 직장 주소지로 발급받으시려면 2번.','주민등록 주소로 받기','회사 주소로 받기'),
 ('주민등록 주소로 받으시려면 1번, 사업장 주소로 받으시려면 2번.','주민등록 주소로 받기','사업장 주소로 받기'),
 ('주민등록상 주소로 받으시려면 1번, 다른 주소로 받으시려면 2번.','주민등록 주소로 받기','다른 주소로 받기'),
])
def test_destination_labels_follow_explicit_source_not_an_assumed_other_address(source,first,second):
    shown=choice_display_options(source,[Option('1','1번'),Option('2','2번')])
    assert [(o.key,o.label) for o in shown]==[('1',first),('2',second)]

@pytest.mark.parametrize('office,resident',[('7','3'),('2','1'),('1','2')])
def test_company_and_resident_meaning_is_independent_of_number_and_option_order(office,resident):
    source=f'회사 주소지 발급은 {office}번, 주민등록상 주소지 발급은 {resident}번.'
    original=[Option(resident,'원문 A','원문 A'),Option(office,'원문 B','원문 B')]
    shown=choice_display_options(source,original)
    assert [(o.key,o.label,o.category) for o in shown]==[(resident,'주민등록 주소로 받기',None),(office,'회사 주소로 받기',None)]
    assert [(o.key,o.label,o.category) for o in original]==[(resident,'원문 A','원문 A'),(office,'원문 B','원문 B')]

@pytest.mark.parametrize('source',[
 '주민등록상 주소지 발급은 1번, 회사 주소지 발급은 1번. 주소 변경은 2번.',
 '주민등록상 주소지 발급은 1번, 회사가 아닌 주소지 발급은 2번.',
 '주민등록상 주소지 발급은 1번, 회사 주소지 발급은 2번이 아닙니다.',
 '주민등록상 주소지가 아닌 곳으로 받으시려면 1번, 회사 주소로 받으시려면 2번.',
 '주민등록상 주소지 발급은 1번, 별도로 안내받은 주소지 발급은 2번.',
 '주민등록상 주소와 회사 주소지 발급은 1번, 다른 주소지 발급은 2번.',
 '회사 및 사업장 주소지 발급은 1번, 다른 주소지 발급은 2번.',
 '회사가 아닌 주민등록상 주소지 발급은 1번, 회사 주소지 발급은 2번.',
])
def test_ambiguous_or_negated_destination_keeps_original_labels(source):
    original=[Option('1','원문 A'),Option('2','원문 B')]
    assert choice_display_options(source,original)==original


def test_private_address_value_is_not_copied_into_destination_labels():
    shown=choice_display_options('주소는 가나시 다라로 123입니다. 주민등록상 주소지 발급은 1번, 회사 주소지 발급은 2번.',[Option('1','1번'),Option('2','2번')])
    assert {o.key:o.label for o in shown}=={'1':'주민등록 주소로 받기','2':'회사 주소로 받기'}


def test_protected_session_displays_actual_destinations_and_only_sends_user_selected_key():
    async def go():
        client=RecoveryAI({});s=make_session(client);s.secure_window=True
        class Offline:
            async def recognize(self,_):return '주민등록상 주소지 발급은 1번. 회사 주소지 발급은 2번을 눌러주십시오.'
        s.d.private_recognizer=Offline()
        try:
            await s._protected_segment(b'local-only',None,s._private_generation)
            req=s.pending
            assert req and req['reason']=='critical_action'
            assert {o['key']:o['label'] for o in req['options']}=={'1':'주민등록 주소로 받기','2':'회사 주소로 받기'}
            assert s.call.keys==[] and client.calls==[]
            assert {o.key:o.label for o in s.current.options}=={'1':'1번','2':'2번'}
            await s.action({'type':'select','request_id':req['request_id'],'key':'2'})
            assert s.call.keys==['2'] and client.calls==[]
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
