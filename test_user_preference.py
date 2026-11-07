"""Communication/delivery choices belong to the user, not to recovery loops."""
import pytest

from test_jev_phrase import item,run
from test_system_recovery import RecoveryAI,make_session


TEXT='문자 안내를 원하시면 1번, 상담원 연결을 원하시면 0번을 눌러 주세요.'


def client(route='unknown',missing=True,key='1'):
    return RecoveryAI({'1':item('문자 안내를 원하시면'),
                       '0':item('상담원 연결을 원하시면','agent')},
                      route=route,key=key,missing_pair=('0','1') if missing else None)


@pytest.mark.parametrize('route',['unknown','proceed'])
def test_grounded_method_preference_asks_even_when_global_routing_is_uncertain(route):
    async def go():
        c=client(route);s=make_session(c)
        try:
            assert not s.audio_out  # No ARS audio listener is required for user requests.
            await s.on_paragraph(TEXT)
            assert s.pending and s.pending['kind']=='clarification'
            assert {o['key'] for o in s.pending['options']}=={'0','1'}
            assert s.status=='awaiting_user' and not s.call.keys
            assert s.pending['message'] and '문자' in s.pending['message'] and '상담원' in s.pending['message']
            assert any(e['event']=='request' for e in s.hub.history)
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_explicit_method_button_is_required_before_a_key_is_sent():
    async def go():
        s=make_session(client())
        try:
            await s.on_paragraph(TEXT)
            assert s.pending and not s.call.keys
            await s.action({'type':'select','request_id':s.pending['request_id'],'key':'1'})
            assert s.call.keys==['1'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_spoken_method_answer_reuses_existing_answer_route():
    async def go():
        c=client();s=make_session(c)
        try:
            await s.on_paragraph(TEXT)
            assert s.pending
            rid=s.pending['request_id'];c.route='proceed';c.missing_pair=None
            await s.action({'type':'answer','request_id':rid,'text':'문자로 받고 싶어요'})
            assert '문자로 받고 싶어요' in s.goal
            assert s.call.keys==['1'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_system_uncertainty_without_user_preference_evidence_still_recovers():
    async def go():
        s=make_session(client(missing=False))
        try:
            await s.on_paragraph(TEXT)
            assert s.pending is None and not s.call.keys
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_goal_that_already_states_method_can_proceed_without_reasking():
    async def go():
        s=make_session(client(route='proceed',missing=False));s.goal='안내를 문자로 받고 싶어요';s.nav.goal=s.goal
        try:
            await s.on_paragraph(TEXT)
            assert s.call.keys==['1'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
