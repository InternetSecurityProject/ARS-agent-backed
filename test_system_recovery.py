"""Recovery is system work; clarification is reserved for missing user details.

Real adapter/navigator/session, scripted external AI and phone only. No API/calls.
"""
import asyncio
import time

import pytest

from interpreter import Option, Prompt
from navigator import Layer, Navigator, NavState
from test_jev_phrase import AI, answer, engine, item, run


class RecoveryAI(AI):
    def __init__(self, *args, missing_pair=None, repair_items=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.missing_pair = missing_pair
        self.repair_items = repair_items
        self.repair_seen = False

    async def decide(self, state, questions):
        if state.get('system_reanalysis'):
            self.repair_seen = True
            if self.repair_items is not None:
                self.items = self.repair_items
        extras = {k:q for k,q in questions.items() if k in ('clarification_pair','user_detail_required')}
        standard = {k:q for k,q in questions.items() if k not in extras}
        out = await super().decide(state, standard)
        for name,q in extras.items():
            if name == 'user_detail_required':
                out[name] = {'type':'noul','noul':.99 if self.missing_pair else .01}
            else:
                pick = next((k for k,v in q['criteria'].items()
                             if isinstance(v,dict) and v.get('keys') == list(self.missing_pair or [])), 'none')
                out[name] = answer(q, pick)
        return out


async def decision(client, text='예약 문의는 1번, 주차 요금은 2번.', goal='예약 변경'):
    i = engine(client)
    p = await i.classify_context(text,goal,[])
    return i,p,await Navigator(i).decide(NavState(goal),p,[])


def test_extraction_uncertainty_is_system_recovery_not_missing_user_info():
    async def go():
        _,_,d = await decision(RecoveryAI({'1':item('예약 문의는',confidence=.4),'2':item('주차 요금은')}))
        assert d.action == 'recover' and d.reason == 'system_uncertain'
    run(go())


def test_route_needs_info_without_identified_user_alternatives_is_recovery():
    async def go():
        _,_,d = await decision(RecoveryAI({'1':item('예약 문의는'),'2':item('주차 요금은')}, route='needs_info'))
        assert d.action == 'recover'
    run(go())


def test_real_missing_detail_asks_only_relevant_options_not_whole_menu():
    async def go():
        _,_,d = await decision(RecoveryAI({'1':item('치과는'),'2':item('내과는'),'4':item('주차는')},
            route='needs_info',missing_pair=('1','2')),
            '치과는 1번, 내과는 2번, 주차는 4번.', '예약 변경')
        assert d.action == 'ask'
        assert [o.key for o in d.options] == ['1','2']
        assert '주차' not in d.message
    run(go())


def test_focused_reanalysis_repairs_source_mapping_and_selects_without_user():
    async def go():
        c=RecoveryAI({'1':item('예약 문의는',confidence=.4),'2':item('주차 요금은')},
                     repair_items={'1':item('예약 문의는'),'2':item('주차 요금은')})
        i,p,d=await decision(c)
        assert d.action=='recover'
        fixed=await i.recover_context(p,'예약 변경',[])
        result=await Navigator(i).decide(NavState('예약 변경'),fixed,[],arrived=False)
        assert result.action=='press' and result.key=='1'
        assert fixed.label_of('1')=='예약 문의'
        assert c.repair_seen
    run(go())


def test_history_contains_source_and_selected_key_even_when_label_unresolved():
    async def go():
        c=RecoveryAI({'3':item('문의는')},key='3');i=engine(c)
        parent=Prompt('menu','장례 용품은 5번, 병원 예약은 7번.',[Option('7','7번 · 메뉴 확인 필요')])
        p=await i.classify_history('문의는 3번.', '병원 예약 변경', [Layer(parent,'7','user')])
        state=next(state for state,q in c.calls if 'matching_key' in q)
        assert state['menu_history'][0]['selected_key']=='7'
        assert state['menu_history'][0]['transcript']=='장례 용품은 5번, 병원 예약은 7번.'
        assert p.label_of('3')=='문의'
    run(go())


def test_reanalysis_also_uses_selected_source_history_without_inventing_a_label():
    class ContextAI(RecoveryAI):
        async def decide(self,state,questions):
            if state.get('system_reanalysis') and state.get('menu_history'):
                self.items={'3':item('문의는')}
            return await super().decide(state,questions)
    async def go():
        c=ContextAI({'3':item('문의는',confidence=.4)},key='3');i=engine(c)
        parent=Prompt('menu','진료 예약은 7번.',[Option('7','7번 · 메뉴 확인 필요')])
        p=await i.classify_history('문의는 3번.','예약 변경',[Layer(parent,'7','user')])
        fixed=await i.recover_context(p,'예약 변경',['7번 · 메뉴 확인 필요'])
        d=await Navigator(i).decide(NavState('예약 변경'),fixed,[])
        assert d.action=='press' and d.key=='3'
        assert fixed.label_of('3')=='문의'  # No missing word is added to ARS text.
    run(go())


@pytest.mark.parametrize('needed,want',[(.65,'ask'),(.4,'recover')])
def test_grounded_user_detail_question_does_not_require_automatic_press_confidence(needed,want):
    class DetailAI(RecoveryAI):
        async def decide(self,state,questions):
            out=await super().decide(state,questions)
            if 'route_status' in out:
                out['route_status']=answer(questions['route_status'],'needs_info',.55)
                out['clarification_pair']=answer(questions['clarification_pair'],'pair_0',.75)
                out['user_detail_required']={'type':'noul','noul':needed}
            return out
    async def go():
        _,_,d=await decision(DetailAI({'1':item('치과는'),'2':item('내과는')},
            route='needs_info',missing_pair=('1','2')),'치과는 1번, 내과는 2번.','예약 변경')
        assert d.action==want
    run(go())


class LocalCall:
    provides_transcripts=True
    def __init__(self):self.keys=[]
    async def send_keys(self,key):self.keys.append(key);return 0.
    async def hangup(self):pass


def make_session(client):
    from app import build_deps
    from session import Session
    from test_flows import make_settings
    deps=build_deps(make_settings());deps.interp=engine(client)
    s=Session('5550000',deps);s.call=LocalCall();s.answered_at=time.monotonic()
    s.goal='예약 변경';s.nav=NavState(s.goal);s.agent_active=True;s.status='listening'
    return s


def test_session_repairs_and_sends_business_key_without_clarification():
    async def go():
        c=RecoveryAI({'1':item('예약 문의는',confidence=.4)},repair_items={'1':item('예약 문의는')})
        s=make_session(c)
        try:
            await s.on_paragraph('예약 문의는 1번.')
            assert s.call.keys==['1'] and s.pending is None
            assert s.layers[-1].prompt.category_of('1')=='예약 문의'
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_system_failure_replays_only_verified_current_key_once_without_user_question():
    async def go():
        c=RecoveryAI({'1':item('예약 문의는',confidence=.4),'9':item('다시 듣기는','repeat')})
        s=make_session(c)
        try:
            for _ in range(4):
                await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            assert s.call.keys==['9'] and s.pending is None and not s.ended
            assert not s.layers
            assert '보류' in s.status_message
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_system_failure_without_replay_does_not_borrow_parent_key_or_ask_goal():
    async def go():
        s=make_session(RecoveryAI({'1':item('예약 문의는',confidence=.4)}))
        s.layers=[Layer(Prompt('menu','상위',[Option('7','이전 서비스')],repeat_key='9'),'7','user')]
        try:
            await s.on_paragraph('예약 문의는 1번.')
            assert s.call.keys==[] and s.pending is None and not s.ended
            assert '보류' in s.status_message
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_reanalysis_transport_failure_preserves_already_verified_current_replay():
    class RepairFailure(RecoveryAI):
        async def decide(self,state,questions):
            if state.get('system_reanalysis'):
                raise TimeoutError('do-not-log-private-error')
            return await super().decide(state,questions)
    async def go():
        s=make_session(RepairFailure({'1':item('예약 문의는',confidence=.4),
                                     '9':item('다시 듣기는','repeat')}))
        try:
            await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            assert s.call.keys==['9'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_hangup_during_reanalysis_never_sends_a_late_key_or_question():
    class EndingAI(RecoveryAI):
        session=None
        async def decide(self,state,questions):
            if state.get('system_reanalysis'):
                self.session.hangup_seen=True
            return await super().decide(state,questions)
    async def go():
        c=EndingAI({'1':item('예약 문의는',confidence=.4)},repair_items={'1':item('예약 문의는')})
        s=make_session(c);c.session=s
        try:
            await s.on_paragraph('예약 문의는 1번.')
            assert not s.call.keys and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_reanalysis_has_a_total_deadline_even_when_the_provider_hangs():
    class HangingAI(RecoveryAI):
        async def decide(self,state,questions):
            if state.get('system_reanalysis'):
                await asyncio.sleep(30)
            return await super().decide(state,questions)
    async def go():
        c=HangingAI({'1':item('예약 문의는',confidence=.4)});i,p,_=await decision(c)
        started=time.monotonic()
        fixed=await asyncio.wait_for(i.recover_context(p,'예약 변경',[]),timeout=2.3)
        assert time.monotonic()-started<2.3
        assert fixed is p
    run(go())


def test_recovery_replay_cannot_be_repeated_by_baseline_no_response_timer():
    async def go():
        s=make_session(RecoveryAI({'1':item('예약 문의는',confidence=.4),'9':item('다시 듣기는','repeat')}))
        try:
            await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            await s._no_response(s.current)
            assert s.call.keys==['9'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_genuine_missing_confirmation_fact_is_asked_instead_of_system_recovery():
    async def go():
        s=make_session(RecoveryAI({'1':item('안내받은 예약일이 맞으면'),'2':item('틀리면')},
            kind='confirm',route='needs_info',missing_pair=('1','2')))
        try:
            await s.on_paragraph('안내받은 예약일이 맞으면 1번, 틀리면 2번.')
            assert s.pending and s.pending['kind']=='clarification'
            assert [o['key'] for o in s.pending['options']]==['1','2']
            assert not s.call.keys
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_actual_confirmation_selection_gives_next_menu_a_fresh_recovery_budget():
    async def go():
        c=RecoveryAI({'1':item('맞으면'),'2':item('틀리면')},kind='confirm')
        s=make_session(c)
        try:
            await s.on_paragraph('맞으면 1번, 틀리면 2번.')
            await s._press('1','user',s.current)
            c.kind='menu';c.key='3';c.items={'3':item('예약 문의는',confidence=.4)}
            c.repair_items={'3':item('예약 문의는')};c.repair_seen=False
            await s.on_paragraph('예약 문의는 3번.')
            assert c.repair_seen and s.call.keys==['1','3'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_recovery_does_not_interrupt_an_announcement_already_being_replayed():
    async def go():
        s=make_session(RecoveryAI({'1':item('예약 문의는',confidence=.4),'9':item('다시 듣기는','repeat')}))
        s._ars_active=lambda:True
        try:
            await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            assert not s.call.keys and s.pending is None
            s._ars_active=lambda:False
            await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            assert s.call.keys==['9']
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_sensitive_input_still_requests_digits_instead_of_recovering():
    async def go():
        s=make_session(RecoveryAI({}))
        try:
            await s.on_paragraph('생년월일 6자리를 입력하고 샵을 눌러 주세요.')
            assert s.pending['kind']=='secure_input' and not s.call.keys
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_critical_approval_still_requires_explicit_user_selection():
    async def go():
        s=make_session(RecoveryAI({'1':item('결제를 진행하려면'),'2':item('취소는')},kind='confirm',critical=.99))
        try:
            await s.on_paragraph('결제를 진행하려면 1번, 취소는 2번.')
            assert s.pending['reason']=='critical_action' and not s.call.keys
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
