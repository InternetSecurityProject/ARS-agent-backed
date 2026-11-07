"""One semantic verdict after source extraction, with code-owned safety gates."""
import asyncio
import copy
import pytest

from test_jev_phrase import AI, item, run
from navigator import Navigator, NavState


class MenuAI(AI):
    def __init__(self,items,*,decision='press_1',confidence=.99,probability=.99,user_choice='none',**kwargs):
        super().__init__(items,**kwargs)
        self.decision=decision;self.confidence=confidence;self.probability=probability
        self.user_choice=user_choice

    async def decide(self,state,questions):
        if 'decision' not in questions:return await super().decide(state,questions)
        self.calls.append((copy.deepcopy(state),copy.deepcopy(questions)))
        q=questions['decision'];criteria=q['criteria'];p=self.probability
        result={'decision':{'type':'choice','choice':self.decision,'confidence':self.confidence,
            'probabilities':{k:p if k==self.decision else (1-p)/max(1,len(criteria)-1)
                             for k in criteria}}}
        if 'user_choice' in questions:
            choices=questions['user_choice']['criteria']
            result['user_choice']={'type':'choice','choice':self.user_choice,'confidence':.99,
                'probabilities':{k:.99 if k==self.user_choice else .01/(len(choices)-1) for k in choices}}
        return result


def engine(client):
    from jev_menu_interpreter import UnifiedJevInterpreter
    return UnifiedJevInterpreter(client)


@pytest.mark.parametrize('key,label,goal',[
    ('1','예약 문의는','예약 변경'),('4','배송 업무는','받을 주소 변경'),
    ('7','회원 업무는','회원증 다시 받기'),('*','상품 업무는','새 상품 신청'),
])
def test_direct_or_parent_menu_uses_one_semantic_request(key,label,goal):
    async def go():
        code={'*':'star','#':'hash'}.get(key,key)
        c=MenuAI({key:item(label)},decision='press_'+code)
        i=engine(c);p=await i.classify_context(f'{label} {key}번.',goal,[])
        d=await Navigator(i).decide(NavState(goal),p,[])
        assert d.action=='press' and d.key==key
        assert sum('decision' in q for _,q in c.calls)==1
        assert not any('matching_key' in q or 'next_step' in q or 'relationship' in q for _,q in c.calls)
        assert len(c.calls)==2  # source extraction + unified decision
    run(go())


@pytest.mark.parametrize('confidence,probability',[(.79,.99),(.99,.79)])
def test_low_decision_strength_recovers_without_asking_or_ending(confidence,probability):
    async def go():
        i=engine(MenuAI({'1':item('예약 문의는')},confidence=confidence,probability=probability))
        p=await i.classify_context('예약 문의는 1번.','예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action=='recover'
    run(go())


def test_unheard_number_cannot_be_selected_even_with_high_confidence():
    async def go():
        i=engine(MenuAI({'1':item('예약 문의는')},decision='press_9'))
        p=await i.classify_context('예약 문의는 1번.','예약 변경',[])
        assert (await Navigator(i).decide(NavState('예약 변경'),p,[])).action=='recover'
    run(go())


def test_user_preference_asks_only_the_grounded_pair():
    async def go():
        c=MenuAI({'1':item('문자로 신청은'),'2':item('상담원과 신청은'),'3':item('주차 요금은')},decision='ask_pair_1_2')
        i=engine(c);p=await i.classify_context('문자로 신청은 1번, 상담원과 신청은 2번, 주차 요금은 3번.','신청',[])
        d=await Navigator(i).decide(NavState('신청'),p,[])
        assert d.action=='ask' and {o.key for o in d.options}=={'1','2'}
    run(go())


def test_single_explicit_entry_condition_requires_a_user_answer_not_a_key():
    async def go():
        i=engine(MenuAI({'1':item('기업 고객 전용 예약은'),'2':item('주차 요금은')},decision='ask_condition_1'))
        p=await i.classify_context('기업 고객 전용 예약은 1번, 주차 요금은 2번.','예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action=='ask' and d.options==[] and d.key is None
        assert '기업 고객 전용 예약' in d.message
    run(go())


@pytest.mark.parametrize('strength', [.89,.99])
def test_absence_requires_strong_complete_evidence(strength):
    async def go():
        i=engine(MenuAI({'1':item('주차 요금은')},decision='unavailable',confidence=strength,probability=strength))
        p=await i.classify_context('주차 요금은 1번.','예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action==('end' if strength>=.9 else 'recover')
    run(go())


def test_normal_confirmation_never_auto_selects_from_goal():
    async def go():
        c=MenuAI({'1':item('맞으면'),'2':item('아니면')},kind='confirm')
        i=engine(c);p=await i.classify_context('맞으면 1번, 아니면 2번.','예약 변경',[])
        route=await i.assess_route('예약 변경',p,p.options,[])
        assert route.status=='needs_info' and set(route.clarification_keys)=={'1','2'}
        assert not any('decision' in q for _,q in c.calls)
    run(go())


def test_new_session_retains_the_unified_policy_and_isolates_state():
    i=engine(MenuAI({'1':item('예약 문의는')}));child=i.new_session()
    assert type(child) is type(i) and child.client is i.client
    assert child._scored is not i._scored and child._routes is not i._routes


def test_cached_success_cannot_outlive_its_source_evidence():
    async def go():
        i=engine(MenuAI({'1':item('예약 문의는')}))
        p=await i.classify_context('예약 문의는 1번.','예약 변경',[])
        i._extractions.pop(p.text)
        assert all(v<=.4 for v in (await i.score_prompt('예약 변경',p,p.options,[])).values())
    run(go())


@pytest.mark.parametrize('failure',['timeout','nan','bad_sum','invalid_choice'])
def test_invalid_decision_never_sends_or_claims_absence(failure):
    class Broken(MenuAI):
        async def decide(self,state,questions):
            if 'decision' not in questions:return await super().decide(state,questions)
            if failure=='timeout':raise TimeoutError('private-do-not-log')
            result=await super().decide(state,questions);a=result['decision']
            if failure=='nan':a['confidence']=float('nan')
            elif failure=='bad_sum':a['probabilities']={k:.99 for k in a['probabilities']}
            else:a['choice']='private-invalid-choice'
            return result
    async def go():
        i=engine(Broken({'1':item('예약 문의는')}));p=await i.classify_context('예약 문의는 1번.','예약 변경',[])
        assert (await Navigator(i).decide(NavState('예약 변경'),p,[])).action=='recover'
    run(go())


def session(client):
    from test_system_recovery import make_session
    s=make_session(client)
    s.interp=engine(client);s.navigator=Navigator(s.interp)
    return s


def test_secure_input_is_not_a_menu_decision_or_generic_key_action():
    from session import ApiError
    async def go():
        c=MenuAI({});s=session(c)
        try:
            await s.on_paragraph('계좌번호를 입력한 후 샵 버튼을 눌러 주세요.')
            assert s.pending['kind']=='secure_input'
            assert not any('decision' in q for _,q in c.calls)
            with pytest.raises(ApiError):await s._act_key({'key':'1'})
            assert s.call.keys==[]
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_critical_approval_stays_button_only():
    from session import ApiError
    async def go():
        c=MenuAI({'1':item('계약 해지 진행은'),'2':item('취소는')},critical=.99)
        s=session(c)
        try:
            await s.on_paragraph('계약 해지 진행은 1번, 취소는 2번.')
            assert s.pending['reason']=='critical_action' and s.call.keys==[]
            assert not any('decision' in q for _,q in c.calls)
            with pytest.raises(ApiError):
                await s._act_answer({'request_id':s.pending['request_id'],'text':'네'})
            assert s.call.keys==[]
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_ordinary_confirmation_is_shown_to_user_even_with_clear_goal():
    async def go():
        c=MenuAI({'1':item('맞으면'),'2':item('아니면')},kind='confirm');s=session(c)
        try:
            await s.on_paragraph('맞으면 1번, 아니면 2번.')
            assert s.pending['kind']=='clarification'
            assert len(s.pending['options'])==2 and s.call.keys==[]
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_late_model_result_cannot_send_after_session_ends():
    async def go():
        started=asyncio.Event();release=asyncio.Event()
        class Slow(MenuAI):
            async def decide(self,state,questions):
                if 'decision' in questions:
                    started.set();await release.wait()
                return await super().decide(state,questions)
        s=session(Slow({'1':item('예약 문의는')}))
        task=asyncio.create_task(s.on_paragraph('예약 문의는 1번.'))
        await started.wait()
        await s.finish('user_canceled',hangup=False)
        release.set();await task
        assert s.call.keys==[]
    run(go())


def test_replay_refreshes_uncertain_decision_once_without_infinite_resampling():
    async def go():
        class Later(MenuAI):
            count=0
            async def decide(self,state,questions):
                if 'decision' in questions:
                    self.count+=1
                    self.decision='press_1' if self.count>=3 else 'uncertain'
                return await super().decide(state,questions)
        c=Later({'1':item('예약 문의는'),'9':item('다시 듣기는','repeat')});s=session(c)
        try:
            await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            assert s.call.keys==['9']
            await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            assert s.call.keys==['9','1'] and c.count==3
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_entry_condition_yes_does_not_answer_a_later_confirmation():
    async def go():
        c=MenuAI({'1':item('기업 고객 전용 예약은'),'2':item('주차 요금은')},decision='ask_condition_1')
        i=engine(c)
        p=await i.classify_context('기업 고객 전용 예약은 1번, 주차 요금은 2번.','예약 변경',[])
        await i.merge_answer_for_prompt('예약 변경',p.text,'네',allow_context=True)
        c.decision='press_1'
        await i.score_prompt('예약 변경',p,p.options,[])
        c.kind='confirm';c.items={'1':item('맞으면'),'2':item('아니면')}
        confirmation=await i.classify_context('맞으면 1번, 아니면 2번.','예약 변경',[])
        assert await i.answer_confirm('예약 변경',confirmation) is None
    run(go())


def test_explicit_same_confirmation_answer_is_consumed_only_once():
    async def go():
        c=MenuAI({'1':item('맞으면'),'2':item('아니면')},kind='confirm');i=engine(c)
        p=await i.classify_context('맞으면 1번, 아니면 2번.','예약 변경',[])
        await i.merge_answer_for_prompt('예약 변경',p.text,'네',allow_context=True)
        assert await i.answer_confirm('예약 변경',p)=='1'
        assert await i.answer_confirm('예약 변경',p) is None
    run(go())


def test_human_choice_guard_prevents_implicit_sms_selection():
    async def go():
        c=MenuAI({'1':item('문자로 신청은'),'2':item('상담원과 신청은')},
                 decision='press_1',user_choice='ask_pair_1_2')
        i=engine(c);p=await i.classify_context('문자로 신청은 1번, 상담원과 신청은 2번.','신청',[])
        d=await Navigator(i).decide(NavState('신청'),p,[])
        assert d.action=='ask' and {o.key for o in d.options}=={'1','2'}
        assert sum('decision' in q for _,q in c.calls)==1
    run(go())


@pytest.mark.parametrize('failure',['uncertain','bad_distribution'])
def test_uncertain_human_choice_never_authorizes_an_automatic_key(failure):
    class Guard(MenuAI):
        async def decide(self,state,questions):
            result=await super().decide(state,questions)
            if 'user_choice' not in result:return result
            a=result['user_choice'];criteria=questions['user_choice']['criteria']
            if failure=='uncertain':
                a['choice']='uncertain';a['probabilities']={k:.99 if k=='uncertain' else .01/(len(criteria)-1) for k in criteria}
            else:a['probabilities']={k:.99 for k in criteria}
            return result
    async def go():
        i=engine(Guard({'1':item('예약 문의는')}))
        p=await i.classify_context('예약 문의는 1번.','예약 변경',[])
        assert (await Navigator(i).decide(NavState('예약 변경'),p,[])).action=='recover'
    run(go())


@pytest.mark.parametrize('key,other,label,other_label,goal',[
    ('4','1','예약 변경 문의는','일반 회원은','예약 변경'),
    ('7','2','배송 주소 변경은','상품 소개는','배송 주소 변경'),
    ('3','8','회원증 신청 문의는','회원 안내는','회원증 신청'),
])
def test_weak_auxiliary_guess_cannot_veto_a_passing_menu(key,other,label,other_label,goal):
    class WeakGuess(MenuAI):
        async def decide(self,state,questions):
            result=await super().decide(state,questions)
            if 'user_choice' in result:
                q=questions['user_choice']['criteria']
                pair=next(k for k,v in q.items() if isinstance(v,dict) and set(v.get('keys',[]))=={key,other})
                result['user_choice']={'type':'choice','choice':pair,'confidence':.38,
                    'probabilities':{k:.38 if k==pair else .62/(len(q)-1) for k in q}}
            return result
    async def go():
        c=WeakGuess({key:item(label),other:item(other_label)},decision='press_'+key,confidence=.87,probability=.87)
        i=engine(c);p=await i.classify_context(f'{other_label} {other}번, {label} {key}번.',goal,[])
        result=await Navigator(i).decide(NavState(goal),p,[])
        assert result.action=='press' and result.key==key
        assert sum('decision' in q for _,q in c.calls)==1
    run(go())


def test_weak_auxiliary_absence_is_not_permission_to_end():
    class WeakNone(MenuAI):
        async def decide(self,state,questions):
            result=await super().decide(state,questions)
            if 'user_choice' in result:result['user_choice']['confidence']=.38
            return result
    async def go():
        i=engine(WeakNone({'1':item('주차 요금은')},decision='unavailable'))
        p=await i.classify_context('주차 요금은 1번.','예약 변경',[])
        assert (await Navigator(i).decide(NavState('예약 변경'),p,[])).action=='recover'
    run(go())


def test_ignoring_a_weak_guess_does_not_raise_a_weak_primary_decision():
    class Weak(MenuAI):
        async def decide(self,state,questions):
            result=await super().decide(state,questions)
            if 'user_choice' in result:result['user_choice']['confidence']=.38
            return result
    async def go():
        i=engine(Weak({'1':item('예약 문의는')},confidence=.79,probability=.79))
        p=await i.classify_context('예약 문의는 1번.','예약 변경',[])
        assert (await Navigator(i).decide(NavState('예약 변경'),p,[])).action=='recover'
    run(go())


def test_user_fact_guard_prevents_premature_absence():
    async def go():
        c=MenuAI({'1':item('개인 고객 예약은'),'2':item('기업 고객 예약은')},
                 decision='unavailable',user_choice='ask_pair_1_2')
        i=engine(c);p=await i.classify_context('개인 고객 예약은 1번, 기업 고객 예약은 2번.','예약 변경',[])
        assert (await Navigator(i).decide(NavState('예약 변경'),p,[])).action=='ask'
    run(go())


def test_confirmation_answer_cannot_be_reused_for_new_evidence_with_identical_text():
    async def go():
        c=MenuAI({'1':item('맞으면'),'2':item('아니면')},kind='confirm');i=engine(c)
        p=await i.classify_context('맞으면 1번, 아니면 2번.','예약 변경',[])
        await i.merge_answer_for_prompt('예약 변경',p.text,'네',allow_context=True)
        stale=copy.deepcopy(p);stale._phrase_evidence_id='different-observation'
        assert await i.answer_confirm('예약 변경',stale) is None
    run(go())


def test_answered_preference_is_available_to_the_one_decision_request():
    async def go():
        c=MenuAI({'1':item('문자로 신청은'),'2':item('상담원과 신청은')},decision='ask_pair_1_2')
        i=engine(c);p=await i.classify_context('문자로 신청은 1번, 상담원과 신청은 2번.','신청',[])
        goal=await i.merge_answer_for_prompt('신청',p.text,'문자로 할게요',allow_context=True)
        c.decision='press_1';c.user_choice='none'
        result=await i.score_prompt(goal,p,p.options,[])
        state,q=next((s,q) for s,q in reversed(c.calls) if 'decision' in q)
        assert result['1']>=.8
        assert state['current_user_answer']['answer']=='문자로 할게요'
        assert state['intent_context']['explicit_user_replies']==['문자로 할게요']
    run(go())
