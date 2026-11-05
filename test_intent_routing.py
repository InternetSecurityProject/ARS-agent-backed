"""User-intent policy across unrelated menus. Fake AI/telephony; no real calls."""
import asyncio
import time
import pytest

from interpreter import Prompt, Option
from jev_interpreter import JevInterpreter
from navigator import Navigator, NavState, Layer


def run(coro):
    return asyncio.run(coro)


class Replies:
    def __init__(self, route='proceed', key='dtmf_1', available=.99, confidence=.99, resolver=None):
        self.route,self.key,self.available,self.confidence=route,key,available,confidence
        self.resolver=resolver
        self.calls=[]

    async def decide(self, state, questions):
        self.calls.append((state,questions))
        route,key,available=(self.resolver(state) if self.resolver else (self.route,self.key,self.available))
        answers={}
        for name,q in questions.items():
            if q['type']=='choice':
                pick='menu' if name=='kind' else route if name=='route_status' else key
                probability=self.confidence if name=='route_status' else .98
                answers[name]={'type':'choice','choice':pick,'confidence':probability,
                    'probabilities':{k:(probability if k==pick else .001) for k in q['criteria']}}
            else:
                answers[name]={'type':'noul','noul':.01 if name=='critical' else available}
        return answers


async def decide(text,goal,client,*,asked=False,history=None,backtracked=False):
    i=JevInterpreter(client)
    path=[l.prompt.category_of(l.key) for l in history or []]
    p=await i.classify_context(text,goal,path)
    state=NavState(goal,backtracked=backtracked)
    if asked:state.low_asked.add(p.signature());state.sibling_asked.add(p.signature())
    d=await Navigator(i).decide(state,p,history or [])
    return d,i,p


@pytest.mark.parametrize('text,goal',[
    ('신용 체크카드는 1번, 현금카드는 2번.', '카드 신규 발급'),
    ('치과는 1번, 내과는 2번.', '예약 변경'),
    ('휴대전화 요금은 1번, 인터넷 요금은 2번.', '요금 확인'),
])
def test_missing_detail_asks_even_when_all_scores_are_very_low(text,goal):
    async def go():
        c=Replies(route='needs_info',available=.01)
        d,_,_=await decide(text,goal,c,asked=True)
        assert d.action=='ask' and d.reason=='ambiguous'
        assert {o.key for o in d.options}=={'1','2'}
        assert len(c.calls)==1
    run(go())


def test_clear_goal_and_confirmed_absence_can_end_with_notice():
    async def go():
        c=Replies(route='unavailable',key='no_selection',available=.01)
        d,_,_=await decide('주차 요금은 1번, 진료 시간은 2번.','예약 변경',c)
        assert d.action=='end' and d.reason=='not_found'
        assert d.note and len(c.calls)==1
    run(go())


@pytest.mark.parametrize('greeting',['이곳은 병원입니다.', '고객 지원입니다.'])
def test_greeting_does_not_hide_numbered_choices_as_a_financial_result(greeting):
    from jev_interpreter import extract_prompt
    p=extract_prompt(greeting+' 진료 예약은 1번, 주차 요금은 2번.')
    assert p.kind=='menu' and {o.key for o in p.options}=={'1','2'}


@pytest.mark.parametrize('text',['잔액은 1000원입니다.', '조회 금액은 **** 원입니다.', '금액은 삼만원입니다.'])
def test_real_or_masked_amount_still_classifies_as_result(text):
    from gemini_interpreter import rule_classify
    assert rule_classify(text).kind=='result'


@pytest.mark.parametrize('route,key,confidence',[
    ('unknown','dtmf_1',.99),('unavailable','dtmf_1',.99),
    ('unavailable','no_selection',.4),('invalid','no_selection',.99),
])
def test_uncertain_or_contradictory_assessment_never_ends_or_presses(route,key,confidence):
    async def go():
        c=Replies(route=route,key=key,confidence=confidence)
        d,_,_=await decide('진료 접수는 1번, 예약은 2번.','진료 접수',c,asked=True)
        assert d.action=='ask'
    run(go())


def test_api_failure_cannot_become_absence_after_a_previous_question():
    class Broken:
        async def decide(self,*args):raise TimeoutError('not-a-real-secret')
    async def go():
        d,_,_=await decide('진료 접수는 1번, 예약은 2번.','진료 접수',Broken(),asked=True)
        assert d.action=='ask' and d.reason=='unknown_prompt'
    run(go())


def test_uncertain_missing_detail_still_offers_the_actual_choices_as_a_question():
    async def go():
        d,_,_=await decide('치과는 1번, 내과는 2번.','예약 변경',
                           Replies(route='needs_info',confidence=.55,available=.01))
        assert d.action=='ask' and d.reason=='ambiguous'
        assert '치과' in d.message and '내과' in d.message
    run(go())


@pytest.mark.parametrize('contradictory',[True,False])
def test_inconsistent_or_low_confidence_matching_answer_cannot_authorize_press(contradictory):
    class UncertainMatch(Replies):
        async def decide(self,state,questions):
            a=await super().decide(state,questions)
            a['matching_key']['confidence']=.01
            if contradictory:
                a['matching_key']['probabilities']['dtmf_1']=.001
                a['matching_key']['probabilities']['dtmf_2']=.98
            return a
    async def go():
        d,_,_=await decide('진료 접수는 1번, 주차 요금은 2번.','진료 접수',UncertainMatch())
        assert d.action=='ask'
    run(go())


def test_repeated_unresolved_answer_does_not_create_a_card_type_preference():
    async def go():
        i=JevInterpreter(Replies());goal='카드 신규 발급'
        p=await i.classify_context('신용 체크카드는 1번, 현금카드는 2번.',goal,[])
        st=NavState(goal);nav=Navigator(i)
        assert (await nav.decide(st,p,[])).action=='ask'
        st.goal=await i.merge_answer(st.goal,'카드 종류를 골라 주세요','카드 신규 발급')
        assert st.goal==goal
        assert (await nav.decide(st,p,[],arrived=False)).action=='ask'
    run(go())


def test_broader_category_can_advance_to_a_more_specific_task_in_one_request():
    async def go():
        c=Replies()
        d,_,_=await decide('예약 문의는 1번, 주차 요금은 2번.','치과 예약 변경',c)
        assert d.action=='press' and d.key=='1' and len(c.calls)==1
        q=c.calls[0][1]
        assert 'route_status' in q
        assert set(q['route_status']['criteria'])=={'proceed','needs_info','unavailable','unknown'}
    run(go())


def test_unknown_previous_route_does_not_authorize_termination_or_wrong_layer_keys():
    def resolver(state):
        return ('unknown','no_selection',.01) if state['current_ars_transcript']=='parent' else ('unavailable','no_selection',.01)
    async def go():
        parent=Prompt('menu','parent',[Option('1','진료 문의'),Option('2','주차 안내')])
        c=Replies(resolver=resolver)
        d,_,_=await decide('인터넷 접수는 7번, 다른 안내는 8번.','진료 예약',c,
                            history=[Layer(parent,'1','ai')],backtracked=True)
        assert d.action=='ask'
        assert not d.options  # Parent-layer numbers must not appear as current choices.
    run(go())


def test_confirmed_absence_checks_previous_alternative_before_ending():
    def resolver(state):
        return ('proceed','dtmf_2',.99) if state['current_ars_transcript']=='parent' else ('unavailable','no_selection',.01)
    async def go():
        parent=Prompt('menu','parent',[Option('1','진료 문의'),Option('2','예약 문의')])
        c=Replies(resolver=resolver)
        d,_,_=await decide('주차 요금은 7번, 이전 메뉴는 별표.','예약 변경',c,history=[Layer(parent,'1','ai')])
        assert d.action=='backtrack' and d.keys=='*'
        assert len(c.calls)==2
    run(go())


def test_number_answer_preserves_the_original_task():
    async def go():
        i=JevInterpreter(Replies(route='needs_info'))
        await i.classify_context('치과는 1번, 내과는 2번.','예약 변경',[])
        goal=await i.merge_answer('예약 변경','어느 진료과인가요?','1번')
        assert '예약 변경' in goal and '치과' in goal
    run(go())


def test_route_cache_is_scoped_to_goal_options_path_and_call():
    async def go():
        c=Replies(route='unavailable',key='no_selection',available=.01);i=JevInterpreter(c)
        p=await i.classify_context('진료 예약은 1번, 주차 요금은 2번.','택배 조회',[])
        a=await i.assess_route('택배 조회',p,p.options,[])
        assert a.status=='unavailable' and len(c.calls)==1
        c.route,c.key,c.available='needs_info','dtmf_1',.01
        assert (await i.assess_route('예약 변경',p,p.options,[])).status=='needs_info'
        assert len(c.calls)==2
        await i.assess_route('예약 변경',p,p.options,['접수']);assert len(c.calls)==3
        await i.assess_route('예약 변경',p,[p.options[0]],['접수']);assert len(c.calls)==4
        other=i.new_session()
        assert not other._routes
        await other.assess_route('택배 조회',p,p.options,[])
        assert len(c.calls)==5
    run(go())


def test_unavailable_does_not_end_if_previous_route_needs_more_information():
    def resolver(state):
        return ('needs_info','no_selection',.01) if state['current_ars_transcript']=='parent' else ('unavailable','no_selection',.01)
    async def go():
        parent=Prompt('menu','parent',[Option('1','치과'),Option('2','내과')])
        d,_,_=await decide('주차 안내는 7번, 진료 시간은 8번.','예약 변경',Replies(resolver=resolver),
                            history=[Layer(parent,'1','user')])
        assert d.action=='ask' and not d.options
    run(go())


def test_session_asks_then_accepts_answer_and_only_ends_for_confirmed_absence():
    from app import build_deps
    from session import Session
    from test_flows import make_settings
    class Call:
        provides_transcripts=True
        def __init__(self):self.keys=[];self.hangups=0;self.notice=None
        async def send_keys(self,keys):self.keys.append(keys);return .01
        async def hangup(self):
            self.hangups+=1;self.notice=self.session.status_message
    def resolver(state):
        if '주차 요금' in state['current_ars_transcript']:
            return 'unavailable','no_selection',.01
        if len(state['options'])==1:
            return 'unavailable','no_selection',.01
        if '예약 변경' in state['user_goal'] and '치과' in state['user_goal']:
            return 'proceed','dtmf_1',.99
        return 'needs_info','dtmf_1',.01
    async def go():
        deps=build_deps(make_settings());deps.interp=JevInterpreter(Replies(resolver=resolver))
        s=Session('5550000',deps);s.call=Call();s.answered_at=time.monotonic()
        s.call.session=s
        s.goal='예약 변경';s.nav=NavState(s.goal);s.agent_active=True;s.status='listening'
        try:
            await s.on_paragraph('치과는 1번, 내과는 2번.')
            assert not s.ended and s.pending['kind']=='clarification'
            req=s.pending
            await s.action({'type':'answer','request_id':req['request_id'],'text':'1번'})
            assert s.call.keys==['1'] and '예약 변경' in s.goal
            await s.on_paragraph('주차 요금은 4번, 진료 시간은 5번.')
            assert s.end_reason=='not_found' and s.call.hangups==1
            assert '찾지 못' in s.call.notice and '종료' in s.call.notice
            ended=[e for e in s.hub.history if e['event']=='ended']
            assert ended[-1]['data']['message']
        finally:
            if not s.ended:await s.finish('user_canceled',hangup=False)
    run(go())
