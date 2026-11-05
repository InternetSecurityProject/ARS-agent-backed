"""Cross-domain intent policy; typed fake Jev responses, real adapter/navigation."""
import pytest
from interpreter import Prompt
from navigator import Navigator,NavState,Layer
from test_jev_phrase import answer,item,run
from test_system_recovery import RecoveryAI,make_session as base_session

def engine(client):
    from jev_phrase_interpreter import PhraseJevInterpreter
    return PhraseJevInterpreter(client,enable_general_intent=True)

def make_session(client):
    s=base_session(client)
    s.interp.enable_general_intent=True
    return s

class MeaningAI(RecoveryAI):
    def __init__(self,*args,basis='grounded',purpose=.01,scope_conf=.99,**kwargs):
        super().__init__(*args,**kwargs);self.basis=basis;self.purpose=purpose;self.scope_conf=scope_conf
    async def decide(self,state,questions):
        extras={k:q for k,q in questions.items() if k in ('meaning_basis','purpose_required_now')}
        a=await super().decide(state,{k:q for k,q in questions.items() if k not in extras})
        for name,q in extras.items():
            a[name]=answer(q,self.basis,self.scope_conf) if name=='meaning_basis' else {'type':'noul','noul':self.purpose}
        return a

@pytest.mark.parametrize('text,goal,label',[
 ('서류 발급은 1번, 자격 상담은 2번.','보험증 서류 필요해','서류 발급은'),
 ('통신 요금 납부는 1번, 납부 확인서 발급은 2번.','요금 종이 필요해요','통신 요금 납부는'),
 ('진료 예약은 1번, 서류 발급은 2번.','병원 종이 하나 떼려고요','진료 예약은'),
])
def test_missing_purpose_blocks_confident_lexical_choice_and_asks_in_everyday_language(text,goal,label):
    async def go():
        second=text.split('1번, ')[1].split(' 2번')[0]
        c=MeaningAI({'1':item(label),'2':item(second)},basis='purpose_missing',purpose=.99)
        i=engine(c);p=await i.classify_context(text,goal,[])
        d=await Navigator(i).decide(NavState(goal),p,[])
        assert d.action=='ask' and d.reason=='ambiguous' and not d.options
        assert '어떤 일' in d.message and '어디' in d.message and '중' not in d.message
    run(go())


def test_missing_purpose_can_be_asked_without_two_current_menu_candidates():
    async def go():
        c=MeaningAI({'1':item('서류 발급은')},basis='purpose_missing',purpose=.99)
        i=engine(c);p=await i.classify_context('서류 발급은 1번.','서류 하나 필요해',[])
        d=await Navigator(i).decide(NavState('서류 하나 필요해'),p,[])
        assert d.action=='ask' and not d.options
    run(go())

@pytest.mark.parametrize('basis,purpose,confidence',[
 ('system_uncertain',.99,.99),('purpose_missing',.01,.99),
 ('grounded',.99,.99),('purpose_missing',.99,.55),
])
def test_system_uncertainty_or_inconsistent_scope_cannot_press_ask_or_end(basis,purpose,confidence):
    async def go():
        c=MeaningAI({'1':item('예약 업무는'),'2':item('주차 안내는')},basis=basis,purpose=purpose,scope_conf=confidence)
        i=engine(c);p=await i.classify_context('예약 업무는 1번, 주차 안내는 2번.','내일 예약 바꾸고 싶어',[])
        d=await Navigator(i).decide(NavState('내일 예약 바꾸고 싶어'),p,[])
        assert d.action=='recover'
    run(go())

@pytest.mark.parametrize('goal',['내일 잡아둔 시간 좀 미룰래','다음 주 예약으로 바꾸고 싶어','예약 관련해서 처리할 일이 있어'])
def test_clear_informal_goal_or_shared_parent_does_not_ask_usage(goal):
    async def go():
        c=MeaningAI({'1':item('예약 업무는'),'2':item('주차 안내는')},basis='grounded')
        s=make_session(c);s.goal=goal;s.nav.goal=s.goal
        try:
            await s.on_paragraph('예약 업무는 1번, 주차 안내는 2번.')
            assert s.call.keys==['1'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_ai_path_is_marked_as_hypothesis_not_confirmed_user_intent():
    async def go():
        c=MeaningAI({'1':item('서류 발급은'),'2':item('자격 문의는')})
        i=engine(c);text='서류 발급은 1번, 자격 문의는 2번.'
        parent=Prompt('menu',text)
        await i.classify_history(text,'보험증 종이',[Layer(parent,'1','ai')])
        state=next(st for st,q in c.calls if 'matching_key' in q)
        assert state['menu_history'][0]['selected_by']=='ai'
        assert state['intent_context']['original_user_words']=='보험증 종이'
    run(go())


def test_user_purpose_answer_reuses_same_form_and_resolves_without_changing_original_words():
    async def go():
        c=MeaningAI({'1':item('서류 발급은'),'2':item('자격 문의는')},basis='purpose_missing',purpose=.99)
        s=make_session(c);s.goal='보험증 서류 필요해';s.nav.goal=s.goal
        try:
            await s.on_paragraph('서류 발급은 1번, 자격 문의는 2번.')
            assert s.pending and not s.call.keys and s.pending['options']==[]
            rid=s.pending['request_id'];c.basis='grounded';c.purpose=.01
            await s.action({'type':'answer','request_id':rid,'text':'가입되어 있다는 걸 회사에 제출하려고요'})
            assert s.call.keys==['1'] and s.pending is None
            state=next(st for st,q in reversed(c.calls) if 'matching_key' in q)
            assert state['intent_context']['original_user_words']=='보험증 서류 필요해'
            assert state['intent_context']['explicit_user_replies'][-1]=='가입되어 있다는 걸 회사에 제출하려고요'
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_parent_menu_missing_purpose_is_asked_after_wrong_ai_branch():
    from interpreter import RouteAssessment,Option
    class ParentPurpose:
        system_recovery=True
        async def score_prompt(self,goal,prompt,options,path):return {o.key:0 for o in options}
        async def assess_route(self,goal,prompt,options,path):
            return RouteAssessment('unavailable',.99) if prompt.text=='current' else RouteAssessment('needs_info',.99,(),'purpose')
    async def go():
        i=ParentPurpose();p=Prompt('menu','current',[Option('3','다른 업무')],back_key='#')
        parent=Prompt('menu','parent',[Option('1','서류 발급'),Option('2','자격 문의')])
        d=await Navigator(i).decide(NavState('보험증 서류'),p,[Layer(parent,'1','ai')])
        assert d.action=='ask' and not d.options and '어디' in d.message
    run(go())
