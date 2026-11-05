"""Source-phrase adapter safety: fixed AI replies, no real calls or remote APIs."""
import asyncio
import re
import time
import pytest

from interpreter import Prompt, Option, RouteAssessment
from navigator import Navigator, NavState, Layer

KEYS='0123456789*#'

def run(coro):
    return asyncio.run(coro)

def answer(q,pick,confidence=.99):
    return {'type':'choice','choice':pick,'confidence':confidence,
            'probabilities':{k:(confidence if k==pick else .0001) for k in q['criteria']}}

class AI:
    def __init__(self,items,*,route='proceed',key='1',kind='menu',kind_conf=.99,critical=.01):
        self.items=items;self.route=route;self.key=key;self.kind=kind
        self.kind_conf=kind_conf;self.critical=critical;self.calls=[]
        self.safe_menu=False
        self.fail_extraction=False;self.bad_span=False

    async def decide(self,state,questions):
        self.calls.append((state,questions))
        if not any(re.fullmatch(r'k\d+_role',name) for name in questions):
            picks={'matching_key':'no_selection' if self.route=='unavailable' else
                'dtmf_'+{'*':'star','#':'hash'}.get(self.key,self.key),'route_status':self.route,'meaning_basis':'grounded'}
            out={}
            for name,q in questions.items():
                if name=='clarification_pair':
                    pick=next((k for k,v in q['criteria'].items() if isinstance(v,dict)),'none') if self.route=='needs_info' else 'none'
                    out[name]=answer(q,pick);continue
                if name.startswith('outcome_'):
                    out[name]={'type':'noul','noul':.99};continue
                if name=='purpose_required_now':
                    out[name]={'type':'noul','noul':.01};continue
                if name=='user_detail_required':
                    out[name]={'type':'noul','noul':.99 if self.route=='needs_info' else .01};continue
                if q['type']=='choice':out[name]=answer(q,picks[name])
                else:out[name]={'type':'noul','noul':.01 if self.route=='unavailable' else .99}
            return out
        if self.fail_extraction:raise TimeoutError('mock-secret-do-not-log')
        out={}
        for name,q in questions.items():
            if name=='kind':out[name]=answer(q,self.kind,self.kind_conf);continue
            if name=='critical':out[name]={'type':'noul','noul':self.critical};continue
            if name=='safe_current_menu':out[name]={'type':'noul','noul':.99 if self.safe_menu else .01};continue
            m=re.fullmatch(r'k(\d+)_(role|span)',name);item=self.items.get(KEYS[int(m[1])],{})
            if m[2]=='role':
                out[name]=answer(q,item.get('role','none'),item.get('role_conf',.99))
            else:
                pick=next((k for k,v in q['criteria'].items() if v==item.get('label')),'none')
                if self.bad_span and item:pick='not-a-source-id'
                out[name]=answer(q,pick,item.get('span_conf',.99))
        return out

def item(label,role='business',confidence=.99):
    return {'label':label,'role':role,'role_conf':confidence,'span_conf':confidence}

def engine(client):
    from jev_phrase_interpreter import PhraseJevInterpreter
    return PhraseJevInterpreter(client)

async def decide(text,goal,client):
    i=engine(client);p=await i.classify_context(text,goal,[])
    d=await Navigator(i).decide(NavState(goal),p,[])
    return i,p,d

def test_ai_extracts_prefix_and_symbol_assignments_not_old_menu_regex(monkeypatch):
    import jev_interpreter
    monkeypatch.setattr(jev_interpreter,'extract_prompt',lambda *_:pytest.fail('Old authoritative regex parser called'))
    async def go():
        c=AI({'3':item('교육 신청'),'*':item('분실 신고'),'#':item('다른 서비스입니다','other')},key='3')
        i,p,d=await decide('3번은 교육 신청, 별표는 분실 신고, 샵은 다른 서비스입니다.','교육 신청',c)
        assert [(o.key,o.label) for o in p.options]==[('3','교육 신청'),('*','분실 신고')]
        assert not p.back_key and not p.home_key
        assert d.action=='press' and d.key=='3'
        assert len(c.calls)==2
        assert set(c.calls[0][0])=={'transcript'}
        assert 'matching_key' in c.calls[1][1]
    run(go())

@pytest.mark.parametrize('low_field',['role_conf','span_conf'])
def test_low_confidence_selected_menu_always_asks(low_field):
    async def go():
        a=item('예약 문의는');a[low_field]=.55
        c=AI({'1':a,'2':item('주차 요금은')},key='1')
        _,p,d=await decide('예약 문의는 1번, 주차 요금은 2번.','예약 변경',c)
        assert d.action=='recover' and d.reason=='system_uncertain'
        assert {o.key for o in p.options}=={'1','2'}
    run(go())

def test_low_confidence_wrong_label_is_not_presented_as_a_known_business():
    async def go():
        c=AI({'1':item('주차 요금은',confidence=.4),'2':item('주차 요금은')})
        _,p,d=await decide('예약 문의는 1번, 주차 요금은 2번.','예약 변경',c)
        a=next(o for o in p.options if o.key=='1')
        assert d.action=='recover' and a.category is None
        assert '메뉴 확인 필요' in a.label and '주차' not in a.label
    run(go())

def test_uncertain_role_keeps_an_observed_number_available_for_manual_confirmation():
    async def go():
        c=AI({'1':item('예약 문의는','other',confidence=.4),'2':item('주차 요금은')})
        _,p,d=await decide('예약 문의는 1번, 주차 요금은 2번.','예약 변경',c)
        a=next(o for o in p.options if o.key=='1')
        assert d.action=='recover' and a.category is None and '메뉴 확인 필요' in a.label
    run(go())

def test_missing_user_detail_asks_and_never_assumes_a_card_type():
    async def go():
        c=AI({'1':item('신용 체크카드는'),'2':item('현금카드는')},route='needs_info')
        _,_,d=await decide('신용 체크카드는 1번, 현금카드는 2번.','카드 발급',c)
        assert d.action=='ask' and {o.key for o in d.options}=={'1','2'}
    run(go())

def test_other_services_is_navigation_not_an_automatic_business_choice():
    async def go():
        c=AI({'1':item('주차 안내는'),'#':item('다른 서비스는','other')},route='unavailable')
        _,p,d=await decide('주차 안내는 1번, 다른 서비스는 샵.','예약 변경',c)
        assert [o.key for o in p.options]==['1']
        assert not p.back_key and not p.home_key
        assert d.action=='recover'
    run(go())

def test_confident_complete_extraction_and_explicit_absence_can_end():
    async def go():
        c=AI({'1':item('주차 안내는'),'2':item('진료 시간은')},route='unavailable')
        _,_,d=await decide('주차 안내는 1번, 진료 시간은 2번.','택배 조회',c)
        assert d.action=='end' and d.reason=='not_found'
    run(go())

def test_incomplete_extraction_cannot_authorize_absence():
    async def go():
        c=AI({'1':item('주차 안내는')},route='unavailable')  # Heard 2 was not extracted.
        _,_,d=await decide('주차 안내는 1번, 진료 문의는 2번.','예약 변경',c)
        assert d.action=='recover'
    run(go())

def test_unheard_key_is_rejected_even_if_ai_is_very_confident():
    async def go():
        c=AI({'1':item('주차 안내는'),'8':item('주차 안내는')},route='unavailable')
        _,p,d=await decide('주차 안내는 1번.','예약 변경',c)
        assert '8' not in {o.key for o in p.options} and d.action=='recover'
    run(go())

@pytest.mark.parametrize('bad',['timeout','span'])
def test_failed_or_invalid_source_extraction_asks_without_regex_fallback(bad,caplog):
    async def go():
        c=AI({'1':item('진료 예약은')})
        c.fail_extraction=bad=='timeout';c.bad_span=bad=='span'
        _,p,d=await decide('진료 예약은 1번.','진료 예약',c)
        assert not p.options and d.action=='recover'
        assert len(c.calls)==1
    run(go())
    assert 'mock-secret-do-not-log' not in caplog.text

def test_conflicting_number_assignment_is_not_auto_selected():
    async def go():
        c=AI({'1':item('신규 예약은')})
        _,p,d=await decide('신규 예약은 1번, 예약 변경도 1번.','예약 변경',c)
        assert not p.options and d.action=='recover'
    run(go())

def test_coordinated_services_keep_full_source_label():
    async def go():
        c=AI({'1':item('신규 예약, 예약 변경은')})
        _,p,d=await decide('신규 예약, 예약 변경은 1번.','예약 변경',c)
        assert p.options[0].label=='신규 예약, 예약 변경'
        assert d.action=='press' and d.key=='1'
    run(go())

def test_english_key_and_unpunctuated_menu_are_read_from_source():
    async def go():
        c=AI({'6':item('for foreign languages service'),'2':item('진료 문의는')},key='6')
        _,p,d=await decide('for foreign languages service press 6 진료 문의는 2번','외국어 상담',c)
        assert {o.key for o in p.options}=={'6','2'} and d.key=='6'
    run(go())

def test_low_classification_confidence_cannot_authorize_automatic_input():
    async def go():
        _,_,d=await decide('진료 문의는 1번.','진료 문의',
                           AI({'1':item('진료 문의는')},kind_conf=.4))
        assert d.action=='recover'
    run(go())

def test_sensitive_request_keeps_secure_input_guard_without_remote_calls():
    async def go():
        c=AI({});p=await engine(c).classify_context('생년월일 6자리를 입력하고 샵을 눌러 주세요.','진료 예약',[])
        assert p.kind=='sensitive' and not c.calls
    run(go())

def test_critical_action_is_not_automatically_approved():
    async def go():
        c=AI({'1':item('결제를 진행하려면'),'2':item('취소는')},kind='confirm',critical=.99)
        i,p,d=await decide('결제를 진행하려면 1번, 취소는 2번.','결제',c)
        assert p.critical and d.action=='ask' and d.reason=='critical_action'
        assert await i.answer_confirm('결제',p) is None
    run(go())

def test_info_and_result_remain_non_menu_states():
    async def go():
        for kind,text in [('info','잠시만 기다려 주세요.'),('result','발송되었습니다.')]:
            c=AI({},kind=kind);p=await engine(c).classify_context(text,'안내',[])
            assert p.kind==kind and not p.options and len(c.calls)==1
    run(go())

@pytest.mark.parametrize('kind',['info','agent_human'])
def test_conflicting_non_menu_classification_and_business_roles_asks(kind):
    async def go():
        _,p,d=await decide('진료 문의는 1번.','진료 문의',
                           AI({'1':item('진료 문의는')},kind=kind))
        assert p.kind=='menu' and d.action=='recover'
    run(go())

def test_session_fork_preserves_new_adapter_and_never_shares_extraction_state():
    async def go():
        i=engine(AI({'1':item('진료 문의는')}));a=i.new_session();b=i.new_session()
        assert type(a) is type(i) and type(b) is type(i)
        await a.classify_context('진료 문의는 1번.','진료 문의',[])
        assert a._extractions and not b._extractions and not i._extractions
    run(go())

def test_cached_extraction_reused_but_changed_goal_gets_new_route_judgment():
    async def go():
        c=AI({'1':item('진료 문의는')});i=engine(c)
        p=await i.classify_context('진료 문의는 1번.','진료 문의',[])
        await i.score_prompt('진료 문의',p,p.options,[])
        assert len(c.calls)==2
        await i.score_prompt('예약 변경',p,p.options,[])
        assert len(c.calls)==3 and sum('k0_role' in q for _,q in c.calls)==1
    run(go())

def test_fresh_low_confidence_extraction_does_not_reuse_old_automatic_route():
    async def go():
        c=AI({'1':item('진료 문의는')});i=engine(c);text='진료 문의는 1번.'
        p=await i.classify_context(text,'진료 문의',[])
        assert (await Navigator(i).decide(NavState('진료 문의'),p,[])).action=='press'
        i._extractions.clear()  # Supported bounded-cache eviction.
        c.items['1']['span_conf']=.3
        p=await i.classify_context(text,'진료 문의',[])
        assert (await Navigator(i).decide(NavState('진료 문의'),p,[])).action=='recover'
    run(go())

def test_normalized_alias_cache_cannot_override_exact_source_uncertainty():
    async def go():
        c=AI({'*':item('진료 문의는')},key='*',kind_conf=.4);i=engine(c)
        uncertain=await i.classify_context('진료 문의는 별표.','진료 문의',[])
        c.kind_conf=.99
        await i.classify_context('진료 문의는 *.','진료 문의',[])
        d=await Navigator(i).decide(NavState('진료 문의'),uncertain,[])
        assert d.action=='recover'
    run(go())

def test_old_prompt_cannot_borrow_a_new_extraction_version():
    async def go():
        c=AI({'1':item('진료 문의는')},kind_conf=.4);i=engine(c)
        old=await i.classify_context('진료 문의는 1번.','진료 문의',[])
        i._extractions.clear();c.kind_conf=.99
        await i.classify_context('진료 문의는 1번.','진료 문의',[])
        assert (await Navigator(i).decide(NavState('진료 문의'),old,[])).action=='recover'
    run(go())

def test_parent_other_navigation_is_checked_even_with_no_business_alternative():
    async def go():
        c=AI({'1':item('주차 안내는'),'#':item('다른 서비스는','other')});i=engine(c)
        parent=await i.classify_context('주차 안내는 1번, 다른 서비스는 샵.','예약 변경',[])
        c.items={'2':item('진료 시간은')};c.route='unavailable'
        child=await i.classify_context('진료 시간은 2번.','예약 변경',['주차 안내'])
        d=await Navigator(i).decide(NavState('예약 변경'),child,[Layer(parent,'1','user')])
        assert d.action=='recover'
    run(go())

def test_complete_parent_without_remaining_options_still_allows_confirmed_absence():
    async def go():
        c=AI({'1':item('주차 안내는')});i=engine(c)
        parent=await i.classify_context('주차 안내는 1번.','예약 변경',[])
        c.items={'2':item('진료 시간은')};c.route='unavailable'
        child=await i.classify_context('진료 시간은 2번.','예약 변경',['주차 안내'])
        d=await Navigator(i).decide(NavState('예약 변경'),child,[Layer(parent,'1','user')])
        assert d.action=='end' and d.reason=='not_found'
    run(go())

def test_uncertain_confirmation_consumes_previous_answer_without_applying_it_later():
    async def go():
        c=AI({'1':item('맞으면',confidence=.4),'2':item('틀리면')},kind='confirm')
        i=engine(c);p=await i.classify_context('맞으면 1번, 틀리면 2번.','예약 변경',[])
        await i.merge_answer('예약 변경','맞나요?','네')
        assert i.last_answer
        assert await i.answer_confirm('예약 변경',p) is None
        assert i.last_answer is None
    run(go())

def test_candidate_pool_failure_asks_without_fallback_or_remote_request(monkeypatch):
    import jev_phrase_interpreter as module
    def overflow(_):raise ValueError('candidate_pool_overflow')
    monkeypatch.setattr(module,'extract_questions',overflow)
    async def go():
        c=AI({'1':item('진료 문의는')})
        _,p,d=await decide('진료 문의는 1번.','진료 문의',c)
        assert not p.options and d.action=='recover' and not c.calls
    run(go())

def test_capping_uncertain_winner_cannot_silently_select_another_key():
    class NearTie(AI):
        async def decide(self,state,questions):
            a=await super().decide(state,questions)
            if 'matching_key' in a:
                a['matching_key']['probabilities']['dtmf_1']=.8
                a['matching_key']['probabilities']['dtmf_2']=.7
            return a
    async def go():
        c=NearTie({'1':item('예약 문의는',confidence=.4),'2':item('주차 요금은')})
        _,_,d=await decide('예약 문의는 1번, 주차 요금은 2번.','예약 변경',c)
        assert d.action=='recover'
    run(go())

def test_uncertain_return_navigation_does_not_end_even_if_parent_has_a_route():
    class Routes:
        async def score_prompt(self,goal,prompt,options,path):
            return {o.key:(.99 if prompt.text=='parent' else 0.) for o in options}
        async def assess_route(self,goal,prompt,options,path):
            return RouteAssessment('proceed' if prompt.text=='parent' else 'unavailable',.99)
    async def go():
        parent=Prompt('menu','parent',[Option('1','주차'),Option('2','진료 예약')])
        current=Prompt('menu','child',[Option('7','다른 업무')])  # No trusted back/home key.
        d=await Navigator(Routes()).decide(NavState('예약 변경'),current,[Layer(parent,'1','ai')])
        assert d.action=='ask'
    run(go())

def test_existing_session_questions_and_explicit_button_selection_work_with_new_engine():
    from app import build_deps
    from session import Session
    from test_flows import make_settings
    class Call:
        provides_transcripts=True
        def __init__(self):self.keys=[]
        async def send_keys(self,key):self.keys.append(key);return .01
        async def hangup(self):pass
    async def go():
        c=AI({'1':item('치과는'),'2':item('내과는')},route='needs_info')
        deps=build_deps(make_settings());deps.interp=engine(c)
        s=Session('5550000',deps);s.call=Call();s.answered_at=time.monotonic()
        s.goal='예약 변경';s.nav=NavState(s.goal);s.agent_active=True;s.status='listening'
        try:
            await s.on_paragraph('치과는 1번, 내과는 2번.')
            assert type(s.interp) is type(deps.interp)
            assert s.pending['kind']=='clarification' and not s.call.keys
            await s.action({'type':'select','request_id':s.pending['request_id'],'key':'1'})
            assert s.call.keys==['1']
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
