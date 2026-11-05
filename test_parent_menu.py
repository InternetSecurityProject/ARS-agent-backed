"""A specific requested action may be inside a proven next parent category."""
import pytest
from test_jev_phrase import item,answer,engine,run
from test_system_recovery import RecoveryAI
from navigator import Navigator,NavState

class ParentAI(RecoveryAI):
    def __init__(self,*args,audit_key='2',relation=.98,bound=.98,ready=.98,pick_conf=.98,malformed=False,**kw):
        super().__init__(*args,route='unknown',**kw)
        self.audit_key=audit_key;self.relation=relation;self.bound=bound
        self.ready=ready;self.pick_conf=pick_conf;self.malformed=malformed
    async def decide(self,state,questions):
        if 'parent_candidates' in state:
            self.calls.append((state,questions))
            if self.malformed:return {}
            result={}
            for name,q in questions.items():
                if name=='next_step':
                    pick=next((k for k,v in q['criteria'].items() if isinstance(v,dict) and v.get('key')==self.audit_key),'none')
                    result[name]=answer(q,pick,self.pick_conf)
                elif name=='user_choice_needed':result[name]={'type':'noul','noul':1-self.ready}
                elif name.endswith('_bound'):result[name]={'type':'noul','noul':self.bound}
                elif name.endswith('_relation'):
                    pick='parent' if name.split('_')[0]=='c'+self.audit_key else 'different'
                    result[name]=answer(q,pick,self.relation)
                    result[name]['probabilities']={k:self.relation if k==pick else (1-self.relation)/3 for k in q['criteria']}
                elif name.endswith(('_target','_action')):result[name]={'type':'noul','noul':.98}
                else:raise AssertionError(name)
            return result
        return await super().decide(state,questions)

@pytest.mark.parametrize('text,goal,labels',[
 ('영수증 발급은 1번, 회원 자격 업무는 2번.','회원증을 다시 받고 싶어요',['영수증 발급은','회원 자격 업무는']),
 ('납부 확인은 1번, 예약 관리는 2번.','잡아 둔 시간을 바꾸고 싶어요',['납부 확인은','예약 관리는']),
 ('상품 소개는 1번, 배송 업무는 2번.','받을 주소를 바꾸고 싶어요',['상품 소개는','배송 업무는']),
])
def test_independent_parent_evidence_can_resolve_uncertain_first_verdict(text,goal,labels):
    async def go():
        c=ParentAI({'1':item(labels[0]),'2':item(labels[1])});i=engine(c)
        p=await i.classify_context(text,goal,[])
        d=await Navigator(i).decide(NavState(goal),p,[])
        assert d.action=='press' and d.key=='2'
    run(go())

@pytest.mark.parametrize('params',[
 {'relation':.7},{'bound':.84},{'ready':.5},{'pick_conf':.5},{'malformed':True},
])
def test_weak_parent_proof_never_authorizes_a_key(params):
    async def go():
        c=ParentAI({'1':item('영수증 발급은'),'2':item('회원 자격 업무는')},**params);i=engine(c)
        p=await i.classify_context('영수증 발급은 1번, 회원 자격 업무는 2번.','회원증 발급',[])
        d=await Navigator(i).decide(NavState('회원증 발급'),p,[])
        assert d.action!='press'
    run(go())


def test_real_user_type_preference_still_asks_instead_of_parent_guessing():
    async def go():
        c=ParentAI({'1':item('개인 고객 예약은'),'2':item('기업 고객 예약은')},missing_pair=('1','2'))
        i=engine(c);p=await i.classify_context('개인 고객 예약은 1번, 기업 고객 예약은 2번.','예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action=='ask' and {o.key for o in d.options}=={'1','2'}
        assert not any('parent_candidates' in state for state,q in c.calls)
    run(go())


def test_similar_word_with_high_first_score_still_requires_relationship_proof():
    class WordOverlap(ParentAI):
        async def decide(self,state,questions):
            a=await super().decide(state,questions)
            if 'matching_key' in questions:
                a['route_status']=answer(questions['route_status'],'proceed')
                for name in questions:
                    if name.startswith('outcome_'):a[name]={'type':'noul','noul':.1}
            return a
    async def go():
        c=WordOverlap({'1':item('영수증 발급은'),'2':item('회원 자격 업무는')})
        i=engine(c);p=await i.classify_context('영수증 발급은 1번, 회원 자격 업무는 2번.','회원증 재발급',[])
        d=await Navigator(i).decide(NavState('회원증 재발급'),p,[])
        assert d.action=='press' and d.key=='2'
        assert any('parent_candidates' in state for state,q in c.calls)
    run(go())


def test_related_word_cannot_pass_when_no_verified_relationship_exists():
    class WordOverlap(ParentAI):
        async def decide(self,state,questions):
            a=await super().decide(state,questions)
            if 'matching_key' in questions:
                a['route_status']=answer(questions['route_status'],'proceed')
                for name in questions:
                    if name.startswith('outcome_'):a[name]={'type':'noul','noul':.1}
            return a
    async def go():
        c=WordOverlap({'1':item('영수증 발급은'),'2':item('회원 자격 업무는')},relation=.2)
        i=engine(c);p=await i.classify_context('영수증 발급은 1번, 회원 자격 업무는 2번.','회원증 재발급',[])
        d=await Navigator(i).decide(NavState('회원증 재발급'),p,[])
        assert d.action=='recover'
    run(go())


def test_sequential_route_checks_share_one_deadline():
    import asyncio,time
    from test_route_validation import RelationAI
    class SlowChecks(RelationAI):
        async def decide(self,state,questions):
            if 'proposal' in state or 'parent_candidates' in state:
                await asyncio.sleep(2)
                return {}
            return await super().decide(state,questions)
    async def go():
        c=SlowChecks({'1':item('예약 변경은'),'2':item('주차 안내는')})
        i=engine(c);t=time.monotonic()
        p=await i.classify_context('예약 변경은 1번, 주차 안내는 2번.','예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action=='recover' and time.monotonic()-t<1.1
    run(go())


def test_existing_strong_parent_choice_is_not_blocked_by_duplicate_review_uncertainty():
    class Established(ParentAI):
        async def decide(self,state,questions):
            a=await super().decide(state,questions)
            if 'matching_key' in questions:
                a['route_status']=answer(questions['route_status'],'proceed')
                for name in questions:
                    if name.startswith('outcome_'):a[name]={'type':'noul','noul':.1}
            return a
    async def go():
        c=Established({'1':item('영수증 발급은'),'2':item('회원 자격 업무는')},key='2',relation=.82)
        i=engine(c);p=await i.classify_context('영수증 발급은 1번, 회원 자격 업무는 2번.','회원증 재발급',[])
        d=await Navigator(i).decide(NavState('회원증 재발급'),p,[])
        assert d.action=='press' and d.key=='2'
    run(go())


def test_legacy_single_proposal_cannot_bypass_full_parent_review():
    from test_route_validation import RelationAI
    class LegacyParent(RelationAI):
        audit_calls=0
        async def decide(self,state,questions):
            if 'parent_candidates' in state:
                self.audit_calls+=1
                out={}
                for name,q in questions.items():
                    if name=='next_step':out[name]=answer(q,'c1')
                    elif name=='user_choice_needed':out[name]={'type':'noul','noul':.99}
                    elif name.endswith('_relation'):out[name]=answer(q,'parent')
                    else:out[name]={'type':'noul','noul':.99}
                return out
            a=await super().decide(state,questions)
            if 'matching_key' in questions:
                for name in questions:
                    if name.startswith('outcome_'):a[name]={'type':'noul','noul':.01}
            return a
    async def go():
        c=LegacyParent({'1':item('방문 예약은'),'2':item('온라인 예약은')},relation='parent')
        i=engine(c);p=await i.classify_context('방문 예약은 1번, 온라인 예약은 2번.','예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action=='recover' and c.audit_calls==1
    run(go())
