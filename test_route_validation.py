"""Independent source-bound route proof, not relaxed confidence thresholds."""
import pytest
from test_binding_validation import BindingAI,decision
from test_jev_phrase import item,answer,run

class RelationAI(BindingAI):
    def __init__(self,*args,relation='direct',relation_conf=.97,source=.98,raw='proceed',**kw):
        super().__init__(*args,route=raw,**kw)
        self.relation=relation;self.relation_conf=relation_conf;self.source=source
    async def decide(self,state,questions):
        if 'proposal' in state:
            return {'relationship':answer(questions['relationship'],self.relation,self.relation_conf),
                    'source_supported':{'type':'noul','noul':self.source}}
        a=await super().decide(state,questions)
        if 'matching_key' in questions:
            a['route_status']['confidence']=.45
            a['matching_key']=answer(questions['matching_key'],a['matching_key']['choice'],.82)
            a['has_matching_service']={'type':'noul','noul':.78}
        return a


def test_new_independent_relation_proof_resolves_weak_broad_proceed():
    async def go():
        c=RelationAI({'1':item('신규 가입 및 요금 변경은'),'2':item('고장 신고는')})
        _,_,d=await decision(c,'신규 가입 및 요금 변경은 1번, 고장 신고는 2번.','신규 가입')
        assert d.action=='press' and d.key=='1'
    run(go())

@pytest.mark.parametrize('params',[
 {'relation':'unrelated'}, {'relation':'needs_info'}, {'relation':'uncertain'},
 {'relation_conf':.84}, {'source':.84}, {'raw':'unknown'}, {'raw':'needs_info'},
])
def test_failed_or_conflicting_relation_proof_never_overrides_uncertainty(params):
    async def go():
        c=RelationAI({'1':item('예약 변경은'),'2':item('주차 안내는')},**params)
        _,_,d=await decision(c,'예약 변경은 1번, 주차 안내는 2번.','예약 변경')
        assert d.action!='press'
    run(go())


def test_missing_user_type_is_still_a_user_question():
    async def go():
        c=RelationAI({'1':item('성인 예약은'),'2':item('소아 예약은')},raw='needs_info')
        _,_,d=await decision(c,'성인 예약은 1번, 소아 예약은 2번.','예약 변경')
        assert d.action=='ask' and {o.key for o in d.options}=={'1','2'}
    run(go())


def test_raw_proceed_with_grounded_user_conflict_cannot_enter_positive_proof():
    class ConflictAI(RelationAI):
        proof_calls=0
        async def decide(self,state,questions):
            if 'proposal' in state:self.proof_calls+=1
            a=await super().decide(state,questions)
            if 'matching_key' in questions:
                q=questions['clarification_pair']
                pair=next(k for k,v in q['criteria'].items() if isinstance(v,dict))
                a['clarification_pair']=answer(q,pair)
                a['user_detail_required']={'type':'noul','noul':.99}
            return a
    async def go():
        c=ConflictAI({'1':item('성인 예약은'),'2':item('소아 예약은')})
        _,_,d=await decision(c,'성인 예약은 1번, 소아 예약은 2번.','예약 변경')
        assert d.action=='ask' and c.proof_calls==0
    run(go())


@pytest.mark.parametrize('slow',[False,True])
def test_malformed_or_timed_out_proof_preserves_original_uncertainty(slow):
    import asyncio
    class BrokenAI(RelationAI):
        async def decide(self,state,questions):
            if 'proposal' in state:
                if slow:await asyncio.sleep(2)
                return {}
            return await super().decide(state,questions)
    async def go():
        c=BrokenAI({'1':item('예약 변경은'),'2':item('주차 안내는')})
        _,_,d=await decision(c,'예약 변경은 1번, 주차 안내는 2번.','예약 변경')
        assert d.action=='recover'
    run(go())
