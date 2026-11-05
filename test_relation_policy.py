"""Four-way relation proof must constrain real navigator decisions, not key names."""
import pytest
from navigator import Navigator,NavState
from test_jev_phrase import answer,engine,item,run
from test_parent_menu import ParentAI

class RelationProofAI(ParentAI):
    def __init__(self,*args,relations=None,target=.98,action=.98,chosen='2',**kwargs):
        super().__init__(*args,audit_key=chosen,**kwargs)
        self.relations=relations or {'1':'different','2':'parent'}
        self.target=target;self.action=action
    async def decide(self,state,questions):
        if 'parent_candidates' not in state:return await super().decide(state,questions)
        self.calls.append((state,questions));out={}
        for name,q in questions.items():
            code=name.split('_')[0];key=code[1:]
            if name=='next_step':out[name]=answer(q,'c'+self.audit_key)
            elif name=='user_choice_needed':out[name]={'type':'noul','noul':.01}
            elif name.endswith('_relation'):out[name]=answer(q,self.relations.get(key,'unknown'))
            elif name.endswith('_target'):out[name]={'type':'noul','noul':self.target}
            elif name.endswith('_action'):out[name]={'type':'noul','noul':self.action}
            elif name.endswith('_bound'):out[name]={'type':'noul','noul':.99}
            elif name.endswith('_parent'):
                # Old overloaded boolean can be high even when atomic judgments disagree.
                out[name]={'type':'noul','noul':.98 if key==self.audit_key else .01}
            else:raise AssertionError(name)
        return out

async def result(client):
    i=engine(client);p=await i.classify_context('납부 확인서 발급은 1번, 회원 자격 업무는 2번.','회원증 재발급',[])
    return await Navigator(i).decide(NavState('회원증 재발급'),p,[])

def make(**kwargs):
    return RelationProofAI({'1':item('납부 확인서 발급은'),'2':item('회원 자격 업무는')},**kwargs)

@pytest.mark.parametrize('relation',['different','unknown'])
def test_positive_old_parent_score_cannot_override_different_or_unknown_relation(relation):
    async def go():
        d=await result(make(relations={'1':'different','2':relation}))
        assert d.action=='recover' and d.key is None
    run(go())

@pytest.mark.parametrize('field',['target','action'])
def test_shared_verb_or_target_does_not_authorize_incompatible_task(field):
    async def go():
        d=await result(make(**{field:.1}))
        assert d.action=='recover' and d.key is None
    run(go())

@pytest.mark.parametrize('relation',['direct','parent'])
def test_supported_direct_or_parent_can_be_selected_without_user_question(relation):
    async def go():
        d=await result(make(relations={'1':'different','2':relation}))
        assert d.action=='press' and d.key=='2'
    run(go())

def test_unique_direct_result_is_preferred_to_a_broader_compatible_parent():
    async def go():
        d=await result(make(relations={'1':'parent','2':'direct'}))
        assert d.action=='press' and d.key=='2'
    run(go())

@pytest.mark.parametrize('relation',['direct','parent'])
def test_two_supported_relations_of_same_specificity_do_not_authorize_guessing(relation):
    async def go():
        d=await result(make(relations={'1':relation,'2':relation}))
        assert d.action=='recover' and d.key is None
    run(go())

def test_selected_parent_cannot_hide_a_more_specific_direct_match():
    async def go():
        d=await result(make(relations={'1':'direct','2':'parent'}))
        assert d.action=='recover' and d.key is None
    run(go())


def test_uncertainty_between_direct_and_parent_is_not_uncertainty_about_advancing():
    class Split(RelationProofAI):
        async def decide(self,state,questions):
            out=await super().decide(state,questions)
            if 'c2_relation' in out:
                out['c2_relation']={'type':'choice','choice':'parent','confidence':.4,
                    'probabilities':{'direct':.47,'parent':.49,'different':.02,'unknown':.02}}
            return out
    async def go():
        d=await result(Split({'1':item('납부 확인서 발급은'),'2':item('회원 자격 업무는')}))
        assert d.action=='press' and d.key=='2'
    run(go())


def test_unknown_or_different_probability_is_not_counted_as_supported_relationship():
    class Split(RelationProofAI):
        async def decide(self,state,questions):
            out=await super().decide(state,questions)
            if 'c2_relation' in out:
                out['c2_relation']={'type':'choice','choice':'parent','confidence':.4,
                    'probabilities':{'direct':.1,'parent':.5,'different':.3,'unknown':.1}}
            return out
    async def go():
        d=await result(Split({'1':item('납부 확인서 발급은'),'2':item('회원 자격 업무는')}))
        assert d.action=='recover' and d.key is None
    run(go())
