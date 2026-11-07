"""Repeated-key semantics need independently verified source evidence."""
import pytest
from test_jev_phrase import AI,item,answer,engine,run
from navigator import Navigator,NavState

class RepeatAI(AI):
    def __init__(self,*args,equivalent=.98,bound=.98,proof_role='business',proof_conf=.98,broken=False,**kwargs):
        super().__init__(*args,**kwargs);self.equivalent=equivalent;self.bound=bound
        self.proof_role=proof_role;self.proof_conf=proof_conf;self.broken=broken
    async def decide(self,state,questions):
        if 'repeated_bindings' in state:
            self.calls.append((state,questions))
            if self.broken:return {}
            out={}
            for name,q in questions.items():
                if name.endswith('_same'):out[name]={'type':'noul','noul':self.equivalent}
                elif name.endswith('_bound'):out[name]={'type':'noul','noul':self.bound}
                else:out[name]=answer(q,self.proof_role,self.proof_conf)
            return out
        return await super().decide(state,questions)

@pytest.mark.parametrize('text,label,key',[
 ('외국어 상담은 6번. For foreign language service, press 6.','외국어 상담은','6'),
 ('예약 변경은 2번. 예약 변경은 2번을 눌러 주세요.','예약 변경은','2'),
 ('분실 신고는 별표. For loss reports press *.','분실 신고는','*'),
])
def test_source_verified_equivalent_repetitions_are_one_usable_option(text,label,key):
    async def go():
        c=RepeatAI({key:item(label)},key=key);i=engine(c)
        p=await i.classify_context(text,label,[])
        d=await Navigator(i).decide(NavState(label),p,[])
        assert d.action=='press' and d.key==key
        assert sum(o.key==key for o in p.options)==1
        assert f'conflicting_key:{key}' not in i._extractions[text].issues
    run(go())

@pytest.mark.parametrize('params',[
 {'equivalent':.2},{'equivalent':.89},{'bound':.2},{'proof_role':'uncertain'},
 {'proof_conf':.5},{'broken':True},
])
def test_repeated_key_stays_blocked_without_strong_consistent_proof(params):
    async def go():
        c=RepeatAI({'1':item('예약 변경은')},**params);i=engine(c)
        p=await i.classify_context('예약 변경은 1번, 주차 문의는 1번.','예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action!='press'
    run(go())


def test_repetition_proof_cannot_raise_an_untrusted_original_label():
    async def go():
        c=RepeatAI({'1':item('예약 변경은',confidence=.4)});i=engine(c)
        p=await i.classify_context('예약 변경은 1번. 예약 변경은 1번.','예약 변경',[])
        assert not p.options or all(o.category is None for o in p.options)
    run(go())


def test_identical_explicit_bindings_can_be_verified_locally_without_relaxing_semantics():
    async def go():
        c=RepeatAI({'1':item('예약 변경은')},bound=.3)
        i=engine(c);text='예약 변경은 1번. 예약 변경은 1번을 눌러 주세요.'
        p=await i.classify_context(text,'예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action=='press' and d.key=='1'
    run(go())


def test_literal_repetition_does_not_override_semantic_stage_conflict():
    async def go():
        c=RepeatAI({'1':item('예약 변경은')},bound=.3,equivalent=.1)
        i=engine(c);text='예약 변경은 1번. 예약 변경은 1번을 눌러 주세요.'
        p=await i.classify_context(text,'예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action!='press'
    run(go())
