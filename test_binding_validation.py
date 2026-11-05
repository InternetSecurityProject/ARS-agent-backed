"""Service-level source/key verification, external AI replies scripted, no calls."""
import asyncio
import copy
import pytest
from navigator import Navigator, NavState
from test_jev_phrase import AI, answer, engine, item, run


class BindingAI(AI):
    def __init__(self,*args,bound=.98,scoped_role='business',scoped_conf=.98,
                 malformed=False,slow=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.bound=bound;self.scoped_role=scoped_role;self.scoped_conf=scoped_conf
        self.malformed=malformed;self.slow=slow

    async def decide(self,state,questions):
        if 'candidate_bindings' not in state:
            return await super().decide(state,questions)
        if self.slow:await asyncio.sleep(2)
        self.calls.append((copy.deepcopy(state),copy.deepcopy(questions)))
        if self.malformed:return {}
        return {name:({'type':'noul','noul':self.bound} if name.endswith('_bound') else
                     answer(q,self.scoped_role,self.scoped_conf)) for name,q in questions.items()}


def confused(label,role='other',span_conf=.98):
    result=item(label);result['role']=role;result['role_conf']=.38
    result['span_conf']=span_conf
    return result


async def decision(client,text,goal):
    i=engine(client);p=await i.classify_context(text,goal,[])
    d=await Navigator(i).decide(NavState(goal),p,[])
    return i,p,d


@pytest.mark.parametrize('text,label,key,goal',[
 ('개인 고객은 1번, 단순 문의 비회원 및 카드 발급 문의는 4번.',
  '단순 문의 비회원 및 카드 발급 문의는','4','새 카드 발급 받고 싶어'),
 ('일반 문의는 6번, 신규 가입 및 요금제 변경은 2번.',
  '신규 가입 및 요금제 변경은','2','새로 가입하려고 해요'),
 ('별표는 검진 예약, 3번은 주차 안내.',
  '검진 예약','*','건강 검진 예약'),
 ('병원 안내는 1번, 샵은 검사 결과 확인.',
  '검사 결과 확인','#','검사 결과 알려줘'),
])
def test_explicit_binding_repairs_role_uncertainty_and_automatically_selects(text,label,key,goal):
    async def go():
        c=BindingAI({key:confused(label)},key=key)
        _,p,d=await decision(c,text,goal)
        assert d.action=='press' and d.key==key
        assert p.label_of(key).startswith(label.rstrip('은는'))
    run(go())


@pytest.mark.parametrize('bad',[
 {'bound':.3}, {'bound':.84}, {'scoped_conf':.5}, {'scoped_role':'uncertain'},
 {'malformed':True}, {'slow':True},
])
def test_weak_conflicting_or_unavailable_verification_never_authorizes_a_key(bad):
    async def go():
        c=BindingAI({'1':confused('예약 문의는')},**bad)
        _,p,d=await decision(c,'예약 문의는 1번.','예약 변경')
        assert d.action=='recover' and d.reason=='system_uncertain'
        assert p.options[0].category is None
    run(go())


def test_wrong_source_key_pair_is_not_repaired_by_strong_business_role():
    async def go():
        c=BindingAI({'1':confused('주차 요금은'),'2':item('주차 요금은')},bound=.02)
        _,p,d=await decision(c,'예약 문의는 1번, 주차 요금은 2번.','예약 변경')
        assert d.action=='recover' and '주차' not in p.label_of('1')
    run(go())


def test_weak_source_span_cannot_be_promoted_by_new_role_confidence():
    async def go():
        c=BindingAI({'1':confused('예약 문의는',span_conf=.5)})
        _,p,d=await decision(c,'예약 문의는 1번.','예약 변경')
        assert d.action=='recover' and p.options[0].category is None
    run(go())


@pytest.mark.parametrize('text',[
 '예약 문의는 1번, 주차 요금은 1번.',
 '예약 문의는 11번.',
 '주차 요금은 2번.',
])
def test_duplicate_unsupported_or_unheard_key_cannot_gain_authorization(text):
    async def go():
        c=BindingAI({'1':confused('예약 문의는')})
        _,_,d=await decision(c,text,'예약 변경')
        assert d.action!='press'
    run(go())


def test_bound_navigation_is_not_converted_to_business():
    async def go():
        c=BindingAI({'1':confused('다시 듣기는')},scoped_role='repeat')
        _,p,d=await decision(c,'다시 듣기는 1번.','예약 변경')
        assert p.repeat_key=='1' and not p.options
        assert d.action!='press'
    run(go())


def test_unrelated_verified_option_survives_validation_failure():
    async def go():
        c=BindingAI({'1':confused('예약 문의는'),'2':item('주차 요금은')},bound=.01,key='2')
        _,p,d=await decision(c,'예약 문의는 1번, 주차 요금은 2번.','주차 요금')
        assert d.action=='press' and d.key=='2'
        assert p.label_of('2')=='주차 요금'
    run(go())


def test_critical_approval_is_never_auto_selected_by_binding_repair():
    async def go():
        c=BindingAI({'1':confused('결제 동의는')},kind='confirm',critical=.99)
        _,p,d=await decision(c,'결제 동의는 1번.','결제')
        assert p.critical and d.action!='press'
    run(go())


def test_sensitive_input_still_requires_secure_form():
    async def go():
        c=BindingAI({'1':confused('계좌 번호를')})
        _,p,d=await decision(c,'계좌번호를 입력한 뒤 샵 버튼을 눌러 주세요.','송금')
        assert p.kind=='sensitive' and d.action!='press'
    run(go())


def test_repaired_evidence_invalidates_old_uncertain_route():
    async def go():
        c=BindingAI({'1':confused('예약 문의는')},bound=.1);i=engine(c)
        p=await i.classify_context('예약 문의는 1번.','예약 변경',[])
        assert (await Navigator(i).decide(NavState('예약 변경'),p,[])).action=='recover'
        c.bound=.99
        fixed=await i.recover_context(p,'예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),fixed,[],arrived=False)
        assert d.action=='press' and d.key=='1'
    run(go())
