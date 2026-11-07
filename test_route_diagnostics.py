"""Numeric gate diagnostics must explain a hold without logging source/user data."""
import json
import logging
import pytest

from jev_parent_route import verify_parent
from test_jev_phrase import engine, item, run
from test_parent_menu import ParentAI


class CapturedParent(ParentAI):
    async def decide(self, state, questions):
        result = await super().decide(state, questions)
        if 'parent_candidates' in state:
            self.parent_proof, self.parent_checks = result, questions
        return result


@pytest.mark.parametrize('params,reason', [
    ({'bound': .84}, 'binding_below_minimum'),
    ({'relation': .7}, 'relationship_below_minimum'),
])
def test_parent_diagnostic_names_failing_candidate_gate(params, reason):
    async def go():
        c = CapturedParent({'1': item('주차 요금은'), '2': item('예약 문의는')}, **params)
        i = engine(c)
        p = await i.classify_context('주차 요금은 1번, 예약 문의는 2번.', '예약 변경', [])
        diagnostic = {}
        result = verify_parent(c.parent_proof, c.parent_checks, i._extractions[p.text],
                               diagnostic=diagnostic)
        assert result is None
        assert reason in diagnostic['candidates']['2']['failed']
        assert diagnostic['candidates']['2']['source'] == .99
        assert diagnostic['candidates']['2']['binding_minimum'] == .85
        assert 'no_unique_valid_candidate' in diagnostic['failed']
        assert '예약' not in json.dumps(diagnostic, ensure_ascii=False)
    run(go())


def test_parent_diagnostic_reports_global_pick_rejection():
    async def go():
        c = CapturedParent({'1': item('주차 요금은'), '2': item('예약 문의는')}, pick_conf=.5)
        i = engine(c)
        p = await i.classify_context('주차 요금은 1번, 예약 문의는 2번.', '예약 변경', [])
        diagnostic = {}
        assert verify_parent(c.parent_proof, c.parent_checks, i._extractions[p.text],
                             diagnostic=diagnostic) is None
        assert 'choice_below_minimum' in diagnostic['failed']
        assert diagnostic['candidates']['2']['failed'] == []
    run(go())


def test_route_and_parent_diagnostics_log_scores_not_source(caplog):
    async def go():
        c = CapturedParent({'1': item('주차 요금은'), '2': item('예약 문의는')}, bound=.84)
        await engine(c).classify_context('주차 요금은 1번, 예약 문의는 2번.',
                                         'RAW-USER-CONTEXT 예약 변경', [])
    with caplog.at_level(logging.INFO):
        run(go())
    records = [json.loads(r.message.split('route_diagnostic ', 1)[1])
               for r in caplog.records if 'route_diagnostic ' in r.message]
    assert any(r['stage'] == 'selection' and 'service' in r for r in records)
    assert any(r['stage'] == 'parent' and
               'binding_below_minimum' in r['candidates']['2']['failed'] for r in records)
    serialized = json.dumps(records, ensure_ascii=False)
    assert 'RAW-USER-CONTEXT' not in serialized
    assert '예약 문의' not in serialized and '주차 요금' not in serialized


def test_single_candidate_diagnostic_explains_rejected_binding():
    from jev_route_validation import verified_route, RELATION
    from test_jev_phrase import answer
    q={'type':'choice','criteria':RELATION}
    diagnostic={}
    assert verified_route({'relationship':answer(q,'parent'),
                           'source_supported':{'type':'noul','noul':.84}},
                          .99,diagnostic=diagnostic) is None
    assert diagnostic['failed']==['binding_below_minimum']
    assert diagnostic['binding']==.84 and diagnostic['binding_minimum']==.85


def test_invalid_provider_answer_never_logs_raw_value(caplog):
    class Broken(CapturedParent):
        async def decide(self,state,questions):
            result=await super().decide(state,questions)
            if 'parent_candidates' in state:
                result['next_step']['choice']='private-provider-body-do-not-log'
            return result
    async def go():
        c=Broken({'1':item('주차 요금은'),'2':item('예약 문의는')})
        i=engine(c)
        p=await i.classify_context('주차 요금은 1번, 예약 문의는 2번.','예약 변경',[])
        assert (await i.assess_route('예약 변경',p,p.options,[])).status=='unknown'
    with caplog.at_level(logging.INFO):run(go())
    assert 'private-provider-body-do-not-log' not in caplog.text
    assert '"error": "ValueError"' in caplog.text


def test_cache_reuse_is_visible_without_logging_menu_contents(caplog):
    async def go():
        c=CapturedParent({'1':item('주차 요금은'),'2':item('예약 문의는')},relation=.7)
        i=engine(c)
        p=await i.classify_context('주차 요금은 1번, 예약 문의는 2번.','예약 변경',[])
        before=len(c.calls)
        await i.assess_route('예약 변경',p,p.options,[])
        assert len(c.calls)==before
    with caplog.at_level(logging.INFO):run(go())
    records=[json.loads(r.message.split('route_diagnostic ',1)[1])
             for r in caplog.records if 'route_diagnostic ' in r.message]
    assert any(r['stage']=='cache' and r['action']=='hit' and r['result']=='unknown'
               for r in records)


def test_logger_failure_cannot_change_a_valid_routing_decision(monkeypatch):
    import jev_diagnostics
    from test_system_recovery import RecoveryAI
    from navigator import Navigator, NavState
    def broken(*args,**kwargs):raise OSError('private handler failure')
    monkeypatch.setattr(jev_diagnostics.log,'info',broken)
    async def go():
        i=engine(RecoveryAI({'1':item('예약 변경은'),'2':item('주차 요금은')}))
        p=await i.classify_context('예약 변경은 1번, 주차 요금은 2번.','예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action=='press' and d.key=='1'
    run(go())


def test_unused_bad_metric_does_not_override_valid_selected_proof(caplog):
    from test_system_recovery import RecoveryAI
    from navigator import Navigator, NavState
    class UnusedBadMetric(RecoveryAI):
        async def decide(self,state,questions):
            result=await super().decide(state,questions)
            if 'outcome_2' in result:
                result['outcome_2']={'type':'noul','noul':'private-invalid-unused-field'}
            return result
    async def go():
        i=engine(UnusedBadMetric({'1':item('예약 변경은'),'2':item('주차 요금은')}))
        p=await i.classify_context('예약 변경은 1번, 주차 요금은 2번.','예약 변경',[])
        d=await Navigator(i).decide(NavState('예약 변경'),p,[])
        assert d.action=='press' and d.key=='1'
    with caplog.at_level(logging.INFO):run(go())
    assert 'private-invalid-unused-field' not in caplog.text


@pytest.mark.parametrize('requirement,confidence,needed,allowed',[
    ('none',.99,.36,True),
    ('none',.99,.65,False),
    ('none',.70,.01,True),
    ('none',.70,.36,False),
    ('unknown',.99,.01,False),
    ('restriction_c2',.99,.01,False),
    ('pair_c1_c2',.99,.01,False),
])
def test_parent_requires_grounded_current_user_requirement_proof(requirement,confidence,needed,allowed):
    from copy import deepcopy
    from test_jev_phrase import answer
    async def go():
        c=CapturedParent({'1':item('주차 요금은'),'2':item('예약 문의는')})
        i=engine(c)
        p=await i.classify_context('주차 요금은 1번, 예약 문의는 2번.','예약 변경',[])
        checks=deepcopy(c.parent_checks);proof=deepcopy(c.parent_proof)
        checks.setdefault('current_user_requirement',{'type':'choice','criteria':{
            'none':'no missing current fact','unknown':'uncertain',
            'restriction_c2':{'key':'2'},'pair_c1_c2':{'keys':['1','2']}}})
        proof['current_user_requirement']=answer(checks['current_user_requirement'],requirement,confidence)
        criteria=checks['current_user_requirement']['criteria']
        proof['current_user_requirement']['probabilities']={k:confidence if k==requirement
            else (1-confidence)/(len(criteria)-1) for k in criteria}
        proof['user_choice_needed']={'type':'noul','noul':needed}
        diagnostic={}
        result=verify_parent(proof,checks,i._extractions[p.text],diagnostic=diagnostic)
        assert (result is not None)==allowed
        assert diagnostic['requirement']==requirement
        assert diagnostic['raw_user_choice_needed']==needed
    run(go())


def test_malformed_requirement_distribution_cannot_authorize_entry():
    from copy import deepcopy
    async def go():
        c=CapturedParent({'1':item('주차 요금은'),'2':item('예약 문의는')})
        i=engine(c)
        p=await i.classify_context('주차 요금은 1번, 예약 문의는 2번.','예약 변경',[])
        checks=deepcopy(c.parent_checks);proof=deepcopy(c.parent_proof)
        q=checks['current_user_requirement']
        proof['current_user_requirement']['probabilities']={k:(.99 if k=='none'
            else .98 if k=='restriction_c2' else 0.) for k in q['criteria']}
        proof['user_choice_needed']={'type':'noul','noul':.36}
        with pytest.raises(ValueError,match='invalid_requirement_distribution'):
            verify_parent(proof,checks,i._extractions[p.text])
    run(go())
