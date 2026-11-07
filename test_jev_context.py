"""Context/latency regression tests; no external calls or institution-specific data."""
import asyncio

from interpreter import Option, Prompt
from jev_interpreter import JevInterpreter
from navigator import Navigator, NavState, Layer


class TypedReplies:
    def __init__(self): self.calls=[]

    async def decide(self, state, questions):
        self.calls.append((state,questions))
        chosen='dtmf_5' if 'dtmf_5' in questions.get('matching_key',{}).get('criteria',{}) else 'dtmf_1'
        answers={}
        for name,q in questions.items():
            if q['type']=='choice':
                pick='menu' if name=='kind' else 'proceed' if name=='route_status' else chosen
                answers[name]={'type':'choice','choice':pick,'probabilities':{k:(.98 if k==pick else .001) for k in q['criteria']},'confidence':.98}
            else: answers[name]={'type':'noul','noul':.01 if name=='critical' else .99}
        return answers


def run(c): return asyncio.run(c)


def test_current_menu_classification_and_navigation_share_one_request():
    async def go():
        c=TypedReplies();i=JevInterpreter(c)
        p=await i.classify_context('잔액 조회는 3번, 카드 업무는 5번.', '카드 발급', [])
        decision=await Navigator(i).decide(NavState('카드 발급'),p,[])
        assert decision.action=='press' and decision.key=='5'
        assert len(c.calls)==1
        assert set(c.calls[0][1])=={'kind','critical','matching_key','has_matching_service','route_status'}
    run(go())


def test_parent_menu_scoring_uses_parent_transcript_and_labels():
    async def go():
        c=TypedReplies();i=JevInterpreter(c)
        parent=Prompt('menu','송금은 1번, 카드 업무는 5번.',[Option('1','송금'),Option('5','카드 업무')])
        i.last_menu=Prompt('menu','카드 신규 문의는 3번, 카드 이용 내역은 4번.',[Option('3','카드 신규 문의'),Option('4','카드 이용 내역')])
        scores=await i.score_prompt('송금',parent,[parent.options[0]],[])
        assert scores['1']>.5
        state,questions=c.calls[-1]
        assert state['current_ars_transcript']=='송금은 1번, 카드 업무는 5번.'
        assert state['options'][0]['label']=='송금'
        assert questions['matching_key']['criteria']['dtmf_1']['service']=='송금'
    run(go())


def test_batch_cache_distinguishes_goal_path_and_filtered_candidates():
    async def go():
        c=TypedReplies();i=JevInterpreter(c)
        text='송금은 1번, 카드 업무는 5번.'
        p=await i.classify_context(text,'카드 발급',[])
        await i.score_prompt('카드 발급',p,p.options,[])
        assert len(c.calls)==1
        await i.score_prompt('카드 발급',p,p.options,['새 경로'])
        await i.score_prompt('카드 발급',p,[p.options[0]],['새 경로'])
        await i.score_prompt('송금',p,p.options,[])
        assert len(c.calls)==4
    run(go())


def test_failed_batch_is_not_retried_during_score_or_auto_pressed():
    class Broken:
        def __init__(self): self.calls=0
        async def decide(self,*args): self.calls+=1;raise TimeoutError('test')
    async def go():
        c=Broken();i=JevInterpreter(c)
        p=await i.classify_context('송금은 1번, 카드 업무는 5번.','카드 발급',[])
        decision=await Navigator(i).decide(NavState('카드 발급'),p,[])
        assert decision.action=='ask' and c.calls==1
    run(go())


def test_call_forks_share_transport_but_never_menu_state():
    async def go():
        c=TypedReplies();parent=JevInterpreter(c)
        a,b=parent.new_session(),parent.new_session()
        await a.classify_context('송금은 1번, 카드 업무는 5번.','카드 발급',[])
        assert a.last_menu is not None and b.last_menu is None and parent.last_menu is None
        assert b.goal=='' and b.last_path==[]
    run(go())


def test_navigator_context_extension_reaches_previous_prompt():
    class ContextOnly:
        async def score_prompt(self,goal,prompt,options,path):
            if prompt.text=='child': return {o.key:0 for o in options}
            assert prompt.text=='parent' and [o.key for o in options]==['1']
            return {'1':.9}
        async def score(self,*args): raise AssertionError('context was lost')
    async def go():
        parent=Prompt('menu','parent',[Option('1','다른 업무'),Option('5','진행 중')])
        child=Prompt('menu','child',[Option('7','불일치')],back_key='*')
        d=await Navigator(ContextOnly()).decide(NavState('목적'),child,[Layer(parent,'5','ai')])
        assert d.action=='backtrack' and d.keys=='*'
    run(go())


def test_bad_matching_answer_cannot_erase_valid_critical_classification():
    class CriticalWithBadScores(TypedReplies):
        async def decide(self,state,questions):
            answer=await super().decide(state,questions)
            if 'kind' in questions:
                answer['critical']['noul']=.99
                answer['matching_key']['probabilities']={}
            return answer
    async def go():
        c=CriticalWithBadScores();i=JevInterpreter(c)
        p=await i.classify_context('신청 확정은 1번, 취소는 2번.','신청 확정',[])
        assert p.critical
        state=NavState('신청 확정',exclude={'2'},exclude_at=p.signature())
        d=await Navigator(i).decide(state,p,[])
        assert d.action=='ask' and d.reason=='critical_action' and len(c.calls)==1
    run(go())
