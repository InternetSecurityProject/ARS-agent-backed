"""A verified replay permits one fresh verdict; repeated uncertainty cannot loop."""
import pytest

from test_jev_phrase import answer, engine, item, run
from test_system_recovery import RecoveryAI, make_session
from test_parent_menu import ParentAI


class BecomesReady(ParentAI):
    def __init__(self, items, key):
        super().__init__(items, key=key, relation=.7)
        self.route_calls = 0

    async def decide(self, state, questions):
        # Initial classification and one internal reanalysis remain uncertain.
        # A NEW response after replay becomes confident; old cache must not hide it.
        if 'matching_key' in questions:
            self.route_calls += 1
            self.route = 'proceed' if self.route_calls >= 3 else 'unknown'
        return await super().decide(state, questions)


@pytest.mark.parametrize('business,replay,label,goal', [
    ('1', '9', '예약 문의는', '예약 변경'),
    ('4', '*', '배송 업무는', '받을 주소 변경'),
    ('7', '#', '회원 업무는', '회원증 다시 받기'),
])
def test_session_replay_rechecks_old_unknown_verdict_once(business, replay, label, goal):
    async def go():
        c = BecomesReady({business: item(label), replay: item('다시 듣기는', 'repeat')}, business)
        s = make_session(c)
        s.goal = s.nav.goal = goal
        replay_name = replay + '번' if replay.isdigit() else replay
        text = f'{label} {business}번, 다시 듣기는 {replay_name}.'
        try:
            await s.on_paragraph(text)
            assert s.call.keys == [replay]
            await s.on_paragraph(text)
            assert s.call.keys == [replay, business]
            assert c.route_calls == 3
            assert s.pending is None
        finally:
            await s.finish('user_canceled', hangup=False)
    run(go())


def test_successful_verdict_is_not_invalidated_or_requeried():
    async def go():
        c = RecoveryAI({'1': item('예약 문의는')})
        i = engine(c)
        p = await i.classify_context('예약 문의는 1번.', '예약 변경', [])
        before = len(c.calls)
        assert not i.retry_uncertain_after_replay(p)
        await i.score_prompt('예약 변경', p, p.options, [])
        assert len(c.calls) == before
    run(go())


def test_uncertain_replay_remains_bounded():
    async def go():
        c = RecoveryAI({'1': item('예약 문의는'), '9': item('다시 듣기는', 'repeat')}, route='unknown')
        s = make_session(c)
        try:
            for _ in range(5):
                await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            assert s.call.keys == ['9'] and s.pending is None and not s.ended
            assert sum('matching_key' in q for _, q in c.calls) <= 3
        finally:
            await s.finish('user_canceled', hangup=False)
    run(go())


def test_genuine_user_choice_cache_is_preserved():
    async def go():
        c = RecoveryAI({'1': item('개인 고객은'), '2': item('기업 고객은')},
                       route='needs_info', missing_pair=('1','2'))
        i = engine(c)
        p = await i.classify_context('개인 고객은 1번, 기업 고객은 2번.', '예약 변경', [])
        before = len(c.calls)
        assert not i.retry_uncertain_after_replay(p)
        assessment = await i.assess_route('예약 변경',p,p.options,[])
        assert assessment.status == 'needs_info' and len(c.calls) == before
    run(go())


def test_other_evidence_or_goal_cannot_evict_current_verdict():
    import copy
    async def go():
        c=ParentAI({'1':item('예약 문의는')},relation=.7)
        i=engine(c)
        p=await i.classify_context('예약 문의는 1번.','예약 변경',[])
        stale=copy.deepcopy(p);stale._phrase_evidence_id='old-evidence'
        assert not i.retry_uncertain_after_replay(stale)
        i.goal='다른 목적'
        assert not i.retry_uncertain_after_replay(p)
        assert i._routes
    run(go())


def test_failed_replay_does_not_invalidate_verdict():
    async def go():
        c=ParentAI({'1':item('예약 문의는'),'9':item('다시 듣기는','repeat')},relation=.7)
        s=make_session(c)
        seen=[]
        s.interp.retry_uncertain_after_replay=lambda p:seen.append(p)
        async def broken(key):raise RuntimeError('synthetic transport failure')
        s.call.send_keys=broken
        try:
            await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            assert seen==[]
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
